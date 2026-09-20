"""Real GR00T fine-tuning VRAM benchmark, using the official launcher and synthetic data.

No model weights or optimizer checkpoints are saved. Forward, backward, and AdamW
updates are unchanged. Each batch size runs in fresh single-GPU or torchrun processes.
"""

import argparse
import json
import math
import os
import runpy
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / "n17"
STORAGE = Path.home() / "raid/vla_finetune"


class DeviceMonitor:
    """Sample whole-device NVML memory; includes CUDA context and non-Torch allocations."""

    def __init__(self, gpu):
        self.gpu = gpu
        self.samples = []
        self.errors = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            self.sample_once()
            self.stop_event.wait(0.2)

    def sample_once(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.gpu),
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=5,
                check=True,
            )
            memory, util = (float(s.strip()) for s in result.stdout.strip().split(","))
            self.samples.append({"time": time.time(), "used_mib": memory, "util": util})
        except (subprocess.SubprocessError, ValueError):
            self.errors += 1

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)


def worker(args):
    import torch
    from transformers import TrainerCallback

    sys.path.insert(0, str(REPO))
    import gr00t.experiment.experiment as experiment

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    run_output = args.output.resolve()
    output = run_output / f"rank-{rank}" if world_size > 1 else run_output
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "batch_size": args.batch_sizes[0],
        "global_batch_size": args.batch_sizes[0],
        "per_gpu_batch_size": args.batch_sizes[0] // world_size,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "max_steps": args.steps,
        "gpu_index": args.gpus[local_rank],
        "gpu_name": torch.cuda.get_device_name(local_rank),
        "gpu_uuid": str(torch.cuda.get_device_properties(local_rank).uuid),
        "gpu_total_gib": torch.cuda.get_device_properties(local_rank).total_memory / 2**30,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "dataset": str(args.dataset.resolve()),
        "base_model": str(args.model.resolve()),
        "tuning_mode": args.mode,
        "checkpoint_saving": False,
        "step_metrics": [],
        "losses": [],
        "trainer_logs": [],
    }
    monitor = DeviceMonitor("GPU-" + report["gpu_uuid"].removeprefix("GPU-"))
    monitor.start()
    start = time.monotonic()
    trainer_ref = []

    def memory():
        return {
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def tensor_shapes(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, Mapping):
            return {k: tensor_shapes(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [tensor_shapes(v) for v in value[:4]]
        return type(value).__name__

    class MemoryCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            model = kwargs["model"]
            counts = Counter()
            for param in model.parameters():
                counts[f"{param.dtype}/{'trainable' if param.requires_grad else 'frozen'}"] += (
                    param.numel()
                )
            report["training_parameters_by_dtype"] = dict(counts)
            engine = trainer_ref[0].model_wrapped
            if hasattr(engine, "zero_optimization_stage"):
                report["actual_zero_stage"] = engine.zero_optimization_stage()
                report["resolved_deepspeed_config"] = {
                    key: value for key, value in engine.config.items() if key != "steps_per_print"
                }
                partitions = getattr(engine.optimizer, "single_partition_of_fp32_groups", [])
                report["optimizer_fp32_master_partition_gib"] = (
                    sum(p.numel() * p.element_size() for p in partitions) / 2**30
                )
            report["before_training"] = memory()
            torch.cuda.reset_peak_memory_stats()
            report["training_started_unix"] = time.time()
            self.step_start = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            monitor.sample_once()  # Also capture short runs at every optimizer step.
            metric = {
                "step": state.global_step,
                "seconds": time.monotonic() - self.step_start,
                **memory(),
            }
            self.step_start = time.monotonic()
            report["step_metrics"].append(metric)
            print(f"VRAM_STEP rank={rank} " + json.dumps(metric), flush=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            report["trainer_logs"].append({"step": state.global_step, **(logs or {})})

    class BenchmarkTrainer(experiment.Gr00tTrainer):
        def __init__(self, *positional, **keywords):
            training_args = keywords["args"]
            training_args.save_strategy = "no"
            training_args.logging_steps = 1
            training_args.logging_first_step = True
            training_args.disable_tqdm = True
            super().__init__(*positional, **keywords)
            self.add_callback(MemoryCallback())
            trainer_ref.append(self)
            params = list(self.model.named_parameters())
            report["total_parameters"] = sum(p.numel() for _, p in params)
            report["trainable_parameters"] = sum(p.numel() for _, p in params if p.requires_grad)
            counts = Counter()
            for _, param in params:
                counts[f"{param.dtype}/{'trainable' if param.requires_grad else 'frozen'}"] += (
                    param.numel()
                )
            report["parameters_by_dtype"] = dict(counts)
            report["training_settings"] = {
                key: getattr(self.args, key)
                for key in (
                    "bf16",
                    "fp16",
                    "tf32",
                    "gradient_checkpointing",
                    "gradient_accumulation_steps",
                    "per_device_train_batch_size",
                    "learning_rate",
                    "weight_decay",
                    "warmup_ratio",
                    "dataloader_num_workers",
                )
            }
            report["optimizer"] = str(self.args.optim)
            report["deepspeed"] = self.args.deepspeed
            report["model_config"] = self.model.config.to_dict()

        def training_step(self, model, inputs, *positional, **keywords):
            if "first_batch" not in report:
                report["first_batch"] = tensor_shapes(inputs)
            loss = super().training_step(model, inputs, *positional, **keywords)
            value = float(loss.detach().float().cpu())
            report["losses"].append(value)
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite training loss: {value}")
            return loss

        def save_model(self, *positional, **keywords):
            print("VRAM benchmark: model checkpoint saving intentionally disabled.", flush=True)

        def _save_checkpoint(self, *positional, **keywords):
            pass

    experiment.Gr00tTrainer = BenchmarkTrainer
    sys.argv = [
        str(REPO / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(args.model.resolve()),
        "--dataset-path",
        str(args.dataset.resolve()),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(REPO / "examples/SO100/so100_config.py"),
        "--output-dir",
        str(run_output / "artifacts"),
        "--num-gpus",
        str(world_size),
        "--global-batch-size",
        str(args.batch_sizes[0]),
        "--max-steps",
        str(args.steps),
        "--dataloader-num-workers",
        "0",
        "--shard-size",
        "128",
        "--episode-sampling-rate",
        "1.0",
        "--num-shards-per-epoch",
        "128" if world_size > 1 else "8",
        "--save-steps",
        "1000000",
    ]
    if args.mode == "full":
        sys.argv += ["--tune-llm", "--tune-visual"]
    report["official_launcher_argv"] = sys.argv[1:]
    exit_code = 0
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
        trainer = trainer_ref[0]
        report["completed_steps"] = trainer.state.global_step
        wrapped_optimizer = trainer.optimizer
        for _ in range(8):
            if isinstance(wrapped_optimizer, torch.optim.Optimizer):
                break
            if hasattr(wrapped_optimizer, "single_partition_of_fp32_groups"):
                report["optimizer_fp32_master_partition_gib"] = (
                    sum(
                        p.numel() * p.element_size()
                        for p in wrapped_optimizer.single_partition_of_fp32_groups
                    )
                    / 2**30
                )
            if not hasattr(wrapped_optimizer, "optimizer"):
                raise RuntimeError("Cannot inspect wrapped optimizer state")
            wrapped_optimizer = wrapped_optimizer.optimizer
        report["optimizer_state_gib"] = (
            sum(
                tensor.numel() * tensor.element_size()
                for state in wrapped_optimizer.state.values()
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor)
            )
            / 2**30
        )
        if trainer.state.global_step != args.steps or not report["losses"]:
            raise RuntimeError("Training did not complete the requested optimizer steps")
        if report["first_batch"]["inputs"]["action"]["shape"][0] != report["per_gpu_batch_size"]:
            raise RuntimeError("Observed per-GPU batch differs from requested batch")
        report["status"] = "ok"
    except Exception as exc:
        import traceback

        report["status"] = "oom" if isinstance(exc, torch.OutOfMemoryError) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        exit_code = 1
    finally:
        monitor.stop()
        report["captured_optimizer_steps"] = len(report["step_metrics"])
        report["final_memory"] = memory()
        report["elapsed_seconds"] = time.monotonic() - start
        report["device_monitor_errors"] = monitor.errors
        report["device_sample_count"] = len(monitor.samples)
        report["device_sampled_peak_gib"] = max(
            (s["used_mib"] / 1024 for s in monitor.samples), default=None
        )
        report["device_sampled_peak_util"] = max((s["util"] for s in monitor.samples), default=None)
        report["device_sampling_valid"] = (
            report["device_sampled_peak_gib"] is not None
            and report["device_sampled_peak_gib"] + 0.1 >= report["final_memory"]["reserved_gib"]
        )
        if report["status"] == "ok" and not report["device_sampling_valid"]:
            report["status"] = "measurement_error"
            report["error"] = "Device samples did not capture resident CUDA allocations"
            exit_code = 1
        (output / "device_samples.json").write_text(json.dumps(monitor.samples, indent=2) + "\n")
        (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(
            "VRAM_RESULT "
            + json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "batch_size",
                        "tuning_mode",
                        "final_memory",
                        "device_sampled_peak_gib",
                    )
                }
            ),
            flush=True,
        )
        if world_size > 1 and report["status"] == "ok" and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Global batch sizes across selected GPUs, before gradient accumulation",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="+", help="Physical GPU indices; overrides --gpu")
    parser.add_argument(
        "--timeout-seconds", type=int, default=600,
        help="Worker timeout in seconds; 0 disables the timeout",
    )
    parser.add_argument("--mode", choices=["default", "full"], default="default")
    parser.add_argument("--dataset", type=Path, default=STORAGE / "datasets/synthetic_so100_vram")
    parser.add_argument("--model", type=Path, default=STORAGE / "models/GR00T-N1.7-3B")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.gpus = args.gpus if args.gpus is not None else [args.gpu]
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be distinct")
    if any(batch % len(args.gpus) for batch in args.batch_sizes):
        parser.error("Global batch sizes must be divisible by the number of GPUs")
    if args.timeout_seconds < 0:
        parser.error("Timeout must be nonnegative (0 means no timeout)")
    if args.steps < 2 or any(b < 1 for b in args.batch_sizes):
        parser.error("Use at least two optimizer steps and positive batch sizes")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            STORAGE / "outputs" / f"vram-benchmark-{len(args.gpus)}gpu-{args.mode}-{stamp}"
        )
    if args.worker:
        return worker(args)
    if not (args.dataset / "SYNTHETIC_DATA.json").is_file():
        parser.error("Synthetic dataset missing; run make_synthetic_gr00t_data.py first")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Call this script directly with --gpus; it launches torchrun itself")
    args.output = args.output.resolve()
    gpu_uuids = []
    for gpu in args.gpus:
        gpu_uuid, baseline = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=uuid,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split(",")
        )
        if float(baseline) > 256:
            parser.error(
                f"GPU {gpu} already uses {baseline.strip()} MiB. Stop keepalive/other jobs first."
            )
        gpu_uuids.append(gpu_uuid.strip())
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Benchmark output: {args.output}", flush=True)
    results = []
    for batch in args.batch_sizes:
        target = args.output / f"batch-{batch}"
        command = [sys.executable, "-u"]
        if len(args.gpus) > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={len(args.gpus)}",
                "--max-restarts=0",
            ]
        command += [
            str(Path(__file__).resolve()),
            "--worker",
            "--batch-sizes",
            str(batch),
            "--steps",
            str(args.steps),
            "--gpu",
            str(args.gpu),
            "--mode",
            args.mode,
            "--dataset",
            str(args.dataset.resolve()),
            "--model",
            str(args.model.resolve()),
            "--output",
            str(target),
            "--gpus",
            *(str(gpu) for gpu in args.gpus),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "NCCL_DEBUG": "WARN",
        }
        print(
            f"Running {args.mode} tuning, global batch {batch}, {len(args.gpus)} GPU(s), {args.steps} real optimizer steps",
            flush=True,
        )
        with (args.output / f"batch-{batch}.log").open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=args.timeout_seconds or None)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)  # Only this newly created benchmark group.
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        if len(args.gpus) > 1:
            rank_results = []
            for rank in range(len(args.gpus)):
                path = target / f"rank-{rank}/result.json"
                rank_results.append(
                    json.loads(path.read_text())
                    if path.exists()
                    else {
                        "rank": rank,
                        "status": "missing_report",
                    }
                )
            status = (
                "ok"
                if proc.returncode == 0 and all(r["status"] == "ok" for r in rank_results)
                else "error"
            )
            if status != "ok":
                log_text = (args.output / f"batch-{batch}.log").read_text(errors="replace")
                if (
                    any(r["status"] == "oom" for r in rank_results)
                    or "out of memory" in log_text.lower()
                ):
                    status = "oom"
            aggregate = {
                "status": "timeout" if timed_out else status,
                "batch_size": batch,
                "global_batch_size": batch,
                "per_gpu_batch_size": batch // len(args.gpus),
                "world_size": len(args.gpus),
                "tuning_mode": args.mode,
                "device_peaks_gib": [r.get("device_sampled_peak_gib") for r in rank_results],
                "rank_results": rank_results,
                "exit_code": proc.returncode,
                "command": command,
            }
            target.mkdir(parents=True, exist_ok=True)
            (target / "result.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        result_path = target / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "status": "error",
                "batch_size": batch,
                "exit_code": proc.returncode,
            }
        )
        results.append(result)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in (
                        "status",
                        "batch_size",
                        "device_sampled_peak_gib",
                        "device_peaks_gib",
                        "final_memory",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        if proc.returncode or result.get("status") != "ok":
            print(
                f"Stopping after failed batch; inspect {args.output / f'batch-{batch}.log'}",
                flush=True,
            )
            return proc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
