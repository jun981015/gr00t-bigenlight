"""Dry-run-first LIBERO BC with frozen VLM; refuses concurrent GPU training."""

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from .data import HERE, RECIPE, dataset_path, storage


REPO = HERE.parents[2]
PRESETS = json.loads((HERE / "qvgm_presets.json").read_text())["profiles"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("suite", *PRESETS), default="suite")
    parser.add_argument("--suite", choices=tuple(RECIPE["datasets"]))
    parser.add_argument("--demos-per-task", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--precision", choices=("bf16-mixed", "fp32"))
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument(
        "--gpu", type=int, choices=(0, 1), help="Physical GPU index for one-GPU runs"
    )
    parser.add_argument("--batch-per-gpu", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--save-at-steps", type=int, nargs="+", default=[])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-run-seconds", type=float, default=28800)
    parser.add_argument("--wandb-project", default="libero-gr00t")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.profile == "suite":
        args.suite = args.suite or "libero_spatial"
        args.suites = [args.suite]
        args.demos_per_task = 0 if args.demos_per_task is None else args.demos_per_task
        args.steps = 20000 if args.steps is None else args.steps
        args.precision = args.precision or "bf16-mixed"
    else:
        preset = PRESETS[args.profile]
        if args.suite is not None or args.demos_per_task is not None:
            parser.error(
                "A Q-VGM profile fixes its suites and demo count; use --profile suite for custom data"
            )
        args.suites = preset["suites"]
        args.demos_per_task = preset["demos_per_task"]
        args.steps = preset["steps"] if args.steps is None else args.steps
        args.precision = args.precision or preset["precision"]
        if args.steps is None:
            parser.error(
                "qvgm-unified requires --steps: Q-VGM does not report its reused checkpoint's SFT update count"
            )
    if args.demos_per_task < 0 or args.seed < 0:
        parser.error("demos-per-task and seed must be nonnegative")
    for key in (
        "steps",
        "num_gpus",
        "batch_per_gpu",
        "save_steps",
        "learning_rate",
        "max_run_seconds",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.num_gpus not in (1, 2):
        parser.error("This server recipe supports one or two GPUs")
    if args.num_gpus == 1 and args.gpu is None:
        parser.error("One-GPU runs require explicit --gpu 0 or --gpu 1")
    if args.num_gpus == 2 and args.gpu is not None:
        parser.error("--gpu selects exactly one GPU; set --num-gpus 1")
    if any(step <= 0 or step > args.steps for step in args.save_at_steps):
        parser.error("save-at-steps must be positive and not exceed steps")
    args.save_at_steps = sorted(set(args.save_at_steps))
    return args


def command(args):
    data = dataset_paths(args)
    model = (args.base_model or storage() / "models" / RECIPE["model"]).resolve()
    cmd = [
        str(storage() / "envs/gr00t-n1.7/bin/python"),
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.num_gpus}",
        str(REPO / "gr00t/experiment/launch_finetune.py"),
        "--base-model-path",
        str(model),
        "--dataset-path",
        os.pathsep.join(map(str, data)),
        "--embodiment-tag",
        RECIPE["embodiment"],
        "--output-dir",
        str(args.output.resolve()),
        "--num-gpus",
        str(args.num_gpus),
        "--global-batch-size",
        str(args.num_gpus * args.batch_per_gpu),
        "--max-steps",
        str(args.steps),
        "--precision",
        args.precision,
        "--learning-rate",
        str(args.learning_rate),
        "--no-tune-llm",
        "--no-tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        "--dataloader-num-workers",
        "2",
        "--shard-size",
        "128",
        "--num-shards-per-epoch",
        "128",
        "--episode-sampling-rate",
        "1.0",
        "--shortest-image-edge",
        "256",
        "--crop-fraction",
        "1.0",
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        "0",
        "--keep-latest-training-state",
        "--max-run-seconds",
        str(args.max_run_seconds),
        "--use-wandb",
        "--wandb-project",
        args.wandb_project,
    ]
    if args.save_at_steps:
        cmd += ["--save-at-steps", *map(str, args.save_at_steps)]
    return cmd


def dataset_paths(args):
    return [dataset_path(suite, args.demos_per_task, args.seed) for suite in args.suites]


def check_ready(args):
    manifests = []
    for suite, data in zip(args.suites, dataset_paths(args), strict=True):
        ready = json.loads((data / "READY.json").read_text())
        validation = json.loads((data / "VALIDATION.json").read_text())
        expected = ready["provenance"]
        if expected != validation["provenance"] or expected["dataset"] != RECIPE["datasets"][suite]:
            raise ValueError("Dataset provenance/validation mismatch")
        if expected["demos_per_task"] != args.demos_per_task or expected["seed"] != args.seed:
            raise ValueError("Prepared selection differs from requested selection")
        if validation["validated_tasks"] != 10:
            raise ValueError("All ten tasks must pass CPU decoding")
        if args.demos_per_task and (
            ready["episodes"] != 10 * args.demos_per_task
            or len(ready["task_counts"]) != 10
            or set(ready["task_counts"].values()) != {args.demos_per_task}
        ):
            raise ValueError("Prepared few-shot count differs from requested count")
        manifests.append(ready)
    model = (args.base_model or storage() / "models" / RECIPE["model"]).resolve()
    if not (model / "config.json").is_file():
        raise FileNotFoundError(model)
    output = args.output.resolve()
    if not output.is_relative_to(storage()):
        raise ValueError("Keep LIBERO checkpoints on VLA_STORAGE_ROOT (RAID)")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Choose an empty output directory; use the recovery launcher for resume"
        )
    return manifests


def selected_gpu_uuids(args, inventory):
    devices = {}
    for line in inventory.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        if not uuid.startswith("GPU-") or any(c not in "0123456789abcdefABCDEF-" for c in uuid[4:]):
            raise ValueError("Unexpected GPU UUID")
        devices[int(index)] = uuid
    indices = [args.gpu] if args.gpu is not None else [0, 1]
    if any(index not in devices for index in indices):
        raise ValueError("Requested GPU is not visible")
    return [devices[index] for index in indices]


def main(argv=None):
    args = parse_args(argv)
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        print("DRY RUN: no training launched. Add --execute only after the current GPU job ends.")
        return
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Run in the allocated GPU container, not the login host")
    manifests = check_ready(args)
    env = os.environ.copy()
    # A new dataset experiment must not accidentally reuse the carrot W&B identity.
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10
    )
    gpu_uuids = selected_gpu_uuids(args, inventory)
    keepalive = REPO / "examples/carrot_in_pot/ensure_keepalive.py"
    env["CARROT_KEEPALIVE_PIDS"] = subprocess.check_output(
        [cmd[0], str(keepalive)], text=True, env=env
    ).strip()
    output = args.output.resolve()
    env.update(
        WANDB_PROJECT=args.wandb_project,
        WANDB_DIR=str(output),
        WANDB_MODE="online",
        WANDB_LOG_MODEL="false",
        WANDB_WATCH="false",
        WANDB_CACHE_DIR=str(storage() / "cache/wandb"),
        WANDB_DATA_DIR=str(storage() / "cache/wandb-data"),
        WANDB_ARTIFACT_DIR=str(storage() / "artifacts/wandb"),
        TOKENIZERS_PARALLELISM="false",
    )
    with ExitStack() as locks:
        lock_root = storage() / "logs/gpu-launch-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        for uuid in sorted(gpu_uuids):
            lock = locks.enter_context((lock_root / f"{uuid}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            subprocess.run(
                [cmd[0], str(keepalive.with_name("gpu_guard.py")), "--gpu-uuid", uuid],
                env=env,
                check=True,
            )
        output.mkdir(parents=True, exist_ok=True)
        lock = locks.enter_context((output / ".launch.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
        from .data import atomic_json

        atomic_json(
            output / "launch.json",
            {
                "command": cmd,
                "datasets": list(map(str, dataset_paths(args))),
                "dataset_manifests": manifests,
                "profile": args.profile,
                "precision": args.precision,
                "selection_seed": args.seed,
                "reference": PRESETS.get(args.profile),
                "gpu_uuids": gpu_uuids,
                "save_at_steps": args.save_at_steps,
            },
        )
        print(f"Log: {output / 'train.log'}", flush=True)
        with (output / "train.log").open("x") as log:
            process = subprocess.run(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        (output / "train.exit").write_text(str(process.returncode) + "\n")
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
