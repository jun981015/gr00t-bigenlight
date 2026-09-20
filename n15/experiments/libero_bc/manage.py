"""Prepare N1.5 LIBERO BC and print launch commands; training requires --execute."""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from data import HERE, PROFILES, REPO, atomic_json, paths, prepare, storage, validate

RECOVERY = {
    "first_save_step": 100,
    "save_interval_seconds": 1800,
    "max_run_seconds": 28800,
    "keep_latest_training_state": True,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare")
    data.add_argument("--profile", choices=PROFILES, required=True)
    data.add_argument("--validate", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("--profile", choices=PROFILES, required=True)
    train.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    train.add_argument("--batch-size", type=int, default=256, help="Effective batch for this one-GPU run")
    train.add_argument("--micro-batch-size", type=int, default=32)
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument(
        "--after-steps", type=int, default=0, help="Extra unsaved temporary steps after successful BC completion"
    )
    train.add_argument("--no-checkpoints", action="store_true", help="No model/optimizer saves or W&B")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        if min(args.batch_size, args.micro_batch_size, args.steps) <= 0:
            parser.error("Batch sizes and steps must be positive")
        if args.batch_size % args.micro_batch_size:
            parser.error("Effective batch must be divisible by micro-batch")
        if args.after_steps < 0 or (args.no_checkpoints and (args.resume or args.after_steps)):
            parser.error("after-steps must be nonnegative; unsaved runs cannot resume or chain")
    return args


def training_recipe(args):
    return {
        "profile": args.profile,
        "action_horizon": 16,
        "precision": "bf16-mixed",
        "checkpoints": not args.no_checkpoints,
        "after_steps": args.after_steps,
        "effective_batch": args.batch_size,
        "gradient_accumulation_steps": args.batch_size // args.micro_batch_size,
        "save_at_steps": []
        if args.no_checkpoints
        else [s for s in (500, 1000, 2000, 3000, 5000) if s <= args.steps],
        "recovery": RECOVERY,
        "wandb_project": "libero-gr00t-n15",
        "config": {
            "dataset_path": [str(p) for p in paths(args.profile)],
            "output_dir": str(args.output.resolve()),
            "run_name": args.output.name,
            "data_config": "libero",
            "base_model_path": str(storage() / "models/GR00T-N1.5-3B"),
            "batch_size": args.micro_batch_size,
            "num_gpus": 1,
            "max_steps": args.steps,
            "save_steps": 2000,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.05,
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "embodiment_tag": "new_embodiment",
            "dataloader_num_workers": 2,
            "video_backend": "torchvision_av",
            "resume": args.resume,
        },
    }


def launch(args):
    recipe = training_recipe(args)
    python = storage() / "envs/deas-gr00t-n1.5/bin/python"
    command = [
        str(python),
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        str(HERE / "worker.py"),
        "--recipe",
        json.dumps(recipe),
    ]
    print(json.dumps(recipe, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if not args.execute:
        print("DRY RUN: no GPU process or W&B run started. Add --execute only when ready.")
        return
    if not Path("/.dockerenv").exists():
        raise ValueError("Training must run inside the allocated GPU container")
    output = args.output.resolve()
    if not output.is_relative_to(storage() / "outputs") or output == storage() / "outputs":
        raise ValueError("Use a dedicated directory beneath RAID outputs")
    for path in paths(args.profile):
        ready = json.loads((path / "READY.json").read_text())
        checked = json.loads((path / "VALIDATION.json").read_text())
        if ready["provenance"] != checked["provenance"] or checked["validated_tasks"] != 10:
            raise ValueError("Run prepare --validate before training")
    model = Path(recipe["config"]["base_model_path"])
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError("N1.5 weights missing; run the documented pinned model download first")
    subprocess.run([str(python), "-c", "import flash_attn"], check=True)
    if args.resume:
        from experiments.robocasa_deas.recovery import latest_checkpoint

        latest_checkpoint(output, 1)
        previous = json.loads((output / "launch_recipe.json").read_text())
        previous["config"]["resume"] = True
        if previous != recipe:
            raise ValueError("Resume must preserve the original recipe")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output; choose a new run directory")
    env = os.environ.copy()
    for key in ("CUDA_VISIBLE_DEVICES", "WANDB_RUN_ID", "WANDB_RESUME"):
        env.pop(key, None)
    n17 = REPO.parent / "n17/examples/carrot_in_pot"
    guard_python = storage() / "envs/gr00t-n1.7/bin/python"
    env["CARROT_KEEPALIVE_PIDS"] = subprocess.check_output(
        [str(guard_python), str(n17 / "ensure_keepalive.py")], env=env, text=True
    ).strip()
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True, timeout=10
    )
    devices = {int(i.strip()): u.strip() for i, u in (r.split(",") for r in rows.splitlines())}
    gpu = devices[args.gpu]
    if not gpu.startswith("GPU-") or any(c not in "0123456789abcdefABCDEF-" for c in gpu[4:]):
        raise ValueError("Invalid GPU UUID")
    with ExitStack() as locks:
        lock_root = storage() / "logs/gpu-launch-locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock = locks.enter_context((lock_root / f"{gpu}.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([str(guard_python), str(n17 / "gpu_guard.py"), "--gpu-uuid", gpu], env=env, check=True)
        output.mkdir(parents=True, exist_ok=True)
        lock = locks.enter_context((output / ".launch.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Check again under the lock; never replace another run's recipe.
        if not args.resume and (output / "launch_recipe.json").exists():
            raise ValueError("A launch already owns this output directory")
        env.update(
            CUDA_VISIBLE_DEVICES=gpu,
            PYTHONPATH=str(REPO),
            USE_TF="0",
            NO_ALBUMENTATIONS_UPDATE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            WANDB_PROJECT=recipe["wandb_project"],
            WANDB_DIR=str(output),
            WANDB_MODE="online",
            WANDB_LOG_MODEL="false",
            WANDB_WATCH="false",
            WANDB_CACHE_DIR=str(storage() / "cache/wandb"),
            WANDB_DATA_DIR=str(storage() / "cache/wandb-data"),
            WANDB_ARTIFACT_DIR=str(storage() / "artifacts/wandb"),
        )
        if args.resume and (output / "STOP_AFTER_CHECKPOINT").exists():
            (output / "STOP_AFTER_CHECKPOINT").rename(output / f"STOP_AFTER_CHECKPOINT.consumed-{time.time_ns()}")
        atomic_json(output / "launch_recipe.json", recipe)
        with (output / "train.log").open("a") as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / "train.exit").write_text(str(result.returncode) + "\n")
        # Event-driven chaining, not polling; hold this GPU's lock throughout.
        if result.returncode == 0 and recipe["after_steps"]:
            from experiments.libero_bc.temporary import temporary_recipe
            from experiments.robocasa_deas.recovery import latest_checkpoint

            checkpoint = latest_checkpoint(output, 1)
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            if state["global_step"] == recipe["config"]["max_steps"]:
                followup = temporary_recipe(recipe, checkpoint)
                temporary_output = Path(followup["config"]["output_dir"])
                temporary_output.mkdir(exist_ok=False)
                atomic_json(temporary_output / "launch_recipe.json", followup)
                atomic_json(
                    output / "after_started.json", {"output": str(temporary_output), "step": state["global_step"]}
                )
                env.update(WANDB_MODE="disabled", WANDB_DIR=str(temporary_output))
                for key in ("WANDB_RUN_ID", "WANDB_RESUME"):
                    env.pop(key, None)
                temporary_command = command[:-1] + [json.dumps(followup)]
                print(
                    f"BC complete; starting {recipe['after_steps']} temporary steps without checkpoints",
                    flush=True,
                )
                with (temporary_output / "train.log").open("x") as log:
                    result = subprocess.run(
                        temporary_command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                (temporary_output / "train.exit").write_text(str(result.returncode) + "\n")
            else:
                print("BC stopped before max_steps; resume BC before temporary run.", flush=True)
    raise SystemExit(result.returncode)


def main():
    args = parse_args()
    if args.command == "prepare":
        for source, target in zip(paths(args.profile, "n17"), paths(args.profile)):
            prepare(source, target)
            if args.validate:
                validate(target)
            print(target, flush=True)
    else:
        launch(args)


if __name__ == "__main__":
    main()
