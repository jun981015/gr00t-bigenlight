"""Dry-run-first N1.7 BC -> filtered BC -> SVF recipe; no implicit GPU launches."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / "recipe.json").read_text())
REPO = HERE.parents[1]


def root():
    return Path(os.environ.get("VLA_STORAGE_ROOT", Path.home() / "raid/vla_finetune")).resolve()


def dataset_paths(stage):
    base = root() / "datasets/robocasa_n17"
    if stage == "bc24":
        return [base / "bc24"]
    groups = ("demos", "success_rollouts" if stage == "filtered-bc" else "rollouts")
    return [base / "offline4" / group / task for group in groups for task in RECIPE["tasks"]]


def command(args):
    python = str(root() / "envs/gr00t-n1.7/bin/python")
    data = dataset_paths(args.stage)
    if args.stage != "bc24" and args.base_model is None:
        raise ValueError("--base-model must explicitly name the preceding BC checkpoint")
    model = args.base_model or root() / "models/GR00T-N1.7-3B"
    model = model.resolve()
    if args.stage == "svf":
        cfg = RECIPE["svf"]
        if args.num_gpus not in (None, 1):
            raise ValueError("SVF CLI is currently single-device; BC alone supports multi-GPU")
        cmd = [
            python,
            "-m",
            "gr00t.rl.train",
            "--backend",
            "gr00t",
            "--algorithm",
            "svf",
            "--dataset-path",
            *map(str, data),
            "--model-path",
            str(model),
            "--output-dir",
            str(args.output.resolve()),
            "--embodiment-tag",
            RECIPE["embodiment"],
            "--annotation-format",
            RECIPE["annotation_format"],
            "--no-bootstrap-on-truncation",
            "--horizon",
            str(RECIPE["horizon"]),
            "--device",
            "cuda:0",
            "--steps",
            str(args.steps or cfg["steps"]),
            "--batch-size",
            str(args.batch_size or cfg["batch_size"]),
            "--save-every",
            "5000",
            "--first-save-step",
            "100",
            "--save-interval-seconds",
            "1800",
            "--max-run-seconds",
            "28800",
            "--keep-latest-training-state",
            "--wandb-project",
            RECIPE["wandb_project"],
        ]
        for key in (
            "learning_rate",
            "gamma",
            "flow_steps",
            "candidates",
            "kappa",
            "lambda_multiplier",
            "q_aggregation",
        ):
            cmd += ["--" + key.replace("_", "-"), str(cfg[key])]
        if args.resume:
            cmd += ["--resume", str(args.resume.resolve())]
    else:
        cfg = RECIPE["bc"]
        gpus, batch = args.num_gpus or cfg["num_gpus"], args.batch_size or cfg["batch_per_gpu"]
        cmd = [
            python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpus}",
            str(REPO / "gr00t/experiment/launch_finetune.py"),
            "--base-model-path",
            str(model),
            "--dataset-path",
            os.pathsep.join(map(str, data)),
            "--modality-config-path",
            str(HERE / "modality_config.py"),
            "--embodiment-tag",
            RECIPE["embodiment"],
            "--output-dir",
            str(args.output.resolve()),
            "--num-gpus",
            str(gpus),
            "--global-batch-size",
            str(gpus * batch),
            "--max-steps",
            str(args.steps or cfg["steps"]),
            "--learning-rate",
            str(cfg["learning_rate"]),
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
            "--ds-weights-alpha",
            "1.0",
            "--save-steps",
            "5000",
            "--save-total-limit",
            "0",
            "--keep-latest-training-state",
            "--first-save-step",
            "100",
            "--save-interval-seconds",
            "1800",
            "--max-run-seconds",
            "28800",
            "--use-wandb",
            "--wandb-project",
            RECIPE["wandb_project"],
        ]
        if args.resume:
            cmd += ["--resume-checkpoint-path", str(args.resume.resolve())]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bc24", "filtered-bc", "svf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--batch-size", type=int, help="Per GPU for BC; single-device batch for SVF"
    )
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument(
        "--resume", type=Path, help="BC: committed checkpoint directory; SVF: full step-N.pt"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key in ("steps", "batch_size", "num_gpus"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        print("DRY RUN. Restore the GPU container, complete data preparation and add --execute.")
        return
    if not Path("/.dockerenv").exists():
        raise RuntimeError("Start from the allocated GPU container, not the login host")
    subprocess.run(["nvidia-smi", "-L"], check=True)
    for path in dataset_paths(args.stage):
        if not (path / "READY.json").is_file():
            raise ValueError(f"Dataset not prepared: {path}")
    if args.resume:
        if args.stage != "svf":
            from gr00t.experiment.checkpoint_policy import verify_manifest

            verify_manifest(
                args.resume.resolve(),
                json.loads((args.resume / "resume_complete.json").read_text()),
            )
        elif not args.resume.is_file():
            raise FileNotFoundError(args.resume)
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory or explicitly --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(REPO),
        WANDB_PROJECT=RECIPE["wandb_project"],
        WANDB_DIR=str(args.output.resolve()),
        WANDB_LOG_MODEL="false",
        WANDB_WATCH="false",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    with (args.output / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Training log: {args.output / 'train.log'}", flush=True)
        with (args.output / "train.log").open("a") as log:
            result = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        (args.output / "train.exit").write_text(str(result.returncode) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
