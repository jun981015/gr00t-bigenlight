"""Resume the last fully committed checkpoint using its original training recipe."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from gr00t.configs.base_config import Config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from gr00t.experiment.checkpoint_policy import STOP_FILE, latest_resumable, verify_manifest


def restored_config(root, checkpoint, *, save_steps=2000, max_run_seconds=32400):
    config = Config.from_pretrained(checkpoint / "experiment_cfg/config.yaml")
    # Config's safe YAML loader deliberately returns nested modality entries as
    # dictionaries. Rehydrate those types, including lowercase enum values.
    for modalities in config.data.modality_configs.values():
        for key, value in modalities.items():
            if not isinstance(value, dict):
                continue
            value = dict(value)
            if value.get("action_configs") is not None:
                value["action_configs"] = [
                    ActionConfig(
                        rep=ActionRepresentation(action["rep"]),
                        type=ActionType(action["type"]),
                        format=ActionFormat(action["format"]),
                        state_key=action.get("state_key"),
                    )
                    if isinstance(action, dict)
                    else action
                    for action in value["action_configs"]
                ]
            modalities[key] = ModalityConfig(**value)
    config.training.output_dir = str(root)
    config.training.experiment_name = None
    config.training.resume_from_checkpoint = False
    config.training.resume_checkpoint_path = str(checkpoint)
    config.training.save_only_model = False
    config.training.save_steps = save_steps
    config.training.save_total_limit = 0
    config.training.keep_latest_training_state = True
    config.training.max_run_seconds = max_run_seconds
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--max-run-seconds", type=float, default=32400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.save_steps <= 0 or args.max_run_seconds <= 0:
        parser.error("save-steps and max-run-seconds must be positive")
    root = args.run_dir.resolve(strict=True)
    if args.worker:
        checkpoint = args.checkpoint.resolve(strict=True)
        if checkpoint.parent != root:
            raise ValueError("Checkpoint must belong to the requested run")
        manifest = json.loads((checkpoint / "resume_complete.json").read_text())
        verify_manifest(checkpoint, manifest)
    else:
        checkpoint, manifest = latest_resumable(root)
    config = restored_config(
        root, checkpoint, save_steps=args.save_steps, max_run_seconds=args.max_run_seconds
    )
    if config.training.num_gpus != manifest["world_size"]:
        raise ValueError("Resume with the same GPU world size as the saved optimizer")
    if manifest["step"] >= config.training.max_steps:
        raise ValueError("This run has already reached its planned max_steps")
    if args.worker:
        from gr00t.configs.data.embodiment_configs import register_modality_config
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.experiment import run

        for tag, modalities in config.data.modality_configs.items():
            register_modality_config(modalities, embodiment_tag=EmbodimentTag(tag))
        run(config)
        return

    subprocess.run([sys.executable, str(Path(__file__).with_name("gpu_guard.py"))], check=True)
    env = dict(os.environ)
    storage = Path(os.environ["VLA_STORAGE_ROOT"])
    env.update(
        {
            "WANDB_DIR": str(root),
            "WANDB_CACHE_DIR": str(storage / "cache/wandb"),
            "WANDB_DATA_DIR": str(storage / "cache/wandb-data"),
            "WANDB_ARTIFACT_DIR": str(storage / "artifacts/wandb"),
            "WANDB_LOG_MODEL": "false",
            "WANDB_WATCH": "false",
            "WANDB_DISABLE_CODE": "true",
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if config.training.use_wandb:
        wb = manifest.get("wandb") or json.loads((root / "wandb_resume.json").read_text())
        if wb["project"] != config.training.wandb_project:
            raise ValueError("W&B project differs from the saved recipe")
        env.update(
            {
                "WANDB_RUN_ID": wb["id"],
                "WANDB_ENTITY": wb["entity"],
                "WANDB_RESUME": "must",
                "WANDB_MODE": "online",
            }
        )
    for key in ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    stop = root / STOP_FILE
    if stop.exists():
        stop.rename(root / f"{STOP_FILE}.consumed-{stamp}")
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config.training.num_gpus}",
        str(Path(__file__).resolve()),
        str(root),
        "--worker",
        "--checkpoint",
        str(checkpoint),
        "--save-steps",
        str(args.save_steps),
        "--max-run-seconds",
        str(args.max_run_seconds),
    ]
    print(
        f"Resuming {checkpoint}; original max_steps={config.training.max_steps}, global batch={config.training.global_batch_size}",
        flush=True,
    )
    with (root / f"resume-{stamp}.log").open("x") as log:
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
