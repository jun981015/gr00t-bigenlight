"""Custom UR7e BC launcher. Dry-run by default; no existing jobs are touched."""

import argparse
import json
import os
import uuid
from pathlib import Path

from experiments.bigenlight_multitask.config import UR7eDataConfig
from experiments.robocasa_deas.recovery import RecoveryCallback, atomic_json, latest_checkpoint
from scripts import gr00t_finetune as bc


def main():
    storage = (Path.home() / "raid/vla_finetune").resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=storage / "datasets/bigenlight_multitask_gr00t/n15")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.save_steps) <= 0:
        parser.error("Batch size and step counts must be positive")
    if args.dataloader_num_workers < 0:
        parser.error("Worker count must be nonnegative")
    root = args.output.resolve()
    dataset = args.dataset_path.resolve()
    bc.DATA_CONFIG_MAP["ur7e_multitask"] = UR7eDataConfig()
    config = bc.ArgsConfig(
        dataset_path=[str(dataset)],
        output_dir=str(root),
        data_config="ur7e_multitask",
        base_model_path=str(storage / "models/GR00T-N1.5-3B"),
        batch_size=args.batch_size,
        max_steps=args.steps,
        save_steps=args.save_steps,
        num_gpus=1,
        dataloader_num_workers=args.dataloader_num_workers,
        video_backend="torchvision_av",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        report_to="wandb",
        run_name=root.name,
        resume=args.resume,
    )
    print(json.dumps(vars(config), indent=2), flush=True)
    if not args.execute:
        print("DRY RUN: add --execute to train; no GPU or W&B run started.")
        return
    if not (dataset / "VALIDATION.json").is_file():
        raise ValueError("Validate N1.5 dataset first")
    if not root.is_relative_to(storage / "outputs") or root == storage / "outputs":
        raise ValueError("Use a dedicated directory beneath RAID outputs")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError("Output exists: choose a new directory or explicitly --resume")
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_json(root / "launch_config.json", vars(config))
    identity = root / "wandb_resume.json"
    if not identity.exists():
        if args.resume:
            raise ValueError("Missing W&B resume identity")
        atomic_json(identity, {"id": uuid.uuid4().hex[:8]})
    os.environ.update(
        WANDB_MODE="online",
        WANDB_PROJECT="bigenlight-multitask-gr00t",
        WANDB_RUN_ID=json.loads(identity.read_text())["id"],
        WANDB_DIR=str(root),
        WANDB_RESUME="must" if args.resume else "allow",
        WANDB_LOG_MODEL="false",
    )
    resume = str(latest_checkpoint(root, 1)) if args.resume else False

    class Runner(bc.TrainRunner):
        def __init__(self, **kwargs):
            training = kwargs["training_args"]
            training.logging_steps = 50
            training.save_total_limit = None
            training.save_only_model = False
            kwargs["resume_from_checkpoint"] = resume
            super().__init__(**kwargs)
            self.trainer.add_callback(
                RecoveryCallback(
                    {
                        "first_save_step": 100,
                        "save_interval_seconds": 1800,
                        "max_run_seconds": 28800,
                        "keep_latest_training_state": True,
                    }
                )
            )

    bc.main(config, runner_class=Runner)


if __name__ == "__main__":
    main()
