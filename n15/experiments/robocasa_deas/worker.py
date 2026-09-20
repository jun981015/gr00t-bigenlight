"""One torchrun worker; HF handles W&B only on rank zero, with restart identity retained."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from recovery import RecoveryCallback, atomic_json, latest_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["bc24", "filtered-bc", "critic"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(args.config)
    logging_steps = config.pop("logging_steps", None)
    recipe = json.loads((HERE / "recipe.json").read_text())
    module = importlib.import_module("scripts.gr00t_deas_critic_finetune" if args.stage == "critic"
                                     else "scripts.gr00t_finetune")
    base = module.CriticTrainRunner if args.stage == "critic" else module.TrainRunner

    class RecipeRunner(base):
        def __init__(self, **kwargs):
            training_args = kwargs["training_args"]
            # All previous standalone weights are retained; only owned optimizer states are pruned.
            training_args.save_total_limit = None
            if logging_steps is not None:
                training_args.logging_steps = logging_steps
            if config["resume"]:
                kwargs["resume_from_checkpoint"] = str(latest_checkpoint(config["output_dir"], config["num_gpus"]))
            if args.stage == "critic":
                # Inference loads the backbone from the actor, never from the critic checkpoint.
                # Freeze ALL shared backbone parameters, including eagle_linear (upstream leaves it trainable).
                kwargs["model"].backbone.requires_grad_(False)
                self.align_actor_statistics(kwargs["train_dataset"])
            super().__init__(**kwargs)
            self.trainer.add_callback(RecoveryCallback(recipe["recovery"]))

        @staticmethod
        def align_actor_statistics(dataset):
            # Both heads see actor-normalized inputs at inference; use those same stats for critic fitting.
            metadata_file = Path(config["base_model_path"]) / "experiment_cfg/metadata.json"
            actor_metadata = json.loads(metadata_file.read_text())
            for tag, metadata in dataset.merged_metadata.items():
                if tag not in actor_metadata:
                    raise ValueError(f"Actor metadata missing embodiment {tag}")
                payload = metadata.model_dump(mode="json")
                payload["statistics"] = actor_metadata[tag]["statistics"]
                dataset.merged_metadata[tag] = type(metadata).model_validate(payload)
            for single in dataset.datasets:
                single.set_transforms_metadata(dataset.merged_metadata[single.tag])

    # Rank 0's HF integration initializes W&B after setup. Launcher and other ranks do not create runs.
    if int(os.environ.get("RANK", "0")) == 0:
        root = Path(config["output_dir"])
        identity = root / "wandb_resume.json"
        if not identity.exists():
            atomic_json(identity, {"id": uuid.uuid4().hex[:8], "project": recipe["wandb_project"]})
        wandb_identity = json.loads(identity.read_text())
        os.environ.update(WANDB_RUN_ID=wandb_identity["id"], WANDB_PROJECT=wandb_identity["project"],
                          WANDB_RESUME="allow", WANDB_DIR=str(root))
    module.main(module.ArgsConfig(**config), runner_class=RecipeRunner)


if __name__ == "__main__":
    main()
