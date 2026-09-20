"""Run the original DEAS BC entry point without its CUDA/W&B CLI side effects."""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.libero_bc.temporary import NoCheckpointMixin, NoSaveStopCallback  # noqa: E402
from experiments.robocasa_deas.recovery import RecoveryCallback, atomic_json, latest_checkpoint  # noqa: E402
from scripts import gr00t_finetune as bc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    recipe = json.loads(parser.parse_args().recipe)
    config = recipe["config"]
    root = Path(config["output_dir"])
    save_checkpoints = recipe.get("checkpoints", True)

    class Milestones(RecoveryCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control = super().on_step_end(args, state, control, **kwargs)
            control.should_save |= state.global_step in recipe["save_at_steps"]
            return control

    class LiberoRunner(bc.TrainRunner):
        def __init__(self, **kwargs):
            training = kwargs["training_args"]
            training.gradient_accumulation_steps = recipe["gradient_accumulation_steps"]
            training.save_total_limit = None
            training.save_only_model = False
            if not save_checkpoints:
                training.save_strategy = "no"
                training.report_to = []
                training.logging_steps = 100
            if config["resume"]:
                checkpoint = latest_checkpoint(root, 1)
                state = json.loads((checkpoint / "trainer_state.json").read_text())
                if state["global_step"] >= config["max_steps"]:
                    raise ValueError("Run already completed its scheduled steps")
                kwargs["resume_from_checkpoint"] = str(checkpoint)
            model = kwargs["model"]
            if model.action_horizon != recipe["action_horizon"]:
                raise ValueError("Model/data horizon mismatch")
            super().__init__(**kwargs)
            if save_checkpoints:
                self.trainer.add_callback(Milestones(recipe["recovery"]))
            else:
                self.trainer.add_callback(NoSaveStopCallback())

        def create_trainer(self, *args, **kwargs):
            trainer = super().create_trainer(*args, **kwargs)
            if not save_checkpoints:
                trainer.__class__ = type("UnsavedLiberoTrainer", (NoCheckpointMixin, type(trainer)), {})
            return trainer

        def train(self):
            if save_checkpoints:
                return super().train()
            # Skip the original runner's unconditional final model/state writes.
            return self.trainer.train()

    if save_checkpoints and int(os.environ.get("RANK", "0")) == 0:
        identity = root / "wandb_resume.json"
        if not identity.exists():
            if config["resume"]:
                raise ValueError("W&B resume identity missing")
            atomic_json(identity, {"id": uuid.uuid4().hex[:8], "project": recipe["wandb_project"]})
        wandb = json.loads(identity.read_text())
        os.environ.update(
            WANDB_RUN_ID=wandb["id"],
            WANDB_PROJECT=wandb["project"],
            WANDB_RESUME="must" if config["resume"] else "allow",
            WANDB_DIR=str(root),
        )
    bc.main(bc.ArgsConfig(**config), runner_class=LiberoRunner)


if __name__ == "__main__":
    main()
