"""Separate, explicitly unsaved post-BC activity."""

from copy import deepcopy
from pathlib import Path

from transformers import TrainerCallback


def temporary_recipe(parent, checkpoint):
    steps = parent["after_steps"]
    if steps <= 0 or not parent["checkpoints"]:
        raise ValueError("Requires saved parent and positive temporary steps")
    config = deepcopy(parent)
    config.update(
        checkpoints=False,
        after_steps=0,
        save_at_steps=[],
        effective_batch=parent["config"]["batch_size"],
        gradient_accumulation_steps=1,
    )
    target = Path(parent["config"]["output_dir"]) / f"temporary-nosave-{steps}"
    config["config"].update(
        output_dir=str(target),
        run_name=target.name,
        max_steps=steps,
        base_model_path=str(checkpoint),
        resume=False,
    )
    return config


class NoCheckpointMixin:
    """Hard guard against periodic and final weight/optimizer saves."""

    def _save_checkpoint(self, *args, **kwargs):
        return None

    def save_model(self, *args, **kwargs):
        return None

    def save_state(self, *args, **kwargs):
        return None


class NoSaveStopCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = False
        if (Path(args.output_dir) / "STOP_AFTER_CHECKPOINT").exists():
            control.should_training_stop = True
        return control
