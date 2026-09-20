"""Committed resumable checkpoints; retain older standalone models, not optimizer copies."""

import json
import logging
import os
from pathlib import Path
import re
import time

from transformers import TrainerCallback

from gr00t.experiment.utils import _broadcast_save_decision
from gr00t.utils.dist_utils import barrier, run_or_wait_on_rank0


MARKER = "resume_complete.json"
LATEST = "latest_resumable.json"
STOP_FILE = "STOP_AFTER_CHECKPOINT"
CHECKPOINT = re.compile(r"checkpoint-(\d+)")
logger = logging.getLogger(__name__)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe checkpoint member: {name}")
    target = root / relative
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink checkpoint member: {name}")
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"Missing or empty checkpoint file: {target}")
    return target


def inspect_checkpoint(checkpoint, world_size):
    """Check the required HF/DeepSpeed inventory, without unpickling checkpoint files."""
    checkpoint = Path(checkpoint)
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if checkpoint.is_symlink() or not CHECKPOINT.fullmatch(checkpoint.name):
        raise ValueError("Expected a real checkpoint-N directory")
    checkpoint = checkpoint.resolve()
    state_file = _safe_file(checkpoint, "trainer_state.json")
    state = json.loads(state_file.read_text())
    step = int(CHECKPOINT.fullmatch(checkpoint.name)[1])
    if state["global_step"] != step or step <= 0:
        raise ValueError("Checkpoint step and TrainerState disagree")
    model_files = None
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (checkpoint / index_name).exists():
            index = json.loads(_safe_file(checkpoint, index_name).read_text())
            model_files = [index_name, *sorted(set(index["weight_map"].values()))]
            break
    if model_files is None:
        model_files = [
            name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (checkpoint / name).is_file()
        ]
    if not model_files:
        raise ValueError("No standalone model weights; cannot retain a model-only checkpoint")
    training_files = ["scheduler.pt"] + (
        ["rng_state.pth"]
        if world_size == 1
        else [f"rng_state_{rank}.pth" for rank in range(world_size)]
    )
    if (checkpoint / "latest").exists():
        tag = _safe_file(checkpoint, "latest").read_text().strip()
        if tag != f"global_step{step}" or (checkpoint / tag).is_symlink():
            raise ValueError("Invalid DeepSpeed checkpoint tag")
        optimizer_files = sorted((checkpoint / tag).glob("*_optim_states.pt"))
        model_states = sorted((checkpoint / tag).glob("*_model_states.pt"))
        if len(optimizer_files) != world_size or not model_states:
            raise ValueError("Incomplete DeepSpeed optimizer/model shards")
        training_files += ["latest"] + [
            str(path.relative_to(checkpoint)) for path in optimizer_files + model_states
        ]
        backend = "deepspeed"
    else:
        training_files += ["optimizer.pt"]
        backend = "torch"
    if (checkpoint / "scaler.pt").exists():
        training_files.append("scaler.pt")
    files = {
        name: _safe_file(checkpoint, name).stat().st_size
        for name in ["trainer_state.json", "training_args.bin", *model_files, *training_files]
    }
    return {
        "version": 1,
        "step": step,
        "world_size": world_size,
        "backend": backend,
        "files": files,
        "training_state_files": training_files,
    }


def verify_manifest(checkpoint, manifest):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported checkpoint manifest")
    if checkpoint.name != f"checkpoint-{manifest['step']}":
        raise ValueError("Manifest directory mismatch")
    allowed_state = re.compile(
        rf"(?:optimizer\.pt|scheduler\.pt|scaler\.pt|latest|rng_state(?:_\d+)?\.pth|"
        rf"global_step{manifest['step']}/[^/]+_(?:optim|model)_states\.pt)"
    )
    if not all(allowed_state.fullmatch(name) for name in manifest["training_state_files"]):
        raise ValueError("Manifest attempts to prune a non-training-state file")
    for name, size in manifest["files"].items():
        if _safe_file(checkpoint, name).stat().st_size != size:
            raise ValueError(f"Checkpoint file size mismatch: {checkpoint / name}")
    if (
        json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
        != manifest["step"]
    ):
        raise ValueError("Manifest step mismatch")


def latest_resumable(root):
    """Use only the last committed full checkpoint, never an incomplete newer directory."""
    root = Path(root).resolve()
    record = json.loads((root / LATEST).read_text())
    name = record["checkpoint"]
    if not CHECKPOINT.fullmatch(name) or (root / name).is_symlink():
        raise ValueError("Unsafe latest-checkpoint pointer")
    checkpoint = root / name
    manifest = json.loads((checkpoint / MARKER).read_text())
    verify_manifest(checkpoint, manifest)
    return checkpoint, manifest


def commit_checkpoint(checkpoint, world_size, *, prune=True):
    checkpoint = Path(checkpoint)
    manifest = inspect_checkpoint(checkpoint, world_size)
    checkpoint = checkpoint.resolve()
    root = checkpoint.parent
    metadata = root / "wandb_resume.json"
    if metadata.exists():
        manifest["wandb"] = json.loads(metadata.read_text())
    # Do not touch the previous resume state until the new inventory is complete.
    atomic_json(checkpoint / MARKER, manifest)
    atomic_json(root / LATEST, {"checkpoint": checkpoint.name, "step": manifest["step"]})
    logger.info("Committed full checkpoint: %s (step %s)", checkpoint, manifest["step"])
    if not prune:
        return manifest
    for old in sorted(root.iterdir()):
        match = CHECKPOINT.fullmatch(old.name)
        if old.is_symlink() or not match or int(match[1]) >= manifest["step"]:
            continue
        old_marker = old / MARKER
        if not old_marker.exists():
            continue  # Never prune an unowned / uncommitted checkpoint.
        previous = json.loads(old_marker.read_text())
        verify_manifest(old, previous)
        # Explicit inventory only; no recursive deletion of checkpoint directories.
        victims = [_safe_file(old, name) for name in previous["training_state_files"]]
        # Mark model-only first: interruption during pruning can only leave extra files.
        atomic_json(
            old / "model_only.json",
            {
                "step": previous["step"],
                "resumable": False,
                "superseded_by": checkpoint.name,
            },
        )
        old_marker.unlink()
        removed = 0
        for path in victims:
            removed += path.stat().st_size
            path.unlink()
        logger.info(
            "Retained %s model; removed %s bytes of older training state", old.name, removed
        )
    return manifest


class ResumableCheckpointCallback(TrainerCallback):
    """Mark saves complete on all ranks, prune old state, and save before a time budget ends."""

    def __init__(
        self,
        *,
        keep_latest=True,
        max_run_seconds=None,
        first_save_step=None,
        save_interval_seconds=None,
        save_at_steps=(),
    ):
        self.keep_latest = keep_latest
        self.max_run_seconds = max_run_seconds
        self.first_save_step = first_save_step
        self.save_interval_seconds = save_interval_seconds
        if any(step <= 0 for step in save_at_steps):
            raise ValueError("save_at_steps must contain positive steps")
        self.save_at_steps = frozenset(save_at_steps)

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_save = time.monotonic()
        # HF restores the previous TrainerState; explicitly apply the requested new cadence.
        state.save_steps = args.save_steps
        if state.is_world_process_zero:
            scheduler = kwargs.get("lr_scheduler")
            scheduler = getattr(scheduler, "scheduler", scheduler)
            groups = getattr(kwargs.get("optimizer"), "param_groups", [])
            logger.info(
                "Training state ready: global_step=%s, scheduler.last_epoch=%s, lr=%s, save_steps=%s",
                state.global_step,
                getattr(scheduler, "last_epoch", None),
                [group["lr"] for group in groups],
                state.save_steps,
            )

    def on_step_end(self, args, state, control, **kwargs):
        should_stop = 0
        should_save = 0
        if state.is_world_process_zero:
            timed_out = (
                self.max_run_seconds is not None
                and time.monotonic() - self.started >= self.max_run_seconds
            )
            should_stop = int(timed_out or (Path(args.output_dir) / STOP_FILE).exists())
            should_save = int(
                state.global_step == self.first_save_step
                or state.global_step in self.save_at_steps
                or (
                    self.save_interval_seconds is not None
                    and time.monotonic() - self.last_save >= self.save_interval_seconds
                )
                or state.global_step == getattr(state, "max_steps", None)
            )
        should_stop, should_save = _broadcast_save_decision(should_stop, float(should_save))
        if should_save:
            control.should_save = True
        if should_stop:
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                logger.info(
                    "Saving full state and stopping at step %s (time budget / stop request)",
                    state.global_step,
                )
        return control

    def on_save(self, args, state, control, **kwargs):
        barrier()  # All optimizer and RNG shards, on every rank, must have finished writing.
        with run_or_wait_on_rank0(label="commit_checkpoint") as is_rank0:
            if is_rank0:
                commit_checkpoint(
                    Path(args.output_dir) / f"checkpoint-{state.global_step}",
                    args.world_size,
                    prune=self.keep_latest,
                )
        self.last_save = time.monotonic()
        return control
