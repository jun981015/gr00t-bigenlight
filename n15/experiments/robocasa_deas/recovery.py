"""DDP-only full-state checkpoint commits. Never prune weights or unowned checkpoints."""

import json
import os
from pathlib import Path
import re
import time

import torch
from transformers import TrainerCallback

MARKER = "resume_complete.json"
LATEST = "latest_resumable.json"
CHECKPOINT = re.compile(r"checkpoint-([0-9]+)")
STATE_FILE = re.compile(r"(?:optimizer\.pt|scheduler\.pt|rng_state(?:_[0-9]+)?\.pth|scaler\.pt)")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe checkpoint file: {name}")
    result = root / relative
    if any(p.is_symlink() for p in (result, *result.parents) if p != root):
        raise ValueError(f"Symlink checkpoint member: {result}")
    if not result.is_file() or result.stat().st_size == 0:
        raise ValueError(f"Missing or empty checkpoint file: {result}")
    return result


def inventory(checkpoint, world_size):
    checkpoint = Path(checkpoint)
    match = CHECKPOINT.fullmatch(checkpoint.name)
    if checkpoint.is_symlink() or not match or world_size < 1:
        raise ValueError("Expected a real checkpoint-N directory and positive world size")
    checkpoint = checkpoint.resolve()
    step = int(match[1])
    state = json.loads(safe_file(checkpoint, "trainer_state.json").read_text())
    if state["global_step"] != step:
        raise ValueError("Trainer state/checkpoint step mismatch")
    names = ["config.json", "trainer_state.json", "experiment_cfg/metadata.json"]
    weights = []
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (checkpoint / index).exists():
            mapping = json.loads(safe_file(checkpoint, index).read_text())["weight_map"]
            weights = [index, *sorted(set(mapping.values()))]
            break
    if not weights:
        weights = [n for n in ("model.safetensors", "pytorch_model.bin") if (checkpoint / n).exists()]
    if not weights:
        raise ValueError("Missing model weights")
    training_files = ["optimizer.pt", "scheduler.pt"]
    training_files += ["rng_state.pth"] if world_size == 1 else [f"rng_state_{r}.pth" for r in range(world_size)]
    if (checkpoint / "scaler.pt").exists():
        training_files.append("scaler.pt")
    sizes = {n: safe_file(checkpoint, n).stat().st_size for n in names + weights + training_files}
    return {"version": 1, "step": step, "world_size": world_size, "files": sizes,
            "training_files": training_files}


def verify(checkpoint, manifest):
    if manifest.get("version") != 1 or checkpoint.name != f"checkpoint-{manifest['step']}":
        raise ValueError("Invalid checkpoint manifest")
    actual = inventory(checkpoint, manifest["world_size"])
    if actual != manifest:
        raise ValueError("Checkpoint inventory changed since commit")
    return manifest


def latest_checkpoint(root, world_size):
    root = Path(root).resolve()
    name = json.loads((root / LATEST).read_text())["checkpoint"]
    if not CHECKPOINT.fullmatch(name) or (root / name).is_symlink():
        raise ValueError("Unsafe resume pointer")
    checkpoint = root / name
    manifest = verify(checkpoint, json.loads((checkpoint / MARKER).read_text()))
    if manifest["world_size"] != world_size:
        raise ValueError("Resume with the original GPU count to restore every RNG state")
    return checkpoint


def commit(checkpoint, world_size, prune=True):
    manifest = inventory(checkpoint, world_size)
    checkpoint = Path(checkpoint).resolve()
    atomic_json(checkpoint / MARKER, manifest)
    atomic_json(checkpoint.parent / LATEST, {"checkpoint": checkpoint.name})
    if prune:
        for old in sorted(checkpoint.parent.iterdir()):
            match = CHECKPOINT.fullmatch(old.name)
            if old.is_symlink() or not match or int(match[1]) >= manifest["step"] or not (old / MARKER).is_file():
                continue
            previous = verify(old, json.loads((old / MARKER).read_text()))
            if not all(STATE_FILE.fullmatch(name) for name in previous["training_files"]):
                raise ValueError("Refusing to prune non-training files")
            victims = [safe_file(old, name) for name in previous["training_files"]]
            atomic_json(old / "model_only.json", {"superseded_by": checkpoint.name})
            (old / MARKER).unlink()
            for victim in victims:
                victim.unlink()
    return manifest


class RecoveryCallback(TrainerCallback):
    def __init__(self, policy):
        self.policy = policy

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_save = time.monotonic()
        state.save_steps = args.save_steps
        # HF restores cadence from trainer_state.json; apply the requested
        # intervals after resume without changing optimizer or scheduler state.
        state.logging_steps = getattr(args, "logging_steps", 10)

    def on_step_end(self, args, state, control, **kwargs):
        stop = save = False
        if state.is_world_process_zero:
            now = time.monotonic()
            stop = now - self.started >= self.policy["max_run_seconds"]
            stop |= (Path(args.output_dir) / "STOP_AFTER_CHECKPOINT").exists()
            save = stop or state.global_step == self.policy["first_save_step"]
            save |= now - self.last_save >= self.policy["save_interval_seconds"]
            save |= state.global_step >= state.max_steps
        if torch.distributed.is_initialized():
            flags = torch.tensor([int(save), int(stop)], device=args.device)
            torch.distributed.broadcast(flags, src=0)
            save, stop = flags.tolist()
        control.should_save |= bool(save)
        control.should_training_stop |= bool(stop)
        return control

    def on_save(self, args, state, control, **kwargs):
        distributed = torch.distributed.is_initialized()
        if distributed:
            torch.distributed.barrier()
        error = [None]
        if state.is_world_process_zero:
            try:
                path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                commit(path, args.world_size, self.policy["keep_latest_training_state"])
                print(f"Committed resumable checkpoint: {path}", flush=True)
            except Exception as exc:
                error[0] = repr(exc)
        if distributed:
            torch.distributed.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(f"Checkpoint commit failed; earlier full state retained: {error[0]}")
        self.last_save = time.monotonic()
        return control
