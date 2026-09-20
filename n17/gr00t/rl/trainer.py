"""Small algorithm-agnostic trainer with explicit, resumable checkpoints."""

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time

import numpy as np
import torch

from .types import OfflineAlgorithm


class OfflineTrainer:
    def __init__(
        self, algorithm: OfflineAlgorithm, device="cpu", batch_encoder=None, metadata=None
    ):
        self.algorithm = algorithm
        self.device, self.batch_encoder = device, batch_encoder
        self.step = 0
        self.stop_requested = False
        self.metadata = deepcopy(metadata)

    def update(self, batch):
        if self.batch_encoder is not None:
            batch = self.batch_encoder(batch)
        batch = batch.to(self.device)
        metrics = self.algorithm.update(batch)
        self.step += 1
        return metrics

    def fit(
        self,
        batches,
        *,
        log_path=None,
        checkpoint_dir=None,
        save_every=0,
        first_save_step=0,
        save_interval_seconds=0,
        max_run_seconds=0,
        keep_latest_training_state=False,
        metrics_callback=None,
    ):
        if save_every < 0:
            raise ValueError("save_every must be nonnegative")
        if any(value < 0 for value in (first_save_step, save_interval_seconds, max_run_seconds)):
            raise ValueError("Recovery intervals must be nonnegative")
        if (
            save_every or first_save_step or save_interval_seconds or max_run_seconds
        ) and checkpoint_dir is None:
            raise ValueError("Recovery saving requires checkpoint_dir")
        started = last_save = time.monotonic()
        log = None
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("a")
        try:
            previous_batch_end = time.perf_counter()
            for batch in batches:
                batch_ready = time.perf_counter()
                metrics = {"step": self.step + 1, **self.update(batch)}
                update_end = time.perf_counter()
                metrics.update(
                    {
                        "time/data_wait_s": batch_ready - previous_batch_end,
                        "time/update_s": update_end - batch_ready,
                        "time/step_s": update_end - previous_batch_end,
                    }
                )
                print(json.dumps(metrics, allow_nan=False), flush=True)
                if log is not None:
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                if metrics_callback is not None:
                    metrics_callback(metrics)
                now = time.monotonic()
                stop = self.stop_requested or bool(
                    max_run_seconds and now - started >= max_run_seconds
                )
                milestone = bool(save_every and self.step % save_every == 0)
                save = milestone or stop or self.step == first_save_step
                save |= bool(save_interval_seconds and now - last_save >= save_interval_seconds)
                if save:
                    self.save_recovery_checkpoint(
                        checkpoint_dir, keep_latest_training_state, milestone or stop
                    )
                    last_save = time.monotonic()
                if stop:
                    break
                previous_batch_end = time.perf_counter()
        finally:
            if log is not None:
                log.close()

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
        state = {
            "format_version": 1,
            "metadata": self.metadata,
            "step": self.step,
            "algorithm": self.algorithm.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        }
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(descriptor)
        try:
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            # Hard-link creates the final path exclusively; a concurrent writer
            # cannot be silently overwritten after the existence check above.
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save_recovery_checkpoint(self, directory, keep_latest=False, archive_model=False):
        """Commit a full state before pruning ONLY the previously owned full-state file."""
        from gr00t.experiment.checkpoint_policy import atomic_json

        directory = Path(directory).resolve()
        path = directory / f"step-{self.step}.pt"
        pointer = directory / "latest_resumable.json"
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        if previous and (
            previous.get("format") != "gr00t-offline-recovery-v1"
            or not re.fullmatch(r"step-[0-9]+\.pt", previous.get("file", ""))
            or previous["file"] != f"step-{previous.get('step')}.pt"
            or previous["step"] >= self.step
        ):
            raise ValueError(
                "Invalid or newer recovery pointer; resume into a fresh output directory"
            )
        self.save_checkpoint(path)
        if not keep_latest:
            return path
        if archive_model:
            # Archive trainable module weights without Adam; the frozen VLM stays in model_path.
            model = {
                key: value
                for key, value in self.algorithm.state_dict().items()
                if key != "optimizer"
            }
            model_path = directory / f"model-step-{self.step}.pt"
            descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".model-", suffix=".pt")
            os.close(descriptor)
            try:
                torch.save(
                    {
                        "model_only": True,
                        "metadata": self.metadata,
                        "step": self.step,
                        "algorithm": model,
                    },
                    temporary,
                )
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.link(temporary, model_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        atomic_json(
            pointer,
            {
                "format": "gr00t-offline-recovery-v1",
                "file": path.name,
                "size": path.stat().st_size,
                "step": self.step,
            },
        )
        if previous and re.fullmatch(r"step-[0-9]+\.pt", previous["file"]):
            old = directory / previous["file"]
            if (
                old != path
                and not old.is_symlink()
                and old.is_file()
                and old.stat().st_size == previous["size"]
            ):
                old.unlink()
        return path

    def load_checkpoint(self, path):
        # This includes optimizer/Python RNG state; load ONLY trusted local files.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported offline RL checkpoint format")
        if state.get("metadata") != self.metadata:
            raise ValueError("Checkpoint data/normalization/training metadata does not match")
        if state["cuda_rng"] is not None and not torch.cuda.is_available():
            raise ValueError("CUDA RNG resume requires the original CUDA execution environment")
        self.algorithm.load_state_dict(state["algorithm"])
        self.step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
