"""CPU checks for checkpoint commit/prune ordering and real optimizer resume."""

import json
from types import SimpleNamespace

from gr00t.experiment.checkpoint_policy import (
    LATEST,
    MARKER,
    STOP_FILE,
    ResumableCheckpointCallback,
    commit_checkpoint,
    latest_resumable,
)
import pytest
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerControl


def _fake_checkpoint(root, step, world_size=1):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for name in ("model.safetensors", "training_args.bin", "scheduler.pt"):
        (path / name).write_bytes(b"test-content")
    if world_size == 1:
        (path / "optimizer.pt").write_bytes(b"optimizer")
        (path / "rng_state.pth").write_bytes(b"rng")
    else:
        tag = f"global_step{step}"
        (path / "latest").write_text(tag)
        (path / tag).mkdir()
        (path / tag / "mp_rank_00_model_states.pt").write_bytes(b"model state")
        for rank in range(world_size):
            (path / f"rng_state_{rank}.pth").write_bytes(b"rng")
            (path / tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
                b"optimizer"
            )
    return path


@pytest.mark.parametrize("world_size", [1, 2])
def test_new_full_checkpoint_preserves_old_model_only(tmp_path, world_size):
    old = _fake_checkpoint(tmp_path, 5000, world_size)
    first = commit_checkpoint(old, world_size)
    new = _fake_checkpoint(tmp_path, 10000, world_size)
    commit_checkpoint(new, world_size)
    assert latest_resumable(tmp_path)[0] == new
    assert (old / "model.safetensors").is_file()
    assert (old / "trainer_state.json").is_file()
    assert (old / "model_only.json").is_file()
    assert not (old / MARKER).exists()
    assert all(not (old / name).exists() for name in first["training_state_files"])
    assert (new / "scheduler.pt").is_file()


def test_incomplete_new_save_does_not_damage_last_resume(tmp_path):
    old = _fake_checkpoint(tmp_path, 5000, 2)
    commit_checkpoint(old, 2)
    broken = _fake_checkpoint(tmp_path, 10000, 2)
    (broken / "global_step10000/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete DeepSpeed"):
        commit_checkpoint(broken, 2)
    assert latest_resumable(tmp_path)[0] == old
    assert (old / "scheduler.pt").exists()


def test_corruption_and_symlinks_are_rejected(tmp_path):
    path = _fake_checkpoint(tmp_path, 5000)
    commit_checkpoint(path, 1)
    (path / "optimizer.pt").write_bytes(b"cut")
    with pytest.raises(ValueError, match="size mismatch"):
        latest_resumable(tmp_path)
    (tmp_path / LATEST).write_text('{"checkpoint": "../outside"}')
    with pytest.raises(ValueError, match="Unsafe"):
        latest_resumable(tmp_path)


def test_pruning_cannot_delete_model_weights(tmp_path):
    old = _fake_checkpoint(tmp_path, 5000)
    manifest = commit_checkpoint(old, 1)
    manifest["training_state_files"].append("model.safetensors")
    (old / MARKER).write_text(json.dumps(manifest))
    new = _fake_checkpoint(tmp_path, 10000)
    with pytest.raises(ValueError, match="non-training-state"):
        commit_checkpoint(new, 1)
    assert (old / "model.safetensors").exists()


def test_stop_request_forces_a_full_save(tmp_path):
    callback = ResumableCheckpointCallback(max_run_seconds=100)
    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=5000)
    state = SimpleNamespace(is_world_process_zero=True, global_step=125, save_steps=2000)
    callback.on_train_begin(args, state, TrainerControl())
    assert state.save_steps == 5000
    (tmp_path / STOP_FILE).touch()
    control = callback.on_step_end(args, state, TrainerControl())
    assert control.should_save and control.should_training_stop


def test_early_and_elapsed_recovery_saves(tmp_path, monkeypatch):
    import gr00t.experiment.checkpoint_policy as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    callback = ResumableCheckpointCallback(first_save_step=100, save_interval_seconds=1800)
    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=5000)
    state = SimpleNamespace(is_world_process_zero=True, global_step=100, max_steps=30000)
    callback.on_train_begin(args, state, TrainerControl())
    control = callback.on_step_end(args, state, TrainerControl())
    assert control.should_save and not control.should_training_stop
    state.global_step = 101
    assert not callback.on_step_end(args, state, TrainerControl()).should_save
    clock[0] = 1801
    assert callback.on_step_end(args, state, TrainerControl()).should_save


def test_explicit_milestones_and_final_step(tmp_path):
    steps = [400, 500, 1000, 2000, 3000, 5000]
    callback = ResumableCheckpointCallback(save_at_steps=steps)
    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=5000)
    state = SimpleNamespace(is_world_process_zero=True, global_step=0, max_steps=5000)
    callback.on_train_begin(args, state, TrainerControl())
    for step in (1, 399, 400, 401, 500, 999, 1000, 2000, 3000, 4000, 5000):
        state.global_step = step
        control = callback.on_step_end(args, state, TrainerControl())
        assert control.should_save == (step in steps)
        assert not control.should_training_stop
    with pytest.raises(ValueError, match="positive"):
        ResumableCheckpointCallback(save_at_steps=[0])


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x, labels):
        return {"loss": (self.fc(x) - labels).square().mean()}


class _PauseAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = control.should_training_stop = True
        return control


def test_real_adam_scheduler_rng_resume(tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=4,
        save_steps=2,
        save_total_limit=0,
        per_device_train_batch_size=2,
        report_to="none",
        learning_rate=0.001,
        optim="adamw_torch",
        disable_tqdm=True,
    )
    data = [{"x": torch.ones(2), "labels": torch.zeros(2)} for _ in range(8)]
    trainer = Trainer(
        model=_TinyModel(),
        args=args,
        train_dataset=data,
        callbacks=[_PauseAtTwo(), ResumableCheckpointCallback()],
    )
    trainer.train()
    first, _ = latest_resumable(tmp_path)
    assert first.name == "checkpoint-2"
    first_optimizer = torch.load(first / "optimizer.pt", weights_only=True)
    first_scheduler = torch.load(first / "scheduler.pt", weights_only=True)
    assert first_scheduler["last_epoch"] == 2
    assert all(float(v["step"]) == 2 for v in first_optimizer["state"].values())
    resumed = Trainer(
        model=_TinyModel(), args=args, train_dataset=data, callbacks=[ResumableCheckpointCallback()]
    )
    resumed.train(resume_from_checkpoint=str(first))
    last, _ = latest_resumable(tmp_path)
    assert resumed.state.global_step == 4 and last.name == "checkpoint-4"
    optimizer = torch.load(last / "optimizer.pt", weights_only=True)
    assert all(float(v["step"]) == 4 for v in optimizer["state"].values())
    assert torch.load(last / "scheduler.pt", weights_only=True)["last_epoch"] == 4
    assert not (first / "optimizer.pt").exists()
