import json

from gr00t.rl.trainer import OfflineTrainer
import pytest
import torch

from .test_algorithms import agent, batch


def test_latest_full_state_retains_model_archives(tmp_path):
    learner = OfflineTrainer(agent())
    learner.update(batch())
    (tmp_path / "step-0.pt").write_bytes(b"unowned")
    learner.save_recovery_checkpoint(tmp_path, keep_latest=True, archive_model=True)
    model = torch.load(tmp_path / "model-step-1.pt", weights_only=False)
    assert "optimizer" not in model["algorithm"]
    learner.update(batch())
    learner.save_recovery_checkpoint(tmp_path, keep_latest=True)
    assert not (tmp_path / "step-1.pt").exists()
    assert (tmp_path / "step-0.pt").read_bytes() == b"unowned"
    assert (tmp_path / "model-step-1.pt").exists()
    restored = OfflineTrainer(agent())
    restored.load_checkpoint(tmp_path / "step-2.pt")
    assert restored.step == 2
    assert json.loads((tmp_path / "latest_resumable.json").read_text())["file"] == "step-2.pt"


def test_failed_new_save_preserves_last_full_state(tmp_path, monkeypatch):
    learner = OfflineTrainer(agent())
    learner.update(batch())
    learner.save_recovery_checkpoint(tmp_path, keep_latest=True)
    learner.update(batch())

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted save")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        learner.save_recovery_checkpoint(tmp_path, keep_latest=True)
    assert (tmp_path / "step-1.pt").exists()
    assert not (tmp_path / "step-2.pt").exists()
    assert json.loads((tmp_path / "latest_resumable.json").read_text())["step"] == 1


def test_time_budget_saves_before_stop(tmp_path, monkeypatch):
    import gr00t.rl.trainer as module

    times = iter([0.0, 9.0, 12.0, 12.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    learner = OfflineTrainer(agent())
    learner.fit(
        [batch(), batch(), batch()],
        checkpoint_dir=tmp_path,
        max_run_seconds=10,
        keep_latest_training_state=True,
    )
    assert learner.step == 2
    assert (tmp_path / "step-2.pt").exists()


def test_stop_request_saves_after_complete_update(tmp_path):
    learner = OfflineTrainer(agent())
    learner.stop_requested = True
    learner.fit([batch(), batch()], checkpoint_dir=tmp_path, keep_latest_training_state=True)
    assert learner.step == 1
    assert (tmp_path / "step-1.pt").exists()
