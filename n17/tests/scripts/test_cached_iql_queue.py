import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


QUEUE = Path(__file__).resolve().parents[2] / "examples/bigenlight_multitask/after_feature_cache.py"


def load_queue(monkeypatch):
    monkeypatch.syspath_prepend(str(QUEUE.parent))
    return runpy.run_path(str(QUEUE))


def test_cache_gate_rejects_partial_missing_and_modified_cache(tmp_path, monkeypatch):
    queue = load_queue(monkeypatch)
    verify = queue["verify_cache"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"file": "episode-000000.npz"}]}))
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    complete = {
        "format": queue["FORMAT"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "episodes": 1,
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="Missing"):
        verify(tmp_path)
    (tmp_path / "episode-000000.npz").touch()
    verify(tmp_path)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="inconsistent"):
        verify(tmp_path)


@pytest.mark.parametrize("interrupted", [False, True])
def test_queue_clears_resume_and_requires_all_completion(tmp_path, monkeypatch, interrupted):
    queue = load_queue(monkeypatch)
    main = queue["main"]
    globals_ = main.__globals__
    calls = []

    def launch(command, **kwargs):
        calls.append((command[2], kwargs["env"].copy()))
        return SimpleNamespace(returncode=0)

    def completed(output, steps):
        assert steps == 10000
        if interrupted:
            raise ValueError("incomplete run")

    monkeypatch.setitem(globals_, "verify_cache", lambda root: None)
    monkeypatch.setitem(globals_, "verify_completion", completed)
    monkeypatch.setattr(queue["subprocess"], "run", launch)
    monkeypatch.setenv("IQL_RESUME", "old-live-critic.pt")
    (tmp_path / "features").mkdir()
    (tmp_path / "features/n17-extraction-gpu-0.lock").touch()
    arguments = [
        "--storage-root",
        str(tmp_path),
        "--state-path",
        str(tmp_path / "state.json"),
        "--all-output",
        str(tmp_path / "all"),
        "--subset-output",
        str(tmp_path / "subset"),
        "--reward",
        "step-cost",
    ]
    if interrupted:
        with pytest.raises(ValueError, match="incomplete run"):
            main(arguments)
    else:
        main(arguments)
    assert [variant for variant, _ in calls] == (["all"] if interrupted else ["all", "50per-task"])
    for _, env in calls:
        assert "IQL_RESUME" not in env
        assert env["IQL_REWARD"] == "step-cost"
        assert env["WANDB_MODE"] == "online"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == ("blocked" if interrupted else "completed")
