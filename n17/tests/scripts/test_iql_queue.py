import json
from pathlib import Path
import runpy

import pytest
import torch


QUEUE = Path(__file__).resolve().parents[2] / "examples/bigenlight_multitask/after_iql.py"


def test_completion_requires_exact_checkpoint_and_matching_metadata(tmp_path):
    verify = runpy.run_path(str(QUEUE))["verify_completion"]
    with pytest.raises(FileNotFoundError):
        verify(tmp_path, 10000)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "run.json").write_text(json.dumps({"metadata": {"source": "test"}}))
    checkpoint = tmp_path / "checkpoints/step-10000.pt"
    state = {
        "step": 10000,
        "algorithm": {"updates": 10000, "algorithm": "iql-critic-only-v1"},
        "metadata": {"source": "test"},
    }
    torch.save(state, checkpoint)
    assert verify(tmp_path, 10000) == checkpoint
    state["algorithm"]["updates"] = 8000
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="update count"):
        verify(tmp_path, 10000)
    state["algorithm"]["updates"] = 10000
    state["metadata"] = {}
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="metadata"):
        verify(tmp_path, 10000)
