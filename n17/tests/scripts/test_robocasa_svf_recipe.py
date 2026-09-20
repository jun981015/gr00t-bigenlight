import argparse
import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[2] / "examples/robocasa_svf/launch.py"
_spec = importlib.util.spec_from_file_location("robocasa_svf_recipe", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
command, dataset_paths = _module.command, _module.dataset_paths


def args(tmp_path, stage):
    return argparse.Namespace(
        stage=stage,
        output=tmp_path / stage,
        base_model=tmp_path / "bc-checkpoint",
        steps=None,
        batch_size=None,
        num_gpus=None,
        resume=None,
    )


def test_four_tasks_and_correct_source_groups():
    assert len(dataset_paths("bc24")) == 1
    assert len(dataset_paths("svf")) == len(dataset_paths("filtered-bc")) == 8
    assert sum("rollouts" in path.parts for path in dataset_paths("svf")) == 4
    assert sum("success_rollouts" in path.parts for path in dataset_paths("filtered-bc")) == 4


def test_bc_frozen_vlm_and_two_gpus(tmp_path):
    cmd = command(args(tmp_path, "bc24"))
    assert "torch.distributed.run" in cmd and "--nproc-per-node=2" in cmd
    assert cmd[cmd.index("--global-batch-size") + 1] == "32"
    assert "--no-tune-llm" in cmd and "--no-tune-visual" in cmd
    assert "--first-save-step" in cmd and "--save-interval-seconds" in cmd


def test_svf_explicit_single_gpu_and_labels(tmp_path):
    config = args(tmp_path, "svf")
    cmd = command(config)
    assert "torch.distributed.run" not in cmd
    assert cmd[cmd.index("--backend") + 1] == "gr00t"
    assert cmd[cmd.index("--annotation-format") + 1] == "deas-robocasa"
    assert "--no-bootstrap-on-truncation" in cmd
    assert cmd[cmd.index("--batch-size") + 1] == "1"
    config.num_gpus = 2
    with pytest.raises(ValueError, match="single-device"):
        command(config)


def test_svf_requires_bc_checkpoint(tmp_path):
    config = args(tmp_path, "svf")
    config.base_model = None
    with pytest.raises(ValueError, match="preceding BC"):
        command(config)
