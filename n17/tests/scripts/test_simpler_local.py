"""CPU-only launcher checks; no GPU, downloads or simulator needed."""

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "simpler_local", ROOT / "examples/SimplerEnv/local/manage.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def args(**updates):
    values = dict(
        command="eval",
        robot="widowx",
        port=5555,
        task=None,
        episodes=10,
        n_envs=1,
        max_episode_steps=300,
        action_steps=None,
        seed=42,
        output_dir="/tmp/simpler-test-only",
        model_path=None,
    )
    values.update(updates)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("robot,config", list(recipe.ROBOTS.items()))
def test_task_commands(robot, config, monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    for task in config["tasks"]:
        command = recipe.build_command(args(robot=robot, task=task))
        assert command[0] == str(tmp_path / "envs/simpler-n17-client/bin/python")
        assert command[command.index("--env-name") + 1] == f"simpler_env_{robot}/{task}"
        assert command[command.index("--n-action-steps") + 1] == str(config["action_steps"])
        assert command[command.index("--seed") + 1] == "42"


@pytest.mark.parametrize("robot", recipe.ROBOTS)
def test_server_is_separate_and_local(robot, monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    command = recipe.build_command(args(command="server", robot=robot))
    assert command[0] == str(tmp_path / "envs/gr00t-n1.7/bin/python")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--embodiment-tag") + 1] == recipe.ROBOTS[robot]["embodiment"]
    assert command[command.index("--model-path") + 1].startswith(str(tmp_path / "models"))


@pytest.mark.parametrize(
    "updates",
    [
        dict(task="google_robot_pick_coke_can"),
        dict(episodes=0),
        dict(n_envs=11),
        dict(n_envs=0),
        dict(port=0),
        dict(port=65536),
        dict(action_steps=0),
        dict(max_episode_steps=0),
    ],
)
def test_invalid_config_rejected(updates):
    with pytest.raises(ValueError):
        recipe.build_command(args(**updates))


def test_custom_model():
    command = recipe.build_command(args(command="server", model_path="/raid/my-bc-checkpoint"))
    assert command[command.index("--model-path") + 1] == "/raid/my-bc-checkpoint"


def test_source_pin_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(recipe, "PINS", {tmp_path: "expected"})
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *a, **kw: "different\n")
    with pytest.raises(RuntimeError, match="revision mismatch"):
        recipe.check_sources()


def test_gpu_denied(monkeypatch):
    monkeypatch.setattr(
        recipe.subprocess,
        "run",
        lambda *a, **kw: argparse.Namespace(
            returncode=4, stdout="", stderr="Insufficient Permissions"
        ),
    )
    with pytest.raises(RuntimeError, match="allocated GPU container"):
        recipe.require_gpu()
