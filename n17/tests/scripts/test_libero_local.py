"""LIBERO setup/launcher contracts without installing or launching a simulator."""

import importlib.util
import json
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "libero_local", ROOT / "examples/LIBERO/local/manage.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


def parse(*args):
    return recipe.parser().parse_args(args)


def test_task_suites():
    tasks = recipe.task_map()
    assert all(len(tasks[suite]) == 10 for suite in recipe.SUITES)
    assert len(tasks["libero_90"]) == 90
    assert len({name for names in tasks.values() for name in names}) == 130


@pytest.mark.parametrize("suite", recipe.SUITES)
def test_suite_specific_model_server(suite, tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    args = parse("server", "--suite", suite)
    assert not args.execute
    cmd = recipe.build_command(args)
    assert cmd[0] == str(tmp_path / "envs/gr00t-n1.7/bin/python")
    assert cmd[cmd.index("--model-path") + 1] == str(tmp_path / "models/GR00T-N1.7-LIBERO" / suite)
    assert cmd[cmd.index("--embodiment-tag") + 1] == "LIBERO_PANDA"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == "5556"


@pytest.mark.parametrize("suite", recipe.SUITES)
def test_eval_all_tasks(suite, tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    args = parse("eval", "--suite", suite, "--all-tasks")
    assert not args.execute
    assert len(recipe.selected_tasks(args)) == 10
    for task in recipe.selected_tasks(args):
        cmd = recipe.build_command(args, task, tmp_path / task)
        assert cmd[0] == str(tmp_path / "envs/libero-n17-client/bin/python")
        assert cmd[cmd.index("--env-name") + 1] == f"libero_sim/{task}"
        assert cmd[cmd.index("--n-action-steps") + 1] == "8"
        assert cmd[cmd.index("--max-episode-steps") + 1] == "720"
        assert cmd[cmd.index("--video-dir") + 1] == str(tmp_path / task / "videos")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--episodes", "0"),
        ("--n-envs", "0"),
        ("--n-envs", "11"),
        ("--port", "0"),
        ("--port", "65536"),
        ("--action-steps", "0"),
        ("--max-episode-steps", "0"),
    ],
)
def test_invalid_eval_rejected(flag, value):
    with pytest.raises(ValueError):
        recipe.build_command(parse("eval", flag, value))


def test_wrong_suite_task_rejected():
    task = recipe.task_map()["libero_goal"][0]
    with pytest.raises(ValueError, match="does not belong"):
        recipe.selected_tasks(parse("eval", "--suite", "libero_spatial", "--task", task))


def test_config_isolated_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    recipe.configure()
    path = tmp_path / "config/libero-n17/config.yaml"
    before = path.stat().st_mtime_ns
    recipe.configure()
    assert path.stat().st_mtime_ns == before
    assert json.loads(path.read_text()) == recipe.config_dict()
    assert json.loads(path.read_text())["datasets"] == str(tmp_path / "datasets/libero")


def test_config_preserves_user_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    recipe.configure()
    path = tmp_path / "config/libero-n17/config.yaml"
    path.write_text("user: setting\n")
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        recipe.configure()
    assert path.read_text() == "user: setting\n"


def test_download_is_pinned_and_weights_only(tmp_path, monkeypatch):
    import sys

    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=lambda *a, **kw: calls.append((a, kw))),
    )
    recipe.download("libero_goal")
    _, kw = calls[0]
    assert kw["revision"] == recipe.MODEL_REVISION
    assert kw["local_dir"] == tmp_path / "models/GR00T-N1.7-LIBERO"
    assert all(p.startswith("libero_goal/") for p in kw["allow_patterns"])
    assert "libero_goal/model-*.safetensors" in kw["allow_patterns"]
    assert not any("optim" in p or "global_step" in p for p in kw["allow_patterns"])


def test_gpu_denied(monkeypatch):
    monkeypatch.setattr(
        recipe.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(
            returncode=4, stdout="Insufficient Permissions", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="allocated container"):
        recipe.require_gpu()


def test_sources_reject_wrong_revision(monkeypatch):
    monkeypatch.setattr(recipe.subprocess, "check_output", lambda *a, **kw: "wrong\n")
    with pytest.raises(RuntimeError, match="revision mismatch"):
        recipe.check_sources()
