"""No downloads, GPU training, W&B writes, or simulator required."""

import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from gr00t.configs.finetune_config import FinetuneConfig
import pytest
import tyro


# tests/examples is a regular package and shadows the repository's namespace
# package under pytest. Load this recipe under an isolated package name.
_root = Path(__file__).resolve().parents[2] / "examples/LIBERO/finetune"
_spec = importlib.util.spec_from_file_location("libero_finetune_recipe", _root / "__init__.py")
_package = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _package
_spec.loader.exec_module(_package)
data = importlib.import_module("libero_finetune_recipe.data")
launch = importlib.import_module("libero_finetune_recipe.launch")
RECIPE, dataset_path = data.RECIPE, data.dataset_path
safe_link, select_episodes, variant = data.safe_link, data.select_episodes, data.variant
command, parse_args = launch.command, launch.parse_args


def test_two_independent_bc_commands(tmp_path):
    for profile, gpu, count, precision in (
        ("qvgm-long", 0, 1, "fp32"),
        ("qvgm-unified", 1, 4, "bf16-mixed"),
    ):
        args = parse_args(
            [
                "--profile",
                profile,
                "--num-gpus",
                "1",
                "--gpu",
                str(gpu),
                "--output",
                str(tmp_path / profile),
                "--steps",
                "5000",
                "--save-steps",
                "5000",
                "--save-at-steps",
                "400",
                "500",
                "1000",
                "2000",
                "3000",
                "5000",
            ]
        )
        cmd = command(args)
        offset = next(i for i, value in enumerate(cmd) if value.endswith("launch_finetune.py"))
        config = tyro.cli(FinetuneConfig, args=cmd[offset + 1 :])
        assert config.num_gpus == 1 and config.global_batch_size == 32
        assert config.precision == precision and config.max_steps == 5000
        assert len(config.dataset_path.split(os.pathsep)) == count
        assert config.save_at_steps == [400, 500, 1000, 2000, 3000, 5000]
        assert not config.tune_visual and not config.tune_llm
        assert args.gpu == gpu


def test_gpu_selection_requires_explicit_isolation(tmp_path):
    base = ["--output", str(tmp_path / "output")]
    for extra in (["--num-gpus", "1"], ["--gpu", "0"], ["--save-at-steps", "0"]):
        with pytest.raises(SystemExit):
            parse_args(base + extra)
    args = parse_args(base + ["--num-gpus", "1", "--gpu", "1"])
    assert launch.selected_gpu_uuids(args, "0, GPU-aaa\n1, GPU-bbb\n") == ["GPU-bbb"]
    with pytest.raises(ValueError, match="not visible"):
        launch.selected_gpu_uuids(args, "0, GPU-aaa\n")


def test_gpu_guard_ignores_only_other_gpu_and_verified_keepalive():
    guard = runpy.run_path(str(_root.parents[1] / "carrot_in_pot/gpu_guard.py"))
    rows = "118, GPU-aaa\n297, GPU-aaa\n125, GPU-bbb\n"
    assert guard["busy_pids"](rows, {118, 125}, "GPU-aaa") == {297}
    assert guard["busy_pids"](rows, {118, 125}, "GPU-bbb") == set()
    assert guard["busy_pids"](rows, {118, 125}) == {297}


@pytest.mark.parametrize("precision", ["bf16-mixed", "fp32"])
def test_precision_and_milestones_reach_training_config(tmp_path, monkeypatch, precision):
    import gr00t.experiment.experiment as experiment

    captured = []
    config = FinetuneConfig(
        base_model_path=str(tmp_path / "base"),
        dataset_path=str(tmp_path / "data"),
        embodiment_tag="LIBERO_PANDA",
        precision=precision,
        save_at_steps=[400, 500, 1000, 2000, 3000, 5000],
    )
    monkeypatch.setattr(tyro, "cli", lambda *args, **kwargs: config)
    monkeypatch.setattr(experiment, "run", captured.append)
    runpy.run_path(
        str(_root.parents[2] / "gr00t/experiment/launch_finetune.py"), run_name="__main__"
    )
    actual = captured[0]
    assert actual.training.save_at_steps == config.save_at_steps
    assert actual.training.bf16 == (precision == "bf16-mixed")
    assert actual.model.use_flash_attention == (precision == "bf16-mixed")
    if precision == "fp32":
        assert not actual.training.fp16 and not actual.training.tf32
        assert actual.model.model_dtype == "float32"


def test_fp32_sdpa_is_scoped_to_nested_qwen(tmp_path, monkeypatch):
    from gr00t.configs.base_config import get_default_config
    from gr00t.model.gr00t_n1d7 import setup
    import torch

    config = get_default_config()
    config.model.model_dtype = "float32"
    config.model.use_flash_attention = False
    config.training.start_from_checkpoint = str(tmp_path / "fake-checkpoint")
    pipeline = setup.Gr00tN1d7Pipeline(config, tmp_path)
    captured = {}

    class Captured(Exception):
        pass

    def intercept(*args, **kwargs):
        captured.update(kwargs)
        raise Captured()

    monkeypatch.setattr(setup.AutoModel, "from_pretrained", intercept)
    with pytest.raises(Captured):
        pipeline._create_model()
    assert captured["torch_dtype"] == torch.float32
    assert not captured["use_flash_attention"]
    assert "attn_implementation" not in captured
    assert captured["transformers_loading_kwargs"]["attn_implementation"] == "sdpa"
    assert "torch_dtype" not in pipeline.transformers_loading_kwargs


def test_selection_is_balanced_reproducible_and_does_not_change_source():
    tasks = [{"task_index": i, "task": f"task-{i}"} for i in range(10)]
    episodes = [
        {"episode_index": i * 10 + j, "tasks": [f"task-{i}"], "length": 20}
        for i in range(10)
        for j in range(8)
    ]
    original = json.dumps(episodes)
    for count in (1, 5):
        selected = select_episodes(episodes, tasks, count, 42)
        assert len(selected) == 10 * count
        assert selected == select_episodes(episodes, tasks, count, 42)
        assert selected != select_episodes(episodes, tasks, count, 43)
        assert all(sum(ep["tasks"] == [task["task"]] for ep in selected) == count for task in tasks)
    assert select_episodes(episodes, tasks) == episodes
    assert json.dumps(episodes) == original
    with pytest.raises(ValueError, match="Not enough"):
        select_episodes(episodes, tasks, 9)
    with pytest.raises(ValueError, match="Missing task"):
        select_episodes(episodes[:-8], tasks)


def test_pins_and_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("VLA_STORAGE_ROOT", str(tmp_path))
    assert sum(spec["episodes"] for spec in RECIPE["datasets"].values()) == 1693
    assert all(len(spec["revision"]) == 40 for spec in RECIPE["datasets"].values())
    assert (
        dataset_path("libero_10", 5, 42)
        == tmp_path / "datasets/libero_n17_bc/5shot-seed42/libero_10"
    )
    with pytest.raises(ValueError):
        variant(-1, 42)


def test_safe_link_preserves_original_and_refuses_unrelated_files(tmp_path):
    source, target = tmp_path / "source", tmp_path / "nested/target"
    source.write_bytes(b"video")
    safe_link(source, target)
    safe_link(source, target)
    assert target.is_symlink() and source.read_bytes() == b"video"
    other = tmp_path / "other"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="Refusing"):
        safe_link(other, target)


@pytest.mark.parametrize("suite", tuple(RECIPE["datasets"]))
def test_bc_command_parses_and_freezes_vlm(suite, tmp_path):
    args = parse_args(
        ["--suite", suite, "--output", str(tmp_path / "output"), "--batch-per-gpu", "128"]
    )
    cmd = command(args)
    offset = next(i for i, value in enumerate(cmd) if value.endswith("launch_finetune.py"))
    config = tyro.cli(FinetuneConfig, args=cmd[offset + 1 :])
    assert config.global_batch_size == 256
    assert not config.tune_llm and not config.tune_visual
    assert config.tune_projector and config.tune_diffusion_model
    assert config.save_steps == 2000 and config.keep_latest_training_state
    assert config.max_run_seconds == 28800
    assert config.embodiment_tag == "LIBERO_PANDA"
    assert not args.execute


def test_dry_run_is_non_mutating(tmp_path, capsys):
    output = tmp_path / "not_created"
    launch.main(["--output", str(output)])
    assert "DRY RUN" in capsys.readouterr().out
    assert not output.exists()
    script = Path(__file__).resolve().parents[2] / "examples/LIBERO/finetune/run.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_prepare_fewshot_isolated_statistics_and_original_ids(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, target = tmp_path / "source", tmp_path / "prepared"
    (source / "meta").mkdir(parents=True)
    tasks = [{"task_index": i, "task": f"task-{i}"} for i in range(10)]
    episodes = [{"episode_index": i, "tasks": [f"task-{i // 3}"], "length": 3} for i in range(30)]
    info = {
        "chunks_size": 1000,
        "fps": 20,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [8]},
            "action": {"dtype": "float32", "shape": [7]},
            "observation.images.image": {"dtype": "video", "shape": [256, 256, 3]},
            "observation.images.wrist_image": {"dtype": "video", "shape": [256, 256, 3]},
        },
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    for name, rows in (("episodes", episodes), ("tasks", tasks)):
        (source / f"meta/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for ep in episodes:
        index = ep["episode_index"]
        fields = dict(episode_index=index, episode_chunk=0)
        file = source / info["data_path"].format(**fields)
        file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "observation.state": np.full((3, 8), index, dtype=np.float32).tolist(),
                    "action": np.full((3, 7), index, dtype=np.float32).tolist(),
                    "episode_index": [index] * 3,
                    "task_index": [index // 3] * 3,
                }
            ),
            file,
        )
        for camera in ("observation.images.image", "observation.images.wrist_image"):
            video = source / info["video_path"].format(**fields, video_key=camera)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video placeholder: no decoding in this unit test")
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    spec = {"episodes": 30, "frames": 90}
    kwargs = dict(suite="libero_spatial", spec=spec, demos_per_task=1, seed=42)
    result = data.prepare(source, target, **kwargs)
    assert not result["rl_ready"]
    ids = result["provenance"]["selected_episode_ids"]
    assert len(ids) == 10 and len(list(target.glob("data/*/*.parquet"))) == 10
    stats = json.loads((target / "meta/stats.json").read_text())
    assert np.allclose(stats["action"]["mean"], np.mean(ids))
    assert json.loads((target / "meta/info.json").read_text())["total_frames"] == 30
    assert [ep["episode_index"] for ep in data.read_jsonl(target / "meta/episodes.jsonl")] == ids
    assert data.prepare(source, target, **kwargs) == result
    assert before == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="different/unowned"):
        data.prepare(source, target, **dict(kwargs, seed=43))
