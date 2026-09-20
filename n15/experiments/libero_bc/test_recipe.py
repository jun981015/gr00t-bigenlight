"""CPU-only recipe tests: no models, downloads, W&B, or GPU processes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("libero_bc_data", HERE / "data.py")
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
spec = importlib.util.spec_from_file_location("libero_bc_manage", HERE / "manage.py")
manage = importlib.util.module_from_spec(spec)
old_data = sys.modules.get("data")
sys.modules["data"] = data
spec.loader.exec_module(manage)
if old_data is None:
    del sys.modules["data"]
else:
    sys.modules["data"] = old_data


@pytest.mark.parametrize("profile,count", [("long", 1), ("unified", 4)])
def test_batch_freeze_and_horizon(tmp_path, profile, count):
    args = manage.parse_args(["train", "--profile", profile, "--gpu", "0", "--output", str(tmp_path)])
    recipe = manage.training_recipe(args)
    cfg = recipe["config"]
    assert recipe["effective_batch"] == 256
    assert cfg["batch_size"] * recipe["gradient_accumulation_steps"] == 256
    assert recipe["action_horizon"] == 16
    assert recipe["precision"] == "bf16-mixed"
    assert cfg["num_gpus"] == 1 and len(cfg["dataset_path"]) == count
    assert cfg["max_steps"] == 5000 and cfg["save_steps"] == 2000
    assert not cfg["tune_llm"] and not cfg["tune_visual"]
    assert cfg["tune_projector"] and cfg["tune_diffusion_model"]
    assert recipe["save_at_steps"] == [500, 1000, 2000, 3000, 5000]
    assert not args.execute


def test_bad_batch_rejected(tmp_path):
    for extra in (["--batch-size", "0"], ["--batch-size", "255"], ["--steps", "-1"]):
        with pytest.raises(SystemExit):
            manage.parse_args(["train", "--profile", "long", "--gpu", "0", "--output", str(tmp_path)] + extra)


def test_schema_matches_original_libero_config():
    from gr00t.data.schema import LeRobotModalityMetadata
    from gr00t.experiment.data_config import LiberoDataConfig

    metadata = LeRobotModalityMetadata.model_validate(data.MODALITY)
    config = LiberoDataConfig()
    for modality in config.modality_config().values():
        for key in modality.modality_keys:
            assert metadata.get_key_meta(key)
    assert config.action_indices == list(range(16))
    assert metadata.state["eef_rot_absolute"].rotation_type.value == "axis_angle"
    assert not metadata.action["eef_pos_delta"].absolute
    assert metadata.state["gripper_close"].end == 8


def test_preparation_preserves_selection_and_sources(tmp_path):
    source = tmp_path / "1shot-seed42/libero_10"
    meta = source / "meta"
    meta.mkdir(parents=True)
    episodes = [{"episode_index": i * 3, "length": 2, "tasks": [f"task-{i}"]} for i in range(10)]
    provenance = {"selected_episode_ids": [ep["episode_index"] for ep in episodes]}
    (source / "READY.json").write_text(json.dumps({"provenance": provenance, "episodes": 10}))
    (source / "VALIDATION.json").write_text(json.dumps({"provenance": provenance, "validated_tasks": 10}))
    info = {
        "total_episodes": 10,
        "total_frames": 20,
        "chunks_size": 1000,
        "data_path": "data/{episode_index}.parquet",
        "video_path": "videos/{video_key}/{episode_index}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    (meta / "tasks.jsonl").write_text("tasks")
    (meta / "stats.json").write_text(json.dumps({"observation.state": {}, "action": {}, "hash": {}}))
    for ep in episodes:
        fields = {"episode_index": ep["episode_index"]}
        members = [info["data_path"].format(**fields)]
        members += [
            info["video_path"].format(**fields, video_key=v["original_key"])
            for v in data.MODALITY["video"].values()
        ]
        for name in members:
            member = source / name
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text("original")
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    target = tmp_path / "overlay"
    data.prepare(source, target)
    data.prepare(source, target)  # Idempotent.
    assert (target / "data/27.parquet").is_symlink()
    assert data.read_jsonl(target / "meta/episodes.jsonl") == episodes
    assert set(json.loads((target / "meta/stats.json").read_text())) == {"observation.state", "action"}
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    (target / "SOURCE.json").write_text("{}")
    with pytest.raises(ValueError, match="Refusing"):
        data.prepare(source, target)


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)


def test_recipe_accepted_by_original_bc_entrypoint(tmp_path):
    from scripts.gr00t_finetune import ArgsConfig

    args = manage.parse_args(["train", "--profile", "long", "--gpu", "1", "--output", str(tmp_path)])
    config = ArgsConfig(**manage.training_recipe(args)["config"])
    assert config.data_config == "libero" and config.video_backend == "torchvision_av"


def test_physical_batch_256_is_explicit_and_no_accumulation(tmp_path):
    args = manage.parse_args(
        ["train", "--profile", "long", "--gpu", "1", "--output", str(tmp_path), "--micro-batch-size", "256"]
    )
    recipe = manage.training_recipe(args)
    assert recipe["config"]["batch_size"] == 256
    assert recipe["gradient_accumulation_steps"] == 1


def test_followup_is_separate_unsaved_and_does_not_change_parent(tmp_path):
    from copy import deepcopy

    from experiments.libero_bc.temporary import temporary_recipe

    args = manage.parse_args(
        ["train", "--profile", "long", "--gpu", "0", "--output", str(tmp_path), "--after-steps", "100000"]
    )
    recipe = manage.training_recipe(args)
    before = deepcopy(recipe)
    child = temporary_recipe(recipe, tmp_path / "checkpoint-5000")
    assert recipe == before and recipe["config"]["max_steps"] == 5000
    assert child["config"]["max_steps"] == 100000
    assert child["config"]["output_dir"] == str(tmp_path / "temporary-nosave-100000")
    assert child["config"]["base_model_path"] == str(tmp_path / "checkpoint-5000")
    assert not child["checkpoints"] and not child["save_at_steps"]
    assert child["after_steps"] == 0 and not child["config"]["resume"]
    assert child["effective_batch"] == 32 and child["gradient_accumulation_steps"] == 1


def test_no_save_guard_blocks_all_checkpoint_entrypoints():
    from experiments.libero_bc.temporary import NoCheckpointMixin

    class Original:
        def _save_checkpoint(self, *args, **kwargs):
            raise AssertionError("Would save")

        save_model = save_state = _save_checkpoint

    class Unsaved(NoCheckpointMixin, Original):
        pass

    trainer = Unsaved()
    trainer._save_checkpoint("model", "trial")
    trainer.save_model("output")
    trainer.save_state()


def test_temporary_runner_bypasses_final_save(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.libero_bc import worker

    captured = {}

    class Trainer:
        def add_callback(self, callback):
            pass

        def train(self):
            captured["trained"] = True
            self.save_model()
            self.save_state()

        def save_model(self):
            raise AssertionError("Model save")

        def save_state(self):
            raise AssertionError("State save")

    class Runner:
        def __init__(self, **kwargs):
            self.trainer = self.create_trainer()
            captured["training"] = kwargs["training_args"]

        def create_trainer(self):
            return Trainer()

        def train(self):
            raise AssertionError("Original final-saving runner must be bypassed")

    def bc_main(config, runner_class):
        runner = runner_class(training_args=SimpleNamespace(), model=SimpleNamespace(action_horizon=16))
        runner.train()

    monkeypatch.setattr(worker.bc, "TrainRunner", Runner)
    monkeypatch.setattr(worker.bc, "main", bc_main)
    args = manage.parse_args(
        ["train", "--profile", "long", "--gpu", "0", "--output", str(tmp_path), "--no-checkpoints"]
    )
    recipe = manage.training_recipe(args)
    monkeypatch.setattr(sys, "argv", ["worker", "--recipe", json.dumps(recipe)])
    worker.main()
    assert captured["trained"]
    assert captured["training"].save_strategy == "no"
    assert captured["training"].report_to == []
    assert not (tmp_path / "wandb_resume.json").exists()


def test_no_save_and_resume_or_chaining_are_incompatible(tmp_path):
    for extra in (["--resume"], ["--after-steps", "100000"]):
        with pytest.raises(SystemExit):
            manage.parse_args(
                ["train", "--profile", "long", "--gpu", "0", "--output", str(tmp_path), "--no-checkpoints"] + extra
            )


def test_timestamp_selection_is_not_keyframe_selection():
    from gr00t.utils.pyav_frames import nearest_indices

    assert nearest_indices([0, 0.05, 0.10, 0.15], [0.10, 0.15, 0, 0.049, 0.6]).tolist() == [2, 3, 0, 1, 3]
    with pytest.raises(ValueError):
        nearest_indices([0, 0.1, 0.05], [0.05])
