"""CPU tests; no authentication, downloads, models, or GPU jobs."""

import argparse
import json
from types import SimpleNamespace

import pytest

from manage import RECIPE, datasets, training_config
from recovery import LATEST, MARKER, RecoveryCallback, commit, latest_checkpoint


def checkpoint(root, step, world=2):
    target = root / f"checkpoint-{step}"
    target.mkdir()
    (target / "experiment_cfg").mkdir()
    for name in ["config.json", "model.safetensors", "optimizer.pt", "scheduler.pt", "experiment_cfg/metadata.json"]:
        (target / name).write_text("stub")
    for name in (["rng_state.pth"] if world == 1 else [f"rng_state_{r}.pth" for r in range(world)]):
        (target / name).write_text("rng")
    (target / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return target


def test_task_splits():
    assert len(RECIPE["tasks"]) == 4
    assert len(datasets("bc24")) == 1
    assert len(datasets("filtered-bc")) == len(datasets("critic")) == 8
    assert sum("success_rollouts" in p.parts for p in datasets("filtered-bc")) == 4
    assert sum("rollouts" in p.parts for p in datasets("critic")) == 4
    assert not any("success_rollouts" in p.parts for p in datasets("critic"))


@pytest.mark.parametrize("stage", ["bc24", "filtered-bc", "critic"])
def test_stage_configuration(tmp_path, stage):
    args = argparse.Namespace(stage=stage, output=tmp_path / stage, base_model=tmp_path / "actor",
                              batch_size=None, num_gpus=None, max_steps=None, resume=False)
    cfg = training_config(args)
    assert cfg["batch_size"] * cfg["num_gpus"] == 32
    assert cfg["max_steps"] == 30000 and cfg["save_steps"] == 5000
    assert cfg["tune_llm"] is False and cfg["tune_visual"] is False
    if stage == "critic":
        assert cfg["critic_action_horizon"] == 16
        assert cfg["expectile"] == 0.7 and cfg["discount1"] == 0.9 and cfg["discount2"] == 0.99


def test_iql_recipe_and_training_arguments(tmp_path):
    from scripts.gr00t_deas_critic_finetune import ArgsConfig
    args = argparse.Namespace(stage="critic", output=tmp_path / "iql", base_model=tmp_path / "actor",
                              batch_size=16, num_gpus=1, max_steps=30000, resume=False,
                              critic_algorithm="iql", save_steps=5000, logging_steps=50,
                              video_backend="torchvision_av")
    cfg = training_config(args)
    assert cfg.pop("logging_steps") == 50
    parsed = ArgsConfig(**cfg)
    assert parsed.critic_algorithm == "iql" and parsed.num_atoms == 1
    assert parsed.iql_discount == 0.99 and not parsed.negative_reward
    assert len(parsed.dataset_path) == 8
    assert all("success_rollouts" not in p for p in parsed.dataset_path)
    args.stage = "filtered-bc"
    with pytest.raises(ValueError, match="only valid for the critic"):
        training_config(args)


def test_commit_then_prune_only_state(tmp_path):
    old = checkpoint(tmp_path, 100)
    commit(old, 2)
    new = checkpoint(tmp_path, 5000)
    commit(new, 2)
    assert latest_checkpoint(tmp_path, 2) == new
    assert not (old / "optimizer.pt").exists()
    assert not (old / MARKER).exists()
    assert (old / "model.safetensors").exists()
    assert (old / "experiment_cfg/metadata.json").exists()
    assert (new / "optimizer.pt").exists()


def test_unowned_checkpoint_untouched(tmp_path):
    unowned = checkpoint(tmp_path, 10)
    new = checkpoint(tmp_path, 100)
    commit(new, 2)
    assert (unowned / "optimizer.pt").exists()


@pytest.mark.parametrize("missing", ["optimizer.pt", "scheduler.pt", "rng_state_1.pth", "model.safetensors"])
def test_partial_new_checkpoint_keeps_previous(tmp_path, missing):
    old = checkpoint(tmp_path, 100)
    commit(old, 2)
    new = checkpoint(tmp_path, 200)
    (new / missing).unlink()
    with pytest.raises(ValueError):
        commit(new, 2)
    assert latest_checkpoint(tmp_path, 2) == old
    assert (old / "optimizer.pt").exists()


def test_resume_rejects_world_size_change(tmp_path):
    commit(checkpoint(tmp_path, 100), 2)
    with pytest.raises(ValueError):
        latest_checkpoint(tmp_path, 1)


def test_resume_rejects_corrupt_inventory(tmp_path):
    target = checkpoint(tmp_path, 100)
    commit(target, 2)
    (target / "optimizer.pt").write_text("changed size")
    with pytest.raises(ValueError):
        latest_checkpoint(tmp_path, 2)


def test_rejects_symlink_weights(tmp_path):
    target = checkpoint(tmp_path, 100)
    (target / "model.safetensors").unlink()
    external = tmp_path / "external-weights"
    external.write_text("test weights")
    (target / "model.safetensors").symlink_to(external)
    with pytest.raises(ValueError):
        commit(target, 2)
    assert external.exists()


def test_unsafe_pointer(tmp_path):
    (tmp_path / LATEST).write_text(json.dumps({"checkpoint": "../checkpoint-100"}))
    with pytest.raises(ValueError):
        latest_checkpoint(tmp_path, 2)


@pytest.mark.parametrize("step,expected", [(1, False), (100, True), (30000, True)])
def test_early_and_final_save(tmp_path, step, expected):
    callback = RecoveryCallback(RECIPE["recovery"])
    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=5000)
    state = SimpleNamespace(is_world_process_zero=True, global_step=step, max_steps=30000)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(args, state, control)
    callback.on_step_end(args, state, control)
    assert control.should_save is expected


def test_stop_saves_full_state(tmp_path):
    callback = RecoveryCallback(RECIPE["recovery"])
    args = SimpleNamespace(output_dir=str(tmp_path), save_steps=5000)
    state = SimpleNamespace(is_world_process_zero=True, global_step=4294, max_steps=30000)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_train_begin(args, state, control)
    (tmp_path / "STOP_AFTER_CHECKPOINT").touch()
    callback.on_step_end(args, state, control)
    assert control.should_save and control.should_training_stop


def test_resume_replaces_checkpoint_cadence():
    callback = RecoveryCallback(RECIPE["recovery"])
    args = SimpleNamespace(save_steps=10000, logging_steps=50)
    state = SimpleNamespace(save_steps=2000, logging_steps=10, global_step=88)
    callback.on_train_begin(args, state, SimpleNamespace())
    assert state.save_steps == 10000 and state.logging_steps == 50
    assert state.global_step == 88


def test_real_hf_save_and_resume_on_cpu(tmp_path):
    """Exercise optimizer/scheduler serialization through the installed HF Trainer, not mocks."""
    import torch
    from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments

    from gr00t.experiment.trainer import DualBrainTrainer
    from gr00t.utils.experiment import CheckpointFormatCallback

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, inputs):
            return {"loss": self.linear(inputs["x"]).square().mean()}

    class StopAtTwo(RecoveryCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control = super().on_step_end(args, state, control, **kwargs)
            if state.global_step == 2:
                control.should_save = control.should_training_stop = True
            return control

    meta = tmp_path / "experiment_cfg"
    meta.mkdir()
    (meta / "metadata.json").write_text("{}")
    train_args = TrainingArguments(output_dir=str(tmp_path), max_steps=3, per_device_train_batch_size=1,
                                   use_cpu=True, report_to=[], save_steps=5000, disable_tqdm=True,
                                   remove_unused_columns=False)
    data = [{"x": torch.ones(2)} for _ in range(5)]
    policy = dict(RECIPE["recovery"], first_save_step=1)
    trainer = DualBrainTrainer(model=TinyModel(PretrainedConfig()), args=train_args, train_dataset=data,
                               compute_dtype=torch.float32)
    trainer.add_callback(CheckpointFormatCallback("test", meta))
    trainer.add_callback(StopAtTwo(policy))
    trainer.train()
    assert trainer.state.global_step == 2
    saved = latest_checkpoint(tmp_path, 1)
    resumed = DualBrainTrainer(model=TinyModel(PretrainedConfig()), args=train_args, train_dataset=data,
                               compute_dtype=torch.float32)
    resumed.add_callback(CheckpointFormatCallback("test", meta))
    resumed.add_callback(RecoveryCallback(policy))
    resumed.train(resume_from_checkpoint=str(saved))
    assert resumed.state.global_step == 3
    assert resumed.lr_scheduler.last_epoch == 3
    assert all(int(state["step"]) == 3 for state in resumed.optimizer.state.values())
    assert latest_checkpoint(tmp_path, 1).name == "checkpoint-3"
