"""Validate launcher flags without launching distributed training or W&B runs."""

from pathlib import Path
import runpy
import shlex
import subprocess

from gr00t.configs.finetune_config import FinetuneConfig
import pytest
import tyro


LAUNCHER = Path(__file__).resolve().parents[2] / "examples/carrot_in_pot/train.sh"


def _arguments():
    source = LAUNCHER.read_text()
    command = source.split("    gr00t/experiment/launch_finetune.py", 1)[1]
    command = command.split('"$@"', 1)[0].replace("\\\n", " ")
    return shlex.split(command)


def test_launcher_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, timeout=5)


def test_launcher_enables_wandb_and_keeps_vlm_frozen():
    config = tyro.cli(FinetuneConfig, args=_arguments())
    assert config.use_wandb
    assert config.wandb_project == "$WANDB_PROJECT"
    assert config.num_gpus == 2 and config.global_batch_size == 32
    assert config.max_steps == 10000 and config.save_steps == 2000
    assert config.keep_latest_training_state and config.save_total_limit == 0
    assert config.max_run_seconds == 32400
    assert not config.tune_llm and not config.tune_visual
    assert config.tune_projector and config.tune_diffusion_model
    assert config.shortest_image_edge == 256 and config.crop_fraction == 1.0
    assert not config.save_only_model


def test_launcher_allows_small_smoke_test_overrides():
    config = tyro.cli(
        FinetuneConfig,
        args=_arguments() + ["--max-steps", "10", "--global-batch-size", "4"],
    )
    assert config.max_steps == 10 and config.global_batch_size == 4
    assert config.use_wandb


def test_launcher_keeps_wandb_files_on_raid_and_disables_model_upload():
    source = LAUNCHER.read_text()
    assert 'export WANDB_PROJECT="${WANDB_PROJECT:-carrot-in-pot-gr00t}"' in source
    assert 'export WANDB_DIR="$carrot_output"' in source
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        assert f'export {key}="$VLA_STORAGE_ROOT/' in source
    assert "export WANDB_LOG_MODEL=false" in source
    assert "export WANDB_WATCH=false" in source
    assert 'tee "$carrot_output/train.log"' in source
    assert "set -eo pipefail" in source


def test_gpu_guard_only_accepts_the_expected_keepalive(tmp_path):
    guard = runpy.run_path(str(LAUNCHER.with_name("gpu_guard.py")))["allowed_keepalives"]
    assert guard("") == set()
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0/safe/gpu_keepalive.py\0--device\x000\0")
    assert guard("123", proc_root=tmp_path, expected_script="/safe/gpu_keepalive.py") == {123}
    (process / "cmdline").write_bytes(b"python\0/train.py\0")
    with pytest.raises(ValueError, match="not the expected"):
        guard("123", proc_root=tmp_path, expected_script="/safe/gpu_keepalive.py")
    for invalid in ("0", "1", "-1", "123;kill"):
        with pytest.raises(ValueError, match="process IDs"):
            guard(invalid, proc_root=tmp_path)


def test_resume_restores_original_recipe_and_modality_types(tmp_path):
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.data.data_config import SingleDatasetConfig
    from gr00t.data.types import ActionRepresentation

    modality = runpy.run_path(str(LAUNCHER.with_name("carrot_config.py")))["carrot_config"]
    config = get_default_config()
    config.data.modality_configs = {"new_embodiment": modality}
    config.data.datasets = [SingleDatasetConfig(["/test/carrot"], "new_embodiment", 1.0)]
    config.training.learning_rate = 0.000023
    config.training.global_batch_size = 24
    config.training.max_steps = 12345
    checkpoint = tmp_path / "checkpoint-5000"
    config.save(checkpoint / "experiment_cfg/config.yaml")
    restore = runpy.run_path(str(LAUNCHER.with_name("resume_training.py")))["restored_config"]
    resumed = restore(tmp_path, checkpoint)
    resumed.validate()
    assert resumed.training.learning_rate == 0.000023
    assert resumed.training.global_batch_size == 24
    assert resumed.training.max_steps == 12345
    assert resumed.training.save_steps == 2000
    assert resumed.training.resume_checkpoint_path == str(checkpoint)
    action = resumed.data.modality_configs["new_embodiment"]["action"]
    assert action.delta_indices == list(range(16))
    assert action.action_configs[0].rep is ActionRepresentation.RELATIVE
