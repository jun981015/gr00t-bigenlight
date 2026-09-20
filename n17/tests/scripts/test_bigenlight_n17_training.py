"""CPU-only launch config tests; never invoke a model, CUDA, or W&B."""

import os
from pathlib import Path
import shlex
import subprocess

from gr00t.configs.finetune_config import FinetuneConfig
import pytest
import tyro


REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "examples/bigenlight_multitask"


@pytest.mark.parametrize(
    "variant,suffix,gpu", [("50per-task", "_50per_task", "0"), ("all", "", "1")]
)
def test_named_runs_dry_run(variant, suffix, gpu):
    env = dict(os.environ, WANDB_PROJECT="old-project", WANDB_RUN_ID="old-run")
    result = subprocess.run(
        ["bash", str(HERE / "run_n17.sh"), variant],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert "DRY RUN" in result.stdout
    assert f"GPU {gpu}" in result.stdout
    command = next(line for line in result.stdout.splitlines() if "--base-model-path" in line)
    tokens = shlex.split(command)
    args = tokens[tokens.index("gr00t/experiment/launch_finetune.py") + 1 :]
    cfg = tyro.cli(FinetuneConfig, args=args)
    assert cfg.base_model_path.endswith("models/GR00T-N1.7-3B")
    assert cfg.dataset_path.endswith(f"bigenlight_multitask_gr00t{suffix}/n17")
    assert f"bigenlight-n17-{variant}-b32-10k-" in cfg.output_dir
    assert not Path(cfg.output_dir).exists()  # dry-run creates no run directory
    assert cfg.num_gpus == 1 and cfg.global_batch_size == 32
    assert cfg.gradient_accumulation_steps == 1 and cfg.max_steps == 10000
    assert cfg.precision == "bf16-mixed" and cfg.logging_steps == 50
    assert cfg.use_wandb and cfg.wandb_project == "bigenlight-multitask-gr00t"
    assert not cfg.tune_llm and not cfg.tune_visual
    assert cfg.tune_projector and cfg.tune_diffusion_model
    assert cfg.save_steps == 5000 and cfg.first_save_step == 100
    assert cfg.save_interval_seconds == 1800 and cfg.max_run_seconds == 28800
    assert cfg.keep_latest_training_state and not cfg.save_only_model
    assert cfg.modality_config_path == "examples/bigenlight_multitask/config.py"


def test_shell_syntax_and_reject_unknown_variant():
    for script in ("train.sh", "run_n17.sh"):
        subprocess.run(["bash", "-n", str(HERE / script)], check=True, timeout=5)
    result = subprocess.run(["bash", str(HERE / "run_n17.sh"), "n15"], capture_output=True)
    assert result.returncode == 2


def test_logging_steps_validation_and_wiring():
    with pytest.raises(ValueError, match="logging_steps"):
        FinetuneConfig(
            base_model_path="model",
            dataset_path="data",
            embodiment_tag="NEW_EMBODIMENT",
            logging_steps=0,
        )
    source = (REPO / "gr00t/experiment/launch_finetune.py").read_text()
    assert "config.training.logging_steps = ft_config.logging_steps" in source
