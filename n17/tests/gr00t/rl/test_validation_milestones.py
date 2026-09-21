import json

from gr00t.rl.validate_action_iql_milestones import completed_report, update_comparison
import pytest


def report(root, step, digest="same"):
    directory = root / f"validation-step-{step}"
    directory.mkdir()
    (directory / "report.json").write_text(
        json.dumps(
            {
                "step": step,
                "q_aggregation": "mean",
                "eval_cache_manifest_sha256": digest,
                "transition_weighted": {"mae": 1},
                "episode_weighted": {},
                "per_task": {},
            }
        )
    )
    return directory


def test_complete_requires_predictions_and_wandb(tmp_path):
    directory = report(tmp_path, 100000)
    assert not completed_report(directory, 100000, False)
    (directory / "predictions.npz").touch()
    assert completed_report(directory, 100000, False)
    assert not completed_report(directory, 100000, True)
    (directory / "wandb.json").write_text('{"url": "test"}')
    assert completed_report(directory, 100000, True)
    assert not completed_report(directory, 150000, True)


def test_comparison_retains_old_milestones_and_checks_cache(tmp_path):
    first, second = report(tmp_path, 5000), report(tmp_path, 100000)
    update_comparison(tmp_path, [second, first])
    comparison = json.loads((tmp_path / "validation-comparison.json").read_text())
    assert list(comparison) == ["5000", "100000"]
    third = report(tmp_path, 150000, "different")
    with pytest.raises(ValueError, match="holdout"):
        update_comparison(tmp_path, [first, third])
