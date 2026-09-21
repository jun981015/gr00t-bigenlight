import hashlib
import json

from gr00t.rl.eval_cached_iql import discounted_returns, evaluate, main, validate_provenance
from gr00t.rl.feature_cache import CachedFeatureDataset, atomic_json
from gr00t.rl.iql import FeatureValue, IQLCriticLearner
from gr00t.rl.networks import FeatureCritic
from gr00t.rl.prepare_critic_holdout import prepare
import numpy as np
import pytest
import torch

from .test_feature_cache import make_cache


@pytest.mark.parametrize(
    "gamma,expected", [(0, [-1, -1, 0]), (0.9, [-1.9, -1, 0]), (1, [-2, -1, 0])]
)
def test_demo_returns(gamma, expected):
    np.testing.assert_allclose(discounted_returns([-1, -1, 0], gamma), expected)


def test_exact_return_critic_and_terminal_mask(tmp_path):
    root = tmp_path / "cache"
    make_cache(root)
    dataset = CachedFeatureDataset(root, reward="step-cost", gamma=0.9)

    class ExactValue(torch.nn.Module):
        def forward(self, obs):
            x = obs["features"][:, 0]
            remaining = torch.where(x >= 100, 103 - x, 4 - x)
            return -(1 - 0.9**remaining) / 0.1

    class ExactQ(ExactValue):
        def forward(self, obs, actions):
            v = super().forward(obs)
            return torch.stack([v, v])

    report, data = evaluate(
        dataset,
        ExactQ(),
        ExactValue(),
        ExactQ(),
        0.7,
        {"0": ["task_a"], "1": ["task_b"]},
        batch_size=2,
    )
    assert report["transitions"] == 7
    assert report["episode_weighted"]["q_mc_mae"] < 1e-5
    assert report["episode_weighted"]["td_mae"] < 1e-5
    assert len(report["per_task"]) == 2
    assert data["terminal"].sum() == 2
    np.testing.assert_allclose(data["td_target"][data["terminal"]], -1)


def setup_eval(tmp_path):
    full, train = tmp_path / "full/n17", tmp_path / "train/n17"
    mapping = [
        {"source_repo": "repo", "source_episode_index": i, "episode_index": i} for i in range(4)
    ]
    for directory, rows in [
        (full, mapping),
        (train, [{**mapping[2], "episode_index": 0}, {**mapping[3], "episode_index": 1}]),
    ]:
        (directory / "meta").mkdir(parents=True)
        atomic_json(directory.parent / "source_mapping.json", rows)
        atomic_json(directory / "meta/info.json", {})
        atomic_json(directory / "meta/modality.json", {})
        atomic_json(directory / "meta/stats.json", {"train_only": directory == train})
        (directory / "meta/tasks.jsonl").write_text("{}\n")
        (directory / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "episode_index": r["episode_index"],
                        "tasks": ["task"],
                        "length": 5 if i % 2 == 0 else 4,
                    }
                )
                + "\n"
                for i, r in enumerate(rows)
            )
        )
        (directory / "data").mkdir()
        (directory / "videos").mkdir()
    holdout = tmp_path / "holdout/n17"
    prepare(full, train, holdout)
    assert prepare(full, train, holdout)["heldout_episode_ids"] == [0, 1]
    assert (holdout / "data").resolve() == full / "data"
    assert json.loads((holdout / "meta/stats.json").read_text())["train_only"]
    roots = [tmp_path / "train_cache", tmp_path / "holdout_cache"]
    for root, directory in zip(roots, [train, holdout], strict=True):
        make_cache(root)
        manifest = json.loads((root / "manifest.json").read_text())
        identity = manifest["identity"]
        identity.update(
            model_path="/model",
            dataset_path=str(directory),
            weight_file_stats={},
            embodiment="test",
            encoder_sha256={"/repo/gr00t/rl/adapters.py": "same"},
            configs_sha256={"/model/config.json": "same"},
        )
        if directory == holdout:
            path = directory / "meta/holdout_split.json"
            identity["configs_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(root / "manifest.json", manifest)
        atomic_json(
            root / "COMPLETE.json",
            {
                "format": identity["format"],
                "manifest_sha256": hashlib.sha256(
                    (root / "manifest.json").read_bytes()
                ).hexdigest(),
            },
        )
    train_manifest = json.loads((roots[0] / "manifest.json").read_text())
    learner = IQLCriticLearner(FeatureCritic(1, (4, 3), (8,)), FeatureValue(1, (8,)))
    algorithm = learner.state_dict()
    algorithm.pop("optimizer")
    checkpoint = {
        "model_only": True,
        "step": 30000,
        "algorithm": algorithm,
        "metadata": {
            "backend": "frozen-bc-cache-v1",
            "bc_identity": train_manifest["identity"],
            "cache_manifest_sha256": hashlib.sha256(
                (roots[0] / "manifest.json").read_bytes()
            ).hexdigest(),
            "args": {"reward": "step-cost", "gamma": 0.9, "hidden_dim": 8, "hidden_layers": 1},
        },
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    return roots, path, checkpoint


def test_eval_cli_and_provenance_guards(tmp_path):
    roots, checkpoint_path, checkpoint = setup_eval(tmp_path)
    output = tmp_path / "eval"
    main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--train-cache",
            str(roots[0]),
            "--eval-cache",
            str(roots[1]),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )
    result = json.loads((output / "report.json").read_text())
    assert result["step"] == 30000 and result["transitions"] == 7
    assert len((output / "predictions.csv").read_text().splitlines()) == 8
    dataset = CachedFeatureDataset(roots[1], reward="step-cost", gamma=0.9)
    dataset.manifest["identity"]["model_path"] = "/wrong-bc"
    with pytest.raises(ValueError, match="BC mismatch"):
        validate_provenance(checkpoint, roots[0], dataset)
    dataset = CachedFeatureDataset(roots[1], reward="step-cost", gamma=0.9)
    checkpoint["metadata"]["cache_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Wrong training cache"):
        validate_provenance(checkpoint, roots[0], dataset)
