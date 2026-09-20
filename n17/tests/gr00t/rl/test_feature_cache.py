import hashlib
import json

from gr00t.rl.adapters import StateActionTransitionCollator
from gr00t.rl.dataset import (
    AllSuccessStepCostAnnotations,
    AllSuccessTerminalAnnotations,
    LeRobotOfflineRLDataset,
)
from gr00t.rl.feature_cache import (
    FORMAT,
    CachedFeatureDataset,
    atomic_json,
    cache_identity,
    identities_match,
    write_episode,
)
from gr00t.rl.train_cached import main
import numpy as np
import pytest
import torch

from .test_dataset import TAG, IdentityNormalizer, MemoryLoader, modalities


def test_identity_allows_only_checkout_relocation():
    from copy import deepcopy

    original = {
        "model_path": "/raid/original-bc",
        "dataset_path": "/raid/original-data",
        "horizon": 16,
        "encoder_sha256": {
            "/old/gr00t/rl/adapters.py": "adapter-hash",
            "/old/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py": "processor-hash",
        },
    }
    relocated = deepcopy(original)
    relocated["encoder_sha256"] = {
        k.replace("/old/", "/workspace/n17/"): v for k, v in original["encoder_sha256"].items()
    }
    assert identities_match(original, relocated)
    changed = deepcopy(relocated)
    changed["encoder_sha256"]["/workspace/n17/gr00t/rl/adapters.py"] = "different"
    assert not identities_match(original, changed)
    for key, value in [
        ("horizon", 8),
        ("model_path", "/raid/another-bc"),
        ("dataset_path", "/raid/another-data"),
    ]:
        assert not identities_match(original, {**relocated, key: value})


def make_cache(root):
    root.mkdir()
    source = MemoryLoader((5, 4))
    manifest = {
        "identity": {"format": FORMAT, "horizon": 2},
        "feature_dim": 1,
        "action_shape": [4, 3],
        "action_indices": [0, 3],
        "episodes": [
            {"file": f"episode-{i:06d}.npz", "rows": n, "valid_starts": n - 1, "episode_index": i}
            for i, n in enumerate(source.episode_lengths)
        ],
    }
    for i, n in enumerate(source.episode_lengths):
        features = (np.arange(n, dtype=np.float32) + 100 * i)[:, None]
        actions = np.stack([features[:-1, 0] + 0.25, features[1:, 0] + 0.25], -1)
        write_episode(
            root / manifest["episodes"][i]["file"],
            features,
            actions,
            {"features": (n, 1), "actions": (n - 1, 2)},
        )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "COMPLETE.json",
        {
            "format": FORMAT,
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        },
    )
    return source


@pytest.mark.parametrize(
    "reward,preset",
    [
        ("terminal-success", AllSuccessTerminalAnnotations),
        ("step-cost", AllSuccessStepCostAnnotations),
    ],
)
def test_cached_batches_match_live_transition_semantics(tmp_path, reward, preset):
    root = tmp_path / "cache"
    source = make_cache(root)
    cached = CachedFeatureDataset(root, reward=reward, gamma=0.9)
    live = LeRobotOfflineRLDataset(source, TAG, preset(), horizon=2, gamma=0.9)
    indices = [0, 1, 3, 4, 6]
    expected = StateActionTransitionCollator(
        IdentityNormalizer(), {TAG.value: modalities()}, gamma=0.9
    )([live[i] for i in indices])
    actual = cached.batch(indices)
    actual.validate()
    for key in ["rewards", "discounts", "terminated", "truncated", "horizons"]:
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key))
    torch.testing.assert_close(actual.observations["features"], expected.observations["features"])
    torch.testing.assert_close(
        actual.next_observations["features"], expected.next_observations["features"]
    )
    torch.testing.assert_close(actual.actions[:, :2, :1], expected.actions)
    assert not actual.actions[:, 2:].any() and not actual.actions[:, :, 1:].any()
    with pytest.raises(IndexError):
        cached.batch([len(cached)])


def test_incomplete_corrupt_and_duplicate_cache_rejected(tmp_path):
    root = tmp_path / "cache"
    make_cache(root)
    with pytest.raises(FileExistsError):
        write_episode(
            root / "episode-000000.npz",
            np.zeros((5, 1)),
            np.zeros((4, 2)),
            {"features": (5, 1), "actions": (4, 2)},
        )
    complete = (root / "COMPLETE.json").read_text()
    (root / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError):
        CachedFeatureDataset(root)
    (root / "COMPLETE.json").write_text(complete)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["feature_dim"] = 999
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest"):
        CachedFeatureDataset(root)


def test_nonfinite_episode_never_published(tmp_path):
    with pytest.raises(ValueError, match="Nonfinite"):
        write_episode(
            tmp_path / "bad.npz",
            np.array([[np.nan]]),
            np.zeros((1, 1)),
            {"features": (1, 1), "actions": (1, 1)},
        )
    assert not (tmp_path / "bad.npz").exists()


def test_cache_identity_changes_with_source_config(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    (data / "meta").mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"test fingerprint only")
    (model / "config.json").write_text("{}")
    a = cache_identity(model, data, 16, "NEW_EMBODIMENT")
    (model / "config.json").write_text('{"different":true}')
    assert a != cache_identity(model, data, 16, "NEW_EMBODIMENT")


def test_cached_training_and_resume_without_model(tmp_path):
    root = tmp_path / "cache"
    make_cache(root)
    output = tmp_path / "run"
    args = [
        "--cache",
        str(root),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--hidden-dim",
        "8",
        "--hidden-layers",
        "1",
        "--reward",
        "step-cost",
    ]
    main([*args, "--steps", "2"])
    checkpoint = output / "checkpoints/step-2.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["step"] == 2 and state["algorithm"]["algorithm"] == "iql-critic-only-v1"
    main([*args, "--steps", "3", "--resume", str(checkpoint)])
    assert (output / "checkpoints/step-3.pt").exists()
    with pytest.raises(ValueError, match="metadata"):
        main(
            [
                *args,
                "--steps",
                "4",
                "--reward",
                "terminal-success",
                "--resume",
                str(output / "checkpoints/step-3.pt"),
            ]
        )
