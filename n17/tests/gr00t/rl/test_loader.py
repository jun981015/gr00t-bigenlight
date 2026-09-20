import pickle

from gr00t.rl.adapters import StateActionTransitionCollator
from gr00t.rl.dataset import EpisodeBatchSampler, LeRobotOfflineRLDataset
from gr00t.rl.train import build_batches
import torch

from .test_dataset import TAG, IdentityNormalizer, MemoryLoader, modalities, terminal_labels


def test_spawn_prefetch_matches_serial_and_resume_sequence():
    ds = LeRobotOfflineRLDataset(MemoryLoader(), TAG, terminal_labels, horizon=2)
    collator = StateActionTransitionCollator(IdentityNormalizer(), {TAG.value: modalities()})
    expected = list(build_batches(ds, EpisodeBatchSampler(ds, 3, 4, 7, 10), collator))
    loader = build_batches(ds, EpisodeBatchSampler(ds, 3, 4, 7, 10), collator, workers=2)
    try:
        actual = list(loader)
        for a, b in zip(actual, expected, strict=True):
            for key in ("actions", "action_mask", "rewards", "discounts", "terminated", "horizons"):
                torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
            torch.testing.assert_close(a.observations["features"], b.observations["features"])
    finally:
        loader._iterator._shutdown_workers()


def test_cache_is_bounded_and_not_pickled():
    loader = MemoryLoader()
    ds = LeRobotOfflineRLDataset(
        loader, TAG, terminal_labels, horizon=2, cache_episodes=2, cache_bytes=100000
    )
    ds[0]
    size = ds._cache_nbytes
    assert size > 0
    ds.cache_bytes = size
    ds[4]
    assert ds._cache_nbytes <= size and len(ds._cache) == 1
    restored = pickle.loads(pickle.dumps(ds))
    assert not restored._cache and restored._cache_nbytes == 0
    restored[0]
    assert restored._cache_nbytes <= size


def test_oversized_episode_decoded_once_per_batch():
    source = MemoryLoader()
    ds = LeRobotOfflineRLDataset(source, TAG, terminal_labels, horizon=2, cache_bytes=1)
    samples = ds.__getitems__([0, 1, 2, 0])
    assert len(samples) == 4 and source.calls == [0]
    assert not ds._cache and ds._active_batch_episode is None
