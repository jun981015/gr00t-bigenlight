from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from experiments.bigenlight_multitask import cache_shards


def test_shards_preserve_episode_and_partial_batch(monkeypatch):
    class Dataset:
        def get_step_data(self, trajectory, index):
            return {"state": np.array([trajectory, index], dtype=np.float32)}

        @staticmethod
        def transforms(item):
            return item

    monkeypatch.setattr(cache_shards, "_dataset", Dataset())
    monkeypatch.setattr(cache_shards, "_collator", lambda rows: {
        "state": torch.from_numpy(np.stack([row["state"] for row in rows]))
    })
    with ThreadPoolExecutor(2) as executor:
        result = list(cache_shards.bounded_map(
            executor, cache_shards.prepare_episode, [(7, 15, 5, 2), (8, 21, 3, 2)], 2
        ))
    for (episode, trajectory, length, batches), expected in zip(result, [(7, 15, 5), (8, 21, 3)]):
        assert (episode, trajectory, length) == expected
        rows = np.concatenate([inputs["state"] for _, inputs in batches])
        np.testing.assert_array_equal(rows[:, 0], np.full(length, trajectory))
        np.testing.assert_array_equal(rows[:, 1], np.arange(length))
        assert len(batches[-1][1]["state"]) == 1


def test_shard_failure_is_not_silently_skipped():
    def failure(job):
        if job == 2:
            raise ValueError("bad episode")
        return job

    with ThreadPoolExecutor(2) as executor:
        result = cache_shards.bounded_map(executor, failure, [1, 2, 3], 2)
        assert next(result) == 1
        with pytest.raises(ValueError, match="bad episode"):
            next(result)
