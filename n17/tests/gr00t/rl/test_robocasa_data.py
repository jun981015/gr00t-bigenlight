from gr00t.rl.dataset import (
    ColumnRLAnnotations,
    DEASRoboCasaAnnotations,
    EpisodeBatchSampler,
    EpisodeRLAnnotations,
    LeRobotOfflineRLDataset,
    OfflineRLConcatDataset,
)
import numpy as np
import pytest

from .test_dataset import TAG, MemoryLoader, terminal_labels


@pytest.mark.parametrize("success", [False, True])
def test_deas_explicit_terminal_vs_timeout(monkeypatch, success):
    rewards = np.array([0, 0, int(success)], dtype=np.float32)
    monkeypatch.setattr(
        ColumnRLAnnotations,
        "__call__",
        lambda *args: EpisodeRLAnnotations(rewards, [0, 0, 1], [0, 0, 0]),
    )
    labels = DEASRoboCasaAnnotations()(MemoryLoader((3,)), 0)
    assert labels.terminated[-1] == success
    assert labels.truncated[-1] != success
    np.testing.assert_array_equal(labels.rewards, rewards)


def test_timeout_needs_explicit_no_bootstrap(monkeypatch):
    monkeypatch.setattr(
        ColumnRLAnnotations,
        "__call__",
        lambda *args: EpisodeRLAnnotations([0, 0, 0], [0, 0, 1], [0, 0, 0]),
    )
    with pytest.raises(ValueError, match="missing final"):
        LeRobotOfflineRLDataset(MemoryLoader((3,)), TAG, DEASRoboCasaAnnotations())
    dataset = LeRobotOfflineRLDataset(
        MemoryLoader((3,)), TAG, DEASRoboCasaAnnotations(), bootstrap_on_truncation=False
    )
    assert dataset[-1].truncated and dataset[-1].discount == 0
    assert not dataset[-1].next_observation_valid


@pytest.mark.parametrize(
    "rewards,done", [([0, 0, 0], [0, 0, 0]), ([0, 1, 0], [0, 0, 1]), ([0, -1, 0], [0, 0, 1])]
)
def test_deas_rejects_ambiguous_labels(monkeypatch, rewards, done):
    monkeypatch.setattr(
        ColumnRLAnnotations,
        "__call__",
        lambda *args: EpisodeRLAnnotations(rewards, done, [0, 0, 0]),
    )
    with pytest.raises(ValueError):
        DEASRoboCasaAnnotations()(MemoryLoader((3,)), 0)


def test_multi_dataset_offsets_and_resume_sampler():
    first = LeRobotOfflineRLDataset(MemoryLoader((5, 4)), TAG, terminal_labels, horizon=2)
    second = LeRobotOfflineRLDataset(MemoryLoader((6,)), TAG, terminal_labels, horizon=2)
    mixture = OfflineRLConcatDataset([first, second])
    assert len(mixture) == len(first) + len(second)
    assert mixture[len(first)].episode_index == 2
    assert mixture[-1].episode_index == 2
    assert mixture.episode_sizes == [4, 3, 5]
    batches = list(EpisodeBatchSampler(mixture, 4, 10, seed=4))
    assert batches[4:] == list(EpisodeBatchSampler(mixture, 4, 6, seed=4, start_step=4))
    for batch in batches:
        assert len({mixture[index].episode_index for index in batch}) == 1
    second.gamma = 0.5
    with pytest.raises(ValueError, match="gamma"):
        OfflineRLConcatDataset([first, second])
