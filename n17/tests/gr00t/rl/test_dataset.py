"""RL transition semantics, using both in-memory episodes and real LeRobot parquet."""

from dataclasses import replace
import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)
from gr00t.rl.adapters import StateActionTransitionCollator
from gr00t.rl.dataset import (
    ColumnRLAnnotations,
    EpisodeBatchSampler,
    EpisodeRLAnnotations,
    LeRobotOfflineRLDataset,
)
import numpy as np
import pandas as pd
import pytest


TAG = EmbodimentTag.NEW_EMBODIMENT


def modalities():
    return {
        "state": ModalityConfig([0], ["joint"]),
        "action": ModalityConfig(
            list(range(4)),
            ["joint"],
            action_configs=[
                ActionConfig(
                    ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT
                )
            ],
        ),
        "language": ModalityConfig([0], ["task"]),
    }


class MemoryLoader:
    def __init__(self, lengths=(5, 4)):
        self.modality_configs = modalities()
        self.episode_lengths = lengths
        self.calls = []

    def __len__(self):
        return len(self.episode_lengths)

    def get_episode_length(self, index):
        return self.episode_lengths[index]

    def __getitem__(self, index):
        self.calls.append(index)
        values = np.arange(self.episode_lengths[index], dtype=np.float32) + index * 100
        return pd.DataFrame(
            {
                "state.joint": [np.array([v], dtype=np.float32) for v in values],
                "action.joint": [np.array([v + 0.25], dtype=np.float32) for v in values],
                "language.task": [f"task {index}"] * len(values),
            }
        )


def terminal_labels(loader, index):
    n = loader.get_episode_length(index)
    return EpisodeRLAnnotations(np.arange(1, n + 1), np.arange(n) == n - 1, np.zeros(n, bool))


class IdentityNormalizer:
    def apply(self, state, action, embodiment_tag):
        return state, action

    def apply_state(self, state, embodiment_tag):
        return state


def test_chunk_return_discount_and_no_boundary_crossing():
    loader = MemoryLoader()
    dataset = LeRobotOfflineRLDataset(loader, TAG, terminal_labels, horizon=2, gamma=0.5)
    assert len(dataset) == 7
    first = dataset[0]
    np.testing.assert_array_equal(first.rewards, [1, 2])
    assert first.next_observation.states["joint"].item() == 2
    assert first.next_observation.actions == {}
    assert first.discount == 0.25
    final = dataset[3]
    assert final.terminated and not final.next_observation_valid and final.discount == 0
    assert final.observation.actions["joint"].shape == (2, 1)
    assert dataset[4].observation.states["joint"].item() == 100
    assert loader.modality_configs["action"].delta_indices == [0, 1, 2, 3]
    collate = StateActionTransitionCollator(
        IdentityNormalizer(), {TAG.value: modalities()}, gamma=0.5
    )
    batch = collate([first, final])
    batch.validate()
    np.testing.assert_allclose(batch.rewards, [2, 6.5])
    np.testing.assert_allclose(batch.discounts, [0.25, 0])
    with pytest.raises(ValueError, match="gamma"):
        StateActionTransitionCollator(IdentityNormalizer(), {TAG.value: modalities()}, 0.99)(
            [first]
        )


def test_history_clamps_only_to_same_episode():
    loader = MemoryLoader()
    loader.modality_configs["state"] = ModalityConfig([-1, 0], ["joint"])
    data = LeRobotOfflineRLDataset(loader, TAG, terminal_labels)
    np.testing.assert_array_equal(data[5].observation.states["joint"], [[100], [100]])
    np.testing.assert_array_equal(data[5].next_observation.states["joint"], [[100], [101]])


def test_future_observation_rejected():
    loader = MemoryLoader()
    loader.modality_configs["state"] = ModalityConfig([0, 1], ["joint"])
    with pytest.raises(ValueError, match="leak"):
        LeRobotOfflineRLDataset(loader, TAG, terminal_labels)


def test_timeout_requires_true_next_observation():
    loader = MemoryLoader((3,))

    def labels(_, index):
        return EpisodeRLAnnotations(np.ones(3), np.zeros(3), [0, 0, 1])

    with pytest.raises(ValueError, match="missing final next observation"):
        LeRobotOfflineRLDataset(loader, TAG, labels)
    data = LeRobotOfflineRLDataset(loader, TAG, labels, bootstrap_on_truncation=False)
    assert data[-1].truncated and not data[-1].terminated
    assert data[-1].discount == 0 and not data[-1].next_observation_valid


def test_final_observation_row_bootstraps_timeout_not_terminal():
    loader = MemoryLoader((4,))

    def labels(*_):
        return EpisodeRLAnnotations(np.ones(3), np.zeros(3), [0, 0, 1])

    data = LeRobotOfflineRLDataset(loader, TAG, labels, gamma=0.9)
    assert len(data) == 3
    assert data[-1].next_observation.states["joint"].item() == 3
    assert data[-1].next_observation_valid and data[-1].discount == 0.9
    assert data[-1].truncated


def test_custom_final_observation_callback():
    loader = MemoryLoader((3,))
    terminal = LeRobotOfflineRLDataset(loader, TAG, terminal_labels)[-1]
    following = replace(terminal.next_observation, states={"joint": np.array([[99]], np.float32)})

    def labels(*_):
        return EpisodeRLAnnotations(np.ones(3), np.zeros(3), [0, 0, 1], following)

    data = LeRobotOfflineRLDataset(loader, TAG, labels)
    assert data[-1].next_observation.states["joint"].item() == 99
    assert data[-1].next_observation_valid and data[-1].discount == pytest.approx(0.99)


@pytest.mark.parametrize(
    "bad",
    [
        EpisodeRLAnnotations([1, np.nan, 1], [0, 0, 1], [0, 0, 0]),
        EpisodeRLAnnotations([1, 1, 1], [0, 2, 1], [0, 0, 0]),
        EpisodeRLAnnotations([1, 1, 1], [0, 1, 0], [0, 0, 0]),
        EpisodeRLAnnotations([1, 1, 1], [0, 0, 1], [0, 0, 1]),
    ],
)
def test_bad_annotations_rejected(bad):
    with pytest.raises(ValueError):
        LeRobotOfflineRLDataset(MemoryLoader((3,)), TAG, lambda *_: bad)


def test_cache_and_episode_sampler_resume():
    loader = MemoryLoader()
    data = LeRobotOfflineRLDataset(loader, TAG, terminal_labels)
    data[0], data[1], data[5], data[0]
    assert loader.calls == [0, 1, 0]
    sampler = list(EpisodeBatchSampler(data, 8, 10, seed=9))
    resumed = list(EpisodeBatchSampler(data, 8, 6, seed=9, start_step=4))
    assert sampler[4:] == resumed
    for batch in sampler:
        assert len({data[i].episode_index for i in batch}) == 1


def write_lerobot(root: Path):
    """Real fixture with noncontiguous episode IDs and explicit RL columns."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "fps": 30,
        "features": {},
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {"joint": {"start": 0, "end": 1}},
                "action": {"joint": {"start": 0, "end": 1}},
            }
        )
    )
    stats = {"min": [-1.0], "max": [1.0], "mean": [0.0], "std": [0.5], "q01": [-1.0], "q99": [1.0]}
    (meta / "stats.json").write_text(json.dumps({"observation.state": stats, "action": stats}))
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test task"}) + "\n")
    episodes = []
    for episode_id in (10, 42):
        episodes.append({"episode_index": episode_id, "length": 6, "tasks": ["test task"]})
        frame = pd.DataFrame(
            {
                "observation.state": [np.array([i / 10], np.float32) for i in range(6)],
                "action": [np.array([i / 20], np.float32) for i in range(6)],
                "reward": [0, 0, 0, 0, 1, 0],
                "terminated": [False, False, False, False, True, False],
                "truncated": [False] * 6,
            }
        )
        path = root / info["data_path"].format(episode_chunk=0, episode_index=episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))


def test_actual_lerobot_loader_and_rl_columns(tmp_path):
    write_lerobot(tmp_path)
    loader = LeRobotEpisodeLoader(tmp_path, modalities())
    labels = ColumnRLAnnotations("reward", "terminated", "truncated", last_row_is_observation=True)
    data = LeRobotOfflineRLDataset(loader, TAG, labels, horizon=2)
    assert len(data) == 8
    assert data[3].terminated and data[3].next_observation_valid
    assert data[3].next_observation.states["joint"].item() == pytest.approx(0.5)
    assert data[4].episode_index == 1
    np.testing.assert_array_equal(data[3].rewards, [0, 1])
    with pytest.raises(ValueError, match="Cannot read explicit RL columns"):
        LeRobotOfflineRLDataset(loader, TAG, ColumnRLAnnotations("missing"))


def test_dual_discounts_are_independent_and_opt_in():
    data = LeRobotOfflineRLDataset(
        MemoryLoader(), TAG, terminal_labels, horizon=2, gamma=0.5, bootstrap_gamma=0.9
    )
    collate = StateActionTransitionCollator(IdentityNormalizer(), {TAG.value: modalities()}, 0.5)
    batch = collate([data[0], data[3]])
    batch.validate()
    np.testing.assert_allclose(batch.rewards, [2, 6.5])
    np.testing.assert_allclose(batch.discounts, [0.9**2, 0])
    assert data[0].reward_gamma == 0.5
    # The gamma guard must also catch mismatches for terminal-only batches.
    with pytest.raises(ValueError, match="gamma"):
        StateActionTransitionCollator(IdentityNormalizer(), {TAG.value: modalities()}, 0.9)(
            [data[3]]
        )
    with pytest.raises(ValueError, match="bootstrap_gamma"):
        LeRobotOfflineRLDataset(MemoryLoader(), TAG, terminal_labels, bootstrap_gamma=float("nan"))


def test_strided_action_indices_rejected():
    loader = MemoryLoader()
    loader.modality_configs["action"] = replace(
        loader.modality_configs["action"], delta_indices=[0, 2, 4]
    )
    with pytest.raises(ValueError, match="contiguous"):
        LeRobotOfflineRLDataset(loader, TAG, terminal_labels)
