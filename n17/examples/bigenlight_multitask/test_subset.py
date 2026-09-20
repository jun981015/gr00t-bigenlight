"""Deterministic per-task selection, independent of input ordering."""

from examples.bigenlight_multitask.subset import select
import pytest


def fixture():
    tasks = [{"task_index": 0, "task": "a"}, {"task_index": 1, "task": "b"}]
    episodes, mapping = [], []
    for task in tasks:
        for index in range(4):
            episode = len(episodes)
            episodes.append({"episode_index": episode, "tasks": [task["task"]]})
            mapping.append(
                {
                    "episode_index": episode,
                    "source_repo": task["task"],
                    "source_episode_index": index,
                }
            )
    return episodes, mapping, tasks


def test_selection_uses_source_index_not_input_order():
    episodes, mapping, tasks = fixture()
    result = select(list(reversed(episodes)), list(reversed(mapping)), tasks, 2)
    assert [r["episode_index"] for r in result] == [0, 1, 4, 5]


@pytest.mark.parametrize("count", [0, -1, 5])
def test_rejects_invalid_counts(count):
    with pytest.raises(ValueError):
        select(*fixture(), count)


def test_rejects_ambiguous_original_indices():
    episodes, mapping, tasks = fixture()
    mapping[1]["source_episode_index"] = 0
    with pytest.raises(ValueError, match="Duplicate source episode"):
        select(episodes, mapping, tasks, 2)
