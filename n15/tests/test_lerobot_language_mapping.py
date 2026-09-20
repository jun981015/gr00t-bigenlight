from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gr00t.data.dataset import LeRobotSingleDataset


@pytest.mark.parametrize("original_key,legacy", [(None, False), (None, True), ("custom_task", False)])
def test_language_mapping_and_temporal_padding(original_key, legacy):
    key = "annotation.human.action.task_description"
    columns = {"task_index": [1, 0]}
    if legacy:
        columns[key] = [0, 1]
    if original_key:
        columns[original_key] = [0, 1]
    dataset = SimpleNamespace(
        curr_traj_data=pd.DataFrame(columns),
        delta_indices={key: np.array([-1, 0, 1, 2])},
        get_trajectory_index=lambda _: 0,
        trajectory_lengths=[2],
        lerobot_modality_meta=SimpleNamespace(annotation={
            "human.action.task_description": SimpleNamespace(original_key=original_key)}),
        tasks=pd.DataFrame({"task": ["close drawer", "open microwave"]}),
    )
    actual = LeRobotSingleDataset.get_language(dataset, 0, key, 0)
    expected = ["close drawer", "close drawer", "open microwave", "open microwave"]
    if not legacy and original_key is None:
        expected = ["open microwave", "open microwave", "close drawer", "close drawer"]
    assert actual == expected


def test_explicit_missing_mapping_does_not_fall_back():
    key = "annotation.human.action.task_description"
    dataset = SimpleNamespace(
        curr_traj_data=pd.DataFrame({"task_index": [0]}),
        delta_indices={key: np.array([0])},
        get_trajectory_index=lambda _: 0,
        trajectory_lengths=[1],
        lerobot_modality_meta=SimpleNamespace(annotation={
            "human.action.task_description": SimpleNamespace(original_key="missing")}),
    )
    with pytest.raises(KeyError, match="missing"):
        LeRobotSingleDataset.get_language(dataset, 0, key, 0)
