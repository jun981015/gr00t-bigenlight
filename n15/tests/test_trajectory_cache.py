"""Keep only the current parquet episode, and never reuse it for another ID."""

from types import SimpleNamespace

import pandas as pd

from gr00t.data.dataset import LeRobotSingleDataset


def test_current_trajectory_cache_tracks_id(tmp_path, monkeypatch):
    for index in (0, 1):
        (tmp_path / f"episode_{index}.parquet").touch()
    calls = []

    def read(path):
        calls.append(path.name)
        return pd.DataFrame({"episode": [int(path.stem.split("_")[-1])]})

    monkeypatch.setattr(pd, "read_parquet", read)
    dataset = SimpleNamespace(
        curr_traj_id=None,
        curr_traj_data=None,
        dataset_path=tmp_path,
        data_path_pattern="episode_{episode_index}.parquet",
        get_episode_chunk=lambda _: 0,
    )
    first = LeRobotSingleDataset.get_trajectory_data(dataset, 0)
    assert LeRobotSingleDataset.get_trajectory_data(dataset, 0) is first
    second = LeRobotSingleDataset.get_trajectory_data(dataset, 1)
    assert second is not first and second.iloc[0].episode == 1
    assert LeRobotSingleDataset.get_trajectory_data(dataset, 1) is second
    assert calls == ["episode_0.parquet", "episode_1.parquet"]
