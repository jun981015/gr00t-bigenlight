"""Conversion invariants; no HF downloads or GPU work."""

from pathlib import Path
import runpy
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


# tests/examples is a regular package and shadows the repository's namespace
# examples package under pytest's configured pythonpath. Load by exact path.
_converter = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/carrot_in_pot/prepare_dataset.py")
)
extract_video = _converter["extract_video"]
modality_metadata = _converter["modality_metadata"]
prepare = _converter["prepare"]
probe_video = _converter["probe_video"]
split_episodes = _converter["split_episodes"]
validate_episode = _converter["validate_episode"]


def test_split_is_disjoint_complete_deterministic():
    split = split_episodes(range(54))
    assert len(split["train"]) == 49 and len(split["val"]) == 5
    assert not set(split["train"]) & set(split["val"])
    assert sorted(split["train"] + split["val"]) == list(range(54))
    assert split == split_episodes(reversed(range(54)))
    with pytest.raises(ValueError):
        split_episodes([0, 0, 1], 1)


def test_robot_groups_and_camera_mapping():
    meta = modality_metadata()
    assert meta["state"]["arm"] == {"start": 0, "end": 6}
    assert meta["action"]["gripper"] == {"start": 6, "end": 7}
    assert meta["video"]["scene"]["original_key"] == "observation.images.cam1"
    assert meta["video"]["wrist"]["original_key"] == "observation.images.cam2"


def test_episode_validation_rejects_bad_timestamps_and_values():
    frame = pd.DataFrame(
        {
            "episode_index": [4] * 3,
            "frame_index": [0, 1, 2],
            "index": [17, 18, 19],
            "timestamp": np.arange(3, dtype=np.float32) / 30,
            "action": [np.zeros(7, np.float32) for _ in range(3)],
            "observation.state": [np.zeros(7, np.float32) for _ in range(3)],
        }
    )
    record = {"episode_index": 4, "length": 3, "dataset_from_index": 17, "dataset_to_index": 20}
    validate_episode(pa.Table.from_pandas(frame), record, 30)
    bad = frame.copy()
    bad["timestamp"] += 0.1
    with pytest.raises(ValueError, match="timestamp"):
        validate_episode(pa.Table.from_pandas(bad), record, 30)
    frame.loc[0, "action"][0] = np.nan
    with pytest.raises(ValueError, match="invalid action"):
        validate_episode(pa.Table.from_pandas(frame), record, 30)


def test_prepare_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "missing_source", output)


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_video_extraction_keeps_last_decodable_frame(tmp_path):
    source, output = tmp_path / "source.mp4", tmp_path / "episode.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "veryfast",
            "-g",
            "10",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    # Start between keyframes and preserve all five requested frames, including the last.
    extract_video(source, output, 3 / 30, 5, 30)
    metadata = probe_video(output)
    assert int(metadata["nb_frames"]) == int(metadata["nb_read_frames"]) == 5
    assert float(metadata["duration"]) == pytest.approx(5 / 30, abs=1e-6)
