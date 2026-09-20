"""CPU-only regression tests for nearest-PTS seeking and bounded frame reuse."""

import os
from fractions import Fraction

import av
import numpy as np
import pytest

from gr00t.utils import pyav_frames as reader


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    reader._CACHE.clear()
    reader._CACHE_BYTES = 0
    monkeypatch.setenv("GR00T_PYAV_CACHE_BYTES", "0")
    monkeypatch.setenv("GR00T_PYAV_DECODE_MODE", "seek")


def make_video(path, timestamps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.time_base = Fraction(1, 10)
        stream.options = {"g": "5", "bf": "2", "sc_threshold": "0"}
        for i, pts in enumerate(timestamps):
            rgb = np.random.default_rng(i).integers(0, 256, (48, 64, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def reference(path, requested):
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        decoded = list(container.decode(video=0))
        times = np.array([float(f.pts * f.time_base) for f in decoded])
        images = np.stack([f.to_ndarray(format="rgb24") for f in decoded])
    return images[reader.nearest_indices(times, requested)], times


@pytest.mark.parametrize("pts", [list(range(40)), [0, 1, 3, 4, 30, 31, 60, 90], list(range(30, 65))])
def test_seek_matches_full_decode_cfr_vfr_offset_and_bframes(tmp_path, pts):
    path = tmp_path / "sample.mp4"
    make_video(path, pts)
    _, times = reference(path, [0])
    queries = np.concatenate(
        [times, (times[:-1] + times[1:]) / 2, times + 1e-6, times - 1e-6, [-10, 100], times[[0, -1, 0]]]
    )[::-1].copy()
    expected, _ = reference(path, queries)
    actual = reader.read_frames(path, queries)
    np.testing.assert_array_equal(actual, expected)


def test_cache_budget_copy_and_file_invalidation(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    make_video(path, list(range(15)))
    size = 48 * 64 * 3
    monkeypatch.setenv("GR00T_PYAV_CACHE_BYTES", str(size))
    original = reader._seek_frame
    calls = []

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(reader, "_seek_frame", tracked)
    first = reader.read_frames(path, [0.2])
    expected = first.copy()
    first[:] = 0
    np.testing.assert_array_equal(reader.read_frames(path, [0.2]), expected)
    assert len(calls) == 1
    reader.read_frames(path, [0.4, 0.6, 0.8])
    assert reader._CACHE_BYTES <= size
    before = len(calls)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    reader.read_frames(path, [0.8])
    assert len(calls) == before + 1
    monkeypatch.setenv("GR00T_PYAV_CACHE_BYTES", "0")
    reader.read_frames(path, [0.8])
    assert reader._CACHE_BYTES == 0 and not reader._CACHE


def test_reference_mode(tmp_path, monkeypatch):
    path = tmp_path / "sample.mp4"
    make_video(path, list(range(15)))
    expected, _ = reference(path, [0, 0.55, 50])
    monkeypatch.setenv("GR00T_PYAV_DECODE_MODE", "full")
    np.testing.assert_array_equal(reader.read_frames(path, [0, 0.55, 50]), expected)


@pytest.mark.parametrize("requested", [[], [[1]], [float("nan")], [float("inf")]])
def test_invalid_requests_rejected(requested):
    with pytest.raises(ValueError, match="timestamps"):
        reader.read_frames("unused.mp4", requested)


def test_nearest_ties_choose_earlier():
    np.testing.assert_array_equal(reader.nearest_indices([0.0, 1.0, 2.0], [0.5, 1.5, -1, 5]), [0, 1, 0, 2])
