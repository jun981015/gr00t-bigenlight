"""Keyframe-seeking PyAV reader with nearest-PTS selection and bounded frame reuse.

The default reads only the GOP preceding a requested timestamp, converts only
selected frames to RGB, and caches selected images (64 MiB per worker). Existing
DataLoader workers provide asynchronous prefetch; sampling/augmentation do not
change. Set GR00T_PYAV_DECODE_MODE=full for the reference decoder, and
GR00T_PYAV_CACHE_BYTES=0 to disable the image cache.
"""

import math
import os
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np

_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CACHE_PID = os.getpid()


def nearest_indices(frame_times, requested):
    frame_times, requested = np.asarray(frame_times), np.asarray(requested)
    if frame_times.ndim != 1 or len(frame_times) == 0 or not np.isfinite(frame_times).all():
        raise ValueError("Invalid frame timestamps")
    if np.any(np.diff(frame_times) < 0) or requested.ndim != 1 or not np.isfinite(requested).all():
        raise ValueError("Expected ordered frame timestamps and finite 1D requests")
    right = np.clip(np.searchsorted(frame_times, requested), 0, len(frame_times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(frame_times[left] - requested) <= np.abs(frame_times[right] - requested), left, right)


def _time(frame):
    if frame.pts is None:
        raise ValueError("Missing video PTS")
    return float(frame.pts * frame.time_base)


def _scan(container, stream, timestamp, seeked):
    previous, previous_time = None, None
    for frame in container.decode(stream):
        current_time = _time(frame)
        if previous_time is not None and current_time < previous_time:
            raise ValueError("Expected ordered frame timestamps")
        # A seek may land after the desired predecessor (sparse/VFR media).
        # Reopen from the start instead of silently returning a wrong frame.
        if previous is None and current_time > timestamp and seeked:
            return None
        if current_time >= timestamp:
            if previous is not None and timestamp - previous_time <= current_time - timestamp:
                return previous.to_ndarray(format="rgb24")
            return frame.to_ndarray(format="rgb24")
        previous, previous_time = frame, current_time
    return None if previous is None else previous.to_ndarray(format="rgb24")


def _seek_frame(source, timestamp):
    # A one-second preroll retains a predecessor across keyframe boundaries.
    # PTS comparisons, not guessed FPS/frame indices, decide which frame wins.
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        seeked = False
        if timestamp > 0:
            offset = math.floor((timestamp - 1.0) / float(stream.time_base))
            try:
                container.seek(offset, stream=stream, backward=True, any_frame=False)
                seeked = True
            except (av.error.FFmpegError, OverflowError):
                pass  # Reopen below; a failed seek may have changed demux state.
            if not seeked:
                result = None
            else:
                result = _scan(container, stream, timestamp, seeked=True)
        else:
            result = _scan(container, stream, timestamp, seeked=False)
    if result is None:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            result = _scan(container, stream, timestamp, seeked=False)
    if result is None:
        raise ValueError(f"Empty video: {source}")
    return result


def _full_frames(source, requested):
    """Original full-RGB-decode behavior, retained for parity tests/rollback."""
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        frames, times = [], []
        for frame in container.decode(stream):
            times.append(_time(frame))
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"Empty video: {source}")
    indices = nearest_indices(times, requested)
    return [frames[index] for index in indices]


def read_frames(path, timestamps):
    global _CACHE_BYTES, _CACHE_PID
    requested = np.asarray(timestamps, dtype=np.float64)
    if requested.ndim != 1 or len(requested) == 0 or not np.isfinite(requested).all():
        raise ValueError("Expected nonempty finite 1D timestamps")
    mode = os.environ.get("GR00T_PYAV_DECODE_MODE", "seek")
    if mode not in ("seek", "full"):
        raise ValueError("GR00T_PYAV_DECODE_MODE must be seek or full")
    limit = int(os.environ.get("GR00T_PYAV_CACHE_BYTES", str(64 * 1024**2)))
    if limit < 0:
        raise ValueError("Video cache byte limit must be nonnegative")
    if _CACHE_PID != os.getpid():
        _CACHE.clear()
        _CACHE_BYTES, _CACHE_PID = 0, os.getpid()
    while _CACHE and _CACHE_BYTES > limit:
        _CACHE_BYTES -= _CACHE.popitem(last=False)[1].nbytes
    source = Path(path).resolve()
    stat = source.stat()
    identity = (str(source), stat.st_size, stat.st_mtime_ns, mode)
    images = {}
    missing = []
    for timestamp in np.unique(requested):
        key = (*identity, float(timestamp))
        if key in _CACHE:
            images[timestamp] = _CACHE.pop(key)
            _CACHE[key] = images[timestamp]
        else:
            missing.append(timestamp)
    if missing:
        decoded = (
            _full_frames(source, missing)
            if mode == "full"
            else [_seek_frame(source, timestamp) for timestamp in missing]
        )
        for timestamp, frame in zip(missing, decoded):
            images[timestamp] = frame
            if frame.nbytes <= limit:
                while _CACHE and _CACHE_BYTES + frame.nbytes > limit:
                    _CACHE_BYTES -= _CACHE.popitem(last=False)[1].nbytes
                _CACHE[(*identity, float(timestamp))] = frame
                _CACHE_BYTES += frame.nbytes
    # Copy so callers/augmentation cannot alter the cache. Preserve request order.
    return np.stack([images[timestamp] for timestamp in requested])
