"""CPU-only merge invariants; no downloads."""

from copy import deepcopy

from examples.bigenlight_multitask.prepare import SOURCES, compatible, prepare, remap_episode
import numpy as np
import pyarrow as pa
import pytest


def test_source_inventory():
    assert sum(count for _, _, count in SOURCES) == 256
    assert len({name for name, _, _ in SOURCES}) == 4
    assert all(len(revision) == 40 for _, revision, _ in SOURCES)


def test_remapping_preserves_vectors_and_does_not_mutate_source():
    table = pa.table(
        {
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
            "timestamp": np.array([0, 1 / 30], dtype=np.float32),
            "action": [list(range(7)), list(range(7))],
            "observation.state": [list(range(7)), list(range(7))],
        }
    )
    merged = remap_episode(table, 65, 20472, {0: 1})
    assert merged["episode_index"].to_pylist() == [65, 65]
    assert merged["index"].to_pylist() == [20472, 20473]
    assert merged["task_index"].to_pylist() == [1, 1]
    for key in ("timestamp", "action", "observation.state"):
        assert merged[key].equals(table[key])
    assert table["episode_index"].to_pylist() == [0, 0]
    with pytest.raises(KeyError):
        remap_episode(table, 65, 20472, {1: 1})


@pytest.mark.parametrize(
    "key,new",
    [
        ("fps", 20),
        ("robot_type", "other"),
        ("codebase_version", "v2.1"),
        ("features", {"action": {"shape": [8]}}),
    ],
)
def test_rejects_schema_mismatch(key, new):
    first = {"fps": 30, "robot_type": "ur7e_gello", "codebase_version": "v3.0", "features": {}}
    other = deepcopy(first)
    other[key] = new
    with pytest.raises(ValueError):
        compatible(first, other)


def test_does_not_overwrite_unowned_destination(tmp_path):
    root = tmp_path / "bigenlight_multitask_gr00t"
    root.mkdir()
    (root / "precious.txt").write_text("keep me")
    with pytest.raises(ValueError, match="unowned"):
        prepare(tmp_path, download=False)
    assert (root / "precious.txt").read_text() == "keep me"
