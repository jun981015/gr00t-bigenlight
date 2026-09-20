"""Reuse the exact N1.7 few-shot selection in an isolated N1.5 metadata overlay."""

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROFILES = {
    "long": [("5shot-seed42", "libero_10")],
    "unified": [
        ("1shot-seed42", suite) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    ],
}
MODALITY = {
    "state": {
        "eef_pos_absolute": {"start": 0, "end": 3},
        "eef_rot_absolute": {"start": 3, "end": 6, "rotation_type": "axis_angle"},
        "gripper_close": {"start": 6, "end": 8},
    },
    "action": {
        "eef_pos_delta": {"start": 0, "end": 3, "absolute": False},
        "eef_rot_delta": {"start": 3, "end": 6, "absolute": False},
        "gripper_close": {"start": 6, "end": 7},
    },
    "video": {
        "front_view": {"original_key": "observation.images.image"},
        "left_wrist_view": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def storage():
    return Path(os.environ.get("VLA_STORAGE_ROOT", Path.home() / "raid/vla_finetune")).resolve()


def paths(profile, version="n15"):
    return [storage() / f"datasets/libero_{version}_bc" / subset / suite for subset, suite in PROFILES[profile]]


def atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and overlay must be separate, non-nested directories")
    ready = json.loads((source / "READY.json").read_text())
    validated = json.loads((source / "VALIDATION.json").read_text())
    if ready["provenance"] != validated["provenance"] or validated["validated_tasks"] != 10:
        raise ValueError("Source selection must have passed N1.7 CPU validation")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    ids = [ep["episode_index"] for ep in episodes]
    if ids != ready["provenance"]["selected_episode_ids"] or len(set(ids)) != len(ids):
        raise ValueError("Selection IDs differ from provenance")
    info = json.loads((source / "meta/info.json").read_text())
    if len(episodes) != ready["episodes"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Episode inventory mismatch")
    expected_count = int(source.parent.name.split("shot-", 1)[0])
    counts = Counter(ep["tasks"][0] for ep in episodes)
    if len(counts) != 10 or set(counts.values()) != {expected_count}:
        raise ValueError("Expected balanced few-shot trajectories for ten tasks")
    copies = ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json")
    provenance = {
        "format": "libero-n15-bc-v1",
        "source": str(source),
        "selection": ready["provenance"],
        "action_horizon": 16,
        "modality": MODALITY,
        "metadata_sha256": {
            name: hashlib.sha256((source / "meta" / name).read_bytes()).hexdigest() for name in copies
        },
    }
    marker = target / "SOURCE.json"
    if target.exists() and any(target.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != provenance:
            raise ValueError(f"Refusing to overwrite unowned/different dataset: {target}")
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, provenance)
    meta = target / "meta"
    meta.mkdir(exist_ok=True)
    for name in copies:
        if name != "stats.json":
            shutil.copyfile(source / "meta" / name, meta / name)
    # N1.7 adds cache/provenance entries that N1.5 mistakes for numerical stats.
    # Keep the selected-subset statistics for the two numerical modalities only.
    source_stats = json.loads((source / "meta/stats.json").read_text())
    atomic_json(meta / "stats.json", {key: source_stats[key] for key in ("observation.state", "action")})
    atomic_json(meta / "modality.json", MODALITY)
    for episode in episodes:
        fields = {
            "episode_index": episode["episode_index"],
            "episode_chunk": episode["episode_index"] // info["chunks_size"],
        }
        assets = [info["data_path"].format(**fields)]
        assets += [
            info["video_path"].format(**fields, video_key=value["original_key"])
            for value in MODALITY["video"].values()
        ]
        for name in assets:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe dataset member")
            src, dst = source / relative, target / relative
            if not src.is_file() or src.stat().st_size == 0:
                raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                if not dst.is_symlink() or dst.resolve() != src.resolve():
                    raise ValueError(f"Refusing to replace {dst}")
            else:
                dst.symlink_to(src.resolve())
    atomic_json(
        target / "READY.json",
        {"provenance": provenance, "episodes": len(episodes), "frames": info["total_frames"], "rl_ready": False},
    )
    return target


def validate(target):
    """Decode one trajectory per task and run the original DEAS LIBERO transforms on CPU."""
    import numpy as np

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.experiment.data_config import LiberoDataConfig

    config = LiberoDataConfig()
    dataset = LeRobotSingleDataset(
        target,
        config.modality_config(),
        "new_embodiment",
        video_backend="torchvision_av",
        transforms=config.transform(),
    )
    seen = set()
    offset = 0
    for episode in read_jsonl(target / "meta/episodes.jsonl"):
        task = episode["tasks"][0]
        if task not in seen:
            sample = dataset[offset]
            assert sample["action"].shape == (16, 32)
            assert sample["state"].shape == (1, 64)
            assert np.isfinite(sample["action"]).all() and np.isfinite(sample["state"]).all()
            assert sample["action_mask"].sum() == 16 * 7
            assert sample["eagle_content"]
            seen.add(task)
        offset += episode["length"]
    if len(seen) != 10:
        raise ValueError("Did not validate all ten tasks")
    ready = json.loads((target / "READY.json").read_text())
    result = {
        "provenance": ready["provenance"],
        "validated_tasks": len(seen),
        "episodes": ready["episodes"],
        "action_horizon": 16,
        "action_dim": 7,
    }
    atomic_json(target / "VALIDATION.json", result)
    return result
