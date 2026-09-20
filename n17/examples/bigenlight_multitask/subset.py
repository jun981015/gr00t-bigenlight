"""Select the first N source episode indices per task, without copying videos."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path

from examples.bigenlight_multitask.prepare import (
    DATA_PATH,
    VIDEO_KEYS,
    VIDEO_PATH,
    atomic_json,
    jsonl,
    remap_episode,
)
import pyarrow.parquet as pq


def select(episodes, mapping, tasks, count):
    if count < 1:
        raise ValueError("count must be positive")
    by_id = {row["episode_index"]: row for row in mapping}
    if len(by_id) != len(mapping):
        raise ValueError("Duplicate source mapping IDs")
    groups = defaultdict(list)
    for episode in episodes:
        if len(episode["tasks"]) != 1:
            raise ValueError("Expected exactly one task per episode")
        groups[episode["tasks"][0]].append(episode)
    if set(groups) != {row["task"] for row in tasks}:
        raise ValueError("Task inventory mismatch")
    selected = []
    for task in sorted(tasks, key=lambda row: row["task_index"]):
        rows = groups[task["task"]]
        origins = [by_id[row["episode_index"]] for row in rows]
        if len({row["source_repo"] for row in origins}) != 1:
            raise ValueError("Source-index selection requires one source per task")
        if len({row["source_episode_index"] for row in origins}) != len(rows):
            raise ValueError("Duplicate source episode indices")
        rows.sort(key=lambda row: by_id[row["episode_index"]]["source_episode_index"])
        if len(rows) < count:
            raise ValueError(f"Not enough episodes for {task['task']}")
        selected.extend(rows[:count])
    return selected


def prepare_subset(source, output, count=30):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Source and subset must be separate, non-nested directories")
    ready = json.loads((source / "READY.json").read_text())
    for view in ("n17", "n15"):
        json.loads((source / view / "VALIDATION.json").read_text())
    episodes = [
        json.loads(line) for line in (source / "n17/meta/episodes.jsonl").read_text().splitlines()
    ]
    mapping = json.loads((source / "source_mapping.json").read_text())
    by_id = {row["episode_index"]: row for row in mapping}
    selected = select(episodes, mapping, ready["tasks"], count)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite another dataset.
    n17, n15 = output / "n17", output / "n15"
    atomic_json(
        output / "SOURCE.json",
        {
            "source": str(source),
            "per_task": count,
            "selection": "first N source_episode_index values per task, ascending; not random",
            "parent_episode_indices": [row["episode_index"] for row in selected],
        },
    )
    rows, selected_mapping, offset = [], [], 0
    task_map = {row["task_index"]: row["task_index"] for row in ready["tasks"]}
    for new_id, episode in enumerate(selected):
        old_id = episode["episode_index"]
        old_fields = {"episode_index": old_id, "episode_chunk": old_id // 1000}
        new_fields = {"episode_index": new_id, "episode_chunk": new_id // 1000}
        original = pq.read_table(source / "n17" / DATA_PATH.format(**old_fields))
        if len(original) != episode["length"]:
            raise ValueError("Episode length mismatch")
        table = remap_episode(original, new_id, offset, task_map)
        path = n17 / DATA_PATH.format(**new_fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        for key in VIDEO_KEYS:
            original_video = source / "n17" / VIDEO_PATH.format(video_key=key, **old_fields)
            if not original_video.is_file():
                raise FileNotFoundError(original_video)
            link = n17 / VIDEO_PATH.format(video_key=key, **new_fields)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(original_video, link.parent))
        rows.append({**episode, "episode_index": new_id})
        selected_mapping.append(
            {**by_id[old_id], "episode_index": new_id, "parent_episode_index": old_id}
        )
        offset += len(table)
    info = json.loads((source / "n17/meta/info.json").read_text())
    info.update(
        total_episodes=len(rows),
        total_frames=offset,
        total_videos=2 * len(rows),
        total_chunks=(len(rows) + 999) // 1000,
        splits={"train": f"0:{len(rows)}"},
    )
    modality = json.loads((source / "n17/meta/modality.json").read_text())
    for root in (n17, n15):
        atomic_json(root / "meta/info.json", info)
        atomic_json(root / "meta/modality.json", modality)
        jsonl(root / "meta/episodes.jsonl", rows)
        jsonl(root / "meta/tasks.jsonl", ready["tasks"])
    atomic_json(output / "source_mapping.json", selected_mapping)
    from gr00t.data.stats import generate_stats

    generate_stats(n17)  # Fit on SELECTED episodes only, not the full parent corpus.
    stats = json.loads((n17 / "meta/stats.json").read_text())
    atomic_json(
        n15 / "meta/stats.json", {key: stats[key] for key in ("action", "observation.state")}
    )
    for name in ("data", "videos"):
        (n15 / name).symlink_to(Path("../n17") / name, target_is_directory=True)
    result = deepcopy(ready)
    result.update(
        episodes=len(rows),
        frames=offset,
        video_clips=len(rows) * 2,
        train_split=f"first {count} episodes per task; no held-out evaluation split",
        per_task=count,
        video="symlinks to previously verified full-corpus H264 clips",
        parent_dataset=str(source),
        statistics_fit="selected episodes only",
    )
    atomic_json(output / "READY.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    storage = Path.home() / "raid/vla_finetune/datasets"
    parser.add_argument("--source", type=Path, default=storage / "bigenlight_multitask_gr00t")
    parser.add_argument(
        "--output", type=Path, default=storage / "bigenlight_multitask_gr00t_30per_task"
    )
    parser.add_argument("--per-task", type=int, default=30)
    args = parser.parse_args()
    prepare_subset(args.source, args.output, args.per_task)
