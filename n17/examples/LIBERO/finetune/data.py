"""Pinned public demos -> isolated GR00T BC datasets; never invent RL labels."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time


HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / "recipe.json").read_text())


def storage():
    return Path(os.environ.get("VLA_STORAGE_ROOT", Path.home() / "raid/vla_finetune")).resolve()


def variant(demos_per_task, seed):
    if demos_per_task < 0 or seed < 0:
        raise ValueError("demos_per_task/seed must be nonnegative")
    return "full" if demos_per_task == 0 else f"{demos_per_task}shot-seed{seed}"


def dataset_path(suite, demos_per_task=0, seed=42):
    return storage() / "datasets/libero_n17_bc" / variant(demos_per_task, seed) / suite


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def download(spec, source, patterns=None):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    print(f"Downloading {spec['repo_id']} @ {spec['revision']}", flush=True)
    for attempt in range(12):
        try:
            snapshot_download(
                spec["repo_id"],
                repo_type="dataset",
                revision=spec["revision"],
                local_dir=source,
                allow_patterns=patterns or ["meta/*", "data/**", "videos/**", "README.md"],
                max_workers=2,
            )
            return
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in (429, 500, 502, 503, 504) or attempt == 11:
                raise
            retry = exc.response.headers.get("Retry-After", "300")
            delay = max(300, int(retry)) if retry.isdigit() else 300
            print(f"HTTP {status}: waiting {delay}s before retry {attempt + 1}/12", flush=True)
            time.sleep(delay)


def download_selected(spec, source, demos_per_task, seed):
    """Fetch just the selected demos, reusing any full-download files/cache locks."""
    download(spec, source, ["meta/*", "README.md"])
    info = json.loads((source / "meta/info.json").read_text())
    selected = select_episodes(
        read_jsonl(source / "meta/episodes.jsonl"),
        read_jsonl(source / "meta/tasks.jsonl"),
        demos_per_task,
        seed,
    )
    cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    patterns = []
    for episode in selected:
        index = episode["episode_index"]
        fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
        patterns.append(info["data_path"].format(**fields))
        patterns.extend(info["video_path"].format(**fields, video_key=camera) for camera in cameras)
    download(spec, source, patterns)


def select_episodes(episodes, tasks, demos_per_task=0, seed=42):
    variant(demos_per_task, seed)
    groups = defaultdict(list)
    known = {task["task"] for task in tasks}
    ids = [ep["episode_index"] for ep in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate episode IDs")
    for episode in episodes:
        if len(episode["tasks"]) != 1 or episode["tasks"][0] not in known:
            raise ValueError("Each LIBERO episode must belong to one known task")
        groups[episode["tasks"][0]].append(episode)
    if set(groups) != known:
        raise ValueError("Missing task demonstrations")
    rng = random.Random(seed)
    selected = []
    for task in sorted(groups):
        members = sorted(groups[task], key=lambda ep: ep["episode_index"])
        if demos_per_task > len(members):
            raise ValueError(f"Not enough demos for {task}")
        selected.extend(rng.sample(members, demos_per_task) if demos_per_task else members)
    return sorted(selected, key=lambda ep: ep["episode_index"])


def safe_link(source, target):
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise ValueError(f"Refusing to replace an existing file: {target}")
    else:
        target.symlink_to(source.resolve())


def prepare(source, target, *, suite, spec, demos_per_task=0, seed=42):
    from gr00t.data.stats import generate_stats
    import numpy as np
    import pyarrow.parquet as pq

    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and prepared dataset must be separate, non-nested directories")
    info = json.loads((source / "meta/info.json").read_text())
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    tasks = read_jsonl(source / "meta/tasks.jsonl")
    if len(episodes) != spec["episodes"] or len(tasks) != 10:
        raise ValueError("Pinned dataset episode/task count mismatch")
    if sum(ep["length"] for ep in episodes) != spec["frames"]:
        raise ValueError("Pinned dataset frame count mismatch")
    selected = select_episodes(episodes, tasks, demos_per_task, seed)
    modality_path = HERE.parent / "modality.json"
    patch = HERE.parent / "patches/episode_000082.mp4"
    provenance = {
        "format": "libero-n17-bc-v1",
        "suite": suite,
        "source": str(source),
        "dataset": spec,
        "demos_per_task": demos_per_task,
        "seed": seed,
        "selected_episode_ids": [ep["episode_index"] for ep in selected],
        "modality_sha256": hashlib.sha256(modality_path.read_bytes()).hexdigest(),
        "goal_wrist_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()
        if suite == "libero_goal"
        else None,
    }
    if target.exists() and any(target.iterdir()):
        marker = target / "SOURCE.json"
        if not marker.is_file() or json.loads(marker.read_text()) != provenance:
            raise ValueError(f"Refusing to overwrite a different/unowned dataset: {target}")
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(target / "SOURCE.json", provenance)
        cameras = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        if set(cameras) != {"observation.images.image", "observation.images.wrist_image"}:
            raise ValueError("Unexpected LIBERO camera schema")
        tasks_by_id = {t["task_index"]: t["task"] for t in tasks}
        for ep in selected:
            index = ep["episode_index"]
            fields = dict(episode_index=index, episode_chunk=index // info["chunks_size"])
            name = info["data_path"].format(**fields)
            frame = pq.read_table(source / name)
            if len(frame) != ep["length"]:
                raise ValueError(f"Incorrect episode length: {name}")
            if set(frame["episode_index"].to_pylist()) != {index}:
                raise ValueError(f"Incorrect episode IDs: {name}")
            actual_tasks = {tasks_by_id[i] for i in frame["task_index"].to_pylist()}
            if actual_tasks != set(ep["tasks"]):
                raise ValueError(f"Incorrect task instruction: {name}")
            for key, width in (("observation.state", 8), ("action", 7)):
                value = np.asarray(frame[key].to_pylist())
                if value.shape != (len(frame), width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key}: {name}")
            safe_link(source / name, target / name)
            for camera in cameras:
                name = info["video_path"].format(**fields, video_key=camera)
                video = source / name
                if suite == "libero_goal" and index == 82 and camera.endswith("wrist_image"):
                    # Official NVIDIA repair; never overwrite the downloaded original.
                    video = patch
                safe_link(video, target / name)
        meta = target / "meta"
        meta.mkdir(exist_ok=True)
        (meta / "episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in selected))
        (meta / "tasks.jsonl").write_text((source / "meta/tasks.jsonl").read_text())
        (meta / "modality.json").write_text(modality_path.read_text())
        prepared_info = deepcopy(info)
        prepared_info.update(
            total_episodes=len(selected),
            total_frames=sum(ep["length"] for ep in selected),
            total_videos=len(selected) * len(cameras),
            splits={"train": f"0:{len(selected)}"},
        )
        atomic_json(meta / "info.json", prepared_info)
        # Only selected parquet symlinks exist here: even a cache recomputation
        # cannot use demonstrations excluded from a few-shot subset.
        generate_stats(target)
        summary = {
            "provenance": provenance,
            "episodes": len(selected),
            "frames": prepared_info["total_frames"],
            "task_counts": dict(Counter(ep["tasks"][0] for ep in selected)),
            "rl_ready": False,
            "reason": "Source contains no reward/terminated/truncated labels",
        }
        atomic_json(target / "READY.json", summary)
    print(
        json.dumps({"dataset": str(target), "episodes": len(selected), "rl_ready": False}),
        flush=True,
    )
    return summary


def validate(target):
    """CPU decode through the actual GR00T loader: one episode per task + known repair."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    import numpy as np

    target = Path(target)
    ready = json.loads((target / "READY.json").read_text())
    loader = LeRobotEpisodeLoader(target, MODALITY_CONFIGS["libero_sim"])
    seen, checked = set(), []
    for index, episode in enumerate(loader.episodes_metadata):
        task = episode["tasks"][0]
        if task in seen and episode["episode_index"] != 82:
            continue
        frame = loader[index]
        if len(frame) != episode["length"]:
            raise ValueError("Decoded episode length mismatch")
        for key in ("video.image", "video.wrist_image"):
            images = np.stack(frame[key].to_list())
            if images.shape != (len(frame), 256, 256, 3) or images.dtype != np.uint8:
                raise ValueError(f"Invalid RGB video: {key}")
        seen.add(task)
        checked.append(episode["episode_index"])
    if len(seen) != 10:
        raise ValueError("Did not validate all ten tasks")
    report = {
        "provenance": ready["provenance"],
        "checked_episode_ids": checked,
        "validated_tasks": len(seen),
        "all_lowdim_rows_checked": True,
        "all_videos_decoded": len(checked) == len(loader),
    }
    atomic_json(target / "VALIDATION.json", report)
    print(
        json.dumps(
            {"dataset": str(target), **{k: v for k, v in report.items() if k != "provenance"}}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download-and-prepare", "prepare", "validate"))
    parser.add_argument(
        "--suite", nargs="+", choices=tuple(RECIPE["datasets"]), default=list(RECIPE["datasets"])
    )
    parser.add_argument(
        "--demos-per-task",
        type=int,
        default=0,
        help="0=all demos; positive=seeded task-balanced few-shot",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    variant(args.demos_per_task, args.seed)
    for suite in args.suite:
        spec = RECIPE["datasets"][suite]
        source = storage() / "datasets/libero_public" / suite
        target = dataset_path(suite, args.demos_per_task, args.seed)
        if args.operation == "download-and-prepare":
            if args.demos_per_task:
                download_selected(spec, source, args.demos_per_task, args.seed)
            else:
                download(spec, source)
        if args.operation != "validate":
            prepare(
                source,
                target,
                suite=suite,
                spec=spec,
                demos_per_task=args.demos_per_task,
                seed=args.seed,
            )
        validate(target)


if __name__ == "__main__":
    main()
