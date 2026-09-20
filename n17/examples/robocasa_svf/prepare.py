"""Pinned downloads and non-destructive N1.7 metadata overlays, entirely on RAID."""

import argparse
import json
import os
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / "recipe.json").read_text())
CAMERAS = {
    "res256_image_side_0": "left_view",
    "res256_image_side_1": "right_view",
    "res256_image_wrist_0": "wrist_view",
}


def root():
    return Path(os.environ.get("VLA_STORAGE_ROOT", Path.home() / "raid/vla_finetune")).resolve()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def download(names):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    for name in names:
        spec = RECIPE["datasets"][name]
        target = root() / "datasets" / spec["directory"]
        print(f"Downloading {name}: {spec['repo_id']} @ {spec['revision']}", flush=True)
        for attempt in range(30):
            try:
                snapshot_download(
                    spec["repo_id"],
                    repo_type="dataset",
                    revision=spec["revision"],
                    local_dir=target,
                    max_workers=2,
                )
                break
            except HfHubHTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in (429, 500, 502, 503, 504) or attempt == 29:
                    raise
                retry = exc.response.headers.get("Retry-After", "300")
                delay = max(300, int(retry)) if retry.isdigit() else 300
                print(f"HTTP {status}: waiting {delay}s before retry {attempt + 1}/30", flush=True)
                time.sleep(delay)
        atomic_json(target / "DOWNLOAD_COMPLETE.json", spec)
        print(f"Completed {name}: {target}", flush=True)


def all_sources(names):
    for name in names:
        source = root() / "datasets" / RECIPE["datasets"][name]["directory"]
        target = root() / "datasets/robocasa_n17" / name
        if name == "bc24":
            yield source, target
        else:
            for group in ("demos", "success_rollouts", "rollouts"):
                for task in RECIPE["tasks"]:
                    yield source / group / task, target / group / task


def prepare_dataset(source, target):
    from gr00t.data.stats import STATS_FINGERPRINTS_KEY, _compute_stats_fingerprint
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and overlay must be separate, non-nested directories")
    provenance = {"version": 1, "source": str(source), "recipe": RECIPE}
    if target.exists() and any(target.iterdir()):
        if (
            not (target / "SOURCE.json").exists()
            or json.loads((target / "SOURCE.json").read_text()) != provenance
        ):
            raise ValueError(f"Refusing to overwrite unowned/different dataset overlay: {target}")
    meta = source / "meta"
    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    episodes = [
        json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines() if line
    ]
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"Incomplete episode metadata: {source}")
    files = []
    for episode in episodes:
        index = episode["episode_index"]
        fields = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        parquet = source / info["data_path"].format(**fields)
        files.append(parquet)
        expected = [parquet]
        for camera in CAMERAS.values():
            video_key = modality["video"][camera].get(
                "original_key", f"observation.images.{camera}"
            )
            expected.append(source / info["video_path"].format(**fields, video_key=video_key))
        for file in expected:
            if not file.is_file() or file.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Download incomplete; no episodes are silently dropped: {file}"
                )
    columns = set(pq.read_schema(files[0]).names)
    language_key = "annotation.human.action.task_description"
    original = modality["annotation"]["human.action.task_description"].get(
        "original_key", language_key
    )
    if original not in columns:
        original = "task_index"
    if original not in columns:
        raise ValueError("No actual instruction index column found")
    modality["annotation"]["human.action.task_description"]["original_key"] = original
    modality["video"] = {key: modality["video"][old] for key, old in CAMERAS.items()}
    target.mkdir(parents=True, exist_ok=True)
    atomic_json(target / "SOURCE.json", provenance)
    (target / "meta").mkdir(exist_ok=True)
    for directory in ("data", "videos"):
        link = target / directory
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != source / directory:
                raise ValueError(f"Unexpected existing overlay member: {link}")
        else:
            link.symlink_to(source / directory, target_is_directory=True)
    for name in ("episodes.jsonl", "tasks.jsonl"):
        (target / "meta" / name).write_text((meta / name).read_text())
    atomic_json(target / "meta/info.json", info)
    atomic_json(target / "meta/modality.json", modality)
    # Read only low-dimensional columns, not the BC dataset's ~126 GB of embedded images.
    features = [key for key, value in info["features"].items() if "float" in value["dtype"]]
    values = {feature: [] for feature in features}
    for file, episode in zip(files, episodes, strict=True):
        frame = pd.read_parquet(file, columns=features)
        if len(frame) != episode["length"]:
            raise ValueError(f"Episode length mismatch: {file}")
        for feature in features:
            values[feature].append(np.vstack(frame[feature].to_numpy()).astype(np.float32))
    stats = {}
    for feature, chunks in values.items():
        array = np.concatenate(chunks)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite training values: {feature}")
        stats[feature] = {
            "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(),
            "min": array.min(0).tolist(),
            "max": array.max(0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }
    stats[STATS_FINGERPRINTS_KEY] = {
        key: _compute_stats_fingerprint(key, info["features"][key]) for key in features
    }
    atomic_json(target / "meta/stats.json", stats)
    atomic_json(
        target / "READY.json",
        {"episodes": len(episodes), "frames": info["total_frames"], "source": str(source)},
    )
    print(f"Prepared {target}: {len(episodes)} episodes; source data unchanged", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("download", "prepare", "download-and-prepare"))
    parser.add_argument("datasets", nargs="+", choices=tuple(RECIPE["datasets"]))
    args = parser.parse_args()
    if args.operation in ("download", "download-and-prepare"):
        # Prepare each dataset as it arrives, without waiting for the large BC archive.
        for name in args.datasets:
            download([name])
            if args.operation == "download-and-prepare":
                for source, target in all_sources([name]):
                    prepare_dataset(source, target)
    else:
        for source, target in all_sources(args.datasets):
            prepare_dataset(source, target)


if __name__ == "__main__":
    main()
