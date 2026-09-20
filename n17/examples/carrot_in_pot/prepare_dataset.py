"""Prepare the pinned carrot LeRobot v3 RGB subset for GR00T's v2.1 loader.

Run in the existing GR00T environment. No lerobot package or GPU is needed.
Original vectors/video snapshots are never modified. Only RGB is consumed.
Videos are frame-accurately re-encoded to H.264, retaining 1280x720 at 30 fps.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ID = "Bigenlight/carrot_in_pot_lerobot_v3"
REVISION = "a079393868f3f97915562309e9664f0941c37398"
VIDEO_KEYS = ("observation.images.cam1", "observation.images.cam2")
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + "\n")


def split_episodes(episode_ids, validation_episodes=5, seed=42):
    ids = sorted(int(i) for i in episode_ids)
    if len(ids) != len(set(ids)) or not 0 < validation_episodes < len(ids):
        raise ValueError("Need unique episode IDs and nonempty train/validation splits")
    val = set(np.random.default_rng(seed).choice(ids, validation_episodes, replace=False).tolist())
    return {"train": [i for i in ids if i not in val], "val": [i for i in ids if i in val]}


def modality_metadata():
    return {
        "state": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {"arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {
            "scene": {"original_key": VIDEO_KEYS[0]},
            "wrist": {"original_key": VIDEO_KEYS[1]},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def probe_video(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,start_time,codec_name,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)["streams"][0]


def extract_video(source, destination, start, length, fps):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Decode from the requested timestamp, not the preceding GOP keyframe.
    # The stock converter uses stream-copy, which can change episode boundaries.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-threads",
            "2",
            "-ss",
            f"{start:.9f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            str(length),
            "-an",
            "-vf",
            f"setpts=N/({fps}*TB)",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(fps),
            "-fps_mode",
            "cfr",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    meta = probe_video(destination)
    num, den = map(int, meta["r_frame_rate"].split("/"))
    if (
        int(meta["nb_frames"]) != length
        or int(meta["nb_read_frames"]) != length
        or abs(float(meta["duration"]) - length / fps) > 1e-5
        or num / den != fps
        or (meta["width"], meta["height"]) != (1280, 720)
        or abs(float(meta["start_time"])) > 1e-6
    ):
        raise ValueError(f"Unexpected converted video metadata: {destination}: {meta}")
    return str(destination)


def validate_episode(table, record, fps):
    episode = int(record["episode_index"])
    length = int(record["length"])
    frame = table.to_pandas()
    if (
        len(frame) != length
        or int(record["dataset_to_index"]) - int(record["dataset_from_index"]) != length
    ):
        raise ValueError(f"Episode {episode}: metadata length mismatch")
    if not np.array_equal(frame["frame_index"], np.arange(length)):
        raise ValueError(f"Episode {episode}: noncontiguous frames")
    if not np.array_equal(
        frame["index"], np.arange(record["dataset_from_index"], record["dataset_to_index"])
    ):
        raise ValueError(f"Episode {episode}: global index mismatch")
    if not np.allclose(frame["timestamp"], np.arange(length) / fps, atol=2e-6, rtol=0):
        raise ValueError(f"Episode {episode}: timestamp mismatch")
    for key in ("action", "observation.state"):
        values = np.stack(frame[key])
        if (
            values.shape != (length, 7)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Episode {episode}: invalid {key}")
        if (values[:, 6] < 0).any() or (values[:, 6] > 1).any():
            raise ValueError(f"Episode {episode}: gripper must remain in [0,1]")


def prepare(source, output, validation_episodes=5, seed=42, workers=4):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing preparation: {output}")
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate directories")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    info = json.loads((source / "meta/info.json").read_text())
    if (
        info["codebase_version"] != "v3.0"
        or info["robot_type"] != "ur7e_gello"
        or info["fps"] != 30
    ):
        raise ValueError("This preparation profile expects UR7e LeRobot v3 at 30Hz")
    for key, names in {
        "action": [f"cmd{i}" for i in range(1, 7)] + ["grip_cmd"],
        "observation.state": [f"ur_q{i}" for i in range(1, 7)] + ["grip_pos"],
    }.items():
        if info["features"][key]["names"] != names:
            raise ValueError(f"Unexpected joint ordering: {key}")
    records = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
        ]
    ).to_dict("records")
    if len(records) != info["total_episodes"]:
        raise ValueError("Episode count mismatch")
    by_id = {int(r["episode_index"]): r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate source episode IDs")
    splits = split_episodes(list(by_id), validation_episodes, seed)
    tasks = pd.read_parquet(source / "meta/tasks.parquet")
    task_rows = [
        {"task_index": int(row["task_index"]), "task": str(task)} for task, row in tasks.iterrows()
    ]
    if task_rows != [{"task_index": 0, "task": "Put carrot in pot"}]:
        raise ValueError("Unexpected task mapping")
    tables = {}
    for record in records:
        key = int(record["data/chunk_index"]), int(record["data/file_index"])
        if key not in tables:
            path = source / info["data_path"].format(chunk_index=key[0], file_index=key[1])
            tables[key] = pq.read_table(path)
    if sum(len(t) for t in tables.values()) != info["total_frames"]:
        raise ValueError("Total parquet row count mismatch")
    output.mkdir(parents=True)
    video_jobs, mapping = [], []
    for split, source_ids in splits.items():
        root, episode_rows, offset = output / split, [], 0
        for local_id, original_id in enumerate(source_ids):
            record = by_id[original_id]
            table = tables[int(record["data/chunk_index"]), int(record["data/file_index"])]
            table = table.filter(pc.equal(table["episode_index"], original_id))
            validate_episode(table, record, info["fps"])
            length = len(table)
            for field, values in (
                ("episode_index", np.full(length, local_id, dtype=np.int64)),
                ("index", np.arange(offset, offset + length, dtype=np.int64)),
            ):
                idx = table.schema.get_field_index(field)
                table = table.set_column(idx, table.schema.field(idx), pa.array(values))
            target = root / DATA_PATH.format(episode_chunk=local_id // 1000, episode_index=local_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            for video_key in VIDEO_KEYS:
                prefix = f"videos/{video_key}"
                src = source / info["video_path"].format(
                    video_key=video_key,
                    chunk_index=int(record[f"{prefix}/chunk_index"]),
                    file_index=int(record[f"{prefix}/file_index"]),
                )
                start, end = (
                    float(record[f"{prefix}/from_timestamp"]),
                    float(record[f"{prefix}/to_timestamp"]),
                )
                if (
                    abs((end - start) * info["fps"] - length) > 1e-3
                    or abs(start * info["fps"] - round(start * info["fps"])) > 1e-3
                ):
                    raise ValueError(f"Episode {original_id}: video is not frame-aligned")
                dst = root / VIDEO_PATH.format(
                    episode_chunk=local_id // 1000, video_key=video_key, episode_index=local_id
                )
                video_jobs.append((src, dst, start, length, info["fps"]))
            episode_rows.append(
                {"episode_index": local_id, "length": length, "tasks": ["Put carrot in pot"]}
            )
            mapping.append(
                {
                    "split": split,
                    "episode_index": local_id,
                    "source_episode_index": original_id,
                    "length": length,
                    "source_from_index": int(record["dataset_from_index"]),
                }
            )
            offset += length
        target_info = deepcopy(info)
        target_info.update(
            codebase_version="v2.1",
            data_path=DATA_PATH,
            video_path=VIDEO_PATH,
            total_episodes=len(source_ids),
            total_frames=offset,
            chunks_size=1000,
            total_chunks=1,
            total_videos=len(source_ids) * 2,
            splits={"train": f"0:{len(source_ids)}"},
        )
        target_info["features"] = {
            k: v
            for k, v in target_info["features"].items()
            if v["dtype"] != "video" or k in VIDEO_KEYS
        }
        for key in VIDEO_KEYS:
            target_info["features"][key]["info"] = {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": False,
            }
        for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
            target_info.pop(key, None)
        write_json(root / "meta/info.json", target_info)
        write_json(root / "meta/modality.json", modality_metadata())
        write_jsonl(root / "meta/episodes.jsonl", episode_rows)
        write_jsonl(root / "meta/tasks.jsonl", task_rows)
    write_json(
        output / "source_mapping.json",
        {
            "source_repo": REPO_ID,
            "revision": REVISION,
            "source_info_sha256": hashlib.sha256(
                (source / "meta/info.json").read_bytes()
            ).hexdigest(),
            "seed": seed,
            "episodes": mapping,
            "state_action_values": "unchanged float32",
            "reward_labels": "absent; no success/terminal/reward labels fabricated",
        },
    )
    print(f"Transcoding {len(video_jobs)} RGB episodes with {workers} CPU workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(extract_video, *job) for job in video_jobs]
        for completed, future in enumerate(as_completed(pending), 1):
            future.result()
            if completed % 10 == 0 or completed == len(pending):
                print(f"Videos verified: {completed}/{len(pending)}", flush=True)
    # Fit statistics on TRAIN ONLY. Validation must use the training normalization.
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import generate_rel_stats, generate_stats

    runpy.run_path(str(Path(__file__).with_name("carrot_config.py")))
    generate_stats(output / "train")
    generate_rel_stats(output / "train", EmbodimentTag.NEW_EMBODIMENT)
    for name in ("stats.json", "relative_stats.json"):
        shutil.copy2(output / "train/meta" / name, output / "val/meta" / name)
    write_json(
        output / "PREPARATION_COMPLETE.json",
        {
            "source_revision": REVISION,
            "total_frames": sum(r["length"] for r in mapping),
            "split_episode_counts": {k: len(v) for k, v in splits.items()},
            "video_count": len(video_jobs),
            "video_frame_counts_verified": True,
            "normalization_fit_split": "train",
            "lossy_video_reencoding": "H264 CRF18, full resolution",
        },
    )
    print(f"PREPARATION_COMPLETE {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    storage = Path.home() / "raid/vla_finetune/datasets"
    parser.add_argument("--source", type=Path, default=storage / "carrot_in_pot_lerobot_v3")
    parser.add_argument("--output", type=Path, default=storage / "carrot_in_pot_gr00t")
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.source, args.output, args.validation_episodes, args.seed, args.workers)


if __name__ == "__main__":
    main()
