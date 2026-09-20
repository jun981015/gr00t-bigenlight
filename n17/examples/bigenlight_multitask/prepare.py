"""Pinned downloads + one 256-episode UR7e corpus, with N1.7/N1.5 metadata views.

Run as a module from Isaac-GR00T using the N1.7 environment. No GPU needed.
The four source snapshots are immutable inputs. An interrupted preparation can
be rerun; existing completed video files are reused, never source files edited.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path

from examples.carrot_in_pot.prepare_dataset import (
    DATA_PATH,
    VIDEO_KEYS,
    VIDEO_PATH,
    extract_video,
    modality_metadata,
    validate_episode,
)
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCES = [
    ("carrot_in_pot_lighting_lerobot_v3", "77695b1bcc8cb615004742eb2ef1f9681863f44c", 65),
    ("bowl_stack_lighting_lerobot_v3", "ab446a6bf6e6641a9eb494e00aa15c6a99fd5522", 71),
    ("bowl_stack_triple_lerobot_v3", "4494eb4bc5d662abe1ec74097304ef64b7461b56", 60),
    ("cube_stack_lerobot_v3", "6b1b2353556f2dd571198ef3b5a89f64ae65e80d", 60),
]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def replace_integer_column(table, key, values):
    index = table.schema.get_field_index(key)
    if index < 0:
        raise ValueError(f"Missing column: {key}")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def remap_episode(table, episode, offset, task_map):
    """Change identifiers only; timestamps and recorded vectors stay bit-exact."""
    mapped = [task_map[int(task)] for task in table["task_index"].to_pylist()]
    for key, values in (
        ("episode_index", np.full(len(table), episode)),
        ("index", np.arange(offset, offset + len(table))),
        ("task_index", mapped),
    ):
        table = replace_integer_column(table, key, values)
    return table


def convert_video(source, destination, start, length, fps):
    # Final files are only installed after ffmpeg + full frame-count validation.
    if destination.is_file():
        return
    temporary = destination.with_name(f".{destination.stem}.partial.mp4")
    if temporary.exists():
        temporary.unlink()  # Only our incomplete staging file, never source data.
    extract_video(source, temporary, start, length, fps)
    temporary.replace(destination)


def compatible(first, other):
    for key in ("codebase_version", "robot_type", "fps", "features"):
        if first[key] != other[key]:
            raise ValueError(f"Incompatible source schema: {key}")
    if first["codebase_version"] != "v3.0" or first["robot_type"] != "ur7e_gello":
        raise ValueError("Expected UR7e LeRobot v3 sources")


def prepare(storage, download=True, workers=4):
    storage = Path(storage).resolve()
    output = storage / "bigenlight_multitask_gr00t"
    provenance = {"format": 1, "sources": SOURCES, "split": "all episodes for BC training"}
    # Normalize tuples to JSON lists before comparing the ownership marker.
    provenance = json.loads(json.dumps(provenance))
    marker = output / "SOURCE.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != provenance:
            raise ValueError(f"Refusing to modify unowned output: {output}")
    atomic_json(marker, provenance)
    if (output / "READY.json").exists():
        print(f"Already prepared: {output}", flush=True)
        return output
    infos, roots = [], []
    for name, revision, expected in SOURCES:
        root = storage / name
        if download:
            snapshot_download(
                "Bigenlight/" + name,
                repo_type="dataset",
                revision=revision,
                local_dir=root,
                max_workers=4,
            )
        info = json.loads((root / "meta/info.json").read_text())
        if info["total_episodes"] != expected:
            raise ValueError(f"Unexpected episode count: {name}")
        if infos:
            compatible(infos[0], info)
        else:
            compatible(info, info)
        roots.append(root)
        infos.append(info)

    n17, n15 = output / "n17", output / "n15"
    tasks, task_lookup, episodes, mapping, jobs = [], {}, [], [], []
    offset = 0
    for (name, revision, _), source, info in zip(SOURCES, roots, infos, strict=True):
        task_map = {}
        for text, row in pd.read_parquet(source / "meta/tasks.parquet").iterrows():
            text = str(text)
            if text not in task_lookup:
                task_lookup[text] = len(tasks)
                tasks.append({"task_index": len(tasks), "task": text})
            task_map[int(row["task_index"])] = task_lookup[text]
        records = pd.concat(
            [
                pd.read_parquet(p)
                for p in sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
            ]
        )
        records = records.sort_values("episode_index").to_dict("records")
        if len(records) != info["total_episodes"] or len(
            {r["episode_index"] for r in records}
        ) != len(records):
            raise ValueError(f"Invalid source episode inventory: {source}")
        tables = {}
        source_frames = 0
        for record in records:
            file_key = int(record["data/chunk_index"]), int(record["data/file_index"])
            if file_key not in tables:
                tables[file_key] = pq.read_table(
                    source
                    / info["data_path"].format(chunk_index=file_key[0], file_index=file_key[1])
                )
            original_id = int(record["episode_index"])
            raw = tables[file_key].filter(pc.equal(tables[file_key]["episode_index"], original_id))
            validate_episode(raw, record, info["fps"])
            episode, length = len(episodes), len(raw)
            table = remap_episode(raw, episode, offset, task_map)
            fields = {"episode_index": episode, "episode_chunk": episode // 1000}
            target = n17 / DATA_PATH.format(**fields)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            descriptions = [tasks[t]["task"] for t in sorted(set(table["task_index"].to_pylist()))]
            if descriptions != list(record["tasks"]):
                raise ValueError(f"Task metadata and frame labels disagree: {source}:{original_id}")
            episodes.append({"episode_index": episode, "length": length, "tasks": descriptions})
            videos = {}
            for key in VIDEO_KEYS:
                prefix = f"videos/{key}"
                relative = info["video_path"].format(
                    video_key=key,
                    chunk_index=int(record[f"{prefix}/chunk_index"]),
                    file_index=int(record[f"{prefix}/file_index"]),
                )
                start, end = (
                    float(record[f"{prefix}/from_timestamp"]),
                    float(record[f"{prefix}/to_timestamp"]),
                )
                if abs((end - start) * info["fps"] - length) > 1e-3:
                    raise ValueError("Video duration and episode length differ")
                if abs(start * info["fps"] - round(start * info["fps"])) > 1e-3:
                    raise ValueError("Video start is not frame-aligned")
                destination = n17 / VIDEO_PATH.format(video_key=key, **fields)
                jobs.append((source / relative, destination, start, length, info["fps"]))
                videos[key] = {"path": relative, "start_frame": round(start * info["fps"])}
            mapping.append(
                {
                    "episode_index": episode,
                    "source_repo": "Bigenlight/" + name,
                    "revision": revision,
                    "source_episode_index": original_id,
                    "length": length,
                    "source_data_path": info["data_path"].format(
                        chunk_index=file_key[0], file_index=file_key[1]
                    ),
                    "videos": videos,
                }
            )
            source_frames += length
            offset += length
        if (
            source_frames != info["total_frames"]
            or sum(len(t) for t in tables.values()) != source_frames
        ):
            raise ValueError(f"Source frame count mismatch: {source}")

    combined = deepcopy(infos[0])
    combined.update(
        codebase_version="v2.1",
        total_episodes=len(episodes),
        total_frames=offset,
        total_tasks=len(tasks),
        total_videos=len(jobs),
        total_chunks=(len(episodes) + 999) // 1000,
        chunks_size=1000,
        data_path=DATA_PATH,
        video_path=VIDEO_PATH,
        splits={"train": f"0:{len(episodes)}"},
    )
    for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
        combined.pop(key, None)
    for key in VIDEO_KEYS:
        combined["features"][key]["info"] = {
            "video.height": 720,
            "video.width": 1280,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.fps": 30,
            "has_audio": False,
        }
    for root in (n17, n15):
        atomic_json(root / "meta/info.json", combined)
        atomic_json(root / "meta/modality.json", modality_metadata())
        jsonl(root / "meta/tasks.jsonl", tasks)
        jsonl(root / "meta/episodes.jsonl", episodes)
    atomic_json(output / "source_mapping.json", mapping)
    print(
        f"Converting {len(jobs)} RGB clips; {len(episodes)} episodes, {offset} frames", flush=True
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(convert_video, *job) for job in jobs]
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            if count % 64 == 0 or count == len(jobs):
                print(f"Verified clips: {count}/{len(jobs)}", flush=True)
    from gr00t.data.stats import generate_stats

    generate_stats(n17)
    stats = json.loads((n17 / "meta/stats.json").read_text())
    # N1.5 treats N1.7's __fingerprints__ entry as numerical data. Isolate metadata,
    # but share all large payload files so this is still ONE physical corpus.
    atomic_json(
        n15 / "meta/stats.json", {key: stats[key] for key in ("action", "observation.state")}
    )
    for name in ("data", "videos"):
        path = n15 / name
        if path.is_symlink() and path.resolve() == (n17 / name).resolve():
            continue
        if path.exists() or path.is_symlink():
            raise ValueError(f"Unexpected existing payload: {path}")
        path.symlink_to(Path("../n17") / name, target_is_directory=True)
    atomic_json(
        output / "READY.json",
        {
            "episodes": len(episodes),
            "frames": offset,
            "tasks": tasks,
            "original_vectors_preserved": True,
            "video_clips": len(jobs),
            "video": "H264 CRF18 full resolution; frame counts verified",
            "train_split": "all 256 episodes; no held-out evaluation split",
            "rl_labels": "not fabricated; BC only",
            "sources": provenance["sources"],
        },
    )
    print(f"READY: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path.home() / "raid/vla_finetune/datasets")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    # Do not run two preparations against the same output simultaneously.
    import fcntl

    args.storage.mkdir(parents=True, exist_ok=True)
    with (args.storage / ".bigenlight-multitask-prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.storage, not args.skip_download, args.workers)
