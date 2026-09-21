"""Build a read-only-media holdout view: full corpus minus BC training episodes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .feature_cache import atomic_json


def source_key(item):
    return item["source_repo"], item["source_episode_index"]


def prepare(full, train, output):
    full, train, output = (Path(p).resolve() for p in (full, train, output))
    mapping_paths = [p.parent / "source_mapping.json" for p in (full, train)]
    all_rows, train_rows = [json.loads(p.read_text()) for p in mapping_paths]
    all_keys, train_keys = ({source_key(e) for e in rows} for rows in (all_rows, train_rows))
    if len(all_keys) != len(all_rows) or len(train_keys) != len(train_rows):
        raise ValueError("Duplicate source episodes")
    if not train_keys < all_keys:
        raise ValueError("Training episodes must be a strict subset of full corpus")
    selected = [e for e in all_rows if source_key(e) not in train_keys]
    ids = {e["episode_index"] for e in selected}
    episodes = [
        json.loads(line) for line in (full / "meta/episodes.jsonl").read_text().splitlines()
    ]
    episodes = [e for e in episodes if e["episode_index"] in ids]
    if len(episodes) != len(selected):
        raise ValueError("Source mapping and episode metadata differ")
    split = {
        "full_dataset": str(full),
        "train_dataset": str(train),
        "full_mapping_sha256": hashlib.sha256(mapping_paths[0].read_bytes()).hexdigest(),
        "train_mapping_sha256": hashlib.sha256(mapping_paths[1].read_bytes()).hexdigest(),
        "train_sources": [list(key) for key in sorted(train_keys)],
        "heldout_sources": selected,
        "heldout_episode_ids": [e["episode_index"] for e in episodes],
        "tasks": {str(e["episode_index"]): e["tasks"] for e in episodes},
        "normalization": "training dataset stats; extraction uses frozen BC processor stats",
    }
    if output.exists():
        if (
            not (output / "READY.json").exists()
            or json.loads((output / "meta/holdout_split.json").read_text()) != split
        ):
            raise FileExistsError("Existing holdout view differs or is incomplete")
        return split
    (output / "meta").mkdir(parents=True)
    for name in ("data", "videos", "masks"):
        if (full / name).exists():
            (output / name).symlink_to(full / name, target_is_directory=True)
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copyfile(full / "meta" / name, output / "meta" / name)
    for name in ("stats.json", "relative_stats.json"):
        if (train / "meta" / name).exists():
            shutil.copyfile(train / "meta" / name, output / "meta" / name)
    info = json.loads((full / "meta/info.json").read_text())
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(e["length"] for e in episodes),
        total_videos=len(episodes) * 2,
        splits={},
    )
    atomic_json(output / "meta/info.json", info)
    with (output / "meta/episodes.jsonl").open("x") as f:
        for item in episodes:
            f.write(json.dumps(item) + "\n")
    atomic_json(output / "meta/holdout_split.json", split)
    atomic_json(
        output / "READY.json",
        {
            "episodes": len(episodes),
            "frames": info["total_frames"],
            "non_contiguous_source_ids": True,
            "evaluation_only": True,
        },
    )
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = prepare(args.full, args.train, args.output)
    print(
        json.dumps(
            {"heldout_episodes": len(split["heldout_episode_ids"]), "output": str(args.output)}
        )
    )


if __name__ == "__main__":
    main()
