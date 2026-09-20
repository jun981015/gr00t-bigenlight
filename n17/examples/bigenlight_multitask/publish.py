"""Stage and publish only approved public Bigenlight corpus files (never logs/tokens).

Large payloads are hardlinked into a RAID export directory, dereferencing local
video/view symlinks. HF receives ordinary file contents in both independent views.
No source files are modified. Requires --execute to create public repositories.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

from examples.bigenlight_multitask.prepare import DATA_PATH, VIDEO_KEYS, VIDEO_PATH, atomic_json
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


VARIANTS = {
    "30per_task": ("_30per_task", 120),
    "50per_task": ("_50per_task", 200),
    "all": ("", 256),
}
REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
STORAGE = (Path.home() / "raid/vla_finetune").resolve()


def card(repo_id, ready, counts, mapping):
    suffix = repo_id.rsplit("_", 1)[-1]
    selection = (
        "All available episodes from the four source datasets."
        if suffix == "all"
        else f"First {ready['per_task']} source episode indices per task (ascending, not random)."
    )
    table = "\n".join(
        f"| [{name}](https://huggingface.co/datasets/{name}) | {count} |"
        for name, count in counts.items()
    )
    revisions = {row["source_repo"]: row["revision"] for row in mapping}
    pins = "\n".join(f"- `{name}`: `{revision}`" for name, revision in revisions.items())
    return f'''---
license: apache-2.0
task_categories:
  - robotics
tags:
  - lerobot
  - gr00t
  - ur7e
  - imitation-learning
  - multitask
configs:
  - config_name: n17
    data_files:
      - split: train
        path: n17/data/**/*.parquet
  - config_name: n15
    data_files:
      - split: train
        path: n15/data/**/*.parquet
---

# Bigenlight / Theo four-task UR7e corpus — {repo_id.split("/")[-1]}

Original demonstrations: **Bigenlight / Theo**. GR00T conversion and subset packaging:
**RLobot-jun**. This is a derived training-format dataset, not new data collection.

**{ready["episodes"]} unique episodes · {ready["frames"]:,} frames · four tasks · 30 Hz.**
{selection} All selected episodes are exposed as `train`; no held-out evaluation
split or measured policy success rate is claimed. The 30/50/all releases overlap
and must not be treated as disjoint train/test sets.

| Original dataset | Included episodes |
| --- | ---: |
{table}

## Contents and processing

- Episode-level **LeRobot v2.1** files, compatible with the GR00T loaders tested here.
- `n17/` and `n15/` expose the **same episodes**, with separate compatible metadata.
  Each view contains real standalone files, not links to the publisher's server.
  They are NOT different splits. Download only the view you need to avoid storing
  the video payload twice locally.
- Absolute UR7e joint commands: six joint positions in radians + one gripper
  command; state is the corresponding measured seven-dimensional vector.
  Gripper convention: 0=open, 1=closed. This is **not EEF-delta action data**.
- Two RGB cameras: `cam1` scene and `cam2` wrist, 1280×720 at 30 Hz. No depth.
- Action horizon **16** in both supplied modality configurations. Native model
  padding is 40×132 for N1.7 and 16×32 for N1.5; only the real 16×7 actions are used.
- Original task text and float32 state/action values are unchanged. Episode and
  global frame IDs are contiguous in this release; `source_mapping.json` preserves
  original IDs and immutable upstream revisions.
- Aggregate source AV1 videos were frame-accurately re-encoded as per-episode
  H.264, CRF18, at original resolution. **Video compression is lossy.** All clip
  frame counts were verified; sampled RGB alignment was checked against sources.
- Normalization statistics were recomputed on **this selected corpus only**.
  N1.7 metadata includes its statistics fingerprints; N1.5 omits those reserved
  entries. N1.7 uses percentile normalization, N1.5's supplied config uses min/max.
- No demonstrations were removed for collection/recovery tags. No reward, success
  or terminal labels were invented; this release is for **BC**, not labeled offline RL.
- CPU loader/preprocessor verification reports are under each view's
  `VALIDATION.json`. These are data-integrity tests, **not policy evaluation**;
  GPU forward/backward and training on this release were not run for publication.

## Download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{repo_id}", repo_type="dataset",
    local_dir="bigenlight-gr00t",
    allow_patterns=["n17/**", "configs/**", "README.md", "LICENSE", "NOTICE",
                    "source_mapping.json", "release.json", "MANIFEST.json"],
)
```

For N1.5, replace `n17/**` with `n15/**`. For both versions, include both patterns.
Use `bigenlight-gr00t/n17` or `bigenlight-gr00t/n15` as the dataset root.
The Hub viewer configurations likewise select a version, not an additional dataset.

## GR00T N1.7

From an installed Isaac-GR00T N1.7 checkout, use the absolute dataset path ending
in `/n17`, `--embodiment-tag NEW_EMBODIMENT`, and
`--modality-config-path /absolute/path/bigenlight-gr00t/configs/n17_modality.py`.
The supplied config uses **absolute** joint targets for a consistent action meaning
across versions; it does not reuse the older single-carrot relative-joint recipe.

## DEAS GR00T N1.5

From the DEAS-Isaac-GR00T N1.5 environment, load
`configs/n15_data_config.py` and register its `UR7eDataConfig()` instance in
`scripts.gr00t_finetune.DATA_CONFIG_MAP` before invoking `main(ArgsConfig(...))`:

```python
import runpy
from scripts import gr00t_finetune as bc

root = "/absolute/path/bigenlight-gr00t"
Config = runpy.run_path(root + "/configs/n15_data_config.py")["UR7eDataConfig"]
bc.DATA_CONFIG_MAP["ur7e_multitask"] = Config()
# Pass dataset_path=[root + "/n15"], data_config="ur7e_multitask",
# embodiment_tag="new_embodiment" in bc.ArgsConfig when starting BC training.
```

Training hyperparameters, GPU allocation and W&B account/project are deliberately
not embedded in this dataset. Keep the N1.5 and N1.7 Python environments isolated.

## Provenance and license

The four original cards declare Apache-2.0. See `LICENSE`, `NOTICE`, source links
above, and the per-episode mapping. Collection details/recovery annotations remain
in the originals, e.g. `meta/source_takes.json`; those tags are not success labels.
Subset and conversion changes are described above. Pinned source revisions:

{pins}
'''


def stage(variant):
    suffix, count = VARIANTS[variant]
    root = STORAGE / "datasets" / ("bigenlight_multitask_gr00t" + suffix)
    target = STORAGE / "exports" / ("bigenlight_multitask_gr00t_" + variant)
    repo_id = "RLobot-jun/" + target.name
    ready = json.loads((root / "READY.json").read_text())
    mapping = json.loads((root / "source_mapping.json").read_text())
    if ready["episodes"] != count or len(mapping) != count:
        raise ValueError("Unexpected inventory")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {target}")
    target.mkdir(parents=True)
    counts = Counter(row["source_repo"] for row in mapping)
    for view in ("n17", "n15"):
        checked = json.loads((root / view / "VALIDATION.json").read_text())
        if checked["episodes"] != count or checked["frames"] != ready["frames"]:
            raise ValueError("Missing/mismatched loader validation")
        relative_paths = [
            f"meta/{name}"
            for name in (
                "info.json",
                "episodes.jsonl",
                "tasks.jsonl",
                "stats.json",
                "modality.json",
            )
        ]
        relative_paths.append("VALIDATION.json")
        for row in mapping:
            fields = {
                "episode_index": row["episode_index"],
                "episode_chunk": row["episode_index"] // 1000,
            }
            relative_paths.append(DATA_PATH.format(**fields))
            relative_paths.extend(VIDEO_PATH.format(**fields, video_key=key) for key in VIDEO_KEYS)
        for relative in relative_paths:
            source = (root / view / relative).resolve(strict=True)
            if not source.is_relative_to(STORAGE / "datasets"):
                raise ValueError("Payload points outside approved datasets")
            destination = target / view / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)  # HF uploads bytes, not the source link/path.
    atomic_json(target / "source_mapping.json", mapping)
    atomic_json(
        target / "release.json",
        {
            "episodes": count,
            "frames": ready["frames"],
            "tasks": ready["tasks"],
            "selection": "all"
            if variant == "all"
            else f"source episode indices 0..{ready['per_task'] - 1} per task",
            "source_counts": counts,
            "dataset_format": "LeRobot v2.1",
            "video_views": ["n17", "n15"],
            "published_by": "RLobot-jun",
        },
    )
    configs = target / "configs"
    configs.mkdir()
    shutil.copyfile(Path(__file__).with_name("config.py"), configs / "n17_modality.py")
    shutil.copyfile(
        WORKSPACE / "DEAS-Isaac-GR00T/experiments/bigenlight_multitask/config.py",
        configs / "n15_data_config.py",
    )
    shutil.copyfile(WORKSPACE / "DEAS-Isaac-GR00T/LICENSE", target / "LICENSE")
    (target / "NOTICE").write_text(
        "Original robot demonstrations: Bigenlight / Theo.\n"
        "Conversion and subset packaging: RLobot-jun.\n"
        "Derived from the four Apache-2.0 datasets linked in README.md.\n"
        "Changes: LeRobot v3 to v2.1; video re-encoding; ID remapping; statistics; optional subset selection.\n"
    )
    (target / "README.md").write_text(card(repo_id, ready, counts, mapping))
    manifest = {}
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("Export must contain regular files only")
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(4 * 1024**2):
                sha.update(chunk)
        manifest[str(path.relative_to(target))] = {
            "bytes": path.stat().st_size,
            "sha256": sha.hexdigest(),
        }
        if path.suffix in (".json", ".jsonl", ".py", ".md"):
            text = path.read_text()
            if "/home/yoon" in text or "/raid/yoon" in text or "hf_" + "token" in text.lower():
                raise ValueError(f"Private path or credential reference in export: {path.name}")
    atomic_json(target / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "staged": repo_id,
                "files": len(manifest) + 1,
                "bytes": sum(row["bytes"] for row in manifest.values()),
            }
        ),
        flush=True,
    )
    return target


def publish(variant):
    target = STORAGE / "exports" / ("bigenlight_multitask_gr00t_" + variant)
    repo_id = "RLobot-jun/" + target.name
    api = HfApi()
    if api.whoami()["name"] != "RLobot-jun":
        raise ValueError("Unexpected active publishing account")
    try:
        api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        pass
    else:
        raise ValueError(f"Refusing to overwrite an existing repository: {repo_id}")
    manifest = json.loads((target / "MANIFEST.json").read_text())
    allow = [*manifest, "MANIFEST.json"]
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=False)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=target,
        allow_patterns=allow,
        commit_message=f"Publish verified {variant} GR00T N1.7/N1.5 corpus",
    )
    # Public, unauthenticated read verifies visibility and committed file sizes.
    remote = HfApi(token=False).dataset_info(repo_id, revision=commit.oid, files_metadata=True)
    if remote.private or remote.gated:
        raise ValueError("Repository is not publicly downloadable")
    inventory = {file.rfilename: file for file in remote.siblings}
    for name in allow:
        file = inventory[name]
        if file.size != (target / name).stat().st_size:
            raise ValueError(f"Remote size mismatch: {name}")
        if file.lfs and name in manifest and file.lfs.sha256 != manifest[name]["sha256"]:
            raise ValueError(f"Remote SHA256 mismatch: {name}")
    result = {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "revision": commit.oid,
        "public_anonymous_access": True,
        "verified_files": len(allow),
    }
    atomic_json(target.parent / (target.name + "_PUBLISHED.json"), result)
    print("PUBLISHED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Upload the prepared export publicly"
    )
    args = parser.parse_args()
    if args.execute:
        publish(args.variant)
    else:
        stage(args.variant)
