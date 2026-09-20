"""Extract frozen N1.7 BC conditioning episode by episode, with resumable files."""

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .feature_cache import (
    FORMAT,
    atomic_json,
    cache_identity,
    identities_match,
    read_episode,
    write_episode,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--sleep-ms", type=float, default=25)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Bounded startup check; 0 processes all episodes",
    )
    args = parser.parse_args(argv)
    if (
        min(args.horizon, args.batch_size, args.cpu_threads) < 1
        or args.sleep_ms < 0
        or args.max_episodes < 0
    ):
        raise ValueError("Invalid extraction settings")
    identity = cache_identity(args.model_path, args.dataset_path, args.horizon, args.embodiment_tag)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / "extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and not identities_match(manifest["identity"], identity):
            raise ValueError("Cache identity differs; use a new directory")
        if manifest is not None:
            identity = manifest["identity"]
        if (root / "COMPLETE.json").exists():
            if (
                manifest is None
                or json.loads((root / "COMPLETE.json").read_text())["manifest_sha256"]
                != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ):
                raise ValueError("Invalid complete cache")
            print("Cache already complete; no GPU model loaded", flush=True)
            return
        torch.set_num_threads(args.cpu_threads)
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        from .adapters import FrozenBCGR00TEncoder

        tag = EmbodimentTag.resolve(args.embodiment_tag)
        policy = Gr00tPolicy(tag, str(args.model_path), device=args.device)
        encoder = FrozenBCGR00TEncoder(policy.model)
        modalities = dict(policy.modality_configs)
        if args.horizon > len(modalities["action"].delta_indices):
            raise ValueError("Cache horizon exceeds trained robot horizon")
        modalities["action"] = replace(
            modalities["action"], delta_indices=list(range(args.horizon))
        )
        loader = LeRobotEpisodeLoader(
            args.dataset_path, modalities, decoder_kwargs={"num_ffmpeg_threads": 2}
        )
        processor = policy.processor
        collator = processor.collator

        def process(frame, index):
            # End-of-episode action padding is never saved as a valid transition;
            # those final rows are needed only as successor observation features.
            step = extract_step_data(frame, index, modalities, tag, allow_padding=True)
            return processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])

        started = time.monotonic()
        completed = 0
        resumed = 0
        for episode in range(len(loader)):
            if manifest is not None:
                spec = manifest["episodes"][episode]
                if (root / spec["file"]).exists():
                    read_episode(root, manifest, spec)
                    completed += 1
                    resumed += 1
                    continue
            frame = loader[episode]
            rows = len(frame)
            if rows != loader.get_episode_length(episode):
                raise ValueError("Decoded episode length differs from metadata")
            valid = max(0, rows - args.horizon + 1)
            features, actions = [], []
            for start in range(0, rows, args.batch_size):
                items = [
                    process(frame, i) for i in range(start, min(rows, start + args.batch_size))
                ]
                inputs = collator(items)["inputs"]
                encoded = encoder.encode_observation(inputs)["features"].float().cpu().numpy()
                mask = inputs["action_mask"].float().cpu().numpy()
                if not np.isin(mask, [0, 1]).all() or not (mask == mask[0]).all():
                    raise ValueError("Feature cache requires a fixed action padding mask")
                indices = np.flatnonzero(mask[0].reshape(-1)).tolist()
                if manifest is None:
                    manifest = {
                        "identity": identity,
                        "feature_dim": encoded.shape[-1],
                        "action_shape": list(mask.shape[1:]),
                        "action_indices": indices,
                        "extraction": {
                            "compute_dtype": "bfloat16",
                            "storage_dtype": "float32",
                            "batch_size": args.batch_size,
                            "batching": "single-episode only; no mixed-language padding",
                            "encoder": "VLM -> frozen BC LN/self-attention once -> masked mean + state + embodiment",
                            "reward_cached": False,
                        },
                        "episodes": [
                            {
                                "episode_index": m["episode_index"],
                                "rows": loader.get_episode_length(i),
                                "valid_starts": max(
                                    0, loader.get_episode_length(i) - args.horizon + 1
                                ),
                                "file": f"episode-{i:06d}.npz",
                            }
                            for i, m in enumerate(loader.episodes_metadata)
                        ],
                    }
                    atomic_json(manifest_path, manifest)
                if (
                    indices != manifest["action_indices"]
                    or list(mask.shape[1:]) != manifest["action_shape"]
                    or encoded.shape[-1] != manifest["feature_dim"]
                ):
                    raise ValueError("Cache dimensions/mask changed within extraction")
                features.append(encoded)
                count = max(0, min(len(items), valid - start))
                if count:
                    actions.append(
                        inputs["action"][:count]
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(count, -1)[:, indices]
                    )
                if args.sleep_ms:
                    time.sleep(args.sleep_ms / 1000)
            spec = manifest["episodes"][episode]
            feature_array = np.concatenate(features)
            action_array = (
                np.concatenate(actions)
                if actions
                else np.empty((0, len(manifest["action_indices"])), np.float32)
            )
            write_episode(
                root / spec["file"],
                feature_array,
                action_array,
                {
                    "features": (rows, manifest["feature_dim"]),
                    "actions": (valid, len(manifest["action_indices"])),
                },
            )
            # First episode: exact disk round-trip check (no extra VLM passes).
            if episode == 0:
                saved_features, saved_actions = read_episode(root, manifest, spec)
                np.testing.assert_array_equal(saved_features, feature_array)
                np.testing.assert_array_equal(saved_actions, action_array)
            completed += 1
            progress = {
                "completed_episodes": completed,
                "total_episodes": len(loader),
                "last_episode_index": spec["episode_index"],
                "elapsed_s": time.monotonic() - started,
                "resumed_episodes": resumed,
                "batch_size": args.batch_size,
                "cpu_threads": args.cpu_threads,
                "sleep_ms": args.sleep_ms,
            }
            atomic_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            del frame, features, actions, feature_array, action_array, items, inputs, encoded
            if args.max_episodes and completed >= args.max_episodes:
                break
        if completed == len(loader):
            if not identities_match(
                cache_identity(
                    args.model_path, args.dataset_path, args.horizon, args.embodiment_tag
                ),
                identity,
            ):
                raise ValueError(
                    "Source/checkpoint changed during extraction; cache not marked complete"
                )
            atomic_json(
                root / "COMPLETE.json",
                {
                    "format": FORMAT,
                    "episodes": completed,
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                },
            )
            print("Feature cache complete", flush=True)


if __name__ == "__main__":
    main()
