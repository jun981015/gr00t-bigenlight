"""Build resumable projected-token cache for an existing completed pooled cache."""

import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import time

import torch

from .actor_cache import (
    FIELDS,
    manifest_for,
    mark_complete,
    validate_episode,
    write_projected_episode,
)
from .feature_cache import CachedFeatureDataset, atomic_json, cache_identity, identities_match
from .projected_actor import FrozenProjectedEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooled-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.cpu_threads) < 1 or args.max_episodes < 0:
        raise ValueError("Invalid extraction settings")
    base = CachedFeatureDataset(args.pooled_cache)
    identity = base.manifest["identity"]

    def current_identity():
        return cache_identity(
            identity["model_path"],
            identity["dataset_path"],
            identity["horizon"],
            identity["embodiment"],
        )

    if not identities_match(identity, current_identity()):
        raise ValueError("BC/data changed since pooled cache extraction")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / "extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = manifest_for(base)
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Projected cache identity changed")
        if not path.exists():
            atomic_json(path, manifest)
        if (root / "COMPLETE.json").exists():
            # Validate completion without loading the VLM.
            from .actor_cache import ProjectedFeatureDataset

            ProjectedFeatureDataset(args.pooled_cache, root)
            print("Projected cache already complete", flush=True)
            return
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        torch.set_num_threads(args.cpu_threads)
        tag = EmbodimentTag.resolve(identity["embodiment"])
        policy = Gr00tPolicy(tag, identity["model_path"], device=args.device)
        policy.model.requires_grad_(False).eval()
        encoder = FrozenProjectedEncoder(policy.model)
        modalities = dict(policy.modality_configs)
        modalities["action"] = replace(
            modalities["action"], delta_indices=list(range(identity["horizon"]))
        )
        loader = LeRobotEpisodeLoader(
            identity["dataset_path"], modalities, decoder_kwargs={"num_ffmpeg_threads": 2}
        )
        started, newly_saved, completed = time.monotonic(), 0, 0
        for episode, spec in enumerate(manifest["episodes"]):
            target = root / spec["file"]
            if target.exists():
                validate_episode(target, spec["rows"])
                completed += 1
                continue
            frame = loader[episode]
            if len(frame) != spec["rows"]:
                raise ValueError("Episode rows changed")
            pieces = {key: [] for key in FIELDS}
            for start in range(0, len(frame), args.batch_size):
                items = [
                    policy.processor(
                        [
                            {
                                "type": MessageType.EPISODE_STEP.value,
                                "content": extract_step_data(
                                    frame, i, modalities, tag, allow_padding=True
                                ),
                            }
                        ]
                    )
                    for i in range(start, min(start + args.batch_size, len(frame)))
                ]
                inputs = policy.processor.collator(items)["inputs"]
                obs = encoder.encode_observation(inputs)
                for key in FIELDS:
                    pieces[key].append(obs[key].cpu())
            write_projected_episode(
                target, {k: torch.cat(v) for k, v in pieces.items()}, spec["rows"]
            )
            completed += 1
            newly_saved += 1
            progress = {
                "completed_episodes": completed,
                "total_episodes": len(loader),
                "elapsed_s": time.monotonic() - started,
                "batch_size": args.batch_size,
            }
            atomic_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            del frame, pieces, obs, inputs, items
            if args.max_episodes and newly_saved >= args.max_episodes:
                break
        if completed == len(manifest["episodes"]):
            if not identities_match(identity, current_identity()):
                raise ValueError("BC/data changed during extraction")
            mark_complete(root, manifest)


if __name__ == "__main__":
    main()
