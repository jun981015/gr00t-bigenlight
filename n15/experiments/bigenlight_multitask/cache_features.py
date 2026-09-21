"""Resumable N1.5 BC critic cache, compatible with n17 CachedFeatureDataset.

Decode each camera once per episode and infer batches of observations. Cache
the frozen BC LN/self-attention pooled output before the new critic projection.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.bigenlight_multitask.config import UR7eDataConfig
from experiments.robocasa_deas.recovery import atomic_json
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import DatasetMetadata
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import GR00TTransform
from gr00t.utils.pyav_frames import _full_frames


FORMAT = "gr00t-frozen-bc-features-v1"


class EpisodeDecodedDataset(LeRobotSingleDataset):
    """Hold just one episode's two timestamp-aligned cameras in CPU RAM."""

    def get_video(self, trajectory_id, modality, key, base_index):
        if getattr(self, "decoded_id", None) != trajectory_id:
            self.decoded_id, self.decoded = trajectory_id, {}
        if key not in self.decoded:
            path = self.get_video_path(trajectory_id, key.removeprefix(modality + "."))
            timestamps = self.curr_traj_data["timestamp"].to_numpy()
            self.decoded[key] = np.stack(_full_frames(path, timestamps))
        frames = self.decoded[key]
        indices = np.clip(self.delta_indices[key] + base_index, 0, len(frames) - 1)
        return frames[indices].copy()


def identity(model, dataset):
    config_files = [
        model / "config.json",
        model / "experiment_cfg/metadata.json",
        *sorted((dataset / "meta").glob("*.json")),
        *sorted((dataset / "meta").glob("*.jsonl")),
    ]
    sources = [Path(__file__), Path(__file__).with_name("config.py"), Path(__file__).with_name("cache_shards.py")]
    gr00t_root = Path(__file__).resolve().parents[2] / "gr00t"
    sources += [
        gr00t_root / name
        for name in (
            "model/gr00t_n1.py",
            "model/transforms.py",
            "data/dataset.py",
            "model/action_head/flow_matching_action_head.py",
            "utils/pyav_frames.py",
            "experiment/data_config.py",
        )
    ]
    weights = list(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("Expected local BC safetensors weights")
    return {
        "format": FORMAT,
        "model_version": "n15",
        "model_path": str(model),
        "dataset_path": str(dataset),
        "horizon": 16,
        "embodiment": "new_embodiment",
        "configs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in config_files},
        "encoder_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "weight_file_stats": {
            str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(weights)
        },
    }


def validate_episode(path, spec, manifest):
    with np.load(path, allow_pickle=False) as data:
        features, actions = data["features"], data["actions"]
        if features.shape != (spec["rows"], manifest["feature_dim"]) or actions.shape != (
            spec["valid_starts"],
            len(manifest["action_indices"]),
        ):
            raise ValueError(f"Invalid cache shape: {path}")
        if not np.isfinite(features).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite cache: {path}")


def make_dataset(model_path, dataset_path):
    config = UR7eDataConfig()
    transforms = config.transform()
    dataset = EpisodeDecodedDataset(
        dataset_path,
        config.modality_config(),
        "new_embodiment",
        video_backend="torchvision_av",
        transforms=transforms,
    )
    # Normalize using the BC's metadata, never recompute stats on a subset.
    metadata = json.loads((model_path / "experiment_cfg/metadata.json").read_text())["new_embodiment"]
    transforms.set_metadata(DatasetMetadata.model_validate(metadata))
    transforms.eval()
    # Only the final packing transform needs training=True to retain actions.
    # Cropping/jitter/noise remain in eval mode; language dropout is disabled.
    packing = [t for t in transforms.transforms if isinstance(t, GR00TTransform)]
    if len(packing) != 1 or packing[0].action_horizon != 16:
        raise ValueError("Expected one horizon-16 GR00T packing transform")
    packing[0].training = True
    packing[0].language_dropout_prob = 0
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-shards", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.prefetch_shards) < 1 or args.max_episodes < 0:
        raise ValueError("Invalid batch/episode limit")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model_path, dataset_path = args.model_path.resolve(), args.dataset_path.resolve()
    cache_identity = identity(model_path, dataset_path)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / "extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest["identity"] != cache_identity:
            # Approved loader-only migration: unchanged transforms/model/normalization.
            previous = json.loads(json.dumps(manifest["identity"]))
            source_key = str(Path(__file__))
            if (
                previous["encoder_sha256"].get(source_key)
                != "8e118877f89aaeb726010b6c745de813024b0d41669b54c1a20370084ccf401c"
            ):
                raise ValueError("Unknown cache encoder revision")
            previous["encoder_sha256"][source_key] = cache_identity["encoder_sha256"][source_key]
            worker_key = str(Path(__file__).with_name("cache_shards.py"))
            previous["encoder_sha256"][worker_key] = cache_identity["encoder_sha256"][worker_key]
            if previous != cache_identity:
                raise ValueError("Cache identity mismatch: use a new output directory")
        atomic_json(
            root / "loader_runtime.json",
            {
                "workers": args.workers,
                "prefetch_shards": args.prefetch_shards,
                "batch_size": args.batch_size,
                "encoder_sha256": cache_identity["encoder_sha256"],
                "loader": "spawned-CPU-episode-shards-v1",
            },
        )
        if (root / "COMPLETE.json").exists():
            complete = json.loads((root / "COMPLETE.json").read_text())
            if complete["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
                raise ValueError("Completion manifest mismatch")
            print("Cache already complete", flush=True)
            return
        dataset = make_dataset(model_path, dataset_path)
        model = GR00T_N1_5.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
        model.requires_grad_(False).eval().to(args.device)
        if model.action_horizon != 16:
            raise ValueError("BC horizon differs from requested cache")
        from experiments.bigenlight_multitask.cache_shards import episode_shards

        started = time.monotonic()
        completed = 0
        jobs = []
        for episode, (trajectory, length) in enumerate(zip(dataset.trajectory_ids, dataset.trajectory_lengths)):
            path = root / f"episode-{episode:06d}.npz"
            if path.exists():
                if manifest is None:
                    raise ValueError("Episode exists without manifest")
                validate_episode(path, manifest["episodes"][episode], manifest)
                completed += 1
            else:
                jobs.append((episode, int(trajectory), int(length), args.batch_size))
        if args.max_episodes:
            jobs = jobs[: args.max_episodes]
        resumed = completed
        previous_end = time.monotonic()
        shards = episode_shards(model_path, dataset_path, jobs, args.workers, args.prefetch_shards)
        for episode, trajectory, length, batches in shards:
            ready = time.monotonic()
            loader_wait = ready - previous_end
            path = root / f"episode-{episode:06d}.npz"
            features, actions = [], []
            valid = max(0, length - 16 + 1)
            for start, inputs in batches:
                mask = inputs["action_mask"].numpy()
                indices = np.flatnonzero(mask[0].reshape(-1)).tolist()
                if not (mask == mask[0]).all():
                    raise ValueError("Action mask changed within batch")
                with torch.inference_mode():
                    backbone_inputs, action_inputs = model.prepare_input(inputs)
                    backbone = model.backbone(backbone_inputs)
                    # Apply frozen BC frontend exactly once, without mutating backbone.
                    tokens = model.action_head.vl_self_attention(
                        model.action_head.vlln(backbone["backbone_features"])
                    )
                    attention_mask = backbone["backbone_attention_mask"].unsqueeze(-1).float()
                    pooled = (tokens.float() * attention_mask).sum(1) / attention_mask.sum(1).clamp_min(1)
                    states = action_inputs["state"].float().flatten(1)
                    onehot = F.one_hot(
                        action_inputs["embodiment_id"].long(), model.action_head.config.max_num_embodiments
                    ).float()
                    encoded = torch.cat((pooled, states, onehot), -1).cpu().numpy()
                if manifest is None:
                    manifest = {
                        "identity": cache_identity,
                        "feature_dim": encoded.shape[-1],
                        "feature_layout": {
                            "vlm_dim": pooled.shape[-1],
                            "state_dim": states.shape[-1],
                            "embodiment_dim": onehot.shape[-1],
                        },
                        "action_shape": list(mask.shape[1:]),
                        "action_indices": indices,
                        "extraction": {
                            "batch_size": args.batch_size,
                            "compute_dtype": "bfloat16",
                            "storage_dtype": "float32",
                            "encoder": "N1.5 VLM -> frozen BC LN/self-attention once -> masked mean + state + embodiment",
                            "video_decode": "once per episode/camera, nearest PTS",
                            "reward_cached": False,
                        },
                        "episodes": [
                            {
                                "episode_index": int(t),
                                "rows": int(n),
                                "valid_starts": max(0, int(n) - 15),
                                "file": f"episode-{i:06d}.npz",
                            }
                            for i, (t, n) in enumerate(zip(dataset.trajectory_ids, dataset.trajectory_lengths))
                        ],
                    }
                    atomic_json(manifest_path, manifest)
                if indices != manifest["action_indices"] or encoded.shape[-1] != manifest["feature_dim"]:
                    raise ValueError("Cache feature/action layout changed")
                features.append(encoded)
                count = max(0, min(len(encoded), valid - start))
                if count:
                    actions.append(inputs["action"][:count].float().numpy().reshape(count, -1)[:, indices])
            feature_array = np.concatenate(features).astype(np.float32)
            action_array = (
                np.concatenate(actions).astype(np.float32) if actions else np.empty((0, len(indices)), np.float32)
            )
            if not np.isfinite(feature_array).all() or not np.isfinite(action_array).all():
                raise ValueError("Nonfinite encoded episode")
            fd, temporary = tempfile.mkstemp(dir=root, prefix=".episode-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    np.savez(handle, features=feature_array, actions=action_array)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            completed += 1
            progress = {
                "completed_episodes": completed,
                "total_episodes": len(dataset.trajectory_ids),
                "elapsed_s": time.monotonic() - started,
                "resumed_episodes": resumed,
                "new_episodes": completed - resumed,
                "loader_wait_s": loader_wait,
                "inference_and_save_s": time.monotonic() - ready,
                "workers": args.workers,
                "prefetch_shards": args.prefetch_shards,
            }
            atomic_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
            previous_end = time.monotonic()
        if completed == len(dataset.trajectory_ids):
            atomic_json(
                root / "COMPLETE.json",
                {
                    "format": FORMAT,
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "episodes": completed,
                },
            )


if __name__ == "__main__":
    main()
