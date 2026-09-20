"""Reward-independent, episode-safe frozen-BC feature cache for scalar critics.

This stores pooled conditioning, NOT the token sequence needed by a DiT actor.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .dataset import AllSuccessStepCostAnnotations, AllSuccessTerminalAnnotations
from .types import OfflineRLBatch


FORMAT = "gr00t-frozen-bc-features-v1"


def identities_match(left, right):
    """Permit a code-checkout relocation, but never different encoder contents.

    Dataset/model paths, config hashes and weight stats remain exact. Old manifests
    keep their original absolute source keys and hashes so existing IQL checkpoints
    remain valid without rewriting caches.
    """
    if left == right:
        return True
    if {k: v for k, v in left.items() if k != "encoder_sha256"} != {
        k: v for k, v in right.items() if k != "encoder_sha256"
    }:
        return False
    known = {"rl/adapters.py", "model/gr00t_n1d7/processing_gr00t_n1d7.py"}

    def encoder_hashes(identity):
        result = {}
        for path, digest in identity.get("encoder_sha256", {}).items():
            relative = path.rsplit("/gr00t/", 1)[-1]
            if relative not in known or relative in result:
                return None
            result[relative] = digest
        return result if result.keys() == known else None

    first, second = encoder_hashes(left), encoder_hashes(right)
    return first is not None and first == second


def atomic_json(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".cache-json-")
    try:
        with os.fdopen(fd, "w") as output:
            json.dump(data, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cache_identity(model_path, dataset_path, horizon, embodiment):
    """Small config content hashes + immutable weight file stat fingerprints.

    Weight files are not reread/hashed in full; model weights must not be modified
    in place while extraction is running. Source paths identify the exact BC.
    """
    model, dataset = Path(model_path).resolve(), Path(dataset_path).resolve()
    files = [
        *model.glob("*.json"),
        *model.glob("processor/*.json"),
        *model.glob("processor/embodiment_mapping.json"),
        *dataset.glob("meta/*.json"),
        *dataset.glob("meta/*.jsonl"),
    ]
    configs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    weights = {
        str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted(model.glob("*.safetensors"))
    }
    if not weights:
        raise ValueError("Expected a local safetensors BC checkpoint")
    source = Path(__file__).resolve().parents[1]
    encoder_files = [
        source / "rl/adapters.py",
        source / "model/gr00t_n1d7/processing_gr00t_n1d7.py",
    ]
    return {
        "format": FORMAT,
        "model_path": str(model),
        "dataset_path": str(dataset),
        "horizon": horizon,
        "embodiment": embodiment,
        "configs_sha256": configs,
        "weight_file_stats": weights,
        "encoder_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in encoder_files
        },
    }


def write_episode(path, features, actions, expected):
    """Atomic exclusive publication; partial temporary files are never consumed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    features, actions = np.asarray(features, np.float32), np.asarray(actions, np.float32)
    if features.shape != tuple(expected["features"]) or actions.shape != tuple(expected["actions"]):
        raise ValueError("Cache episode shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite cached features/actions")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".episode-")
    try:
        with os.fdopen(fd, "wb") as output:
            np.savez(output, features=features, actions=actions)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_episode(root, manifest, episode):
    filename = episode["file"]
    if Path(filename).name != filename:
        raise ValueError("Episode cache filename must not escape cache root")
    with np.load(Path(root) / filename, allow_pickle=False) as data:
        features, actions = data["features"], data["actions"]
    if features.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError("Feature cache must use float32 storage")
    if features.shape != (episode["rows"], manifest["feature_dim"]):
        raise ValueError("Cached feature shape mismatch")
    if actions.shape != (episode["valid_starts"], len(manifest["action_indices"])):
        raise ValueError("Cached action shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite cache data")
    return features, actions


class CachedFeatureDataset:
    """Keep compact features/actions in RAM; no images, processor, VLM or workers.

    Reward annotations are rebuilt independently on each training run. Currently
    supports explicitly all-success datasets; no missing terminal successor can
    contribute to a TD target. Padding/horizon match the original live encoder.
    """

    def __init__(self, root, *, reward="terminal-success", gamma=0.99):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        complete = json.loads((self.root / "COMPLETE.json").read_text())
        if self.manifest["identity"]["format"] != FORMAT or complete.get("format") != FORMAT:
            raise ValueError("Unsupported/incomplete feature cache")
        digest = hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest()
        if complete.get("manifest_sha256") != digest:
            raise ValueError("Feature cache completion/manifest mismatch")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0,1]")
        annotations = {
            "terminal-success": AllSuccessTerminalAnnotations,
            "step-cost": AllSuccessStepCostAnnotations,
        }
        if reward not in annotations:
            raise ValueError("Explicit reward preset required")
        self.gamma = gamma
        self.horizon = self.manifest["identity"]["horizon"]
        self.feature_dim = self.manifest["feature_dim"]
        self.action_shape = tuple(self.manifest["action_shape"])
        self.action_indices = np.asarray(self.manifest["action_indices"], np.int64)
        if len(self.action_shape) != 2 or self.horizon < 1 or self.horizon > self.action_shape[0]:
            raise ValueError("Invalid cached horizon/shape")
        mask = np.zeros(np.prod(self.action_shape), np.float32)
        if (
            (self.action_indices < 0).any()
            or (self.action_indices >= len(mask)).any()
            or len(np.unique(self.action_indices)) != len(self.action_indices)
        ):
            raise ValueError("Invalid cached action mask indices")
        mask[self.action_indices] = 1
        self.mask = mask.reshape(self.action_shape)
        if not np.array_equal(self.mask.any(-1), np.arange(self.action_shape[0]) < self.horizon):
            raise ValueError("Cached action mask must have exactly H valid prefix timesteps")
        self.episodes = self.manifest["episodes"]
        self.episode_sizes = [max(0, e["rows"] - self.horizon + 1) for e in self.episodes]
        if self.episode_sizes != [e["valid_starts"] for e in self.episodes]:
            raise ValueError("Cached valid-start counts do not match episode lengths")
        self.offsets = np.concatenate(([0], np.cumsum(self.episode_sizes)))
        self.arrays = [read_episode(self.root, self.manifest, e) for e in self.episodes]
        self.annotations = [
            annotations[reward]()(self, i).validated(e["rows"]) for i, e in enumerate(self.episodes)
        ]

    def get_episode_length(self, index):
        return self.episodes[index]["rows"]

    def __len__(self):
        return int(self.offsets[-1])

    def batch(self, indices):
        indices = np.asarray(indices, np.int64)
        if (
            indices.ndim != 1
            or not len(indices)
            or (indices < 0).any()
            or (indices >= len(self)).any()
        ):
            raise IndexError("Invalid cached transition indices")
        episodes = np.searchsorted(self.offsets, indices, side="right") - 1
        current, following, rewards, terminated = [], [], [], []
        actions = np.zeros((len(indices), np.prod(self.action_shape)), np.float32)
        for row, (index, episode) in enumerate(zip(indices, episodes)):
            start = int(index - self.offsets[episode])
            end = start + self.horizon
            features, packed_actions = self.arrays[episode]
            labels = self.annotations[episode]
            terminal = bool(labels.terminated[end - 1])
            if end >= len(features) and not terminal:
                raise ValueError("Missing nonterminal cached successor")
            current.append(features[start])
            following.append(features[min(end, len(features) - 1)])
            actions[row, self.action_indices] = packed_actions[start]
            rewards.append(np.dot(self.gamma ** np.arange(self.horizon), labels.rewards[start:end]))
            terminated.append(terminal)
        terminal = torch.tensor(terminated, dtype=torch.bool)
        return OfflineRLBatch(
            {"features": torch.from_numpy(np.stack(current))},
            {"features": torch.from_numpy(np.stack(following))},
            torch.from_numpy(actions.reshape(len(indices), *self.action_shape)),
            torch.from_numpy(np.broadcast_to(self.mask, (len(indices), *self.action_shape)).copy()),
            torch.tensor(rewards, dtype=torch.float32),
            (~terminal).float() * self.gamma**self.horizon,
            terminal,
            torch.zeros_like(terminal),
            torch.full((len(indices),), self.horizon, dtype=torch.long),
        )
