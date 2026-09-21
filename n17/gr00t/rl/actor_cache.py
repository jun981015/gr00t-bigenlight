"""Disk cache of frozen BC projected tokens/state for DiT LoRA SVF."""

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .feature_cache import CachedFeatureDataset, atomic_json
from .projected_actor import FrozenProjectedEncoder, project_observation


FORMAT = "gr00t-projected-actor-v1"
FIELDS = (
    "projected_vl_features",
    "projected_state_features",
    "backbone_attention_mask",
    "image_mask",
    "embodiment_id",
)


def projection_signature():
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    functions = [
        project_observation,
        FrozenProjectedEncoder,
        Gr00tN1d7ActionHead._encode_features,
        Gr00tN1d7ActionHead.process_backbone_output,
    ]
    return hashlib.sha256("\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def manifest_for(base_cache):
    base = base_cache.manifest
    return {
        "format": FORMAT,
        "pooled_manifest_sha256": hashlib.sha256(
            (base_cache.root / "manifest.json").read_bytes()
        ).hexdigest(),
        "identity": base["identity"],
        "projection_sha256": projection_signature(),
        "episodes": [
            {"file": f"episode-{i:06d}.safetensors", "rows": ep["rows"]}
            for i, ep in enumerate(base["episodes"])
        ],
    }


def write_projected_episode(path, tensors, rows):
    path = Path(path)
    payload = {key: tensors[key].detach().cpu().contiguous() for key in FIELDS}
    if any(value.shape[0] != rows for value in payload.values()):
        raise ValueError("Projected episode row count mismatch")
    if any(
        value.is_floating_point() and not torch.isfinite(value).all() for value in payload.values()
    ):
        raise ValueError("Nonfinite projected features")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".projected-")
    os.close(fd)
    try:
        save_file(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)  # exclusive atomic publication
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_episode(path, rows):
    with safe_open(path, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(FIELDS):
            raise ValueError("Projected cache fields differ")
        shapes = {key: source.get_slice(key).get_shape() for key in FIELDS}
        if any(shape[0] != rows for shape in shapes.values()):
            raise ValueError("Projected cache row count differs")
        tokens, state = shapes["projected_vl_features"], shapes["projected_state_features"]
        if len(tokens) != 3 or len(state) != 3 or state[1] != 1:
            raise ValueError("Invalid projected token/state shape")
        if shapes["embodiment_id"] != [rows] or any(
            shapes[key] != tokens[:2] for key in ("image_mask", "backbone_attention_mask")
        ):
            raise ValueError("Projected mask/embodiment shape mismatch")


def mark_complete(root, manifest):
    for ep in manifest["episodes"]:
        validate_episode(root / ep["file"], ep["rows"])
    atomic_json(
        root / "COMPLETE.json",
        {
            "format": FORMAT,
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        },
    )


class ProjectedFeatureDataset:
    def __init__(self, pooled_cache, actor_cache, *, reward="step-cost", gamma=0.99):
        self.base = CachedFeatureDataset(pooled_cache, reward=reward, gamma=gamma)
        self.root = Path(actor_cache)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        complete = json.loads((self.root / "COMPLETE.json").read_text())
        if self.manifest != manifest_for(self.base) or complete != {
            "format": FORMAT,
            "manifest_sha256": hashlib.sha256(
                (self.root / "manifest.json").read_bytes()
            ).hexdigest(),
        }:
            raise ValueError("Incomplete or mismatched projected cache / BC / projection")
        for ep in self.manifest["episodes"]:
            validate_episode(self.root / ep["file"], ep["rows"])
        self.episode_sizes = self.base.episode_sizes
        self.offsets = self.base.offsets

    def __len__(self):
        return len(self.base)

    def batch(self, indices):
        base = self.base.batch(indices)
        episode_ids = np.searchsorted(self.offsets, indices, side="right") - 1
        if not (episode_ids == episode_ids[0]).all():
            raise ValueError(
                "Use EpisodeBatchSampler: projected tokens require one episode per batch"
            )
        episode = int(episode_ids[0])
        starts = np.asarray(indices) - self.offsets[episode]
        spec = self.manifest["episodes"][episode]
        following = np.minimum(starts + self.base.horizon, spec["rows"] - 1)
        with safe_open(self.root / spec["file"], framework="pt", device="cpu") as source:

            def gather(rows):
                return {
                    key: torch.stack([source.get_slice(key)[int(i)] for i in rows])
                    for key in FIELDS
                }

            current, next_obs = gather(starts), gather(following)
        if any(
            v.is_floating_point() and not torch.isfinite(v).all()
            for obs in (current, next_obs)
            for v in obs.values()
        ):
            raise ValueError("Nonfinite projected cache sample")
        # Use original cached critic vectors exactly, including their BF16 rounding.
        current.update(base.observations)
        next_obs.update(base.next_observations)
        return replace(base, observations=current, next_observations=next_obs)


def load_action_head(model_path, device="cpu"):
    """Load only action-head tensors from BC safetensors; never construct a VLM."""
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    root = Path(model_path)
    head = Gr00tN1d7ActionHead(Gr00tN1d7Config.from_pretrained(root)).to(dtype=torch.bfloat16)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key.startswith("action_head."):
                    name = key.removeprefix("action_head.")
                    if name in state:
                        raise ValueError("Duplicate action-head tensor")
                    state[name] = source.get_tensor(key)
    head.load_state_dict(state, strict=True)
    return head.to(device).requires_grad_(False).eval()
