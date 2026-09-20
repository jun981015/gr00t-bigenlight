"""SVF policy improvement against a frozen, cached-feature IQL environment Q.

Q conditioning is a frozen copy of BC LN/self-attention. Raw VLM tokens are kept
separately for actor/reference; training an actor cannot change Q's state space.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from .adapters import FrozenGR00TEncoder
from .algorithms import SoftValueFlow
from .feature_cache import FORMAT, identities_match
from .networks import FeatureCritic


class FrozenQConditioningEncoder(FrozenGR00TEncoder):
    """Snapshot BEFORE making the actor trainable/FP32; keep BC compute dtype."""

    def __init__(self, model):
        super().__init__(model)
        self.q_ln = deepcopy(model.action_head.vlln).requires_grad_(False).eval()
        self.q_attention = (
            deepcopy(model.action_head.vl_self_attention).requires_grad_(False).eval()
        )

    def pooling_features(self, raw_features):
        self.q_ln.eval()
        self.q_attention.eval()
        parameter = next(self.q_ln.parameters(), None)
        dtype = parameter.dtype if parameter is not None else raw_features.dtype
        return self.q_attention(self.q_ln(raw_features.to(dtype))).float()


class FrozenQSoftValueFlow(SoftValueFlow):
    """No env-Q TD, optimizer entries, next-action rollout, or env-Q EMA."""

    def __init__(self, actor, critic, inner_critic, config, reference=None):
        if not config.freeze_reference:
            raise ValueError("Frozen-IQL-Q mode requires a frozen BC reference")
        critic.requires_grad_(False).eval()
        critic.zero_grad(set_to_none=True)
        super().__init__(actor, critic, inner_critic, config, reference)

    def outer_objective(self, batch, actions):
        self.critic.eval()
        # No bootstrap computation: Q is an immutable scoring function.
        return actions.new_zeros(()), actions.new_zeros(actions.shape[0])

    @torch.no_grad()
    def _update_targets(self):
        # Maintain the existing inner target for checkpoint compatibility, but
        # never touch the imported environment Q or its frozen scoring copy.
        for parameter, target in zip(
            self.inner_critic.parameters(), self.target_inner_critic.parameters(), strict=True
        ):
            target.lerp_(parameter, self.config.tau)
        for buffer, target in zip(
            self.inner_critic.buffers(), self.target_inner_critic.buffers(), strict=True
        ):
            target.copy_(buffer)

    def losses(self, batch):
        loss, metrics = super().losses(batch)
        metrics.pop("critic/td_target_mean")
        with torch.no_grad():
            q = self.critic(batch.observations, batch.actions * batch.action_mask)
        metrics.update(
            {
                "critic/env_q_frozen": loss.new_ones(()),
                "q/min": q.min(),
                "q/mean": q.mean(),
                "q/max": q.max(),
            }
        )
        return loss, metrics


def load_frozen_iql_q(
    checkpoint, cache, *, identity, feature_dim, action_mask, gamma, reward, device="cpu"
):
    """Import ONLINE twin Q, not IQL V/target/optimizer, from a trusted full state.

    Require exact cache provenance, BC/normalization/dataset identity, action mask,
    reward and discount. Legacy live-VLM checkpoints are intentionally rejected.
    """
    checkpoint, cache = Path(checkpoint), Path(cache)
    manifest_bytes = (cache / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    complete = json.loads((cache / "COMPLETE.json").read_text())
    if complete.get("format") != FORMAT or complete.get("manifest_sha256") != digest:
        raise ValueError("Require a completed, consistent feature cache")
    if not identities_match(manifest["identity"], identity):
        raise ValueError("BC/dataset/normalization/horizon identity differs from IQL cache")
    if action_mask.ndim != 3 or not torch.isin(action_mask, action_mask.new_tensor([0, 1])).all():
        raise ValueError("Expected binary batched action mask")
    if not torch.equal(action_mask, action_mask[:1].expand_as(action_mask)):
        raise ValueError("Q requires the same mask for every sample")
    indices = action_mask[0].flatten().nonzero().flatten().tolist()
    if (
        list(action_mask.shape[1:]) != manifest["action_shape"]
        or indices != manifest["action_indices"]
        or feature_dim != manifest["feature_dim"]
    ):
        raise ValueError("IQL feature dimension or action horizon/padding mask differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    algorithm = state["algorithm"]
    if (
        state.get("format_version") != 1
        or algorithm.get("algorithm") != "iql-critic-only-v1"
        or state.get("step", 0) < 1
        or state["step"] != algorithm["updates"]
        or metadata.get("backend") != "frozen-bc-cache-v1"
        or metadata.get("cache_manifest_sha256") != digest
        or metadata.get("bc_identity") != manifest["identity"]
    ):
        raise ValueError("Require a matching cached-IQL full training checkpoint")
    settings = metadata["args"]
    if settings["gamma"] != gamma or settings["reward"] != reward:
        raise ValueError("IQL reward/discount mismatch")
    hidden = (settings["hidden_dim"],) * settings["hidden_layers"]
    critic = FeatureCritic(feature_dim, tuple(action_mask.shape[1:]), hidden)
    critic.load_state_dict(algorithm["critic"], strict=True)
    if not all(torch.isfinite(p).all() for p in critic.parameters()):
        raise ValueError("Nonfinite IQL Q weights")
    with checkpoint.open("rb") as stream:
        checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "iql_step": state["step"],
        "cache_manifest_sha256": digest,
        "source": "online IQL Q1/Q2; no IQL V or optimizer",
        "metadata": metadata,
    }
    return critic.to(device).requires_grad_(False).eval(), provenance
