"""SVF policy improvement against frozen cached IQL or DEAS environment Q.

Q conditioning is a frozen copy of BC LN/self-attention. Raw VLM tokens are kept
separately for actor/reference; training an actor cannot change Q's state space.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from .adapters import FrozenGR00TEncoder
from .algorithms import SoftValueFlow, aggregate_heads, soft_value
from .feature_cache import FORMAT, identities_match
from .networks import FeatureCritic


class FrozenDEASQ(torch.nn.Module):
    """Online DEAS projection + twin distribution means, with action gradients.

    No V, target-Q, or optimizer is retained. Frozen parameters still permit
    differentiation with respect to actions when a consumer needs it.
    """

    def __init__(self, state, feature_dim, action_shape, indices):
        super().__init__()
        from .deas_cached import DEASCachedLearner, DEASConfig

        config = DEASConfig(**state["config"])
        if feature_dim != config.vlm_dim + config.state_dim + config.embodiment_dim:
            raise ValueError("DEAS feature layout differs from cache")
        if state["action_indices"].tolist() != indices:
            raise ValueError("DEAS action coordinates differ from cache")
        learner = DEASCachedLearner(indices, config)
        self.projection, self.heads, self.hlg = learner.projection, learner.critic, learner.hlg
        self.projection.load_state_dict(state["projection"], strict=True)
        self.heads.load_state_dict(state["critic"], strict=True)
        self.register_buffer("action_indices", torch.tensor(indices, dtype=torch.long))
        self.vlm_dim, self.state_dim = config.vlm_dim, config.state_dim
        self.feature_dim, self.action_shape = feature_dim, tuple(action_shape)
        self.requires_grad_(False).eval()

    def forward(self, observations, actions):
        features = observations["features"].float()
        if features.shape[-1] != self.feature_dim or tuple(actions.shape[1:]) != self.action_shape:
            raise ValueError("DEAS scoring input layout differs from training")
        projected = self.projection(features[:, : self.vlm_dim])
        state = features[:, self.vlm_dim : self.vlm_dim + self.state_dim]
        packed = actions.flatten(1).float()[:, self.action_indices]
        inputs = torch.cat((projected, state, packed), -1)
        return torch.stack([self.hlg.decode(head(inputs)) for head in self.heads])


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

    def initialize_from_svf(self, state):
        if state["algorithm"] not in ("FrozenQSoftValueFlow", "FrozenQInnerOnly") or state[
            "config"
        ] != asdict(self.config):
            raise ValueError("Require matching fixed-Q SVF initialization")
        if self.updates:
            raise ValueError("Initialize only a fresh learner")
        for name in (
            "actor",
            "reference",
            "critic",
            "inner_critic",
            "target_critic",
            "target_inner_critic",
        ):
            getattr(self, name).load_state_dict(state[name], strict=True)
        # Preserve the destination's trainability and fresh optimizer/counter.

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


class FrozenQInnerOnly(FrozenQSoftValueFlow):
    """Train only inner soft value; actor/reference/Q are immutable scorers."""

    def __init__(self, actor, critic, inner_critic, config, reference=None):
        actor.requires_grad_(False).eval()
        super().__init__(actor, critic, inner_critic, config, reference)

    def initialize_from_svf(self, state):
        if state["algorithm"] != "FrozenQSoftValueFlow" or state["config"] != asdict(self.config):
            raise ValueError("Require matching fixed-Q SVF initialization")
        if self.updates:
            raise ValueError("Initialize only a fresh inner-only learner")
        for name in (
            "actor",
            "reference",
            "critic",
            "inner_critic",
            "target_critic",
            "target_inner_critic",
        ):
            getattr(self, name).load_state_dict(state[name], strict=True)
        # New inner-only optimizer and step counter; don't import actor Adam state.

    def losses(self, batch):
        batch.validate()
        actions, mask, observations = (
            batch.actions * batch.action_mask,
            batch.action_mask,
            batch.observations,
        )
        time = actions.new_empty(actions.shape[0]).uniform_(self.config.t_min, 1)
        t = time[:, None, None]
        noisy = ((1 - t) * torch.randn_like(actions) + t * actions) * mask
        endpoints = self.base_sde_endpoints(observations, noisy, time, mask)
        q = self._endpoint_q(observations, endpoints)
        temperature = self.estimate_temperature(batch, endpoint_q=q)
        target = soft_value(q, temperature).detach()
        prediction = aggregate_heads(
            self.inner_critic(observations, noisy, time), self.config.q_aggregation
        )
        loss = (prediction - target).square().mean()
        metrics = {
            "loss": loss,
            "critic/inner_loss": loss,
            "critic/inner_abs_error": (prediction - target).abs().mean(),
            "critic/outer_loss": loss.new_zeros(()),
            "critic/env_q_frozen": loss.new_ones(()),
            "actor/frozen": loss.new_ones(()),
            "sv/lambda": temperature,
            "critic/sv_q_spread": q.std(0, correction=0).mean(),
        }
        for label, values in (("q", q), ("inner", prediction), ("inner_target", target)):
            for stat in ("min", "mean", "max"):
                metrics[f"{label}/{stat}"] = getattr(values, stat)()
        return loss, metrics

    def update(self, batch):
        self.actor.eval()
        self.reference.eval()
        self.critic.eval()
        self.inner_critic.train()
        loss, metrics = self.losses(batch)
        norm = self._step(loss, [self.inner_critic], self._update_targets)
        return {
            **{key: float(value.detach()) for key, value in metrics.items()},
            "grad_norm": norm,
            "inner/grad_norm": norm,
        }


def load_frozen_iql_q(
    checkpoint, cache, *, identity, feature_dim, action_mask, gamma, reward, device="cpu"
):
    """Import ONLINE twin Q, not V/target/optimizer, from a trusted full state.

    Require exact cache provenance, BC/normalization/dataset identity, action mask,
    reward and discount. Legacy live-VLM checkpoints are intentionally rejected.
    DEAS restores its projection and decodes each head's distribution expectation.
    Its two source discounts are recorded; fixed-Q SVF performs no TD backup.
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
        or algorithm.get("algorithm") not in ("iql-critic-only-v1", "deas-cached-critic-v1")
        or state.get("step", 0) < 1
        or state["step"] != algorithm["updates"]
        or metadata.get("backend") != "frozen-bc-cache-v1"
        or metadata.get("cache_manifest_sha256") != digest
        or metadata.get("bc_identity") != manifest["identity"]
    ):
        raise ValueError("Require a matching cached-IQL or DEAS full training checkpoint")
    settings = metadata["args"]
    is_deas = algorithm["algorithm"] == "deas-cached-critic-v1"
    source_gamma = settings["discount2"] if is_deas else settings["gamma"]
    if source_gamma != gamma or settings["reward"] != reward:
        raise ValueError("IQL reward/discount mismatch")
    if is_deas:
        if metadata.get("deas_config") != algorithm["config"] or any(
            settings[key] != algorithm["config"][key]
            for key in ("discount1", "discount2", "expectile", "learning_rate")
        ):
            raise ValueError("DEAS configuration provenance mismatch")
        critic = FrozenDEASQ(algorithm, feature_dim, action_mask.shape[1:], indices)
    else:
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
        "source": (
            "online DEAS projection + Q1/Q2 distribution means; no V or optimizer"
            if is_deas
            else "online IQL Q1/Q2; no IQL V or optimizer"
        ),
        "metadata": metadata,
    }
    if is_deas:
        # Frozen-Q SVF never uses batch rewards/discounts in its objective.
        # Preserve BOTH source discounts rather than claiming a single-gamma TD.
        provenance["discount1"] = settings["discount1"]
        provenance["discount2"] = settings["discount2"]
    return critic.to(device).requires_grad_(False).eval(), provenance
