"""Frozen observation projection and DiT-only LoRA for SVF.

Cached projected tokens have distinct keys and bypass _encode_features entirely.
Action encoding/decoding remain differentiable forwards with frozen weights.
"""

from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import FrozenGR00TEncoder


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Require positive LoRA rank and alpha")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        update = F.linear(F.linear(value.float(), self.lora_A), self.lora_B) * self.scale
        return self.base(value) + update.to(value.dtype)


def inject_dit_lora(head, rank=16, alpha=32.0):
    """Only DiT attention Q/K/V/output projections; zero initial residual."""
    if any(isinstance(module, LoRALinear) for module in head.modules()):
        raise ValueError("LoRA already installed")
    targets = [
        (name, module)
        for name, module in head.model.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(suffix) for suffix in ("to_q", "to_k", "to_v", "to_out.0"))
    ]
    if not targets:
        raise ValueError("No supported DiT attention projections found")
    head.set_trainable_parameters(False, True, False)
    head.requires_grad_(False)
    for name, module in targets:
        parent, _, leaf = name.rpartition(".")
        setattr(head.model.get_submodule(parent), leaf, LoRALinear(module, rank, alpha))
    return [name for name, _ in targets]


@torch.no_grad()
def project_observation(head, observations):
    """Run frozen BC observation projection once, never on already projected tokens."""
    from transformers.feature_extraction_utils import BatchFeature

    for module in (head.vlln, head.vl_self_attention, head.state_encoder):
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError("Observation projection must be frozen before caching")
        module.eval()
    dtype = next(head.action_encoder.parameters()).dtype
    backbone = BatchFeature(
        data={
            key: observations[key].to(dtype)
            if observations[key].is_floating_point()
            else observations[key]
            for key in ("backbone_features", "backbone_attention_mask", "image_mask")
        }
    )
    encoded = head._encode_features(
        backbone,
        BatchFeature(
            data={
                "state": observations["state"].to(dtype),
                "embodiment_id": observations["embodiment_id"],
            }
        ),
    )
    return {
        "projected_vl_features": encoded["backbone_features"].detach(),
        "projected_state_features": encoded["state_features"].detach(),
        "backbone_attention_mask": observations["backbone_attention_mask"].detach(),
        "image_mask": observations["image_mask"].detach(),
        "embodiment_id": observations["embodiment_id"].detach(),
    }


class FrozenProjectedEncoder(FrozenGR00TEncoder):
    """Batch-local cache: one VLM and BC projection pass per observation batch."""

    def __init__(self, model, critic_cache=None):
        super().__init__(model)
        self.critic_cache = critic_cache

    def __call__(self, processed):
        batch = super().__call__(processed)
        if self.critic_cache is not None:
            batch = reuse_critic_cache(batch, processed.transitions, self.critic_cache)
        return batch

    @torch.no_grad()
    def encode_observation(self, inputs):
        # Parent only pools raw tokens; its result does not mutate backbone tokens.
        raw = super().encode_observation(inputs)
        projected = project_observation(self.model.action_head, raw)
        tokens = projected["projected_vl_features"].float()
        mask = projected["backbone_attention_mask"].to(tokens.dtype).unsqueeze(-1)
        pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
        one_hot = F.one_hot(raw["embodiment_id"], self.model.config.max_num_embodiments).float()
        projected["features"] = torch.cat((pooled, raw["state"].float().flatten(1), one_hot), -1)
        return projected


def reuse_critic_cache(batch, transitions, cache):
    """Keep freshly projected actor tokens; read Q/inner features from existing disk cache."""
    indices = []
    for transition in transitions:
        ep, start = transition.episode_index, transition.step_index
        if not 0 <= ep < len(cache.episode_sizes) or not 0 <= start < cache.episode_sizes[ep]:
            raise ValueError("Transition is outside the original critic cache")
        indices.append(int(cache.offsets[ep]) + start)
    cached = cache.batch(indices).to(batch.actions.device)
    for key in (
        "actions",
        "action_mask",
        "horizons",
        "rewards",
        "discounts",
        "terminated",
        "truncated",
    ):
        left, right = getattr(batch, key), getattr(cached, key)
        if left.shape != right.shape or not torch.allclose(
            left.float(), right.float(), rtol=1e-5, atol=1e-6
        ):
            raise ValueError(f"Live transition differs from critic cache: {key}")
    if batch.observations["features"].shape != cached.observations["features"].shape:
        raise ValueError("Cached critic feature dimension differs from BC")
    return replace(
        batch,
        observations={**batch.observations, **cached.observations},
        next_observations={**batch.next_observations, **cached.next_observations},
    )


class ProjectedFlowActor(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, observations, actions, time):
        head = self.head
        head.set_frozen_modules_to_eval_mode()
        if actions.shape[1:] != (head.action_horizon, head.action_dim):
            raise ValueError("Expected processor-padded action shape")
        dtype = next(head.action_encoder.parameters()).dtype
        embodiment = observations["embodiment_id"]
        buckets = (time * head.num_timestep_buckets).long().clamp(0, head.num_timestep_buckets - 1)
        action_features = head.action_encoder(actions.to(dtype), buckets, embodiment)
        if head.config.add_pos_embed:
            positions = torch.arange(actions.shape[1], device=actions.device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        kwargs = {
            "hidden_states": torch.cat(
                (observations["projected_state_features"].to(dtype), action_features), 1
            ),
            "encoder_hidden_states": observations["projected_vl_features"].to(dtype),
            "encoder_attention_mask": observations["backbone_attention_mask"],
            "timestep": buckets,
            "return_all_hidden_states": True,
        }
        if head.config.use_alternate_vl_dit:
            kwargs.update(
                image_mask=observations["image_mask"],
                backbone_attention_mask=observations["backbone_attention_mask"],
            )
        hidden, _ = head.model(**kwargs)
        return head.action_decoder(hidden, embodiment)[:, -actions.shape[1] :].float()


def lora_actor_pair(head, rank=16, alpha=32.0):
    """Construct reference before attaching adapters; preserve BC compute dtype."""
    reference = ProjectedFlowActor(deepcopy(head).requires_grad_(False).eval())
    targets = inject_dit_lora(head, rank, alpha)
    return ProjectedFlowActor(head), reference, targets
