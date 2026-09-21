"""Action-reinjected scalar IQL. Independent of the actor and legacy IQL/SVF.

Public Q layout is [batch, heads], NOT the legacy [heads, batch] layout.
All critic math is FP32; only state conditioning is detached.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class FrozenStateFeatures(nn.Module):
    """Read cached or already encoded BC conditioning without an action path."""

    def __init__(self, feature_dim, exclude_proprio=False):
        super().__init__()
        self.feature_dim = feature_dim
        if exclude_proprio and feature_dim != 2212:
            raise ValueError("No-proprio ablation requires N1.7 2048+132+32 cache layout")
        self.exclude_proprio = exclude_proprio
        self.output_dim = feature_dim - (132 if exclude_proprio else 0)

    def forward(self, observations):
        features = observations["features"] if isinstance(observations, dict) else observations
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("Expected [B, feature_dim] frozen state conditioning")
        features = features.detach().float()
        if self.exclude_proprio:
            features = torch.cat((features[:, :2048], features[:, 2180:]), -1)
        return features


class FrozenGR00TStateExtractor:
    """Optional live interface; same representation as the existing BC cache.

    Freezes the entire supplied BC. No DiT call, action labels or noisy actions.
    Keep this outside Q so frozen backbone evaluation cannot disable dQ/da.
    """

    def __init__(self, model):
        from .adapters import FrozenBCGR00TEncoder

        self.encoder = FrozenBCGR00TEncoder(model)

    def __call__(self, processed_observation):
        return self.encoder.encode_observation(processed_observation)["features"]


def q_statistics(q_values, beta=0.0):
    """Population ensemble std (finite even for E=1); pessimism is diagnostic only."""
    if q_values.ndim != 2 or q_values.shape[-1] < 1:
        raise ValueError("Q ensemble must be [B, E]")
    mean, std = q_values.mean(-1), q_values.std(-1, correction=0)
    return {
        "mean": mean,
        "min": q_values.min(-1).values,
        "max": q_values.max(-1).values,
        "std": std,
        "mean_minus_beta_std": mean - beta * std,
    }


class ActionInjectedQHead(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        width = state_dim
        for hidden in hidden_dims:
            self.layers.append(nn.Linear(width + action_dim, hidden))
            width = hidden
        self.out = nn.Linear(width + action_dim, 1)

    def forward(self, state, action):
        h = state
        for layer in self.layers:
            h = F.gelu(layer(torch.cat((h, action), -1)))
        return self.out(torch.cat((h, action), -1)).squeeze(-1)


class ActionConditionedQEnsemble(nn.Module):
    """Q(z,A)->[B,E]. Accepts padded clean chunks; gathers only valid coordinates.

    No action encoder is shared between heads: raw compact action is re-injected
    into every hidden layer AND the scalar output layer. Gather/cast preserve
    gradients back to the input chunk; padded coordinates have zero gradient.
    """

    def __init__(
        self,
        state_dim,
        action_shape,
        action_indices=None,
        hidden_dims=(512, 512, 256),
        num_q_heads=10,
        exclude_proprio=False,
    ):
        super().__init__()
        self.action_shape = tuple(action_shape)
        self.hidden_dims = tuple(hidden_dims)
        if len(self.action_shape) != 2 or min(self.action_shape) < 1:
            raise ValueError("action_shape must be positive (H, A)")
        if num_q_heads < 1 or not hidden_dims or min(hidden_dims) < 1 or state_dim < 1:
            raise ValueError("Invalid critic dimensions")
        indices = torch.as_tensor(
            list(range(math.prod(action_shape))) if action_indices is None else action_indices,
            dtype=torch.long,
        )
        if (
            indices.ndim != 1
            or not indices.numel()
            or indices.min() < 0
            or indices.max() >= math.prod(action_shape)
            or indices.unique().numel() != indices.numel()
        ):
            raise ValueError("Invalid action_indices")
        self.register_buffer("action_indices", indices)
        self.state_features = FrozenStateFeatures(state_dim, exclude_proprio)
        self.heads = nn.ModuleList(
            [
                ActionInjectedQHead(self.state_features.output_dim, indices.numel(), hidden_dims)
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, state_feat, action):
        if action.ndim != 3 or tuple(action.shape[1:]) != self.action_shape:
            raise ValueError("Action must have the configured [B,H,A] shape")
        with torch.autocast(device_type=action.device.type, enabled=False):
            state = self.state_features(state_feat)
            a = action.float().flatten(1).index_select(1, self.action_indices)
            if state.shape[0] != a.shape[0]:
                raise ValueError("State/action batch mismatch")
            return torch.stack([head(state, a) for head in self.heads], -1)


class StateValue(nn.Module):
    def __init__(self, state_dim, hidden_dims=(512, 512, 256), exclude_proprio=False):
        super().__init__()
        if state_dim < 1 or not hidden_dims or min(hidden_dims) < 1:
            raise ValueError("Invalid value dimensions")
        self.state_features = FrozenStateFeatures(state_dim, exclude_proprio)
        layers, width = [], self.state_features.output_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.GELU()))
            width = hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, state_feat):
        state = self.state_features(state_feat)
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.net(state).squeeze(-1)


@dataclass(frozen=True)
class ActionIQLConfig:
    critic_lr: float = 1e-4
    value_lr: float = 1e-4
    expectile_tau: float = 0.8
    target_tau: float = 0.005
    max_grad_norm: float = 10.0
    diagnostic_every: int = 50
    perturb_sigma: float = 0.05
    diagnostic_beta: float = 1.0
    assume_all_success: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Configuration must be finite")
        if (
            min(self.critic_lr, self.value_lr, self.max_grad_norm) <= 0
            or not 0 < self.expectile_tau < 1
            or not 0 < self.target_tau <= 1
            or self.diagnostic_every < 1
            or self.perturb_sigma < 0
        ):
            raise ValueError("Invalid IQL hyperparameters")


class ActionIQLLearner:
    """Q fits r_H+d*EMA(V)(s'); V fits detached online ensemble mean.

    `batch.discounts` is the COMPLETE bootstrap multiplier, including terminal
    masking and gamma**H. Never apply gamma or (1-done) a second time here.
    """

    def __init__(self, critic, value, config=ActionIQLConfig()):
        self.critic, self.value, self.config = critic.float(), value.float(), config
        if set(critic.parameters()) & set(value.parameters()):
            raise ValueError("Q and V must have independent parameters")
        self.target_value = deepcopy(value).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(
            [
                {"params": critic.parameters(), "lr": config.critic_lr},
                {"params": value.parameters(), "lr": config.value_lr},
            ]
        )
        self.updates = 0

    def objectives(self, batch):
        batch.validate()
        action = batch.actions * batch.action_mask
        q = self.critic(batch.observations, action)
        v = self.value(batch.observations)
        with torch.no_grad():
            target = batch.rewards + batch.discounts * self.target_value(batch.next_observations)
        diff = q.mean(-1).detach() - v
        weight = torch.where(diff > 0, self.config.expectile_tau, 1 - self.config.expectile_tau)
        return (q - target[:, None]).square().mean(), (weight * diff.square()).mean(), q, v, target

    @torch.enable_grad()
    def action_diagnostics(self, batch):
        # Frozen Q parameters still permit dQ/da. Never enclose this in no_grad.
        action = batch.actions.detach().float().clone().requires_grad_(True)
        q = self.critic(batch.observations, action * batch.action_mask).mean(-1)
        grad = torch.autograd.grad(q.sum(), action)[0]
        # Local deterministic generator: logging does not alter training RNG.
        generator = torch.Generator(device=action.device).manual_seed(1729 + self.updates)
        noise = torch.randn(action.shape, generator=generator, device=action.device)
        with torch.no_grad():
            perturbed = (action + self.config.perturb_sigma * noise) * batch.action_mask
            q_perturbed = self.critic(batch.observations, perturbed).mean(-1)
        return {
            "critic/action_grad_norm": grad.flatten(1).norm(dim=-1).mean().item(),
            "critic/action_grad_max": grad.abs().max().item(),
            "critic/q_dataset_action": q.mean().item(),
            "critic/q_perturbed_action": q_perturbed.mean().item(),
            "critic/perturb_abs_delta": (q_perturbed - q.detach()).abs().mean().item(),
        }

    def update(self, batch):
        self.critic.train()
        self.value.train()
        self.target_value.eval()
        q_loss, v_loss, q, v, target = self.objectives(batch)
        loss = q_loss + v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite action-conditioned IQL loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        q_grad = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        v_grad = nn.utils.clip_grad_norm_(
            self.value.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            for dst, src in zip(
                self.target_value.parameters(), self.value.parameters(), strict=True
            ):
                dst.lerp_(src, self.config.target_tau)
        self.updates += 1
        stats = q_statistics(q.detach(), self.config.diagnostic_beta)
        metrics = {
            "loss/q": q_loss.item(),
            "loss/v": v_loss.item(),
            "q/mean": stats["mean"].mean().item(),
            "q/std": stats["std"].mean().item(),
            "q/min": q.detach().min().item(),
            "q/max": q.detach().max().item(),
            "q/state_std": stats["mean"].std(correction=0).item(),
            "q/mean_minus_beta_std": stats["mean_minus_beta_std"].mean().item(),
            "v/mean": v.detach().mean().item(),
            "v/min": v.detach().min().item(),
            "v/max": v.detach().max().item(),
            "target_q/mean": target.mean().item(),
            "reward/mean": batch.rewards.mean().item(),
            "batch/terminal_ratio": batch.terminated.float().mean().item(),
            "critic/abs_td_loss": (q.detach() - target[:, None]).abs().mean().item(),
            "critic/grad_norm": q_grad.item(),
            "value/grad_norm": v_grad.item(),
        }
        if self.updates == 1 or self.updates % self.config.diagnostic_every == 0:
            metrics.update(self.action_diagnostics(batch))
        if self.config.assume_all_success:
            metrics.update({"data/success_ratio": 1.0, "data/success_ratio_is_assumed": 1.0})
        return metrics

    def state_dict(self):
        return {
            "algorithm": "action-conditioned-iql-v1",
            "config": asdict(self.config),
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "target_value": self.target_value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if state["algorithm"] != "action-conditioned-iql-v1" or state["config"] != asdict(
            self.config
        ):
            raise ValueError("IQL variant/configuration mismatch")
        for name in ("critic", "value", "target_value", "optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.updates = state["updates"]


def load_action_iql_models(path, device="cpu"):
    """Load this variant's trusted model-only archive for scoring or dQ/da.

    Returns Q,V,metadata; all weights are frozen, action gradients remain enabled.
    Not compatible with legacy IQL/SVF loaders (different ensemble axis/targets).
    """
    archive = torch.load(path, map_location="cpu", weights_only=True)
    metadata, state = archive["metadata"], archive["algorithm"]
    if (
        metadata.get("backend") != "action-conditioned-iql-cache-v1"
        or state.get("algorithm") != "action-conditioned-iql-v1"
        or metadata.get("q_layout") != "batch,heads"
    ):
        raise ValueError("Not an action-conditioned IQL model")
    args = metadata["args"]
    q = ActionConditionedQEnsemble(
        metadata["feature_dim"],
        metadata["action_shape"],
        metadata["action_indices"],
        args["hidden_dims"],
        args["num_q_heads"],
        exclude_proprio=args.get("exclude_proprio", False),
    )
    v = StateValue(
        metadata["feature_dim"],
        args["hidden_dims"],
        exclude_proprio=args.get("exclude_proprio", False),
    )
    for model, key in ((q, "critic"), (v, "value")):
        model.load_state_dict(state[key], strict=True)
        model.to(device).requires_grad_(False).eval()
    return q, v, metadata
