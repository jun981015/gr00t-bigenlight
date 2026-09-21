"""DEAS distributional critic learning on immutable N1.7 pooled features.

The frozen N1.5 token attention frontend is not reproduced by this cache backend.
Q MLPs, residual V, trainable projection, and losses follow n15 DEASCritic.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .feature_cache import CachedFeatureDataset
from .networks import mlp


class DEASDataset(CachedFeatureDataset):
    def __init__(self, root, *, discount1=0.9, discount2=0.99):
        if not 0 <= discount1 < 1 or not 0 <= discount2 < 1:
            raise ValueError("Discounts must be in [0, 1)")
        super().__init__(root, reward="step-cost", gamma=discount1)
        self.discount2 = discount2

    def batch(self, indices):
        batch = super().batch(indices)
        return replace(batch, discounts=(~batch.terminated).float() * self.discount2**self.horizon)


class HLGauss(nn.Module):
    def __init__(self, minimum=-100.0, bins=101, sigma_ratio=0.1):
        super().__init__()
        edges = torch.linspace(minimum, 0, bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2)
        self.sigma = sigma_ratio * (-minimum / bins)

    def decode(self, logits):
        return (logits.softmax(-1) * self.centers).sum(-1)

    def probabilities(self, target):
        # Clip out-of-support targets before erf to prevent 0/0 normalization.
        target = target.clamp(self.edges[0], self.edges[-1])
        cdf = torch.erf((self.edges - target.unsqueeze(-1)) / (math.sqrt(2) * self.sigma))
        return (cdf[..., 1:] - cdf[..., :-1]) / (cdf[..., -1:] - cdf[..., :1])


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, x):
        return x + self.block(x)


@dataclass(frozen=True)
class DEASConfig:
    learning_rate: float = 1e-4
    expectile: float = 0.7
    target_tau: float = 0.005
    discount1: float = 0.9
    discount2: float = 0.99
    vlm_dim: int = 2048
    state_dim: int = 132
    embodiment_dim: int = 32
    projection_width: int = 1024
    feature_dim: int = 64
    hidden_dim: int = 512
    depth: int = 4
    bins: int = 101
    sigma_ratio: float = 0.1
    max_grad_norm: float = 10.0


class DEASCachedLearner:
    def __init__(self, action_indices, config=DEASConfig(), device="cpu"):
        self.config = config
        c = config
        if not (0 < c.expectile < 1 and 0 < c.target_tau <= 1 and c.learning_rate > 0):
            raise ValueError("Invalid DEAS learning configuration")
        # One embodiment in these datasets: train only its projection slice.
        self.action_indices = torch.as_tensor(action_indices, dtype=torch.long, device=device)
        self.projection = nn.Sequential(
            nn.Linear(c.vlm_dim, c.projection_width),
            nn.SiLU(),
            nn.Linear(c.projection_width, c.projection_width),
            nn.SiLU(),
            nn.Linear(c.projection_width, c.projection_width),
            nn.SiLU(),
            nn.Linear(c.projection_width, c.feature_dim),
            nn.Tanh(),
        ).to(device)
        obs_dim = c.feature_dim + c.state_dim
        self.critic = nn.ModuleList(
            [
                mlp(obs_dim + len(action_indices), c.bins, (c.hidden_dim,) * c.depth, True)
                for _ in range(2)
            ]
        ).to(device)
        self.value = nn.Sequential(
            nn.Linear(obs_dim, c.hidden_dim),
            nn.LayerNorm(c.hidden_dim),
            nn.ReLU(),
            *[ResidualBlock(c.hidden_dim) for _ in range(c.depth)],
            nn.Linear(c.hidden_dim, c.bins),
        ).to(device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.hlg = HLGauss(-1 / (1 - c.discount2), c.bins, c.sigma_ratio).to(device)
        self.parameters = [
            p for m in (self.projection, self.critic, self.value) for p in m.parameters()
        ]
        self.optimizer = torch.optim.Adam(self.parameters, lr=c.learning_rate)
        self.updates = 0

    def features(self, observations):
        x = observations["features"].float()
        c = self.config
        if x.shape[-1] != c.vlm_dim + c.state_dim + c.embodiment_dim:
            raise ValueError("Cached conditioning layout mismatch")
        return torch.cat(
            (self.projection(x[:, : c.vlm_dim]), x[:, c.vlm_dim : c.vlm_dim + c.state_dim]), -1
        )

    def update(self, batch):
        batch.validate()
        actions = (batch.actions * batch.action_mask).flatten(1)[:, self.action_indices]
        features = self.features(batch.observations)
        v_logits = self.value(features)
        v = self.hlg.decode(v_logits)
        with torch.no_grad():
            q_target_logits = torch.stack(
                [q(torch.cat((features.detach(), actions), -1)) for q in self.target_critic]
            )
            q_target_values = self.hlg.decode(q_target_logits)
            choice = q_target_values.argmin(0)
            target_probs = q_target_logits[choice, torch.arange(len(v), device=v.device)].softmax(
                -1
            )
            target_q = (target_probs * self.hlg.centers).sum(-1)
            next_v = self.hlg.decode(self.value(self.features(batch.next_observations)))
            target = batch.rewards + batch.discounts * next_v
            target_distribution = self.hlg.probabilities(target)
            weights = torch.where(target_q >= v, self.config.expectile, 1 - self.config.expectile)
        v_loss = (weights * -(target_probs * v_logits.log_softmax(-1)).sum(-1)).mean()
        q_logits = torch.stack([q(torch.cat((features, actions), -1)) for q in self.critic])
        q_loss = -(target_distribution.unsqueeze(0) * q_logits.log_softmax(-1)).sum(-1).mean()
        loss = q_loss + v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DEAS loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        def norm(module):
            return (
                torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None])
                .norm()
                .item()
            )

        q_grad, v_grad, projection_grad = norm(self.critic), norm(self.value), norm(self.projection)
        grad = nn.utils.clip_grad_norm_(
            self.parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)
        self.updates += 1
        q = self.hlg.decode(q_logits.detach())
        metrics = {
            "loss/q": q_loss.item(),
            "loss/v": v_loss.item(),
            "critic/loss": q_loss.item(),
            "critic/abs_td_loss": (q - target).abs().mean().item(),
            "critic/grad_norm": q_grad,
            "value/grad_norm": v_grad,
            "projection/grad_norm": projection_grad,
            "grad_norm": grad.item(),
            "target/mean": target.mean().item(),
            "reward/mean": batch.rewards.mean().item(),
            "value/upper_expectile_fraction": (target_q >= v.detach()).float().mean().item(),
        }
        for name, values in (("q", q), ("v", v.detach())):
            for stat in ("min", "mean", "max"):
                metrics[f"{name}/{stat}"] = getattr(values, stat)().item()
        return metrics

    def state_dict(self):
        return {
            "algorithm": "deas-cached-critic-v1",
            "config": asdict(self.config),
            "action_indices": self.action_indices.cpu(),
            "updates": self.updates,
            **{
                name: getattr(self, name).state_dict()
                for name in ("projection", "critic", "value", "target_critic", "optimizer")
            },
        }

    def load_state_dict(self, state):
        if state["algorithm"] != "deas-cached-critic-v1" or state["config"] != asdict(self.config):
            raise ValueError("DEAS checkpoint configuration mismatch")
        if not torch.equal(state["action_indices"].cpu(), self.action_indices.cpu()):
            raise ValueError("DEAS checkpoint action mask mismatch")
        for name in ("projection", "critic", "value", "target_critic", "optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.updates = state["updates"]
