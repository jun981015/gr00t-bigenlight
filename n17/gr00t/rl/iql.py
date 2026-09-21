"""Critic-only scalar IQL. No policy optimization or SVF inner critic."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .networks import mlp


@dataclass(frozen=True)
class IQLConfig:
    learning_rate: float = 3e-4
    expectile: float = 0.7
    target_tau: float = 0.005
    max_grad_norm: float = 10.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("IQL configuration must be finite")
        if not 0 < self.expectile < 1 or not 0 < self.target_tau <= 1:
            raise ValueError("Invalid expectile/target_tau")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate and gradient limit must be positive")


class FeatureValue(nn.Module):
    """State value V(s), not the time-conditioned SVF soft-value critic."""

    def __init__(self, feature_dim, hidden_dims=(512,) * 4):
        super().__init__()
        self.net = mlp(feature_dim, 1, hidden_dims, layer_norm=True)

    def forward(self, observations):
        return self.net(observations["features"].float()).squeeze(-1)


class IQLCriticLearner:
    def __init__(self, critic, value, config=IQLConfig()):
        self.critic, self.value, self.config = critic, value, config
        self.target_critic = deepcopy(critic).requires_grad_(False).eval()
        # Q and V may share a pooled-feature encoder. Optimize each weight once.
        self.parameters = list(dict.fromkeys([*critic.parameters(), *value.parameters()]))
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.learning_rate)
        self.updates = 0

    def update(self, batch):
        batch.validate()
        actions = batch.actions * batch.action_mask
        # All bootstrap targets use the pre-update networks. Only Q has an EMA.
        self.target_critic.eval()
        self.value.eval()
        with torch.no_grad():
            target_q = self.target_critic(batch.observations, actions).min(0).values
            target = batch.rewards + batch.discounts * self.value(batch.next_observations)
        self.critic.train()
        self.value.train()
        q = self.critic(batch.observations, actions)
        v = self.value(batch.observations)
        difference = target_q - v
        weights = torch.where(difference > 0, self.config.expectile, 1 - self.config.expectile)
        v_loss = (weights * difference.square()).mean()
        q_loss = (q - target.unsqueeze(0)).square().mean()
        loss = q_loss + v_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite IQL loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Separate, pre-clipping L2 norms: Q only versus the auxiliary IQL V.
        def gradient_norm(module):
            return torch.stack(
                [p.grad.detach().float().norm(2) for p in module.parameters() if p.grad is not None]
            ).norm(2)

        critic_grad_norm = gradient_norm(self.critic)
        value_grad_norm = gradient_norm(self.value)
        grad_norm = nn.utils.clip_grad_norm_(
            self.parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)
        self.updates += 1
        return {
            "loss/q": q_loss.item(),
            "loss/v": v_loss.item(),
            "critic/loss": q_loss.item(),
            # Diagnostic MAE of the Bellman residual; optimization still uses MSE.
            "critic/abs_td_loss": (q.detach() - target.unsqueeze(0)).abs().mean().item(),
            "critic/grad_norm": critic_grad_norm.item(),
            "value/grad_norm": value_grad_norm.item(),
            "q/min": q.detach().min().item(),
            "q/mean": q.detach().mean().item(),
            "q/max": q.detach().max().item(),
            "v/min": v.detach().min().item(),
            "v/mean": v.detach().mean().item(),
            "v/max": v.detach().max().item(),
            "target/mean": target.mean().item(),
            "reward/mean": batch.rewards.mean().item(),
            "reward/nonzero_fraction": (batch.rewards != 0).float().mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def state_dict(self):
        return {
            "algorithm": "iql-critic-only-v1",
            "config": asdict(self.config),
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if state["algorithm"] != "iql-critic-only-v1" or state["config"] != asdict(self.config):
            raise ValueError("IQL checkpoint configuration mismatch")
        for key in ("critic", "value", "target_critic", "optimizer"):
            getattr(self, key).load_state_dict(state[key])
        self.updates = state["updates"]
