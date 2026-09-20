"""Torch offline algorithms independent of GR00T's data and model internals.

SVF's default equations follow fmrl origin/dh 7304b615, qflow_svf/agents/qflow_rc.py.
The default soft-value/actor path is ported, not every experimental switch.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .types import OfflineRLBatch, map_tensors


@contextmanager
def evaluating(module):
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Flow prediction, target, and mask shapes must match")
    return ((prediction.float() - target.float()).square() * mask).sum() / mask.sum().clamp_min(1)


def aggregate_heads(values, mode):
    if values.ndim != 2:
        raise ValueError("Critics must return [ensemble, batch], including for a single head")
    if mode == "mean":
        return values.mean(0)
    if mode == "min":
        return values.min(0).values
    raise ValueError(f"Unknown ensemble aggregation {mode}")


def soft_value(q_values, temperature):
    """Stable lambda * logmeanexp(Q/lambda), over independent candidates [K,B]."""
    if q_values.ndim != 2 or not torch.all(torch.as_tensor(temperature) > 0):
        raise ValueError("Require Q[K,B] and positive temperature")
    return temperature * (
        torch.logsumexp(q_values / temperature, dim=0) - math.log(q_values.shape[0])
    )


def svf_coefficient(time, temperature, kappa, t_min):
    safe = time.clamp_min(t_min)
    return torch.where(time >= t_min, kappa**2 * (1 - safe) / (safe * temperature), 0.0)


@dataclass(frozen=True)
class SVFConfig:
    learning_rate: float = 3e-4
    tau: float = 0.005
    flow_steps: int = 10
    candidates: int = 8
    t_min: float = 0.1
    kappa: float = 1.0
    lambda_multiplier: float = 1.0
    lambda_batch_size: int = 64
    soft_lambda: float | None = None
    q_aggregation: str = "mean"
    freeze_reference: bool = False
    action_clip: float | None = 1.0
    max_grad_norm: float | None = None

    def __post_init__(self):
        for name in ("learning_rate", "lambda_multiplier"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.t_min < 1 or not 0 <= self.tau <= 1:
            raise ValueError("Require 0<t_min<1 and 0<=tau<=1")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        for name in ("flow_steps", "candidates", "lambda_batch_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("soft_lambda", "action_clip", "max_grad_norm"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be None or finite and positive")
        if self.q_aggregation not in ("mean", "min"):
            raise ValueError("q_aggregation must be mean or min")


class FlowBC:
    """A minimal second algorithm exercising the same batch/trainer interfaces."""

    def __init__(self, actor: nn.Module, config: SVFConfig = SVFConfig()):
        self.actor, self.config = actor, config
        self.optimizer = torch.optim.Adam(
            [p for p in actor.parameters() if p.requires_grad], lr=config.learning_rate
        )
        self.updates = 0

    def _clip(self, actions, mask):
        if self.config.action_clip is not None:
            actions = actions.clamp(-self.config.action_clip, self.config.action_clip)
        return actions * mask

    @torch.no_grad()
    def sample_actions(self, observations, action_mask, initial_noise=None):
        with evaluating(self.actor):
            actions = (
                torch.randn_like(action_mask, dtype=torch.float32)
                if initial_noise is None
                else initial_noise.float().clone()
            )
            if actions.shape != action_mask.shape:
                raise ValueError("initial_noise and mask shape mismatch")
            actions = actions * action_mask
            for step in range(self.config.flow_steps):
                time = actions.new_full((actions.shape[0],), step / self.config.flow_steps)
                actions = (
                    actions
                    + self.actor(observations, actions, time).float() / self.config.flow_steps
                ) * action_mask
            return self._clip(actions, action_mask)

    def _step(self, loss, modules, before_step=None):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite offline RL loss; optimizer was not stepped")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for module in modules for p in module.parameters() if p.requires_grad]
        norm = nn.utils.clip_grad_norm_(
            params, self.config.max_grad_norm or float("inf"), error_if_nonfinite=True
        )
        if before_step is not None:
            before_step()
        self.optimizer.step()
        self.updates += 1
        return float(norm.detach())

    def update(self, batch: OfflineRLBatch):
        batch.validate()
        self.actor.train()
        noise = torch.randn_like(batch.actions) * batch.action_mask
        time = batch.actions.new_empty(batch.actions.shape[0]).uniform_()
        t = time[:, None, None]
        actions = batch.actions * batch.action_mask
        noisy = (1 - t) * noise + t * actions
        loss = masked_mse(
            self.actor(batch.observations, noisy, time), actions - noise, batch.action_mask
        )
        grad_norm = self._step(loss, [self.actor])
        return {
            "loss": float(loss.detach()),
            "actor/bc_loss": float(loss.detach()),
            "grad_norm": grad_norm,
        }

    def state_dict(self):
        return {
            "algorithm": type(self).__name__,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if state["algorithm"] != type(self).__name__ or state["config"] != asdict(self.config):
            raise ValueError("Checkpoint algorithm/config does not match this learner")
        self.actor.load_state_dict(state["actor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]


class SoftValueFlow(FlowBC):
    """SVF with injected flow actor/reference and outer/time-conditioned critics.

    Actor and reference must have independent parameters. Only frozen observation
    encodings may be shared. All four objectives use pre-update parameters and
    one Adam step, as in the dh reference. Target EMA also uses pre-step weights
    (the reference target_update reads self.network, not new_network).
    """

    def __init__(self, actor, critic, inner_critic, config=SVFConfig(), reference=None):
        self.actor, self.config = actor, config
        self.reference = deepcopy(actor) if reference is None else reference
        self.critic, self.inner_critic = critic, inner_critic
        modules = [actor, self.reference, critic, inner_critic]
        parameter_ids = [id(p) for module in modules for p in module.parameters()]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("Actor, reference, and critics must not share parameters")
        if config.freeze_reference:
            self.reference.requires_grad_(False).eval()
        self.target_critic = deepcopy(critic).requires_grad_(False).eval()
        self.target_inner_critic = deepcopy(inner_critic).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(
            [p for module in modules for p in module.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )
        self.updates = 0

    @torch.no_grad()
    def base_sde_endpoints(self, observations, noisy, time, mask):
        if not ((time >= self.config.t_min) & (time <= 1)).all():
            raise ValueError("Reference SDE anchors must satisfy t_min <= t <= 1")
        endpoints = []
        with evaluating(self.reference):
            for _ in range(self.config.candidates):
                x, s = noisy.clone() * mask, time.clone()
                for _ in range(self.config.flow_steps):
                    ds = (1 - s).clamp(0, 1 / self.config.flow_steps)
                    u = self.reference(observations, x, s).float() * mask
                    st, dt = s[:, None, None], ds[:, None, None]
                    drift = u - self.config.kappa**2 * (x - st * u) / st
                    noise = (
                        self.config.kappa
                        * (2 * (1 - st) / st * dt).clamp_min(0).sqrt()
                        * torch.randn_like(x)
                    )
                    x = (x + drift * dt + noise) * mask
                    s = (s + ds).clamp_max(1)
                endpoints.append(self._clip(x, mask))
        return torch.stack(endpoints).detach()

    @torch.no_grad()
    def _endpoint_q(self, observations, endpoints):
        return torch.stack(
            [
                aggregate_heads(self.target_critic(observations, action), self.config.q_aggregation)
                for action in endpoints
            ]
        )

    @torch.no_grad()
    def estimate_temperature(self, batch):
        if self.config.soft_lambda is not None:
            return batch.actions.new_tensor(self.config.soft_lambda)
        n = min(self.config.lambda_batch_size, batch.actions.shape[0])
        observations = map_tensors(batch.observations, lambda x: x[:n])
        actions, mask = batch.actions[:n], batch.action_mask[:n]
        noise = torch.randn_like(actions) * mask
        time = actions.new_empty(n).uniform_(self.config.t_min, 1)
        t = time[:, None, None]
        endpoints = self.base_sde_endpoints(
            observations, ((1 - t) * noise + t * actions) * mask, time, mask
        )
        q = self._endpoint_q(observations, endpoints)
        # Population std, as jnp.std; torch's default correction=1 is NOT equivalent.
        spread = q.std(dim=0, correction=0).mean()
        return (self.config.lambda_multiplier * spread.clamp_min(1e-3)).detach()

    def losses(self, batch):
        batch.validate()
        actions, mask, obs = (
            batch.actions * batch.action_mask,
            batch.action_mask,
            batch.observations,
        )
        temperature = self.estimate_temperature(batch)
        with torch.no_grad():
            next_actions = self.sample_actions(batch.next_observations, mask)
            next_q = aggregate_heads(
                self.target_critic(batch.next_observations, next_actions), self.config.q_aggregation
            )
            td_target = batch.rewards + batch.discounts * next_q
        outer_loss = (self.critic(obs, actions) - td_target.unsqueeze(0)).square().mean()

        # Independent anchors/draws from temperature estimation.
        inner_time = actions.new_empty(actions.shape[0]).uniform_(self.config.t_min, 1)
        it = inner_time[:, None, None]
        inner_x = ((1 - it) * torch.randn_like(actions) + it * actions) * mask
        endpoints = self.base_sde_endpoints(obs, inner_x, inner_time, mask)
        q = self._endpoint_q(obs, endpoints)
        inner_target = soft_value(q, temperature).detach()
        inner_prediction = aggregate_heads(
            self.inner_critic(obs, inner_x, inner_time), self.config.q_aggregation
        )
        inner_loss = (inner_prediction - inner_target).square().mean()

        # Actor CFM covers [0,1); target is DATA velocity, not reference(obs,x,t).
        time = actions.new_empty(actions.shape[0]).uniform_()
        t = time[:, None, None]
        noise = torch.randn_like(actions) * mask
        x = ((1 - t) * noise + t * actions).detach().requires_grad_(True)
        with torch.enable_grad():
            value = self.inner_critic(obs, x * mask, time).mean(0).sum()
            gradient = torch.autograd.grad(value, x, create_graph=False)[0].detach() * mask
        coefficient = svf_coefficient(time, temperature, self.config.kappa, self.config.t_min)
        guidance = coefficient[:, None, None] * gradient
        data_velocity = actions - noise
        actor_target = (data_velocity + guidance).detach()
        actor_loss = masked_mse(self.actor(obs, x.detach(), time), actor_target, mask)
        if self.config.freeze_reference:
            with torch.no_grad():
                reference_prediction = self.reference(obs, x.detach(), time)
        else:
            reference_prediction = self.reference(obs, x.detach(), time)
        bc_loss = masked_mse(reference_prediction, data_velocity, mask)
        loss = (
            outer_loss + inner_loss + actor_loss + (0 if self.config.freeze_reference else bc_loss)
        )
        metrics = {
            "loss": loss,
            "critic/outer_loss": outer_loss,
            "critic/inner_loss": inner_loss,
            "actor/loss": actor_loss,
            "actor/bc_flow_loss": bc_loss,
            "sv/lambda": temperature,
            "critic/sv_q_spread": q.std(0, correction=0).mean(),
            "critic/sv_weight_max": (q / temperature).softmax(0).max(0).values.mean(),
            "actor/sv_coef_mean": coefficient.mean(),
            "actor/sv_guidance_norm": guidance.flatten(1).norm(dim=1).mean(),
            "critic/td_target_mean": td_target.mean(),
        }
        return loss, metrics

    @torch.no_grad()
    def _update_targets(self):
        for online, target in (
            (self.critic, self.target_critic),
            (self.inner_critic, self.target_inner_critic),
        ):
            for parameter, target_parameter in zip(
                online.parameters(), target.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.tau)
            for buffer, target_buffer in zip(online.buffers(), target.buffers(), strict=True):
                target_buffer.copy_(buffer)

    def update(self, batch):
        self.actor.train()
        self.critic.train()
        self.inner_critic.train()
        self.reference.train(not self.config.freeze_reference)
        loss, metrics = self.losses(batch)
        norm = self._step(
            loss, [self.actor, self.reference, self.critic, self.inner_critic], self._update_targets
        )
        return {**{key: float(value.detach()) for key, value in metrics.items()}, "grad_norm": norm}

    def state_dict(self):
        return {
            **super().state_dict(),
            **{
                key: getattr(self, key).state_dict()
                for key in (
                    "reference",
                    "critic",
                    "inner_critic",
                    "target_critic",
                    "target_inner_critic",
                )
            },
        }

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for key in ("reference", "critic", "inner_critic", "target_critic", "target_inner_critic"):
            getattr(self, key).load_state_dict(state[key])
