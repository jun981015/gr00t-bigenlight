"""DEAS feature adapter with independent outer-Q and inner soft-value ensembles.

Architecture/loss primitives, not a complete SVF agent. Reference:
fmrl dh@710cde1, docs/SVF_IMPLEMENTATION.md and qflow_svf/agents/qflow_rc.py.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from gr00t.model.critic.hlg import HLGaussLoss

from .deas_critic import CategorySpecificMLP


@dataclass(frozen=True)
class SVFCriticConfig:
    backbone_dim: int = 1536
    state_dim: int = 64
    action_dim: int = 32
    action_horizon: int = 16
    num_embodiments: int = 32
    projection_hidden_dim: int = 1024
    feature_dim: int = 64
    hidden_dims: tuple = (512, 512, 512, 512)
    num_ensembles: int = 2
    time_embed_dim: int = 16
    q_aggregation: str = "mean"
    inner_aggregation: str = "mean"
    guidance_aggregation: str = "mean"
    tau: float = 0.005
    outer_update: str = "td"
    q_loss_type: str = "mse"
    v_loss_type: str = "mse"  # IQL V(s): scalar expectile or DEAS distributional CE
    iql_expectile: float = 0.7
    hl_gauss_min: float | None = None
    hl_gauss_max: float | None = None
    hl_gauss_num_bins: int = 101
    hl_gauss_sigma_ratio: float = 0.1  # DEAS sigma / bin width

    def __post_init__(self):
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        dimensions = (
            self.backbone_dim,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.num_embodiments,
            self.projection_hidden_dim,
            self.feature_dim,
            self.num_ensembles,
        )
        if any(d <= 0 for d in dimensions) or any(d <= 0 for d in self.hidden_dims):
            raise ValueError("All network dimensions must be positive")
        if self.time_embed_dim < 2 or self.time_embed_dim % 2:
            raise ValueError("time_embed_dim must be positive and even")
        if any(
            mode not in ("mean", "min")
            for mode in (self.q_aggregation, self.inner_aggregation, self.guidance_aggregation)
        ):
            raise ValueError("Aggregation must be mean or min")
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0, 1]")
        if self.outer_update not in ("td", "iql"):
            raise ValueError("outer_update must be td or iql")
        if self.q_loss_type not in ("mse", "hl_gauss"):
            raise ValueError("q_loss_type must be mse or hl_gauss")
        if self.v_loss_type not in ("mse", "deas"):
            raise ValueError("v_loss_type must be mse or deas")
        if self.v_loss_type == "deas" and (self.outer_update != "iql" or self.q_loss_type != "hl_gauss"):
            raise ValueError("v_loss_type='deas' requires outer_update='iql' and q_loss_type='hl_gauss'")
        if not 0 < self.iql_expectile < 1:
            raise ValueError("iql_expectile must be in (0,1)")
        if self.q_loss_type == "hl_gauss":
            if (
                self.hl_gauss_min is None
                or self.hl_gauss_max is None
                or not math.isfinite(self.hl_gauss_min)
                or not math.isfinite(self.hl_gauss_max)
                or self.hl_gauss_min >= self.hl_gauss_max
            ):
                raise ValueError("HL-Gauss requires explicit finite hl_gauss_min < hl_gauss_max")
            if self.hl_gauss_num_bins < 2 or not isinstance(self.hl_gauss_num_bins, int):
                raise ValueError("hl_gauss_num_bins must be an integer >= 2")
            if not math.isfinite(self.hl_gauss_sigma_ratio) or self.hl_gauss_sigma_ratio <= 0:
                raise ValueError("hl_gauss_sigma_ratio must be finite and positive")


@dataclass(frozen=True)
class CriticObservation:
    """Reusable frozen context: pooled [B,D], state [B,S], embodiment_id [B]."""

    pooled: torch.Tensor
    state: torch.Tensor
    embodiment_id: torch.Tensor


def aggregate(values, mode):
    """Ensemble-first [E,B] -> [B]."""
    if mode == "mean":
        return values.mean(0)
    if mode == "min":
        return values.min(0).values
    raise ValueError(f"Unknown aggregation: {mode}")


def fourier_time_embed(times, dimension):
    if dimension < 2 or dimension % 2:
        raise ValueError("Fourier dimension must be positive and even")
    if times.ndim == 1:
        times = times[:, None]
    if times.ndim != 2 or times.shape[1] != 1:
        raise ValueError("times must have shape [B] or [B,1]")
    frequencies = torch.linspace(0, math.log(256), dimension // 2, device=times.device, dtype=torch.float32).exp()
    angles = times.float() * frequencies
    return torch.cat((angles.sin(), angles.cos()), -1)


@torch.no_grad()
def soft_value_target(endpoint_q, temperature):
    """lambda * logmeanexp(Q/lambda), endpoint_q [K,B], detached float32.

    Q must already be ensemble-aggregated. Endpoints must be conditional BC-SDE
    continuations from THIS (s,x_t,t), not independent actor samples from noise.
    Temperature is a positive scalar; adaptive calibration belongs to the agent.
    """
    if endpoint_q.ndim != 2 or endpoint_q.shape[0] < 1:
        raise ValueError("endpoint_q must have shape [K,B], K >= 1")
    temperature = torch.as_tensor(temperature, device=endpoint_q.device, dtype=torch.float32)
    if temperature.numel() != 1 or not torch.isfinite(temperature).all() or temperature.item() <= 0:
        raise ValueError("temperature must be a finite positive scalar")
    q = endpoint_q.float()
    # Center before division to retain stability for small temperatures.
    maximum = q.max(0).values
    return maximum + temperature * (torch.logsumexp((q - maximum) / temperature, 0) - math.log(q.shape[0]))


@torch.no_grad()
def outer_td_target(chunk_returns, bootstrap_discounts, next_q):
    """Inputs [B]; bootstrap_discounts includes gamma**H AND continuation mask.

    chunk_returns is already discounted/terminal-masked. Never sum fmrl's
    cumulative reward sequence again. Per-step LeRobot rewards need conversion
    upstream, including terminal versus time-limit truncation semantics.
    """
    if chunk_returns.ndim != 1 or not (chunk_returns.shape == bootstrap_discounts.shape == next_q.shape):
        raise ValueError("All outer TD inputs must have matching [B] shapes")
    if (
        not torch.isfinite(bootstrap_discounts).all()
        or not ((bootstrap_discounts >= 0) & (bootstrap_discounts <= 1)).all()
    ):
        raise ValueError("bootstrap_discounts must be finite and in [0,1]")
    return chunk_returns.float() + bootstrap_discounts.float() * next_q.float()


class _ValueEnsemble(nn.Module):
    def __init__(self, config, time_conditioned, *, output_dim=1, action_conditioned=True, num_members=None):
        super().__init__()
        self.config = config
        self.time_conditioned = time_conditioned
        self.action_conditioned = action_conditioned
        # Independent trainable projection per critic family, included in EMA.
        self.projection = CategorySpecificMLP(
            config.num_embodiments, config.backbone_dim, config.projection_hidden_dim, config.feature_dim
        )
        input_dim = config.feature_dim + config.state_dim
        if action_conditioned:
            input_dim += config.action_horizon * config.action_dim
        if time_conditioned:
            input_dim += config.time_embed_dim
        self.members = nn.ModuleList()
        for _ in range(config.num_ensembles if num_members is None else num_members):
            layers = []
            previous = input_dim
            for width in config.hidden_dims:
                # fmrl MLP order is Dense -> GELU -> LayerNorm.
                layers.extend(
                    (nn.Linear(previous, width), nn.GELU(approximate="tanh"), nn.LayerNorm(width, eps=1e-6))
                )
                previous = width
            layers.append(nn.Linear(previous, output_dim))
            self.members.append(nn.Sequential(*layers))

    def forward(self, observation, actions=None, times=None, action_mask=None):
        config = self.config
        batch = observation.pooled.shape[0]
        if self.action_conditioned:
            if actions is None or actions.shape != (batch, config.action_horizon, config.action_dim):
                raise ValueError("actions/x_t must match [B,action_horizon,action_dim]")
            if action_mask is not None:
                if action_mask.shape != actions.shape:
                    raise ValueError("action_mask must have the same shape as actions/x_t")
                actions = actions * action_mask.to(actions)
        parameter = next(self.projection.parameters())
        features = self.projection(
            observation.pooled.to(parameter).unsqueeze(1), observation.embodiment_id.to(device=parameter.device)
        )
        parts = [features.squeeze(1).tanh(), observation.state.to(parameter)]
        if self.action_conditioned:
            parts.append(actions.to(parameter).flatten(1))
        if self.time_conditioned:
            if times is None or times.shape[0] != batch:
                raise ValueError("inner critic requires one time per observation")
            parts.append(fourier_time_embed(times, config.time_embed_dim).to(parameter))
        x = torch.cat(parts, -1)
        return torch.stack([member(x).squeeze(-1) for member in self.members])


class SVFCriticHead(nn.Module):
    """Frozen BC LN/attention -> pooling -> Q(s,a) and V(s,x_t,t).

    The supplied BC modules are COPIED, never shared/mutated. Targets include
    each family's projection. Q and soft values are [E,B]; optional IQL V(s)
    is [B], and HL-Gauss Q logits are [E,B,N]. Default paths remain scalar TD.
    """

    def __init__(self, config, vlln, vl_self_attention):
        super().__init__()
        self.config = config
        self.vlln = deepcopy(vlln).requires_grad_(False).eval()
        self.vl_self_attention = deepcopy(vl_self_attention).requires_grad_(False).eval()
        self.hlg = None
        if config.q_loss_type == "hl_gauss":
            self.hlg = HLGaussLoss(
                config.hl_gauss_min,
                config.hl_gauss_max,
                config.hl_gauss_num_bins,
                config.hl_gauss_sigma_ratio
                * (config.hl_gauss_max - config.hl_gauss_min)
                / config.hl_gauss_num_bins,
            )
        self.critic = _ValueEnsemble(
            config, time_conditioned=False, output_dim=config.hl_gauss_num_bins if self.hlg is not None else 1
        )
        self.tc_critic = _ValueEnsemble(config, time_conditioned=True)
        self.iql_value = (
            _ValueEnsemble(
                config,
                time_conditioned=False,
                action_conditioned=False,
                num_members=1,
                output_dim=config.hl_gauss_num_bins if config.v_loss_type == "deas" else 1,
            )
            if config.outer_update == "iql"
            else None
        )
        self.target_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.target_tc_critic = deepcopy(self.tc_critic).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        for module in (self.vlln, self.vl_self_attention, self.target_critic, self.target_tc_critic):
            module.eval()
        return self

    @torch.no_grad()
    def encode_observation(self, backbone_output, state, embodiment_id):
        # Raw backbone features only: do not pass actor-mutated backbone_output.
        tokens = backbone_output["backbone_features"]
        batch = tokens.shape[0]
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.backbone_dim or tokens.shape[1] < 1:
            raise ValueError("backbone_features must be nonempty [B,T,backbone_dim]")
        state = state.reshape(batch, -1)
        if state.shape != (batch, self.config.state_dim):
            raise ValueError("Flattened state must match [B,state_dim]")
        if embodiment_id.shape != (batch,) or embodiment_id.dtype != torch.long:
            raise ValueError("embodiment_id must be int64 [B]")
        if not ((embodiment_id >= 0) & (embodiment_id < self.config.num_embodiments)).all():
            raise ValueError("embodiment_id outside configured range")
        parameter = next(self.vlln.parameters(), next(self.vl_self_attention.parameters(), None))
        if parameter is not None:
            tokens = tokens.to(parameter)
        # Match DEAS unmasked mean pooling; self-attention has no mask API.
        pooled = self.vl_self_attention(self.vlln(tokens)).mean(1)
        return CriticObservation(pooled.detach(), state.detach(), embodiment_id.detach())

    def q_values(self, observation, actions, *, target=False, action_mask=None):
        module = self.target_critic if target else self.critic
        return self._decode_q(module(observation, actions, action_mask=action_mask))

    def _decode_q(self, prediction):
        if self.hlg is None:
            return prediction
        return self.hlg.transform_from_probs(prediction.float().softmax(-1))

    def _q_predictions(self, observation, actions, action_mask):
        raw = self.critic(observation, actions, action_mask=action_mask)
        output = {"q_values": self._decode_q(raw)}
        if self.hlg is not None:
            output["q_logits"] = raw
        return output

    def state_values(self, observation):
        """Scalar IQL V(s) [B] (distribution expectation in DEAS mode)."""
        if self.iql_value is None:
            raise ValueError("state_values requires outer_update='iql'")
        return self._decode_state_value(self.iql_value(observation).squeeze(0))

    def _decode_state_value(self, prediction):
        if self.config.v_loss_type == "deas":
            return self.hlg.transform_from_probs(prediction.float().softmax(-1))
        return prediction

    def soft_values(self, observation, noisy_actions, times, *, target=False, action_mask=None):
        module = self.target_tc_critic if target else self.tc_critic
        return module(observation, noisy_actions, times, action_mask)

    def forward(self, observation, actions, noisy_actions, times, *, action_mask=None):
        output = self._q_predictions(observation, actions, action_mask)
        output["soft_values"] = self.soft_values(observation, noisy_actions, times, action_mask=action_mask)
        return output

    def q_loss(self, predictions, q_targets):
        """Q representation/loss choice is independent of the TD target builder."""
        q = predictions["q_values"].float()
        if q_targets.shape != q.shape[1:]:
            raise ValueError("Q targets must have shape [B]")
        target = q_targets.detach().to(q)
        if not torch.isfinite(target).all():
            raise ValueError("Q targets must be finite")
        if self.hlg is None:
            return {"critic_loss": (q - target.unsqueeze(0)).square().mean()}
        logits = predictions["q_logits"].float()
        # Original DEAS erf normalization can be 0/0 far outside the support.
        # Explicitly saturate targets and report how often the range is exceeded.
        bounded = target.clamp(self.config.hl_gauss_min, self.config.hl_gauss_max)
        probabilities = self.hlg.transform_to_probs(bounded)
        if not torch.isfinite(probabilities).all():
            raise ValueError("Invalid HL-Gauss probabilities; check support and sigma")
        loss = -(probabilities.unsqueeze(0) * logits.log_softmax(-1)).sum(-1).mean()
        return {"critic_loss": loss, "critic/target_clipped_fraction": (bounded != target).float().mean().detach()}

    def loss(self, predictions, q_targets, soft_targets):
        """SVF default agg-before-MSE inner distillation; independent losses."""
        q, values = predictions["q_values"].float(), predictions["soft_values"].float()
        if q_targets.shape != q.shape[1:] or soft_targets.shape != values.shape[1:]:
            raise ValueError("Targets must have shape [B]")
        metrics = self.q_loss(predictions, q_targets)
        inner_loss = F.mse_loss(aggregate(values, self.config.inner_aggregation), soft_targets.detach().to(values))
        return {**metrics, "loss": metrics["critic_loss"] + inner_loss, "inner_loss": inner_loss}

    def iql_loss(
        self,
        observation,
        next_observation,
        actions,
        chunk_returns,
        bootstrap_discounts,
        *,
        noisy_actions=None,
        times=None,
        soft_targets=None,
        action_mask=None,
    ):
        """IQL Q/V update, optionally combined with SVF inner distillation.

        V uses scalar expectile or DEAS weighted CE against min-Q's distribution;
        Q target is R + gamma^H*mask*V(s') (scalar expectation for categorical V).
        All targets use the same pre-update snapshot and are detached. No actor
        samples, advantage-weighted actor loss, or target V are used here.
        """
        if self.iql_value is None:
            raise ValueError("iql_loss requires outer_update='iql'")
        inner_args = (noisy_actions, times, soft_targets)
        if any(x is not None for x in inner_args) and not all(x is not None for x in inner_args):
            raise ValueError("Supply noisy_actions, times and soft_targets together")
        predictions = self._q_predictions(observation, actions, action_mask)
        value_prediction = self.iql_value(observation).squeeze(0)
        value = self._decode_state_value(value_prediction).float()
        with torch.no_grad():
            target_raw = self.target_critic(observation, actions, action_mask=action_mask)
            q_for_value, selected = self._decode_q(target_raw).float().min(0)
            if self.config.v_loss_type == "deas":
                # Select a whole member distribution per sample, not a binwise
                # minimum or a Gaussian re-encoding of the scalar Q expectation.
                target_probabilities = target_raw.float().softmax(-1)[
                    selected, torch.arange(selected.shape[0], device=selected.device)
                ]
            next_value = self.state_values(next_observation).float()
            q_target = outer_td_target(
                chunk_returns.to(next_value), bootstrap_discounts.to(next_value), next_value
            )
        diff = q_for_value - value
        weights = torch.where(diff >= 0, self.config.iql_expectile, 1 - self.config.iql_expectile)
        if self.config.v_loss_type == "deas":
            per_sample = -(target_probabilities * value_prediction.float().log_softmax(-1)).sum(-1)
            value_loss = (weights.detach() * per_sample).mean()
            predictions["iql_value_logits"] = value_prediction
        else:
            value_loss = (weights * diff.square()).mean()
        losses = self.q_loss(predictions, q_target)
        losses.update(
            loss=losses["critic_loss"] + value_loss,
            iql_value_loss=value_loss,
            **{
                "iql/value_mean": value.detach().mean(),
                "iql/target_q_mean": q_for_value.mean(),
                "critic/target_mean": q_target.mean(),
            },
        )
        if soft_targets is not None:
            predictions["soft_values"] = self.soft_values(
                observation, noisy_actions, times, action_mask=action_mask
            )
            values = aggregate(predictions["soft_values"].float(), self.config.inner_aggregation)
            if soft_targets.shape != values.shape:
                raise ValueError("Soft targets must have shape [B]")
            inner_loss = F.mse_loss(values, soft_targets.detach().to(values))
            losses["inner_loss"] = inner_loss
            losses["loss"] = losses["loss"] + inner_loss
        return {**predictions, **losses}

    @torch.enable_grad()
    def guidance_gradient(self, observation, noisy_actions, times, *, target=False, action_mask=None):
        """Detached guidance target dV/dx_t, usable even inside torch.no_grad().

        For higher-order/actor differentiation use soft_values directly instead.
        This does not include kappa**2 * (1-t)/(t*lambda), or the t_min gate.
        """
        x = noisy_actions.detach().requires_grad_(True)
        values = self.soft_values(observation, x, times, target=target, action_mask=action_mask)
        gradient = torch.autograd.grad(aggregate(values.float(), self.config.guidance_aggregation).sum(), x)[0]
        return gradient.detach()

    @torch.no_grad()
    def update_targets(self, tau=None):
        """Call once AFTER optimizer.step(), not per gradient accumulation microbatch."""
        tau = self.config.tau if tau is None else tau
        if not 0 < tau <= 1:
            raise ValueError("tau must be in (0,1]")
        for online, target in ((self.critic, self.target_critic), (self.tc_critic, self.target_tc_critic)):
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.lerp_(source, tau)
