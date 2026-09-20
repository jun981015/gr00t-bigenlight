"""Scalar IQL Q/V on frozen GR00T features (H=1 or action-chunk transitions).

The BC actor is fixed: this module does not implement advantage-weighted actor
updates. Q/V losses are evaluated at the same parameter snapshot and summed.
"""

from copy import deepcopy

import torch
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature

from .deas_critic import DEASCritic, DEASCriticConfig


def expectile_loss(diff, expectile):
    if not 0 < expectile < 1:
        raise ValueError("expectile must be between 0 and 1")
    return (torch.where(diff > 0, expectile, 1 - expectile) * diff.square()).mean()


def chunk_target(rewards, dones, next_value, discount, reward_shift=0.0):
    """Include terminal reward, mask later rewards, and never bootstrap past done.

    dones is the per-transition terminal indicator, not a continuation mask.
    Callers must distinguish time-limit truncations upstream if bootstrapping
    across them is desired. Inputs are never modified.
    """
    if rewards.ndim != 2 or rewards.shape != dones.shape or rewards.shape[1] < 1:
        raise ValueError("rewards and dones must have matching [B,H] shapes with H >= 1")
    if next_value.shape != rewards.shape[:1]:
        raise ValueError("next_value must have shape [B]")
    if not 0 < discount < 1:
        raise ValueError("discount must be between 0 and 1")
    rewards = rewards.float() + reward_shift
    continuation = 1 - dones.to(dtype=torch.float32).clamp(0, 1)
    alive = torch.cat([torch.ones_like(continuation[:, :1]), continuation[:, :-1]], dim=1).cumprod(1)
    discounts = discount ** torch.arange(rewards.shape[1], device=rewards.device, dtype=torch.float32)
    returns = (rewards * alive * discounts).sum(1)
    bootstrap = continuation.prod(1) * discount ** rewards.shape[1]
    return returns + bootstrap * next_value.float()


class IQLCritic(DEASCritic):
    """Shared trainable bottleneck + scalar Q1/Q2/V, with a full EMA Q encoder.

    Names match the existing actor+critic loader. Legacy DEAS checkpoints keep
    their original head; config.rl_config['algorithm']='iql' selects this head.
    """

    def __init__(self, config: DEASCriticConfig):
        config.rl_config = dict(config.rl_config, algorithm="iql", num_atoms=1)
        if config.rl_config.get("nstep", 1) != 1:
            raise ValueError("IQL expects next observations at t+H; nstep must be 1")
        if config.expand_batch not in (None, 1):
            raise ValueError("IQL does not expand/repeat input batches")
        if config.rl_config.get("q_agg", "min") != "min":
            raise ValueError("IQL uses the minimum of target Q1 and Q2")
        super().__init__(config)
        self.discount = config.rl_config.get("discount", 0.99)
        if not 0 < self.discount < 1 or not 0 < self.rl_config.expectile < 1:
            raise ValueError("discount and expectile must be between 0 and 1")
        self.target_backbone_encoder = deepcopy(self.backbone_encoder).requires_grad_(False)
        self.target_backbone_encoder.eval()

    def set_trainable_parameters(self, tune_value, tune_critic):
        super().set_trainable_parameters(tune_value, tune_critic)
        if hasattr(self, "target_backbone_encoder"):
            self.target_backbone_encoder.requires_grad_(False)

    def process_backbone_output(self, backbone_output):
        # A new container prevents both feature overwrite and double processing.
        # no_grad also avoids allocating graphs through the frozen VLM features.
        self.vlln.eval()
        self.vl_self_attention.eval()
        with torch.no_grad():
            features = self.vl_self_attention(self.vlln(backbone_output.backbone_features))
        return BatchFeature(data={**dict(backbone_output), "backbone_features": features})

    def forward(self, backbone_output, next_backbone_output, action_input):
        self.set_frozen_modules_to_eval_mode()
        self.target_critic.eval()
        self.target_backbone_encoder.eval()
        current = self.process_backbone_output(backbone_output).backbone_features.mean(1, keepdim=True)
        following = self.process_backbone_output(next_backbone_output).backbone_features.mean(1, keepdim=True)
        ids = action_input.embodiment_id
        features = torch.tanh(self.backbone_encoder(current, ids))
        actions = action_input.action[:, :self.critic_action_horizon]
        if actions.shape[1] != self.critic_action_horizon:
            raise ValueError("Action chunk is shorter than critic_action_horizon")
        if action_input.reward.shape != action_input.done.shape or action_input.reward.shape != actions.shape[:2]:
            raise ValueError("reward/done must match the [B,H] action chunk")

        with torch.no_grad():
            target_features = torch.tanh(self.target_backbone_encoder(current, ids))
            tq1, tq2 = self.target_critic(target_features, action_input.state, actions)
            target_q = torch.minimum(tq1, tq2).float()
            next_features = torch.tanh(self.backbone_encoder(following, ids))
            next_value = self.value(next_features, action_input.next_state).reshape(-1).float()
            target = chunk_target(action_input.reward, action_input.done, next_value,
                                  self.discount, -1.0 if self.rl_config.negative_reward else 0.0)

        value = self.value(features, action_input.state).reshape(-1).float()
        q1, q2 = self.critic(features, action_input.state, actions)
        q1, q2 = q1.float(), q2.float()
        value_loss = expectile_loss(target_q - value, self.rl_config.expectile)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        return BatchFeature(data={
            "loss": value_loss + critic_loss,
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "value/v_mean": value.detach().mean(),
            "value/target_q_mean": target_q.mean(),
            "critic/q1_mean": q1.detach().mean(),
            "critic/q2_mean": q2.detach().mean(),
            "critic/target_mean": target.mean(),
        })
