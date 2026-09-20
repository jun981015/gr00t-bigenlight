"""Algorithm-independent, explicit semi-Markov transition contracts."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

import numpy as np
import torch

from gr00t.data.types import VLAStepData


@dataclass(frozen=True)
class RLTransition:
    observation: VLAStepData
    next_observation: VLAStepData
    rewards: np.ndarray  # Per-action rewards, not already discounted.
    terminated: bool
    truncated: bool
    bootstrap_mask: float
    discount: float  # bootstrap_gamma ** horizon * mask; DO NOT multiply gamma again.
    episode_index: int
    step_index: int
    reward_gamma: float  # Independent within-chunk discount; collator must match.
    next_observation_valid: bool = True


def map_tensors(tree: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return {key: map_tensors(value, fn) for key, value in tree.items()}
    raise TypeError(f"Observation trees must contain tensors/dicts, got {type(tree).__name__}")


@dataclass(frozen=True)
class OfflineRLBatch:
    observations: dict[str, torch.Tensor]
    next_observations: dict[str, torch.Tensor]
    actions: torch.Tensor  # [B, H, A], in the policy's normalized action coordinates.
    action_mask: torch.Tensor  # Same shape; padding never enters flow/critic objectives.
    rewards: torch.Tensor  # [B], sum_i gamma**i * reward_i.
    discounts: torch.Tensor  # [B], COMPLETE bootstrap multiplier.
    terminated: torch.Tensor
    truncated: torch.Tensor
    horizons: torch.Tensor

    def to(self, device: str | torch.device) -> "OfflineRLBatch":
        return replace(
            self,
            **{
                key: map_tensors(value, lambda x: x.to(device)) for key, value in vars(self).items()
            },
        )

    def validate(self) -> None:
        if self.actions.ndim != 3 or self.actions.shape != self.action_mask.shape:
            raise ValueError("actions/action_mask must have the same [B, H, A] shape")
        size = self.actions.shape[0]
        if size == 0:
            raise ValueError("Empty RL batch")
        if not self.actions.is_floating_point():
            raise ValueError("Normalized actions must be floating point")
        for key in ("rewards", "discounts", "terminated", "truncated", "horizons"):
            if getattr(self, key).shape != (size,):
                raise ValueError(f"{key} must have shape [B]")
        for key in ("actions", "action_mask", "rewards", "discounts"):
            if not torch.isfinite(getattr(self, key)).all():
                raise ValueError(f"Non-finite {key}")
        if not ((self.action_mask == 0) | (self.action_mask == 1)).all():
            raise ValueError("action_mask must be binary")
        if (self.action_mask.flatten(1).sum(1) == 0).any():
            raise ValueError("Every sample must have a valid action")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise ValueError("terminated/truncated must be boolean")
        if (self.terminated & self.truncated).any():
            raise ValueError("A transition cannot be both terminated and truncated")
        if self.horizons.dtype not in (torch.int32, torch.int64):
            raise ValueError("horizons must be integer")
        valid_timesteps = self.action_mask.bool().any(-1)
        if not torch.equal(valid_timesteps.sum(-1), self.horizons):
            raise ValueError("horizons must count exactly the non-padded action timesteps")
        prefix = (
            torch.arange(self.actions.shape[1], device=self.actions.device)[None]
            < self.horizons[:, None]
        )
        if not torch.equal(valid_timesteps, prefix):
            raise ValueError("Valid action timesteps must form a contiguous prefix")
        if not ((self.discounts >= 0) & (self.discounts <= 1)).all():
            raise ValueError("discounts must be in [0, 1]")
        if (self.discounts[self.terminated.bool()] != 0).any():
            raise ValueError("True terminals cannot bootstrap")
        for tree in (self.observations, self.next_observations):

            def check(x):
                if x.ndim == 0 or x.shape[0] != size:
                    raise ValueError("Observation tensors must have batch as their first dimension")
                if x.is_floating_point() and not torch.isfinite(x).all():
                    raise ValueError("Non-finite observation")
                return x

            map_tensors(tree, check)


class OfflineAlgorithm(Protocol):
    """Inject another algorithm without coupling it to LeRobot or a particular encoder."""

    def update(self, batch: OfflineRLBatch) -> dict[str, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
