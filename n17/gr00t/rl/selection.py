"""Optional critic-based policy extraction, independent of the RL update rule.

Inspired by DEAS's GR00T best-of-N path. This is a generic selector, not a DEAS
critic implementation or a robot-serving wrapper. SVF's default actor is unchanged.
"""

from dataclasses import dataclass
import math

import torch

from .algorithms import aggregate_heads, evaluating


@dataclass(frozen=True)
class CandidateSelection:
    actions: torch.Tensor  # [B,H,A], still normalized and padded.
    scores: torch.Tensor  # [N,B], ensemble-aggregated candidate Q values.
    indices: torch.Tensor  # [B], selected candidate for each observation.


@torch.no_grad()
def select_action_candidates(
    sample_actions,
    critic,
    observations,
    action_mask,
    *,
    num_candidates=10,
    aggregation="min",
    temperature=0.0,
):
    """Sample independent chunks; greedily rank or sample softmax(Q/temperature).

    sample_actions(observations, action_mask) must handle its own evaluation mode.
    Critic returns scalar Q [ensemble,B] (convert distributional logits first).
    Features are reused and candidates evaluated sequentially to bound VRAM.
    The mask/horizon/normalization must match how this critic was trained.
    """
    if not isinstance(num_candidates, int) or num_candidates < 1:
        raise ValueError("num_candidates must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if aggregation not in ("min", "mean"):
        raise ValueError("aggregation must be min or mean")
    if action_mask.ndim != 3 or action_mask.shape[0] == 0:
        raise ValueError("action_mask must have shape [B,H,A] with B>0")
    if not ((action_mask == 0) | (action_mask == 1)).all():
        raise ValueError("action_mask must be binary")
    if (action_mask.flatten(1).sum(1) == 0).any():
        raise ValueError("Each observation needs a valid action")
    candidates, scores = [], []
    with evaluating(critic):
        for _ in range(num_candidates):
            action = sample_actions(observations, action_mask)
            if action.shape != action_mask.shape or not torch.isfinite(action).all():
                raise ValueError("Sampler must return finite actions with action_mask's shape")
            action = action * action_mask
            score = aggregate_heads(critic(observations, action), aggregation)
            if score.shape != (action.shape[0],) or not torch.isfinite(score).all():
                raise ValueError("Critic must return a finite scalar score per observation")
            candidates.append(action)
            scores.append(score)
    candidates, scores = torch.stack(candidates), torch.stack(scores)
    if temperature == 0:
        indices = scores.argmax(0)
    else:
        logits = (scores - scores.max(0).values) / temperature
        indices = torch.multinomial(logits.softmax(0).T, 1).squeeze(-1)
    actions = candidates[indices, torch.arange(action_mask.shape[0], device=indices.device)]
    return CandidateSelection(actions, scores, indices)
