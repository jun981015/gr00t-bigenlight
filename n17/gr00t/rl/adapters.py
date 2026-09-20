"""GR00T preprocessing and differentiable action-head adapters for offline RL."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gr00t.data.types import MessageType

from .types import OfflineRLBatch


def make_batch(transitions, gamma, observations, next_observations, actions, action_mask):
    if not transitions:
        raise ValueError("Cannot collate empty transitions")
    horizons = [len(sample.rewards) for sample in transitions]
    for sample in transitions:
        if not np.isclose(sample.reward_gamma, gamma, rtol=0, atol=1e-12):
            raise ValueError("Dataset and collator reward gamma differ")
    device = actions.device
    return OfflineRLBatch(
        observations,
        next_observations,
        actions.float(),
        action_mask.float(),
        torch.tensor(
            [np.dot(gamma ** np.arange(len(s.rewards)), s.rewards) for s in transitions],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor([s.discount for s in transitions], dtype=torch.float32, device=device),
        torch.tensor([s.terminated for s in transitions], dtype=torch.bool, device=device),
        torch.tensor([s.truncated for s in transitions], dtype=torch.bool, device=device),
        torch.tensor(horizons, dtype=torch.long, device=device),
    )


class StateActionTransitionCollator:
    """Normalize with GR00T StateActionProcessor; intentionally ignores images/text.

    This is a low-dimensional baseline/smoke path, NOT the GR00T VLM actor.
    """

    def __init__(self, state_action_processor, modality_configs, gamma=0.99):
        self.processor = state_action_processor
        self.configs, self.gamma = modality_configs, gamma

    def __call__(self, transitions):
        states, next_states, actions = [], [], []
        for sample in transitions:
            tag = sample.observation.embodiment.value
            configs = self.configs[tag]
            state, action = self.processor.apply(
                state=sample.observation.states,
                action=sample.observation.actions,
                embodiment_tag=tag,
            )
            following = self.processor.apply_state(sample.next_observation.states, tag)
            state_keys, action_keys = (
                configs["state"].modality_keys,
                configs["action"].modality_keys,
            )
            states.append(np.concatenate([state[key].reshape(-1) for key in state_keys]))
            next_states.append(np.concatenate([following[key].reshape(-1) for key in state_keys]))
            actions.append(np.concatenate([action[key] for key in action_keys], axis=-1))
        action_tensor = torch.from_numpy(np.stack(actions)).float()
        return make_batch(
            transitions,
            self.gamma,
            {"features": torch.from_numpy(np.stack(states)).float()},
            {"features": torch.from_numpy(np.stack(next_states)).float()},
            action_tensor,
            torch.ones_like(action_tensor),
        )


@dataclass
class ProcessedGR00TTransitions:
    # VLM pixel_values have a packed patch dimension, NOT a leading B dimension.
    # Keep these separate from the algorithm's uniformly batched observation tree.
    current: dict
    following: dict
    transitions: list
    gamma: float


class Gr00tTransitionCollator:
    """Use the original processor/collator on current and next observations separately.

    Supply a separate eval-mode processor: stochastic crops/state noise would
    change the critic state on every read, and training mode rejects actionless
    next observations. No next-action labels are fabricated or leaked.
    """

    def __init__(self, processor, gamma=0.99):
        if getattr(processor, "training", True):
            raise ValueError("Use a dedicated processor in eval mode for RL transitions")
        self.processor, self.gamma = processor, gamma
        self.collator = processor.collator

    def __call__(self, transitions):
        def process(step):
            return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])

        current = self.collator([process(s.observation) for s in transitions])["inputs"]
        following = self.collator([process(s.next_observation) for s in transitions])["inputs"]
        return ProcessedGR00TTransitions(current, following, transitions, self.gamma)


class FrozenGR00TEncoder:
    """Run the frozen VLM once per current/next batch, not per SVF flow step.

    The action head remains trainable and belongs to the algorithm. The critics
    use masked mean VLM features + normalized state + one-hot embodiment; callers
    can substitute a token-aware critic without changing the dataset or SVF.
    """

    def __init__(self, model):
        self.model = model
        model.backbone.requires_grad_(False).eval()

    @torch.no_grad()
    def encode_observation(self, inputs):
        self.model.backbone.eval()
        clean_inputs = {
            key: value for key, value in inputs.items() if key not in ("action", "action_mask")
        }
        backbone_inputs, action_inputs = self.model.prepare_input(clean_inputs)
        result = dict(self.model.backbone(backbone_inputs))
        features = result["backbone_features"].float()
        valid = result["backbone_attention_mask"].to(features.dtype).unsqueeze(-1)
        pooled = (features * valid).sum(1) / valid.sum(1).clamp_min(1)
        state = action_inputs["state"]
        embodiment = action_inputs["embodiment_id"].long()
        one_hot = F.one_hot(embodiment, self.model.config.max_num_embodiments).float()
        result.update(
            state=state.detach(),
            embodiment_id=embodiment,
            features=torch.cat((pooled, state.float().flatten(1), one_hot), -1).detach(),
        )
        # Only the raw backbone outputs are cached; trainable head projections are
        # recomputed by each actor/reference so their parameters remain independent.
        return {
            key: value.detach() for key, value in result.items() if isinstance(value, torch.Tensor)
        }

    def __call__(self, processed: ProcessedGR00TTransitions):
        observations = self.encode_observation(processed.current)
        following = self.encode_observation(processed.following)
        device = observations["features"].device
        return make_batch(
            processed.transitions,
            processed.gamma,
            observations,
            following,
            processed.current["action"].to(device),
            processed.current["action_mask"].to(device),
        )


class Gr00tFlowActor(nn.Module):
    """Expose differentiable v(s,x,t); get_action() is no_grad and cannot train SVF.

    Mirrors Gr00tN1d7ActionHead.forward's deterministic velocity path, without
    sampling its own x/t or computing an internal CFM loss. State-dropout is
    intentionally omitted so the SVF reference dynamics use a stable condition.
    """

    def __init__(self, action_head):
        super().__init__()
        self.head = action_head

    def forward(self, observations, actions, time):
        from transformers.feature_extraction_utils import BatchFeature

        head = self.head
        head.set_frozen_modules_to_eval_mode()
        if actions.shape[1:] != (head.action_horizon, head.action_dim):
            raise ValueError(
                "GR00T actor needs processor-padded [B, action_horizon, max_action_dim]"
            )
        parameter = next(head.parameters())
        backbone = BatchFeature(
            data={
                key: value.to(dtype=parameter.dtype) if value.is_floating_point() else value
                for key, value in observations.items()
                if key in ("backbone_features", "backbone_attention_mask", "image_mask")
            }
        )
        embodiment = observations["embodiment_id"]
        action_input = BatchFeature(
            data={
                "state": observations["state"].to(dtype=parameter.dtype),
                "embodiment_id": embodiment,
            }
        )
        encoded = head._encode_features(backbone, action_input)
        buckets = (time * head.num_timestep_buckets).long().clamp(0, head.num_timestep_buckets - 1)
        action_features = head.action_encoder(
            actions.to(dtype=parameter.dtype), buckets, embodiment
        )
        if head.config.add_pos_embed:
            positions = torch.arange(actions.shape[1], device=actions.device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        kwargs = {
            "hidden_states": torch.cat((encoded["state_features"], action_features), dim=1),
            "encoder_hidden_states": encoded["backbone_features"],
            "encoder_attention_mask": backbone["backbone_attention_mask"],
            "timestep": buckets,
            "return_all_hidden_states": True,
        }
        if head.config.use_alternate_vl_dit:
            kwargs.update(
                image_mask=backbone["image_mask"],
                backbone_attention_mask=backbone["backbone_attention_mask"],
            )
        hidden, _ = head.model(**kwargs)
        return head.action_decoder(hidden, embodiment)[:, -actions.shape[1] :].float()
