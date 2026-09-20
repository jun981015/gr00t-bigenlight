"""Actual tiny GR00T DiT/head tests, without loading pretrained VLM weights."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.rl.adapters import FrozenGR00TEncoder, Gr00tFlowActor, Gr00tTransitionCollator
from gr00t.rl.algorithms import SoftValueFlow, SVFConfig
from gr00t.rl.dataset import LeRobotOfflineRLDataset
from gr00t.rl.networks import FeatureCritic
from gr00t.rl.types import OfflineRLBatch
import numpy as np
import pytest
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .test_dataset import TAG, MemoryLoader, terminal_labels


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def small_head(alternate=False):
    config = Gr00tN1d7Config(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=3,
        max_action_dim=2,
        action_horizon=3,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=alternate,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    return Gr00tN1d7ActionHead(config).eval()


def observations():
    return {
        "backbone_features": torch.randn(2, 8, 64),
        "backbone_attention_mask": torch.ones(2, 8, dtype=torch.bool),
        "image_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]] * 2, dtype=torch.bool),
        "state": torch.randn(2, 1, 3),
        "embodiment_id": torch.tensor([0, 1]),
        "features": torch.randn(2, 71),
    }


@pytest.mark.parametrize("alternate", [False, True])
def test_velocity_matches_original_forward_and_keeps_gradients(monkeypatch, alternate):
    head = small_head(alternate)
    actor = Gr00tFlowActor(head)
    obs = observations()
    original_features = obs["backbone_features"].clone()
    target = torch.randn(2, 3, 2)
    noise = torch.randn_like(target)
    time = torch.full((2,), 0.35)
    noisy = ((1 - time[:, None, None]) * noise + time[:, None, None] * target).requires_grad_(True)
    prediction = actor(obs, noisy, time)
    monkeypatch.setattr(head, "sample_time", lambda *args, **kwargs: time)
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: noise.clone())
    backbone = BatchFeature(
        data={
            key: value.clone()
            for key, value in obs.items()
            if key in ("backbone_features", "backbone_attention_mask", "image_mask")
        }
    )
    inputs = BatchFeature(
        data={
            "state": obs["state"],
            "embodiment_id": obs["embodiment_id"],
            "action": target,
            "action_mask": torch.ones_like(target),
        }
    )
    original = head(backbone, inputs)
    torch.testing.assert_close(original["loss"], (prediction - (target - noise)).square().mean())
    prediction.square().mean().backward()
    assert noisy.grad is not None and noisy.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.model.parameters())
    torch.testing.assert_close(obs["backbone_features"], original_features)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return BatchFeature(
            data={
                "backbone_features": inputs["backbone_features"] * self.scale,
                "backbone_attention_mask": inputs["backbone_attention_mask"],
                "image_mask": inputs["image_mask"],
            }
        )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.action_head = small_head(True)
        self.config = SimpleNamespace(max_num_embodiments=4)

    def prepare_input(self, inputs):
        assert "action" not in inputs and "action_mask" not in inputs
        return inputs, {"state": inputs["state"], "embodiment_id": inputs["embodiment_id"]}


def test_actual_gr00t_head_svf_step_with_frozen_backbone():
    model = TinyModel()
    encoder = FrozenGR00TEncoder(model)
    obs = observations()
    encoded = encoder.encode_observation(obs)
    following = encoder.encode_observation(observations())
    assert model.backbone.calls == 2
    assert not any(value.requires_grad for value in encoded.values())
    assert encoded["features"].shape == (2, 71)
    before = deepcopy(model.action_head.state_dict())
    learner = SoftValueFlow(
        Gr00tFlowActor(model.action_head),
        FeatureCritic(71, (3, 2), (16,)),
        FeatureCritic(71, (3, 2), (16,), time_embed_dim=16),
        SVFConfig(flow_steps=2, candidates=2, soft_lambda=1.0),
    )
    data = OfflineRLBatch(
        encoded,
        following,
        torch.randn(2, 3, 2).clamp(-1, 1),
        torch.ones(2, 3, 2),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.99**3, 0.0]),
        torch.tensor([False, True]),
        torch.zeros(2, dtype=torch.bool),
        torch.full((2,), 3),
    )
    metrics = learner.update(data)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert model.backbone.calls == 2, "SVF inner rollouts must not rerun the VLM"
    assert model.backbone.scale.grad is None and model.backbone.scale.item() == 1
    assert any(
        not torch.equal(value, before[key]) for key, value in model.action_head.state_dict().items()
    )


def test_real_gr00t_processor_handles_short_rl_chunks(monkeypatch):
    """Only the external text/image tokenizer is stubbed; all GR00T processing is real."""
    from gr00t.data.types import ModalityConfig
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    class Tokenizer:
        tokenizer = SimpleNamespace(padding_side="left")

        def apply_chat_template(self, *args, **kwargs):
            return "test task"

        def __call__(self, *, text, images, **kwargs):
            return {
                "input_ids": torch.ones(len(text), 8, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 8, dtype=torch.long),
                "pixel_values": torch.zeros(len(images) * 5, 64),
            }

    monkeypatch.setattr(processing, "build_processor", lambda *args: Tokenizer())
    loader = MemoryLoader((5,))
    configs = {**loader.modality_configs, "video": ModalityConfig([0], ["camera"])}
    stats = {"min": [-10.0], "max": [10.0], "mean": [0.0], "std": [1.0]}
    processor = processing.Gr00tN1d7Processor(
        {TAG.value: configs},
        statistics={TAG.value: {"state": {"joint": stats}, "action": {"joint": stats}}},
        max_state_dim=3,
        max_action_dim=2,
        max_action_horizon=4,
        image_target_size=[32, 32],
        image_crop_size=[32, 32],
        embodiment_id_mapping={TAG.value: 0},
    )
    with pytest.raises(ValueError, match="eval mode"):
        Gr00tTransitionCollator(processor)
    processor.eval()
    data = LeRobotOfflineRLDataset(loader, TAG, terminal_labels, horizon=2)
    sample = data[0]
    images = {"camera": [np.zeros((32, 32, 3), dtype=np.uint8)]}
    sample = replace(
        sample,
        observation=replace(sample.observation, images=images),
        next_observation=replace(sample.next_observation, images=images),
    )
    processed = Gr00tTransitionCollator(processor)([sample, sample])
    assert "action" not in processed.following and "action_mask" not in processed.following
    assert processed.current["action"].shape == (2, 4, 2)
    expected_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(processed.current["action_mask"], expected_mask.expand(2, -1, -1))
    torch.testing.assert_close(processed.following["state"][:, :, 0], torch.full((2, 1), 0.2))
    assert processed.current["pixel_values"].shape == (10, 64)


def test_collator_next_observation_is_actionless_and_packed_inputs_are_encoded():
    class Processor:
        training = False

        def __init__(self):
            self.seen_actions = []

        def __call__(self, messages):
            step = messages[0]["content"]
            self.seen_actions.append(bool(step.actions))
            item = {
                "state": torch.nn.functional.pad(torch.from_numpy(step.states["joint"]), (0, 2)),
                "embodiment_id": torch.tensor(0),
            }
            if step.actions:
                item["action"] = torch.nn.functional.pad(
                    torch.from_numpy(step.actions["joint"]), (0, 1, 0, 1)
                )
                item["action_mask"] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
            return item

        def collator(self, items):
            result = {key: torch.stack([item[key] for item in items]) for key in items[0]}
            count = len(items)
            result.update(
                backbone_features=torch.ones(count, 8, 64),
                backbone_attention_mask=torch.ones(count, 8, dtype=torch.bool),
                image_mask=torch.ones(count, 8, dtype=torch.bool),
                pixel_values=torch.zeros(count * 5, 64),
            )
            return {"inputs": result}

    processor = Processor()
    dataset = LeRobotOfflineRLDataset(MemoryLoader((5,)), TAG, terminal_labels, horizon=2)
    collator = Gr00tTransitionCollator(processor)
    processed = collator([dataset[0], dataset[3]])
    assert processor.seen_actions == [True, True, False, False]
    assert "action" not in processed.following
    assert processed.current["pixel_values"].shape[0] != 2
    model = TinyModel()
    encoded = FrozenGR00TEncoder(model)(processed)
    encoded.validate()
    assert "action" not in encoded.observations
    assert encoded.actions.shape == (2, 3, 2)
    assert encoded.discounts[-1] == 0
    assert model.backbone.calls == 2
