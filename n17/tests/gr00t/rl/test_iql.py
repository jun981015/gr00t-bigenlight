from copy import deepcopy
from dataclasses import replace
import json

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.rl.adapters import FrozenBCGR00TEncoder, StateActionTransitionCollator
from gr00t.rl.dataset import (
    AllSuccessStepCostAnnotations,
    AllSuccessTerminalAnnotations,
    LeRobotOfflineRLDataset,
)
from gr00t.rl.iql import FeatureValue, IQLConfig, IQLCriticLearner
from gr00t.rl.networks import FeatureCritic
from gr00t.rl.train import main
from gr00t.rl.types import OfflineRLBatch
import numpy as np
import pytest
import torch

from .test_algorithms import batch
from .test_dataset import TAG, IdentityNormalizer, MemoryLoader, modalities, write_lerobot
from .test_gr00t_adapter import TinyModel, observations


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def learner(feature_dim=5, shape=(3, 3)):
    return IQLCriticLearner(
        FeatureCritic(feature_dim, shape, (16,)), FeatureValue(feature_dim, (16,)), IQLConfig()
    )


def test_iql_formulas_masking_ema_and_resume():
    data, agent = batch(), learner()
    before = deepcopy(agent.target_critic.state_dict())
    with torch.no_grad():
        target_q = (
            agent.target_critic(data.observations, data.actions * data.action_mask).min(0).values
        )
        v = agent.value(data.observations)
        residual = target_q - v
        expected_v = (torch.where(residual > 0, 0.7, 0.3) * residual.square()).mean()
        target = data.rewards + data.discounts * agent.value(data.next_observations)
        assert torch.equal(target[data.terminated], data.rewards[data.terminated])
        expected_q = (
            (agent.critic(data.observations, data.actions * data.action_mask) - target)
            .square()
            .mean()
        )
    identical = learner()
    identical.load_state_dict(deepcopy(agent.state_dict()))
    metrics = agent.update(data)
    altered = replace(data, actions=data.actions + (1 - data.action_mask) * 100)
    masked_metrics = identical.update(altered)
    assert metrics == pytest.approx(masked_metrics)
    assert metrics["loss/v"] == pytest.approx(expected_v.item())
    assert metrics["loss/q"] == pytest.approx(expected_q.item())
    assert metrics["critic/loss"] == metrics["loss/q"]
    assert metrics["critic/abs_td_loss"] >= 0
    for prefix in ("q", "v"):
        assert metrics[f"{prefix}/min"] <= metrics[f"{prefix}/mean"] <= metrics[f"{prefix}/max"]
    assert metrics["critic/grad_norm"] > 0 and metrics["value/grad_norm"] > 0
    assert metrics["grad_norm"] ** 2 == pytest.approx(
        metrics["critic/grad_norm"] ** 2 + metrics["value/grad_norm"] ** 2, rel=1e-5
    )
    for key, value in agent.target_critic.state_dict().items():
        torch.testing.assert_close(value, before[key].lerp(agent.critic.state_dict()[key], 0.005))
    assert all(p.grad is None for p in agent.target_critic.parameters())
    resumed = learner()
    resumed.load_state_dict(deepcopy(agent.state_dict()))
    assert agent.update(data) == pytest.approx(resumed.update(data))
    assert resumed.updates == 2
    assert set(agent.state_dict()) == {
        "algorithm",
        "config",
        "critic",
        "value",
        "target_critic",
        "optimizer",
        "updates",
    }


def test_success_assumption_and_terminal_chunk():
    loader = MemoryLoader((5,))
    ds = LeRobotOfflineRLDataset(loader, TAG, AllSuccessTerminalAnnotations(), horizon=2, gamma=0.9)
    labels = ds.annotations[0]
    np.testing.assert_array_equal(labels.rewards, [0, 0, 0, 0, 1])
    first, last = ds[0], ds[len(ds) - 1]
    assert first.discount == pytest.approx(0.9**2)
    assert last.terminated and last.discount == 0 and not last.next_observation_valid
    np.testing.assert_array_equal(last.rewards, [0, 1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_bc_attention_once_and_no_dit_or_feature_overwrite(dtype):
    model = TinyModel().to(dtype=dtype)
    encoder = FrozenBCGR00TEncoder(model)
    before = deepcopy(model.state_dict())
    obs = observations()
    raw = obs["backbone_features"].clone()
    calls = []
    hook = model.action_head.vl_self_attention.register_forward_hook(lambda *args: calls.append(1))

    def forbid_dit(*args):
        raise AssertionError("Critic training must not run DiT")

    dit_hook = model.action_head.model.register_forward_pre_hook(forbid_dit)
    try:
        encoded = encoder.encode_observation(obs)
        following = encoder.encode_observation(obs)
        assert len(calls) == 2 and model.backbone.calls == 2
        torch.testing.assert_close(encoded["backbone_features"], raw)
        torch.testing.assert_close(obs["backbone_features"], raw)
        torch.testing.assert_close(encoded["features"], following["features"])
        assert not any(p.requires_grad for p in model.parameters())
        data = OfflineRLBatch(
            encoded,
            following,
            torch.zeros(2, 3, 2),
            torch.ones(2, 3, 2),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.99**3, 0.0]),
            torch.tensor([False, True]),
            torch.zeros(2, dtype=torch.bool),
            torch.full((2,), 3),
        )
        agent = learner(71, (3, 2))
        metrics = agent.update(data)
        assert all(np.isfinite(v) for v in metrics.values())
        assert len(calls) == 2 and model.backbone.calls == 2
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters())
    finally:
        hook.remove()
        dit_hook.remove()


@pytest.mark.parametrize("annotation_format", ["all-success-terminal", "all-success-step-cost"])
def test_iql_cli_success_labels_resume(tmp_path, monkeypatch, annotation_format):
    data, output = tmp_path / "data", tmp_path / "run"
    write_lerobot(data)
    monkeypatch.setitem(MODALITY_CONFIGS, TAG.value, modalities())
    args = [
        "--dataset-path",
        str(data),
        "--output-dir",
        str(output),
        "--algorithm",
        "iql",
        "--annotation-format",
        annotation_format,
        "--horizon",
        "2",
        "--batch-size",
        "2",
        "--hidden-dim",
        "8",
        "--hidden-layers",
        "1",
        "--cpu-threads",
        "2",
    ]
    main([*args, "--steps", "2"])
    checkpoint = output / "checkpoints/step-2.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["algorithm"]["algorithm"] == "iql-critic-only-v1"
    provenance = json.loads((output / "reward_assumption.json").read_text())
    assert provenance["source_data_modified"] is False
    assert len(provenance["episodes"]) == 2
    main([*args, "--steps", "3", "--resume", str(checkpoint)])
    assert (output / "checkpoints/step-3.pt").is_file()


def test_step_cost_is_per_action_and_terminal_zero():
    source = MemoryLoader((5,))
    positive = AllSuccessTerminalAnnotations()(source, 0)
    ds = LeRobotOfflineRLDataset(source, TAG, AllSuccessStepCostAnnotations(), horizon=2, gamma=0.9)
    np.testing.assert_array_equal(ds.annotations[0].rewards, [-1, -1, -1, -1, 0])
    np.testing.assert_array_equal(positive.rewards, [0, 0, 0, 0, 1])
    collator = StateActionTransitionCollator(
        IdentityNormalizer(), {TAG.value: modalities()}, gamma=0.9
    )
    data = collator([ds[0], ds[len(ds) - 1]])
    torch.testing.assert_close(data.rewards, torch.tensor([-1.9, -1.0]))
    torch.testing.assert_close(data.discounts, torch.tensor([0.81, 0.0]))


@pytest.mark.parametrize(
    "kwargs", [{"expectile": 1}, {"target_tau": 0}, {"learning_rate": float("nan")}]
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        IQLConfig(**kwargs)
