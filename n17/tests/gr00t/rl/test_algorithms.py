from copy import deepcopy
from dataclasses import replace

from gr00t.rl.algorithms import FlowBC, SoftValueFlow, SVFConfig, soft_value, svf_coefficient
from gr00t.rl.networks import FeatureCritic, FeatureFlowActor, fourier_time_embed
from gr00t.rl.trainer import OfflineTrainer
from gr00t.rl.types import OfflineRLBatch
import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch():
    actions = torch.randn(4, 3, 3).clamp(-1, 1)
    mask = torch.ones_like(actions)
    mask[:, 2] = 0
    mask[:, :, 2] = 0
    return OfflineRLBatch(
        {"features": torch.randn(4, 5)},
        {"features": torch.randn(4, 5)},
        actions,
        mask,
        torch.tensor([0.0, 1.0, -1.0, 0.5]),
        torch.tensor([0.99**2, 0.0, 0.99**2, 0.0]),
        torch.tensor([False, True, False, True]),
        torch.tensor([False, False, True, False]),
        torch.full((4,), 2),
    )


def agent(**kwargs):
    config = SVFConfig(flow_steps=2, candidates=2, soft_lambda=0.5, **kwargs)
    return SoftValueFlow(
        FeatureFlowActor(5, (3, 3), (16,)),
        FeatureCritic(5, (3, 3), (16,)),
        FeatureCritic(5, (3, 3), (16,), time_embed_dim=16),
        config,
    )


def test_soft_value_and_time_formula():
    q = torch.tensor([[1.0, 3.0], [2.0, 3.0]], dtype=torch.float64)
    expected = 0.5 * torch.log(torch.exp(q / 0.5).mean(0))
    torch.testing.assert_close(soft_value(q, 0.5), expected)
    assert soft_value(q, 0.5)[1] == 3
    times = torch.tensor([0.0, 0.05, 0.1, 0.5, 1.0])
    torch.testing.assert_close(
        svf_coefficient(times, 2.0, 0.6, 0.1), torch.tensor([0, 0, 1.62, 0.18, 0])
    )
    torch.testing.assert_close(
        fourier_time_embed(torch.zeros(2), 4), torch.tensor([[0.0, 0.0, 1.0, 1.0]]).expand(2, -1)
    )
    with pytest.raises(ValueError):
        soft_value(q, 0)


class ConstantFlow(nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, observations, actions, time):
        return self.value.expand_as(actions)


def test_reference_sde_deterministic_limit_and_t_one():
    model = SoftValueFlow(
        ConstantFlow(),
        FeatureCritic(5, (3, 3), (8,)),
        FeatureCritic(5, (3, 3), (8,), time_embed_dim=16),
        SVFConfig(kappa=0, action_clip=None, flow_steps=10, candidates=3),
    )
    data = batch()
    x = torch.zeros_like(data.actions)
    time = torch.tensor([0.1, 0.33, 0.9, 1.0])
    endpoints = model.base_sde_endpoints(data.observations, x, time, data.action_mask)
    expected = (1 - time[:, None, None]) * 0.2 * data.action_mask
    torch.testing.assert_close(endpoints, expected.unsqueeze(0).expand(3, -1, -1, -1))
    assert not endpoints.requires_grad
    with pytest.raises(ValueError, match="t_min"):
        model.base_sde_endpoints(data.observations, x, torch.zeros(4), data.action_mask)


def test_nonzero_kappa_euler_maruyama_formula(monkeypatch):
    model = SoftValueFlow(
        ConstantFlow(),
        FeatureCritic(5, (3, 3), (8,)),
        FeatureCritic(5, (3, 3), (8,), time_embed_dim=16),
        SVFConfig(kappa=0.6, action_clip=None, flow_steps=1, candidates=2),
    )
    data = batch()
    x = torch.full_like(data.actions, 0.3) * data.action_mask
    time = torch.tensor([0.5, 0.8, 0.9, 1.0])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    result = model.base_sde_endpoints(data.observations, x, time, data.action_mask)
    t = time[:, None, None]
    dt = 1 - t
    u = 0.2 * data.action_mask
    expected = (
        x + (u - 0.6**2 * (x - t * u) / t) * dt + 0.6 * (2 * (1 - t) / t * dt).sqrt()
    ) * data.action_mask
    torch.testing.assert_close(result, expected.unsqueeze(0).expand(2, -1, -1, -1))


def test_td_uses_complete_discount_without_second_gamma(monkeypatch):
    model = agent()
    data = batch()
    monkeypatch.setattr(
        model, "target_critic", lambda obs, action: torch.full((2, action.shape[0]), 3.0)
    )
    _, metrics = model.losses(data)
    torch.testing.assert_close(
        metrics["critic/td_target_mean"], (data.rewards + 3 * data.discounts).mean()
    )


def test_temperature_floor_and_independent_draws(monkeypatch):
    model = agent()
    model.config = replace(model.config, soft_lambda=None, lambda_multiplier=4)
    seen = []
    original = model.base_sde_endpoints

    def capture(obs, noisy, time, mask):
        seen.append((noisy.clone(), time.clone()))
        return original(obs, noisy, time, mask)

    monkeypatch.setattr(model, "base_sde_endpoints", capture)
    monkeypatch.setattr(model, "_endpoint_q", lambda obs, ends: torch.zeros(ends.shape[:2]))
    data = batch()
    loss, info = model.losses(data)
    assert info["sv/lambda"].item() == pytest.approx(0.004)
    assert len(seen) == 2 and not torch.equal(seen[0][0], seen[1][0])
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "objective,updated",
    [
        ("actor/loss", "actor"),
        ("critic/outer_loss", "critic"),
        ("critic/inner_loss", "inner_critic"),
        ("actor/bc_flow_loss", "reference"),
    ],
)
def test_gradient_isolation(objective, updated):
    model = agent()
    _, metrics = model.losses(batch())
    metrics[objective].backward()
    for name in (
        "actor",
        "critic",
        "inner_critic",
        "reference",
        "target_critic",
        "target_inner_critic",
    ):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()
        )
        assert has_grad == (name == updated), name


def test_padding_does_not_change_losses_or_sampled_actions():
    model = agent()
    data = batch()
    changed = replace(data, actions=data.actions + (1 - data.action_mask) * 10000)
    torch.manual_seed(11)
    loss, _ = model.losses(data)
    torch.manual_seed(11)
    changed_loss, _ = model.losses(changed)
    torch.testing.assert_close(loss, changed_loss)
    sampled = model.sample_actions(data.observations, data.action_mask)
    assert torch.all(sampled[data.action_mask == 0] == 0)
    assert sampled.abs().max() <= 1


def test_joint_update_freeze_and_pre_update_target_ema():
    model = agent(freeze_reference=True, tau=0.2)
    data = batch()
    before = {
        name: deepcopy(getattr(model, name).state_dict())
        for name in ("actor", "critic", "inner_critic", "reference")
    }
    for parameter in model.target_critic.parameters():
        parameter.zero_()
    metrics = model.update(data)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    for name in ("actor", "critic", "inner_critic"):
        assert any(
            not torch.equal(value, before[name][key])
            for key, value in getattr(model, name).state_dict().items()
        )
    for key, value in model.reference.state_dict().items():
        torch.testing.assert_close(value, before["reference"][key], rtol=0, atol=0)
    for key, value in model.target_critic.state_dict().items():
        torch.testing.assert_close(value, before["critic"][key] * 0.2)
    assert all(p.grad is None for p in model.reference.parameters())


def test_checkpoint_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(31)
    data = batch()
    first = OfflineTrainer(agent())
    first.update(data)
    checkpoint = tmp_path / "state.pt"
    first.save_checkpoint(checkpoint)
    with pytest.raises(FileExistsError):
        first.save_checkpoint(checkpoint)
    expected_metrics = first.update(data)
    second = OfflineTrainer(agent())
    second.load_checkpoint(checkpoint)
    actual_metrics = second.update(data)
    assert second.step == first.step == 2
    assert actual_metrics == expected_metrics
    for name in (
        "actor",
        "reference",
        "critic",
        "inner_critic",
        "target_critic",
        "target_inner_critic",
    ):
        for key, value in getattr(first.algorithm, name).state_dict().items():
            torch.testing.assert_close(
                value, getattr(second.algorithm, name).state_dict()[key], rtol=0, atol=0
            )


def test_bc_uses_same_trainer_and_rejects_nonfinite_batch():
    learner = FlowBC(FeatureFlowActor(5, (3, 3), (16,)), SVFConfig(flow_steps=2))
    trainer = OfflineTrainer(learner)
    data = batch()
    assert trainer.update(data)["loss"] > 0
    broken = replace(data, rewards=torch.full((4,), float("nan")))
    with pytest.raises(ValueError, match="Non-finite"):
        trainer.update(broken)
    assert trainer.step == 1


def test_parameter_sharing_and_config_errors_rejected():
    model = agent()
    with pytest.raises(ValueError, match="share"):
        SoftValueFlow(model.actor, model.critic, model.inner_critic, reference=model.actor)
    for values in (
        {"candidates": 0},
        {"soft_lambda": 0},
        {"kappa": float("nan")},
        {"t_min": 0},
        {"q_aggregation": "invalid"},
    ):
        with pytest.raises(ValueError):
            SVFConfig(**values)
