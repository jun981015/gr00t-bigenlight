from copy import deepcopy
from dataclasses import replace
import json

from gr00t.rl.action_conditioned_iql import (
    ActionConditionedQEnsemble,
    ActionIQLConfig,
    ActionIQLLearner,
    FrozenGR00TStateExtractor,
    StateValue,
    load_action_iql_models,
    q_statistics,
)
from gr00t.rl.feature_cache import CachedFeatureDataset
from gr00t.rl.train_action_iql import main
import pytest
import torch

from .test_algorithms import batch
from .test_feature_cache import make_cache
from .test_gr00t_adapter import TinyModel, observations


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(13)
    yield
    torch.set_num_threads(old)


def learner(config=None):
    return ActionIQLLearner(
        ActionConditionedQEnsemble(5, (3, 3), [0, 1, 3, 4], (32, 16), 10),
        StateValue(5, (32, 16)),
        config or ActionIQLConfig(),
    )


def test_shapes_independent_heads_reinjection_and_action_gradient():
    agent = learner()
    state = torch.randn(4, 5, requires_grad=True)
    action = torch.randn(4, 3, 3, requires_grad=True)
    q = agent.critic(state, action)
    assert q.shape == (4, 10)
    assert agent.value(state).shape == (4,)
    for head in agent.critic.heads:
        assert [layer.in_features for layer in head.layers] == [9, 36]
        assert head.out.in_features == 20
    groups = [{p.data_ptr() for p in head.parameters()} for head in agent.critic.heads]
    assert all(not (groups[i] & groups[j]) for i in range(10) for j in range(i))
    grad = torch.autograd.grad(q.mean(), action)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert grad.flatten(1)[:, [2, 5, 6, 7, 8]].count_nonzero() == 0
    epsilon = 1e-3
    plus, minus = action.detach().clone(), action.detach().clone()
    plus[0, 0, 0] += epsilon
    minus[0, 0, 0] -= epsilon
    finite_diff = (agent.critic(state, plus).mean() - agent.critic(state, minus).mean()) / (
        2 * epsilon
    )
    torch.testing.assert_close(grad[0, 0, 0], finite_diff, atol=2e-5, rtol=0.02)
    assert state.grad is None
    agent.critic.requires_grad_(False)
    q = agent.critic(state, action)
    assert torch.autograd.grad(q.mean(), action)[0].norm() > 0
    stats = q_statistics(q)
    assert stats["mean"].shape == (4,) and (stats["std"] > 0).all()
    assert q_statistics(q[:, :1])["std"].eq(0).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert agent.critic(state, action).dtype == torch.float32


def test_losses_exact_ema_masking_and_resume():
    data, agent = batch(), learner()
    q_loss, v_loss, q, v, target = agent.objectives(data)
    expected_target = data.rewards + data.discounts * agent.target_value(data.next_observations)
    torch.testing.assert_close(target, expected_target)
    torch.testing.assert_close(target[data.terminated], data.rewards[data.terminated])
    delta = q.mean(-1).detach() - v
    torch.testing.assert_close(v_loss, (torch.where(delta > 0, 0.8, 0.2) * delta.square()).mean())
    torch.testing.assert_close(q_loss, (q - target[:, None]).square().mean())
    v_loss.backward()
    assert all(p.grad is None for p in agent.critic.parameters())
    before = deepcopy(agent.target_value.state_dict())
    metrics = agent.update(data)
    assert metrics["critic/action_grad_norm"] > 0
    assert metrics["critic/perturb_abs_delta"] > 0
    for key, val in agent.target_value.state_dict().items():
        torch.testing.assert_close(val, before[key].lerp(agent.value.state_dict()[key], 0.005))
    assert all(p.grad is None for p in agent.target_value.parameters())
    twin = learner()
    twin.load_state_dict(deepcopy(agent.state_dict()))
    # Logging randomness is local, and padded actions cannot affect losses.
    changed = replace(data, actions=data.actions + (1 - data.action_mask) * 100)
    assert agent.update(data) == pytest.approx(twin.update(changed))


def test_frozen_backbone_live_interface_and_q_training():
    model = TinyModel()
    extractor = FrozenGR00TStateExtractor(model)
    before = deepcopy(model.state_dict())
    features = extractor(observations())
    q = ActionConditionedQEnsemble(features.shape[1], (2, 1), hidden_dims=(8,), num_q_heads=2)
    action = torch.randn(2, 2, 1, requires_grad=True)
    q(features, action).mean().backward()
    assert action.grad.norm() > 0
    assert not features.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for key, val in model.state_dict().items():
        torch.testing.assert_close(val, before[key], rtol=0, atol=0)


def test_fixed_terminal_batch_overfits():
    agent = learner(ActionIQLConfig(critic_lr=1e-3, value_lr=1e-3, diagnostic_every=100))
    data = batch()
    rewards = data.actions.flatten(1)[:, 0] + 0.5 * data.observations["features"][:, 0]
    data = replace(
        data,
        rewards=rewards,
        discounts=torch.zeros(4),
        terminated=torch.ones(4, dtype=torch.bool),
        truncated=torch.zeros(4, dtype=torch.bool),
    )
    initial = agent.objectives(data)[0].item()
    for _ in range(150):
        agent.update(data)
    final = agent.objectives(data)[0].item()
    assert final < initial * 0.05
    assert agent.action_diagnostics(data)["critic/action_grad_norm"] > 0


def test_cache_discount_cli_checkpoints_and_resume(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "run"
    make_cache(cache)
    ds = CachedFeatureDataset(cache, reward="step-cost", gamma=0.9)
    data = ds.batch([0, 3])
    torch.testing.assert_close(data.rewards, torch.tensor([-1.9, -1.0]))
    torch.testing.assert_close(data.discounts, torch.tensor([0.81, 0.0]))
    args = [
        "--cache",
        str(cache),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--assume-all-success",
        "--num-q-heads",
        "3",
        "--hidden-dims",
        "16",
        "8",
        "--batch-size",
        "2",
        "--log-every",
        "1",
        "--save-every",
        "2",
    ]
    main([*args, "--steps", "2"])
    archive = torch.load(output / "checkpoints/model-step-2.pt", weights_only=True)
    assert "optimizer" not in archive["algorithm"]
    assert archive["metadata"]["q_layout"] == "batch,heads"
    q, v, _ = load_action_iql_models(output / "checkpoints/model-step-2.pt")
    action = data.actions.clone().requires_grad_(True)
    assert q(data.observations, action).shape == (2, 3)
    assert torch.autograd.grad(q(data.observations, action).mean(), action)[0].norm() > 0
    assert not any(p.requires_grad for p in q.parameters())
    assert v(data.observations).shape == (2,)
    metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
    assert metrics["data/success_ratio"] == 1
    assert metrics["critic/action_grad_norm"] > 0
    main([*args, "--steps", "3", "--resume", str(output / "checkpoints/step-2.pt")])
    assert (output / "checkpoints/step-3.pt").exists()
    assert not (output / "checkpoints/step-2.pt").exists()
    assert (output / "checkpoints/model-step-2.pt").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expectile_tau": 1},
        {"value_lr": 0},
        {"target_tau": 0},
        {"critic_lr": float("nan")},
        {"diagnostic_every": 0},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        ActionIQLConfig(**kwargs)


def test_validation_uses_mean_and_ema_value(tmp_path):
    from gr00t.rl.eval_action_iql import evaluate

    root = tmp_path / "cache"
    make_cache(root)
    ds = CachedFeatureDataset(root, reward="step-cost", gamma=0.9)
    q = ActionConditionedQEnsemble(1, ds.action_shape, ds.action_indices, (8,), 3)
    v, target_v = StateValue(1, (8,)), StateValue(1, (8,))
    q.requires_grad_(False)
    v.requires_grad_(False)
    target_v.requires_grad_(False)
    report, data = evaluate(ds, q, v, target_v, 0.8, {"0": "task", "1": "task"}, batch_size=2)
    batch_data = ds.batch([0])
    values = q(batch_data.observations, batch_data.actions)
    target = batch_data.rewards + batch_data.discounts * target_v(batch_data.next_observations)
    assert data["q_mean"][0] == pytest.approx(values.mean().item())
    assert data["td_target"][0] == pytest.approx(target.item())
    assert data["td_squared"][0] == pytest.approx((values - target[:, None]).square().mean().item())
    diff = values.mean().item() - v(batch_data.observations).item()
    assert data["v_expectile_loss"][0] == pytest.approx((0.8 if diff > 0 else 0.2) * diff**2)
    assert report["q_aggregation"] == "mean"
    assert report["transition_weighted"]["action_grad_norm"] > 0
    assert report["terminal"] is not None
    assert report["nonterminal"] is not None


def test_no_proprio_invariant_q_v_and_loaded_model(tmp_path):
    from gr00t.rl.action_conditioned_iql import FrozenStateFeatures

    features = torch.randn(4, 2212)
    changed = features.clone()
    changed[:, 2048:2180] += torch.randn(4, 132) * 100
    extractor = FrozenStateFeatures(2212, exclude_proprio=True)
    assert extractor(features).shape == (4, 2080)
    torch.testing.assert_close(extractor(features)[:, :2048], features[:, :2048])
    torch.testing.assert_close(extractor(features)[:, 2048:], features[:, 2180:])
    q = ActionConditionedQEnsemble(
        2212, (2, 7), hidden_dims=(16, 8), num_q_heads=2, exclude_proprio=True
    )
    v = StateValue(2212, (16, 8), exclude_proprio=True)
    a = torch.randn(4, 2, 7, requires_grad=True)
    torch.testing.assert_close(q(features, a), q(changed, a), rtol=0, atol=0)
    torch.testing.assert_close(v(features), v(changed), rtol=0, atol=0)
    assert torch.autograd.grad(q(features, a).mean(), a)[0].norm() > 0
    assert q.heads[0].layers[0].in_features == 2080 + 14
    assert v.net[0].in_features == 2080
    path = tmp_path / "no-proprio.pt"
    agent = ActionIQLLearner(q, v)
    torch.save(
        {
            "metadata": {
                "backend": "action-conditioned-iql-cache-v1",
                "q_layout": "batch,heads",
                "feature_dim": 2212,
                "action_shape": [2, 7],
                "action_indices": list(range(14)),
                "args": {"hidden_dims": [16, 8], "num_q_heads": 2, "exclude_proprio": True},
            },
            "algorithm": agent.state_dict(),
        },
        path,
    )
    restored_q, restored_v, _ = load_action_iql_models(path)
    torch.testing.assert_close(restored_q(changed, a), q(features, a), rtol=0, atol=0)
    torch.testing.assert_close(restored_v(changed), v(features), rtol=0, atol=0)
    torch.testing.assert_close(agent.target_value(changed), v(features), rtol=0, atol=0)
    with pytest.raises(ValueError, match="layout"):
        FrozenStateFeatures(2048, exclude_proprio=True)
