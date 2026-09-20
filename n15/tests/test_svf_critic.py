from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gr00t.model.action_head.cross_attention_dit import SelfAttentionTransformer
from gr00t.model.action_head.svf_critic import (
    SVFCriticConfig,
    SVFCriticHead,
    aggregate,
    fourier_time_embed,
    outer_td_target,
    soft_value_target,
)
from gr00t.model.gr00t_n1_svf import GR00TN15SVF


def config(**kwargs):
    return SVFCriticConfig(
        **dict(
            {
                "backbone_dim": 8,
                "state_dim": 3,
                "action_dim": 2,
                "action_horizon": 2,
                "num_embodiments": 1,
                "projection_hidden_dim": 8,
                "feature_dim": 4,
                "hidden_dims": (8, 8),
                "time_embed_dim": 4,
            },
            **kwargs,
        )
    )


def attention():
    return SelfAttentionTransformer(num_attention_heads=1, attention_head_dim=8, num_layers=1, dropout=0.1)


def head_batch():
    torch.manual_seed(9)
    head = SVFCriticHead(config(), nn.LayerNorm(8), attention()).train()
    raw = {"backbone_features": torch.randn(2, 3, 8, requires_grad=True)}
    state = torch.randn(2, 1, 3, requires_grad=True)
    ids = torch.zeros(2, dtype=torch.long)
    actions = torch.randn(2, 2, 2, requires_grad=True)
    noisy = torch.randn(2, 2, 2, requires_grad=True)
    times = torch.tensor([0.2, 0.8])
    return head, raw, state, ids, actions, noisy, times


def test_once_only_attention_nonmutation_and_shapes():
    head, raw, state, ids, actions, noisy, times = head_batch()
    before = raw["backbone_features"].detach().clone()
    calls = []
    head.vl_self_attention.register_forward_hook(lambda *args: calls.append(1))
    observation = head.encode_observation(raw, state, ids)
    predictions = head(observation, actions, noisy, times)
    head.q_values(observation, actions, target=True)
    head.soft_values(observation, noisy, times, target=True)
    assert len(calls) == 1
    assert all(value.shape == (2, 2) for value in predictions.values())
    torch.testing.assert_close(raw["backbone_features"], before)
    assert raw["backbone_features"].requires_grad
    assert not observation.pooled.requires_grad and not observation.state.requires_grad
    for module in (head.vlln, head.vl_self_attention, head.target_critic, head.target_tc_critic):
        assert not module.training and not any(p.requires_grad for p in module.parameters())
    repeated = head.encode_observation(raw, state, ids)
    torch.testing.assert_close(repeated.pooled, observation.pooled)


@pytest.mark.parametrize(
    "loss_key,active,inactive", [("critic_loss", "critic", "tc_critic"), ("inner_loss", "tc_critic", "critic")]
)
def test_loss_gradients_isolated(loss_key, active, inactive):
    head, raw, state, ids, actions, noisy, times = head_batch()
    predictions = head(head.encode_observation(raw, state, ids), actions, noisy, times)
    q_target = torch.randn(2, requires_grad=True)
    v_target = torch.randn(2, requires_grad=True)
    losses = head.loss(predictions, q_target, v_target)
    losses[loss_key].backward()
    assert q_target.grad is None and v_target.grad is None
    assert raw["backbone_features"].grad is None and state.grad is None
    assert all(p.grad is not None for p in getattr(head, active).parameters())
    for module in (
        getattr(head, inactive),
        head.target_critic,
        head.target_tc_critic,
        head.vlln,
        head.vl_self_attention,
    ):
        assert all(p.grad is None for p in module.parameters())


def test_loss_reference_aggregation_before_mse():
    head, *_ = head_batch()
    predictions = {
        "q_values": torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
        "soft_values": torch.tensor([[0.0, 2.0], [2.0, 4.0]]),
    }
    losses = head.loss(predictions, torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0]))
    assert losses["critic_loss"].item() == 1
    assert losses["inner_loss"].item() == 0


def test_guidance_and_action_padding():
    head, raw, state, ids, _actions, noisy, times = head_batch()
    observation = head.encode_observation(raw, state, ids)
    mask = torch.ones_like(noisy)
    mask[:, :, -1] = 0
    with torch.no_grad():
        guidance = head.guidance_gradient(observation, noisy, times, action_mask=mask)
    assert guidance[:, :, 0].abs().sum() > 0
    assert guidance[:, :, -1].count_nonzero() == 0
    assert not guidance.requires_grad
    assert all(p.grad is None for p in head.parameters())
    changed = noisy.detach().clone()
    changed[:, :, -1] += 100
    torch.testing.assert_close(
        head.soft_values(observation, noisy, times, action_mask=mask),
        head.soft_values(observation, changed, times, action_mask=mask),
    )
    v = head.soft_values(observation, noisy, times, target=True, action_mask=mask)
    grad = torch.autograd.grad(v.sum(), noisy)[0]
    assert grad[:, :, 0].abs().sum() > 0


def test_fourier_and_soft_target_reference():
    t = torch.tensor([0.0, 1.0])
    expected_angles = t[:, None] * torch.tensor([1.0, 256.0])
    torch.testing.assert_close(
        fourier_time_embed(t, 4), torch.cat((expected_angles.sin(), expected_angles.cos()), -1)
    )
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    expected = 0.5 * (torch.logsumexp(q / 0.5, 0) - torch.log(torch.tensor(2.0)))
    actual = soft_value_target(q, 0.5)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
    torch.testing.assert_close(soft_value_target(q.detach()[:1], 0.5), q.detach()[0])
    assert torch.isfinite(soft_value_target(torch.full((3, 2), 1e20), 1e-20)).all()
    with pytest.raises(ValueError):
        soft_value_target(q, 0)
    torch.testing.assert_close(aggregate(q, "min"), q[0])


def test_outer_target_no_double_discount_or_reward_sum():
    target = outer_td_target(
        torch.tensor([2.0, 3.0]), torch.tensor([0.0, 0.81]), torch.tensor([100.0, 10.0], requires_grad=True)
    )
    torch.testing.assert_close(target, torch.tensor([2.0, 11.1]))
    assert not target.requires_grad
    with pytest.raises(ValueError):
        outer_td_target(torch.ones(2, 3), torch.ones(2), torch.ones(2))


def test_full_ema_after_optimizer_step_and_checkpoint():
    head, raw, state, ids, actions, noisy, times = head_batch()
    old_q, old_v = deepcopy(head.target_critic), deepcopy(head.target_tc_critic)
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    output = head(head.encode_observation(raw, state, ids), actions, noisy, times)
    head.loss(output, torch.ones(2), torch.zeros(2))["loss"].backward()
    optimizer.step()
    head.update_targets(0.2)
    for online, target, old in (
        (head.critic, head.target_critic, old_q),
        (head.tc_critic, head.target_tc_critic, old_v),
    ):
        for p, tp, op in zip(online.parameters(), target.parameters(), old.parameters()):
            torch.testing.assert_close(tp, op * 0.8 + p * 0.2)
    replica = SVFCriticHead(config(), nn.LayerNorm(8), attention())
    replica.load_state_dict(deepcopy(head.state_dict()))
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, replica.state_dict()[key])


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return {"backbone_features": self.linear(inputs["tokens"])}


class FakeBC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.action_head = nn.Module()
        self.action_head.vlln = nn.LayerNorm(8)
        self.action_head.vl_self_attention = attention()
        self.action_head.config = SimpleNamespace(
            backbone_embedding_dim=8, max_state_dim=3, action_dim=2, action_horizon=2, max_num_embodiments=1
        )

    def prepare_input(self, inputs):
        return inputs, inputs


def test_bc_wrapper_forward_and_checkpoint_roundtrip(tmp_path):
    bc = FakeBC()
    originals = deepcopy(bc.state_dict())
    model = GR00TN15SVF(bc, config()).train()
    batch = {
        "tokens": torch.randn(2, 3, 8),
        "state": torch.randn(2, 1, 3),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }
    kwargs = {
        "actions": torch.randn(2, 2, 2),
        "noisy_actions": torch.randn(2, 2, 2),
        "times": torch.rand(2),
        "q_targets": torch.randn(2),
        "soft_targets": torch.randn(2),
    }
    output = model(batch, **kwargs)
    output["loss"].backward()
    assert bc.backbone.calls == 1 and not bc.training
    assert all(p.grad is None for p in bc.parameters())
    for key, value in originals.items():
        torch.testing.assert_close(bc.state_dict()[key], value)
    path = tmp_path / "critics.pt"
    torch.save(model.critic_checkpoint(), path)
    replica = GR00TN15SVF(deepcopy(bc), config())
    replica.load_critic_checkpoint(torch.load(path, weights_only=True))
    torch.testing.assert_close(output["loss"], replica(batch, **kwargs)["loss"])
    assert model.critic_head.vlln is not bc.action_head.vlln
    bad = model.critic_checkpoint()
    bad["config"] = asdict(config(tau=0.1))
    with pytest.raises(ValueError):
        replica.load_critic_checkpoint(bad)


def test_bfloat16_frozen_adapter_float32_heads():
    model = GR00TN15SVF(FakeBC().to(dtype=torch.bfloat16), config())
    batch = {
        "tokens": torch.randn(2, 3, 8, dtype=torch.bfloat16),
        "state": torch.randn(2, 1, 3, dtype=torch.bfloat16),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }
    out = model(
        batch,
        actions=torch.randn(2, 2, 2),
        noisy_actions=torch.randn(2, 2, 2),
        times=torch.rand(2),
        q_targets=torch.zeros(2),
        soft_targets=torch.ones(2),
    )
    out["loss"].backward()
    assert out["loss"].dtype == torch.float32 and torch.isfinite(out["loss"])


def test_bad_shapes_fail_explicitly():
    head, raw, state, ids, actions, noisy, _times = head_batch()
    obs = head.encode_observation(raw, state, ids)
    with pytest.raises(ValueError):
        head.q_values(obs, actions[:, :1])
    with pytest.raises(ValueError):
        head.soft_values(obs, noisy, torch.ones(2, 2))
    with pytest.raises(ValueError):
        head.encode_observation(raw, state, ids + 1)
    with pytest.raises(ValueError):
        SVFCriticConfig(time_embed_dim=3)


@pytest.mark.parametrize("dimension", [16, 64])
def test_selectable_fourier_width(dimension):
    original_config = config()
    model = GR00TN15SVF(FakeBC(), original_config, time_embed_dim=dimension)
    head = model.critic_head
    assert original_config.time_embed_dim == 4  # overrides do not mutate caller config
    assert head.config.time_embed_dim == dimension
    inputs = {
        "tokens": torch.randn(2, 3, 8),
        "state": torch.randn(2, 1, 3),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }
    obs = model.encode_observation(inputs)
    times = torch.rand(2)
    noisy = torch.randn(2, 2, 2, requires_grad=True)
    assert fourier_time_embed(times, dimension).shape == (2, dimension)
    base_width = original_config.feature_dim + original_config.state_dim + 4
    assert head.critic.members[0][0].in_features == base_width
    assert head.tc_critic.members[0][0].in_features == base_width + dimension
    values = head.soft_values(obs, noisy, times)
    assert values.shape == (2, 2)
    values.sum().backward()
    assert noisy.grad.abs().sum() > 0
    checkpoint = deepcopy(model.critic_checkpoint())
    replica = GR00TN15SVF(deepcopy(model.bc_policy), config(time_embed_dim=dimension))
    replica.load_critic_checkpoint(checkpoint)
    torch.testing.assert_close(values, replica.critic_head.soft_values(obs, noisy, times))
    wrong_width = GR00TN15SVF(FakeBC(), config(time_embed_dim=64 if dimension == 16 else 16))
    with pytest.raises(ValueError, match="configuration mismatch"):
        wrong_width.load_critic_checkpoint(checkpoint)


def q_config(loss_type, **kwargs):
    return config(q_loss_type=loss_type, hl_gauss_min=-2.0, hl_gauss_max=2.0, hl_gauss_num_bins=11, **kwargs)


def transition_inputs():
    return {
        "tokens": torch.randn(2, 3, 8),
        "state": torch.randn(2, 1, 3),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "action_mask": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]] * 2),
    }


@pytest.mark.parametrize("loss_type", ["mse", "hl_gauss"])
@pytest.mark.parametrize("mode", ["td", "iql"])
@pytest.mark.parametrize("dimension", [16, 64])
def test_q_modes_update_and_checkpoint(loss_type, mode, dimension):
    cfg = q_config(loss_type, outer_update=mode, time_embed_dim=dimension)
    model = GR00TN15SVF(FakeBC(), cfg).train()
    head = model.critic_head
    current, following = transition_inputs(), transition_inputs()
    snapshot = deepcopy(current)
    calls = []
    head.vl_self_attention.register_forward_hook(lambda *args: calls.append(1))
    kwargs = {
        "actions": torch.randn(2, 2, 2),
        "noisy_actions": torch.randn(2, 2, 2),
        "times": torch.rand(2),
        "soft_targets": torch.tensor([0.3, -0.5]),
    }
    if mode == "iql":
        kwargs.update(
            next_inputs=following,
            chunk_returns=torch.tensor([0.0, 1.0]),
            bootstrap_discounts=torch.tensor([0.81, 0.0]),
        )
    else:
        kwargs["q_targets"] = torch.tensor([0.2, 0.5])
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    output = model(current, **kwargs)
    assert len(calls) == (2 if mode == "iql" else 1)
    assert output["q_values"].shape == (2, 2)
    if loss_type == "hl_gauss":
        assert output["q_logits"].shape == (2, 2, 11)
        assert output["q_values"].abs().max() <= 2
    for key in current:
        torch.testing.assert_close(current[key], snapshot[key])
    old = deepcopy(head.target_critic.state_dict())
    output["loss"].backward()
    optimizer.step()
    head.update_targets(0.2)
    for key, value in head.target_critic.state_dict().items():
        torch.testing.assert_close(value, old[key] * 0.8 + head.critic.state_dict()[key] * 0.2)
    replica = GR00TN15SVF(deepcopy(model.bc_policy), cfg)
    replica.load_critic_checkpoint(deepcopy(model.critic_checkpoint()))
    torch.testing.assert_close(replica(current, **kwargs)["loss"], model(current, **kwargs)["loss"])
    assert all(p.grad is None for p in model.bc_policy.parameters())
    assert all(p.grad is None for p in head.target_critic.parameters())


@pytest.mark.parametrize("loss_type,v_loss_type", [("mse", "mse"), ("hl_gauss", "mse"), ("hl_gauss", "deas")])
@pytest.mark.parametrize(
    "loss_key,active", [("critic_loss", "critic"), ("iql_value_loss", "iql_value"), ("inner_loss", "tc_critic")]
)
def test_iql_three_way_gradient_isolation(loss_type, v_loss_type, loss_key, active):
    model = GR00TN15SVF(FakeBC(), q_config(loss_type, outer_update="iql", v_loss_type=v_loss_type))
    head = model.critic_head
    obs = model.encode_observation(transition_inputs())
    next_obs = model.encode_observation(transition_inputs())
    returns = torch.randn(2, requires_grad=True)
    soft_target = torch.randn(2, requires_grad=True)
    out = head.iql_loss(
        obs,
        next_obs,
        torch.randn(2, 2, 2),
        returns,
        torch.tensor([0.9, 0.0]),
        noisy_actions=torch.randn(2, 2, 2),
        times=torch.rand(2),
        soft_targets=soft_target,
    )
    out[loss_key].backward()
    for name in ("critic", "iql_value", "tc_critic", "target_critic", "target_tc_critic"):
        parameters = list(getattr(head, name).parameters())
        if name == active:
            assert all(p.grad is not None for p in parameters)
        else:
            assert all(p.grad is None for p in parameters)
    assert returns.grad is None and soft_target.grad is None


@pytest.mark.parametrize("loss_type", ["mse", "hl_gauss"])
def test_iql_exact_targets_and_expectile(loss_type):
    cfg = q_config(loss_type, outer_update="iql", iql_expectile=0.8, q_aggregation="mean")
    model = GR00TN15SVF(FakeBC(), cfg)
    head = model.critic_head
    with torch.no_grad():
        for module in (head.critic, head.target_critic, head.iql_value):
            for p in module.parameters():
                p.zero_()
        head.iql_value.members[0][-1].bias.fill_(0.25)
        if loss_type == "mse":
            head.target_critic.members[0][-1].bias.fill_(-0.2)
            head.target_critic.members[1][-1].bias.fill_(0.6)
        else:
            head.target_critic.members[0][-1].bias.copy_(torch.linspace(2.0, -2.0, 11))
            head.target_critic.members[1][-1].bias.copy_(torch.linspace(-2.0, 2.0, 11))
    obs = model.encode_observation(transition_inputs())
    next_obs = model.encode_observation(transition_inputs())
    actions = torch.randn(2, 2, 2)
    q_min = head.q_values(obs, actions, target=True).min(0).values
    expected_target = torch.tensor([0.1 + 0.9 * 0.25, 1.0])
    result = head.iql_loss(obs, next_obs, actions, torch.tensor([0.1, 1.0]), torch.tensor([0.9, 0.0]))
    torch.testing.assert_close(result["critic/target_mean"], expected_target.mean())
    diff = q_min - 0.25
    expected_v_loss = (torch.where(diff > 0, 0.8, 0.2) * diff.square()).mean()
    torch.testing.assert_close(result["iql_value_loss"], expected_v_loss)
    torch.testing.assert_close(result["critic_loss"], head.q_loss(result, expected_target)["critic_loss"])
    assert "inner_loss" not in result  # critic-only warmup needs no SVF targets


def test_hl_gauss_matches_deas_loss_and_clips_outliers():
    from gr00t.model.critic.hlg import HLGaussLoss

    cfg = q_config("hl_gauss")
    head = SVFCriticHead(cfg, nn.LayerNorm(8), attention())
    logits = torch.randn(2, 3, 11, requires_grad=True)
    targets = torch.tensor([-0.5, 0.0, 1.0], requires_grad=True)
    predicted = {"q_values": head._decode_q(logits), "q_logits": logits}
    deas = HLGaussLoss(-2.0, 2.0, 11, 0.1 * 4 / 11)
    expected = torch.stack([deas(member, targets.detach()) for member in logits]).mean()
    actual = head.q_loss(predicted, targets)
    torch.testing.assert_close(actual["critic_loss"], expected)
    actual["critic_loss"].backward()
    assert targets.grad is None and logits.grad.abs().sum() > 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outliers = head.q_loss(predicted, torch.tensor([-1e20, 0.0, 1e20]))
    assert torch.isfinite(outliers["critic_loss"])
    assert outliers["critic_loss"].dtype == torch.float32
    assert outliers["critic/target_clipped_fraction"].item() == pytest.approx(2 / 3)


def test_iql_wrapper_shortcuts_and_required_inputs():
    model = GR00TN15SVF(
        FakeBC(), config(), outer_update="iql", q_loss_type="hl_gauss", hl_gauss_min=-2.0, hl_gauss_max=2.0
    )
    assert model.critic_head.config.outer_update == "iql"
    assert model.critic_head.config.q_loss_type == "hl_gauss"
    with pytest.raises(ValueError, match="IQL requires"):
        model(transition_inputs(), actions=torch.randn(2, 2, 2))
    with pytest.raises(ValueError, match="builds Q targets"):
        model(transition_inputs(), actions=torch.randn(2, 2, 2), q_targets=torch.ones(2))
    with pytest.raises(ValueError, match="explicit finite"):
        config(q_loss_type="hl_gauss")
    with pytest.raises(ValueError, match="iql_expectile"):
        config(outer_update="iql", iql_expectile=1.0)


def test_bc_loader_does_not_receive_critic_options(monkeypatch):
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    received = {}

    def fake_loader(path, **kwargs):
        received.update(path=path, **kwargs)
        return FakeBC()

    monkeypatch.setattr(GR00T_N1_5, "from_pretrained", fake_loader)
    model = GR00TN15SVF.from_bc_checkpoint(
        "local-bc",
        config(),
        time_embed_dim=64,
        outer_update="iql",
        q_loss_type="hl_gauss",
        v_loss_type="deas",
        hl_gauss_min=-2.0,
        hl_gauss_max=2.0,
        local_files_only=True,
    )
    assert model.critic_head.config.time_embed_dim == 64
    assert model.critic_head.config.q_loss_type == "hl_gauss"
    assert model.critic_head.config.v_loss_type == "deas"
    assert received == {
        "path": "local-bc",
        "local_files_only": True,
        "tune_visual": False,
        "tune_llm": False,
        "tune_projector": False,
        "tune_diffusion_model": False,
    }


def test_previous_td_mse_checkpoint_config_still_loads():
    model = GR00TN15SVF(FakeBC(), config())
    previous = deepcopy(model.critic_checkpoint())
    for key in (
        "outer_update",
        "q_loss_type",
        "v_loss_type",
        "iql_expectile",
        "hl_gauss_min",
        "hl_gauss_max",
        "hl_gauss_num_bins",
        "hl_gauss_sigma_ratio",
    ):
        del previous["config"][key]
    model.load_critic_checkpoint(previous)


@pytest.mark.parametrize("tie", [False, True])
def test_deas_value_matches_distribution_ce_and_scalar_bootstrap(monkeypatch, tie):
    model = GR00TN15SVF(FakeBC(), q_config("hl_gauss", outer_update="iql", v_loss_type="deas", iql_expectile=0.8))
    head = model.critic_head
    obs = model.encode_observation(transition_inputs())
    next_obs = model.encode_observation(transition_inputs())
    low = torch.linspace(2.0, -2.0, 11)
    high = low.flip(0)
    # Opposite ensemble member wins for each batch row.
    q_logits = torch.stack([torch.stack([low, high]), torch.stack([high, low])]).requires_grad_(True)
    v_logits = (
        torch.stack([low, low]) if tie else torch.stack([low * 2.5, torch.zeros_like(low)])
    ).requires_grad_(True)
    monkeypatch.setattr(head.target_critic, "forward", lambda *args, **kwargs: q_logits)
    monkeypatch.setattr(head.iql_value, "forward", lambda *args, **kwargs: v_logits.unsqueeze(0))
    output = head.iql_loss(obs, next_obs, torch.randn(2, 2, 2), torch.tensor([0.1, 0.2]), torch.tensor([0.9, 0.0]))
    q_probs = q_logits.detach().softmax(-1)
    q_values = head.hlg.transform_from_probs(q_probs)
    q_min, selected = q_values.min(0)
    assert selected.tolist() == [0, 1]
    selected_probs = q_probs[selected, torch.arange(2)]
    v = head.hlg.transform_from_probs(v_logits.softmax(-1))
    weights = torch.where(q_min >= v.detach(), 0.8, 0.2)
    assert weights.tolist() == pytest.approx([0.8, 0.8] if tie else [0.8, 0.2])
    reference = (weights * -(selected_probs * v_logits.log_softmax(-1)).sum(-1)).mean()
    torch.testing.assert_close(output["iql_value_loss"], reference)
    torch.testing.assert_close(head.state_values(obs), v)
    expected_q_target = torch.tensor([0.1, 0.2]) + torch.tensor([0.9, 0.0]) * v.detach()
    torch.testing.assert_close(output["critic/target_mean"], expected_q_target.mean())
    reference_grad = torch.autograd.grad(reference, v_logits, retain_graph=True)[0]
    output["iql_value_loss"].backward()
    torch.testing.assert_close(v_logits.grad, reference_grad)
    assert q_logits.grad is None
    assert all(p.grad is None for p in head.critic.parameters())


@pytest.mark.parametrize("dimension", [16, 64])
def test_deas_value_optimizer_bf16_and_checkpoint(dimension):
    cfg = q_config("hl_gauss", outer_update="iql", v_loss_type="deas", time_embed_dim=dimension)
    model = GR00TN15SVF(FakeBC(), cfg).train()
    head = model.critic_head
    current, following = transition_inputs(), transition_inputs()
    before = deepcopy(current)
    calls = []
    head.vl_self_attention.register_forward_hook(lambda *args: calls.append(1))
    args = {
        "actions": torch.randn(2, 2, 2),
        "next_inputs": following,
        "chunk_returns": torch.tensor([0.0, 1.0]),
        "bootstrap_discounts": torch.tensor([0.9, 0.0]),
        "noisy_actions": torch.randn(2, 2, 2),
        "times": torch.rand(2),
        "soft_targets": torch.randn(2),
    }
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(current, **args)
    assert len(calls) == 2
    assert output["iql_value_logits"].shape == (2, 11)
    assert head.tc_critic.members[0][-1].out_features == 1
    assert output["iql_value_loss"].dtype == torch.float32
    assert torch.isfinite(output["loss"])
    old_v = deepcopy(head.iql_value.state_dict())
    output["loss"].backward()
    optimizer.step()
    head.update_targets()
    assert any(not torch.equal(v, old_v[k]) for k, v in head.iql_value.state_dict().items())
    for key in current:
        torch.testing.assert_close(current[key], before[key])
    assert all(p.grad is None for p in model.bc_policy.parameters())
    assert all(p.grad is None for p in head.target_critic.parameters())
    replica = GR00TN15SVF(deepcopy(model.bc_policy), cfg)
    replica.load_critic_checkpoint(deepcopy(model.critic_checkpoint()))
    torch.testing.assert_close(replica(current, **args)["loss"], model(current, **args)["loss"])
    scalar = GR00TN15SVF(FakeBC(), q_config("hl_gauss", outer_update="iql", time_embed_dim=dimension))
    with pytest.raises(ValueError, match="configuration mismatch"):
        scalar.load_critic_checkpoint(model.critic_checkpoint())


def test_deas_value_requires_distributional_q_and_iql():
    with pytest.raises(ValueError, match="requires outer_update"):
        config(outer_update="iql", q_loss_type="mse", v_loss_type="deas")
    with pytest.raises(ValueError, match="requires outer_update"):
        q_config("hl_gauss", outer_update="td", v_loss_type="deas")
    with pytest.raises(ValueError, match="v_loss_type must"):
        config(v_loss_type="unknown")
