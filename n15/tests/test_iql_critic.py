from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.deas_critic import DEASCritic, DEASCriticConfig
from gr00t.model.action_head.iql_critic import IQLCritic, chunk_target, expectile_loss
from gr00t.utils.experiment import PolyakUpdateCallback


def config(**overrides):
    options = dict(backbone_embedding_dim=8, input_embedding_dim=8, hidden_size=8,
                   action_dim=2, action_horizon=2, max_state_dim=3, max_num_embodiments=1,
                   vl_self_attention_cfg=dict(num_attention_heads=1, attention_head_dim=8,
                                              num_layers=1, dropout=0.0),
                   critic_config=dict(hidden_dim=8, depth=1), value_config=dict(hidden_dim=8, depth=1),
                   rl_config=dict(algorithm="iql", critic_action_horizon=2, feature_dim=4,
                                  num_atoms=1, sigma=0.75, tau=0.005, expectile=0.7,
                                  negative_reward=False, discount=0.99), expand_batch=1)
    options.update(overrides)
    return DEASCriticConfig(**options)


def batch():
    return (BatchFeature(data={"backbone_features": torch.randn(2, 3, 8, requires_grad=True)}),
            BatchFeature(data={"backbone_features": torch.randn(2, 3, 8, requires_grad=True)}),
            BatchFeature(data={"embodiment_id": torch.zeros(2, dtype=torch.long),
                               "state": torch.randn(2, 1, 3), "next_state": torch.randn(2, 1, 3),
                               "action": torch.randn(2, 2, 2), "reward": torch.tensor([[0., 1.], [0., 0.]]),
                               "done": torch.tensor([[0., 1.], [0., 0.]])}))


def test_expectile_reference_equation():
    diff = torch.tensor([-2., 1.], requires_grad=True)
    loss = expectile_loss(diff, 0.7)
    assert loss.item() == pytest.approx((0.3 * 4 + 0.7) / 2)
    loss.backward()
    torch.testing.assert_close(diff.grad, torch.tensor([-0.6, 0.7]))


def test_chunk_target_terminal_and_one_step():
    r = torch.tensor([[1., 2., 100.], [1., 2., 3.], [1., 2., 3.]])
    d = torch.tensor([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.]])
    target = chunk_target(r, d, torch.tensor([10., 10., 10.]), 0.5)
    torch.testing.assert_close(target, torch.tensor([2., 4., 1.]))
    torch.testing.assert_close(chunk_target(r[:, :1], d[:, :1], torch.full((3,), 10.), 0.5),
                               torch.tensor([6., 6., 1.]))
    shifted = chunk_target(r, d, torch.tensor([10., 10., 10.]), 0.5, -1)
    torch.testing.assert_close(shifted, torch.tensor([0.5, 2.25, 0.]))
    assert r[0, 0] == 1  # no reward -= 1 side effect


def test_once_only_features_inputs_unchanged_and_repeatable():
    torch.manual_seed(1)
    head = IQLCritic(config()).train()
    b, nb, a = batch()
    snapshots = [{k: v.detach().clone() for k, v in x.items()} for x in (b, nb, a)]
    calls = []
    head.vl_self_attention.register_forward_hook(lambda m, args, out: calls.append(out.detach().clone()))
    output = head(b, nb, a)
    assert len(calls) == 2  # once each for current and next
    for x, snapshot in zip((b, nb, a), snapshots):
        for k in snapshot:
            torch.testing.assert_close(x[k], snapshot[k])
    repeated = head(b, nb, a)
    torch.testing.assert_close(output.loss, repeated.loss)
    output.loss.backward()
    assert b.backbone_features.grad is None and nb.backbone_features.grad is None
    for module in (head.vlln, head.vl_self_attention, head.target_critic, head.target_backbone_encoder):
        assert all(p.grad is None for p in module.parameters())
    assert any(p.grad is not None for p in head.critic.parameters())
    assert any(p.grad is not None for p in head.value.parameters())


@pytest.mark.parametrize("loss_name,unused", [("value_loss", "critic"), ("critic_loss", "value")])
def test_q_and_v_targets_do_not_backpropagate(loss_name, unused):
    head = IQLCritic(config())
    output = head(*batch())
    output[loss_name].backward()
    assert all(p.grad is None for p in getattr(head, unused).parameters())
    assert any(p.grad is not None for p in head.backbone_encoder.parameters())


def test_full_target_ema_and_optimizer_step():
    head = IQLCritic(config())
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=1e-3)
    output = head(*batch())
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    for source, target in [(head.critic, head.target_critic),
                           (head.backbone_encoder, head.target_backbone_encoder)]:
        before = [p.detach().clone() for p in target.parameters()]
        PolyakUpdateCallback(target, source, tau=0.2).on_step_end(None, None, None)
        for old, actual, online in zip(before, target.parameters(), source.parameters()):
            torch.testing.assert_close(actual, old * 0.8 + online * 0.2)
            assert not actual.requires_grad
    head.set_trainable_parameters(True, True)
    assert all(not p.requires_grad for p in head.target_backbone_encoder.parameters())


@pytest.mark.parametrize("bad", [dict(expand_batch=2), dict(rl_config={"nstep": 2}),
                               dict(rl_config={"q_agg": "mean"})])
def test_reject_unsupported_transitions(bad):
    with pytest.raises(ValueError):
        IQLCritic(config(**bad))


def test_checkpoint_dispatch_and_roundtrip(tmp_path, monkeypatch):
    import gr00t.model.gr00t_n1_deas_critic as model_module

    class TinyBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(8, 8)

    monkeypatch.setattr(model_module, "EagleBackbone", TinyBackbone)
    cfg = model_module.GR00T_N1_5_DEAS_Critic_Config(
        backbone_cfg={}, critic_cfg=config().to_dict(), action_dim=2, action_horizon=2)
    model = model_module.GR00T_N1_5_DEAS_Critic(cfg, local_model_path=str(tmp_path))
    assert isinstance(model.critic_head, IQLCritic)
    model.save_pretrained(tmp_path)
    restored = PreTrainedModel.from_pretrained.__func__(
        model_module.GR00T_N1_5_DEAS_Critic, tmp_path, local_model_path=str(tmp_path), local_files_only=True)
    assert isinstance(restored.critic_head, IQLCritic)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    legacy = config(rl_config=dict(critic_action_horizon=2, feature_dim=4, num_atoms=11,
                                  sigma=0.75, tau=0.005)).to_dict()
    cfg.critic_cfg = legacy
    assert type(model_module.GR00T_N1_5_DEAS_Critic(cfg, str(tmp_path)).critic_head) is DEASCritic


def test_bon_scores_with_scalar_q():
    from gr00t.model.action_head.deas_action_head_bon import DEASActionHeadBoN, DEASActionHeadBoNConfig
    cfg = config().to_dict()
    cfg.update(diffusion_model_cfg=dict(num_attention_heads=1, attention_head_dim=8,
                                        num_layers=1, dropout=0., cross_attention_dim=8,
                                        output_dim=8), num_target_vision_tokens=2,
               num_inference_timesteps=1, max_seq_len=8)
    cfg["rl_config"].update(num_samples=3, temperature=0.)
    head = DEASActionHeadBoN(DEASActionHeadBoNConfig(**cfg)).eval()
    captured = {}

    def record(module, inputs, output):
        captured["actions"] = inputs[2].clone().reshape(3, 2, 2, 2)
        captured["q"] = torch.minimum(*output).reshape(3, 2)

    head.critic.register_forward_hook(record)
    # Distributional decoding must never run for scalar Q.
    def fail(*args, **kwargs):
        raise AssertionError("IQL cannot use HLGauss decoding")
    head.hlg.transform_from_probs = fail
    b, _, a = batch()
    result = head.get_action(b, a)
    expected = captured["actions"][captured["q"].argmax(0), torch.arange(2)]
    torch.testing.assert_close(result.action_pred, expected)


def test_actor_and_iql_checkpoint_assembly_preserves_actor_tokens(monkeypatch):
    from transformers import PretrainedConfig
    import gr00t.model.gr00t_n1_deas_bon as bon_module
    from gr00t.model.action_head.deas_action_head_bon import DEASActionHeadBoN, DEASActionHeadBoNConfig

    class TinyBackbone(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def set_trainable_parameters(self, **kwargs):
            self.requires_grad_(False)

    cfg = config().to_dict()
    cfg.update(diffusion_model_cfg=dict(num_attention_heads=1, attention_head_dim=8,
                                        num_layers=1, dropout=0., cross_attention_dim=8, output_dim=8),
               num_target_vision_tokens=2, num_inference_timesteps=1, max_seq_len=8)
    actor = SimpleNamespace(
        config=PretrainedConfig(backbone_cfg={}, action_head_cfg=cfg, action_dim=2, action_horizon=2),
        backbone=TinyBackbone(), action_head=DEASActionHeadBoN(DEASActionHeadBoNConfig(**cfg)),
        local_model_path="unused")
    with torch.no_grad():
        actor.action_head.future_tokens.weight.fill_(0.123)
    critic = SimpleNamespace(critic_head=IQLCritic(config()))
    monkeypatch.setattr(bon_module, "EagleBackbone", TinyBackbone)
    monkeypatch.setattr(bon_module.GR00T_N1_5, "from_pretrained", lambda *a, **k: deepcopy(actor))
    monkeypatch.setattr(bon_module.GR00T_N1_5_DEAS_Critic, "from_pretrained", lambda *a, **k: deepcopy(critic))
    assembled = bon_module.GR00T_N1_5_DEAS_BoN.from_pretrained_bc_and_critic("actor", "critic")
    assert assembled.action_head.rl_config.algorithm == "iql"
    torch.testing.assert_close(assembled.action_head.future_tokens.weight, actor.action_head.future_tokens.weight)
    for name, tensor in critic.critic_head.critic.state_dict().items():
        torch.testing.assert_close(assembled.action_head.critic.state_dict()[name], tensor)
