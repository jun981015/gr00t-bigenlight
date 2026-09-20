from copy import deepcopy
from dataclasses import replace
import json

from gr00t.rl.adapters import FrozenBCGR00TEncoder, Gr00tFlowActor
from gr00t.rl.algorithms import SVFConfig
from gr00t.rl.feature_cache import CachedFeatureDataset
from gr00t.rl.frozen_q import FrozenQConditioningEncoder, FrozenQSoftValueFlow, load_frozen_iql_q
from gr00t.rl.networks import FeatureCritic, FeatureFlowActor
from gr00t.rl.train import main as train_main
from gr00t.rl.train_cached import main as cached_main
from gr00t.rl.trainer import OfflineTrainer
from gr00t.rl.types import OfflineRLBatch
import pytest
import torch

from .test_algorithms import batch
from .test_feature_cache import make_cache
from .test_gr00t_adapter import TinyModel, observations


@pytest.fixture(autouse=True)
def cpu_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(previous)


def learner():
    return FrozenQSoftValueFlow(
        FeatureFlowActor(5, (3, 3), (16,)),
        FeatureCritic(5, (3, 3), (16,)),
        FeatureCritic(5, (3, 3), (16,), time_embed_dim=16),
        SVFConfig(flow_steps=2, candidates=2, soft_lambda=0.5, freeze_reference=True),
    )


def test_only_actor_and_inner_update_without_next_action_or_q_ema(monkeypatch):
    model = learner()
    names = ["actor", "inner_critic", "critic", "target_critic", "reference"]
    before = {name: deepcopy(getattr(model, name).state_dict()) for name in names}
    optimizer_ids = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
    for name in ["critic", "target_critic", "reference"]:
        assert not optimizer_ids.intersection(id(p) for p in getattr(model, name).parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen Q must not compute next-action TD targets")

    monkeypatch.setattr(model, "sample_actions", forbidden)
    metrics = model.update(batch())
    for name in names:
        same = all(
            torch.equal(value, before[name][key])
            for key, value in getattr(model, name).state_dict().items()
        )
        assert same == (name not in ["actor", "inner_critic"])
    assert all(p.grad is None for p in model.critic.parameters())
    assert not model.critic.training
    assert metrics["critic/outer_loss"] == 0
    assert metrics["critic/env_q_frozen"] == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["actor/sv_guidance_norm"] > 0


def test_frozen_q_objective_does_not_depend_on_reward_or_successor():
    model, data = learner(), batch()
    changed = replace(
        data, rewards=data.rewards + 100, next_observations={"features": torch.randn(4, 5) * 100}
    )
    torch.manual_seed(50)
    original, _ = model.losses(data)
    torch.manual_seed(50)
    other, _ = model.losses(changed)
    torch.testing.assert_close(original, other, rtol=0, atol=0)


def test_frozen_q_checkpoint_resume(tmp_path):
    data = batch()
    first = OfflineTrainer(learner())
    first.update(data)
    first.save_checkpoint(tmp_path / "state.pt")
    expected = first.update(data)
    second = OfflineTrainer(learner())
    second.load_checkpoint(tmp_path / "state.pt")
    actual = second.update(data)
    assert actual == pytest.approx(expected)
    assert all(not p.requires_grad for p in second.algorithm.critic.parameters())


def test_q_conditioning_matches_bc_and_survives_actor_changes():
    model = TinyModel()
    source = deepcopy(model)
    expected_encoder = FrozenBCGR00TEncoder(source)
    encoder = FrozenQConditioningEncoder(model)
    inputs = observations()
    raw = inputs["backbone_features"].clone()
    expected = expected_encoder.encode_observation(inputs)
    original = encoder.encode_observation(inputs)
    torch.testing.assert_close(original["features"], expected["features"], rtol=0, atol=0)
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.add_(0.5)
    model.action_head.train()
    current = encoder.encode_observation(inputs)
    torch.testing.assert_close(current["features"], original["features"], rtol=0, atol=0)
    torch.testing.assert_close(current["backbone_features"], original["backbone_features"])
    torch.testing.assert_close(inputs["backbone_features"], raw, rtol=0, atol=0)
    assert model.action_head.training  # encoder must not call model.eval()
    assert not any(p.requires_grad for p in encoder.q_attention.parameters())


def test_real_tiny_dit_fixed_q_svf_update():
    model = TinyModel()
    encoder = FrozenQConditioningEncoder(model)
    reference = Gr00tFlowActor(deepcopy(model.action_head))
    obs = encoder.encode_observation(observations())
    algorithm = FrozenQSoftValueFlow(
        Gr00tFlowActor(model.action_head),
        FeatureCritic(71, (3, 2), (16,)),
        FeatureCritic(71, (3, 2), (16,), time_embed_dim=16),
        SVFConfig(flow_steps=2, candidates=2, soft_lambda=1, freeze_reference=True, t_min=1e-6),
        reference,
    )
    before = deepcopy(model.action_head.state_dict())
    data = OfflineRLBatch(
        obs,
        obs,
        torch.randn(2, 3, 2),
        torch.ones(2, 3, 2),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        torch.full((2,), 3),
    )
    metrics = algorithm.update(data)
    assert model.backbone.calls == 1  # no extra VLM calls inside SDE/actor
    assert any(not torch.equal(v, before[k]) for k, v in model.action_head.state_dict().items())
    assert metrics["actor/sv_guidance_norm"] > 0
    assert model.backbone.scale.grad is None


def test_load_actual_cached_iql_and_reject_mismatches(tmp_path):
    cache, output = tmp_path / "cache", tmp_path / "iql"
    make_cache(cache)
    cached_main(
        [
            "--cache",
            str(cache),
            "--output-dir",
            str(output),
            "--reward",
            "step-cost",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--hidden-dim",
            "8",
            "--hidden-layers",
            "1",
        ]
    )
    dataset = CachedFeatureDataset(cache, reward="step-cost")
    kwargs = dict(
        identity=dataset.manifest["identity"],
        feature_dim=1,
        action_mask=dataset.batch([0, 1]).action_mask,
        gamma=0.99,
        reward="step-cost",
    )
    checkpoint = output / "checkpoints/step-2.pt"
    q, provenance = load_frozen_iql_q(checkpoint, cache, **kwargs)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for name, value in q.state_dict().items():
        torch.testing.assert_close(value, state["algorithm"]["critic"][name], rtol=0, atol=0)
    assert provenance["iql_step"] == 2 and not any(p.requires_grad for p in q.parameters())
    for change in [
        {"reward": "terminal-success"},
        {"gamma": 0.9},
        {"feature_dim": 2},
        {"identity": {"wrong": "BC"}},
        {"action_mask": torch.ones(2, 4, 3)},
    ]:
        with pytest.raises(ValueError):
            load_frozen_iql_q(checkpoint, cache, **{**kwargs, **change})
    # Identical dimensions are not enough: cache provenance must match as well.
    state["metadata"]["cache_manifest_sha256"] = "wrong"
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="matching cached-IQL"):
        load_frozen_iql_q(tmp_path / "bad.pt", cache, **kwargs)
    (cache / "COMPLETE.json").write_text(json.dumps({"format": "partial"}))
    with pytest.raises(ValueError, match="completed"):
        load_frozen_iql_q(checkpoint, cache, **kwargs)


def test_cli_rejects_partial_or_wrong_mode_before_model_load(tmp_path):
    args = [
        "--dataset-path",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--fixed-iql-checkpoint",
        "trusted.pt",
    ]
    with pytest.raises(ValueError, match="both"):
        train_main(args)
    with pytest.raises(ValueError, match="GR00T SVF"):
        train_main(args + ["--fixed-iql-cache", "cache"])
