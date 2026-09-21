from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch

from gr00t.rl.deas_cached import DEASCachedLearner, DEASConfig, DEASDataset, HLGauss
from gr00t.rl.types import OfflineRLBatch
from .test_feature_cache import make_cache


def test_dual_discount_and_terminal(tmp_path):
    make_cache(tmp_path / "cache")
    data = DEASDataset(tmp_path / "cache", discount1=0.9, discount2=0.99)
    batch = data.batch([0, 3])
    assert batch.rewards.tolist() == pytest.approx([-1.9, -1.0])
    assert batch.discounts.tolist() == pytest.approx([0.99**2, 0.0])
    assert batch.terminated.tolist() == [False, True]


def test_hl_gauss_matches_original():
    path = Path(__file__).resolve().parents[4] / "n15/gr00t/model/critic/hlg.py"
    spec = importlib.util.spec_from_file_location("reference_hlg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.HLGaussLoss(-100, 0, 101, 0.1 * 100 / 101)
    ours = HLGauss()
    targets = torch.tensor([-100.0, -70.3, -10.7, -1.0, 0.0])
    torch.testing.assert_close(ours.probabilities(targets), reference.transform_to_probs(targets))
    logits = torch.randn(5, 101)
    torch.testing.assert_close(
        ours.decode(logits), reference.transform_from_probs(logits.softmax(-1))
    )
    assert torch.isfinite(ours.probabilities(torch.tensor([-1000.0, 1000.0]))).all()


def test_losses_isolation_and_resume():
    torch.set_num_threads(2)
    config = DEASConfig(
        vlm_dim=3,
        state_dim=2,
        embodiment_dim=1,
        projection_width=8,
        feature_dim=4,
        hidden_dim=8,
        depth=1,
    )
    learner = DEASCachedLearner([0, 2], config)
    batch = OfflineRLBatch(
        {"features": torch.randn(4, 6)},
        {"features": torch.randn(4, 6)},
        torch.randn(4, 2, 2),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]).expand(4, -1, -1),
        torch.tensor([-1.9, -1.0, -1.9, -1.0]),
        torch.tensor([0.99**2, 0.0, 0.99**2, 0.0]),
        torch.tensor([False, True, False, True]),
        torch.zeros(4, dtype=torch.bool),
        torch.full((4,), 2, dtype=torch.long),
    )
    original = batch.observations["features"].clone()
    with torch.no_grad():
        features = learner.features(batch.observations)
        actions = batch.actions.flatten(1)[:, [0, 2]]
        q_logits = torch.stack([q(torch.cat((features, actions), -1)) for q in learner.critic])
        q_values = learner.hlg.decode(q_logits)
        probabilities = q_logits[q_values.argmin(0), torch.arange(4)].softmax(-1)
        v_logits = learner.value(features)
        weights = torch.where(q_values.min(0).values >= learner.hlg.decode(v_logits), 0.7, 0.3)
        expected_v = -(weights * (probabilities * v_logits.log_softmax(-1)).sum(-1)).mean()
        targets = batch.rewards + batch.discounts * learner.hlg.decode(
            learner.value(learner.features(batch.next_observations))
        )
        expected_q = (
            -(learner.hlg.probabilities(targets)[None] * q_logits.log_softmax(-1)).sum(-1).mean()
        )
    old = deepcopy(learner.state_dict())
    metrics = learner.update(batch)
    assert metrics["loss/q"] == pytest.approx(expected_q.item())
    assert metrics["loss/v"] == pytest.approx(expected_v.item())
    assert metrics["projection/grad_norm"] > 0
    assert all(p.grad is None for p in learner.target_critic.parameters())
    torch.testing.assert_close(original, batch.observations["features"])
    for key, target in learner.target_critic.state_dict().items():
        torch.testing.assert_close(
            target,
            old["target_critic"][key].lerp(learner.critic.state_dict()[key], config.target_tau),
        )
    restored = DEASCachedLearner([0, 2], config)
    restored.load_state_dict(deepcopy(learner.state_dict()))
    first, second = learner.update(batch), restored.update(batch)
    assert first == pytest.approx(second)
