from copy import deepcopy

from gr00t.rl.iql import IQLConfig, IQLCriticLearner
from gr00t.rl.projected_iql import ProjectedIQLQ, ProjectedIQLV, pooled_encoder
import torch

from .test_algorithms import batch


def make_learner():
    projection = pooled_encoder(3, 8, 2)
    return IQLCriticLearner(
        ProjectedIQLQ(projection, 5, (3, 3), (8,), vlm_dim=3, output_dim=2),
        ProjectedIQLV(projection, 5, (8,), vlm_dim=3, output_dim=2),
        IQLConfig(),
    )


def test_shared_encoder_updates_once_and_target_has_independent_ema():
    torch.set_num_threads(2)
    learner = make_learner()
    assert learner.critic.projection is learner.value.projection
    assert len(learner.parameters) == len({id(p) for p in learner.parameters})
    assert learner.target_critic.projection is not learner.critic.projection
    before = deepcopy(learner.target_critic.state_dict())
    data = batch()
    inputs = data.observations["features"].clone()
    metrics = learner.update(data)
    assert metrics["critic/grad_norm"] > 0
    assert any(
        not torch.equal(before[k], v)
        for k, v in learner.critic.state_dict().items()
        if k.startswith("projection.")
    )
    for k, v in learner.target_critic.state_dict().items():
        torch.testing.assert_close(v, before[k].lerp(learner.critic.state_dict()[k], 0.005))
    assert all(p.grad is None for p in learner.target_critic.parameters())
    torch.testing.assert_close(inputs, data.observations["features"])
    restored = make_learner()
    restored.load_state_dict(deepcopy(learner.state_dict()))
    learner.update(data)
    restored.update(data)
    for a, b in zip(learner.parameters, restored.parameters):
        torch.testing.assert_close(a, b)
