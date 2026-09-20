from gr00t.rl.selection import select_action_candidates
from gr00t.rl.trainer import OfflineTrainer
import pytest
import torch
from torch import nn

from .test_algorithms import batch


class RankingCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, obs, action):
        assert not self.training
        score = action[:, 0, 0] * obs["sign"] * self.scale
        return torch.stack((score, score + 0.5))


@pytest.mark.parametrize("aggregation,offset", [("min", 0.0), ("mean", 0.25)])
def test_best_of_n_selection_masks_and_no_grad(aggregation, offset):
    critic = RankingCritic().train()
    obs = {"sign": torch.tensor([1.0, -1.0])}
    mask = torch.tensor([[[1.0, 0.0]]] * 2)
    candidates = iter([-1.0, 0.0, 1.0])

    def sample(observations, action_mask):
        assert observations is obs
        return torch.full_like(action_mask, next(candidates), requires_grad=True)

    result = select_action_candidates(
        sample, critic, obs, mask, num_candidates=3, aggregation=aggregation
    )
    assert result.indices.tolist() == [2, 0]
    torch.testing.assert_close(result.actions[:, 0, 0], torch.tensor([1.0, -1.0]))
    assert (result.actions[:, 0, 1] == 0).all()
    torch.testing.assert_close(result.scores[0], torch.tensor([-1.0, 1.0]) + offset)
    assert critic.training and critic.scale.grad is None
    assert not result.actions.requires_grad and not result.scores.requires_grad


def test_softmax_selection_is_seeded_and_rejects_bad_scores():
    critic = RankingCritic()
    obs = {"sign": torch.zeros(128)}
    mask = torch.ones(128, 1, 1)

    def run():
        return select_action_candidates(
            lambda obs, mask: torch.ones_like(mask),
            critic,
            obs,
            mask,
            num_candidates=3,
            temperature=1.0,
        )

    torch.manual_seed(5)
    first = run()
    torch.manual_seed(5)
    assert torch.equal(first.indices, run().indices)
    assert first.indices.unique().numel() == 3
    with pytest.raises(ValueError, match="finite actions"):
        select_action_candidates(lambda obs, mask: mask * float("nan"), critic, obs, mask)


def test_generic_trainer_accepts_critic_only_algorithm():
    class CriticOnly:
        def __init__(self):
            self.updates = 0

        def update(self, data):
            data.validate()
            self.updates += 1
            return {"critic/loss": float(data.rewards.square().mean())}

        def state_dict(self):
            return {"updates": self.updates}

        def load_state_dict(self, state):
            self.updates = state["updates"]

    trainer = OfflineTrainer(CriticOnly())
    assert trainer.update(batch())["critic/loss"] >= 0
    assert trainer.step == trainer.algorithm.updates == 1
