"""Scalar IQL with a shared DEAS-style encoder after frozen token pooling.

Only the pooled VLM vector is projected. Preserve the existing scalar IQL
state/embodiment/action layout and losses to isolate the encoder change.
"""

import torch
from torch import nn

from .iql import FeatureValue
from .networks import FeatureCritic


def pooled_encoder(vlm_dim=2048, width=1024, output_dim=64):
    return nn.Sequential(
        nn.Linear(vlm_dim, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, output_dim),
        nn.Tanh(),
    )


class ProjectedIQLQ(nn.Module):
    def __init__(self, projection, feature_dim, action_shape, hidden, vlm_dim=2048, output_dim=64):
        super().__init__()
        self.projection, self.vlm_dim = projection, vlm_dim
        self.q = FeatureCritic(feature_dim - vlm_dim + output_dim, action_shape, hidden)

    def forward(self, observations, actions):
        features = observations["features"].float()
        encoded = torch.cat(
            (self.projection(features[:, : self.vlm_dim]), features[:, self.vlm_dim :]), -1
        )
        return self.q({"features": encoded}, actions)


class ProjectedIQLV(nn.Module):
    def __init__(self, projection, feature_dim, hidden, vlm_dim=2048, output_dim=64):
        super().__init__()
        self.projection, self.vlm_dim = projection, vlm_dim
        self.v = FeatureValue(feature_dim - vlm_dim + output_dim, hidden)

    def forward(self, observations):
        features = observations["features"].float()
        encoded = torch.cat(
            (self.projection(features[:, : self.vlm_dim]), features[:, self.vlm_dim :]), -1
        )
        return self.v({"features": encoded})
