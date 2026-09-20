"""Small reference networks; replace these with robot/model-specific modules."""

import math

import torch
from torch import nn


def fourier_time_embed(time: torch.Tensor, embed_dim: int = 16) -> torch.Tensor:
    if embed_dim < 2 or embed_dim % 2:
        raise ValueError("time embedding dimension must be positive and even")
    time = time.reshape(-1, 1).float()
    frequencies = torch.exp(torch.linspace(0, math.log(256), embed_dim // 2, device=time.device))
    phase = time * frequencies
    return torch.cat((phase.sin(), phase.cos()), -1)


def mlp(input_dim, output_dim, hidden_dims, layer_norm=False):
    layers = []
    for width in hidden_dims:
        layers.append(nn.Linear(input_dim, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.GELU())
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class FeatureFlowActor(nn.Module):
    """CPU-testable flow actor on observations['features']; not a VLA replacement."""

    def __init__(self, feature_dim, action_shape, hidden_dims=(512,) * 4):
        super().__init__()
        self.action_shape = tuple(action_shape)
        action_dim = math.prod(action_shape)
        self.net = mlp(feature_dim + action_dim + 1, action_dim, hidden_dims)

    def forward(self, observations, actions, time):
        inputs = torch.cat(
            (
                observations["features"].float(),
                actions.flatten(1).float(),
                time.reshape(-1, 1).float(),
            ),
            -1,
        )
        return self.net(inputs).reshape_as(actions)


class FeatureCritic(nn.Module):
    """Ensemble output [E, B]; optional Fourier-time conditioning for inner V."""

    def __init__(
        self, feature_dim, action_shape, hidden_dims=(512,) * 4, ensemble_size=2, time_embed_dim=0
    ):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.time_embed_dim = time_embed_dim
        self.heads = nn.ModuleList(
            [
                mlp(feature_dim + math.prod(action_shape) + time_embed_dim, 1, hidden_dims, True)
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, observations, actions, time=None):
        inputs = [observations["features"].float(), actions.flatten(1).float()]
        if self.time_embed_dim:
            if time is None:
                raise ValueError("Time-conditioned critic requires time")
            inputs.append(fourier_time_embed(time, self.time_embed_dim))
        combined = torch.cat(inputs, -1)
        return torch.stack([head(combined).squeeze(-1) for head in self.heads])
