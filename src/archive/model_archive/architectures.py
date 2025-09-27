from typing import Iterable
import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """
    Einfaches MLP-Backbone.
    Default: 2 Hidden-Layer à 64 Neuronen, ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Iterable[int] = (64, 64),
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(
                activation if isinstance(activation, nn.Module) else activation()
            )
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim  # Feature-Dimension für den Kopf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
