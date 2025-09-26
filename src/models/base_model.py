from abc import ABC, abstractmethod
from typing import Iterable, List
import torch
import torch.nn as nn

from src.utils import create_activation


class BaseModel(ABC, nn.Module):
    """
    Base: definiert die Architektur (Anzahl Layer & Units), Aktivierung & Dropout.
    Subklassen bestimmen:
      - welchen Hidden-Layer-Typ wir nutzen (nn.Linear, VariationalLinear, ...)
      - welchen Head wir anhängen (DenseNormal, DenseNormalGamma, ...)
      - wie die Loss berechnet wird
    """

    def __init__(
        self,
        input_dim: int,
        hidden_units: Iterable[int],
        activations: nn.Module,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: str = None,
        p_drop: float = 0.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_units = list(int(h) for h in hidden_units)
        self.activations = create_activation(activations)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.p_drop = float(p_drop)
        self.output_dim = int(output_dim)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

        layers: List[nn.Module] = []
        in_dim = self.input_dim
        for activation, units in zip(self.activations, self.hidden_units):
            layers.append(self.make_hidden_layer(in_dim, units))
            layers.append(activation)
            if self.p_drop > 0.0:
                layers.append(nn.Dropout(p_drop))
            in_dim = units

        self.backbone = nn.Sequential(*layers)
        self.backbone_out_dim = in_dim

        # Head von Subklasse bauen
        self.head = self.make_head(self.backbone_out_dim, self.output_dim)

    # ---- Hooks, die Subklassen überschreiben können ----
    @abstractmethod
    def make_hidden_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Welche Art Hidden-Layer? (z.B. nn.Linear oder VariationalLinear)"""
        raise NotImplementedError

    @abstractmethod
    def make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        """Welcher Kopf? (z.B. DenseNormal, DenseNormalGamma, ...)"""
        raise NotImplementedError

    @abstractmethod
    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """Daten-/Likelihood-Term (z.B. Gaussian NLL oder DER-NLL)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
