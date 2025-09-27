from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn


def create_activations(activation: list[str]) -> list[nn.Module]:
    """Convert a list of activation names into their corresponding nn.Module instances."""
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "softplus": nn.Softplus,
    }

    modules: list[nn.Module] = []
    for name in activation:
        key = name.lower()
        if key not in mapping:
            raise ValueError(f"Unknown Activation: {name}")
        modules.append(mapping[key]())
    return modules


class BaseModel(ABC, nn.Module):
    """Abstract base for models with a configurable backbone plus custom head and loss.

    Builds a stack of hidden layers with activations and optional dropout.
    Subclasses define the hidden layer type, the head, and the loss function.

    Architecture: Input -> [Hidden -> Activation -> (Dropout)] x n -> Head -> Output
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Iterable[int],
        activations: nn.Module,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        shuffle: bool,
        target_dim: int,
        p_drop: float = 0.0,
        device: str | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            in_features: Number of input features.
            hidden_features: Hidden layer sizes, one per layer.
            activations: Activation spec processed by create_activations.
            epochs: Planned number of training epochs (informational).
            batch_size: Planned batch size (informational).
            lr: Planned learning rate (informational).
            weight_decay: Planned weight decay (informational).
            shuffle: Whether to shuffle data during training (informational).
            target_dim: Output dimensionality from the head.
            p_drop: Dropout probability after each activation.
            device: Torch device string (e.g. cuda, cpu). If None, auto-selects cuda if available.

        Notes:
            create_activations(activations) must return as many activations as hidden layers.
            If not, zip will truncate silently.
        """
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_features: list[int] = [int(h) for h in hidden_features]
        self.activations: list[nn.Module] = create_activations(activations)
        self.epochs: int = int(epochs)
        self.batch_size: int = int(batch_size)
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.shuffle: bool = bool(shuffle)
        self.target_dim: int = int(target_dim)
        self.p_drop: float = float(p_drop)
        self.device: torch.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build backbone: [Hidden -> Activation -> (Dropout)]*
        layers: list[nn.Module] = []
        current_in_features = self.in_features
        for activation, h in zip(self.activations, self.hidden_features):
            layers.append(self.make_hidden_layer(current_in_features, h))
            layers.append(activation)
            if self.p_drop > 0.0:
                layers.append(nn.Dropout(self.p_drop))
            current_in_features = h

        self.backbone: nn.Sequential = nn.Sequential(*layers)
        self.backbone_out_features: int = current_in_features

        # Build head
        self.head: nn.Module = self.make_head(
            self.backbone_out_features, self.target_dim
        )

        # Device placement
        self.to(self.device)

    @abstractmethod
    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Create one hidden layer.

        Args:
            in_features: Input feature size.
            hidden_features: Output feature size.

        Returns:
            Module mapping in_features -> hidden_features.
        """
        raise NotImplementedError

    @abstractmethod
    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Create the output head.

        Args:
            backbone_out_features: Feature size output by the backbone.
            target_dim: Desired output dimensionality.

        Returns:
            Module mapping backbone_out_features -> target_dim.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, y_true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Compute training loss for targets and predictions.

        Args:
            y_true: Target tensor.
            pred: Prediction tensor from forward().

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head.

        Args:
            x: Input tensor shaped (batch_size, in_features).

        Returns:
            Output tensor shaped (batch_size, target_dim).
        """
        return self.head(self.backbone(x))
