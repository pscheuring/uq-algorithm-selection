from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn


def create_activations(activation: list[str]) -> list[nn.Module]:
    """Create activation modules from a list of activation names.

    Args:
        activation: List of activation names, e.g. ["relu", "tanh"].

    Returns:
        List of instantiated nn.Module activations.

    Raises:
        ValueError: If an unknown activation name is provided.
    """
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
    """Abstract base class for deep probabilistic regression models.

    Builds a configurable feed-forward backbone consisting of linear layers,
    activations, and optional dropout, followed by a model-specific output head.

    Subclasses must define:
        - make_hidden_layer: how to construct a single hidden layer,
        - make_head: how to construct the output head,
        - loss: model-specific training loss,
        - fit: the training loop,
        - predict_with_uncertainties: inference with uncertainty decomposition.

    Architecture:
        Input → [Dropout → Linear → Activation] × n → Head → Output
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Iterable[int],
        activations: list[str],
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        shuffle: bool,
        target_dim: int,
        seed: int,
        p_drop: float = 0.0,
        device: str | None = None,
    ) -> None:
        """Initialize the base model.

        Args:
            in_features: Number of input features.
            hidden_features: Sizes of each hidden layer.
            activations: List of activation names, e.g. ["relu", "tanh"].
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            weight_decay: Weight decay for optimizer.
            shuffle: Whether to shuffle data during training.
            target_dim: Output dimensionality of the model head.
            seed: Random seed for reproducibility.
            p_drop: Dropout probability applied after each activation.
            device: Torch device string (e.g. "cuda", "cpu"). If None, auto-selects CUDA if available.

        Notes:
            The number of activations should match the number of hidden layers.
            If mismatched, `zip()` will silently truncate.
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
        self.seed: int = int(seed)
        self.device: torch.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build backbone: [Dropout → Linear → Activation] × n
        layers: list[nn.Module] = []
        current_in_features = self.in_features
        for activation, h in zip(self.activations, self.hidden_features):
            if self.p_drop > 0.0:
                layers.append(nn.Dropout(self.p_drop))
            layers.append(self.make_hidden_layer(current_in_features, h))
            layers.append(activation)
            current_in_features = h

        self.backbone: nn.Sequential = nn.Sequential(*layers)
        self.backbone_out_features: int = current_in_features

        # Construct model head
        self.head: nn.Module = self.make_head(
            self.backbone_out_features, self.target_dim
        )

        # Move model to target device
        self.to(self.device)

    @abstractmethod
    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Create a single hidden layer.

        Args:
            in_features: Input feature dimensionality.
            hidden_features: Output feature dimensionality.

        Returns:
            Module mapping in_features → hidden_features.
        """
        raise NotImplementedError

    @abstractmethod
    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Create the output head of the model.

        Args:
            backbone_out_features: Feature size output by the backbone.
            target_dim: Desired dimensionality of the model’s output.

        Returns:
            Module mapping backbone_out_features → target_dim.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, y_true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Compute model-specific training loss.

        Args:
            y_true: Ground truth target tensor.
            pred: Model prediction tensor (output of forward()).

        Returns:
            Scalar tensor representing the loss.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> list[float]:
        """Train the model.

        Args:
            X_train: Training inputs, shape (N, in_features).
            y_train: Training targets, shape (N,) or (N, D).

        Returns:
            List of average loss values per epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict mean and uncertainties on test data.

        Args:
            X_test: Input array, shape (N, in_features).

        Returns:
            Tuple (mu, epistemic, aleatoric), each shaped (N, D).
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and head.

        Args:
            x: Input tensor, shape (B, in_features).

        Returns:
            Output tensor, shape (B, target_dim).
        """
        return self.head(self.backbone(x))
