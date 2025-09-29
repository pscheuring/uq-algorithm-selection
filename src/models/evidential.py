import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.losses import der_loss
from src.utils.utils_logging import logger
from src.models.base_model import BaseModel
from src.models.layers import DenseNormalGamma


class DeepEvidentialRegression(BaseModel):
    """Deep Evidential Regression model.

    Based on Amini et al. (2020). Predicts parameters of a Normal-Inverse-Gamma
    distribution to capture aleatoric and epistemic uncertainty.
    """

    def __init__(self, coeff: float, **kwargs) -> None:
        """Init model with DER-specific loss coefficient.

        Args:
            *args: Forwarded to BaseModel.
            coeff: Regularization coefficient for DER loss.
            **kwargs: Forwarded to BaseModel.
        """
        super().__init__(**kwargs)
        self.coeff = float(coeff)

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Standard linear hidden layer."""
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Head that outputs [mu, v, alpha, beta] for each target dim."""
        return DenseNormalGamma(
            in_features=backbone_out_features, target_dim=target_dim
        )

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """DER loss with regularization term."""
        return der_loss(y_true, head_out, coeff=self.coeff)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
    ) -> list[float]:
        """Train model by minimizing DER loss.

        Args:
            X_train: Training inputs, shape (N, in_dim).
            y_train: Training targets, shape (N,) or (N, D).
            shuffle: Shuffle dataset each epoch.
            clip_grad_norm: If set, clip gradient norm.

        Returns:
            List of average epoch losses.
        """
        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if X_train.ndim == 1:
            X_train = X_train.unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        losses: list[float] = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)  # (B, 4*D)
                loss = self.loss(batch_y, pred)
                loss.backward()
                # if clip_grad_norm is not None:
                #     torch.nn.utils.clip_grad_norm_(
                #         self.parameters(), max_norm=clip_grad_norm
                #     )
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / n_batches
            losses.append(avg)
            logger.info(f"Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict mean and decompose uncertainty.

        Returns:
            Tuple (mu, epistemic, aleatoric), each shaped (N, D).
        """
        self.eval()
        device = next(self.parameters()).device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        pred = self(X_test)
        gamma, v, alpha, beta = torch.chunk(pred, 4, dim=-1)

        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (v * (alpha - 1.0))

        return gamma.cpu().numpy(), epistemic.cpu().numpy(), aleatoric.cpu().numpy()
