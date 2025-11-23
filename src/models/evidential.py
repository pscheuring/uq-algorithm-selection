"""Deep Evidential Regression model based on Amini et al. (2020)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import DenseNormalGamma
from src.models.losses import _nig_nll, der_loss
from src.utils.utils_logging import logger


class DeepEvidentialRegression(BaseModel):
    """Deep Evidential Regression model.

    Based on Amini et al. (2020). Predicts parameters of a Normal-Inverse-Gamma
    distribution to capture aleatoric and epistemic uncertainty.
    """

    def __init__(self, coeff: float, **kwargs) -> None:
        """Init model with DER-specific loss coefficient.

        Args:
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
    ) -> list[float]:
        """Train model by minimizing DER loss.

        Args:
            X_train (np.ndarray): Training inputs of shape (N, in_dim).
            y_train (np.ndarray): Training targets of shape (N,) or (N, D).

        Returns:
            list[float]: Average loss per epoch during training.
        """
        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if X_train.ndim == 1:
            X_train = X_train.unsqueeze(1)

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=generator,
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
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / n_batches
            losses.append(avg)
            logger.info(f"Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_mean: np.ndarray | float,
        y_std: np.ndarray | float,
        test_interval: list[float, float],
        train_interval: list[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts mean, epistemic uncertainty, aleatoric uncertainty, and NIG-based NLL
        for Deep Evidential Regression (Normal–Inverse-Gamma model).

        The model outputs the NIG parameters (γ, v, α, β) for each target dimension.
        These parameters define a Normal–Inverse-Gamma distribution over the target
        mean and variance. From these, the predictive mean and uncertainty
        components are computed and returned in the original target scale.

        The NLL is computed only over rows of X_test whose inputs lie inside
        the interval [lo, hi] specified by test_interval. Predictions
        (mu, epistemic, aleatoric) are returned for all test samples.

        Args:
            X_test (np.ndarray): Test inputs of shape (N, in_dim)
            y_test (np.ndarray): Ground-truth targets in original scale, shape (N,) or (N, D).
            y_mean (np.ndarray | float): Training target mean(s), shape (D,) or scalar.
            y_std (np.ndarray | float): Training target standard deviation(s), shape (D,) or scalar.
            test_interval (list[float, float]): Interval [lo, hi]; used to select test samples
                included in the NLL computation.
            train_interval (list[float, float]): Interval [lo, hi]; used to select training samples
                included in the NLL computation.

        Returns:
            tuple:
                - mu (np.ndarray): Predictive mean γ, shape (N, D)
                - epistemic (np.ndarray): Epistemic uncertainty, shape (N, D), computed as β / (v * (α - 1))
                - aleatoric (np.ndarray): Aleatoric uncertainty, shape (N, D), computed as β / (α - 1)
                - nll_id (np.ndarray): NIG negative log-likelihood on ID data
                - nll_ood (np.ndarray): NIG negative log-likelihood on full test interval (including OOD)
        """
        self.eval()
        device = next(self.parameters()).device

        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(1)

        N = X_test.shape[0]
        D = y_test.shape[1]

        # Convert y_mean/y_std to tensors
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, -1)
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1)

        pred = self(X_test)
        gamma_norm, v, alpha, beta_norm = torch.chunk(pred, 4, dim=-1)

        gamma = gamma_norm * y_std + y_mean  # (N, D)
        beta = beta_norm * (y_std**2)  # (N, D)
        assert gamma.shape == (N, D)
        assert beta.shape == (N, D)

        aleatoric = beta / (alpha - 1.0)  # E[σ^2]
        epistemic = beta / (v * (alpha - 1.0))  # Var[μ]

        # Build OOD mask
        lo_test, hi_test = map(float, test_interval)
        if X_test.ndim == 1:
            X_mask_ood = (X_test >= lo_test) & (X_test <= hi_test)
        else:
            X_mask_ood = ((X_test >= lo_test) & (X_test <= hi_test)).all(dim=1)  # (N,)

        # Build ID mask
        train_interval = np.array(train_interval, dtype=float)
        if train_interval.ndim == 2:  # in-between OOD
            lo_train = train_interval[:, 0].min()
            hi_train = train_interval[:, 1].max()
        else:
            lo_train, hi_train = map(float, train_interval)
        if X_test.ndim == 1:
            X_mask_id = (X_test >= lo_train) & (X_test <= hi_train)
        else:
            X_mask_id = ((X_test >= lo_train) & (X_test <= hi_train)).all(dim=1)  # (N,)

        # Normal-Inverse-Gamma NLL
        nll_ood = _nig_nll(
            y_test[X_mask_ood],
            gamma[X_mask_ood],
            v[X_mask_ood],
            alpha[X_mask_ood],
            beta[X_mask_ood],
            reduce=True,
            per_dimension=True,
        )

        nll_id = _nig_nll(
            y_test[X_mask_id],
            gamma[X_mask_id],
            v[X_mask_id],
            alpha[X_mask_id],
            beta[X_mask_id],
            reduce=True,
            per_dimension=True,
        )

        return (
            gamma.cpu().numpy(),
            epistemic.cpu().numpy(),
            aleatoric.cpu().numpy(),
            nll_ood.cpu().numpy(),
            nll_id.cpu().numpy(),
        )
