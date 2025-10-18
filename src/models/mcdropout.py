"""MC Dropout model based on Gal & Ghahramani (2016)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.models.losses import gaussian_mixture_mc_nll
from src.utils.utils_logging import logger


class MCDropout(BaseModel):
    """Monte Carlo Dropout model. Based on Gal & Ghahramani (2016).

    Uses dropout at train and test time to approximate Bayesian inference.
    Predicts [mu, log_var] and estimates epistemic + aleatoric uncertainty
    via multiple stochastic forward passes.
    """

    def __init__(
        self,
        length_scale: float,
        tau: float,
        n_mc_samples: int,
        **kwargs,
    ) -> None:
        """Initialize MC Dropout model.

        Args:
            n_mc_samples: Number of Monte Carlo forward passes at inference time.
            length_scale: Prior length scale parameter (for weight decay / regularization).
            tau: Precision (inverse variance) parameter for model noise (to scale
                 epistemic variance term).
            **kwargs: Forwarded to BaseModel (e.g. in_dim, hidden_dims, dropout_rate, etc.).
        """
        super().__init__(**kwargs)
        self.length_scale = length_scale
        self.tau = tau
        self.n_mc_samples = n_mc_samples

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Return plain linear hidden layer."""
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Builds the model head that outputs the mean and log-variance for each target dimension.

        Args:
            backbone_out_features (int): Number of features produced by the backbone network.
            target_dim (int): Dimensionality of the output target (D).
                The head predicts both mean and log-variance for each dimension.

        Returns:
            nn.Module: A DenseNormal head producing concatenated [mu, log_var]
            of shape(..., 2 * D).
        """
        return DenseNormal(in_features=backbone_out_features, target_dim=target_dim)

    def loss(self, y_true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Computes the Gaussian negative log-likelihood (NLL) from predicted mean and log-variance.

        The prediction tensor pred is expected to contain concatenated [mu, log_var]
        along the last dimension.

        Args:
            y_true (torch.Tensor): Ground truth targets.
            pred (torch.Tensor): Model predictions containing [mu, log_var].

        Returns:
            torch.Tensor: Scalar loss value representing the mean Gaussian NLL.
        """
        mu, logvar = torch.chunk(pred, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> list[float]:
        """Trains the model on the given training data.

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
        N = len(X_train)
        p = self.p_drop  # Dropout rate
        l = self.length_scale  # prior length scale
        tau = self.tau  # precision
        weight_decay = l**2 * (1 - p) / (2.0 * N * tau)

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

        losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)
                loss = self.loss(batch_y, pred)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(1, n_batches)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts mean, epistemic uncertainty, and aleatoric uncertainty via MC Dropout,
        and returns the MC mixture NLL. Dropout is kept active during sampling.

        All outputs are in the original target scale (denormalized using y_mean/y_std).

        Args:
            X_test (np.ndarray): Test inputs of shape (N, in_dim)
            y_test (np.ndarray): Ground-truth targets of shape (N,) or (N, D), in original scale.
            y_mean (np.ndarray | float): Training-set target mean (D,) or scalar.
            y_std (np.ndarray | float): Training-set target std (D,) or scalar.

        Returns:
            tuple:
                - mu (np.ndarray): Predictive mean, shape (N, D).
                - epistemic (np.ndarray): Epistemic uncertainty across MC samples, shape (N, D).
                - aleatoric (np.ndarray): Predicted aleatoric uncertainty, shape (N, D).
                - nll (np.ndarray): MC mixture negative log-likelihood (averaged over N and D).
        """

        self.train()  # Keep dropout active during MC sampling
        device = next(self.parameters()).device

        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(1)

        # Convert y_mean/y_std to tensors
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, -1)
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1)

        S = self.n_mc_samples
        mu_samples, var_samples = [], []

        # MC sampling
        for _ in range(S):
            out = self(X_test)  # (N, 2D)
            mu, log_var = torch.chunk(out, 2, dim=-1)
            mu_samples.append(mu)
            var_samples.append(torch.exp(log_var))  # σ²_s

        mu_stack = torch.stack(mu_samples, dim=0)  # (S, N, D)
        var_stack = torch.stack(var_samples, dim=0)  # (S, N, D)

        # Denormalize all stochastic samples
        mu_stack_real = mu_stack * y_std + y_mean
        var_stack_real = var_stack * (y_std**2)

        # Aggregate stochastic samples
        mu_mean_real = mu_stack_real.mean(dim=0)  # (N, D)
        epistemic_real = mu_stack_real.var(dim=0, unbiased=False)  # (N, D)
        aleatoric_real = var_stack_real.mean(dim=0)  # (N, D)

        # MC-based NLL
        mc_nll = gaussian_mixture_mc_nll(
            y=y_test,
            mu_stack=mu_stack_real,
            var_stack=var_stack_real,
            reduce=True,
        )

        return (
            mu_mean_real.cpu().numpy(),
            epistemic_real.cpu().numpy(),
            aleatoric_real.cpu().numpy(),
            mc_nll.cpu().numpy(),
        )
