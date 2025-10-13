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
        self.n_mc_samples = n_mc_samples

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Return plain linear hidden layer."""
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Return probabilistic head projecting to [mu, log_var]."""
        return DenseNormal(in_features=backbone_out_features, target_dim=target_dim)

    def loss(self, y_true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL with predicted mean and variance."""
        mu, logvar = torch.chunk(pred, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
    ) -> list[float]:
        """Train model on (X, y).

        Args:
            X_train: Training inputs, shape (N, in_dim).
            y_train: Training targets, shape (N,) or (N, D).
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

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=generator,
        )
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        losses: list[float] = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)
                loss = self.loss(batch_y, pred)
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), max_norm=clip_grad_norm
                    )
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Predict mean, epistemic, aleatoric variances and MC-based NLL via MC Dropout."""
        self.train()  # keep dropout active during MC sampling
        device = next(self.parameters()).device

        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(1)

        S = int(self.n_mc_samples)
        mu_samples, var_samples = [], []

        # MC sampling
        for _ in range(S):
            out = self(X_test)  # (N, 2D)
            mu, log_var = torch.chunk(out, 2, dim=-1)
            mu_samples.append(mu)
            var_samples.append(torch.exp(log_var))  # σ²_s

        mu_stack = torch.stack(mu_samples, dim=0)  # (S, N, D)
        var_stack = torch.stack(var_samples, dim=0)  # (S, N, D)

        # Aggregate statistics
        mu_mean = mu_stack.mean(dim=0)  # (N, D)
        epistemic = mu_stack.var(dim=0, unbiased=False)  # (N, D)
        aleatoric = var_stack.mean(dim=0)  # (N, D)

        # MC-based NLL
        nll_tensor = gaussian_mixture_mc_nll(
            y=y_test, mu_stack=mu_stack, var_stack=var_stack, reduce=True
        )
        nll = float(nll_tensor.item())

        return (
            mu_mean.cpu().numpy(),
            epistemic.cpu().numpy(),
            aleatoric.cpu().numpy(),
            nll,
        )
