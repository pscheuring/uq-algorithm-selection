"""Bayes by Backprop model based on Blundell et al. (2015)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import VariationalDense, VariationalDenseNormal
from src.models.losses import elbo_loss, gaussian_mixture_mc_nll
from src.utils.utils_logging import logger


class BayesByBackprop(BaseModel):
    """Bayesian NN with mean-field variational inference (Bayes by Backprop).

    Based on Blundell et al. (2015). Trains with ELBO and a front-loaded beta
    schedule per batch. Head outputs [mu, log_var].
    """

    def __init__(
        self,
        kl_beta: float,
        prior_mean: float,
        prior_sigma: float,
        mu_init_mean: float,
        mu_init_std: float,
        rho_init: float,
        n_mc_samples: int,
        **kwargs,
    ) -> None:
        """Init model and store prior/posterior hyperparameters.

        Args:
            kl_beta: Global KL weight.
            prior_mean: mean of gaussian prior.
            prior_sigma: std of gaussian prior.
            mu_init_mean: Mean for init of posterior mus.
            mu_init_std: Std for init of posterior mus.
            rho_init: Init value for posterior rhos.
            n_mc_samples: Number of MC samples at test time.
            **kwargs: Forwarded to BaseModel.
        """
        self._prior: dict[str, float] = {
            "prior_mean": prior_mean,
            "prior_sigma": prior_sigma,
        }
        self._post_init: dict[str, float] = {
            "mu_init_mean": mu_init_mean,
            "mu_init_std": mu_init_std,
            "rho_init": rho_init,
        }
        self.kl_beta: float = kl_beta
        self.n_mc_samples: int = int(n_mc_samples)
        self.dataset_size: int | None = None

        # call after setting _prior/_post_init, because BaseModel.__init__ uses them
        super().__init__(**kwargs)

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Return a variational dense layer (in_dim -> out_dim)."""
        return VariationalDense(
            in_features=in_features,
            out_features=hidden_features,
            **self._prior,
            **self._post_init,
        )

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Builds the variational model head that outputs the mean and log-variance for each target dimension.

        Args:
            backbone_out_features (int): Number of features produced by the backbone network.
            target_dim (int): Dimensionality of the output target (D).
                The head predicts both mean and log-variance for each dimension.

        Returns:
            nn.Module: A VariationalDenseNormal head producing concatenated [mu, log_var]
            of shape(..., 2 * D).
        """
        return VariationalDenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
            **self._prior,
            **self._post_init,
        )

    def kl_total_loss(self) -> torch.Tensor:
        """Sum KL divergences from all variational submodules."""
        kl_total = torch.zeros((), device=self.device)
        for m in self.modules():
            if hasattr(m, "kl_loss") and callable(getattr(m, "kl_loss")):
                kl_total = kl_total + m.kl_loss()
        return kl_total

    def loss(
        self, y_true: torch.Tensor, head_out: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """Compute negative ELBO (NLL_mean + beta*KL/N)."""
        kl = self.kl_total_loss()
        assert self.dataset_size is not None, "dataset_size not set, call fit() first"
        return elbo_loss(y_true, head_out, kl, N=self.dataset_size, beta=beta)

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

        self.dataset_size = int(X_train.shape[0])

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
            num_batches = len(loader)

            # front-loaded beta schedule
            M = num_batches
            den = 2.0 - 2.0 ** (-(M - 1))

            for b_idx, (batch_x, batch_y) in enumerate(loader, start=1):
                pi_i = 2.0 ** (-(b_idx - 1)) / den
                beta = self.kl_beta * pi_i

                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)
                loss = self.loss(batch_y, pred, beta=beta)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg = epoch_loss / n_batches
            losses.append(avg)
            logger.info(f"Epoch {epoch:3d}/{self.epochs}  elbo={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_mean: np.ndarray | float,
        y_std: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts mean, epistemic uncertainty, aleatoric uncertainty, and MC-based NLL using Bayes by Backprop.

        Performs multiple stochastic forward passes by sampling weights from the learned
        variational posterior (BBB). The predictive mean and uncertainties are computed
        from these weight samples. All returned quantities are in the original target scale.

        Args:
            X_test (np.ndarray): Test inputs of shape (N, in_dim)
            y_test (np.ndarray): Ground-truth targets of shape (N,) or (N, D), in original scale.
            y_mean (np.ndarray | float): Training-set target mean, shape (D,) or scalar.
            y_std (np.ndarray | float): Training-set target std, shape (D,) or scalar.

        Returns:
            tuple:
                - mu (np.ndarray): Predictive mean, shape (N, D).
                - epistemic (np.ndarray): Epistemic uncertainty (across BBB MC samples), shape (N, D).
                - aleatoric (np.ndarray): Predicted aleatoric uncertainty, shape (N, D).
                - nll (np.ndarray): MC mixture negative log-likelihood averaged over the batch.
        """
        self.eval()
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
