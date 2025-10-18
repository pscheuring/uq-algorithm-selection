"""Deep Ensemble model based on Lakshminarayanan et al. (2017)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.models.losses import gaussian_mixture_mc_nll
from src.utils.utils_logging import logger


class DeepEnsembleMember(BaseModel):
    """One member of a deep ensemble (standard MLP + probabilistic head).

    Each member outputs the parameters of a factorized Gaussian per target
    dimension (mean and variance) to capture aleatoric uncertainty.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize one ensemble member.

        Args:
            **kwargs: Forwarded to BaseModel (e.g., in_dim, hidden_dims, etc.).
        """
        super().__init__(**kwargs)

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
        return DenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
        )

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """Computes the Gaussian negative log-likelihood (NLL) from predicted mean and log-variance.

        The prediction tensor pred is expected to contain concatenated [mu, log_var]
        along the last dimension.

        Args:
            y_true (torch.Tensor): Ground truth targets.
            pred (torch.Tensor): Model predictions containing [mu, log_var].

        Returns:
            torch.Tensor: Scalar loss value representing the mean Gaussian NLL.
        """
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit_member(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        member_seed: int,
    ) -> dict[str, list[float]]:
        """Trains a single ensemble member by minimizing the Gaussian NLL.

        Args:
            X_train (np.ndarray): Training inputs of shape (N, in_dim).
            y_train (np.ndarray): Training targets of shape (N,) or (N, D).
            member_seed (int): Random seed used to initialize the data shuffling and dropout.

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

        generator = torch.Generator(device="cpu").manual_seed(member_seed)

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

        losses = []
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)  # (B, 2*D)
                loss = self.loss(batch_y, pred)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(1, n_batches)
            losses.append(avg)
            logger.info(f"[Member] Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties_member(
        self, X_test: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance for one member.

        Args:
            X_test: Input array, shape (N, in_dim) or (N,) for 1D inputs.

        Returns:
            Tuple (mu, var), each shaped (N, D).
        """
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        out = self(X_test)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        var = torch.exp(logvar)
        return mu, var


class DeepEnsemble:
    """Collection of independently trained ensemble members.

    Based on Lakshminarayanan et al. (2017). Aggregates member predictions to
    decompose predictive uncertainty into epistemic and aleatoric parts.
    """

    def __init__(self, seed: int, n_members: int, **kwargs) -> None:
        """Init deep ensemble with n_members.

        Args:
            seed: Base random seed; member m uses seed + m.
            n_members: Number of ensemble members to create/train.
            **kwargs: Forwarded to DeepEnsembleMember constructor/BaseModel.
        """
        self.seed = int(seed)
        self.members: list[DeepEnsembleMember] = []

        for m_idx in range(n_members):
            logger.debug(f"[DeepEnsemble] Creating member {m_idx + 1}/{n_members}")
            member = DeepEnsembleMember(**kwargs, seed=self.seed + m_idx)
            self.members.append(member)

            # Quick sanity check to confirm initialization differs across members.
            w0 = next(member.parameters()).detach().view(-1)[0].item()
            logger.debug(f"[DeepEnsemble] Member {m_idx + 1} first param[0]={w0:.6f}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> list[float]:
        """Train all ensemble members independently.

        Args:
            X_train: Training inputs, shape (N, in_dim).
            y_train: Training targets, shape (N,) or (N, D).

        Returns:
            list[float]: Average loss per epoch during training
        """
        losses: list[dict[str, list[float]]] = []
        for m_idx, member in enumerate(self.members):
            member_seed = self.seed + m_idx
            logger.info(
                f"\n=== Train member {m_idx + 1}/{len(self.members)} (seed={member_seed}) ==="
            )
            loss = member.fit_member(
                X_train,
                y_train,
                member_seed=member_seed,
            )
            losses.append(loss)
        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_mean: np.ndarray | float,
        y_std: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Aggregate predictions across ensemble members and compute NLL over all members.

        Args:
            X_test (np.ndarray): Test inputs, shape (N, in_dim).
            y_test (np.ndarray): Ground-truth targets, shape (N,) or (N, D).

        Returns:
            Tuple:
                - mu_mean (np.ndarray): Predictive mean, shape (N, D)
                - epistemic (np.ndarray): Epistemic variance, shape (N, D)
                - aleatoric (np.ndarray): Aleatoric variance, shape (N, D)
                - nll (float): Negative log-likelihood over all members
        """
        device = self.members[0].device

        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
        if y_test.ndim == 1:
            y_test = y_test.unsqueeze(1)
        if X_test.ndim == 1:
            X_test = X_test.unsqueeze(1)

        # Convert y_mean/y_std to tensors
        y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device).view(1, -1)
        y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device).view(1, -1)

        mu_samples, var_samples = [], []

        # Sample over ensemble members
        for member in self.members:
            mu, var = member.predict_with_uncertainties_member(X_test)
            mu_samples.append(mu)  # (N, D)
            var_samples.append(var)  # (N, D)

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
