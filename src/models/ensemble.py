import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.utils.utils_logging import logger
from src.utils.utils_pipeline import set_global_seed


class DeepEnsembleMember(BaseModel):
    """One member of a deep ensemble (standard MLP + probabilistic head)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Return a plain linear hidden layer (no variational layers)."""
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Return probabilistic head that outputs [mu, log_var]."""
        return DenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
        )

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL with predicted mean and variance."""
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
        member_seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Train one ensemble member.

        Args:
            X_train: Training inputs, shape (N, in_dim).
            y_train: Training targets, shape (N,) or (N, D).
            shuffle: Shuffle data each epoch.
            clip_grad_norm: If set, clip gradient norm.
            verbose: If True, print progress.
            member_seed: Optional seed for this member.

        Returns:
            Dict with key "losses" containing average loss per epoch.
        """
        set_global_seed(member_seed)

        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

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
                pred = self(batch_x)  # (B, 2*D)
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
            logger.info(f"[Member] Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return {"losses": losses}

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for one member.

        Args:
            X: Input array, shape (N, in_dim).

        Returns:
            Tuple (mu, var), each shaped (N, D).
        """
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        out = self(X_test)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        var = torch.exp(logvar)
        mu = mu.cpu().numpy()
        var = var.cpu().numpy()
        return mu, var


class DeepEnsemble:
    """Collection of independently trained ensemble members. Based on Lakshminarayanan et al. (2017)."""

    def __init__(self, seed: int, n_members: int, **kwargs) -> None:
        """Init deep ensemble with n_members."""
        self.members = [DeepEnsembleMember(**kwargs) for _ in range(n_members)]
        self.seed = seed

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
    ) -> list[float]:
        """Train all ensemble members independently.

        Args:
            X_train: Training inputs.
            y_train: Training targets.
            shuffle_each: If True, shuffle batches per member.
            clip_grad_norm: Optional gradient clipping.

        Returns:
            List of dicts with training losses per member.
        """
        losses: list[dict[str, list[float]]] = []
        for m_idx, member in enumerate(self.members):
            member_seed = self.seed + m_idx
            logger.debug(
                f"\n=== Train member {m_idx + 1}/{len(self.members)} (seed={member_seed}) ==="
            )
            loss = member.fit(
                X_train,
                y_train,
                clip_grad_norm=clip_grad_norm,
                member_seed=member_seed,
            )
            losses.append(loss)
        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate predictions across members.

        Computes:
            mu_mean = average of member means
            epistemic = variance across member means
            aleatoric = average of member variances

        Args:
            X: Input array, shape (N, in_dim).

        Returns:
            Tuple (mu_mean, epistemic, aleatoric), each shaped (N, D).
        """
        mus, vars_ = [], []
        for member in self.members:
            mu, var = member.predict_with_uncertainties(X_test)
            mus.append(mu)
            vars_.append(var)

        mu_stack = np.stack(mus, axis=0)  # (M, N, D)
        var_stack = np.stack(vars_, axis=0)  # (M, N, D)

        mu_mean = mu_stack.mean(axis=0)  # (N, D)
        epistemic = mu_stack.var(axis=0, ddof=0)  # (N, D)
        aleatoric = var_stack.mean(axis=0)  # (N, D)

        return mu_mean, epistemic, aleatoric
