import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
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
        """Standard linear hidden layer.

        Args:
            in_features: Number of input features to the layer.
            hidden_features: Number of output features (hidden units).

        Returns:
            A plain nn.Linear layer (no variational components).
        """
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Head that outputs [mu, log_var] for each target dim.

        Args:
            backbone_out_features: Number of features from the backbone.
            target_dim: Output target dimensionality D.

        Returns:
            A DenseNormal head producing concatenated [mu, log_var] (shape (..., 2*D)).
        """
        return DenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
        )

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood with predicted mean/variance.

        Args:
            y_true: Ground-truth targets, shape (B, D).
            head_out: Head outputs, concatenated [mu, log_var], shape (B, 2*D).

        Returns:
            Scalar loss (mean NLL over batch).
        """
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        member_seed: int,
        clip_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        """Train this ensemble member by minimizing Gaussian NLL.

        Args:
            X_train: Training inputs, shape (N, in_dim) or (N,) for 1D inputs.
            y_train: Training targets, shape (N,) or (N, D).
            clip_grad_norm: If set, clip gradient norm to this value.
            member_seed: RNG seed used for shuffling batches.

        Returns:
            Dict with key `"losses"` containing the average loss per epoch.
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
            X_test: Input array, shape (N, in_dim) or (N,) for 1D inputs.

        Returns:
            Tuple (mu, var), each shaped (N, D).
        """
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        out = self(X_test)  # (N, 2*D)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        var = torch.exp(logvar)
        mu = mu.cpu().numpy()
        var = var.cpu().numpy()
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
            member = DeepEnsembleMember(**kwargs, seed=self.seed)
            self.members.append(member)

            # Quick sanity check to confirm initialization differs across members.
            w0 = next(member.parameters()).detach().view(-1)[0].item()
            logger.debug(f"[DeepEnsemble] Member {m_idx + 1} first param[0]={w0:.6f}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
    ) -> list[dict[str, list[float]]]:
        """Train all ensemble members independently.

        Args:
            X_train: Training inputs, shape (N, in_dim) or (N,) for 1D inputs.
            y_train: Training targets, shape (N,) or (N, D).
            clip_grad_norm: Optional gradient clipping value.

        Returns:
            List of dicts with training losses per member, each having key "losses".
        """
        losses: list[dict[str, list[float]]] = []
        for m_idx, member in enumerate(self.members):
            member_seed = self.seed + m_idx
            logger.info(
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
            - mu_mean: Average of member means.
            - epistemic: Variance across member means.
            - aleatoric: Average of member variances.

        Args:
            X_test: Input array, shape (N, in_dim) or (N,) for 1D inputs.

        Returns:
            Tuple (mu_mean, epistemic, aleatoric), each shaped (N, D).
        """
        mus, vars_ = [], []
        for member in self.members:
            mu, var = member.predict_with_uncertainties(X_test)
            mus.append(mu)
            vars_.append(var)

        mu_stack = np.stack(mus, axis=0)
        var_stack = np.stack(vars_, axis=0)

        mu_mean = mu_stack.mean(axis=0)
        epistemic = mu_stack.var(axis=0, ddof=0)
        aleatoric = var_stack.mean(axis=0)

        return mu_mean, epistemic, aleatoric
