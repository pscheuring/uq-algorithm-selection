import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.base_model import BaseModel
from src.models.layers import DenseNormal

from src.logging import logger


class MCDropout(BaseModel):
    def __init__(
        self,
        *args,
        dropout: float = 0.1,
        n_mc_samples: int = 50,
        device: str = None,
        **kwargs,
    ):
        super().__init__(*args, dropout=dropout, **kwargs)

    def make_hidden_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Linear(in_dim, out_dim)

    def make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        return DenseNormal(in_dim, out_dim)

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shuffle: bool = True,
        clip_grad_norm: float | None = None,
        verbose: bool = True,
    ):
        """
        Trainiert das Modell auf (X, y).
        - X: (N, input_dim)
        - y: (N,) oder (N, output_dim)
        """
        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
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

                pred = self(batch_x)  # (B, 4*output_dim)
                loss = self.loss(batch_y, pred)  # <- deine DER-Loss

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
            logger.debug(f"Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return losses

    def predict_with_uncertainties(
        self,
        X_test: np.ndarray,
        n_mc_samples: int = 50,
    ):
        """
        MC Dropout-Predictions.
        Gibt (mu_mean, epistemic, aleatoric) zurück.
        """
        self.train()  # wichtig: Dropout bleibt aktiv!
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)

        mu_samples, var_samples = [], []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                out = self(X_test).cpu().numpy()
                mu, sigma = np.split(out, 2, axis=-1)
                mu_samples.append(mu)
                var_samples.append(sigma**2)

        mu_samples = np.stack(mu_samples, axis=0)  # (T, N, D)
        var_samples = np.stack(var_samples, axis=0)  # (T, N, D)

        mu_mean = mu_samples.mean(axis=0)  # (N, D)
        epistemic = mu_samples.var(axis=0)  # (N, D)
        aleatoric = var_samples.mean(axis=0)  # (N, D)

        return mu_mean, epistemic, aleatoric
