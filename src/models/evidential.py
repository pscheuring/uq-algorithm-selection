import torch
import torch.nn as nn
from models.base_model import BaseModel
from src.models.layers import DenseNormalGamma
from models.archive.der import der_loss

import numpy as np
import torch.optim as optim

from src.logging import logger


class DeepEvidentialRegression(BaseModel):
    def __init__(
        self,
        *args,
        coeff: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.coeff = float(coeff)

    def make_hidden_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Linear(in_dim, out_dim)

    def make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        return DenseNormalGamma(in_dim, out_dim)

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        # head_out = concat([mu, v, alpha, beta])
        return der_loss(y_true, head_out, coeff=self.coeff)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shuffle: bool = True,
        clip_grad_norm: float | None = None,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gibt (mu, epistemic, aleatoric) zurück.
        Shapes: (N, D)
        """
        self.eval()
        device = next(self.parameters()).device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        with torch.no_grad():
            pred = self(X_test).cpu().numpy()  # (N, 4*D)

        gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        # Numerische Sicherheit
        eps = 1e-8
        nu = np.maximum(nu, eps)
        alpha = np.maximum(alpha, 1.0 + eps)  # sollte eh >1 sein
        beta = np.maximum(beta, eps)

        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))

        return gamma, epistemic, aleatoric
