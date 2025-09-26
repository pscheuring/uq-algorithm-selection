# deep_ensemble.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Dict, Any

from models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.utils import set_global_seed

from src.logging import logger


class DeepEnsembleMember(BaseModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    # Standard-MLP-Layer; kein variational stuff
    def make_hidden_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Linear(in_dim, out_dim)

    # Probabilistic Head: [mu, logvar]
    def make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        return DenseNormal(in_dim, out_dim)

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        # logvar = torch.clamp(logvar, min=-20.0, max=5.0)  # numerische Stabilität
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shuffle: bool = True,
        clip_grad_norm: float | None = None,
        verbose: bool = True,
        member_seed: Optional[int] = None,  # eigener Seed pro Mitglied
    ) -> Dict[str, list]:
        """Trainiert ein einzelnes Ensemble-Mitglied."""
        set_global_seed(member_seed)

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
            if verbose:
                print(f"[Member] Epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_mu_var(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gibt mu und var (nicht logvar) für ein Mitglied zurück."""
        self.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        out = self(X_t)
        mu_t, logvar_t = torch.chunk(out, 2, dim=-1)
        # logvar_t = torch.clamp(logvar_t, min=-20.0, max=5.0)
        var_t = torch.exp(logvar_t)
        mu = mu_t.cpu().numpy()
        var = var_t.cpu().numpy()
        return mu, var


class DeepEnsemble:
    def __init__(
        self,
        n_members: int,
        seed: int,
    ):
        """Erzeugt ein Deep Ensemble mit num_members unabhängigen Mitgliedern."""
        self.members = [DeepEnsembleMember() for _ in range(n_members)]
        self.seed = seed

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shuffle_each: bool = True,
        clip_grad_norm: float | None = None,
    ) -> List[dict]:
        """Trainiert alle Mitglieder unabhängig (Paper: volles Dataset; Bootstrapping optional)."""
        losses = []
        for m_idx, member in enumerate(self.members):
            member_seed = self.seed + m_idx
            logger.debug(
                f"\n=== Train member {m_idx + 1}/{len(self.members)} (seed={member_seed}) ==="
            )
            loss = member.fit(
                X_train,
                y_train,
                shuffle=True if shuffle_each else False,
                clip_grad_norm=clip_grad_norm,
                member_seed=member_seed,
            )
            losses.append(loss)
        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregiert Vorhersagen der Mitglieder:
          mu* = mean_m mu_m
          var* = mean_m (var_m + mu_m^2) - mu*^2
          epistemic = var_mittelwerte(mu_m)     (Model-Unsicherheit)
          aleatoric = mean_m(var_m)             (Daten-Unsicherheit)
        """
        mus, vars_ = [], []
        for member in self.members:
            mu, var = member.predict_mu_var(X)
            mus.append(mu)
            vars_.append(var)

        mu_stack = np.stack(mus, axis=0)  # [M, N, D]
        var_stack = np.stack(vars_, axis=0)  # [M, N, D]

        mu_mean = mu_stack.mean(axis=0)  # [N, D]
        epistemic = mu_stack.var(axis=0, ddof=0)  # [N, D]
        aleatoric = var_stack.mean(axis=0)  # [N, D]

        return mu_mean, epistemic, aleatoric
