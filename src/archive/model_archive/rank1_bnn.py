import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ============ Rank-1 building blocks (model-centric) ============


class GaussianParameter(nn.Module):
    """Normal(μ, σ) via reparameterization; σ = softplus(ρ)."""

    def __init__(self, dim: int, rho_init: float = -3.0):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.rho = nn.Parameter(torch.full((dim,), rho_init))

    def sigma(self):
        return F.softplus(self.rho)

    def sample(self):
        eps = torch.randn_like(self.mean)
        return self.mean + self.sigma() * eps

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        sigma2 = self.sigma().pow(2)
        prior_var = prior_std**2
        # KL[q || p] for factorized Gaussians
        return (
            0.5
            * (
                ((sigma2 + self.mean.pow(2)) / prior_var)
                - 1.0
                + 2.0 * math.log(prior_std)
                - torch.log(sigma2)
            ).sum()
        )

    def sign_init(self):
        with torch.no_grad():
            self.mean.normal_(0, 0.02)
            self.rho.fill_(-3.0)


class Rank1Linear(nn.Module):
    """
    Full Rank-1 VI linear layer:
      y = (W (x ⊙ s)) ⊙ r + b_c
    s ∈ R^{in}, r ∈ R^{out} are GaussianParameter; bias kept per-component.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 0.1,
        l2_scale: float = 0.0,
        bias: bool = True,
        components: int = 1,
    ):
        super().__init__()
        self.prior_std = prior_std
        self.l2_scale = l2_scale
        self.components = components

        self.layer = nn.Linear(in_features, out_features, bias=False)
        self.s = nn.ModuleList(
            [GaussianParameter(in_features) for _ in range(components)]
        )
        self.r = nn.ModuleList(
            [GaussianParameter(out_features) for _ in range(components)]
        )
        self.bias = (
            nn.Parameter(torch.zeros(components, out_features)) if bias else None
        )
        self._c = 0
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()
        for g in list(self.s) + list(self.r):
            g.sign_init()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        c = self._c
        s = self.s[c].sample() if (self.training and sample) else self.s[c].mean
        r = self.r[c].sample() if (self.training and sample) else self.r[c].mean
        y = self.layer(x * s) * r
        if self.bias is not None:
            y = y + self.bias[c].unsqueeze(0)
        self._c = (c + 1) % self.components
        return y

    def kl(self) -> torch.Tensor:
        # use the last-used component for KL (round-robin)
        c = (self._c - 1) % self.components
        kl = self.s[c].kl_divergence(self.prior_std) + self.r[c].kl_divergence(
            self.prior_std
        )
        if self.l2_scale > 0.0:
            kl = kl + 0.5 * self.l2_scale * self.layer.weight.pow(2).sum()
            if self.bias is not None:
                kl = kl + 0.5 * self.l2_scale * self.bias[c].pow(2).sum()
        return kl


def total_kl(module: nn.Module) -> torch.Tensor:
    """Sum KL() over all submodules that implement it."""
    device = next(module.parameters()).device
    kl = torch.zeros((), device=device)
    for m in module.modules():
        if hasattr(m, "kl") and callable(getattr(m, "kl")):
            kl = kl + m.kl()
    return kl


# ============ Full Rank-1 VI MLP for tabular regression ============


class FullRank1TabularRegressor(nn.Module):
    """
    Architecture (all Linear layers are Rank-1 VI):
      in_dim → Rank1(→ 128) → ReLU → Dropout
             → Rank1(→ 64)  → ReLU → Dropout
             → Rank1(→ 2)   (outputs [mu, log_var])
    """

    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, int] = (128, 64),
        p_drop: float = 0.1,
        use_norm: bool = True,
        prior_std: float = 0.1,
        l2_scale: float = 0.0,
        components: int = 1,
    ):
        super().__init__()
        h1, h2 = hidden

        self.l1 = Rank1Linear(
            in_dim, h1, prior_std=prior_std, l2_scale=l2_scale, components=components
        )
        self.n1 = nn.LayerNorm(h1) if use_norm else nn.Identity()
        self.d1 = nn.Dropout(p_drop)

        self.l2 = Rank1Linear(
            h1, h2, prior_std=prior_std, l2_scale=l2_scale, components=components
        )
        self.n2 = nn.LayerNorm(h2) if use_norm else nn.Identity()
        self.d2 = nn.Dropout(p_drop)

        self.head = Rank1Linear(
            h2, 2, prior_std=prior_std, l2_scale=l2_scale, components=components
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        z = self.l1(x, sample=sample)
        z = self.n1(z)
        z = F.relu(z)
        z = self.d1(z)
        z = self.l2(z, sample=sample)
        z = self.n2(z)
        z = F.relu(z)
        z = self.d2(z)
        out = self.head(z, sample=sample)  # [B, 2] = [mu, log_var]
        return out


def gaussian_nll(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    y_pred: [B, 2] -> [mu, log_var]; y_true: [B]
    """
    mu, log_var = y_pred[:, 0], y_pred[:, 1].clamp(-10.0, 10.0)
    var = log_var.exp().clamp_min(1e-6)
    return (0.5 * (var.log() + (y_true - mu) ** 2 / var)).mean()


# ============ High-level wrapper with fit() & predict_with_uncertainties() ============


@dataclass
class TrainConfig:
    hidden: Tuple[int, int] = (128, 64)
    p_drop: float = 0.1
    use_norm: bool = True
    prior_std: float = 0.1
    l2_scale: float = 0.0
    components: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    kl_warmup_epochs: int = 10
    num_workers: int = 0
    device: Optional[str] = None  # "cuda"/"cpu" (auto if None)


class FullRank1VIRegressor:
    """
    Full Rank-1 VI across all Linear layers.
      - fit(X_df, y)
      - predict_with_uncertainties(X_df, T=50)
    Includes z-score standardization.
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(
            self.cfg.device
            if self.cfg.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[nn.Module] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._fitted = False
        self._feature_cols: Optional[list] = None

    # --- standardization ---
    def _fit_standardizer(self, X: np.ndarray):
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std < 1e-12] = 1.0

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    # --- public API ---
    def fit(self, X_df: pd.DataFrame, y: np.ndarray, verbose: bool = True):
        assert isinstance(X_df, pd.DataFrame), "X_df must be a pandas DataFrame"
        X = X_df.values.astype(np.float32)
        y = y.astype(np.float32).reshape(-1)

        self._feature_cols = list(X_df.columns)
        self._fit_standardizer(X)
        Xn = self._transform_X(X).astype(np.float32)

        in_dim = Xn.shape[1]
        self.model = FullRank1TabularRegressor(
            in_dim=in_dim,
            hidden=self.cfg.hidden,
            p_drop=self.cfg.p_drop,
            use_norm=self.cfg.use_norm,
            prior_std=self.cfg.prior_std,
            l2_scale=self.cfg.l2_scale,
            components=self.cfg.components,
        ).to(self.device)

        ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(y))
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=False,
        )

        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        N = len(ds)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            # KL annealing 0 → 1
            kl_scale = min(1.0, epoch / max(1, self.cfg.kl_warmup_epochs))
            running = 0.0

            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)

                preds = self.model(xb, sample=True)  # stochastic in ALL layers
                nll = gaussian_nll(preds, yb)
                kl = total_kl(self.model)  # KL over all Rank-1 layers
                loss = nll + kl_scale * kl / N
                loss.backward()
                opt.step()

                running += loss.item() * xb.size(0)

            if verbose and (epoch % 10 == 0 or epoch == self.cfg.epochs - 1):
                print(
                    f"[Epoch {epoch:03d}] loss={running / N:.6f}  (kl_scale={kl_scale:.2f})"
                )

        self._fitted = True
        return self

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X_df: pd.DataFrame, T: int = 50
    ) -> pd.DataFrame:
        assert self._fitted and self.model is not None, "Call fit() before predict."
        if self._feature_cols is not None:
            X_df = X_df[self._feature_cols]

        X = X_df.values.astype(np.float32)
        Xn = self._transform_X(X).astype(np.float32)

        self.model.eval()
        X_tensor = torch.from_numpy(Xn).to(self.device)

        mus, ale_vars = [], []
        for _ in range(T):
            out = self.model(X_tensor, sample=True)  # sample across ALL Rank-1 layers
            mu, log_var = out[:, 0], out[:, 1].clamp(-10.0, 10.0)
            mus.append(mu)
            ale_vars.append(log_var.exp().clamp_min(1e-6))

        mus = torch.stack(mus, 0)  # [T, N]
        ale_vars = torch.stack(ale_vars, 0)  # [T, N]

        mean = mus.mean(0)
        epistemic_var = mus.var(0, unbiased=False)
        aleatoric_var = ale_vars.mean(0)
        total_var = epistemic_var + aleatoric_var

        return pd.DataFrame(
            {
                "pred_mean": mean.cpu().numpy(),
                "pred_std_total": total_var.sqrt().cpu().numpy(),
                "pred_std_epistemic": epistemic_var.sqrt().cpu().numpy(),
                "pred_std_aleatoric": aleatoric_var.sqrt().cpu().numpy(),
            },
            index=X_df.index,
        )
