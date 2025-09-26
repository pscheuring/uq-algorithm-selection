# layers.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


# class DenseNormal(nn.Module):
#     """Gibt (mu, sigma>0) aus – je `units` Dimensionen."""

#     def __init__(self, in_features: int, units: int, eps: float = 1e-6):
#         super().__init__()
#         self.units = int(units)
#         self.eps = float(eps)
#         self.lin = nn.Linear(in_features, 2 * self.units)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.lin(x)
#         mu, logsigma = torch.chunk(out, 2, dim=-1)
#         sigma = F.softplus(logsigma) + self.eps  # sigma > 0
#         return torch.cat([mu, sigma], dim=-1)  # Shape: (B, 2*units)


class DenseNormal(nn.Module):
    """
    Einfacher Head: mappt auf 2 * out_dim, splittet in (mu, log_var).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        mu, log_var = torch.chunk(out, 2, dim=-1)
        return torch.cat([mu, log_var], dim=-1)


class DenseNormalGamma(nn.Module):
    """
    Evidential Regression (Amini): gibt (gamma, v>0, alpha>1, beta>0) aus.
    Erwartet Eingabefeatures mit Dim = in_features; units = Zieldim.
    """

    def __init__(self, in_features: int, units: int):
        super().__init__()
        self.units = int(units)
        self.lin = nn.Linear(in_features, 4 * self.units)

    @staticmethod
    def evidence(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        mu, logv, logalpha, logbeta = torch.chunk(out, 4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1.0
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)


class VariationalLinear(nn.Module):
    """
    Bayes-by-Backprop Linear Layer mit diagonalem Gaussian-Posterior:
        w ~ N(μ, σ^2),  σ = softplus(ρ)
    Prior: Scale-Mixture zweier Gaussians (0-mean):
        p(w_j) = π N(0, σ1^2) + (1-π) N(0, σ2^2)

    Beim Forward wird 1 Monte-Carlo-Sample w gezogen (reparam trick),
    die KL pro Layer für dieses Sample gespeichert und über .kl_loss() abrufbar.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.5,
        prior_sigma2: float = 0.1,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        # Posterior-Parameter (μ, ρ) für Gewichte
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0, 0.02)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(-5.0)
        )  # kleine σ-Startwerte
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.02))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(-5.0))

        # Prior-Hyperparameter (fix, nicht lernbar)
        self.register_buffer("prior_pi", torch.tensor(float(prior_pi)))
        self.register_buffer("prior_sigma1", torch.tensor(float(prior_sigma1)))
        self.register_buffer("prior_sigma2", torch.tensor(float(prior_sigma2)))

        # Zwischenspeicher für KL der letzten Vorwärtsrechnung
        self._kl = None

    def _sample(
        self, mu: torch.Tensor, rho: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = F.softplus(rho)  # σ > 0
        eps = torch.randn_like(mu)
        w = mu + sigma * eps  # reparametrization trick
        return w, mu, sigma

    @staticmethod
    def _log_gaussian(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        # Erwartet passende device/dtype; sigma > 0!
        return dist.Normal(mu, sigma).log_prob(x)

    def _log_mixture_prior(self, w: torch.Tensor) -> torch.Tensor:
        device, dtype = w.device, w.dtype
        pi = torch.as_tensor(self.prior_pi, device=device, dtype=dtype).clamp(
            1e-8, 1 - 1e-8
        )
        s1 = torch.as_tensor(self.prior_sigma1, device=device, dtype=dtype)
        s2 = torch.as_tensor(self.prior_sigma2, device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)

        a = self._log_gaussian(w, zero, s1) + torch.log(pi)
        b = self._log_gaussian(w, zero, s2) + torch.log1p(-pi)  # stabil
        return torch.logaddexp(a, b)  # elementweise

    def _kl_term(
        self, w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        log_q = self._log_gaussian(w, mu, sigma).sum()
        log_p = self._log_mixture_prior(w).sum()
        return log_q - log_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weights
        w, w_mu, w_sigma = self._sample(self.weight_mu, self.weight_rho)
        kl = self._kl_term(w, w_mu, w_sigma)

        # bias
        if self.use_bias:
            b, b_mu, b_sigma = self._sample(self.bias_mu, self.bias_rho)
            kl = kl + self._kl_term(b, b_mu, b_sigma)
        else:
            b = None

        self._kl = kl  # speichere KL dieses Forward-Passes
        return F.linear(x, w, b)

    def kl_loss(self) -> torch.Tensor:
        # Falls forward noch nicht gelaufen ist: 0
        if self._kl is None:
            return torch.tensor(0.0, device=self.weight_mu.device)
        return self._kl


class VariationalDenseNormal(nn.Module):
    """
    Head: VariationalLinear -> 2*D und split in (mu, logvar).
    Hat eine eigene KL über den internen VariationalLinear.
    """

    def __init__(
        self,
        in_features: int,
        out_dim: int,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.5,
        prior_sigma2: float = 0.1,
    ):
        super().__init__()
        # nutzt deinen VariationalLinear
        self.var_proj = VariationalLinear(
            in_features,
            2 * out_dim,
            prior_pi=prior_pi,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.var_proj(x)  # (B, 2*D)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        return torch.cat([mu, logvar], dim=-1)

    # optional, falls du kl_loss() direkt am Head aufrufen möchtest:
    def kl_loss(self) -> torch.Tensor:
        return self.var_proj.kl_loss()
