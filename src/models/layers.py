import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class DenseNormal(nn.Module):
    """Dense layer that outputs Normal parameters (mu, log_var)."""

    def __init__(self, in_features: int, target_dim: int) -> None:
        super().__init__()
        self.target_dim: int = int(target_dim)
        self.out_features: int = 2 * self.target_dim
        self.proj: nn.Linear = nn.Linear(in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to [mu, log_var].

        Args:
            x: Input tensor shaped (B, in_features).

        Returns:
            Tensor shaped (B, 2*D) with [mu, log_var] concatenated on the last dim.
        """
        out = self.proj(x)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        return torch.cat([mu, logvar], dim=-1)


class DenseNormalGamma(nn.Module):
    """Evidential Regression head (Amini et al.): outputs mu, v>0, alpha>1, beta>0."""

    def __init__(self, in_features: int, target_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.out_features = 4 * self.target_dim
        self.proj = nn.Linear(in_features, self.out_features)
        self.eps = eps

    @staticmethod
    def evidence(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to [mu, v, alpha, beta].

        Args:
            x: Input tensor shaped (B, in_features).

        Returns:
            Tensor shaped (B, 4*D) with [mu, v, alpha, beta] concatenated on the last dim.
        """
        out = self.proj(x)
        gamma, logv, logalpha, logbeta = torch.chunk(out, 4, dim=-1)
        v = self.evidence(logv) + self.eps  # > 0
        alpha = self.evidence(logalpha) + 1.0 + self.eps  # > 1
        beta = self.evidence(logbeta) + self.eps  # > 0
        return torch.cat([gamma, v, alpha, beta], dim=-1)


class VariationalDense(nn.Module):
    """Bayes-by-Backprop linear layer with diagonal Gaussian posterior.

    Posterior: w ~ N(mu, sigma^2), with sigma = softplus(rho).
    Prior: Gaussian N(mu_0, sigma_0^2), with sigma_0 = softplus(rho_0).
    Stores KL from the last forward pass for retrieval via kl_loss().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mean: float,
        prior_sigma: float,
        mu_init_mean: float,
        mu_init_std: float,
        rho_init: float,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)

        # Posterior parameters (mu, rho)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(mu_init_mean, mu_init_std)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(rho_init)
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features).normal_(mu_init_mean, mu_init_std)
        )
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(rho_init))

        # Prior hyperparameters (buffers, not trainable)
        self.register_buffer("prior_mean", torch.tensor(float(prior_mean)))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma)))

        self._kl: torch.Tensor | None = None  # set on forward()

    def _sample(
        self, mu: torch.Tensor, rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample: returns (w, mu, sigma)."""
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        w = mu + sigma * eps
        return w

    @staticmethod
    def _kl_normal_closed(
        mu: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed form KL(q(w) || p(w)) elementwise
        """
        return (
            torch.log(sigma_p / sigma_q)
            + (sigma_q.square() + (mu - mu_p).square()) / (2.0 * sigma_p.square())
            - 0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear projection with MC-sampled weights.

        Args:
            x: Input tensor shaped (B, in_features).

        Returns:
            Tensor shaped (B, out_features).
        """
        # Weight
        w = self._sample(self.weight_mu, self.weight_rho)
        # Biase
        b = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(x, w, b)

    def kl_loss(self) -> torch.Tensor:
        """Closed form KL q(w)||p(w)"""
        device = self.weight_mu.device
        mu_p = self.prior_mean.to(device)
        sig_p = self.prior_sigma.to(device)
        w_sig = F.softplus(self.weight_rho)
        b_sig = F.softplus(self.bias_rho)

        w_kl = self._kl_normal_closed(self.weight_mu, w_sig, mu_p, sig_p).sum()
        b_kl = self._kl_normal_closed(self.bias_mu, b_sig, mu_p, sig_p).sum()
        return w_kl + b_kl


class VariationalDenseNormal(nn.Module):
    """Variational head producing Normal parameters (mu, log_var)."""

    def __init__(
        self,
        in_features: int,
        target_dim: int,
        prior_mean: float,
        prior_sigma: float,
        mu_init_mean: float,
        mu_init_std: float,
        rho_init: float,
    ) -> None:
        super().__init__()
        self.target_dim: int = int(target_dim)
        out_features = 2 * self.target_dim

        self.var_proj = VariationalDense(
            in_features=in_features,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_sigma=prior_sigma,
            mu_init_mean=mu_init_mean,
            mu_init_std=mu_init_std,
            rho_init=rho_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to [mu, log_var].

        Args:
            x: Input tensor shaped (B, in_features).

        Returns:
            Tensor shaped (B, 2*D) with [mu, log_var].
        """
        out = self.var_proj(x)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        return torch.cat([mu, logvar], dim=-1)
