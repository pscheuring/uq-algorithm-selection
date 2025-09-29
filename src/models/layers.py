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
        out = self.proj(x)
        gamma, logv, logalpha, logbeta = torch.chunk(out, 4, dim=-1)
        v = self.evidence(logv) + self.eps  # > 0
        alpha = self.evidence(logalpha) + 1.0 + self.eps  # > 1
        beta = self.evidence(logbeta) + self.eps  # > 0
        return torch.cat([gamma, v, alpha, beta], dim=-1)


class VariationalDense(nn.Module):
    """Bayes-by-Backprop linear layer with diagonal Gaussian posterior.

    Posterior: w ~ N(mu, sigma^2), with sigma = softplus(rho).
    Prior: Gaussian scale mixture with two zero-mean components.
    Stores KL from the last forward pass for retrieval via kl_loss().
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_pi: float,
        prior_sigma1: float,
        prior_sigma2: float,
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
        self.register_buffer("prior_pi", torch.tensor(float(prior_pi)))
        self.register_buffer("prior_sigma1", torch.tensor(float(prior_sigma1)))
        self.register_buffer("prior_sigma2", torch.tensor(float(prior_sigma2)))

        self._kl: torch.Tensor | None = None  # set on forward()

    def _sample(
        self, mu: torch.Tensor, rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterized sample: returns (w, mu, sigma)."""
        sigma = F.softplus(rho)
        eps = torch.randn_like(mu)
        w = mu + sigma * eps
        return w, mu, sigma

    @staticmethod
    def _log_gaussian(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Elementwise log N(x | mu, sigma)."""
        return dist.Normal(mu, sigma).log_prob(x)

    def _log_mixture_prior(self, w: torch.Tensor) -> torch.Tensor:
        """Elementwise log probability under the scale-mixture prior."""
        device, dtype = w.device, w.dtype
        pi = torch.as_tensor(self.prior_pi, device=device, dtype=dtype)
        s1 = torch.as_tensor(self.prior_sigma1, device=device, dtype=dtype)
        s2 = torch.as_tensor(self.prior_sigma2, device=device, dtype=dtype)
        zero = torch.zeros((), device=device, dtype=dtype)

        a = self._log_gaussian(w, zero, s1) + torch.log(pi)
        b = self._log_gaussian(w, zero, s2) + torch.log1p(-pi)
        return torch.logaddexp(a, b)

    def _kl_term(
        self, w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """KL contribution for one sample of parameters."""
        log_q = self._log_gaussian(w, mu, sigma).sum()
        log_p = self._log_mixture_prior(w).sum()
        return log_q - log_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear projection with MC-sampled weights.

        Args:
            x: Input tensor shaped (B, in_features).

        Returns:
            Tensor shaped (B, out_features).
        """
        # Weights
        w, w_mu, w_sigma = self._sample(self.weight_mu, self.weight_rho)
        kl = self._kl_term(w, w_mu, w_sigma)
        # Bias
        b, b_mu, b_sigma = self._sample(self.bias_mu, self.bias_rho)
        kl += self._kl_term(b, b_mu, b_sigma)

        self._kl = kl
        return F.linear(x, w, b)

    def kl_loss(self) -> torch.Tensor:
        """KL from the last forward pass; zero if forward hasn't run yet."""
        if self._kl is None:
            return torch.tensor(0.0, device=self.weight_mu.device)
        return self._kl


class VariationalDenseNormal(nn.Module):
    """Variational head producing Normal parameters (mu, log_var)."""

    def __init__(
        self,
        in_features: int,
        target_dim: int,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.5,
        prior_sigma2: float = 0.1,
        mu_init_mean: float = 0.0,
        mu_init_std: float = 0.02,
        rho_init: float = -5.0,
    ) -> None:
        super().__init__()
        self.target_dim: int = int(target_dim)
        out_features = 2 * self.target_dim

        self.var_proj = VariationalDense(
            in_features=in_features,
            out_features=out_features,
            prior_pi=prior_pi,
            prior_sigma1=prior_sigma1,
            prior_sigma2=prior_sigma2,
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

    def kl_loss(self) -> torch.Tensor:
        """KL of the internal variational layer."""
        return self.var_proj.kl_loss()
