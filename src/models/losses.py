import math
import torch
import torch.nn.functional as F


def elbo_loss(y, head_out, kl_value, N: int, beta: float) -> torch.Tensor:
    """Evidence Lower Bound (ELBO) loss.

    Computes:
        mean Gaussian NLL + (beta * KL) / N

    Args:
        y: Targets, shape (N, D).
        head_out: Model outputs [mu, log_var], shape (N, 2*D).
        kl_value: KL divergence term from variational layers.
        N: Dataset size, for scaling KL.
        beta: KL weight.

    Returns:
        Scalar loss tensor.
    """
    mu, logvar = torch.chunk(head_out, 2, dim=-1)
    var = torch.exp(logvar).clamp_min(1e-6)
    assert mu.shape == y.shape == var.shape, f"{mu.shape} vs {y.shape} vs {var.shape}"
    nll_mean = F.gaussian_nll_loss(mu, y, var, reduction="mean", eps=1e-6)
    return nll_mean + (beta * kl_value) / float(N)


def _nig_nll(
    y: torch.Tensor,
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Negative log-likelihood of Student-t marginal Amini et al. (2020)."""
    twoBlambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, dtype=y.dtype, device=y.device) / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean() if reduce else nll


def _kl_nig(
    mu1: torch.Tensor,
    v1: torch.Tensor,
    a1: torch.Tensor,
    b1: torch.Tensor,
    mu2: torch.Tensor,
    v2: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two Normal-Inverse-Gamma distributions."""
    KL = (
        0.5 * (a1 - 1.0) / b1 * (v2 * (mu2 - mu1).pow(2))
        + 0.5 * v2 / v1
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1))
        - 0.5
        + a2 * torch.log(b1 / b2)
        - (torch.lgamma(a1) - torch.lgamma(a2))
        + (a1 - a2) * torch.digamma(a1)
        - (b1 - b2) * a1 / b1
    )
    return KL


def _nig_reg(
    y: torch.Tensor,
    gamma: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    omega: float = 0.01,
    reduce: bool = True,
    kl: bool = False,
) -> torch.Tensor:
    """Evidential regularization term Amini et al., 2020.

    Can use simple evidence penalty (default) or KL-based version.
    """
    error = (y - gamma).abs()

    if kl:
        omega = torch.as_tensor(omega, dtype=y.dtype, device=y.device)
        kl_term = _kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl_term
    else:
        evi = 2.0 * v + alpha
        reg = error * evi

    return reg.mean() if reduce else reg


def der_loss(
    y_true: torch.Tensor, evidential_output: torch.Tensor, coeff: float
) -> torch.Tensor:
    """Deep Evidential Regression loss.

    Total loss = NIG NLL + coeff * regularization term.

    Args:
        y_true: Targets, shape (N, D).
        evidential_output: Model outputs [gamma, v, alpha, beta], shape (N, 4*D).
        coeff: Weight for regularization term.

    Returns:
        Scalar loss tensor.
    """
    gamma, v, alpha, beta = torch.chunk(evidential_output, 4, dim=-1)
    assert y_true.shape == gamma.shape, f"y{y_true.shape} vs gamma{gamma.shape}"
    loss_nll = _nig_nll(y_true, gamma, v, alpha, beta)
    loss_reg = _nig_reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * loss_reg
