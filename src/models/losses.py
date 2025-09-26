# losses.py
from typing import Tuple
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def elbo_loss(y, head_out, kl_value, N: int, beta: float):
    """
    NLL_mean + (beta * KL) / N
    beta steuert die KL-Stärke; N stellt korrekte Skalierung sicher.
    """
    mu, logvar = torch.chunk(head_out, 2, dim=-1)
    var = torch.exp(logvar).clamp_min(1e-6)
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
    """
    Negative Log-Likelihood der Student-t-Marginal (Amini).
    """
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
    """
    KL-Divergenz zwischen zwei NIG-Verteilungen.
    """
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
    """
    Evidenz-Regularisierung (Amini) ODER KL-basierte Variante (optional).
    """
    error = (y - gamma).abs()

    if kl:
        omega_v = torch.as_tensor(omega, dtype=y.dtype, device=y.device)
        omega_a = torch.as_tensor(1.0 + omega, dtype=y.dtype, device=y.device)
        kl_term = _kl_nig(gamma, v, alpha, beta, gamma, omega_v, omega_a, beta)
        reg = error * kl_term
    else:
        evi = 2.0 * v + alpha
        reg = error * evi

    return reg.mean() if reduce else reg


def der_loss(
    y_true: torch.Tensor, evidential_output: torch.Tensor, coeff: float
) -> torch.Tensor:
    """
    Gesamtloss = NIG_NLL + coeff * NIG_Reg  (Amini-Variante; KL ist hier NICHT aktiv).
    Erwartet evidential_output = concat([gamma, v, alpha, beta], dim=-1).
    """
    gamma, v, alpha, beta = torch.chunk(evidential_output, 4, dim=-1)
    loss_nll = _nig_nll(y_true, gamma, v, alpha, beta, reduce=True)
    loss_reg = _nig_reg(y_true, gamma, v, alpha, beta, reduce=True, kl=False)
    return loss_nll + coeff * loss_reg
