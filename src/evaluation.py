"""
Unified evaluation utilities for (single- and multi-output) regression with uncertainty.
"""

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.stats import t as student_t
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.constants import SUMMARY_PATH


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def gaussian_mixture_mc_nll(
    y: torch.Tensor,  # (N, D)
    mu_stack: torch.Tensor,  # (S, N, D)
    var_stack: torch.Tensor,  # (S, N, D)
) -> torch.Tensor:
    """MC Gaussian mixture negative NLL computed separately for each output dimension.

    Args:
        y: Ground-truth targets, shape (N, D).
        mu_stack: Predicted means from S samples, shape (S, N, D).
        var_stack: Predicted variances from S samples, shape (S, N, D).
        reduce: If True, returns mean NLL per dimension (shape (D,));
            if False, returns per-sample NLLs (shape (N, D)).

    Returns:
        torch.Tensor: NLL per dimension (D,).
    """
    eps = 1e-8
    S = mu_stack.shape[0]
    var_stack = var_stack.clamp_min(eps)

    # (S, N, D): element-wise Gaussian NLL
    nll_elems = F.gaussian_nll_loss(
        mu_stack,
        y.unsqueeze(0),
        var_stack,
        full=True,
        reduction="none",
    )

    # Combine over S: log( (1/S) * Σ_s exp(log p_s) )
    logp_mix = torch.logsumexp(-nll_elems, dim=0) - math.log(S)  # (N, D)
    nll_per_point_dim = -logp_mix  # (N, D)

    return nll_per_point_dim.mean(dim=0)


# def crps_gaussian_mixture_analytic(
#     y: torch.Tensor,  # (N,)
#     mu_stack: torch.Tensor,  # (S, N)
#     var_stack: torch.Tensor,  # (S, N)
# ) -> torch.Tensor:
#     """Analytische CRPS einer univariaten Gaussian-Mixture pro Datenpunkt."""
#     S, N = mu_stack.shape
#     sigma = var_stack.sqrt()
#     y = y.unsqueeze(0)  # (1,N)

#     # --- Term1: E|X - y|
#     z = (y - mu_stack) / sigma
#     phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
#     Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
#     term1 = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
#     term1 = term1.mean(dim=0)  # (N,)

#     # --- Term2: 0.5 * E|X - X'|
#     mu_i = mu_stack.unsqueeze(1)  # (S,1,N)
#     mu_j = mu_stack.unsqueeze(0)  # (1,S,N)
#     s2_i = var_stack.unsqueeze(1)  # (S,1,N)
#     s2_j = var_stack.unsqueeze(0)  # (1,S,N)
#     delta = (mu_i - mu_j).abs()
#     sig_sum = torch.sqrt(s2_i + s2_j)
#     z = delta / sig_sum
#     phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
#     Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
#     exp_abs = sig_sum * (math.sqrt(2 / math.pi) * phi + z * (2 * Phi - 1))
#     term2 = 0.5 * exp_abs.mean(dim=(0, 1))  # (N,)

#     crps = term1 - term2
#     return crps  # (N,)


# def crps_gm_analytic_multid(
#     y: torch.Tensor,  # (N, D)
#     mu_stack: torch.Tensor,  # (S, N, D)
#     var_stack: torch.Tensor,  # (S, N, D)
#     reduce: bool = True,  # True -> (D,), False -> (N, D)
# ):
#     S, N, D = mu_stack.shape
#     out = []
#     for d in range(D):
#         crps_d = crps_gaussian_mixture_analytic(
#             y=y[:, d], mu_stack=mu_stack[:, :, d], var_stack=var_stack[:, :, d]
#         )  # (N,)
#         out.append(crps_d)
#     crps_nd = torch.stack(out, dim=1)  # (N, D)
#     return crps_nd.mean(0) if reduce else crps_nd


def kuleshov_calibration_score_mc(
    y: torch.Tensor,  # (N, D)
    mu_stack: torch.Tensor,  # (S, N, D)
    var_stack: torch.Tensor,  # (S, N, D)
    thresholds: torch.Tensor = None,  # (m,)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calibration Score (cal_score) per output dimension for MC sampled Gaussian mixtures
    as proposed by Kuleshov et al. (2018).

    Computes, for each dimension d, the sum over thresholds of squared
    deviations between nominal coverage p_j and empirical coverage p̂_j,
    where p̂_j = mean( F_i(y_i) <= p_j ) and F_i is the predictive CDF
    of the Gaussian mixture built from (mu_stack, var_stack).

    Args:
        y: Ground-truth targets, shape (N, D).
        mu_stack: Predicted means from S samples, shape (S, N, D).
        var_stack: Predicted variances from S samples, shape (S, N, D).
        thresholds: Confidence thresholds in [0, 1], shape (m,). If None,
            uses torch.linspace(0.05, 1.0, 20) on the same device/dtype as y.
        eps: Small constant to clamp variances for numerical stability.

    Returns:
        torch.Tensor: Calibration score per dimension, shape (D,).
    """
    S, N, D = mu_stack.shape
    var_stack = var_stack.clamp_min(eps)
    std = var_stack.sqrt()

    # Predictive mixture CDF at true targets: F_mix(y) = mean_s Phi((y - mu_s)/sigma_s)
    z = (y.unsqueeze(0) - mu_stack) / std  # (S, N, D)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    F_mix_y = Phi.mean(dim=0)  # (N, D)

    if thresholds is None:
        thresholds = torch.linspace(0.05, 1.0, 20, device=y.device, dtype=y.dtype)

    m = thresholds.numel()
    p_hat = torch.empty((D, m), device=y.device, dtype=y.dtype)
    for d in range(D):
        Fd = F_mix_y[:, d]  # (N,)
        p_hat[d] = (Fd.unsqueeze(0) <= thresholds.unsqueeze(1)).float().mean(dim=1)

    delta = thresholds.unsqueeze(0) - p_hat  # (D, m)
    cal_score = (delta**2).sum(dim=1)  # (D,)
    return cal_score


def kuleshov_calibration_score_der(
    y: torch.Tensor,  # (N, D)
    gamma: torch.Tensor,  # (N, D)
    v: torch.Tensor,  # (N, D)
    alpha: torch.Tensor,  # (N, D)
    beta: torch.Tensor,  # (N, D)
    thresholds: torch.Tensor = None,  # (m,)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calibration Score (CalScore) per output dimension for Deep Evidential Regression
    using the Student-t posterior predictive as defined by Kuleshov et al. (2018).

    The Student-t predictive distribution implied by the Normal-Inverse-Gamma parameters
    (γ, v, α, β) has:
        degrees of freedom:  ν = 2α
        scale:               s² = β (v + 1) / (α v)

    For each dimension:
        CalScore = Σ_j (p_j - p̂_j)², where
        p̂_j = mean_i [ F_i(y_i) ≤ p_j ],
        and F_i is the predictive CDF of Student-t(ν_i, γ_i, s_i).

    Args:
        y: Ground-truth targets, shape (N, D).
        gamma: Predicted mean (location), shape (N, D).
        v: NIG precision parameter v > 0, shape (N, D).
        alpha: NIG α > 0, shape (N, D).
        beta: NIG β > 0, shape (N, D).
        thresholds: Confidence thresholds p_j ∈ [0, 1], shape (m,).
                    If None, uses torch.linspace(0.05, 1.0, 20).
        eps: Small constant for numerical stability.

    Returns:
        torch.Tensor: Calibration score per dimension, shape (D,).
    """
    device, dtype = y.device, y.dtype
    N, D = y.shape

    # Ensure numerical stability
    v = v.clamp_min(eps)
    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)

    # Student-t parameters
    nu = 2.0 * alpha
    scale = torch.sqrt(beta * (v + 1.0) / (alpha * v)).clamp_min(eps)

    # Default thresholds
    if thresholds is None:
        thresholds = torch.linspace(0.05, 1.0, 20, device=device, dtype=dtype)
    m = thresholds.numel()

    cal_score = torch.empty(D, device=device, dtype=dtype)

    for d in range(D):
        # Convert tensors to NumPy (scipy is CPU-only)
        y_np = y[:, d].detach().cpu().numpy()
        gamma_np = gamma[:, d].detach().cpu().numpy()
        scale_np = scale[:, d].detach().cpu().numpy()
        nu_np = nu[:, d].detach().cpu().numpy()

        # Standardized variable and CDF under Student-t
        t_std = (y_np - gamma_np) / scale_np
        Fi_np = student_t.cdf(t_std, df=nu_np)

        # Clip numerical range and convert back to torch
        Fi = torch.from_numpy(np.clip(Fi_np, 0.0, 1.0)).to(device=device, dtype=dtype)

        # Empirical coverage for each threshold p_j
        p_hat_d = (Fi.unsqueeze(0) <= thresholds.unsqueeze(1)).float().mean(dim=1)

        # Squared deviation between nominal and empirical coverage
        delta = thresholds - p_hat_d
        cal_score[d] = (delta**2).sum()

    return cal_score


def ause_score(
    y: np.ndarray,  # (N, D)
    y_pred_mean: np.ndarray,  # (N, D)
    predicted_variances: np.ndarray,  # (N, D)
    n_bins: int = 100,
) -> np.ndarray:
    """Area Under Sparsification Error (AUSE) per output dimension.

    As proposed by Ilg et al. (2018).

    Computes the normalized area between the sparsification curve
    (MAE after removing α-fraction with highest predicted uncertainty)
    and the oracle curve (MAE after removing α-fraction with highest true errors).

    Args:
        y: Ground-truth targets, shape (N, D).
        y_pred_mean: Predicted means, shape (N, D).
        predicted_variances: Predictive variances (uncertainties), shape (N, D).
        n_bins: Number of α grid steps (0 ≤ α ≤ 1).

    Returns:
        np.ndarray: AUSE per dimension, shape (D,). Lower is better.
    """
    y = np.asarray(y, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)
    predicted_variances = np.asarray(predicted_variances, dtype=float)

    N, D = y.shape
    ause = np.zeros(D, dtype=float)

    for d in range(D):
        abs_error = np.abs(y[:, d] - y_pred_mean[:, d])  # (N,)
        uncertainty = predicted_variances[:, d]  # (N,)

        mae_all = abs_error.mean()

        # Sort indices by ascending uncertainty / error
        idx_uncertainty = np.argsort(uncertainty)
        idx_error = np.argsort(abs_error)

        oracle_curve, uncertainty_curve = [], []
        step = max(1, N // n_bins)

        for i in range(0, N + 1, step):
            n_keep = N - i
            if n_keep <= 0:
                mae_uncertainty = 0.0
                mae_oracle = 0.0
            else:
                mae_uncertainty = abs_error[idx_uncertainty[:n_keep]].mean()
                mae_oracle = abs_error[idx_error[:n_keep]].mean()
            uncertainty_curve.append(mae_uncertainty)
            oracle_curve.append(mae_oracle)

        uncertainty_curve = np.asarray(uncertainty_curve, dtype=float)
        oracle_curve = np.asarray(oracle_curve, dtype=float)

        diff = uncertainty_curve - oracle_curve
        ause[d] = max(diff.mean() / mae_all, 0.0)

    return ause


def _bin_uncertainty(
    predictive_variances: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantile binning of predictive variances for ENCE.

    Args:
        predictive_variances: Predicted variances per sample, shape (N,).
        n_bins: Number of quantile bins.

    Returns:
        tuple:
            bin_indices (np.ndarray): Bin index for each sample, shape (N,).
            bin_edges (np.ndarray): Bin edges (unique, ascending), shape (B+1,).
            variances (np.ndarray): Predictive variances, shape (N,).
    """
    variances = np.asarray(predictive_variances, dtype=float)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.unique(np.percentile(variances, quantiles, method="linear"))

    # Handle the rare case of constant variance (all values identical)
    if bin_edges.size < 2:
        bin_edges = np.array([variances.min(), variances.max() + 1e-12], dtype=float)

    bin_indices = np.digitize(variances, bin_edges[1:-1], right=True)
    return bin_indices, bin_edges, variances


def ence(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    predictive_variances: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Normalized Calibration Error (ENCE).

    ENCE quantifies how well predicted variances correspond to observed errors.
    Lower values indicate better uncertainty calibration.

    Args:
        y_true: Ground-truth target values, shape (N,).
        y_pred_mean: Predicted mean values, shape (N,).
        predictive_variances: Predictive variances per sample, shape (N,).
        n_bins: Number of quantile bins.

    Returns:
        float: ENCE scalar (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)
    variances = np.asarray(predictive_variances, dtype=float)

    bin_indices, bin_edges, _ = _bin_uncertainty(variances, n_bins=n_bins)

    rmv_per_bin, rmse_per_bin = [], []

    # Compute RMV (predicted uncertainty) and RMSE (observed error) per bin
    for bin_id in range(bin_edges.size - 1):
        mask = bin_indices == bin_id
        root_mean_variance = float(np.sqrt(np.mean(variances[mask])))
        root_mean_squared_error = float(
            np.sqrt(np.mean((y_true[mask] - y_pred_mean[mask]) ** 2))
        )

        rmv_per_bin.append(root_mean_variance)
        rmse_per_bin.append(root_mean_squared_error)

    rmv_per_bin = np.asarray(rmv_per_bin, dtype=float)
    rmse_per_bin = np.asarray(rmse_per_bin, dtype=float)

    ence_value = np.mean(np.abs(rmv_per_bin - rmse_per_bin) / rmv_per_bin)
    return float(ence_value)


def mae(y_true: np.ndarray, predictions: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, predictions))


def spearman_aleatoric_rank_corr(
    pred_aleatoric: np.ndarray,
    true_aleatoric: np.ndarray,
) -> float:
    if pred_aleatoric.size == 0 or true_aleatoric.size == 0:
        return float("nan")
    if np.all(pred_aleatoric == pred_aleatoric[0]) or np.all(
        true_aleatoric == true_aleatoric[0]
    ):
        return 0.0
    rho, _ = spearmanr(pred_aleatoric, true_aleatoric)
    return float(0.0 if np.isnan(rho) else rho)


def auroc_ood_from_uncertainties(
    X_test: np.ndarray,
    config: Union[dict, str, Path],
    epistemic_uncertainties: np.ndarray,
) -> float:
    if not isinstance(config, dict):
        cfg_path = Path(config)
        with open(cfg_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    lo, hi = config["train_interval"][0], config["train_interval"][1]
    X = np.asarray(X_test)
    u = np.asarray(epistemic_uncertainties)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if u.ndim != 1:
        raise ValueError("epistemic_uncertainties must be 1D (n_samples,)")
    if not np.isfinite(u).all():
        raise ValueError("epistemic_uncertainties contains NaN/Inf values")
    iid_mask = np.all((X >= lo) & (X <= hi), axis=1)
    ood_mask = np.all((X < lo) | (X > hi), axis=1)
    use_mask = iid_mask | ood_mask
    y_true = np.where(ood_mask[use_mask], 1, 0)
    y_score = u[use_mask]
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------


def _load_maybe(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _as_1d(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    return np.asarray(a).squeeze()


# ---------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------


def _coerce_runs(arr: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if arr is None:
        return None
    A = np.asarray(arr)
    if A.ndim == 3:
        return A
    if A.ndim == 2:
        if A.shape[1] == n_samples:
            return A[..., None]
        if A.shape[0] == n_samples:
            return A[None, ...]
        return A[..., None]
    if A.ndim == 1:
        if A.size != n_samples:
            raise ValueError(f"1D input size {A.size} != n_samples={n_samples}.")
        return A[None, :, None]
    raise ValueError(f"Unexpected array shape {A.shape}.")


def _select_dim(A_RND: np.ndarray | None, d: int) -> np.ndarray | None:
    return None if A_RND is None else A_RND[..., d]


# ---------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------


def _normalize_interval(
    iv: tuple[float, float] | str | None,
    X1d: np.ndarray | None,
) -> tuple[float, float] | None:
    if iv is None:
        return None
    if isinstance(iv, str):
        if iv.lower() != "auto":
            raise ValueError(f"Unknown interval keyword '{iv}'.")
        if X1d is None:
            raise ValueError("'auto' interval requires X_test.")
        lo, hi = float(np.nanmin(X1d)), float(np.nanmax(X1d))
        return (lo, hi) if lo <= hi else (hi, lo)
    lo, hi = iv
    return (lo, hi) if lo <= hi else (hi, lo)


def _apply_interval_mask_1d(
    X1d: np.ndarray | None,
    y_true_1d: np.ndarray,
    YP_RN: np.ndarray,
    ALEA_RN: np.ndarray | None,
    EPIS_RN: np.ndarray | None,
    interval: tuple[float, float] | None,
    R: int,
    return_index: bool = False,
):
    if X1d is None or interval is None:
        out = (y_true_1d, YP_RN, ALEA_RN, EPIS_RN, None)
        return out if return_index else out[:4]

    lo, hi = interval
    mask = np.isfinite(X1d) & (X1d >= lo) & (X1d <= hi)
    idx = np.where(mask)[0]

    if idx.size == 0:
        y_nan = np.full(1, np.nan)
        YP_nan = np.full((R, 1), np.nan)
        ALEA_nan = None if ALEA_RN is None else np.full((R, 1), np.nan)
        EPIS_nan = None if EPIS_RN is None else np.full((R, 1), np.nan)
        out = (y_nan, YP_nan, ALEA_nan, EPIS_nan, idx)
        return out if return_index else out[:4]

    out = (
        y_true_1d[idx],
        YP_RN[:, idx],
        None if ALEA_RN is None else ALEA_RN[:, idx],
        None if EPIS_RN is None else EPIS_RN[:, idx],
        idx,
    )
    return out if return_index else out[:4]


# ---------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------


def _make_metric_registry(config_path: Path | None):
    def _auroc_wrapper(y, yp, ale, epi, a_true, X):
        if epi is None or config_path is None or not config_path.exists():
            return np.nan
        return auroc_ood_from_uncertainties(X, config_path, epi)

    return {
        "mae_performance": lambda y, yp, a, e, t, X: mae(y, yp),
        "mae_aleatoric": lambda y, yp, a, e, t, X: np.nan
        if (a is None or t is None)
        else mae(t, a),
        "ence_aleatoric": lambda y, yp, a, e, t, X: np.nan
        if a is None
        else ence(y, yp, a),
        "ence_total": lambda y, yp, a, e, t, X: np.nan
        if (a is None or e is None)
        else ence(y, yp, a + e),
        "spearman_aleatoric": lambda y, yp, a, e, t, X: np.nan
        if (a is None or t is None)
        else spearman_aleatoric_rank_corr(a, t),
        "auroc_ood": _auroc_wrapper,
    }


# ---------------------------------------------------------------------
# Per-run computation for ONE output dimension
# ---------------------------------------------------------------------


def _compute_metrics_for_dim(
    run_dir: Path,
    metrics: list[str],
    metric_intervals: dict[str, tuple[float, float] | str],
    dim_index: int,
) -> dict[str, np.ndarray] | None:
    """
    Returns dict: { metric_name: (R,) } + pass-through arrays: "nll","train_time","infer_time".
    """
    y_true = _load_maybe(run_dir / "y_true.npy")
    y_pred = _load_maybe(run_dir / "y_pred_all.npy")
    X_test = _load_maybe(run_dir / "X_test.npy")
    alea_pred = _load_maybe(run_dir / "aleatoric_all.npy")
    alea_true = _load_maybe(run_dir / "aleatoric_true.npy")
    epis_pred = _load_maybe(run_dir / "epistemic_all.npy")

    # Extras (present by assumption):
    train_times = _as_1d(_load_maybe(run_dir / "train_times.npy"))
    infer_times = _as_1d(_load_maybe(run_dir / "infer_times.npy"))
    nll_vals = _as_1d(_load_maybe(run_dir / "nll_all.npy"))

    config_path = run_dir / "config.json"

    if y_true is None or y_pred is None:
        return None

    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        n_samples = y_true.shape[0]
        y_true_1d = y_true
    elif y_true.ndim == 2:
        n_samples = y_true.shape[0]
        if dim_index >= y_true.shape[1]:
            raise IndexError(
                f"dim_index {dim_index} >= y_true.shape[1]={y_true.shape[1]}"
            )
        y_true_1d = y_true[:, dim_index]
    else:
        raise ValueError(f"Unexpected y_true shape {y_true.shape}")

    YP_RN = _select_dim(_coerce_runs(y_pred, n_samples), dim_index)  # (R,N)
    ALEA_RN = (
        None
        if alea_pred is None
        else _select_dim(_coerce_runs(alea_pred, n_samples), dim_index)
    )
    EPIS_RN = (
        None
        if epis_pred is None
        else _select_dim(_coerce_runs(epis_pred, n_samples), dim_index)
    )

    if YP_RN is None:
        return None

    # 1D feature for masking
    X1d = None
    if X_test is not None:
        X = np.asarray(X_test)
        if X.ndim == 2 and X.shape[1] >= 1:
            X1d = X[:, 0]
        elif X.ndim == 1:
            X1d = X
        elif X.ndim == 2 and X.shape[1] == 0:
            X1d = None
        else:
            X1d = X.reshape(X.shape[0], -1)[:, 0]

    # Intervals
    resolved_intervals: dict[str, tuple[float, float] | None] = {}
    for m in metrics:
        if "time" in m:
            resolved_intervals[m] = None
        else:
            if m not in metric_intervals:
                raise ValueError(f"Missing interval for metric '{m}'")
            resolved_intervals[m] = _normalize_interval(metric_intervals[m], X1d)

    # Pre-mask
    R = YP_RN.shape[0]
    needed = {iv for iv in resolved_intervals.values() if iv is not None}
    cache: dict[tuple[float, float], dict] = {}
    for iv in needed:
        y_t, YP_m, ALEA_m, EPIS_m, idx = _apply_interval_mask_1d(
            X1d, y_true_1d, YP_RN, ALEA_RN, EPIS_RN, iv, R, return_index=True
        )
        a_true_m = None
        if alea_true is not None and idx is not None and idx.size > 0:
            a_full = np.asarray(alea_true)
            if a_full.ndim == 2:
                if dim_index >= a_full.shape[1]:
                    raise IndexError(
                        f"dim_index {dim_index} >= aleatoric_true.shape[1]={a_full.shape[1]}"
                    )
                a_full = a_full[:, dim_index]
            a_true_m = a_full[idx]
        X_iv = None if (X1d is None or idx is None or idx.size == 0) else X1d[idx]
        cache[iv] = dict(
            y=y_t, YP=YP_m, ALEA=ALEA_m, EPI=EPIS_m, ATRUE=a_true_m, X=X_iv
        )

    out = {m: np.full(R, np.nan, dtype=float) for m in metrics}
    REG = _make_metric_registry(config_path if config_path.exists() else None)

    for r in range(R):
        for m in metrics:
            iv = resolved_intervals[m]
            if iv is None:
                continue  # time stats handled elsewhere if requested
            masked = cache[iv]
            y = masked["y"]
            if np.isnan(y).all():
                continue
            yp = masked["YP"][r]
            ale = None if masked["ALEA"] is None else masked["ALEA"][r]
            epi = None if masked["EPI"] is None else masked["EPI"][r]
            a_true = masked["ATRUE"]
            X_iv = masked["X"] if masked["X"] is not None else X1d
            try:
                if m in REG:
                    out[m][r] = REG[m](y, yp, ale, epi, a_true, X_iv)
            except Exception:
                out[m][r] = np.nan

    nll_vec: np.ndarray = np.array([], dtype=float)
    if nll_vals is not None:
        A = np.asarray(nll_vals)
        if A.ndim == 1:
            # (R,)
            nll_vec = A
        elif A.ndim == 2:
            # typischer Fall: (R, D) oder (D, R)
            if A.shape[0] == R:
                # (R, D) -> Dim über Achse 1
                if A.shape[1] == 1:
                    nll_vec = A[:, 0]
                else:
                    nll_vec = A[:, dim_index]
            elif A.shape[1] == R:
                # (D, R) -> Dim über Achse 0
                if A.shape[0] == 1:
                    nll_vec = A[0, :]
                else:
                    nll_vec = A[dim_index, :]
            else:
                raise ValueError(
                    f"nll_all.npy has shape {A.shape}, cannot align with R={R} runs."
                )
        else:
            raise ValueError(f"nll_all.npy unexpected ndim={A.ndim}")
    else:
        nll_vec = np.array([], dtype=float)

    # Zeit-Vektoren bleiben pro Run
    to_1d = (
        lambda x: np.ravel(np.asarray(x, dtype=float))
        if x is not None
        else np.array([], dtype=float)
    )
    out["nll"] = np.ravel(nll_vec.astype(float, copy=False))
    out["train_time"] = to_1d(train_times)
    out["infer_time"] = to_1d(infer_times)
    return out


# ---------------------------------------------------------------------
# Public table builders
# ---------------------------------------------------------------------

METRIC_INTERVALS_DEFAULT: dict[str, tuple[float, float] | str] = {
    "mae_performance": (-4.0, 4.0),
    "rmse_performance": (-4.0, 4.0),
    "mse_performance": (-4.0, 4.0),
    "spearman_aleatoric": (-4.0, 4.0),
    "mae_aleatoric": (-4.0, 4.0),
    "ence_aleatoric": (-4.0, 4.0),
    "ence_total": (-6.0, 6.0),
    "auroc_ood": (-6.0, 6.0),
}


def build_results_df(
    metrics: Iterable[str],
    filter_dict: Optional[dict[str, Union[str, int, float, list]]] = None,
    summary_csv: Union[str, Path, None] = SUMMARY_PATH,
    metric_intervals: Optional[dict[str, tuple[float, float] | str]] = None,
) -> pd.DataFrame:
    if summary_csv is None:
        raise ValueError("summary_csv must be provided.")
    metric_intervals = metric_intervals or METRIC_INTERVALS_DEFAULT

    requested_raw = [m.strip().lower() for m in metrics]
    extras = ["cal_score", "nll", "train_time", "infer_time"]
    known_metrics = set(METRIC_INTERVALS_DEFAULT.keys())

    # Rechenbare Metriken
    requested_compute = [m for m in requested_raw if (m in known_metrics)]
    # Nicht berechenbare (Ausgabewünsche)
    requested_unknown = [m for m in requested_raw if (m not in known_metrics)]

    missing = [
        m for m in requested_compute if ("time" not in m and m not in metric_intervals)
    ]
    if missing:
        raise ValueError(f"Missing intervals for metrics: {', '.join(missing)}")

    summary_csv = Path(summary_csv)
    base_dir = summary_csv.parent
    summary_df = pd.read_csv(summary_csv)

    if filter_dict:
        for k, v in filter_dict.items():
            summary_df = (
                summary_df[summary_df[k].isin(v)]
                if isinstance(v, list)
                else summary_df[summary_df[k] == v]
            )

    per_row_results: list[dict[str, np.ndarray] | None] = []
    max_runs_global = 0

    for _, row in summary_df.iterrows():
        rf = Path(str(row["result_folder"]))
        run_dir = rf if rf.is_absolute() else (base_dir / rf)
        res = _compute_metrics_for_dim(
            run_dir=run_dir,
            metrics=requested_compute,
            metric_intervals=metric_intervals,
            dim_index=0,
        )
        per_row_results.append(res)
        if res is not None:
            for k, v in res.items():
                if k in requested_compute or k in extras:
                    max_runs_global = max(
                        max_runs_global, int(np.ravel(np.asarray(v)).size)
                    )

    out_rows: list[dict] = []
    for row, res in zip(summary_df.to_dict(orient="records"), per_row_results):
        base = dict(row)
        if res is None:
            res = {}

        # 1) Rechenbare Metriken
        for m in requested_compute:
            vals = np.ravel(np.asarray(res.get(m, np.array([])), dtype=float))
            base[f"{m}_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
            for i in range(max_runs_global):
                base[f"{m}_run{i + 1}"] = (
                    float(vals[i])
                    if (i < vals.size and np.isfinite(vals[i]))
                    else np.nan
                )

        # 2) Extras (immer)
        for ex in extras:
            vals = np.ravel(np.asarray(res.get(ex, np.array([])), dtype=float))
            base[f"{ex}_median"] = float(np.nanmedian(vals)) if vals.size else np.nan
            for i in range(max_runs_global):
                base[f"{ex}_run{i + 1}"] = (
                    float(vals[i])
                    if (i < vals.size and np.isfinite(vals[i]))
                    else np.nan
                )

        # 3) Nicht berechenbare Metriken: Spalten anlegen (NaN), ohne echte Werte zu überschreiben
        for unk in requested_unknown:
            # nur anlegen, wenn noch nicht existiert (z.B. 'train_time_median' würde sonst Extras überschreiben)
            if f"{unk}_median" not in base:
                base[f"{unk}_median"] = np.nan
            for i in range(max_runs_global):
                col = f"{unk}_run{i + 1}"
                if col not in base:
                    base[col] = np.nan

        out_rows.append(base)

    return pd.DataFrame(out_rows)


def _get_dim_order_from_config(config_path: Path) -> list[tuple[str, str]]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    fn_field = cfg.get("function")

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for xi in x:
                yield from flatten(xi)
        else:
            yield str(x)

    dims: list[tuple[str, str]] = []
    if isinstance(fn_field, list) and len(fn_field) > 0:
        if all(isinstance(el, (list, tuple)) for el in fn_field):
            for pair in fn_field:
                f = next(
                    (p for p in pair if isinstance(p, str) and p.startswith("f_")), None
                )
                s = next(
                    (p for p in pair if isinstance(p, str) and p.startswith("sigma")),
                    None,
                )
                if f is None or s is None:
                    f = f or next(
                        (p for p in flatten(pair) if p.startswith("f_")), None
                    )
                    s = s or next(
                        (p for p in flatten(pair) if p.startswith("sigma")), None
                    )
                dims.append((f, s))
        else:
            cur_f, cur_s = None, None
            for token in fn_field:
                token = str(token)
                if token.startswith("f_"):
                    cur_f = token
                elif token.startswith("sigma"):
                    cur_s = token
                if cur_f and cur_s:
                    dims.append((cur_f, cur_s))
                    cur_f, cur_s = None, None
    return dims


def build_multi_vs_single_df(
    metrics: Iterable[str],
    multi_experiment_name: str,
    single_experiment_name: str,
    summary_csv: Union[str, Path] = SUMMARY_PATH,
    metric_intervals: Optional[dict[str, tuple[float, float] | str]] = None,
    require_same_model: bool = True,
    unique_key_fields: tuple[str, ...] = ("seed", "function", "noise", "model"),
) -> pd.DataFrame:
    metric_intervals = metric_intervals or METRIC_INTERVALS_DEFAULT

    requested_raw = [m.strip().lower() for m in metrics]
    extras = ["nll", "train_time", "infer_time"]

    known_metrics = set(METRIC_INTERVALS_DEFAULT.keys())

    req = [m for m in requested_raw if (m in known_metrics)]
    requested_unknown = [m for m in requested_raw if (m not in known_metrics)]

    missing = [m for m in req if ("time" not in m and m not in metric_intervals)]
    if missing:
        raise ValueError(f"Missing intervals for metrics: {', '.join(missing)}")

    summary_csv = Path(summary_csv)
    base_dir = summary_csv.parent
    df = pd.read_csv(summary_csv)

    df_multi = df[df["experiment_name"] == multi_experiment_name].copy()
    df_single = df[df["experiment_name"] == single_experiment_name].copy()

    rows_out: list[dict] = []
    seen_keys: set[tuple] = set()

    def _row_key(seed, function, noise, model):
        mapping = {"seed": seed, "function": function, "noise": noise, "model": model}
        return tuple(mapping[f] for f in unique_key_fields)

    for _, mrow in df_multi.iterrows():
        run_dir_multi = Path(str(mrow["result_folder"]))
        if not run_dir_multi.is_absolute():
            run_dir_multi = base_dir / run_dir_multi

        cfg_path = run_dir_multi / "config.json"
        if not cfg_path.exists():
            continue

        dims = _get_dim_order_from_config(cfg_path)
        multi_model = mrow.get("model_name")

        for d, (f_name, s_name) in enumerate(dims):
            key = _row_key(mrow["seed"], str(f_name), str(s_name), str(multi_model))
            if key in seen_keys:
                continue

            filt = (
                (df_single["seed"] == mrow["seed"])
                & (df_single["function"].astype(str) == str(f_name))
                & (df_single["noise"].astype(str) == str(s_name))
                & (
                    df_single["train_interval"].astype(str)
                    == str(mrow["train_interval"])
                )
                & (df_single["test_interval"].astype(str) == str(mrow["test_interval"]))
                & (df_single["test_grid_length"] == mrow["test_grid_length"])
            )
            if require_same_model:
                filt &= df_single["model_name"].astype(str) == str(multi_model)

            candidates = df_single[filt]
            if len(candidates) != 1:
                continue

            srow = candidates.iloc[0]
            run_dir_single = Path(str(srow["result_folder"]))
            if not run_dir_single.is_absolute():
                run_dir_single = base_dir / run_dir_single

            mres = _compute_metrics_for_dim(
                run_dir_multi, req, metric_intervals=metric_intervals, dim_index=d
            )
            sres = _compute_metrics_for_dim(
                run_dir_single, req, metric_intervals=metric_intervals, dim_index=0
            )
            if mres is None or sres is None:
                continue

            base = {
                "seed": mrow["seed"],
                "dim": d,
                "function": str(f_name),
                "noise": str(s_name),
                "multi_result_folder": str(run_dir_multi),
                "single_result_folder": str(run_dir_single),
                "multi_model": str(multi_model),
                "single_model": str(srow["model_name"]),
            }

            def _fill(prefix: str, res_dict: dict[str, np.ndarray]):
                # Bestimme R aus allen verfügbaren Serien in diesem Dict
                R = 0
                for v in res_dict.values():
                    R = max(R, int(np.ravel(np.asarray(v)).size))

                # 1) rechenbare Metriken
                for m in req:
                    vals = np.ravel(
                        np.asarray(res_dict.get(m, np.array([])), dtype=float)
                    )
                    base[f"{prefix}_{m}_median"] = (
                        float(np.nanmedian(vals)) if vals.size else np.nan
                    )
                    for i in range(R):
                        base[f"{prefix}_{m}_run{i + 1}"] = (
                            float(vals[i])
                            if (i < vals.size and np.isfinite(vals[i]))
                            else np.nan
                        )

                # 2) Extras
                for ex in extras:
                    vals = np.ravel(
                        np.asarray(res_dict.get(ex, np.array([])), dtype=float)
                    )
                    base[f"{prefix}_{ex}_median"] = (
                        float(np.nanmedian(vals)) if vals.size else np.nan
                    )
                    for i in range(R):
                        base[f"{prefix}_{ex}_run{i + 1}"] = (
                            float(vals[i])
                            if (i < vals.size and np.isfinite(vals[i]))
                            else np.nan
                        )

                # 3) Nicht berechenbare Metriken (nur Spalten anlegen, echte Werte überschreiben wir nicht)
                for unk in requested_unknown:
                    med_col = f"{prefix}_{unk}_median"
                    if med_col not in base:
                        base[med_col] = np.nan
                    for i in range(R):
                        col = f"{prefix}_{unk}_run{i + 1}"
                        if col not in base:
                            base[col] = np.nan

            _fill("multi", mres)
            _fill("single", sres)

            rows_out.append(base)
            seen_keys.add(key)

    return pd.DataFrame(rows_out)


### ARCHIVE ###
def compare_model_results(
    df: pd.DataFrame,
    function: str | None = None,
    noise: str | None = None,
    model: str = "",
    seed: int | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Creates a comparison table aggregated across (function, noise) combinations.
    Rows = metrics; columns = single vs multi model results.

    Args:
        df: DataFrame produced by build_multi_vs_single_df
        function: optional; if None, use all functions (aggregate across them)
        noise: optional; if None, use all noise settings (aggregate across them)
        model: model name (e.g. "mcdropout")
        seed: optional; if provided, restrict to one seed
        metrics: list of metric column base names (already with _median suffix)
                 default uses medians from the builder.
    """
    import numpy as np
    import pandas as pd

    # defaults (passen zu deiner Median-Auswertung)
    if metrics is None:
        metrics = [
            "mae_performance_median",
            "mae_aleatoric_median",
            "auroc_ood_median",
            "ence_total_median",
            "nll_median",
            "train_time_median",
            "infer_time_median",
        ]

    # Grundfilter: Modell-Gleichheit (multi & single) + optional Seed
    filt = (df["multi_model"].astype(str) == str(model)) & (
        df["single_model"].astype(str) == str(model)
    )
    if seed is not None:
        filt &= df["seed"] == seed

    # Optional: falls function/noise gesetzt sind, zusätzlich filtern.
    if function is not None:
        filt &= df["function"].astype(str) == str(function)
    if noise is not None:
        filt &= df["noise"].astype(str) == str(noise)

    subset = df[filt]
    if subset.empty:
        raise ValueError("No matching rows found for the given filters.")

    # Aggregation: Durchschnitt (mean) über ALLE verbleibenden
    # (function, noise, dim, seed)-Kombinationen hinweg
    sub_mean = subset.mean(numeric_only=True)

    # Vergleichstabelle bauen
    rows = []
    for m in metrics:
        multi_val = sub_mean.get(f"multi_{m}", np.nan)
        single_val = sub_mean.get(f"single_{m}", np.nan)
        rows.append(
            {
                "metric": m,
                "single_model_result": single_val,
                "multi_model_result": multi_val,
            }
        )

    return pd.DataFrame(rows)
