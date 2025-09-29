import math
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from src.constants import SUMMARY_PATH


def nll_from_variance(
    y_true: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
) -> float:
    """
    Computes the mean Negative Log-Likelihood (NLL) for a Gaussian distribution
    with given mean predictions and variances.

    The model assumes:
        y ~ N(mu, sigma^2), where uncertainties = sigma^2.

    Args:
        y_true (np.ndarray of shape (n_samples,)):
            True target values.
        predictions (np.ndarray of shape (n_samples,)):
            Predicted means for each sample (mu).
        uncertainties (np.ndarray of shape (n_samples,)):
            Predicted variances (sigma^2) for each sample. Must be > 0.

    Returns:
        float: Mean negative log-likelihood across all samples.
    """
    # numerical stability: avoid division by zero
    variances = np.maximum(uncertainties, 1e-8)

    nll_i = 0.5 * (
        np.log(2 * np.pi * variances) + ((y_true - predictions) ** 2) / variances
    )

    return float(np.mean(nll_i))


def auroc_ood_from_uncertainties(
    X_test: np.ndarray,
    config: Union[dict, str, Path],
    epistemic_uncertainties: np.ndarray,
) -> float:
    """
    Compute AUROC for OOD detection based on epistemic uncertainty.

    Assumptions:
      - Higher epistemic uncertainty => more likely to be OOD.
      - OOD (positive, y=1): Only samples where ALL features are outside [lo, hi].
      - IID (negative, y=0): Only samples where ALL features are inside [lo, hi].
      - All other samples (mixed cases) are ignored.

    Args:
        X_test: Test samples, shape (n_samples, n_features) or (n_samples,)
        config: Dict or path to config.json with 'train_interval' = [lo, hi]
        epistemic_uncertainties: Uncertainty values, shape (n_samples,)

    Returns:
        float: AUROC score in [0, 1].

    Raises:
        ValueError / FileNotFoundError / KeyError if inputs are invalid.
    """
    train_interval = config["train_interval"]
    lo, hi = train_interval[0], train_interval[1]

    X = np.asarray(X_test)
    u = np.asarray(epistemic_uncertainties)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X_test must be 1D or 2D (n_samples[, n_features])")
    if u.ndim != 1:
        raise ValueError("epistemic_uncertainties must be 1D (n_samples,)")

    n = X.shape[0]
    if len(u) != n:
        raise ValueError(
            f"length mismatch: X_test has {n} samples but uncertainties has {len(u)}"
        )
    if not np.isfinite(u).all():
        raise ValueError("epistemic_uncertainties contains NaN/Inf values")

    # IID: all features in [lo, hi]
    iid_mask = np.all((X >= lo) & (X <= hi), axis=1)

    # OOD: all features outside (each feature < lo or > hi)
    ood_mask = np.all((X < lo) | (X > hi), axis=1)

    # Ignore mixed samples
    use_mask = iid_mask | ood_mask
    if not use_mask.any():
        raise ValueError(
            "No usable samples: all are mixed (neither strictly IID nor strictly OOD)"
        )

    y_score = u[use_mask]
    y_true = np.where(ood_mask[use_mask], 1, 0)  # OOD=1 (positive), IID=0 (negative)
    return roc_auc_score(y_true, y_score)


def rmse(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """
    Computes Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray of shape (n_samples,)):
            Ground-truth target values.
        predictions (np.ndarray of shape (n_samples,)):
            Predicted values.

    Returns:
        float: RMSE (single scalar).
    """
    return np.sqrt(mean_squared_error(y_true, predictions))


def mse(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """
    Computes Mean Squared Error (MSE) between true and predicted values.

    Args:
        y_true (np.ndarray of shape (n_samples,)):
            Ground-truth target values.
        predictions (np.ndarray of shape (n_samples,)):
            Predicted values.

    Returns:
        float: MSE (single scalar).
    """
    return mean_squared_error(y_true, predictions)


def mae(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """
    Computes Mean Absolute Error (MAE) between true and predicted values.

    Args:
        y_true (np.ndarray of shape (n_samples,)):
            Ground-truth target values.
        predictions (np.ndarray of shape (n_samples,)):
            Predicted values.

    Returns:
        float: MAE (single scalar).
    """
    return mean_absolute_error(y_true, predictions)


def _bin_uncertainty(
    uncertainties: np.ndarray,
    n_bins: int,
    min_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal helper: assigns samples to quantile bins based on predicted uncertainties.

    Args:
        uncertainties (np.ndarray of shape (n_samples,)):
            Predicted uncertainties (interpreted as variances, i.e., sigma^2). Must be >= 0.
        n_bins (int):
            Number of bins to create (quantile bins of approximately equal counts).
        min_var (float):
            Lower bound applied to uncertainties for numerical stability.

    Returns:
        Tuple:
            - idx (np.ndarray of shape (n_samples,)):
                Bin index (0..n_bins-1) assigned to each sample.
            - edges (np.ndarray of shape (n_bins+1,)):
                Quantile bin edges (in uncertainty units).
            - unc (np.ndarray of shape (n_samples,)):
                Stabilized uncertainties after applying min_var.
            - std (np.ndarray of shape (n_samples,)):
                Standard deviations (sqrt(unc)).
    """
    unc = np.maximum(uncertainties, min_var)
    std = np.sqrt(unc)

    # Quantile bin edges (equal-count bins); unique to reduce empty bins due to ties
    q = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.unique(np.percentile(unc, q, interpolation="linear"))
    if edges.size < 2:  # all values identical
        edges = np.array(
            [unc.min(), unc.max() + (1e-12 if unc.max() == unc.min() else 0.0)]
        )

    # Assign bins (right edge inclusive)
    idx = np.digitize(unc, edges[1:-1], right=True)
    return idx, edges, unc, std


def rmv(
    uncertainties: np.ndarray,
    n_bins: int = 10,
    min_var: float = 1e-12,
) -> np.ndarray:
    """
    Computes per-bin RMV (root mean variance) values for regression calibration
    using quantile binning.

    RMV(j) = sqrt(mean(uncertainty)) in bin j

    Args:
        uncertainties (np.ndarray of shape (n_samples,)):
            Predicted uncertainties (interpreted as variances, i.e., sigma^2).
        n_bins (int, optional):
            Number of quantile bins. Default 10.
        min_var (float, optional):
            Lower bound for uncertainties for numerical stability. Default 1e-12.

    Returns:
        np.ndarray of shape (n_bins,):
            RMV values per bin.
    """
    idx, edges, unc, std = _bin_uncertainty(
        uncertainties, n_bins=n_bins, min_var=min_var
    )

    rmv_list = []
    for b in range(edges.size - 1):
        mask = idx == b
        if mask.sum() == 0:
            raise ValueError("Empty bin encountered. Reduce n_bins or check for ties.")
        rmv_val = float(np.sqrt(np.mean(unc[mask])))
        rmv_list.append(rmv_val)

    return np.asarray(rmv_list, dtype=float)


def ence(
    y_true: np.ndarray,
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    min_var: float = 1e-12,
) -> float:
    """
    Computes the Expected Normalized Calibration Error (ENCE) using quantile bins.

    ENCE = mean_j( |RMV(j) - RMSE(j)| / RMV(j) ).

    Args:
        y_true (np.ndarray of shape (n_samples,)):
            Ground-truth targets.
        predictions (np.ndarray of shape (n_samples,)):
            Predicted means (mu).
        uncertainties (np.ndarray of shape (n_samples,)):
            Predicted uncertainties (interpreted as variances, i.e., sigma^2).
        n_bins (int, optional):
            Number of quantile bins. Default 10.
        min_var (float, optional):
            Lower bound for uncertainties for numerical stability. Default 1e-12.

    Returns:
        float:
            ENCE value (smaller is better; 0 indicates perfect calibration).
    """
    idx, edges, unc, std = _bin_uncertainty(
        uncertainties, n_bins=n_bins, min_var=min_var
    )

    rmv_vals, rmse_vals = [], []
    for b in range(edges.size - 1):
        mask = idx == b
        if mask.sum() == 0:
            raise ValueError("Empty bin encountered. Reduce n_bins or check for ties.")

        # Per-bin RMSE
        errors_sq = (y_true[mask] - predictions[mask]) ** 2
        rmse_b = float(np.sqrt(np.mean(errors_sq)))

        # Per-bin RMV
        rmv_b = float(np.sqrt(np.mean(unc[mask])))

        rmv_vals.append(rmv_b)
        rmse_vals.append(rmse_b)

    rmv_arr = np.asarray(rmv_vals, dtype=float)
    rmse_arr = np.asarray(rmse_vals, dtype=float)
    rmv_arr = np.maximum(rmv_arr, 1e-12)  # guard against division by tiny RMV

    return float(np.mean(np.abs(rmv_arr - rmse_arr) / rmv_arr))


def spearman_aleatoric_rank_corr(
    pred_aleatoric: np.ndarray,
    true_aleatoric: np.ndarray,
) -> float:
    """
    Computes Spearman's rank correlation (rho) between predicted and true aleatoric uncertainty.

    Args:
        pred_aleatoric (np.ndarray of shape (n_samples,)):
            Predicted aleatoric uncertainties.
        true_aleatoric (np.ndarray of shape (n_samples,)):
            Ground-truth aleatoric uncertainties.

    Returns:
        float:
            Spearman's rho in [-1.0, 1.0]; 1.0 means perfect monotonic agreement.
    """
    rho, _ = spearmanr(pred_aleatoric, true_aleatoric)
    return float(rho)


# ==== Helper functions ====


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Union[int, float, str, bool]]:
    """Flatten nested dicts into a flat dict with prefixes (for model_params etc.)"""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key))
        else:
            out[key] = v
    return out


def _load_opt(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _as_1d(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    a = np.asarray(a)
    return a.squeeze()


# ==== Hauptfunktion ====


def build_results_df(
    metrics: Iterable[str],
    filter_dict: Optional[dict[str, Union[str, int, float, list]]] = None,
    summary_csv: Union[str, Path] = SUMMARY_PATH,
) -> pd.DataFrame:
    """
    Loads `benchmark_summary.csv`, filters it using a dictionary of {column_name: value or [values]},
    and computes the requested metrics for each filtered row based on the corresponding result_folder.

    Args:
        summary_csv (str | Path):
            Path to the `benchmark_summary.csv`.
        metrics (Iterable[str]):
            List of metric names to compute.
        filter_dict (dict | None):
            Dictionary of {column_name: value or [values]} used for filtering.

    Returns:
        pd.DataFrame:
            DataFrame containing the filtered summary columns along with the requested metrics.
    """

    summary_csv = Path(summary_csv)
    base_dir = summary_csv.parent

    summary_df = pd.read_csv(summary_csv)

    # --- einfaches Filtern ---
    if filter_dict:
        for key, value in filter_dict.items():
            if isinstance(value, list):
                summary_df = summary_df[summary_df[key].isin(value)]
            else:
                summary_df = summary_df[summary_df[key] == value]

    requested = set(m.strip().lower() for m in metrics)

    def compute_for_run(run_dir: Path) -> dict[str, float]:
        y_true = _as_1d(_load_opt(run_dir / "y_true.npy"))
        y_pred = _as_1d(_load_opt(run_dir / "y_pred_all.npy"))
        X_test = _as_1d(_load_opt(run_dir / "X_test.npy"))
        alea_pred = _as_1d(_load_opt(run_dir / "aleatoric_all.npy"))
        alea_true = _as_1d(_load_opt(run_dir / "aleatoric_true.npy"))
        epis_pred = _as_1d(_load_opt(run_dir / "epistemic_all.npy"))
        train_times = _as_1d(_load_opt(run_dir / "train_times.npy"))
        infer_times = _as_1d(_load_opt(run_dir / "infer_times.npy"))
        cfg_path = run_dir / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        out: dict[str, float] = {}

        def add(name: str, val):
            if (
                val is not None
                and isinstance(val, (float, int))
                and math.isfinite(float(val))
            ):
                out[name] = float(val)
            else:
                out[name] = np.nan

        total_var = alea_pred + epis_pred

        if "mae_performance" in requested:
            add("mae_performance", mae(y_true, y_pred))
        if "rmse_performance" in requested:
            add("rmse_perforamnce", rmse(y_true, y_pred))
        if "mse_performance" in requested:
            add("mse_performance", mse(y_true, y_pred))
        if "mae_aleatoric" in requested:
            add("mae_aleatoric", mae(alea_true, alea_pred))
        if "nll_aleatoric" in requested:
            add("nll_aleatoric", nll_from_variance(y_true, y_pred, alea_pred))
        if "nll_total" in requested:
            add("nll_total", nll_from_variance(y_true, y_pred, total_var))
        if "ence_aleatoric" in requested:
            add("ence_aleatoric", ence(y_true, y_pred, alea_pred))
        if "ence_total" in requested:
            add("ence_total", ence(y_true, y_pred, total_var))
        if "spearman_aleatoric" in requested:
            add(
                "spearman_aleatoric", spearman_aleatoric_rank_corr(alea_pred, alea_true)
            )
        if "auroc_ood" in requested:
            add("auroc_ood", auroc_ood_from_uncertainties(X_test, config, epis_pred))
        if "train_time_mean" in requested:
            add("train_time_mean", float(np.mean(train_times)))
        if "train_time_std" in requested:
            add("train_time_std", float(np.std(train_times)))
        if "infer_time_mean" in requested:
            add("infer_time_mean", float(np.mean(infer_times)))
        if "infer_time_std" in requested:
            add("infer_time_std", float(np.std(infer_times)))

        return out

    metric_rows = []
    for _, row in summary_df.iterrows():
        rf = Path(str(row["result_folder"]))
        run_dir = rf if rf.is_absolute() else (base_dir / rf)
        metric_rows.append(compute_for_run(run_dir))

    metrics_df = pd.DataFrame(metric_rows, index=summary_df.index)

    out_df = pd.concat(
        [summary_df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1
    )

    metric_cols = [c for c in out_df.columns if c in requested]
    other_cols = [c for c in out_df.columns if c not in requested]
    return out_df[other_cols + metric_cols]
