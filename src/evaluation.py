import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.constants import MULTI_TARGET_EVAL_FUNC, RESULTS_DIR


### Evaluation metrics ###
def gaussian_mixture_nll(
    y: torch.Tensor,  # (N, D)
    mu_stack: torch.Tensor,  # (M, N, D)
    var_stack: torch.Tensor,  # (M, N, D)
) -> torch.Tensor:
    """Gaussian mixture NLL computed separately for each output dimension.

    Args:
        y: Ground-truth targets, shape (N, D).
        mu_stack: Predicted means from M samples, shape (M, N, D).
        var_stack: Predicted variances from M samples, shape (M, N, D).

    Returns:
        torch.Tensor: NLL per dimension (D,).
    """
    eps = 1e-8
    M = mu_stack.shape[0]
    var_stack = var_stack.clamp_min(eps)

    # (M, N, D): element-wise Gaussian NLL
    nll_elems = F.gaussian_nll_loss(
        mu_stack,  # (M, N, D)
        y.unsqueeze(0),  # (1, N, D) -> is broadcasted to (M, N, D) in calculation
        var_stack,  # (M, N, D)
        full=True,
        reduction="none",
    )

    # Combine over M: log( (1/M) * Σ_m exp(log p_m) )
    logp_mix = torch.logsumexp(-nll_elems, dim=0) - math.log(M)  # (N, D)
    nll_per_point_dim = -logp_mix  # (N, D)

    return nll_per_point_dim.mean(dim=0)  # (D,)


def _bin_uncertainty(predictive_variances: np.ndarray, n_bins: int):
    """Quantile binning of predictive (total) uncertainty for ENCE."""
    variances = np.asarray(predictive_variances, dtype=float)
    stds = np.sqrt(np.clip(variances, 0.0, None))

    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(stds, quantiles, method="linear")
    edges = np.unique(edges)

    if edges.size < 2:
        edges = np.array([stds.min(), stds.max() + 1e-12], dtype=float)

    bin_idx = np.digitize(stds, edges[1:-1], right=True)
    return bin_idx, edges, variances


def ence(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    predictive_variances: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Normalized Calibration Error (ENCE) as proposed by Levi et al. (2022)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_mean = np.asarray(y_pred_mean, dtype=float)
    variances = np.asarray(predictive_variances, dtype=float)

    bin_idx, edges, _ = _bin_uncertainty(variances, n_bins=n_bins)

    rmv_list, rmse_list = [], []
    for b in range(edges.size - 1):
        mask = bin_idx == b
        if not np.any(mask):
            continue  # skip empty bins

        rmv = float(np.sqrt(np.mean(np.clip(variances[mask], 0.0, None))))
        rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred_mean[mask]) ** 2)))

        rmv_list.append(rmv)
        rmse_list.append(rmse)

    rmv_arr = np.asarray(rmv_list, dtype=float)
    rmse_arr = np.asarray(rmse_list, dtype=float)
    if rmv_arr.size == 0:
        return float("nan")

    eps = 1e-12
    diff = np.abs(rmv_arr - rmse_arr)
    denom = np.where(rmv_arr > 0, rmv_arr, eps)
    # If RMV=0 and RMSE=0 → 0
    term = np.where((rmv_arr <= 0) & (rmse_arr <= 0), 0.0, diff / denom)
    return float(np.mean(term))


def mae(y_true: np.ndarray, predictions: np.ndarray) -> float:
    """Mean Absolute Error (MAE) between true and predicted values."""
    return float(mean_absolute_error(y_true, predictions))


def spearman_aleatoric_rank_corr(
    pred_aleatoric: np.ndarray,
    true_aleatoric: np.ndarray,
) -> float:
    """Spearman rank correlation between predicted and true aleatoric uncertainties."""
    rho, _ = spearmanr(pred_aleatoric, true_aleatoric)
    return float(0.0 if np.isnan(rho) else rho)


def auroc_ood_from_uncertainties(
    X_test: np.ndarray,
    config: dict,
    epistemic_uncertainties: np.ndarray,
) -> float:
    """
    Computes AUROC for OOD detection using epistemic uncertainties.

    Supported cases (minimal version):
        1. 1D input:
               train_interval = [lo, hi]
           → ID = samples with lo <= x <= hi
           → OOD = samples outside this range

        2. 1D in-between OOD:
               train_interval = [[lo1, hi1], [lo2, hi2]]
           → ID = union of intervals
           → OOD = everything else

        3. 2D input:
               train_interval = [lo, hi]
           → ID  if both coordinates lie within [lo, hi]
           → OOD if both coordinates lie below lo or both above hi
           → Mixed cases (one dim ID, one dim OOD) are ignored

    Args:
        X_test (np.ndarray):
            Test inputs, shape (n,), (n,1), or (n,2).

        config (dict):
            Must contain key:
                "train_interval"
            either [lo, hi] or [[lo1,hi1],[lo2,hi2]].

        epistemic_uncertainties (np.ndarray):
            1D array (n,) with uncertainty scores
            — higher means more likely OOD.

    Returns:
        float:
            AUROC for distinguishing ID vs. OOD based on uncertainty.
    """
    train_int = config["train_interval"]

    # Normalize interval format
    if isinstance(train_int[0], (int, float)):
        intervals = [(float(train_int[0]), float(train_int[1]))]
    else:
        intervals = [tuple(map(float, iv)) for iv in train_int]

    X = np.asarray(X_test)
    u = np.asarray(epistemic_uncertainties).reshape(-1)

    # Normalize to (n_samples, n_dims)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_dims = X.shape
    if u.shape[0] != n_samples:
        raise ValueError(
            f"Inconsistent sample counts: X_test has {n_samples}, "
            f"uncertainties have {u.shape[0]}."
        )

    # -------- 1D CASE --------
    if n_dims == 1:
        x = X[:, 0]
        id_mask = np.zeros(n_samples, dtype=bool)
        for lo, hi in intervals:
            id_mask |= (x >= lo) & (x <= hi)
        y_true = (~id_mask).astype(int)  # 1 = OOD
        return float(roc_auc_score(y_true, u))

    # -------- 2D CASE --------
    if n_dims == 2:
        if len(intervals) != 1:
            raise ValueError("2D input only supports a single [lo, hi] interval.")

        lo, hi = intervals[0]
        x, y = X[:, 0], X[:, 1]

        id_mask = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
        ood_mask = ((x < lo) & (y < lo)) | ((x > hi) & (y > hi))
        use_mask = id_mask | ood_mask

        if not use_mask.any():
            raise ValueError("No usable samples for AUROC in 2D case.")

        y_true = ood_mask[use_mask].astype(int)
        return float(roc_auc_score(y_true, u[use_mask]))


### Utility helpers ###
def _squeeze_last(a: np.ndarray) -> np.ndarray:
    """
    Squeeze the last dimension if it is singleton (shape == 1).

    Used to convert arrays like (N, 1) or (runs, N, 1) into (N,) or (runs, N),
    so that masking and metric computations expecting 1D outputs work correctly.
    """
    if a.ndim >= 2 and a.shape[-1] == 1:
        return np.squeeze(a, axis=-1)
    return a


def _is_multioutput(cfg: dict) -> bool:
    """Return True if this experiment has more than one target dimension."""
    target_dim = cfg["model_params"].get("target_dim", 1)
    return int(target_dim) > 1


def _get_target_index(cfg: dict, target_name: str) -> int | None:
    """
    Determine the index of the desired target in cfg["function"] for multi-output experiments.

    For exp5, cfg["function"] is a list of [f_name, sigma_name] pairs. We look for
    the pair whose first entry matches `target_name`. If not found, return None.
    """
    fn = cfg["function"]
    if not fn:
        return None

    # Multi-output structure: list of [f_name, sigma_name] pairs
    if isinstance(fn[0], (list, tuple)):
        for i, pair in enumerate(fn):
            if pair and pair[0] == target_name:
                return i
        return None

    # For single-output or other structures, we don't do anything here
    return None


def _select_target_from_multioutput(
    cfg: dict,
    y_clean: np.ndarray,
    y_pred_all: np.ndarray,
    alea_true: np.ndarray,
    alea_pred_all: np.ndarray,
    epi_all: np.ndarray,
    target_idx: int | None,
):
    """
    For multi-output experiments (target_dim > 1), select only the desired target
    dimension based on `target_idx`. For single-output, arrays are returned unchanged.
    """
    if not _is_multioutput(cfg) or target_idx is None:
        return y_clean, y_pred_all, alea_true, alea_pred_all, epi_all

    def _slice_last_dim(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim >= 2 and a.shape[-1] > 1:
            return a[..., target_idx]
        return a

    y_clean_sel = _slice_last_dim(y_clean)
    alea_true_sel = _slice_last_dim(alea_true)
    y_pred_all_sel = _slice_last_dim(y_pred_all)
    alea_pred_all_sel = _slice_last_dim(alea_pred_all)
    epi_all_sel = _slice_last_dim(epi_all)

    return y_clean_sel, y_pred_all_sel, alea_true_sel, alea_pred_all_sel, epi_all_sel


def _mask_between(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Return a boolean mask indicating which samples lie inside [lo, hi].
    Works for 1D (n,) / (n,1) and multi-dimensional (n,d) inputs.
    """
    x = np.asarray(x)

    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]

    if x.ndim == 1:
        return (x >= lo) & (x <= hi)

    return np.all((x >= lo) & (x <= hi), axis=1)


def _normalize_interval(interval):
    """
    Normalize an interval specification.

    If given multiple sub-intervals like [[lo1, hi1], [lo2, hi2], ...],
    this function returns the single bounding interval [min, max].
    This intentionally removes any "in-between" gaps.

    Consequence:
        After normalization, the full range [min, max] is treated as the
        ID region. Any original gap between sub-intervals is therefore counted
        as ID when computing ID-only metrics.

        The following metrics in the evaluation pipeline operate only on
        the ID region defined by train_interval:
            - MAE for aleatoric uncertainty (mae_aleatoric_run*)
            - MAE for predictions (mae_prediction_run*)
            - Rank correlation aleatoric uncertainty (aleatoric_rank_corr_run*)

        These metrics will therefore include the gap region as ID, allowing
        you to compare how much the model degrades inside that gap.

    Simple intervals [lo, hi] or empty values are returned unchanged.
    """
    if not interval:
        return interval
    if isinstance(interval[0], (list, tuple)):
        flat = [v for pair in interval for v in pair]
        return [min(flat), max(flat)]
    return interval


def _load_config(path: Path) -> dict:
    """Load and normalize a config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    train_interval = _normalize_interval(raw["train_interval"])
    test_interval = _normalize_interval(raw["test_interval"])

    cfg: dict[str, object] = {
        "experiment_name": raw["experiment_name"],
        "model_runs": int(raw["model_runs"]),
        "seed": int(raw["seed"]),
        "function": list(raw["function"]),
        "train_interval": list(map(float, train_interval)),
        "train_instances": int(raw["train_instances"]),
        "train_repeats": int(raw["train_repeats"]),
        "test_interval": list(map(float, test_interval)),
        "test_grid_length": int(raw["test_grid_length"]),
        "model_name": str(raw["model_name"]),
        "model_params": dict(raw["model_params"]),
    }
    return cfg


def _collect_metrics_for_experiment(results_dir: Path) -> dict[str, float]:
    """Collect evaluation metrics for a single experiment from its results directory."""
    cfg = _load_config(results_dir / "config.json")

    # Target used for evaluation (for multi-output experiments)
    target_name = MULTI_TARGET_EVAL_FUNC
    target_idx = _get_target_index(cfg, target_name) if _is_multioutput(cfg) else None

    # Load NLL and select the matching target if multi-output
    nll_ood_raw = np.load(results_dir / "nll_ood_all.npy").astype(float)
    nll_id_raw = np.load(results_dir / "nll_id_all.npy").astype(float)

    if nll_ood_raw.ndim == 2 and target_idx is not None:
        nll_ood = nll_ood_raw[:, target_idx]
        nll_id = nll_id_raw[:, target_idx]
    else:
        nll_ood = nll_ood_raw.reshape(-1)
        nll_id = nll_id_raw.reshape(-1)

    # Run-level times (always 1D in this setup)
    train_times = np.load(results_dir / "train_times.npy").astype(float)
    infer_times = np.load(results_dir / "infer_times.npy").astype(float)

    X_test = np.load(results_dir / "X_test.npy")
    X_test = _squeeze_last(X_test)

    # Load raw arrays (possibly multi-output)
    y_clean_raw = np.load(results_dir / "y_clean.npy")
    y_pred_all_raw = np.load(results_dir / "y_pred_all.npy")
    alea_true_raw = np.load(results_dir / "aleatoric_true.npy")
    alea_pred_all_raw = np.load(results_dir / "aleatoric_all.npy")
    epi_all_raw = np.load(results_dir / "epistemic_all.npy")

    # Select the appropriate target dimension if multi-output
    (
        y_clean,
        y_pred_all,
        alea_true,
        alea_pred_all,
        epi_all,
    ) = _select_target_from_multioutput(
        cfg,
        y_clean_raw,
        y_pred_all_raw,
        alea_true_raw,
        alea_pred_all_raw,
        epi_all_raw,
        target_idx=target_idx,
    )

    # Squeeze potential trailing singleton dimensions
    y_clean = _squeeze_last(y_clean)
    y_pred_all = _squeeze_last(y_pred_all)
    alea_true = _squeeze_last(alea_true)
    alea_pred_all = _squeeze_last(alea_pred_all)
    epi_all = _squeeze_last(epi_all)

    if alea_pred_all.ndim != 2:
        raise ValueError("aleatoric_pred must be (runs, n_samples) after squeeze")

    n_runs, _ = alea_pred_all.shape

    te_lo, te_hi = float(cfg["test_interval"][0]), float(cfg["test_interval"][1])
    tr_lo, tr_hi = float(cfg["train_interval"][0]), float(cfg["train_interval"][1])

    mask_test_interval = _mask_between(X_test, te_lo, te_hi)
    mask_train_interval = _mask_between(X_test, tr_lo, tr_hi)

    metrics: dict[str, float] = {}

    for i in range(n_runs):
        metrics[f"train_times_run{i + 1}"] = float(train_times[i])
        metrics[f"infer_times_run{i + 1}"] = float(infer_times[i])
        metrics[f"nll_ood_run{i + 1}"] = float(nll_ood[i])
        metrics[f"nll_id_run{i + 1}"] = float(nll_id[i])

    # ENCE on test interval
    y_true_te = y_clean[mask_test_interval]
    for i in range(n_runs):
        y_mean_te = y_pred_all[i, mask_test_interval]
        var_te = alea_pred_all[i, mask_test_interval] + epi_all[i, mask_test_interval]
        metrics[f"ence_run{i + 1}"] = ence(y_true_te, y_mean_te, var_te, n_bins=10)

    # MAE on train interval
    y_true_tr = y_clean[mask_train_interval]
    alea_true_tr = alea_true[mask_train_interval]
    for i in range(n_runs):
        alea_pred_tr = alea_pred_all[i, mask_train_interval]
        y_mean_tr = y_pred_all[i, mask_train_interval]
        metrics[f"mae_aleatoric_run{i + 1}"] = mae(alea_true_tr, alea_pred_tr)
        metrics[f"mae_prediction_run{i + 1}"] = mae(y_true_tr, y_mean_tr)

    # Spearman rank correlation (train interval)
    for i in range(n_runs):
        alea_pred_tr = alea_pred_all[i, mask_train_interval]
        metrics[f"aleatoric_rank_corr_run{i + 1}"] = spearman_aleatoric_rank_corr(
            alea_pred_tr, alea_true_tr
        )

    # AUROC OOD (test interval)
    X_te = X_test[mask_test_interval]
    for i in range(n_runs):
        epi_te = epi_all[i, mask_test_interval]
        metrics[f"auroc_ood_run{i + 1}"] = auroc_ood_from_uncertainties(
            X_te, {"train_interval": cfg["train_interval"]}, epi_te
        )

    for base in (
        "train_times",
        "infer_times",
        "nll_ood",
        "nll_id",
        "ence",
        "mae_aleatoric",
        "mae_prediction",
        "aleatoric_rank_corr",
        "auroc_ood",
    ):
        vals = [metrics[f"{base}_run{j + 1}"] for j in range(n_runs)]
        metrics[f"{base}_median"] = float(np.nanmedian(vals))

    return metrics


def build_eval_dataframe(benchmark_csv_path: str | Path) -> pd.DataFrame:
    """Build a comprehensive evaluation DataFrame from the benchmark summary CSV."""
    benchmark_csv_path = Path(benchmark_csv_path)
    df_raw = pd.read_csv(benchmark_csv_path)
    rows: list[dict[str, object]] = []

    for _, row in df_raw.iterrows():
        results_dir = RESULTS_DIR / row["result_folder"]

        cfg = _load_config(results_dir / "config.json")

        # Defaults from CSV
        function_value = row["function"]
        noise_value = row["noise"]

        # For multi-output, pick the function/noise pair for the target of interest
        if _is_multioutput(cfg):
            idx = _get_target_index(cfg, MULTI_TARGET_EVAL_FUNC)
            if idx is not None:
                pair = cfg["function"][idx]
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    function_value = pair[0]
                    noise_value = pair[1]

        exp_config: dict[str, object] = {
            "experiment_name": row["experiment_name"],
            "model_runs": int(row["model_runs"]),
            "seed": int(row["seed"]),
            "function": function_value,
            "noise": noise_value,
            "train_interval": row["train_interval"],
            "train_instances": int(row["train_instances"]),
            "train_repeats": int(row["train_repeats"]),
            "test_interval": row["test_interval"],
            "test_grid_length": int(row["test_grid_length"]),
            "model_name": row["model_name"],
            "model_params": row["model_params"],
            "results_folder": row["result_folder"],
            "timestamp": row["timestamp"],
        }

        metrics = _collect_metrics_for_experiment(results_dir)
        rows.append({**exp_config, **metrics})

    eval_df = pd.DataFrame(rows)

    exp_config_cols = [
        "experiment_name",
        "model_runs",
        "seed",
        "function",
        "noise",
        "train_interval",
        "train_instances",
        "train_repeats",
        "test_interval",
        "test_grid_length",
        "model_name",
        "model_params",
        "results_folder",
        "timestamp",
    ]

    metric_bases = sorted(
        set(
            c.split("_run")[0].split("_median")[0]
            for c in eval_df.columns
            if c not in exp_config_cols
        )
    )

    ordered_metric_cols: list[str] = []
    for base in metric_bases:
        run_cols = sorted(
            [c for c in eval_df.columns if c.startswith(f"{base}_run")],
            key=lambda x: int(x.split("_run")[-1]),
        )
        median_col = [c for c in eval_df.columns if c == f"{base}_median"]
        ordered_metric_cols.extend(run_cols + median_col)

    eval_df = eval_df[exp_config_cols + ordered_metric_cols]
    return eval_df
