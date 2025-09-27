import json
import math
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


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
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
) -> float:
    """
    Computes AUROC for OOD detection using epistemicuncertainty estimates.

    Assumes that larger epistemic uncertainty indicates OOD. OOD samples are treated
    as the positive class (label=1), ID samples as the negative class (label=0).

    Args:
        id_uncertainties (np.ndarray of shape (n_id,)):
            Uncertainty scores for in-distribution (ID) samples.
        ood_uncertainties (np.ndarray of shape (n_ood,)):
            Uncertainty scores for out-of-distribution (OOD) samples.

    Returns:
        float: AUROC in [0.0, 1.0], where 1.0 means perfect separation
        (OOD > ID by uncertainty).

    Raises:
        ValueError: If inputs are not 1D arrays, contain NaNs/Infs,
        or are empty.
    """
    # Scores: higher = more likely OOD
    y_score = np.concatenate([id_uncertainties, ood_uncertainties], axis=0)
    y_true = np.concatenate(
        [
            np.zeros_like(id_uncertainties, dtype=int),
            np.ones_like(ood_uncertainties, dtype=int),
        ],
        axis=0,
    )

    return float(roc_auc_score(y_true, y_score))


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
    return float(np.sqrt(np.mean((y_true - predictions) ** 2)))


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
    return float(np.mean((y_true - predictions) ** 2))


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
    return float(np.mean(np.abs(y_true - predictions)))


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
    """verschachtelte Dicts in flaches Dict mit Präfixen auflösen (für model_params usw.)"""
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
    results_path: Union[str, Path], metrics: Iterable[str]
) -> pd.DataFrame:
    """
    Liest einen Ergebnis-Ordner (oder einen Oberordner mit mehreren Ergebnis-Unterordnern)
    und baut ein DataFrame mit Settings + gewünschten Metriken.

    Parameters
    ----------
    results_path : str | Path
        Pfad zu einem Ordner, der eine `config.json` und die .npy-Dateien enthält.
        Wenn der Pfad Unterordner enthält, werden alle Unterordner mit `config.json`
        als separate Zeilen erfasst.
    metrics : Iterable[str]
        Liste der Metrik-Namen, die berechnet werden sollen.
        Verfügbare Namen (alle liefern Skalar-Spalten):
            - "rmse"
            - "mse"
            - "nll_aleatoric"     (NLL mit aleatorischer Varianz)
            - "nll_total"         (NLL mit totaler Varianz = aleatoric + epistemic; falls epistemic fehlt, = aleatoric)
            - "ence_aleatoric"    (ENCE mit aleatorischer Varianz)
            - "ence_total"        (ENCE mit totaler Varianz; s.o.)
            - "spearman_aleatoric" (Rangkorrelation zwischen vorhergesagter und wahrer Aleatorik)
            - "train_time_mean", "train_time_std"
            - "infer_time_mean", "infer_time_std"
        (AUROC-OOD wird hier nicht automatisch berechnet, weil dafür ein OOD-Satz nötig ist.)

    Returns
    -------
    pd.DataFrame
        Eine Zeile pro Resultat-Ordner, Spalten = Settings + gewünschte Metriken.
    """
    results_path = Path(results_path)

    # Kandidatenordner bestimmen (selbst oder Unterordner)
    candidate_dirs: list[Path] = []
    if (results_path / "config.json").exists():
        candidate_dirs = [results_path]
    else:
        for p in sorted(results_path.iterdir()):
            if p.is_dir() and (p / "config.json").exists():
                candidate_dirs.append(p)

    rows = []

    for run_dir in candidate_dirs:
        # --- Dateien laden (optional robust) ---
        cfg_path = run_dir / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        # Arrays
        y_true = _as_1d(_load_opt(run_dir / "y_true.npy"))
        y_pred = _as_1d(_load_opt(run_dir / "y_pred_all.npy"))
        alea_pred = _as_1d(_load_opt(run_dir / "aleatoric_all.npy"))  # Varianz (σ²)
        alea_true = _as_1d(_load_opt(run_dir / "aleatoric_true.npy"))  # Varianz (σ²)
        epis_pred = _as_1d(_load_opt(run_dir / "epistemic_all.npy"))  # Varianz (σ²)

        train_times = _as_1d(_load_opt(run_dir / "train_times.npy"))
        infer_times = _as_1d(_load_opt(run_dir / "infer_times.npy"))

        # sanity checks
        if y_true is None or y_pred is None:
            raise FileNotFoundError(
                f"{run_dir}: y_true.npy und/oder y_pred_all.npy fehlen."
            )

        # totale Varianz falls vorhanden
        if alea_pred is not None and epis_pred is not None:
            total_var = alea_pred + epis_pred
        else:
            total_var = alea_pred  # fallback: nur aleatorisch

        # --- Settings flach machen ---
        base_cfg = {k: v for k, v in cfg.items() if k != "model_params"}
        flat_cfg = {
            **base_cfg,
            **_flatten_dict({"model_params": cfg.get("model_params", {})}),
        }

        row: dict[str, Union[int, float, str, bool]] = {}
        row.update({f"cfg.{k}": v for k, v in flat_cfg.items()})

        # --- Metriken berechnen je nach Wunsch ---
        requested = set(m.strip().lower() for m in metrics)

        def add(name: str, val: float | None):
            if (
                val is not None
                and (isinstance(val, float) or isinstance(val, int))
                and math.isfinite(float(val))
            ):
                row[name] = float(val)
            else:
                row[name] = np.nan  # falls nicht berechenbar

        if "mae" in requested:
            add("mae", mae(y_true, y_pred))

        if "rmse" in requested:
            add("rmse", rmse(y_true, y_pred))

        if "mse" in requested:
            add("mse", mse(y_true, y_pred))

        if "nll_aleatoric" in requested:
            add(
                "nll_aleatoric",
                nll_from_variance(y_true, y_pred, alea_pred)
                if alea_pred is not None
                else np.nan,
            )

        if "nll_total" in requested:
            add(
                "nll_total",
                nll_from_variance(y_true, y_pred, total_var)
                if total_var is not None
                else np.nan,
            )

        if "ence_aleatoric" in requested:
            add(
                "ence_aleatoric",
                ence(y_true, y_pred, alea_pred) if alea_pred is not None else np.nan,
            )

        if "ence_total" in requested:
            add(
                "ence_total",
                ence(y_true, y_pred, total_var) if total_var is not None else np.nan,
            )

        if "spearman_aleatoric" in requested:
            add(
                "spearman_aleatoric",
                spearman_aleatoric_rank_corr(alea_pred, alea_true)
                if (alea_pred is not None and alea_true is not None)
                else np.nan,
            )

        if "train_time_mean" in requested:
            add(
                "train_time_mean",
                float(np.mean(train_times)) if train_times is not None else np.nan,
            )
        if "train_time_std" in requested:
            add(
                "train_time_std",
                float(np.std(train_times)) if train_times is not None else np.nan,
            )

        if "infer_time_mean" in requested:
            add(
                "infer_time_mean",
                float(np.mean(infer_times)) if infer_times is not None else np.nan,
            )
        if "infer_time_std" in requested:
            add(
                "infer_time_std",
                float(np.std(infer_times)) if infer_times is not None else np.nan,
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    # Konsistente Spaltenreihenfolge: erst cfg.*, dann Metriken
    cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
    metric_cols = [c for c in df.columns if not c.startswith("cfg.")]
    return df[cfg_cols + metric_cols]
