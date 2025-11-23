import random
import re
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.constants import ALGORITHM_PERFORMANCE_PATH, META_FEATURES_PATH
from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.utils.utils_pipeline import set_global_seed


class FeatureComparisonModel(BaseModel):
    def __init__(self, in_features: int):
        super().__init__(
            in_features=in_features,
            hidden_features=[64, 64],
            activations=["relu", "relu"],
            epochs=200,
            lr=1e-3,
            weight_decay=1e-3,
            shuffle=True,
            target_dim=1,
            device=None,
            seed=42,
            p_drop=0.0,
            batch_size=32,
        )

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Standard linear hidden layer.

        Args:
            in_features: Number of input features to the layer.
            hidden_features: Number of output features (hidden units).

        Returns:
            A plain nn.Linear layer (no variational components).
        """
        return nn.Linear(in_features=in_features, out_features=hidden_features)

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Head that outputs [mu, log_var] for each target dim.

        Args:
            backbone_out_features: Number of features from the backbone.
            target_dim: Output target dimensionality D.

        Returns:
            A DenseNormal head producing concatenated [mu, log_var] (shape (..., 2*D)).
        """
        return DenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
        )

    def loss(
        self, y_true: torch.Tensor, head_out: torch.Tensor, full_nll: bool = False
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood with predicted mean/variance.

        Args:
            y_true: Ground-truth targets, shape (B, D).
            head_out: Head outputs, concatenated [mu, log_var], shape (B, 2*D).

        Returns:
            Scalar loss (mean NLL over batch).
        """
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(
            mu, y_true, var, reduction="mean", eps=1e-6, full=full_nll
        )

    def fit_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[float, dict]:
        """
        Train NN on a single feature (X shape: (N,1)) and return test NLL.
        Returns (test_nll, extras_dict).
        """

        # to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(
            y_train.reshape(-1, 1), dtype=torch.float32, device=self.device
        )
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(
            y_test.reshape(-1, 1), dtype=torch.float32, device=self.device
        )

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        train_ds = TensorDataset(X_train, y_train)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=generator,
        )

        self.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in train_dl:
                optimizer.zero_grad(set_to_none=True)
                out = self(batch_x)
                loss = self.loss(batch_y, out)
                loss.backward()
                optimizer.step()

        self.eval()
        with torch.no_grad():
            pred = self(X_test)
            test_nll = self.loss(y_test, pred, full_nll=True).item()

        return test_nll

    def fit():
        pass

    def predict_with_uncertainties():
        pass


def feature_combo_performance(
    test_algorithm_file: str,
    n_features: int,
    sample: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate all (or a random subset of) feature combinations of size `n_features`
    and rank them by test NLL (lower is better).

    Loads meta-features and per-dataset algorithm accuracies, computes the mean
    accuracy per dataset, merges both on the dataset index, and trains the same
    base model for each feature combination. Each model is evaluated on a fixed
    test split using Gaussian negative log-likelihood (NLL). Combinations are
    ranked by their test NLL.

    Args:
        test_algorithm_file: Filename under ALGORITHM_PERFORMANCE_PATH containing
            per-dataset accuracies across folds (rows = datasets).
        n_features: Number of features to include in each combination (combination size).
        sample: If set and smaller than the total number of combinations, randomly
            sample this many combinations for evaluation.
        random_state: Random seed used for train/test splitting and random sampling.

    Returns:
        DataFrame with columns:
            - features: Tuple of feature names used in the combination.
            - nll: Test negative log-likelihood for that combination.
        Sorted in ascending order by nll (lower is better).
    """
    algorithm_performance = pd.read_csv(
        ALGORITHM_PERFORMANCE_PATH / test_algorithm_file, index_col=0
    )
    meta_features = pd.read_csv(META_FEATURES_PATH, index_col=0)

    y_df = pd.DataFrame({"mean_accuracy": algorithm_performance.mean(axis=1)})
    df = meta_features.merge(y_df, left_index=True, right_index=True, how="inner")

    X_all = df.drop(columns=["mean_accuracy"])
    y = df["mean_accuracy"].to_numpy(np.float32)

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=random_state
    )

    cols = list(X_all.columns)
    all_combos_iter = list(combinations(cols, n_features))

    if sample is not None and sample < len(all_combos_iter):
        rnd = random.Random(random_state)
        all_combos_iter = rnd.sample(all_combos_iter, sample)

    results = []

    for combo in all_combos_iter:
        X = df[list(combo)].to_numpy(np.float32)
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        set_global_seed(random_state)
        model = FeatureComparisonModel(in_features=n_features)
        nll = model.fit_and_eval(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        nll_val = nll[0] if isinstance(nll, (tuple, list)) else nll
        results.append({"features": tuple(combo), "nll": float(nll_val)})

    df_combo_comparison = (
        pd.DataFrame(results).sort_values("nll", ascending=True).reset_index(drop=True)
    )
    return df_combo_comparison


def define_base_function_single_feature(
    df_feature_comparison: pd.DataFrame,
    target: str,
    test_algorithm_file: str,
    candidate_funcs: dict[str, Callable],
) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray]:
    """Fit candidate functions on the top-ranked feature and pick the best by MAE.

    The top feature in df_feature_comparison (assumed pre-ranked) is scaled to
    [-6, 6]. Each candidate function is fitted to predict target using
    non-linear least squares. The function with the lowest MAE on the training
    points is selected.

    Args:
        df_feature_comparison: DataFrame with at least a "feature" column; the
            first row is treated as the best feature.
        target: Target column to fit ("mean_accuracy"`` or "std_accuracy").
        test_algorithm_file: File name under ALGORITHM_PERFORMANCE_PATH that
            contains per-dataset accuracies across folds (rows = datasets).
        candidate_funcs: mapping name -> f(x, *params) returning predictions.

    Returns:
        Tuple of:
            - results: DataFrame with columns ["Function", "Params", "MAE"],
              sorted by MAE ascending.
            - best_feat: Name of the selected feature.
            - x: Scaled 1D array of the selected feature in roughly [-6, 6].
            - y: Target values aligned to x.
    """
    meta_features = pd.read_csv(META_FEATURES_PATH, index_col=0)
    algorithm_performance = pd.read_csv(
        ALGORITHM_PERFORMANCE_PATH / test_algorithm_file, index_col=0
    )

    df_mean_std = pd.DataFrame(
        {
            "mean_accuracy": algorithm_performance.mean(axis=1),
            "std_accuracy": algorithm_performance.std(axis=1),
        }
    )

    df_merged = meta_features.merge(
        df_mean_std, left_index=True, right_index=True, how="inner"
    )

    best_feat = str(df_feature_comparison.iloc[0]["features"][0])

    scaler = MinMaxScaler(feature_range=(-6, 6))
    x = scaler.fit_transform(df_merged[[best_feat]]).ravel()
    y = df_merged[target].to_numpy()

    rows: list[dict[str, object]] = []
    for name, func in candidate_funcs.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=[0, 0, 0, 0, 0], maxfev=10000)
            y_pred = func(x, *popt)
            mae = mean_absolute_error(y, y_pred)
            rows.append(
                {
                    "non_linear_term": name,
                    "params": np.round(popt, 6).tolist(),
                    "mae": float(mae),
                }
            )
        except Exception:
            rows.append({"non_linear_term": name, "params": None, "mae": float("inf")})

    df_func_results = (
        pd.DataFrame(rows).sort_values(by="mae", ascending=True).reset_index(drop=True)
    )
    return df_func_results, best_feat, x, y


def define_base_function_two_features(
    df_feature_comparison: pd.DataFrame,
    target: str,
    test_algorithm_file: str,
    candidate_funcs: dict[str, Callable],
) -> tuple[pd.DataFrame, tuple[str, str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit candidate two-input functions on the top-ranked *pair* of features and pick the best by MAE.

    The top two features in df_feature_comparison (assumed pre-ranked) are each scaled to [-6, 6].
    For every candidate function f(x1, x2, a, b, c, d, e, activation), two variants are evaluated:
        - x1: x1 is used as the linear term and x2 as the non-linear input (as defined in the candidate).
        - x2: inputs are swapped so x2 is linear and x1 is the non-linear input.
    Each variant is fitted via non-linear least squares (curve_fit) to predict the target.
    The best (lowest MAE) across all functions and both linear-term assignments is selected.

    Args:
        df_feature_comparison: DataFrame indicating feature ranking.
            Accepted forms:
              - Column "features" with tuples; first row must contain at least two feature names.
              - Column "features" with single names; first two rows are taken.
        target: Target column to fit ("mean_accuracy" or "std_accuracy").
        test_algorithm_file: File name under ALGORITHM_PERFORMANCE_PATH that
            contains per-dataset accuracies across folds (rows = datasets).
        candidate_funcs: mapping name -> f(x1, x2, a, b, c, d, e, activation)
            where the function treats x1 as the linear path and x2 via the non-linearity.

    Returns:
        Tuple of:
            - results: DataFrame with columns ["Function", "LinearTerm", "Params", "MAE"],
              sorted ascending by MAE. "LinearTerm" is either "x1" or "x2".
            - best_feats: Tuple (feat_x1, feat_x2) in the order used for the "x1" variant.
            - x1: Scaled 1D array for the first selected feature in [-6, 6].
            - x2: Scaled 1D array for the second selected feature in [-6, 6].
            - y: Target values aligned to x1/x2.
    """
    meta_features = pd.read_csv(META_FEATURES_PATH, index_col=0)
    algorithm_performance = pd.read_csv(
        ALGORITHM_PERFORMANCE_PATH / test_algorithm_file, index_col=0
    )
    df_mean_std = pd.DataFrame(
        {
            "mean_accuracy": algorithm_performance.mean(axis=1),
            "std_accuracy": algorithm_performance.std(axis=1),
        }
    )
    df_merged = meta_features.merge(
        df_mean_std, left_index=True, right_index=True, how="inner"
    )

    top_tuple = df_feature_comparison.iloc[0]["features"]
    feat1, feat2 = str(top_tuple[0]), str(top_tuple[1])

    # Scale both features to[-6, 6]
    scaler = MinMaxScaler(feature_range=(-6, 6))

    x1 = scaler.fit_transform(df_merged[[feat1]]).ravel()
    x2 = scaler.fit_transform(df_merged[[feat2]]).ravel()

    y = df_merged[target].to_numpy()

    # Fit each candidate in two variants: x1-linear (native) and x2-linear (swapped)
    rows = []

    # helper to run curve_fit robustly
    def _fit_and_score(
        func: Callable, X1: np.ndarray, X2: np.ndarray
    ) -> tuple[list[float] | None, float]:
        try:

            def f_vec(X, a, b, c, d, e):
                z1, z2 = X
                return func(z1, z2, a, b, c, d, e)

            popt, _ = curve_fit(f_vec, (X1, X2), y, p0=[0, 0, 0, 0, 0], maxfev=10000)
            y_pred = f_vec((X1, X2), *popt)
            mae = mean_absolute_error(y, y_pred)
            return np.round(popt, 6).tolist(), float(mae)
        except Exception:
            return None, float("inf")

    for name, f in candidate_funcs.items():

        def f_as_is(z1, z2, a, b, c, d, e):
            return f(z1, z2, a, b, c, d, e)

        params, mae = _fit_and_score(f_as_is, x1, x2)
        rows.append(
            {
                "non_linear_term": name,
                "linear_term": "x1",
                "params": params,
                "mae": mae,
            }
        )

        def f_swapped(z1, z2, a, b, c, d, e):
            return f(z2, z1, a, b, c, d, e)  # x2 linear, x1 non-linear

        params, mae = _fit_and_score(f_swapped, x1, x2)
        rows.append(
            {
                "non_linear_term": name,
                "linear_term": "x2",
                "params": params,
                "mae": mae,
            }
        )

    df_func_results = pd.DataFrame(rows).sort_values(by="mae").reset_index(drop=True)
    best_feats = (feat1, feat2)
    return df_func_results, best_feats, x1, x2, y


def plot_best_fit(
    df_func_results: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    candidate_funcs: dict[str, Callable],
    best_feat: str,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot data, point predictions, and the best fitted candidate function.

    Args:
        df_func_results: Output of define_base_function (sorted by MAE ascending);
            must include columns "Function" and "Params" in row 0.
        x: Scaled feature values used for fitting.
        y: Target values aligned to x.
        candidate_funcs: Mapping of candidate names to callables used during fitting.
        best_feat: Feature name used for the x-axis label.
        title: Optional plot title; if None, includes the MAE.

    Returns:
        Matplotlib axes with the visualization.
    """
    fig, ax = plt.subplots()

    best_name = str(df_func_results.iloc[0]["non_linear_term"])
    params = df_func_results.iloc[0]["params"]

    func = candidate_funcs[best_name]
    y_pred = func(x, *params)
    mae = mean_absolute_error(y, y_pred)

    x_fit = np.linspace(-6, 6, 500)
    y_fit = func(x_fit, *params)

    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]
    fit_color = tab10[3]

    ax.scatter(x, y, label="Data", alpha=0.35)
    ax.scatter(x, y_pred, label="Prediction (points)", alpha=0.35, color=data_color)
    ax.plot(x_fit, y_fit, label=f"Fitted: {best_name}", linewidth=2, color=fit_color)

    ax.set_xlim(-6, 6)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    ax.set_xlabel(f"{best_feat}")
    ax.set_ylabel("Target")
    ax.set_title(title or f"{best_name} fit — MAE = {mae:.4f}")
    return ax


def generate_function_defs(results_mean, results_std):
    """Generate clean Python function definitions for each algorithm's f(x) and σ(x).

    This utility automatically extracts the best-fitting candidate functions
    from provided results dictionaries and generates well-formatted Python
    function definitions for both the mean (f) and heteroscedastic noise
    (σ) functions of each algorithm.

    The generated function strings are printed to stdout and can be copied
    to the data_sampler.py module for later use.

    Args:
        results_mean (dict[str, dict]):
            Mapping from algorithm name → dict containing:
                - "df_func_results": DataFrame of candidate fits with MAE values.
                - "x", "y": input/output data arrays.
                - "best_feat": the feature name.
                - "candidate_funcs": dict of candidate functions.
        results_std (dict[str, dict]):
            Same structure as results_mean, but for standard deviation fits.

    Example output::

    # ROCKET
    def f_sinusoidal_rocket_1_feat(X: np.ndarray) -> np.ndarray:
        x1 = X[:, 0]
        return expit(
            - 13.51 - 0.05 * (x1 + 316.76) - 0.14 * np.sin(x1 + 0.90)
        )

    def sigma_sinusoidal_rocket_1_feat(X: np.ndarray) -> np.ndarray:
        x1 = X[:, 0]
        return softplus(
            - 7.39 + 0.17 * (x1 + 21.14) - 0.05 * np.sin(x1 + 0.18)
        )
    """

    def sanitize_name(name: str) -> str:
        """Convert any name into a valid lowercase Python identifier."""
        s = re.sub(r"\W+", "_", str(name).strip().lower())
        if re.match(r"^\d", s):  # cannot start with a digit
            s = "_" + s
        return s

    def format_coeff(val: float) -> str:
        """Return a properly signed coefficient string ('+ 3.1' or '- 3.1')."""
        return f"- {abs(val)}" if val < 0 else f"+ {val}"

    def format_shift(val: float) -> str:
        """Return a formatted shift string for use in '(x1 ± shift)'."""
        return f"- {abs(val)}" if val < 0 else f"+ {val}"

    def nonlinear_expr(term: str, x: str, offset: float) -> str:
        """Return the formatted nonlinear expression term."""
        shift = format_shift(offset)
        if term == "Cubic":
            return f"({x} {shift}) ** 3"
        elif term == "Quadratic":
            return f"({x} {shift}) ** 2"
        elif term == "Exponential":
            return f"np.exp({x} {shift})"
        elif term == "Sinusoidal":
            return f"np.sin({x} {shift})"
        elif term == "Cosine":
            return f"np.cos({x} {shift})"
        else:
            raise ValueError(f"Unknown nonlinear term: {term}")

    #  Main generation loop
    for algo in results_mean.keys():
        df_mean = results_mean[algo]["df_func_results"]
        df_std = results_std[algo]["df_func_results"]

        # Select best fit by minimum MAE
        best_mean = df_mean.loc[df_mean["mae"].idxmin()]
        best_std = df_std.loc[df_std["mae"].idxmin()]

        nonlin_mean = str(best_mean["non_linear_term"])
        nonlin_std = str(best_std["non_linear_term"])

        params_mean = best_mean["params"]  # [a, b, c, d, e]
        params_std = best_std["params"]  # [a, b, c, d, e]

        # Nonlinear term strings
        mean_nonlin_expr = nonlinear_expr(nonlin_mean, "x1", params_mean[4])
        std_nonlin_expr = nonlinear_expr(nonlin_std, "x1", params_std[4])

        # Safe names for function definitions
        algo_name_safe = sanitize_name(algo)
        mean_name_safe = sanitize_name(nonlin_mean)
        std_name_safe = sanitize_name(nonlin_std)

        print(f"# {algo}")

        # ---- f(x) ----
        a, b, c, d, e = params_mean
        print(
            f"def f_{mean_name_safe}_{algo_name_safe}_1_feat(X: np.ndarray) -> np.ndarray:"
        )
        print('    """Compute the true mean function f(x) for 1D input."""')
        print("    x1 = X[:, 0]")
        print("    return expit(")
        print(
            f"        {a} "
            f"{format_coeff(b)} * (x1 {format_shift(c)}) "
            f"{format_coeff(d)} * {mean_nonlin_expr}"
        )
        print("    )\n")

        # ---- σ(x) ----
        a, b, c, d, e = params_std
        print(
            f"def sigma_{std_name_safe}_{algo_name_safe}_1_feat(X: np.ndarray) -> np.ndarray:"
        )
        print('    """Compute the heteroscedastic noise σ(x) for 1D input."""')
        print("    x1 = X[:, 0]")
        print("    return softplus(")
        print(
            f"        {a} "
            f"{format_coeff(b)} * (x1 {format_shift(c)}) "
            f"{format_coeff(d)} * {std_nonlin_expr}"
        )
        print("    )\n")


def plot_best_fit_grid_all(
    fit_configs_mean: Union[List[Dict[str, object]], Dict[str, Dict[str, object]]],
    fit_configs_std: Union[List[Dict[str, object]], Dict[str, Dict[str, object]]],
    y_label_mean: str = "Mean Accuracy",
    y_label_std: str = "Std Accuracy",
    figsize_per_col: float = 4.0,  # width per column (figure width scales with #cols)
    figsize_per_group: float = 6.0,  # height per 2-row group
    x_fit_range: Tuple[float, float] = (-6, 6),
    x_fit_points: int = 500,
    per_row: int = 5,  # always produce 2×5 blocks (or fewer columns in the last block)
    sharey_mean: bool = True,  # share y across ALL mean panels
    sharey_std: bool = False,  # share y across ALL std panels
    suptitle: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot an arbitrary number of algorithms in paged 2×N blocks (default 2×5 per block):
      - Top row of each block: mean fit (observed points + best candidate function)
      - Bottom row of each block: std fit (parametric std function if available; otherwise constant mean(std))

    The two inputs must be aligned (same algorithms/order). You can pass either lists or dicts.

    Required keys per config dict:
        - "df_func_results": DataFrame (row 0 has "non_linear_term", "params", optional "std")
        - "x": np.ndarray of inputs
        - "y": np.ndarray of targets (for mean row)
        - "candidate_funcs": dict[str, Callable], look up best model by "non_linear_term"
        - "best_feat": str for x-axis labeling
        - "title" (optional): column title

    Returns:
        (fig, axs) where axs has shape (2 * n_groups, n_cols) with unused axes hidden in the last block.
    """

    # --- Normalize inputs to ordered lists with titles preserved ---
    def _to_list(
        cfgs: Union[List[Dict[str, object]], Dict[str, Dict[str, object]]],
    ) -> List[Dict[str, object]]:
        if isinstance(cfgs, dict):
            # preserve insertion order (Py3.7+)
            out = []
            for name, cfg in cfgs.items():
                merged = dict(cfg)
                merged.setdefault("title", name)
                out.append(merged)
            return out
        elif isinstance(cfgs, (list, tuple)):
            return list(cfgs)
        else:
            raise TypeError(
                "fit_configs must be a list/tuple of dicts or a dict[str, dict]."
            )

    mean_list = _to_list(fit_configs_mean)
    std_list = _to_list(fit_configs_std)

    if len(mean_list) != len(std_list):
        raise ValueError(
            "fit_configs_mean and fit_configs_std must have the same length."
        )

    n = len(mean_list)
    if n == 0:
        raise ValueError("No columns to plot (empty configs).")

    # --- Layout geometry: 2 rows per group; up to `per_row` columns per group ---
    n_cols = min(per_row, n)
    n_groups = int(np.ceil(n / per_row))
    n_rows = n_groups * 2  # 2 rows per group

    # --- Figure size scales with #cols and #groups ---
    figsize = (
        max(figsize_per_col * n_cols, 6.0),
        max(figsize_per_group * n_groups, 4.0),
    )
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    # --- Colors (Tab10) ---
    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]  # blue
    fit_color = tab10[1]  # orange
    std_color = tab10[3]  # red

    # --- Helpers ---
    def _sanitize(text: str) -> str:
        """Escape $ to avoid mathtext parse errors."""
        if text is None:
            return ""
        return str(text).replace("$", r"\$")

    x_fit = np.linspace(x_fit_range[0], x_fit_range[1], int(x_fit_points))

    mean_ymins, mean_ymaxs = [], []
    std_ymins, std_ymaxs = [], []

    for idx in range(n):
        g = idx // per_row  # group index (0..n_groups-1)
        c = idx % per_row  # column inside the group (0..n_cols-1)
        r_mean = 2 * g  # row index for mean
        r_std = 2 * g + 1  # row index for std

        ax_top = axs[r_mean, c]
        ax_bot = axs[r_std, c]

        # ------------------ MEAN (top row) ------------------
        cfg_m = mean_list[idx]
        df_m = cfg_m["df_func_results"]
        x_m = np.asarray(cfg_m["x"])
        y_m = np.asarray(cfg_m["y"])
        funcs_m: Dict[str, Callable] = cfg_m["candidate_funcs"]
        best_feat_m = _sanitize(str(cfg_m["best_feat"]))
        col_title = _sanitize(str(cfg_m.get("title", f"Model {idx + 1}")))

        if df_m.shape[0] == 0:
            raise ValueError(f"df_func_results (mean) at column {idx} is empty.")
        best_name_m = str(df_m.iloc[0]["non_linear_term"])
        params_m = df_m.iloc[0]["params"]
        f_m = funcs_m[best_name_m]

        y_pred_m = f_m(x_m, *params_m)
        mae = mean_absolute_error(y_m, y_pred_m)
        y_fit_m = f_m(x_fit, *params_m)

        ax_top.scatter(
            x_m, y_m, label="Observed data points", alpha=0.5, color=data_color
        )
        ax_top.plot(
            x_fit,
            y_fit_m,
            label="Best candidate function",
            linewidth=2,
            color=fit_color,
        )
        ax_top.set_xlim(x_fit_range)
        ax_top.grid(True, which="both", alpha=0.4)
        ax_top.set_xlabel(f"{best_feat_m}")
        if c == 0:
            ax_top.set_ylabel(_sanitize(y_label_mean))
        ax_top.set_title(f"{col_title} — MAE = {mae:.4f}")

        yl_top = ax_top.get_ylim()
        mean_ymins.append(yl_top[0])
        mean_ymaxs.append(yl_top[1])

        # ------------------ STD (bottom row) ------------------
        cfg_s = std_list[idx]
        df_s = cfg_s["df_func_results"]
        x_s = np.asarray(cfg_s["x"])
        y_s = np.asarray(cfg_s["y"])
        funcs_s: Dict[str, Callable] = cfg_s["candidate_funcs"]
        best_feat_s = _sanitize(str(cfg_s["best_feat"]))

        if df_s.shape[0] == 0:
            raise ValueError(f"df_func_results (std) at column {idx} is empty.")
        best_name_s = str(df_s.iloc[0]["non_linear_term"])
        params_s = df_s.iloc[0]["params"]

        # Try parametric std function; fallback to constant mean(std)
        y_std_fit = None
        if best_name_s in funcs_s:
            f_s = funcs_s[best_name_s]
            try:
                y_std_fit = f_s(x_fit, *params_s)
            except Exception:
                y_std_fit = None

        if y_std_fit is None:
            if "std" in df_s.columns:
                std_vals = np.asarray(df_s.iloc[0]["std"])
                std_mean = float(np.nanmean(std_vals)) if np.size(std_vals) > 0 else 0.0
            else:
                std_mean = 0.0
            y_std_fit = np.full_like(x_fit, std_mean, dtype=float)

        ax_bot.plot(
            x_fit, y_std_fit, linewidth=2, color=std_color, label="Predicted Std."
        )
        ax_bot.scatter(
            x_s, y_s, label="Observed data points", alpha=0.5, color=data_color
        )
        ax_bot.set_xlim(x_fit_range)
        ax_bot.grid(True, which="both", alpha=0.4)
        ax_bot.set_xlabel(f"{best_feat_s} (scaled to [-6, 6])")
        ax_bot.set_ylim(0, 0.03)
        if c == 0:
            ax_bot.set_ylabel(_sanitize(y_label_std))

        yl_bot = ax_bot.get_ylim()
        std_ymins.append(yl_bot[0])
        std_ymaxs.append(yl_bot[1])

    # Hide any unused columns in the last group (if n % per_row != 0)
    remainder = n % per_row
    if remainder != 0:
        last_cols_to_hide = range(remainder, n_cols)
        last_group_row_mean = 2 * (n_groups - 1)
        last_group_row_std = last_group_row_mean + 1
        for c in last_cols_to_hide:
            axs[last_group_row_mean, c].set_visible(False)
            axs[last_group_row_std, c].set_visible(False)

    # Shared y across ALL mean panels
    if sharey_mean and mean_ymins and mean_ymaxs:
        y_min, y_max = min(mean_ymins), max(mean_ymaxs)
        for g in range(n_groups):
            for c in range(n_cols):
                axs[2 * g, c].set_ylim(y_min, y_max)

    # Shared y across ALL std panels
    if sharey_std and std_ymins and std_ymaxs:
        y_min, y_max = min(std_ymins), max(std_ymaxs)
        for g in range(n_groups):
            for c in range(n_cols):
                axs[2 * g + 1, c].set_ylim(y_min, y_max)

    # Shared legend (merge handles from one mean and one std axis)
    first_mean_ax = axs[0, 0]
    handles, labels = first_mean_ax.get_legend_handles_labels()
    first_std_ax = axs[1, 0]
    h2, l2 = first_std_ax.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h)
            labels.append(l)

    if suptitle:
        fig.suptitle(_sanitize(suptitle), fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def build_fit_configs(
    results_mean: Dict[str, dict],
    results_std: Dict[str, dict],
    candidate_funcs_mean: Dict[str, Callable],
    candidate_funcs_std: Dict[str, Callable],
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Build paired configuration dictionaries for mean and std model fitting results across all algorithms.

    Each entry contains all required elements for plotting or further analysis:
        - df_func_results: DataFrame with fitted model results (first row = best fit)
        - x, y: arrays of input and target data
        - candidate_funcs: dictionary of available candidate functions
        - best_feat: name of the best-performing feature
        - title: algorithm name (for plot labeling)

    Args:
        results_mean (dict): Mapping algorithm_name -> {"df_func_results", "x", "y", "best_feat"} for mean results.
        results_std (dict):  Mapping algorithm_name -> {"df_func_results", "x", "y", "best_feat"} for std results.
        candidate_funcs_mean (dict): Mapping function_name -> callable for mean target functions.
        candidate_funcs_std (dict):  Mapping function_name -> callable for std target functions.

    Returns:
        tuple[dict[str, dict], dict[str, dict]]:
            Two dictionaries keyed by algorithm name:
                - fit_configs_mean[algo_name]: config dict for mean fit
                - fit_configs_std[algo_name]:  config dict for std fit
    """
    algo_names = sorted(results_mean.keys() & results_std.keys())

    fit_configs_mean = {}
    fit_configs_std = {}

    for algo_name in algo_names:
        # --- Mean case ---
        fit_configs_mean[algo_name] = dict(
            df_func_results=results_mean[algo_name]["df_func_results"],
            x=results_mean[algo_name]["x"],
            y=results_mean[algo_name]["y"],
            candidate_funcs=candidate_funcs_mean,
            best_feat=results_mean[algo_name]["best_feat"],
            title=algo_name,
        )

        # --- Std case ---
        fit_configs_std[algo_name] = dict(
            df_func_results=results_std[algo_name]["df_func_results"],
            x=results_std[algo_name]["x"],
            y=results_std[algo_name]["y"],
            candidate_funcs=candidate_funcs_std,
            best_feat=results_std[algo_name]["best_feat"],
            title=algo_name,
        )

    return fit_configs_mean, fit_configs_std
