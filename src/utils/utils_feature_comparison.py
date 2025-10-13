from typing import Dict, Optional, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.constants import ALGORITHM_PERFORMANCE_PATH, META_FEATURES_PATH
from src.models.base_model import BaseModel
from src.models.layers import DenseNormal
from src.utils.utils_pipeline import set_global_seed


class FeatureComparisonModel(BaseModel):
    def __init__(self):
        super().__init__(
            in_features=1,
            hidden_features=[64, 64],
            activations=["relu", "relu"],
            epochs=500,
            lr=1e-3,
            weight_decay=0.0,
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

    def loss(self, y_true: torch.Tensor, head_out: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood with predicted mean/variance.

        Args:
            y_true: Ground-truth targets, shape (B, D).
            head_out: Head outputs, concatenated [mu, log_var], shape (B, 2*D).

        Returns:
            Scalar loss (mean NLL over batch).
        """
        mu, logvar = torch.chunk(head_out, 2, dim=-1)
        var = torch.exp(logvar)
        return F.gaussian_nll_loss(mu, y_true, var, reduction="mean", eps=1e-6)

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
            test_nll = self.loss(y_test, pred).item()

        return test_nll

    def fit():
        pass

    def predict_with_uncertainties():
        pass


def single_feature_performance(test_algorithm_file: str) -> pd.DataFrame:
    """Rank meta-features by training a base model on each single feature.

    Loads meta-features and per-dataset algorithm accuracies, computes the mean
    accuracy per dataset, merges both on the dataset index, and then trains the
    same base network once per individual feature. Each model is evaluated on
    a fixed test split using Gaussian NLL; features are ranked by this NLL
    (lower is better).

    Args:
        test_algorithm_file: File name under ALGORITHM_PERFORMANCE_PATH that
            contains per-dataset accuracies across folds (rows = datasets).

    Returns:
        DataFrame with columns:
            - feature: feature name.
            - nll: test negative log-likelihood for that single-feature model.
        Sorted ascending by nll.
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
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

    results: list[dict[str, float | str]] = []

    for feat in X_all.columns:
        X = df[[feat]].to_numpy(np.float32)
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        set_global_seed(42)
        model = FeatureComparisonModel()
        nll = model.fit_and_eval(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
        nll_val = nll[0] if isinstance(nll, (tuple, list)) else nll
        results.append({"feature": feat, "nll": float(nll_val)})

    df_feature_comparison = (
        pd.DataFrame(results).sort_values("nll", ascending=True).reset_index(drop=True)
    )
    return df_feature_comparison


def define_base_function(
    df_feature_comparison: pd.DataFrame,
    target: str,
    test_algorithm_file: str,
    candidate_funcs: dict[str, Callable],
) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray]:
    """Fit candidate functions on the top-ranked feature and pick the best by MAE.

    The top feature in df_feature_comparison (assumed pre-ranked) is scaled to
    [-2, 2]. Each candidate function is fitted to predict target using
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
            - x: Scaled 1D array of the selected feature in roughly [-2, 2].
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

    best_feat = str(df_feature_comparison.iloc[0]["feature"])

    scaler = MinMaxScaler(feature_range=(-2, 2))
    x = scaler.fit_transform(df_merged[[best_feat]]).ravel()
    y = df_merged[target].to_numpy()

    rows: list[dict[str, object]] = []
    for name, func in candidate_funcs.items():
        try:
            popt, _ = curve_fit(func, x, y, maxfev=10_000)
            y_pred = func(x, *popt)
            mae = mean_absolute_error(y, y_pred)
            rows.append(
                {
                    "Function": name,
                    "Params": np.round(popt, 6).tolist(),
                    "MAE": float(mae),
                }
            )
        except Exception:
            rows.append({"Function": name, "Params": None, "MAE": float("inf")})

    df_func_results = (
        pd.DataFrame(rows).sort_values(by="MAE", ascending=True).reset_index(drop=True)
    )
    return df_func_results, best_feat, x, y


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

    best_name = str(df_func_results.iloc[0]["Function"])
    params = df_func_results.iloc[0]["Params"]

    func = candidate_funcs[best_name]
    y_pred = func(x, *params)
    mae = mean_absolute_error(y, y_pred)

    x_fit = np.linspace(-3, 3, 500)
    y_fit = func(x_fit, *params)

    ax.scatter(x, y, label="Data", alpha=0.35)
    ax.scatter(x, y_pred, label="Prediction (points)", alpha=0.35)
    ax.plot(x_fit, y_fit, label=f"Fitted: {best_name}", linewidth=2)

    ax.set_xlim(-3, 3)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    ax.set_xlabel(f"{best_feat} (scaled to [-2, 2])")
    ax.set_ylabel("Target")
    ax.set_title(title or f"{best_name} fit — MAE = {mae:.4f}")
    return ax
