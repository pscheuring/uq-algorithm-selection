from pathlib import Path
from typing import List, Optional, Union

import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import clear_output, display
from ipywidgets import interact
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler

from src.constants import META_MODEL_RESULTS_DIR

# style_path = BASE_DIR / "src" / "styling.mplstyle"
# plt.style.use(str(style_path.resolve()))


def plot_corr_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    annot: bool = False,
):
    """Plots a heatmap of the correlation matrix for numeric columns in a DataFrame."""
    if columns:
        data = df[columns].select_dtypes(include="number")
    else:
        data = df.select_dtypes(include="number")

    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=90, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap_plotly(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    annot: bool = False,
):
    """Interactive plotly heatmap of the correlation matrix for numeric columns in a DataFrame."""
    if columns:
        data = df[columns].select_dtypes(include="number")
    else:
        data = df.select_dtypes(include="number")

    corr_matrix = data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f" if annot else False,
        color_continuous_scale=px.colors.sequential.RdBu[::-1],
        aspect="square",
        labels=dict(color="Correlation"),
        title="Correlation Heatmap",
    )
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        width=800,
        height=800,
    )
    fig.show()


def interactive_scatter_plot(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    bool_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

    @interact(
        x_col=widgets.Dropdown(options=numeric_cols, description="X-Axis"),
        y_col=widgets.Dropdown(options=numeric_cols, description="Y-Axis"),
        hue_cols=widgets.SelectMultiple(
            options=bool_cols,
            description="Hue(s)",
            layout=widgets.Layout(height="100px"),
        ),
    )
    def plot(x_col, y_col, hue_cols):
        clear_output(wait=True)
        if x_col == y_col:
            print("Please select two different columns.")
            return

        plot_df = df.copy()

        if hue_cols:
            hue_col = "_combined_hue_"
            plot_df[hue_col] = (
                plot_df[list(hue_cols)]
                .astype(str)
                .agg(
                    lambda row: ",".join(
                        f"{col}={val}" for col, val in zip(hue_cols, row)
                    ),
                    axis=1,
                )
            )
        else:
            hue_col = None

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df, x=x_col, y=y_col, hue=hue_col if hue_col else None, ax=ax
        )

        ax.set_title(
            f"{y_col} vs {x_col}" + (f" by {', '.join(hue_cols)}" if hue_cols else "")
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        if hue_cols:
            ax.legend(
                title="Hue",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        else:
            ax.legend().remove()

        plt.tight_layout()
        plt.show()


def interactive_scatter_plot_plotly(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    bool_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all()]

    @interact(
        x_col=widgets.Dropdown(options=numeric_cols, description="X-Axis"),
        y_col=widgets.Dropdown(options=numeric_cols, description="Y-Axis"),
        hue_cols=widgets.SelectMultiple(
            options=bool_cols,
            description="Hue(s)",
            layout=widgets.Layout(height="100px"),
        ),
    )
    def plot(x_col, y_col, hue_cols):
        clear_output(wait=True)
        if x_col == y_col:
            print("Please select two different columns.")
            return

        plot_df = df.copy()

        if hue_cols:
            hue_col = "_combined_hue_"
            plot_df[hue_col] = (
                plot_df[list(hue_cols)]
                .astype(str)
                .agg(
                    lambda row: ",".join(
                        f"{col}={val}" for col, val in zip(hue_cols, row)
                    ),
                    axis=1,
                )
            )
        else:
            hue_col = None

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=hue_col if hue_col else None,
            title=f"{y_col} vs {x_col}"
            + (f" by {', '.join(hue_cols)}" if hue_cols else ""),
            height=600,
            width=800,
        )

        fig.update_layout(legend_title="Hue", margin=dict(t=50, r=50, b=50, l=50))
        fig.show()


def interactive_boxplot(
    df: pd.DataFrame,
    eval_col: str = "mse",
    feature_cols: List[str] = [
        "shifting",
        "seasonal",
        "stationary",
        "trend",
        "transition",
    ],
):
    """Interactive boxplot for selected features."""
    cols = [
        col
        for col in feature_cols
        if col in df.columns and df[col].dropna().isin([0, 1]).all()
    ]

    @interact(
        show_outliers=widgets.Dropdown(
            options=[("with outliers", True), ("without outliers", False)],
            value=True,
            description="Outlier:",
        )
    )
    def plot(show_outliers):
        clear_output(wait=True)
        rows = []
        for feature in cols:
            for val in [0, 1]:
                subset = df[df[feature] == val]
                for score in subset[eval_col]:
                    rows.append(
                        {
                            "Feature": feature,
                            "Value": score,
                            "Condition": bool(val),
                        }
                    )

        df_plot = pd.DataFrame(rows)

        if df_plot.empty:
            print("No data available.")
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_plot,
            x="Feature",
            y="Value",
            hue="Condition",
            showfliers=show_outliers,
        )
        handles = [
            mpatches.Patch(color="#006BA4", label="False"),
            mpatches.Patch(color="#FF800E", label="True"),
        ]
        plt.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.title(f"{eval_col} distribution grouped by binary features (False vs True)")
        plt.tight_layout()
        plt.show()


def interactive_boxplot_plotly(
    df: pd.DataFrame,
    eval_col: str = "mse",
    feature_cols: List[str] = [
        "shifting",
        "seasonal",
        "stationary",
        "trend",
        "transition",
    ],
):
    import ipywidgets as widgets
    import pandas as pd
    import plotly.express as px
    from IPython.display import clear_output
    from ipywidgets import interact

    cols = [
        col
        for col in feature_cols
        if col in df.columns and df[col].dropna().isin([0, 1]).all()
    ]

    @interact(
        show_outliers=widgets.Dropdown(
            options=[("with outliers", True), ("without outliers", False)],
            value=True,
            description="Outlier:",
        )
    )
    def plot(show_outliers):
        clear_output(wait=True)
        rows = []

        for feature in cols:
            for val in [0, 1]:
                subset = df[df[feature] == val][eval_col]
                if not show_outliers:
                    Q1 = subset.quantile(0.25)
                    Q3 = subset.quantile(0.75)
                    IQR = Q3 - Q1
                    subset = subset[
                        (subset >= Q1 - 1.5 * IQR) & (subset <= Q3 + 1.5 * IQR)
                    ]

                for score in subset:
                    rows.append(
                        {
                            "Feature": feature,
                            "Value": score,
                            "Condition": bool(val),
                        }
                    )

        df_plot = pd.DataFrame(rows)

        if df_plot.empty:
            print("No data available.")
            return

        fig = px.box(
            df_plot,
            x="Feature",
            y="Value",
            color="Condition",
            points="outliers" if show_outliers else False,
            color_discrete_map={False: "#006BA4", True: "#FF800E"},
            title=f"{eval_col} distribution grouped by binary features (False vs True)",
        )

        fig.update_layout(
            boxmode="group",
            legend=dict(
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                bordercolor="gray",
                borderwidth=1,
            ),
            margin=dict(t=50, r=100, b=50, l=50),
            height=600,
            width=800,
        )
        fig.show()


def interactive_distribution_plot(df: pd.DataFrame, bins: int = 50):
    """Interactive distribution plot for numeric columns in a DataFrame."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    @interact(
        variable=widgets.Dropdown(options=numeric_cols, description="Variable:"),
        show_kde=widgets.Checkbox(value=True, description="Show KDE"),
    )
    def plot(variable, show_kde):
        data = df[variable].dropna()
        plt.figure(figsize=(8, 6))
        sns.histplot(data, kde=show_kde, bins=bins, color="steelblue")
        plt.title(f"Distribution of '{variable}'")
        plt.xlabel(variable)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


def interactive_distribution_plot_plotly(df: pd.DataFrame, bins: int = 50):
    """Interactive distribution plot (histogram + optional KDE) using Plotly."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    @interact(
        variable=widgets.Dropdown(options=numeric_cols, description="Variable:"),
        show_kde=widgets.Checkbox(value=True, description="Show KDE"),
    )
    def plot(variable, show_kde):
        data = df[variable].dropna()

        hist = go.Histogram(
            x=data,
            nbinsx=bins,
            name="Histogram",
            marker_color="steelblue",
            opacity=0.75,
        )

        fig = go.Figure(data=[hist])

        if show_kde:
            kde_x = np.linspace(data.min(), data.max(), 200)
            kde = sns.kdeplot(data, bw_adjust=1).get_lines()[0].get_data()
            plt.close()
            kde_line = go.Scatter(
                x=kde[0],
                y=kde[1] * len(data) * (kde_x[1] - kde_x[0]),
                name="KDE",
                mode="lines",
                line=dict(color="firebrick"),
            )
            fig.add_trace(kde_line)

        fig.update_layout(
            title=f"Distribution of '{variable}'",
            xaxis_title=variable,
            yaxis_title="Count",
            bargap=0.05,
            height=500,
            width=800,
            showlegend=False,
        )

        fig.show()


def plot_forecast_vs_truth(
    true_data_dir: str,
    predictions_dir: str,
    config: str,
    filename: str,
    model_names: List[str],
    context_window: Union[int, str] = "max",
    normalize: bool = False,
) -> None:
    """
    Plots time series ground truth alongside predictions from one or more models.

    Parameters:
    - true_data_dir (str or Path): Directory containing the ground truth CSV file.
    - predictions_dir (str or Path): Base directory containing prediction subfolders per model/config.
    - config (str): Configuration name used to locate the prediction subfolder.
    - filename (str): Name of the CSV file containing the true series.
    - model_names (List[str]): Names of models whose predictions should be loaded and plotted.
    - context_window (int or 'max'): Number of steps shown before forecast begins (can be 'max' to use all available history).
    - normalize (bool): If True, the true series will be z-score normalized.
    """

    true_path = Path(true_data_dir, filename)
    df = pd.read_csv(true_path)

    # Extract the true series
    if "data" in df.columns:
        series = df["data"].values
    else:
        raise ValueError("Expected column named 'data' in the CSV file.")

    # Optionally normalize the series
    if normalize:
        scaler = StandardScaler()
        series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    forecast_horizon = None
    preds_by_model = {}

    # Load predictions for each model
    for model_name in model_names:
        pred_base = Path(predictions_dir, model_name, config)
        target_name = filename.replace(".csv", "")

        candidates = [
            p
            for p in pred_base.glob("*")
            if p.is_dir() and p.name.endswith(target_name)
        ]

        if not candidates:
            print(
                f"No prediction folder found for model '{model_name}' and file '{filename}'"
            )
            continue

        latest_folder = max(candidates, key=lambda p: p.name)
        y_pred = np.load(latest_folder / "y_pred.npy")

        if forecast_horizon is None:
            forecast_horizon = len(y_pred)
        preds_by_model[model_name] = y_pred

    if forecast_horizon is None:
        print("No predictions found for any model.")
        return

    # Determine context window
    if context_window == "max":
        context_window = len(series) - forecast_horizon
    total_length = context_window + forecast_horizon
    full_series = series[-total_length:]
    time = np.arange(total_length)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(time, full_series, label="True values", color="black")
    plt.axvline(context_window, color="red", linestyle=":", label="Forecast start")

    for model_name, y_pred in preds_by_model.items():
        plt.plot(
            time[context_window:], y_pred, linestyle="--", label=f"{model_name} (pred)"
        )

    plt.title(f"Forecast vs. Truth\n{filename} ({config})")
    plt.xlabel("Time step")
    plt.ylabel("Normalized value" if normalize else "Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_forecast_vs_truth_plotly(
    true_data_dir: str,
    predictions_dir: str,
    config: str,
    filename: str,
    model_names: List[str],
    context_window: Union[int, str] = "max",
    normalize: bool = False,
) -> None:
    """
    Plot true time series values and model predictions using Plotly.

    Parameters:
    - true_data_dir (str): Path to directory containing the ground truth .csv file.
    - predictions_dir (str): Base directory where predictions for each model/config are stored.
    - config (str): Subfolder name for the model configuration.
    - filename (str): CSV file containing the true time series data.
    - model_names (List[str]): List of model names whose predictions will be plotted.
    - context_window (int or 'max'): Number of steps before the forecast starts; 'max' uses all available history.
    - normalize (bool): Whether to z-score normalize the true series.
    """

    true_path = Path(true_data_dir, filename)
    df = pd.read_csv(true_path)

    if "data" in df.columns:
        series = df["data"].values
    else:
        raise ValueError("Expected a column named 'data' in the CSV file.")

    if normalize:
        scaler = StandardScaler()
        series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    forecast_horizon = None
    preds_by_model: dict[str, np.ndarray] = {}

    for model_name in model_names:
        pred_base = Path(predictions_dir, model_name, config)
        target_name = filename.replace(".csv", "")

        candidates = [
            p
            for p in pred_base.glob("*")
            if p.is_dir() and p.name.endswith(target_name)
        ]

        if not candidates:
            print(
                f"No prediction folder found for model '{model_name}' and file '{filename}'"
            )
            continue

        latest_folder = max(candidates, key=lambda p: p.name)
        y_pred = np.load(latest_folder / "y_pred.npy")

        if forecast_horizon is None:
            forecast_horizon = len(y_pred)
        preds_by_model[model_name] = y_pred

    if forecast_horizon is None:
        print("⚠️ No predictions found for any model.")
        return

    if context_window == "max":
        context_window = len(series) - forecast_horizon

    total_length = context_window + forecast_horizon
    full_series = series[-total_length:]
    time = np.arange(total_length)

    fig = go.Figure()

    # True values
    fig.add_trace(
        go.Scatter(
            x=time,
            y=full_series,
            mode="lines",
            name="True values",
            line=dict(color="black"),
        )
    )

    # Predictions
    for model_name, y_pred in preds_by_model.items():
        fig.add_trace(
            go.Scatter(
                x=time[context_window:],
                y=y_pred,
                mode="lines",
                name=f"{model_name} (pred)",
                line=dict(dash="dash"),
            )
        )

    # Forecast boundary
    fig.add_shape(
        type="line",
        x0=context_window,
        x1=context_window,
        y0=min(full_series),
        y1=max(full_series),
        line=dict(color="red", dash="dot"),
    )

    fig.update_layout(
        title=f"Forecast vs. Truth<br>{filename} ({config})",
        xaxis_title="Time step",
        yaxis_title="Normalized value" if normalize else "Value",
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1,
        ),
        height=500,
        width=800,
    )

    fig.show()


def interactive_dual_heatmap(df: pd.DataFrame, target_col: str = "mse", bins: int = 3):
    """Interactive dual heatmap showing how feature pairs relate to the target column's standard deviation."""

    feature_cols = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()

    def smart_bin(series, bins=bins):
        if series.nunique() <= 2:
            return pd.cut(series, bins=2, include_lowest=True, duplicates="drop")
        else:
            return pd.cut(series, bins=bins, include_lowest=True, duplicates="drop")

    def dual_heatmap(feat_x1, feat_y1, feat_x2, feat_y2):
        df_temp = df.copy()

        # Heatmap 1
        df_temp["x_bin1"] = smart_bin(df_temp[feat_x1])
        df_temp["y_bin1"] = smart_bin(df_temp[feat_y1])
        grouped1 = (
            df_temp.groupby(["x_bin1", "y_bin1"], observed=False)[target_col]
            .agg(std="std", count="count")
            .reset_index()
            .dropna()
        )
        heatmap1 = grouped1.pivot(index="y_bin1", columns="x_bin1", values="std")
        count1 = grouped1.pivot(index="y_bin1", columns="x_bin1", values="count")

        # Heatmap 2
        df_temp["x_bin2"] = smart_bin(df_temp[feat_x2])
        df_temp["y_bin2"] = smart_bin(df_temp[feat_y2])
        grouped2 = (
            df_temp.groupby(["x_bin2", "y_bin2"], observed=False)[target_col]
            .agg(std="std", count="count")
            .reset_index()
            .dropna()
        )
        heatmap2 = grouped2.pivot(index="y_bin2", columns="x_bin2", values="std")
        count2 = grouped2.pivot(index="y_bin2", columns="x_bin2", values="count")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            heatmap1,
            annot=count1,
            fmt=".0f",
            cmap="coolwarm",
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": f"{target_col} Std"},
            ax=ax1,
        )
        ax1.set_title(f"{feat_x1} vs. {feat_y1}")
        ax1.set_xlabel(feat_x1)
        ax1.set_ylabel(feat_y1)

        sns.heatmap(
            heatmap2,
            annot=count2,
            fmt=".0f",
            cmap="coolwarm",
            linewidths=0.5,
            linecolor="gray",
            cbar=False,
            ax=ax2,
        )
        ax2.set_title(f"{feat_x2} vs. {feat_y2}")
        ax2.set_xlabel(feat_x2)
        ax2.set_ylabel(feat_y2)

        plt.tight_layout()
        plt.show()

    interact(
        dual_heatmap,
        feat_x1=widgets.Dropdown(options=feature_cols, description="Feature X1"),
        feat_y1=widgets.Dropdown(options=feature_cols, description="Feature Y1"),
        feat_x2=widgets.Dropdown(options=feature_cols, description="Feature X2"),
        feat_y2=widgets.Dropdown(options=feature_cols, description="Feature Y2"),
    )


def interactive_dual_heatmap_plotly(
    df: pd.DataFrame, target_col: str = "mse", bins: int = 3
):
    """Interactive dual heatmap with Plotly showing feature-pair effects on target stddev + count annotations."""

    feature_cols = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()

    def smart_bin(series, bins=bins):
        if series.nunique() <= 2:
            return pd.cut(series, bins=2, include_lowest=True, duplicates="drop")
        else:
            return pd.cut(series, bins=bins, include_lowest=True, duplicates="drop")

    def dual_heatmap(feat_x1, feat_y1, feat_x2, feat_y2):
        df_temp = df.copy()

        # Heatmap 1
        df_temp["x_bin1"] = smart_bin(df_temp[feat_x1])
        df_temp["y_bin1"] = smart_bin(df_temp[feat_y1])
        grouped1 = (
            df_temp.groupby(["x_bin1", "y_bin1"], observed=False)[target_col]
            .agg(std="std", count="count")
            .reset_index()
            .dropna()
        )
        heatmap1 = grouped1.pivot(index="y_bin1", columns="x_bin1", values="std")
        count1 = grouped1.pivot(index="y_bin1", columns="x_bin1", values="count")

        # Heatmap 2
        df_temp["x_bin2"] = smart_bin(df_temp[feat_x2])
        df_temp["y_bin2"] = smart_bin(df_temp[feat_y2])
        grouped2 = (
            df_temp.groupby(["x_bin2", "y_bin2"], observed=False)[target_col]
            .agg(std="std", count="count")
            .reset_index()
            .dropna()
        )
        heatmap2 = grouped2.pivot(index="y_bin2", columns="x_bin2", values="std")
        count2 = grouped2.pivot(index="y_bin2", columns="x_bin2", values="count")

        # Gemeinsamer Farbbereich
        vmin = min(heatmap1.min().min(), heatmap2.min().min())
        vmax = max(heatmap1.max().max(), heatmap2.max().max())

        # Subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"{feat_x1} vs. {feat_y1}", f"{feat_x2} vs. {feat_y2}"],
        )

        xlabels1 = [f"{c.left:.2f}–{c.right:.2f}" for c in heatmap1.columns]
        ylabels1 = [f"{i.left:.2f}–{i.right:.2f}" for i in heatmap1.index]
        xlabels2 = [f"{c.left:.2f}–{c.right:.2f}" for c in heatmap2.columns]
        ylabels2 = [f"{i.left:.2f}–{i.right:.2f}" for i in heatmap2.index]

        # Heatmap 1
        fig.add_trace(
            go.Heatmap(
                z=heatmap1.values,
                x=xlabels1,
                y=ylabels1,
                text=count1.values.astype(str),
                texttemplate="%{text}",
                colorscale=px.colors.sequential.RdBu[::-1],
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title=f"{target_col} Std"),
                hovertemplate="x: %{x}<br>y: %{y}<br>std: %{z:.2f}<br>count: %{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Heatmap 2
        fig.add_trace(
            go.Heatmap(
                z=heatmap2.values,
                x=xlabels2,
                y=ylabels2,
                text=count2.values.astype(str),
                texttemplate="%{text}",
                colorscale=px.colors.sequential.RdBu[::-1],
                zmin=vmin,
                zmax=vmax,
                showscale=False,
                hovertemplate="x: %{x}<br>y: %{y}<br>std: %{z:.2f}<br>count: %{text}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=600,
            width=1000,
            title_text="Dual Heatmap: Std Dev of Target with Count Annotations",
        )
        fig.update_xaxes(title_text=feat_x1, row=1, col=1)
        fig.update_yaxes(title_text=feat_y1, row=1, col=1)
        fig.update_xaxes(title_text=feat_x2, row=1, col=2)
        fig.update_yaxes(title_text=feat_y2, row=1, col=2)

        fig.show()

    interact(
        dual_heatmap,
        feat_x1=widgets.Dropdown(options=feature_cols, description="Feature X1"),
        feat_y1=widgets.Dropdown(options=feature_cols, description="Feature Y1"),
        feat_x2=widgets.Dropdown(options=feature_cols, description="Feature X2"),
        feat_y2=widgets.Dropdown(options=feature_cols, description="Feature Y2"),
    )


def interactive_binned_boxplot(df: pd.DataFrame, target_col: str = "mse"):
    """
    Interactive Plotly boxplot showing the distribution of a target variable (e.g. MSE)
    across bins of a selected numeric feature.

    Parameters:
    - df: DataFrame containing the data
    - target_col: the column to use as boxplot y-values (e.g. 'mse')
    - exclude_cols: optional list of columns to exclude from dropdown
    """

    # Select numeric meta-features, excluding target_col and any specified
    meta_features = df.select_dtypes(include="number").columns

    # Widgets
    feature_dropdown = widgets.Dropdown(
        options=meta_features.tolist(), description="Feature:"
    )
    bin_dropdown = widgets.Dropdown(
        options=list(range(2, 11)), value=3, description="Bins:"
    )

    output_plot = widgets.Output()

    def update_plot(change=None):
        with output_plot:
            clear_output(wait=True)
            feat = feature_dropdown.value
            num_bins = bin_dropdown.value

            df_feat = df.copy()
            df_feat["bin"] = pd.cut(df_feat[feat], bins=num_bins)
            df_feat["bin_label"] = df_feat["bin"].astype(str)

            fig = go.Figure()
            for bin_label in sorted(df_feat["bin_label"].dropna().unique()):
                y_vals = df_feat.loc[df_feat["bin_label"] == bin_label, target_col]
                if len(y_vals) == 0:
                    continue
                fig.add_trace(
                    go.Box(
                        y=y_vals,
                        name=bin_label,
                        boxpoints="all",
                        marker_color="royalblue",
                    )
                )

            fig.update_layout(
                title=f"{target_col.upper()} by binned {feat}",
                xaxis_title=f"{feat} (binned)",
                yaxis_title=target_col.upper(),
                template="plotly_white",
                height=500,
                width=800,
                margin=dict(t=60, r=20, b=60, l=60),
                legend=dict(x=1, y=1, xanchor="right", yanchor="top"),
            )

            fig.show()

    # Trigger plot updates
    feature_dropdown.observe(update_plot, names="value")
    bin_dropdown.observe(update_plot, names="value")

    # Initial display
    display(widgets.HBox([feature_dropdown, bin_dropdown]))
    display(output_plot)
    update_plot()


def interactive_mse_feature_combo_plot(
    df: pd.DataFrame, target_col: str = "mse", max_features: int = 5, bin_count: int = 4
):
    """
    Interactive plot showing how the standard deviation of a target (e.g., MSE)
    evolves when combining up to N meta-features using binning.

    Parameters:
    - df: DataFrame containing meta-features and a target column
    - target_col: the column used to calculate std (e.g. 'mse')
    - max_features: maximum number of features to combine
    - bin_count: number of bins for pd.cut
    """

    meta_features = df.columns.drop(target_col).tolist()

    # Create dropdowns for selecting features
    dropdowns = [
        widgets.Dropdown(options=[""] + meta_features, description=f"Feature {i + 1}")
        for i in range(max_features)
    ]
    display(widgets.VBox(dropdowns))

    button = widgets.Button(description="Update Plot")
    output = widgets.Output()
    display(button, output)

    def compute_combo_bins(df_input, selected_features):
        df_copy = df_input.copy()
        bin_cols = []
        for f in selected_features:
            col = f + "_bin"
            df_copy[col] = pd.cut(df_copy[f], bins=bin_count)
            bin_cols.append(col)
        df_copy["combo_bin"] = df_copy[bin_cols].astype(str).agg("_".join, axis=1)
        return df_copy

    def update_plot(change=None):
        with output:
            clear_output(wait=True)

            selected = [d.value for d in dropdowns if d.value]
            if not selected:
                print("Please select at least one feature.")
                return

            results = []
            for i in range(1, len(selected) + 1):
                combo = selected[:i]
                df_combo = compute_combo_bins(df, combo)

                grouped = df_combo.groupby("combo_bin")[target_col].agg(
                    ["mean", "std", "count"]
                )

                avg_std = grouped["std"].mean()
                max_std = grouped["std"].max()
                weighted_std = (grouped["std"] * grouped["count"]).sum() / grouped[
                    "count"
                ].sum()

                # filter: only include groups with at least 30 samples
                valid_bins = df_combo["combo_bin"].value_counts()
                keep_bins = valid_bins[valid_bins >= 30].index
                df_filtered = df_combo[df_combo["combo_bin"].isin(keep_bins)]
                grouped_filt = df_filtered.groupby("combo_bin")[target_col].agg(
                    ["std", "count"]
                )
                avg_std_filt30 = (
                    grouped_filt["std"].mean() if not grouped_filt.empty else np.nan
                )

                results.append(
                    {
                        "features": " + ".join(
                            [f"f{i + 1}" for i in range(len(combo))]
                        ),
                        "avg_std": avg_std,
                        "max_std": max_std,
                        "weighted_std": weighted_std,
                        "avg_std_filt30": avg_std_filt30,
                    }
                )

            results_df = pd.DataFrame(results)

            # Plotting
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results_df["features"],
                    y=results_df["avg_std"],
                    mode="lines+markers",
                    name="Average Std of MSE",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=results_df["features"],
                    y=results_df["weighted_std"],
                    mode="lines+markers",
                    name="Weighted Std (all bins)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=results_df["features"],
                    y=results_df["avg_std_filt30"],
                    mode="lines+markers",
                    name="Avg Std (bins ≥ 30)",
                )
            )
            fig.update_layout(
                title="MSE Standard Deviation across Feature Combinations",
                xaxis_title="Feature Combination",
                yaxis_title="MSE Std",
                template="plotly_white",
                height=500,
                width=850,
            )
            fig.show()

    # Connect button
    button.on_click(update_plot)


def load_meta_model_benchmarking_results(folder_name, base_path=META_MODEL_RESULTS_DIR):
    folder = Path(base_path) / folder_name

    # Load arrays
    aleatoric_all = np.load(folder / "aleatoric_all.npy")
    aleatoric_pred = np.mean(aleatoric_all, axis=0)
    aleatoric_true = np.load(folder / "aleatoric_true.npy")

    epistemic_all = np.load(folder / "epistemic_all.npy")
    epistemic_pred = np.mean(epistemic_all, axis=0)

    y_pred_all = np.load(folder / "y_pred_all.npy")
    y_pred = np.mean(y_pred_all, axis=0)

    y_test = np.load(folder / "y_true.npy")
    y_train = np.load(folder / "y_train.npy")
    X_test = np.load(folder / "X_test.npy")
    X_train = np.load(folder / "X_train.npy")

    # Determine number of features
    n_features = X_test.shape[1] if X_test.ndim > 1 else 1
    feature_names = [f"x{i + 1}" for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X_test, columns=feature_names)
    df["aleatoric_pred"] = aleatoric_pred
    df["epistemic_pred"] = epistemic_pred
    df["y_pred"] = y_pred
    df["y_true"] = y_test
    df["aleatoric_true"] = aleatoric_true

    # Return both dict and df
    return {
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "y_pred": y_pred,
        "y_pred_all": y_pred_all,
        "aleatoric_all": aleatoric_all,
        "aleatoric_pred": aleatoric_pred,
        "aleatoric_true": aleatoric_true,
        "epistemic_all": epistemic_all,
        "epistemic_pred": epistemic_pred,
        "df": df,
        "feature_names": feature_names,
    }


def load_meta_model_benchmarking_filtered_results(
    folder_name, min_val, max_val, base_path=META_MODEL_RESULTS_DIR
):
    data = load_meta_model_benchmarking_results(folder_name, base_path)
    df = data["df"]
    feature_names = data["feature_names"]

    # Build filter dynamically for each feature
    filter_mask = np.ones(len(df), dtype=bool)
    for fname in feature_names:
        filter_mask &= (df[fname] > min_val) & (df[fname] < max_val)

    df_filtered = df[filter_mask]

    return {
        "X_test": df_filtered[feature_names].values,
        "y_test": df_filtered["y_true"].values,
        "y_pred": df_filtered["y_pred"].values,
        "aleatoric_pred": df_filtered["aleatoric_pred"].values,
        "aleatoric_true": df_filtered["aleatoric_true"].values,
        "epistemic_pred": df_filtered["epistemic_pred"].values,
        "df": df_filtered,
        "feature_names": feature_names,
    }


def plot_uncertainties_and_predictions(
    X_test,
    aleatoric_all,
    aleatoric_true,
    epistemic_all,
    y_test,
    y_pred_all,
    ood_bounds=(-2, 2),
    figsize=(18, 5),
    title="Uncertainty and Prediction Overview",
    x_min=None,
    x_max=None,
    y_margin_factor=0.1,  # % margin for y-limits on epistemic
):
    """
    Plot aleatoric and epistemic uncertainties and predicted outputs.
    Middle plot (epistemic) y-axis auto-scaled to its mean prediction curve.
    """
    x = X_test.squeeze()

    # Optional x-axis filtering
    if x_min is not None and x_max is not None:
        mask = (x >= x_min) & (x <= x_max)
        x = x[mask]
        aleatoric_all = aleatoric_all[:, mask]
        aleatoric_true = aleatoric_true[mask]
        epistemic_all = epistemic_all[:, mask]
        y_test = y_test[mask]
        y_pred_all = y_pred_all[:, mask]

    aleatoric_pred = np.mean(aleatoric_all, axis=0)
    epistemic_pred = np.mean(epistemic_all, axis=0)
    y_pred = np.mean(y_pred_all, axis=0)

    ood_left, ood_right = ood_bounds

    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True)

    # 1. Aleatoric
    for pred in aleatoric_all:
        axs[0].plot(x, pred, color="blue", alpha=0.2, linewidth=1)
    axs[0].plot(x, aleatoric_pred, color="blue", linewidth=2, label="Predicted (mean)")
    axs[0].plot(x, aleatoric_true, color="red", linewidth=2, label="True")
    axs[0].axvline(x=ood_left, color="gray", linestyle="--")
    axs[0].axvline(x=ood_right, color="gray", linestyle="--")
    axs[0].set_title("Aleatoric Uncertainty")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Uncertainty (σ)")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Epistemic (center y-axis on predicted mean only)
    for pred in epistemic_all:
        axs[1].plot(x, pred, color="blue", alpha=0.2, linewidth=1)
    axs[1].plot(x, epistemic_pred, color="blue", linewidth=2, label="Predicted (mean)")
    axs[1].axvline(x=ood_left, color="gray", linestyle="--")
    axs[1].axvline(x=ood_right, color="gray", linestyle="--")
    axs[1].set_title("Epistemic Uncertainty")
    axs[1].set_xlabel("x")
    axs[1].legend()
    axs[1].grid(True)

    if np.all(np.isfinite(epistemic_pred)):
        y_min = np.min(epistemic_pred)
        y_max = np.max(epistemic_pred)
        margin = y_margin_factor * (y_max - y_min + 1e-8)  # vermeiden von 0-Spanne
        axs[1].set_ylim(y_min - margin, y_max + margin)
    else:
        # → Dynamische y-Achse auf Basis des Mittelwerts
        y_min = np.min(epistemic_pred)
        y_max = np.max(epistemic_pred)
        margin = y_margin_factor * (y_max - y_min)
        axs[1].set_ylim(y_min - margin, y_max + margin)

    # 3. Prediction
    for pred in y_pred_all:
        axs[2].plot(x, pred, color="blue", alpha=0.2, linewidth=1)
    axs[2].plot(x, y_pred, color="blue", linewidth=2, label="Predicted Output (mean)")
    axs[2].plot(x, y_test, color="red", linewidth=2, label="True Output")
    axs[2].axvline(x=ood_left, color="gray", linestyle="--")
    axs[2].axvline(x=ood_right, color="gray", linestyle="--")
    axs[2].set_title("Prediction vs. Ground Truth")
    axs[2].set_xlabel("x")
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compute_surface_grid(X, values, feature_indices=(0, 1)):
    """
    Reshapes values onto a 2D grid based on two selected features.
    Assumes X is structured as a full grid (e.g., created via meshgrid).
    """
    f1_idx, f2_idx = feature_indices
    f1 = X[:, f1_idx]
    f2 = X[:, f2_idx]

    f1_unique = np.unique(f1)
    f2_unique = np.unique(f2)
    grid_shape = (len(f2_unique), len(f1_unique))  # rows = y, columns = x

    f1_grid = f1.reshape(grid_shape)
    f2_grid = f2.reshape(grid_shape)
    surface_values = values.flatten().reshape(grid_shape)

    return f1_grid, f2_grid, surface_values


def plot_3d_surfaces(X, noise_true, aleatoric_pred, feature_indices=(0, 1)):
    """
    Plots three 3D surfaces side by side:
    1. True aleatoric uncertainty
    2. Predicted aleatoric uncertainty
    3. Absolute prediction error
    Assumes X is grid-structured (from meshgrid).
    """
    # Flatten and compute error
    noise_true = noise_true.flatten()
    aleatoric_pred = aleatoric_pred.flatten()
    prediction_error = np.abs(noise_true - aleatoric_pred)

    # Compute surfaces (no interpolation needed)
    f1_grid, f2_grid, surface_true = compute_surface_grid(
        X, noise_true, feature_indices
    )
    _, _, surface_pred = compute_surface_grid(X, aleatoric_pred, feature_indices)
    _, _, surface_error = compute_surface_grid(X, prediction_error, feature_indices)

    # Plotting
    fig = plt.figure(figsize=(20, 6))

    for i, (surface, title, cmap) in enumerate(
        [
            (surface_true, "True Uncertainty", "viridis"),
            (surface_pred, "Predicted Aleatoric Uncertainty", "viridis"),
            (surface_error, "Prediction Error (|true - pred|)", "viridis"),
        ]
    ):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # Set vmin/vmax only for the third plot
        if i == 2:
            surf = ax.plot_surface(
                f1_grid,
                f2_grid,
                surface,
                cmap=cmap,
                edgecolor="none",
                alpha=0.95,
                vmin=0,
                vmax=2,
            )
            ax.set_zlim(0, 2)
        else:
            surf = ax.plot_surface(
                f1_grid, f2_grid, surface, cmap=cmap, edgecolor="none", alpha=0.95
            )

        ax.set_xlabel(f"x{feature_indices[0] + 1}")
        ax.set_ylabel(f"x{feature_indices[1] + 1}")
        ax.set_zlabel("Uncertainty")
        ax.set_title(title)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()


def plot_3d_surfaces_interactive(X, noise_true, aleatoric_pred, feature_indices=(0, 1)):
    """
    Plots three interactive 3D surfaces using Plotly:
    - True aleatoric uncertainty
    - Predicted aleatoric uncertainty
    - Absolute prediction error
    Assumes X is a regular grid.
    """
    noise_true = noise_true.flatten()
    aleatoric_pred = aleatoric_pred.flatten()
    prediction_error = np.abs(noise_true - aleatoric_pred)

    # Compute surface grids directly
    f1_grid, f2_grid, surface_true = compute_surface_grid(
        X, noise_true, feature_indices
    )
    _, _, surface_pred = compute_surface_grid(X, aleatoric_pred, feature_indices)
    _, _, surface_error = compute_surface_grid(X, prediction_error, feature_indices)

    surfaces = [
        (surface_true, "True Uncertainty", "Viridis", "True"),
        (surface_pred, "Predicted Aleatoric Uncertainty", "Viridis", "Predicted"),
        (surface_error, "Prediction Error (|true - pred|)", "Viridis", "Error"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "surface"}] * 3],
        subplot_titles=[s[1] for s in surfaces],
        horizontal_spacing=0.08,
    )

    # Colorbar positions
    colorbar_offsets = [0.34, 0.67, 1.0]

    for i, (Z, title, cmap, cb_title) in enumerate(surfaces):
        fig.add_trace(
            go.Surface(
                z=Z,
                x=f1_grid,
                y=f2_grid,
                colorscale=cmap,
                showscale=True,
                colorbar=dict(
                    title=cb_title, len=0.7, x=colorbar_offsets[i], xanchor="left"
                ),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        height=400,
        title_text="Interactive 3D Uncertainty Surfaces",
        margin=dict(l=0, r=20, t=40, b=0),
    )

    fig.show()


def plot_3d_uncertainty_surface_grid(
    X, uncertainty, feature_indices=(0, 1), title="Uncertainty Surface", cmap="viridis"
):
    """
    Plots a 3D uncertainty surface assuming X is structured as a full grid
    (i.e., generated from np.meshgrid).

    Parameters:
    - X: Input data array of shape (n_samples, n_features), generated from a grid
    - uncertainty: Array of uncertainty values, shape (n_samples,) or (n_samples, 1)
    - feature_indices: Tuple of two feature indices to plot (e.g., (0, 1))
    - title: Title of the plot
    - cmap: Colormap for the surface
    """
    f1_idx, f2_idx = feature_indices
    f1 = X[:, f1_idx]
    f2 = X[:, f2_idx]

    # Automatically determine grid shape from unique values
    f1_unique = np.unique(f1)
    f2_unique = np.unique(f2)
    grid_shape = (len(f2_unique), len(f1_unique))  # rows = y-axis, columns = x-axis

    # Reshape data for surface plot
    f1_grid = f1.reshape(grid_shape)
    f2_grid = f2.reshape(grid_shape)
    uncertainty = uncertainty.flatten()  # ensure 1D
    uncertainty_grid = uncertainty.reshape(grid_shape)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        f1_grid, f2_grid, uncertainty_grid, cmap=cmap, edgecolor="none", alpha=0.95
    )
    ax.set_xlabel(f"x{f1_idx + 1}")
    ax.set_ylabel(f"x{f2_idx + 1}")
    ax.set_zlabel("Uncertainty")
    ax.set_title(title)
    # ax.view_init(elev=30, azim=135)  # Adjust elevation and azimuth
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()


def plot_3d_uncertainty_surface_grid_plotly(
    X, uncertainty, feature_indices=(0, 1), title="Uncertainty Surface", cmap="Plasma"
):
    """
    Plots a 3D uncertainty surface assuming X is structured as a full grid
    (i.e., generated from np.meshgrid).

    Parameters:
    - X: Input data array of shape (n_samples, n_features), generated from a grid
    - uncertainty: Array of uncertainty values, shape (n_samples,) or (n_samples, 1)
    - feature_indices: Tuple of two feature indices to plot (e.g., (0, 1))
    - title: Title of the plot
    - cmap: Colormap for the surface
    """
    f1_idx, f2_idx = feature_indices
    f1 = X[:, f1_idx]
    f2 = X[:, f2_idx]

    # define grid form
    f1_unique = np.unique(f1)
    f2_unique = np.unique(f2)
    grid_shape = (len(f2_unique), len(f1_unique))

    # construct grid data
    f1_grid = f1.reshape(grid_shape)
    f2_grid = f2.reshape(grid_shape)
    uncertainty_grid = uncertainty.flatten().reshape(grid_shape)

    fig = go.Figure(
        data=[
            go.Surface(
                x=f1_grid,
                y=f2_grid,
                z=uncertainty_grid,
                colorscale=cmap,
                colorbar=dict(title="Uncertainty"),
                showscale=True,
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"x{f1_idx + 1}",
            yaxis_title=f"x{f2_idx + 1}",
            zaxis_title="Uncertainty",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.show()
