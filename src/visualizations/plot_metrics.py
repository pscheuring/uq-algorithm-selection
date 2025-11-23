import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
    ScalarFormatter,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_metrics_grid_exp1(
    df_metrics: pd.DataFrame,
    model_col: str = "model_name",
    config_col: str = "train_instances",
    metric_bases: tuple[str, str, str] = ("nll_id", "mae_aleatoric", "auroc_ood"),
    mean_suffix: str = "_median",
    bottom_titles: tuple[str, str, str] = (
        "↓ NLL (ID)",
        "↓ MAE aleatoric uncertainty",
        "↑ AUROC OOD",
    ),
    figsize: tuple[float, float] = (20.0, 6.0),
) -> tuple[Figure, Legend]:
    """Plot a 1×3 metrics grid for experiment 1 with a zoomed inset on the left.

    Panels (fixed order):
        - left:   NLL (ID)
        - middle: MAE aleatoric (log-y)
        - right:  AUROC OOD

    All panels use a log-x scale with base 5. The left panel includes a zoom
    rectangle and a zoomed inset with connector lines.

    Args:
        df_metrics: Tidy DataFrame with metrics. Must contain ``model_col``,
            ``config_col`` and metric columns named ``{base}_run*`` and
            ``{base}{mean_suffix}`` for each base in ``metric_bases``.
        model_col: Column name for raw model keys.
        config_col: Column name for the x-axis (must be > 0 for log-x).
        metric_bases: Metric base names for the three panels (left, middle, right).
        mean_suffix: Suffix of the aggregate metric column.
        bottom_titles: Titles for the three subplots.
        figsize: Figure size in inches.

    Returns:
        tuple[Figure, Legend]: The created figure and the bottom legend.
    """

    # Fixed behavior / local style
    show_runs = True
    show_median_markers = True
    x_log_base = 5.0

    zoom_xrange = (200.0, 100_000.0)
    zoom_ylim = (-2.8, -2.7)
    inset_height_frac = 0.8
    inset_y_shift = 0.15
    shrink_factor = 0.85

    # Model label mapping and plotting order
    name_map: dict[str, str] = {
        "bbb": "Bayes by Backprop",
        "mcdropout": "MC Dropout",
        "ensemble": "Deep Ensemble",
        "evidential": "DER",
    }
    raw_order: list[str] = ["bbb", "mcdropout", "ensemble", "evidential"]

    # Colors per pretty label
    pretty_colors: dict[str, str] = {
        "Bayes by Backprop": "tab:olive",
        "MC Dropout": "tab:purple",
        "Deep Ensemble": "tab:brown",
        "DER": "tab:pink",
    }

    def _run_cols_and_med(base: str) -> tuple[list[str], str]:
        """Return run columns and median/aggregate column for a metric base."""
        runs = sorted(
            [c for c in df_metrics.columns if c.startswith(f"{base}_run")],
            key=lambda x: int(x.split("run")[-1])
            if x.split("run")[-1].isdigit()
            else 9999,
        )
        med = f"{base}{mean_suffix}"
        return runs, med

    # Figure & data
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=False)
    ax_left, ax_mid, ax_right = axs
    y_labels = ["NLL", "MAE", "AUROC"]

    work = df_metrics.copy()
    work[config_col] = pd.to_numeric(work[config_col], errors="coerce")
    x_vals = np.array(
        sorted(work.loc[work[config_col] > 0, config_col].unique()),
        dtype=float,
    )

    # Main panels
    for idx, (ax, base, title) in enumerate(zip(axs, metric_bases, bottom_titles)):
        run_cols, med_col = _run_cols_and_med(base)

        for raw_key in raw_order:
            sub = work[work[model_col] == raw_key].sort_values(config_col)
            if sub.empty:
                continue

            label = name_map.get(raw_key, raw_key)
            color = pretty_colors[label]

            if show_runs:
                for rc in run_cols:
                    if rc == med_col or rc not in sub:
                        continue
                    ax.plot(
                        sub[config_col],
                        sub[rc],
                        color=color,
                        linewidth=0.8,  # per-run linewidth (no rcParam available)
                        alpha=0.35,  # per-run alpha
                    )

            # Median line uses global `lines.linewidth` from style
            ax.plot(
                sub[config_col],
                sub[med_col],
                color=color,
                label=label,
                zorder=5,
                marker=("o" if show_median_markers else None),
                markersize=(6 if show_median_markers else 0),
                markerfacecolor=("white" if show_median_markers else None),
                markeredgewidth=(1.5 if show_median_markers else 0),
            )

        ax.set_title(title)
        ax.set_xlabel(r"$N_\mathrm{unique}$")
        ax.set_ylabel(y_labels[idx])
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

        # log-x (base 5), ticks at data values, no minor ticks
        ax.set_xscale("log", base=x_log_base)
        ax.xaxis.set_major_locator(LogLocator(base=x_log_base))
        if x_vals.size:
            ax.set_xticks(x_vals)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())

    # Middle panel: log-y
    ax_mid.set_yscale("log", base=10)
    ax_mid.yaxis.set_major_locator(LogLocator(base=10))
    ax_mid.yaxis.set_minor_formatter(NullFormatter())

    # x-ticks: e.g. 1000 → "1k"
    def _format_k(x: float, _pos: int) -> str:
        return f"{int(x / 1000)}k" if x >= 1000 else str(int(x))

    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k))

    # Zoom rectangle + inset on left panel
    x0_r, x1_r = zoom_xrange
    y0_r, y1_r = zoom_ylim

    rect = Rectangle(
        (x0_r, y0_r),
        x1_r - x0_r,
        y1_r - y0_r,
        fill=False,
        ec="black",
        lw=1.0,
    )
    ax_left.add_patch(rect)

    # Map zoom region to axes-fraction coordinates
    x0_disp = ax_left.transData.transform((x0_r, 0))[0]
    x1_disp = ax_left.transData.transform((x1_r, 0))[0]
    ax0_disp = ax_left.transAxes.transform((0, 0))[0]
    ax1_disp = ax_left.transAxes.transform((1, 0))[0]
    ax_w_disp = ax1_disp - ax0_disp
    left_frac = (x0_disp - ax0_disp) / ax_w_disp
    right_frac = (x1_disp - ax0_disp) / ax_w_disp

    center_frac = (left_frac + right_frac) / 2
    half_width = (right_frac - left_frac) * shrink_factor / 2
    left_frac = center_frac - half_width
    right_frac = center_frac + half_width
    width_frac = max(1e-6, right_frac - left_frac)

    axins = inset_axes(
        ax_left,
        width=f"{width_frac * 100:.6f}%",
        height=f"{inset_height_frac * 100:.6f}%",
        loc="upper left",
        bbox_to_anchor=(
            left_frac,
            1 - inset_height_frac - inset_y_shift,
            1,
            inset_height_frac,
        ),
        bbox_transform=ax_left.transAxes,
        borderpad=0.0,
    )

    axins.set_xscale("log", base=x_log_base)
    axins.set_xlim(x0_r, x1_r)
    axins.set_ylim(y0_r, y1_r)
    axins.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axins.xaxis.set_minor_locator(NullLocator())
    axins.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=False,
        length=3.5,
    )
    axins.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=False,
        length=3.5,
    )

    # Inset data (same metric as left panel)
    base_left = metric_bases[0]
    run_cols_l, med_col_l = _run_cols_and_med(base_left)
    for raw_key in raw_order:
        sub = work[work[model_col] == raw_key].sort_values(config_col)
        if sub.empty:
            continue

        label = name_map.get(raw_key, raw_key)
        color = pretty_colors[label]
        mask = (sub[config_col] >= x0_r) & (sub[config_col] <= x1_r)

        if show_runs:
            for rc in run_cols_l:
                if rc == med_col_l or rc not in sub:
                    continue
                axins.plot(
                    sub.loc[mask, config_col],
                    sub.loc[mask, rc],
                    color=color,
                    linewidth=0.8,
                    alpha=0.35,
                )

        if med_col_l in sub:
            axins.plot(
                sub.loc[mask, config_col],
                sub.loc[mask, med_col_l],
                color=color,
                marker=("o" if show_median_markers else None),
                markersize=(6 if show_median_markers else 0),
                markerfacecolor=("white" if show_median_markers else None),
                markeredgewidth=(1.5 if show_median_markers else 0),
            )

    # Connector lines
    con_2_to_3 = ConnectionPatch(
        xyA=(x0_r, y1_r),
        xyB=(0, 0),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax_left,
        axesB=axins,
        color="black",
        lw=1.0,
    )
    con_1_to_4 = ConnectionPatch(
        xyA=(x1_r, y1_r),
        xyB=(1, 0),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax_left,
        axesB=axins,
        color="black",
        lw=1.0,
    )
    ax_left.add_artist(con_2_to_3)
    ax_left.add_artist(con_1_to_4)

    # Legend (figure-level, fixed order)
    pretty_order = [name_map[k] for k in raw_order if k in name_map]
    line_width = plt.rcParams.get("lines.linewidth", 2.5)
    proxies = [
        Line2D([0], [0], color=pretty_colors[p], lw=line_width, label=p)
        for p in pretty_order
    ]
    legend = fig.legend(
        proxies,
        pretty_order,
        loc="lower center",
        ncol=max(1, len(pretty_order)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    return fig, legend


def plot_metrics_grid_exp2(
    df_metrics: pd.DataFrame,
    model_col: str = "model_name",
    config_col: str = "train_repeats",
    metric_bases: tuple[str, str, str] = (
        "nll_id",
        "mae_aleatoric",
        "auroc_ood",
    ),
    mean_suffix: str = "_median",
    bottom_titles: tuple[str, str, str] = (
        "↓ NLL (ID)",
        "↓ MAE aleatoric uncertainty",
        "↑ AUROC OOD",
    ),
    figsize: tuple[float, float] = (20.0, 6.0),
    show_runs: bool = True,
    show_median_markers: bool = True,
) -> tuple[Figure, list[plt.Axes]]:
    """Plot a 1×3 metrics grid for experiment 2 with a zoomed inset on the left.

    Panels (fixed order, semantics like exp1):
        - left:   first metric in ``metric_bases``
        - middle: second metric (log-y)
        - right:  third metric

    This function mirrors the style and layout of :func:`plot_metrics_grid_exp1`,
    but uses a linear x-axis (train repeats) and returns the axes instead of
    a legend object.

    Args:
        df_metrics: Tidy DataFrame with metrics.
        model_col: Column name for raw model keys.
        config_col: Column name for the x-axis (number of repeats).
        metric_bases: Metric base names for the three panels.
        mean_suffix: Suffix of the aggregate metric column.
        bottom_titles: Titles for the three subplots.
        figsize: Figure size in inches.
        show_runs: Whether to plot all individual run curves.
        show_median_markers: Whether to show markers on median curves.

    Returns:
        tuple[Figure, list[plt.Axes]]: The created figure and the list of axes.
    """

    work = df_metrics.copy()
    work[config_col] = pd.to_numeric(work[config_col], errors="coerce").clip(lower=0)

    # Pretty labels and ordering
    name_map: dict[str, str] = {
        "bbb": "Bayes by Backprop",
        "mcdropout": "MC Dropout",
        "ensemble": "Deep Ensemble",
        "evidential": "DER",
    }
    work["__model_label__"] = work[model_col].map(name_map).fillna(work[model_col])

    desired_order = ["Bayes by Backprop", "MC Dropout", "Deep Ensemble", "DER"]
    found = list(pd.unique(work["__model_label__"]))
    model_names = [m for m in desired_order if m in found] + [
        m for m in found if m not in desired_order
    ]

    default_colors: dict[str, str] = {
        "Bayes by Backprop": "tab:olive",
        "MC Dropout": "tab:purple",
        "Deep Ensemble": "tab:brown",
        "DER": "tab:pink",
    }
    tab10 = plt.cm.tab10.colors
    label_to_color: dict[str, str] = {
        m: default_colors.get(m, tab10[i % 10]) for i, m in enumerate(model_names)
    }

    def _run_cols_and_med(base: str) -> tuple[list[str], str]:
        runs = sorted(
            [c for c in work.columns if c.startswith(f"{base}_run")],
            key=lambda x: int(x.split("run")[-1])
            if x.split("run")[-1].isdigit()
            else 9999,
        )
        return runs, f"{base}{mean_suffix}"

    # Figure & axes
    fig, axs_arr = plt.subplots(1, 3, figsize=figsize, sharex=False)
    axs = list(axs_arr)
    ax_left, ax_mid, ax_right = axs
    y_labels = ["NLL", "MAE", "AUROC"]

    # Fixed x ticks
    x_vals = np.array([0, 3, 6, 9, 12, 15], dtype=float)

    member_linewidth = 0.8
    member_alpha = 0.35
    line_width = plt.rcParams.get("lines.linewidth", 2.5)

    # Main panels
    for idx, (ax, base, title) in enumerate(zip(axs, metric_bases, bottom_titles)):
        run_cols, med_col = _run_cols_and_med(base)
        for m in model_names:
            sub = work[work["__model_label__"] == m].sort_values(config_col)
            if sub.empty:
                continue
            color = label_to_color[m]

            if show_runs:
                for rc in run_cols:
                    if rc == med_col or rc not in sub:
                        continue
                    ax.plot(
                        sub[config_col],
                        sub[rc],
                        color=color,
                        linewidth=member_linewidth,
                        alpha=member_alpha,
                    )

            ax.plot(
                sub[config_col],
                sub[med_col],
                color=color,
                linewidth=line_width,
                label=m,
                zorder=5,
                marker=("o" if show_median_markers else None),
                markersize=(6 if show_median_markers else 0),
                markerfacecolor=("white" if show_median_markers else None),
                markeredgewidth=(1.5 if show_median_markers else 0),
            )

        ax.set_xlim(left=0)
        ax.set_xticks(x_vals)
        ax.set_ylabel(y_labels[idx])
        ax.set_xlabel(r"$N_\text{rep}$")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)

    # Middle panel: log-y
    ax_mid.set_yscale("log", base=10)
    ax_mid.yaxis.set_major_locator(LogLocator(base=10))
    ax_mid.yaxis.set_minor_formatter(NullFormatter())

    # Left panel: zoom rectangle + inset
    x0_r, x1_r = 3.0, 15.0
    y0_r, y1_r = -2.8, -2.63

    rect = Rectangle(
        (x0_r, y0_r),
        x1_r - x0_r,
        y1_r - y0_r,
        fill=False,
        ec="black",
        lw=1.0,
    )
    ax_left.add_patch(rect)

    x0_disp = ax_left.transData.transform((x0_r, 0))[0]
    x1_disp = ax_left.transData.transform((x1_r, 0))[0]
    ax0_disp = ax_left.transAxes.transform((0, 0))[0]
    ax1_disp = ax_left.transAxes.transform((1, 0))[0]
    ax_w_disp = ax1_disp - ax0_disp
    left_frac = (x0_disp - ax0_disp) / ax_w_disp
    right_frac = (x1_disp - ax0_disp) / ax_w_disp

    shrink_factor = 0.85
    center_frac = (left_frac + right_frac) / 2
    half_width = (right_frac - left_frac) * shrink_factor / 2
    left_frac = center_frac - half_width
    right_frac = center_frac + half_width
    width_frac = right_frac - left_frac

    inset_height_frac = 0.8
    inset_y_shift = 0.15

    axins = inset_axes(
        ax_left,
        width=f"{width_frac * 100:.6f}%",
        height=f"{inset_height_frac * 100:.6f}%",
        loc="upper left",
        bbox_to_anchor=(
            left_frac,
            1 - inset_height_frac - inset_y_shift,
            1,
            inset_height_frac,
        ),
        bbox_transform=ax_left.transAxes,
        borderpad=0.0,
    )

    axins.set_xlim(x0_r, x1_r)
    axins.set_ylim(y0_r, y1_r)
    axins.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    xticks_inset = [t for t in x_vals if x0_r <= t <= x1_r]
    axins.set_xticks(xticks_inset)
    axins.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=False,
        length=3.5,
    )
    axins.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=False,
        length=3.5,
    )

    # Inset data
    run_cols_l, med_col_l = _run_cols_and_med(metric_bases[0])
    for m in model_names:
        sub = work[work["__model_label__"] == m].sort_values(config_col)
        if sub.empty:
            continue
        color = label_to_color[m]
        mask = (sub[config_col] >= x0_r) & (sub[config_col] <= x1_r)

        if show_runs:
            for rc in run_cols_l:
                if rc == med_col_l or rc not in sub:
                    continue
                axins.plot(
                    sub.loc[mask, config_col],
                    sub.loc[mask, rc],
                    color=color,
                    linewidth=member_linewidth,
                    alpha=member_alpha,
                )

        axins.plot(
            sub.loc[mask, config_col],
            sub.loc[mask, med_col_l],
            color=color,
            linewidth=line_width,
            marker=("o" if show_median_markers else None),
            markersize=(6 if show_median_markers else 0),
            markerfacecolor=("white" if show_median_markers else None),
            markeredgewidth=(1.5 if show_median_markers else 0),
        )

    # Connector lines
    con_2_to_3 = ConnectionPatch(
        xyA=(x0_r, y1_r),
        xyB=(0, 0),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax_left,
        axesB=axins,
        color="black",
        lw=1.0,
    )
    con_1_to_4 = ConnectionPatch(
        xyA=(x1_r, y1_r),
        xyB=(1, 0),
        coordsA="data",
        coordsB="axes fraction",
        axesA=ax_left,
        axesB=axins,
        color="black",
        lw=1.0,
    )
    ax_left.add_artist(con_2_to_3)
    ax_left.add_artist(con_1_to_4)

    # x-axis labels as ints
    def _fmt_int(x: float, _pos: int) -> str:
        return str(int(x))

    for ax in axs:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_int))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    return fig, axs
