from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
)

from src.data_sampler import DataSampler


def sample_and_plot_true(
    n_points: int,
    train_repeats: int,
    seed: int,
    f_true: Callable[[np.ndarray], np.ndarray],
    sigma_true: Callable[[np.ndarray], np.ndarray],
    ood_bounds: tuple[float, float] | list[list[float]],
    figsize: tuple[float, float],
) -> tuple[Figure, np.ndarray]:
    """Sample synthetic data and plot true function and variance.

    The figure has two panels:

      - Left:  true function f(x), 95% CI band (±2σ(x)), sampled train data.
      - Right: true aleatoric variance σ²(x).

    Args:
        n_points: Number of training points to draw.
        train_repeats: Number of repeats per training location.
        seed: Random seed for the sampler.
        f_true: Ground-truth mean function f(x), input shape (N, 1).
        sigma_true: Ground-truth std function σ(x), input shape (N, 1).
        ood_bounds: In-distribution interval(s):
            - single tuple (lo, hi), or
            - list of [lo, hi] intervals for multiple ID regions.
        figsize: Figure size in inches.

    Returns:
        Tuple (fig, axs) where:
            - fig: Matplotlib Figure.
            - axs: array of axes [left, right].
    """
    # Internal defaults (were previously parameters)
    x_range = (-6.0, 6.0)
    ci_alpha = 0.15
    shade_alpha = 0.15

    lo, hi = float(x_range[0]), float(x_range[1])

    # -------------------------------------------------------------------------
    # ID / OOD helpers
    # -------------------------------------------------------------------------
    def _normalize_id_intervals(
        bounds: tuple[float, float] | list[list[float]],
    ) -> list[tuple[float, float]]:
        """Normalize ID bounds into a sorted, merged list of (a, b) intervals."""
        if isinstance(bounds[0], (float, int)):  # single interval
            id_int = [(float(bounds[0]), float(bounds[1]))]
        else:  # list of intervals
            id_int = [tuple(map(float, ab)) for ab in bounds]

        id_int.sort(key=lambda t: t[0])
        merged: list[list[float]] = []
        for a, b in id_int:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        return [(a, b) for a, b in merged]

    def _ood_segments(
        id_intervals: list[tuple[float, float]],
        xlo: float,
        xhi: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Split domain into OOD 'outside' and 'in-between' segments."""
        outside: list[tuple[float, float]] = []
        between: list[tuple[float, float]] = []

        if not id_intervals:
            return [(xlo, xhi)], []

        # Left outside
        if id_intervals[0][0] > xlo:
            outside.append((xlo, min(id_intervals[0][0], xhi)))

        # Gaps between ID intervals
        for (a1, b1), (a2, b2) in zip(id_intervals, id_intervals[1:]):
            gap_left = max(b1, xlo)
            gap_right = min(a2, xhi)
            if gap_right > gap_left:
                between.append((gap_left, gap_right))

        # Right outside
        if id_intervals[-1][1] < xhi:
            outside.append((max(id_intervals[-1][1], xlo), xhi))

        return outside, between

    id_intervals = _normalize_id_intervals(ood_bounds)
    if len(id_intervals) == 1:
        train_interval: list[float] | list[list[float]] = [
            id_intervals[0][0],
            id_intervals[0][1],
        ]
    else:
        train_interval = [list(ab) for ab in id_intervals]

    ood_outside, ood_between = _ood_segments(id_intervals, lo, hi)

    # -------------------------------------------------------------------------
    # 1) Sample training data
    # -------------------------------------------------------------------------
    FUNCTIONS = {"f_true": lambda X: f_true(X)}
    SIGMAS = {"sigma_true": lambda X: sigma_true(X)}

    job = {
        "seed": int(seed),
        "function": ["f_true", "sigma_true"],
        "train_interval": train_interval,
        "train_instances": int(n_points),
        "train_repeats": int(train_repeats),
        "test_interval": [lo, hi],
        "test_grid_length": 1000,
    }

    sampler = DataSampler(job=job, functions=FUNCTIONS, sigmas=SIGMAS)
    train = sampler.sample_train_data()

    X_train = train["X"]
    y_train = train["y"]

    x = X_train.reshape(-1, 1)
    y = y_train.reshape(-1, 1)

    # -------------------------------------------------------------------------
    # 2) Dense grid for true f and σ
    # -------------------------------------------------------------------------
    xg = np.linspace(lo, hi, 1000).reshape(-1, 1)
    fg = f_true(xg).ravel()
    sg = sigma_true(xg).ravel()
    xg = xg.ravel()

    # Preserve original clipping to [0, 1]
    y = np.clip(y, 0.0, 1.0)
    fg = np.clip(fg, 0.0, 1.0)
    ci_lo = np.clip(fg - 2.0 * sg, 0.0, 1.0)
    ci_hi = np.clip(fg + 2.0 * sg, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # 3) Plot
    # -------------------------------------------------------------------------
    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]
    f_color = tab10[1]
    var_color = tab10[3]
    between_color = tab10[3]
    outside_color = "gray"

    line_width = float(plt.rcParams.get("lines.linewidth", 2.5))

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    def shade_segments(ax: plt.Axes, segments, color: str) -> None:
        """Shade a list of (left, right) segments within current x-limits."""
        if not segments:
            return
        x_min, x_max = ax.get_xlim()
        for left, right in segments:
            l = max(left, x_min)
            r = min(right, x_max)
            if r > l:
                ax.axvspan(l, r, color=color, alpha=shade_alpha)

    # Left: true function + CI + data
    ax_left = axs[0]
    ax_left.set_xlim(lo, hi)
    shade_segments(ax_left, ood_outside, outside_color)
    shade_segments(ax_left, ood_between, between_color)

    ax_left.fill_between(xg, ci_lo, ci_hi, color=f_color, alpha=ci_alpha)
    ax_left.plot(xg, fg, color=f_color, linewidth=line_width)
    ax_left.scatter(x, y, s=14, alpha=0.6, color=data_color)

    ax_left.set_title(r"True function $f^\ast(x^{(m)})$ with 95% CI")
    ax_left.set_xlabel(r"$x^{(m)}$")
    ax_left.set_ylabel(r"$y^{(m)}$")
    ax_left.grid(True)

    legend_handles_left: list[Line2D | Patch] = [
        Line2D([], [], color=data_color, marker="o", lw=0, label="Train data"),
        Line2D(
            [],
            [],
            color=f_color,
            lw=line_width,
            label=r"True function $f^\ast(x^{(m)})$",
        ),
        Line2D(
            [],
            [],
            color=f_color,
            lw=6,
            alpha=ci_alpha,
            label=r"95% CI ($\pm 2\sigma(x^{(m)})$)",
        ),
    ]
    if ood_between:
        legend_handles_left.insert(
            0, Patch(color=between_color, alpha=shade_alpha, label="In-between OOD")
        )
    if ood_outside:
        legend_handles_left.insert(
            0, Patch(color=outside_color, alpha=shade_alpha, label="OOD")
        )

    ax_left.legend(
        handles=legend_handles_left,
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
    )

    # Right: variance
    ax_right = axs[1]
    ax_right.set_xlim(lo, hi)
    shade_segments(ax_right, ood_outside, outside_color)
    shade_segments(ax_right, ood_between, between_color)

    ax_right.plot(xg, sg**2, color=var_color, linewidth=line_width)
    ax_right.set_title(r"True variance $\sigma_{\text{alea}}^2(x^{(m)})$")
    ax_right.set_xlabel(r"$x^{(m)}$")
    ax_right.set_ylabel(r"$\sigma_{\text{alea}}^2(x^{(m)})$")
    ax_right.grid(True)

    legend_handles_right: list[Line2D | Patch] = [
        Line2D(
            [],
            [],
            color=var_color,
            lw=line_width,
            label=r"True variance $\sigma_{\text{alea}}^2(x^{(m)})$",
        ),
    ]
    if ood_between:
        legend_handles_right.insert(
            0, Patch(color=between_color, alpha=shade_alpha, label="In-between OOD")
        )
    if ood_outside:
        legend_handles_right.insert(
            0, Patch(color=outside_color, alpha=shade_alpha, label="OOD")
        )

    ax_right.legend(
        handles=legend_handles_right,
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
    )

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_true_variance_grid(
    configs: list[dict[str, Any]],
    figsize: tuple[float, float],
    seed: int,
) -> tuple[Figure, Legend]:
    """Plot a 2×N grid of true functions and variances for multiple configs.

    Top row:    f*(x) with 95% CI and train data.
    Bottom row: σ_alea²(x).

    Each entry in ``configs`` describes one experiment.

    Args:
        configs: List of configs, each dict with keys:
            - "f_true": callable, f(x)
            - "sigma_true": callable, σ(x)
            - optional "n_points": int
            - optional "train_repeats": int
            - optional "ood_bounds": (a, b) or [(a1, b1), ...]
        figsize: Figure size in inches.
        seed: Base random seed; per column uses ``seed + i``.

    Returns:
        (fig, legend): Figure and shared legend at the bottom.
    """
    x_range = (-6.0, 6.0)
    lo, hi = float(x_range[0]), float(x_range[1])
    ci_alpha = 0.15
    shade_alpha = 0.15
    mean_lw = float(plt.rcParams.get("lines.linewidth", 2.5))

    n_cols = len(configs)
    fig, axs = plt.subplots(
        2,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey="row",
    )

    if n_cols == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]
    f_color = tab10[1]
    var_color = tab10[3]
    between_color = tab10[3]
    outside_color = "gray"

    def _normalize_id_intervals(bounds) -> list[tuple[float, float]]:
        if isinstance(bounds[0], (float, int)):
            id_int = [(float(bounds[0]), float(bounds[1]))]
        else:
            id_int = [tuple(map(float, ab)) for ab in bounds]
        id_int.sort(key=lambda t: t[0])
        merged: list[list[float]] = []
        for a, b in id_int:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)
        return [(a, b) for a, b in merged]

    def _ood_segments(
        id_intervals: list[tuple[float, float]],
        xlo: float,
        xhi: float,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        outside: list[tuple[float, float]] = []
        between: list[tuple[float, float]] = []
        if not id_intervals:
            return [(xlo, xhi)], []
        if id_intervals[0][0] > xlo:
            outside.append((xlo, min(id_intervals[0][0], xhi)))
        for (a1, b1), (a2, b2) in zip(id_intervals, id_intervals[1:]):
            gap_left, gap_right = max(b1, xlo), min(a2, xhi)
            if gap_right > gap_left:
                between.append((gap_left, gap_right))
        if id_intervals[-1][1] < xhi:
            outside.append((max(id_intervals[-1][1], xlo), xhi))
        return outside, between

    for i, cfg in enumerate(configs):
        f_true = cfg["f_true"]
        sigma_true = cfg["sigma_true"]
        n_points = int(cfg.get("n_points", 200))
        train_repeats = int(cfg.get("train_repeats", 0))
        ood_bounds = cfg.get("ood_bounds", (-4.0, 4.0))

        ax_top, ax_bot = axs[0, i], axs[1, i]

        id_intervals = _normalize_id_intervals(ood_bounds)
        ood_outside, ood_between = _ood_segments(id_intervals, lo, hi)

        FUNCTIONS = {"f_true": lambda X, f=f_true: f(X)}
        SIGMAS = {"sigma_true": lambda X, s=sigma_true: s(X)}

        if len(id_intervals) == 1:
            train_interval: list[float] | list[list[float]] = [
                id_intervals[0][0],
                id_intervals[0][1],
            ]
        else:
            train_interval = [list(ab) for ab in id_intervals]

        job = {
            "seed": seed + i,
            "function": ["f_true", "sigma_true"],
            "train_interval": train_interval,
            "train_instances": n_points,
            "train_repeats": train_repeats,
            "test_interval": [lo, hi],
            "test_grid_length": 1000,
        }

        sampler = DataSampler(job=job, functions=FUNCTIONS, sigmas=SIGMAS)
        train = sampler.sample_train_data()
        X_train = train["X"].reshape(-1, 1)
        y_train = train["y"].reshape(-1, 1)

        xg = np.linspace(lo, hi, 1000).reshape(-1, 1)
        fg = f_true(xg).ravel()
        sg = sigma_true(xg).ravel()
        var = sg**2

        ci_lo = fg - 2 * sg
        ci_hi = fg + 2 * sg

        # Top row: f*(x) + CI + train
        ax_top.set_xlim(lo, hi)
        for l, r in ood_outside:
            ax_top.axvspan(l, r, color=outside_color, alpha=shade_alpha)
        for l, r in ood_between:
            ax_top.axvspan(l, r, color=between_color, alpha=shade_alpha)

        ax_top.fill_between(xg.ravel(), ci_lo, ci_hi, color=f_color, alpha=ci_alpha)
        ax_top.plot(xg, fg, color=f_color, lw=mean_lw)
        ax_top.scatter(X_train, y_train, s=14, alpha=0.6, color=data_color)

        ax_top.set_title(rf"Experiment $E_{{{i + 1}}}$")
        if i == 0:
            ax_top.set_ylabel(r"$f^\ast(x^{(m)})$")
        ax_top.grid(True)

        # Bottom row: variance
        ax_bot.set_xlim(lo, hi)
        for l, r in ood_outside:
            ax_bot.axvspan(l, r, color=outside_color, alpha=shade_alpha)
        for l, r in ood_between:
            ax_bot.axvspan(l, r, color=between_color, alpha=shade_alpha)

        ax_bot.plot(xg, var, color=var_color, lw=mean_lw)
        if i == 0:
            ax_bot.set_ylabel(r"$\sigma_{\text{alea}}^2(x^{(m)})$")
        ax_bot.set_xlabel(r"$x^{(m)}$")
        ax_bot.grid(True)

    legend_elements = [
        Patch(color=outside_color, alpha=shade_alpha, label="OOD"),
        Patch(color=between_color, alpha=shade_alpha, label="In-between OOD"),
        Line2D(
            [], [], color=f_color, lw=mean_lw, label=r"True function $f^\ast(x^{(m)})$"
        ),
        Patch(
            color=f_color,
            alpha=ci_alpha,
            label=r"95% CI ($\pm 2\sigma_{\text{alea}}(x^{(m)})$)",
        ),
        Line2D(
            [], [], color=data_color, marker="o", lw=0, markersize=6, label="Train data"
        ),
        Line2D(
            [],
            [],
            color=var_color,
            lw=mean_lw,
            label=r"True variance $\sigma_{\text{alea}}^2(x^{(m)})$",
        ),
    ]

    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.show()

    return fig, legend


def plot_dgp_block(
    func_pairs: list[
        tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
    ],
    col_titles: list[str] | None = None,
    y_lim_true: tuple[float, float] | list[float] | None = None,
    y_lim_var: tuple[float, float] | list[float] | None = None,
    title: str = "Data-Generating Process",
    axs: np.ndarray | None = None,
    show: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Plot a 2×6 block of DGPs (true functions and variances).

    For each (f_true, sigma_true) pair this function draws:

      - Top row: true function f(x) with 95% CI and sampled train data.
      - Bottom row: variance σ²(x).

    Args:
        func_pairs: List of six (f_true, sigma_true) callables.
        col_titles: Optional list of six column titles.
        y_lim_true: Optional y-limits for all top-row panels.
        y_lim_var: Optional y-limits for all bottom-row panels.
        title: Block title.
        axs: Optional existing 2×6 axes array. If None, a new figure is created.
        show: If True and axes are created here, calls plt.show() at the end.

    Returns:
        (fig, axs): Figure and 2×6 axes array.
    """
    # Internal defaults
    x_range = (-6.0, 6.0)
    id_interval = (-4.0, 4.0)
    n_points = 200
    train_repeats = 0
    test_grid_length = 1000
    base_seed = 42
    ci_alpha = 0.15
    shade_alpha = 0.15
    clip01 = True

    lo, hi = float(x_range[0]), float(x_range[1])
    a_id, b_id = float(id_interval[0]), float(id_interval[1])
    mean_lw = float(plt.rcParams.get("lines.linewidth", 2.5))

    xg = np.linspace(lo, hi, int(test_grid_length)).reshape(-1, 1)

    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]
    gt_color = tab10[1]
    bottom_color = tab10[3]
    outside_color = "gray"

    created_here = False
    if axs is None:
        fig, axs = plt.subplots(2, 6, figsize=(20.0, 5.0), sharex=True, sharey="row")
        created_here = True
    else:
        fig = plt.gcf()

    if title:
        if created_here:
            fig.suptitle(title, y=0.99, fontsize=14, weight="bold")
        else:
            axs[0, 0].set_title(title, loc="left", y=1.4, fontweight="bold")

    def shade_outside(
        ax: plt.Axes, a: float, b: float, x_min: float, x_max: float
    ) -> None:
        left_start, left_end = x_min, min(a, x_max)
        right_start, right_end = max(b, x_min), x_max
        if left_end > left_start:
            ax.axvspan(left_start, left_end, color=outside_color, alpha=shade_alpha)
        if right_end > right_start:
            ax.axvspan(right_start, right_end, color=outside_color, alpha=shade_alpha)

    def render_pair(
        ax_top: plt.Axes,
        ax_bottom: plt.Axes,
        f_true: Callable[[np.ndarray], np.ndarray],
        sigma_true: Callable[[np.ndarray], np.ndarray],
        col_seed: int,
    ) -> None:
        FUNCTIONS = {"f_true": lambda X, f=f_true: f(X)}
        SIGMAS = {"sigma_true": lambda X, s=sigma_true: s(X)}
        job = {
            "seed": int(col_seed),
            "function": ["f_true", "sigma_true"],
            "train_interval": [a_id, b_id],
            "train_instances": int(n_points),
            "train_repeats": int(train_repeats),
            "test_interval": [lo, hi],
            "test_grid_length": int(test_grid_length),
        }
        sampler = DataSampler(job=job, functions=FUNCTIONS, sigmas=SIGMAS)
        train = sampler.sample_train_data()
        X_train = train["X"]
        y_train = train["y"]

        x = X_train.reshape(-1, 1)
        y = y_train.reshape(-1, 1)

        fg = f_true(xg).ravel()
        sg = sigma_true(xg).ravel()

        if clip01:
            y = np.clip(y, 0.0, 1.0)
            fg = np.clip(fg, 0.0, 1.0)
            ci_lo = np.clip(fg - 2.0 * sg, 0.0, 1.0)
            ci_hi = np.clip(fg + 2.0 * sg, 0.0, 1.0)
        else:
            ci_lo = fg - 2.0 * sg
            ci_hi = fg + 2.0 * sg

        # Top: true function
        ax_top.set_xlim(lo, hi)
        ax_top.set_xticks([-4.0, 0.0, 4.0])
        shade_outside(ax_top, a_id, b_id, lo, hi)
        ax_top.fill_between(xg.ravel(), ci_lo, ci_hi, color=gt_color, alpha=ci_alpha)
        ax_top.plot(xg.ravel(), fg, color=gt_color, linewidth=mean_lw)
        ax_top.scatter(x, y, s=12, alpha=0.6, color=data_color)
        ax_top.grid(True)

        # Bottom: variance
        ax_bottom.set_xlim(lo, hi)
        ax_bottom.set_xticks([-4.0, 0.0, 4.0])
        shade_outside(ax_bottom, a_id, b_id, lo, hi)
        ax_bottom.plot(xg.ravel(), sg**2, color=bottom_color, linewidth=mean_lw)
        ax_bottom.grid(True)

    for j, (f_j, s_j) in enumerate(func_pairs):
        render_pair(axs[0, j], axs[1, j], f_j, s_j, base_seed + j)
        axs[0, j].set_title(
            col_titles[j] if col_titles is not None else f"Model {j + 1}"
        )

    axs[0, 0].set_ylabel(r"$y^{(m)}$")
    axs[1, 0].set_ylabel(r"$\sigma_{\text{alea}}^2(x^{(m)})$")
    for j in range(len(func_pairs)):
        axs[1, j].set_xlabel(r"$x^{(m)}$")

    if y_lim_true is not None:
        for j in range(len(func_pairs)):
            axs[0, j].set_ylim(y_lim_true)
    if y_lim_var is not None:
        for j in range(len(func_pairs)):
            axs[1, j].set_ylim(y_lim_var)

    if created_here:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()

    return fig, axs


def create_dgp_figure(
    high_correlation_funcs: list[
        tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
    ],
    low_correlation_funcs: list[
        tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
    ],
    high_corr_titles: list[str],
    low_corr_titles: list[str],
    figsize: tuple[float, float],
) -> tuple[Figure, Legend]:
    """Create a composite figure with two stacked 2×6 DGP blocks.

    Upper block:  high-correlation target functions.
    Lower block:  low- or uncorrelated target functions.

    Args:
        high_correlation_funcs: Six (f_true, sigma_true) pairs for the top block.
        low_correlation_funcs: Six (f_true, sigma_true) pairs for the bottom block.
        high_corr_titles: Column titles for the high-correlation block.
        low_corr_titles: Column titles for the low-correlation block.
        plot_dgp_block: Function that renders a 2×6 DGP block.
        figsize: Figure size in inches.

    Returns:
        (fig, legend): Composite figure and shared legend beneath it.
    """
    fig = plt.figure(figsize=figsize)

    gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.65)
    gs_high = gridspec.GridSpecFromSubplotSpec(
        2, 6, subplot_spec=gs_outer[0], hspace=0.15, wspace=0.1
    )
    gs_low = gridspec.GridSpecFromSubplotSpec(
        2, 6, subplot_spec=gs_outer[1], hspace=0.15, wspace=0.1
    )

    axs_high = np.empty((2, 6), dtype=object)
    axs_low = np.empty((2, 6), dtype=object)

    # High block axes
    for j in range(6):
        if j == 0:
            axs_high[0, j] = fig.add_subplot(gs_high[0, j])
            axs_high[1, j] = fig.add_subplot(gs_high[1, j], sharex=axs_high[0, j])
        else:
            axs_high[0, j] = fig.add_subplot(gs_high[0, j], sharey=axs_high[0, 0])
            axs_high[1, j] = fig.add_subplot(
                gs_high[1, j],
                sharex=axs_high[0, j],
                sharey=axs_high[1, 0],
            )

    # Low block axes
    for j in range(6):
        if j == 0:
            axs_low[0, j] = fig.add_subplot(gs_low[0, j])
            axs_low[1, j] = fig.add_subplot(gs_low[1, j], sharex=axs_low[0, j])
        else:
            axs_low[0, j] = fig.add_subplot(gs_low[0, j], sharey=axs_low[0, 0])
            axs_low[1, j] = fig.add_subplot(
                gs_low[1, j],
                sharex=axs_low[0, j],
                sharey=axs_low[1, 0],
            )

    # Hide redundant y labels, and x labels on top rows
    for row_axs in [axs_high[0], axs_high[1], axs_low[0], axs_low[1]]:
        for ax in row_axs[1:]:
            ax.tick_params(labelleft=False)

    for ax in np.r_[axs_high[0, :], axs_low[0, :]]:
        ax.tick_params(labelbottom=False)

    # Unified y ticks for variance (bottom row)
    var_ticks = [0.0, 5e-4, 1e-3]
    var_labels = ["0", "5e-4", "1e-3"]

    for ax in np.r_[axs_high[1, :], axs_low[1, :]]:
        ax.yaxis.set_major_locator(FixedLocator(var_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(var_labels))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.yaxis.offsetText.set_visible(False)

    # Render both blocks using the provided plot_dgp_block
    plot_dgp_block(
        high_correlation_funcs,
        col_titles=high_corr_titles,
        y_lim_true=[0.5, 1.0],
        y_lim_var=[0.0, 0.001],
        title=r"Similar function shapes ($\mathcal{G}_{\text{multi-target}}^{(1)}$)",
        axs=axs_high,
        show=False,
    )

    plot_dgp_block(
        low_correlation_funcs,
        col_titles=low_corr_titles,
        y_lim_true=[0.5, 1.0],
        y_lim_var=[0.0, 0.001],
        title=r"Distinct function shapes ($\mathcal{G}_{\text{multi-target}}^{(2)}$)",
        axs=axs_low,
        show=False,
    )

    # Shared legend
    tab10 = plt.cm.tab10.colors
    data_color = tab10[0]
    f_color = tab10[1]
    var_color = tab10[3]
    outside_color = "gray"
    shade_alpha = 0.15
    line_width = float(plt.rcParams.get("lines.linewidth", 2.5))

    legend_elements = [
        Patch(color=outside_color, alpha=shade_alpha, label="OOD"),
        Line2D(
            [],
            [],
            color=f_color,
            lw=line_width,
            label=r"True function $f^\ast(x^{(m)})$",
        ),
        Patch(
            color=f_color,
            alpha=0.15,
            label=r"95% CI ($\pm 2\sigma_{\text{alea}}(x^{(m)})$)",
        ),
        Line2D(
            [], [], color=data_color, marker="o", lw=0, markersize=6, label="Train data"
        ),
        Line2D(
            [],
            [],
            color=var_color,
            lw=line_width,
            label=r"True variance $\sigma_{\text{alea}}^2(x^{(m)})$",
        ),
    ]

    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    return fig, legend
