from collections.abc import Sequence
from typing import Any
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter


# Line widths & alphas used in this module
MEMBER_LINEWIDTH: float = 0.8  # per-run thin lines
MEMBER_ALPHA: float = 0.25  # per-run transparency
SHADE_ALPHA: float = 0.15  # OOD shading alpha

# Tick density
YTICKS_TARGET: int = 6


def _set_yticks_target(ax: plt.Axes) -> None:
    """Apply a consistent y-tick density for both linear and log scales."""
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_locator(LogLocator(numticks=YTICKS_TARGET))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=YTICKS_TARGET))


def plot_model_comparison(
    groups: Sequence[Sequence[dict[str, Any]]],
    labels: Sequence[Sequence[str]] | None = None,
    metric: str = "epistemic",
    column_titles: Sequence[str] | None = None,
    ood_bounds: Sequence[Any] | None = None,
    x_min: float | Sequence[float] | None = None,
    x_max: float | Sequence[float] | None = None,
    y_limits: Sequence[tuple[float, float] | None] | None = None,
    y_logscale: bool | Sequence[bool | dict[str, bool]] = False,
    show_runs: bool = True,
    figsize: tuple[float, float] | None = None,
    model_colors: Sequence[Sequence[Any]] | None = None,
    share_yaxis: bool = False,
) -> tuple[Figure, Legend]:
    """Visualize epistemic or aleatoric uncertainty for four groups of models.

    Each of the four columns represents a model family or training condition.
    Within each column, multiple models are plotted over the same 1D input grid.

    Args:
        groups:
            Four groups of model result dictionaries. Each dictionary must contain:
            - "X_test": array-like, shape (n,) or (n, 1)
            - uncertainty arrays ("epistemic_all" or "aleatoric_all")
            - optionally "aleatoric_true" for ground-truth aleatoric curves
        labels:
            Per-column labels for the models. If None, defaults to "Model i".
        metric:
            "epistemic" or "aleatoric".
        column_titles:
            Optional titles for the four columns.
        ood_bounds:
            Per-column OOD intervals such as [(a, b), (c, d), ...].
        x_min, x_max:
            Per-column or global x-limits.
        y_limits:
            Per-column y-limits as (ymin, ymax).
        y_logscale:
            Per-column or global logscale specification.
        show_runs:
            Whether to show individual member trajectories.
        figsize:
            Optional figure size; if None, uses the style defaults.
        model_colors:
            Optional per-column color specifications.
        share_yaxis:
            If True, share the y-axis across all columns.

    Returns:
        The created figure and legend.
    """
    metric_lower = metric.lower()
    if metric_lower not in {"epistemic", "aleatoric"}:
        raise ValueError("`metric` must be 'epistemic' or 'aleatoric'.")

    metric_cfg = {
        "epistemic": {
            "key": "epistemic_all",
            "ylabel": r"$\sigma_{\text{epis}}^2(x^{(m)})$",
            "title": "Epistemic Uncertainty",
        },
        "aleatoric": {
            "key": "aleatoric_all",
            "ylabel": r"$\sigma_{\text{alea}}^2(x^{(m)})$",
            "title": "Aleatoric Uncertainty",
        },
    }
    metric_key = metric_cfg[metric_lower]["key"]
    ylabel_text = metric_cfg[metric_lower]["ylabel"]
    default_title_text = metric_cfg[metric_lower]["title"]
    show_ground_truth = metric_lower == "aleatoric"

    if len(groups) != 4:
        raise ValueError("`groups` must contain exactly four columns.")
    groups_list = [list(g) for g in groups]
    for i, g in enumerate(groups_list):
        if len(g) == 0:
            raise ValueError(f"Column {i + 1} contains no result dictionaries.")

    def _as_four(value: Any, default: Any | None = None) -> list[Any | None]:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return list(value)
        if value is None:
            return [default] * 4
        return [value] * 4

    labels_list = _as_four(labels)
    ood_bounds_list = _as_four(ood_bounds)
    y_limits_list = _as_four(y_limits)
    y_logscale_list = _as_four(y_logscale, False)
    model_colors_list = _as_four(model_colors)
    x_min_list = _as_four(x_min)
    x_max_list = _as_four(x_max)

    if figsize is not None:
        fig, axs = plt.subplots(1, 4, sharey=share_yaxis, figsize=figsize)
    else:
        fig, axs = plt.subplots(1, 4, sharey=share_yaxis)

    axs = np.array(axs).ravel()
    collected_legends: list[tuple[list[Any], list[str]]] = []

    for j in range(4):
        results = groups_list[j]
        labels_j = (
            labels_list[j]
            if labels_list[j] is not None
            else [f"Model {i + 1}" for i in range(len(results))]
        )
        if len(labels_j) != len(results):
            raise ValueError(f"Column {j + 1}: label count does not match results.")

        if model_colors_list[j] is None:
            base = plt.cm.tab10.colors
            model_cols = [base[i % len(base)] for i in range(len(results))]
        else:
            colspec = model_colors_list[j]
            if isinstance(colspec, (str, tuple)):
                model_cols = [colspec] * len(results)
            else:
                model_cols = list(
                    itertools.islice(itertools.cycle(colspec), len(results))
                )

        x_test = np.asarray(results[0]["X_test"]).ravel()
        x_min_j = x_min_list[j]
        x_max_j = x_max_list[j]
        mask = (
            (x_test >= x_min_j) & (x_test <= x_max_j)
            if x_min_j is not None and x_max_j is not None
            else slice(None)
        )
        x_plot = x_test[mask]
        ax = axs[j]

        legend_patches: list[Any] = []
        legend_labels: list[str] = []
        gt_handle: Line2D | None = None

        if show_ground_truth:
            try:
                gt = np.asarray(results[0]["aleatoric_true"])[mask]
                gt_col = plt.cm.tab10.colors[1]
                ax.plot(x_plot, gt, color=gt_col, linestyle="--", zorder=10)
                gt_handle = Line2D(
                    [0], [0], color=gt_col, linestyle="--", label="Ground Truth"
                )
            except KeyError:
                pass

        proxy_models: list[Line2D] = []
        for res, lab, col in zip(results, labels_j, model_cols, strict=True):
            arr_all = np.asarray(res[metric_key])[:, mask]
            med = np.median(arr_all, axis=0)
            dist = np.mean(np.abs(arr_all - med), axis=1)
            median_curve = arr_all[np.argmin(dist)]

            if show_runs:
                for arr in arr_all:
                    ax.plot(
                        x_plot,
                        arr,
                        color=col,
                        alpha=MEMBER_ALPHA,
                        linewidth=MEMBER_LINEWIDTH,
                    )

            ax.plot(x_plot, median_curve, color=col)
            proxy_models.append(Line2D([0], [0], color=col, label=lab))

        ood_j = ood_bounds_list[j]
        if ood_j is not None:
            if (
                isinstance(ood_j, (list, tuple))
                and len(ood_j) > 0
                and isinstance(ood_j[0], (list, tuple))
            ):
                intervals = list(ood_j)
            elif isinstance(ood_j, (list, tuple)) and len(ood_j) == 2:
                intervals = [tuple(ood_j)]
            else:
                raise ValueError(
                    "`ood_bounds` must be a (left, right) tuple or a list of such tuples."
                )

            xg_min = float(np.min(x_plot))
            xg_max = float(np.max(x_plot))

            def _clip(a: float | None, b: float | None) -> tuple[float, float] | None:
                a = xg_min if a is None or np.isneginf(a) else float(a)
                b = xg_max if b is None or np.isposinf(b) else float(b)
                left, right = max(xg_min, a), min(xg_max, b)
                return (left, right) if left < right else None

            if metric_lower == "epistemic":
                if len(intervals) == 2:
                    interval_colors = ["gray", "gray"]
                    interval_labels = ["OOD", "OOD region"]
                elif len(intervals) == 3:
                    interval_colors = ["gray", plt.cm.tab10(3), "gray"]
                    interval_labels = ["Outer OOD", "In-between OOD", "Outer OOD"]
                else:
                    interval_colors = ["gray"] * len(intervals)
                    interval_labels = ["OOD"] * len(intervals)
            else:
                interval_colors = ["gray"] * len(intervals)
                interval_labels = ["OOD"] * len(intervals)

            for i, (a, b) in enumerate(intervals):
                clipped = _clip(a, b)
                if clipped is None:
                    continue
                l, r = clipped
                patch = ax.axvspan(l, r, color=interval_colors[i], alpha=SHADE_ALPHA)
                if interval_labels[i] not in legend_labels:
                    legend_patches.append(patch)
                    legend_labels.append(interval_labels[i])

        ax.set_xlim(
            x_min_j if x_min_j is not None else float(np.min(x_plot)),
            x_max_j if x_max_j is not None else float(np.max(x_plot)),
        )
        ax.set_xlabel(r"$x^{(m)}$")
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])

        if j == 0 or not share_yaxis:
            ax.set_ylabel(ylabel_text)

        if column_titles and len(column_titles) == 4 and column_titles[j]:
            ax.set_title(column_titles[j])
        else:
            ax.set_title(default_title_text)

        sci = ScalarFormatter(useMathText=True)
        sci.set_scientific(True)
        sci.set_powerlimits((0, 0))
        sci.set_useOffset(False)
        ax.yaxis.set_major_formatter(sci)

        yl = y_limits_list[j]
        if yl is not None:
            ax.set_ylim(*yl)

        ylog_j = y_logscale_list[j]
        apply_log = (
            isinstance(ylog_j, dict) and ylog_j.get(metric_lower, False)
            if isinstance(ylog_j, dict)
            else bool(ylog_j)
        )
        if apply_log:
            ax.set_yscale("log")

        _set_yticks_target(ax)

        handles = ([gt_handle] if gt_handle else []) + proxy_models
        labels_all = (["Ground Truth"] if gt_handle else []) + list(labels_j)
        if legend_patches:
            handles += legend_patches
            labels_all += legend_labels
        collected_legends.append((handles, labels_all))

    handles, labels_all = collected_legends[-1]
    leg = fig.legend(
        handles,
        labels_all,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.subplots_adjust(wspace=0.1)
    plt.show()

    return fig, leg
