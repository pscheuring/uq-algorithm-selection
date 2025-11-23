import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from src.constants import RESULTS_DIR


def plot_uq_multi(
    *results: dict[str, Any],
    labels: list[str] | None = None,
    ood_bounds: tuple[float, float] | list[tuple[float, float]] | None = (-4, 4),
    figsize: tuple[float, float] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    show_runs: bool = True,
    y_limits: (
        dict[str, tuple[float, float] | None]
        | list[tuple[float, float] | None]
        | tuple[tuple[float, float] | None, ...]
        | None
    ) = None,
    model_colors: str | tuple[float, float, float] | list[Any] | None = None,
    y_logscale: bool | dict[str, bool] = False,
) -> tuple[Figure, Legend]:
    """Plot prediction, aleatoric uncertainty, and epistemic uncertainty for 1D inputs.

    For one or more models evaluated on the same 1D test grid, this function generates:
    - A target prediction panel
    - An aleatoric uncertainty panel
    - An epistemic uncertainty panel

    Ground truth (if provided) is shown with a dashed line. Each model's multiple runs
    (e.g., ensemble members, MC samples) can be optionally visualized as faint curves,
    while one representative member per panel is emphasized.

    Args:
        *results: One or more dictionaries containing:
            - ``"X_test"`` (array-like): x-values, shape (n,) or (n, 1)
            - ``"y_pred_all"`` (array-like): shape (n_members, n)
            - ``"aleatoric_all"`` (array-like): shape (n_members, n)
            - ``"epistemic_all"`` (array-like): shape (n_members, n)
            - ``"y_clean"`` or ``"y_test"`` (array-like): ground truth target
            - Optional: ``"aleatoric_true"`` for ground-truth aleatoric variance
        labels: Optional list of labels for each model. If None, generated automatically.
        ood_bounds: One interval ``(a, b)`` or list of intervals ``[(a1, b1), ...]`` to highlight
            as out-of-distribution regions. Use ``None`` to disable shading.
        figsize: Optional figure size. If None, the global Matplotlib style is used.
        x_min: Optional left x-limit. Defaults to min of the provided data.
        x_max: Optional right x-limit. Defaults to max of the provided data.
        show_runs: If True, all member trajectories are plotted in the background.
        y_limits: Optional per-axis y-limits. Can be:
            - dict with keys {"prediction", "aleatoric", "epistemic"}
            - list or tuple of three limit tuples
        model_colors: Color spec(s) for models. If None, uses tab10 minus the orange slot
            (which is reserved for ground truth).
        y_logscale: If True, all y-axes use log-scale. If a dict is provided, keys
            {"prediction", "aleatoric", "epistemic"} specify per-panel scaling.

    Returns:
        tuple[Figure, Legend]: The created figure and its associated bottom legend.
    """

    # -------------------------------------------------------------------------
    # Labels & colors
    # -------------------------------------------------------------------------
    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(results))]

    # Ground truth color = orange (tab10[1])
    gt_color = plt.cm.tab10.colors[1]

    if model_colors is None:
        base_colors = list(plt.cm.tab10.colors)
        base_colors.pop(1)  # remove orange
        colors = [base_colors[i % len(base_colors)] for i in range(len(results))]
    else:
        if isinstance(model_colors, (str, tuple)):
            colors = [model_colors] * len(results)
        else:
            colors = list(itertools.islice(itertools.cycle(model_colors), len(results)))

    # -------------------------------------------------------------------------
    # Prepare data slice
    # -------------------------------------------------------------------------
    x_all = np.asarray(results[0]["X_test"]).ravel()
    y_clean = np.asarray(results[0].get("y_clean", results[0].get("y_test"))).ravel()

    mask = (
        (x_all >= x_min) & (x_all <= x_max)
        if x_min is not None and x_max is not None
        else slice(None)
    )
    x = x_all[mask]
    y = y_clean[mask]

    # -------------------------------------------------------------------------
    # Create figure & axes
    # -------------------------------------------------------------------------
    subplot_kwargs = {"nrows": 1, "ncols": 3, "sharex": True}
    if figsize is not None:
        subplot_kwargs["figsize"] = figsize

    fig, axs = plt.subplots(**subplot_kwargs)
    ax_pred, ax_alea, ax_epi = axs

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _representative_curve(arr_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return all member curves (masked) and a representative curve."""
        arr = np.asarray(arr_2d)[:, mask]
        median = np.median(arr, axis=0)
        idx = np.argmin(np.mean(np.abs(arr - median), axis=1))
        return arr, arr[idx]

    def _format_axis(ax: plt.Axes) -> None:
        """Apply axis formatting that is not style-dependent."""
        ax.set_xlim(
            x_min if x_min is not None else float(np.min(x)),
            x_max if x_max is not None else float(np.max(x)),
        )
        ax.set_xlabel(r"$x^{(m)}$")

    # -------------------------------------------------------------------------
    # Ground truth
    # -------------------------------------------------------------------------
    ax_pred.plot(x, y, color=gt_color, linestyle="--", label="Ground Truth", zorder=10)

    if "aleatoric_true" in results[0]:
        ax_alea.plot(
            x,
            np.asarray(results[0]["aleatoric_true"])[mask],
            color=gt_color,
            linestyle="--",
            label="Ground Truth",
            zorder=10,
        )

    # -------------------------------------------------------------------------
    # Model predictions
    # -------------------------------------------------------------------------
    n_models = len(results)
    preferred = 1 if n_models > 1 else 0
    draw_order = [preferred] + [i for i in range(n_models) if i != preferred]

    member_alpha = 0.25
    member_linewidth = 0.8

    for idx in draw_order:
        res = results[idx]
        label = labels[idx]
        color = colors[idx]

        # Prediction
        pred_all, pred_rep = _representative_curve(res["y_pred_all"])
        if show_runs:
            for arr in pred_all:
                ax_pred.plot(
                    x, arr, color=color, alpha=member_alpha, linewidth=member_linewidth
                )
        ax_pred.plot(x, pred_rep, color=color, label=label)

        # Aleatoric
        alea_all, alea_rep = _representative_curve(res["aleatoric_all"])
        if show_runs:
            for arr in alea_all:
                ax_alea.plot(
                    x, arr, color=color, alpha=member_alpha, linewidth=member_linewidth
                )
        ax_alea.plot(x, alea_rep, color=color)

        # Epistemic
        epi_all, epi_rep = _representative_curve(res["epistemic_all"])
        if show_runs:
            for arr in epi_all:
                ax_epi.plot(
                    x, arr, color=color, alpha=member_alpha, linewidth=member_linewidth
                )
        ax_epi.plot(x, epi_rep, color=color)

    # -------------------------------------------------------------------------
    # Axis cosmetics
    # -------------------------------------------------------------------------
    for ax in axs:
        _format_axis(ax)

    ax_pred.set_title("Target Prediction")
    ax_alea.set_title("Aleatoric Uncertainty")
    ax_epi.set_title("Epistemic Uncertainty")

    ax_pred.set_ylabel(r"$y^{(m)}$")
    ax_alea.set_ylabel(r"$\sigma_{\text{alea}}^2(x^{(m)})$")
    ax_epi.set_ylabel(r"$\sigma_{\text{epis}}^2(x^{(m)})$")

    sci = ScalarFormatter(useMathText=True)
    sci.set_scientific(True)
    sci.set_powerlimits((0, 0))
    sci.set_useOffset(False)
    ax_alea.yaxis.set_major_formatter(sci)
    ax_epi.yaxis.set_major_formatter(sci)

    # -------------------------------------------------------------------------
    # OOD shading
    # -------------------------------------------------------------------------
    legend_patches: list[Any] = []
    legend_labels: list[str] = []
    shade_alpha = 0.15

    if ood_bounds is not None:
        if (
            isinstance(ood_bounds, (list, tuple))
            and len(ood_bounds) > 0
            and isinstance(ood_bounds[0], (list, tuple))
        ):
            intervals = [tuple(b) for b in ood_bounds]
        else:
            intervals = [tuple(ood_bounds)]

        xg_min, xg_max = float(np.min(x)), float(np.max(x))

        def _clip_interval(
            a: float | None, b: float | None
        ) -> tuple[float, float] | None:
            left = xg_min if a is None or np.isneginf(a) else float(a)
            right = xg_max if b is None or np.isposinf(b) else float(b)
            left, right = max(xg_min, left), min(xg_max, right)
            return (left, right) if left < right else None

        if len(intervals) == 2:
            interval_colors = ["gray", "gray"]
            interval_labels_list = ["OOD", "OOD"]
        elif len(intervals) == 3:
            interval_colors = ["gray", plt.cm.tab10(3), "gray"]
            interval_labels_list = ["Outer OOD", "In-between OOD", "Outer OOD"]
        else:
            interval_colors = ["gray"] * len(intervals)
            interval_labels_list = ["OOD"] * len(intervals)

        for ax in axs:
            for i, (a, b) in enumerate(intervals):
                clipped = _clip_interval(a, b)
                if clipped is None:
                    continue
                left, right = clipped
                patch = ax.axvspan(
                    left, right, color=interval_colors[i], alpha=shade_alpha
                )
                if interval_labels_list[i] not in legend_labels:
                    legend_patches.append(patch)
                    legend_labels.append(interval_labels_list[i])

    # -------------------------------------------------------------------------
    # y-limits
    # -------------------------------------------------------------------------
    if y_limits is not None:

        def _apply_limits(axis: plt.Axes, lim: tuple[float, float] | None) -> None:
            if lim is not None:
                axis.set_ylim(*lim)

        if isinstance(y_limits, dict):
            _apply_limits(ax_pred, y_limits.get("prediction"))
            _apply_limits(ax_alea, y_limits.get("aleatoric"))
            _apply_limits(ax_epi, y_limits.get("epistemic"))
        else:
            if len(y_limits) == 3:
                _apply_limits(ax_pred, y_limits[0])
                _apply_limits(ax_alea, y_limits[1])
                _apply_limits(ax_epi, y_limits[2])

    # -------------------------------------------------------------------------
    # y-logscale
    # -------------------------------------------------------------------------
    if isinstance(y_logscale, dict):
        if y_logscale.get("prediction"):
            ax_pred.set_yscale("log")
        if y_logscale.get("aleatoric"):
            ax_alea.set_yscale("log")
        if y_logscale.get("epistemic"):
            ax_epi.set_yscale("log")
    elif y_logscale:
        for ax in axs:
            ax.set_yscale("log")

    # -------------------------------------------------------------------------
    # Legend (bottom, single legend)
    # -------------------------------------------------------------------------
    proxy_gt = Line2D([0], [0], color=gt_color, linestyle="--", label="Ground Truth")
    proxy_models = [
        Line2D([0], [0], color=color, label=label)
        for color, label in zip(colors, labels)
    ]

    handles: list[Any] = [proxy_gt] + proxy_models
    labels_all: list[str] = ["Ground Truth"] + list(labels)

    if legend_patches:
        handles.extend(legend_patches)
        labels_all.extend(legend_labels)

    legend = fig.legend(
        handles,
        labels_all,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    return fig, legend


def _squeeze_tail(a):
    """Recursively squeeze trailing singleton dimensions along the last axis."""
    a = np.asarray(a)
    while a.ndim > 0 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def plot_all_outputs_with_plot_uq_multi(
    folder: str | Path,
    x_min: float = -6.0,
    x_max: float = 6.0,
    ood_bounds: tuple[tuple[float, float], ...] | list[tuple[float, float]] = (
        (-np.inf, -4.0),
        (-1.0, 1.0),
        (4.0, np.inf),
    ),
    labels_prefix: str | None = None,
    **plot_kwargs: Any,
) -> None:
    """Load a multi-output results folder and call ``plot_uq_multi`` once per output.

    Expected files in ``RESULTS_DIR / folder`` (shapes in parentheses):

        - ``X_test.npy``         (N,) or (N, 1) or (N, d)  [assumed 1D here]
        - ``y_test.npy``         (N,) or (N, O)
        - ``y_clean.npy``        (N,) or (N, O)
        - ``y_pred_all.npy``     (R, N, O)
        - ``aleatoric_all.npy``  (R, N, O)
        - ``epistemic_all.npy``  (R, N, O)
        - ``aleatoric_true.npy`` (N,) or (N, O)
        - ``config.json``        with metadata, in particular:
            * ``"function"``: list of [f_name, sigma_name] pairs per output
            * ``"model_name"``: optional string used for titles/labels

    Behavior:
        * Squeezes trailing singleton dimensions (via ``_squeeze_tail``).
        * Assumes a 1D input; uses ``X_test`` directly as x-axis.
        * Uses the config (if present) to build titles/labels per output.
        * For each output dimension, constructs a results dict and forwards it
          to ``plot_uq_multi(...)``, including x-limits and OOD bounds.

    Args:
        folder: Directory (relative to ``RESULTS_DIR``) containing the result
            arrays and ``config.json``.
        x_min: Left x-axis limit forwarded to ``plot_uq_multi``.
        x_max: Right x-axis limit forwarded to ``plot_uq_multi``.
        ood_bounds: OOD intervals forwarded to ``plot_uq_multi``, e.g.
            ``[(-np.inf, -4), (-1, 1), (4, np.inf)]``.
        labels_prefix: Optional prefix for legend labels. If None, the
            ``"model_name"`` from the config is used if available.
        **plot_kwargs: Arbitrary keyword arguments forwarded to ``plot_uq_multi``.
    """
    folder = Path(folder)

    # Load arrays
    X_test = np.load(RESULTS_DIR / folder / "X_test.npy")
    y_test = np.load(RESULTS_DIR / folder / "y_test.npy")
    y_clean = np.load(RESULTS_DIR / folder / "y_clean.npy")
    y_pred_all = np.load(RESULTS_DIR / folder / "y_pred_all.npy")
    aleatoric_all = np.load(RESULTS_DIR / folder / "aleatoric_all.npy")
    epistemic_all = np.load(RESULTS_DIR / folder / "epistemic_all.npy")
    aleatoric_true = np.load(RESULTS_DIR / folder / "aleatoric_true.npy")

    # Squeeze trailing singleton dimensions (e.g. (..., 1))
    X_test = _squeeze_tail(X_test)
    y_test = _squeeze_tail(y_test)
    y_clean = _squeeze_tail(y_clean)
    y_pred_all = _squeeze_tail(y_pred_all)
    aleatoric_all = _squeeze_tail(aleatoric_all)
    epistemic_all = _squeeze_tail(epistemic_all)
    if aleatoric_true is not None:
        aleatoric_true = _squeeze_tail(aleatoric_true)

    # Read config for function names and model name
    cfg_path = RESULTS_DIR / folder / "config.json"
    cfg = json.loads(cfg_path.read_text())
    pairs = cfg.get("function", [])  # list of [f_name, sigma_name] per output
    model_name = cfg.get("model_name", "")
    title_prefix = model_name if model_name else "Model"
    if labels_prefix is None:
        labels_prefix = title_prefix

    # Enforce 1D input
    if X_test.ndim != 1:
        raise ValueError("X_test must be 1D for this plotting helper.")
    x_1d = X_test

    # Number of outputs (O) from shape (R, N, O)
    _, _, O = y_pred_all.shape

    # Call plot_uq_multi for each output dimension
    for o in range(O):
        res: dict[str, np.ndarray] = {
            "X_test": x_1d,
            "y_test": (y_test if y_test.ndim == 1 else y_test[:, o]),
            "y_clean": (y_clean if y_clean.ndim == 1 else y_clean[:, o]),
            "y_pred_all": y_pred_all[:, :, o],
            "aleatoric_all": aleatoric_all[:, :, o],
            "epistemic_all": epistemic_all[:, :, o],
        }
        if aleatoric_true is not None:
            res["aleatoric_true"] = (
                aleatoric_true if aleatoric_true.ndim == 1 else aleatoric_true[:, o]
            )

        # Title/label from config when available
        if o < len(pairs):
            f_name, s_name = pairs[o]
            title = f"{f_name}  |  {s_name}  ({title_prefix})"
            label = f"{labels_prefix} — {f_name}"
        else:
            title = f"Output {o + 1}  ({title_prefix})"
            label = f"{labels_prefix} — Output {o + 1}"

        plot_uq_multi(
            res,
            labels=[label],
            x_min=float(x_min),
            x_max=float(x_max),
            ood_bounds=ood_bounds,
            **plot_kwargs,
        )


def _load_folder_for_fn_sigma(
    folder: str | Path,
    function_name: str,
    sigma_name: str,
) -> tuple[dict[str, np.ndarray], str, tuple[float, float]]:
    """Load a folder and extract the output matching a (function, sigma) pair.

    This returns a single-output result dict in the format expected by
    ``plot_uq_multi``, even if the underlying data is multi-output.

    Assumptions:
        * ``X_test`` is effectively 1D (shape (N,) or squeezable to (N,)).
        * For multi-output results, the relevant output index is determined by
          matching the pair ``[function_name, sigma_name]`` in
          ``config.json["function"]``.

    Args:
        folder: Directory (relative to ``RESULTS_DIR``) containing the arrays
            and ``config.json``.
        function_name: Name of the ground truth function to select.
        sigma_name: Name of the aleatoric/noise function to select.

    Returns:
        Tuple ``(res, label, train_interval)`` where:
            res: Result dict for a single output dimension with keys:
                ``"X_test"``, ``"y_test"``, ``"y_clean"``,
                ``"y_pred_all"``, ``"aleatoric_all"``, ``"epistemic_all"``,
                and optionally ``"aleatoric_true"``.
            label: A string label derived from the model name or folder name.
            train_interval: A (low, high) tuple representing the training interval.

    Raises:
        ValueError: If ``X_test`` is not 1D or the (function, sigma) pair
            cannot be found in the config for a multi-output folder.
    """
    folder = Path(folder)

    X_test = np.load(RESULTS_DIR / folder / "X_test.npy")
    y_test = np.load(RESULTS_DIR / folder / "y_test.npy")
    y_clean = np.load(RESULTS_DIR / folder / "y_clean.npy")
    y_pred_all = np.load(RESULTS_DIR / folder / "y_pred_all.npy")
    aleatoric_all = np.load(RESULTS_DIR / folder / "aleatoric_all.npy")
    epistemic_all = np.load(RESULTS_DIR / folder / "epistemic_all.npy")
    alea_true = np.load(RESULTS_DIR / folder / "aleatoric_true.npy")

    # Squeeze trailing singleton dims
    X_test = _squeeze_tail(X_test)
    y_test = _squeeze_tail(y_test)
    y_clean = _squeeze_tail(y_clean)
    y_pred_all = _squeeze_tail(y_pred_all)
    aleatoric_all = _squeeze_tail(aleatoric_all)
    epistemic_all = _squeeze_tail(epistemic_all)
    if alea_true is not None:
        alea_true = _squeeze_tail(alea_true)

    # Enforce 1D input
    if X_test.ndim != 1:
        raise ValueError("X_test must be 1D for this plotting helper.")
    X_1d = X_test

    # Multi- vs single-output
    if y_pred_all.ndim == 3:
        # Multi-output → find matching output index in config.json
        cfg = json.loads((RESULTS_DIR / folder / "config.json").read_text())
        pairs = cfg.get("function", [])
        out_idx: int | None = None

        for i, (f, s) in enumerate(pairs):
            if str(f) == str(function_name) and str(s) == str(sigma_name):
                out_idx = i
                break
        if out_idx is None:
            raise ValueError(f"{function_name} | {sigma_name} not found in {folder}.")

        y_pred_all_o = y_pred_all[:, :, out_idx]
        aleatoric_all_o = aleatoric_all[:, :, out_idx]
        epistemic_all_o = epistemic_all[:, :, out_idx]
        y_test_o = y_test if y_test.ndim == 1 else y_test[:, out_idx]
        y_clean_o = y_clean if y_clean.ndim == 1 else y_clean[:, out_idx]
        alea_true_o = (
            None
            if alea_true is None
            else (alea_true if alea_true.ndim == 1 else alea_true[:, out_idx])
        )
    else:
        # Single-output
        y_pred_all_o = y_pred_all
        aleatoric_all_o = aleatoric_all
        epistemic_all_o = epistemic_all
        y_test_o = y_test
        y_clean_o = y_clean
        alea_true_o = alea_true

    # Label and training interval from config (fallback: folder name / [-4, 4])
    try:
        cfg = json.loads((folder / "config.json").read_text())
        label = str(cfg.get("model_name", "")) or folder.name
        train_interval_raw = cfg.get("train_interval", [-4.0, 4.0])
        train_interval = tuple(map(float, train_interval_raw))
    except Exception:
        label = folder.name
        train_interval = (-4.0, 4.0)

    res: dict[str, np.ndarray] = {
        "X_test": X_1d,
        "y_test": y_test_o.reshape(-1),
        "y_clean": y_clean_o.reshape(-1),
        "y_pred_all": y_pred_all_o,
        "aleatoric_all": aleatoric_all_o,
        "epistemic_all": epistemic_all_o,
    }
    if alea_true_o is not None:
        res["aleatoric_true"] = alea_true_o.reshape(-1)

    return res, label, train_interval


def plot_uq_multi_target(
    folder_multi_similar: str | Path,
    folder_multi_distinct: str | Path,
    folder_single: str | Path,
    function_name: str,
    sigma_name: str,
    x_min: float = -6.0,
    x_max: float = 6.0,
    ood_bounds: tuple[tuple[float, float], ...] | list[tuple[float, float]] = (
        (-np.inf, -4.0),
        (4.0, np.inf),
    ),
    **plot_kwargs: Any,
) -> tuple[Figure, Legend]:
    """Compare a (function, sigma) pair across three folders in one UQ plot.

    Each folder is expected to contain results compatible with ``plot_uq_multi``
    (possibly multi-output). The specific output dimension is chosen by matching
    ``(function_name, sigma_name)`` via ``config.json["function"]`` in the
    multi-output folders.

    The three compared settings are:

    - multi-similar:  ``folder_multi_similar``
    - multi-distinct: ``folder_multi_distinct``
    - single-target:  ``folder_single``

    Args:
        folder_multi_similar: Folder with multi-output (similar functions) results.
        folder_multi_distinct: Folder with multi-output (distinct functions) results.
        folder_single: Folder with single-output results.
        function_name: Name of the ground truth function to select.
        sigma_name: Name of the noise/aleatoric function to select.
        x_min: Left x-axis limit forwarded to ``plot_uq_multi``.
        x_max: Right x-axis limit forwarded to ``plot_uq_multi``.
        ood_bounds: OOD intervals forwarded to ``plot_uq_multi``, e.g.
            ``[(-np.inf, -4), (-1, 1), (4, np.inf)]``.
        **plot_kwargs: Arbitrary keyword arguments forwarded to ``plot_uq_multi``.

    Returns:
        A tuple ``(fig, leg)`` where ``fig`` is the Figure and ``leg`` is the
        shared legend object returned by ``plot_uq_multi``.
    """
    folders: dict[str, Path] = {
        r"$\mathcal{G}_{\text{multi-target}}^{(1)}$ (similar functions)": Path(
            folder_multi_similar
        ),
        r"$\mathcal{G}_{\text{multi-target}}^{(2)}$ (distinct functions)": Path(
            folder_multi_distinct
        ),
        r"$\mathcal{G}_{\text{single}}$ (single target)": Path(folder_single),
    }

    results: list[dict[str, np.ndarray]] = []
    labels: list[str] = []
    _train_intervals: list[tuple[float, float]] = []  # kept for potential debugging

    for label, folder in folders.items():
        res, _, train_interval = _load_folder_for_fn_sigma(
            folder, function_name=function_name, sigma_name=sigma_name
        )
        results.append(res)
        labels.append(label)
        _train_intervals.append(train_interval)

    fig, leg = plot_uq_multi(
        *results,
        labels=labels,
        x_min=float(x_min),
        x_max=float(x_max),
        ood_bounds=ood_bounds,
        **plot_kwargs,
    )

    return fig, leg
