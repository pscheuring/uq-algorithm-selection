from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# Line width for focus-box outlines on 3D surfaces
LINES_LINEWIDTH: float = 2.5


def _select_median_run(
    all_runs: np.ndarray,
    X: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return the run whose mean is closest to the median of run means.

    Runs can be shaped (n_runs, n_points) or (n_points, n_runs), with optional
    singleton dimensions. If X and bounds are provided, only points with
    all features in [bounds[0], bounds[1]] are used to compute the run means.

    Args:
        all_runs: Array of shape (n_runs, n_points) or (n_points, n_runs), possibly
            with extra singleton dimensions.
        X: Optional array of shape (n_points, n_features) with the inputs
            corresponding to the points in each run.
        bounds: Optional (low, high) interval. If given together with X,
            only points whose features lie in [low, high] for all dimensions
            are used when computing run means.

    Returns:
        A 1D array of length n_points representing the selected median run.

    Raises:
        ValueError: If shapes are inconsistent or all run means become NaN.
    """
    arr = np.asarray(all_runs, dtype=float).squeeze()

    if arr.ndim == 1:
        runs_first = arr[None, :]
    else:
        runs_first = arr if arr.shape[0] <= arr.shape[1] else arr.T

    if X is not None and bounds is not None:
        X = np.asarray(X)
        low, high = bounds

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_points, n_features), got shape {X.shape}"
            )

        if X.shape[0] != runs_first.shape[1]:
            raise ValueError(
                f"Mismatch: X has {X.shape[0]} points, but runs have "
                f"{runs_first.shape[1]} points."
            )

        # Only keep points whose features lie within [low, high]
        mask = np.all((X >= low) & (X <= high), axis=1)
        if not np.any(mask):
            raise ValueError(
                "No points in X lie within the specified bounds; "
                "cannot select a median run."
            )

        runs_first = runs_first.copy()
        runs_first[:, ~mask] = np.nan

    runs_first = np.where(np.isfinite(runs_first), runs_first, np.nan)
    run_means = np.nanmean(runs_first, axis=1)

    if np.all(np.isnan(run_means)):
        raise ValueError("All run means are NaN after filtering/non-finite removal.")

    med = np.nanmedian(run_means)
    idx = int(np.nanargmin(np.abs(run_means - med)))
    return runs_first[idx]


def _grid_from_pairs_and_values(
    x_pairs: np.ndarray,
    z_vals: np.ndarray,
    clip_range: tuple[float, float] = (-6.0, 6.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular ij-grid (X, Y, Z) from (x1, x2) pairs and values.

    The function filters to the square clip_range × clip_range, sorts
    lexicographically by (x1, x2), and reshapes to a complete meshgrid.

    Args:
        x_pairs: Array of shape (N, >=2) with at least two columns [x1, x2].
        z_vals: Array of shape (N,) matching ``x_pairs``.
        clip_range: Interval (low, high) to keep only points within
            [low, high] on both axes.

    Returns:
        Tuple of (X, Y, Z) as 2D arrays in ij-layout (rows=x1, cols=x2).

    Raises:
        AssertionError: If the filtered data does not form a complete grid.
    """
    x_pairs = np.asarray(x_pairs, dtype=float)[:, :2]
    z_vals = np.asarray(z_vals, dtype=float).squeeze()

    fmin, fmax = clip_range
    mask = (
        (x_pairs[:, 0] >= fmin)
        & (x_pairs[:, 0] <= fmax)
        & (x_pairs[:, 1] >= fmin)
        & (x_pairs[:, 1] <= fmax)
    )
    x_pairs = x_pairs[mask]
    z_vals = z_vals[mask]

    order = np.lexsort((x_pairs[:, 1], x_pairs[:, 0]))
    x_pairs = x_pairs[order]
    z_vals = z_vals[order]

    x1_u = np.unique(x_pairs[:, 0])
    x2_u = np.unique(x_pairs[:, 1])
    n1, n2 = x1_u.size, x2_u.size
    assert n1 * n2 == x_pairs.shape[0], "Incomplete grid after filtering."

    X = x_pairs[:, 0].reshape(n1, n2)
    Y = x_pairs[:, 1].reshape(n1, n2)
    Z = z_vals.reshape(n1, n2)
    return X, Y, Z


def _nice_125_ticks(
    z_min: float,
    z_max: float,
    n_target: int = 5,
) -> tuple[float, float, np.ndarray]:
    """Compute a 1–2–5-based tick grid starting at zero.

    Args:
        z_min: Minimum data value (unused except for completeness).
        z_max: Maximum data value.
        n_target: Approximate number of ticks desired.

    Returns:
        A tuple (z0, z_top, ticks) where:
            z0: Lower bound (always 0).
            z_top: Upper bound as a multiple of a 1–2–5 step.
            ticks: Tick positions from z0 to z_top.
    """
    z0 = 0.0
    span = max(z_max - z0, 1e-300)
    raw = span / max(n_target - 1, 1)
    exp = np.floor(np.log10(raw))
    base = 10.0**exp

    for m in (1, 2, 5, 10):
        step = m * base
        if step >= raw - 1e-12:
            break

    z_top = step * np.ceil(z_max / step)
    ticks = np.arange(z0, z_top + 0.5 * step, step)
    return z0, z_top, ticks


def _add_focus_box_lines(
    ax: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    bounds: tuple[float, float] = (-4.0, 4.0),
    color: str = "white",
    lw: float = LINES_LINEWIDTH,
) -> None:
    """Draw four 3D lines along the boundary of a focus box on the surface.

    The lines are drawn at x1 = ±bounds and x2 = ±bounds in the (X, Y, Z) grid.

    Args:
        ax: Matplotlib 3D axes to draw on.
        X: 2D array of x1 coordinates.
        Y: 2D array of x2 coordinates.
        Z: 2D array of z values.
        bounds: (low, high) interval for the focus box in both x1 and x2.
        color: Line color.
        lw: Line width.
    """
    fmin, fmax = bounds
    i_min = int(np.argmin(np.abs(X[:, 0] - fmin)))
    i_max = int(np.argmin(np.abs(X[:, 0] - fmax)))
    ax.plot3D(
        X[i_min, :], Y[i_min, :], Z[i_min, :], color=color, linewidth=lw, zorder=20
    )
    ax.plot3D(
        X[i_max, :], Y[i_max, :], Z[i_max, :], color=color, linewidth=lw, zorder=20
    )

    j_min = int(np.argmin(np.abs(Y[0, :] - fmin)))
    j_max = int(np.argmin(np.abs(Y[0, :] - fmax)))
    ax.plot3D(
        X[:, j_min], Y[:, j_min], Z[:, j_min], color=color, linewidth=lw, zorder=20
    )
    ax.plot3D(
        X[:, j_max], Y[:, j_max], Z[:, j_max], color=color, linewidth=lw, zorder=20
    )


def plot_true_vs_pred_2_feat(
    meta: dict[str, Any],
    mode: str = "aleatoric",  # "aleatoric" or "prediction"
    titles: Sequence[str] | None = None,
    bounds: tuple[float, float] = (-4.0, 4.0),
    figsize: tuple[float, float] = (20, 6),
    sensitivity: float = 0.98,
    z_lim_left_mid: tuple[float, float] | None = None,
    z_ticks_left_mid: Sequence[float] | None = None,
    z_lim_right: tuple[float, float] | None = None,
    z_ticks_right: Sequence[float] | None = None,
    show: bool = True,
) -> Figure:
    """Plot 3D surfaces for true, predicted, and absolute error on a 2D feature grid.

    Two modes are supported:

    - ``mode="aleatoric"``:
        Left: true aleatoric variance,
        Middle: predicted aleatoric variance (median run),
        Right: absolute error.
    - ``mode="prediction"``:
        Left: true targets,
        Middle: predicted targets (median run),
        Right: absolute error.

    The left and middle panels share a colormap normalization; the error panel
    uses an independent robust normalization.

    Args:
        meta: Result dictionary containing at least:
            - "X_test": 2D array of test inputs (n_points, n_features >= 2).
            - For mode="aleatoric":
                "aleatoric_true" and "aleatoric_all".
            - For mode="prediction":
                "y_clean" and "y_pred_all".
        mode: Either "aleatoric" or "prediction".
        titles: Optional sequence of three titles for the subplots
            (true, predicted, error). If None, no titles are set.
        bounds: (low, high) interval used for drawing the focus box.
        figsize: Figure size in inches.
        sensitivity: Quantile used for robust colormap normalization.
        z_lim_left_mid: Optional z-limits for the left and middle subplots.
        z_ticks_left_mid: Optional z-ticks for the left and middle subplots.
        z_lim_right: Optional z-limits for the right (error) subplot.
        z_ticks_right: Optional z-ticks for the right subplot.
        show: If True, ``plt.show()`` is called before returning.

    Returns:
        The created Matplotlib Figure.
    """
    X_test = meta["X_test"]

    if mode == "aleatoric":
        z_true_vec = meta["aleatoric_true"]
        z_pred_vec = _select_median_run(
            meta["aleatoric_all"],
            X=X_test,
            bounds=(-6, 6),
        )

        zlabel_left_mid = r"$\sigma_{\mathrm{alea}}^{2}(x^{(m)})$"
        zlabel_right = (
            r"$|\sigma_{\mathrm{alea,true}}^{2}(x)"
            r"-\sigma_{\mathrm{alea}}^{2}(x^{(m)})|$"
        )
        fmt_func = lambda v, _pos: f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        one_sided = True

    elif mode == "prediction":
        z_true_vec = meta["y_clean"]
        z_pred_vec = _select_median_run(
            meta["y_pred_all"],
            X=X_test,
            bounds=(-6, 6),
        )
        zlabel_left_mid = r"$y^{(m)}$"
        zlabel_right = r"$|y^{(m)}-\hat{y}^{(m)}|$"
        fmt_func = lambda v, _pos: f"{v:.2f}"
        one_sided = False

        if z_ticks_left_mid is None:
            z_ticks_left_mid = [0.8, 1.0, 1.2]
        if z_lim_left_mid is None and z_ticks_left_mid is not None:
            z_lim_left_mid = (min(z_ticks_left_mid), max(z_ticks_left_mid))

    else:
        raise ValueError("mode must be 'aleatoric' or 'prediction'.")

    if titles is None:
        titles = (None, None, None)

    X1, Y1, Z1 = _grid_from_pairs_and_values(meta["X_test"], z_true_vec)
    X2, Y2, Z2 = _grid_from_pairs_and_values(meta["X_test"], z_pred_vec)
    Z3 = np.abs(Z1 - Z2)

    cmap = cm.get_cmap("viridis")

    lm_vals = np.concatenate([Z1.ravel(), Z2.ravel()])
    lm_vals = lm_vals[np.isfinite(lm_vals)]
    if one_sided:
        vmin_lm, vmax_lm = 0.0, float(np.quantile(lm_vals, sensitivity))
    else:
        q_low, q_high = max(0.0, 1.0 - float(sensitivity)), float(sensitivity)
        vmin_lm = float(np.quantile(lm_vals, q_low))
        vmax_lm = float(np.quantile(lm_vals, q_high))
        if vmin_lm == vmax_lm:
            vmax_lm = vmin_lm + 1.0
    norm_lm = mpl.colors.Normalize(vmin=vmin_lm, vmax=vmax_lm)

    r_vals = Z3.ravel()
    r_vals = r_vals[np.isfinite(r_vals)]
    if one_sided:
        vmin_r, vmax_r = 0.0, float(np.quantile(r_vals, sensitivity))
    else:
        q_low_r, q_high_r = max(0.0, 1.0 - float(sensitivity)), float(sensitivity)
        vmin_r = float(np.quantile(r_vals, q_low_r))
        vmax_r = float(np.quantile(r_vals, q_high_r))
        if vmin_r == vmax_r:
            vmax_r = vmin_r + 1.0
    norm_r = mpl.colors.Normalize(vmin=vmin_r, vmax=vmax_r)

    if mode == "aleatoric" and z_lim_left_mid is None and z_ticks_left_mid is None:
        z0, z_top, z_ticks = _nice_125_ticks(
            0.0, max(np.nanmax(Z1), np.nanmax(Z2)), n_target=5
        )
        z_lim_left_mid = (z0, z_top)
        z_ticks_left_mid = list(z_ticks)

    fig = plt.figure(figsize=figsize)
    datasets = [(X1, Y1, Z1, norm_lm), (X2, Y2, Z2, norm_lm), (X1, Y1, Z3, norm_r)]

    eps = 1e-12  # used to drop the bottom-most tick robustly

    for i, (X, Y, Z, norm_here) in enumerate(datasets, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=cmap(norm_here(Z)),
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 0.7))
        ax.set_xlabel(r"$x_1^{(m)}$", labelpad=10)
        ax.set_ylabel(r"$x_2^{(m)}$", labelpad=10)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
        ax.tick_params(axis="z", pad=10)
        ax.zaxis.set_major_formatter(FuncFormatter(fmt_func))

        if i in (1, 2):
            if z_lim_left_mid is not None:
                ax.set_zlim(*z_lim_left_mid)

            if z_ticks_left_mid is not None:
                ticks = [
                    t
                    for t in z_ticks_left_mid
                    if z_lim_left_mid is None or t > z_lim_left_mid[0] + eps
                ]
                ax.set_zticks(ticks)
            else:
                ticks = [t for t in ax.get_zticks() if t > ax.get_zlim()[0] + eps]
                ax.set_zticks(ticks)
        else:
            if z_lim_right is not None:
                ax.set_zlim(*z_lim_right)
            else:
                ax.set_zlim(np.nanmin(Z), np.nanmax(Z))

            if z_ticks_right is not None:
                ticks = [
                    t
                    for t in z_ticks_right
                    if z_lim_right is None or t > z_lim_right[0] + eps
                ]
                ax.set_zticks(ticks)
            else:
                ticks = [t for t in ax.get_zticks() if t > ax.get_zlim()[0] + eps]
                ax.set_zticks(ticks)

        if mode == "prediction":
            if i == 1:
                ax.set_zlabel(r"$y^{(m)}$", labelpad=20)
            elif i == 2:
                ax.set_zlabel(r"$\hat{y}^{(m)}$", labelpad=20)
            else:
                ax.set_zlabel(r"$|y^{(m)}-\hat{y}^{(m)}|$", labelpad=20)
        else:
            ax.set_zlabel(zlabel_left_mid if i < 3 else zlabel_right, labelpad=20)

        if titles[i - 1] is not None:
            ax.set_title(titles[i - 1])

        _add_focus_box_lines(
            ax, X, Y, Z, bounds=bounds, color="white", lw=LINES_LINEWIDTH
        )

    plt.tight_layout(rect=[0, 0.15, 0.98, 1])
    if show:
        plt.show()
    return fig


def plot_epis_2_feat(
    d1: dict[str, Any],
    d2: dict[str, Any],
    d3: dict[str, Any],
    titles: Sequence[str] = (
        r"$\mathcal{G}_{\text{multi-target}}^{(1)}$",
        r"$\mathcal{G}_{\text{multi-target}}^{(2)}$",
        r"$\mathcal{G}_{\text{single}}$",
    ),
    focus_box: tuple[float, float] = (-4.0, 4.0),
    figsize: tuple[float, float] = (20, 5),
    # ---- Z-axis control (applies to all three subplots) ----
    z_lim: tuple[float, float] | None = None,
    z_ticks: Sequence[float] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot three epistemic surfaces with region colors.

    Each subplot shows epistemic uncertainty over a 2D input grid. Regions are
    colored according to their relation to a focus interval:

    - green:  both x1 and x2 within ``focus_box``
    - orange: exactly one of x1, x2 within ``focus_box``
    - red:    both x1 and x2 outside ``focus_box``

    Args:
        d1: Result dict with keys ``"X_test"`` and ``"epistemic_all"``.
        d2: Result dict with keys ``"X_test"`` and ``"epistemic_all"``.
        d3: Result dict with keys ``"X_test"`` and ``"epistemic_all"``.
        titles: Three titles (MathText allowed) placed above the subplots.
        focus_box: (low, high) interval used to classify points as in- or out-of-domain
            along both feature axes.
        figsize: Figure size in inches.
        z_lim: Shared (zmin, zmax) applied to all three subplots. If None, a shared
            1–2–5 scheme from the data is used.
        z_ticks: Shared z-ticks applied to all three subplots. If None, ticks from
            the shared 1–2–5 scheme are used. In all cases, the lowest tick is
            removed.

    Returns:
        A tuple ``(fig, axes)`` where ``fig`` is the Matplotlib Figure and
        ``axes`` is the list of three 3D Axes.
    """
    tab10 = cm.get_cmap("tab10")
    palette = {
        "green": tab10(2)[:3],
        "orange": tab10(1)[:3],
        "red": tab10(3)[:3],
    }

    def _facecolors_regions(
        X: np.ndarray, Y: np.ndarray, fb: tuple[float, float]
    ) -> np.ndarray:
        fmin, fmax = fb
        in_x1 = (X >= fmin) & (X <= fmax)
        in_x2 = (Y >= fmin) & (Y <= fmax)
        both_in = in_x1 & in_x2
        one_in = in_x1 ^ in_x2
        both_out = ~(in_x1 | in_x2)

        C = np.zeros(X.shape + (4,), dtype=float)
        C[both_in] = (*palette["green"], 1.0)
        C[one_in] = (*palette["orange"], 1.0)
        C[both_out] = (*palette["red"], 1.0)
        return C

    def _prep(
        meta: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_test = np.asarray(meta["X_test"], dtype=float)
        median_run = _select_median_run(meta["epistemic_all"])
        X, Y, Z = _grid_from_pairs_and_values(X_test, median_run)
        C = _facecolors_regions(X, Y, focus_box)
        return X, Y, Z, C

    data = [_prep(d) for d in (d1, d2, d3)]

    # Shared z-limits / ticks if not provided
    if z_lim is None or z_ticks is None:
        z_min = min(np.nanmin(Z) for (_, _, Z, _) in data)
        z_max = max(np.nanmax(Z) for (_, _, Z, _) in data)
        z0_auto, z_top_auto, z_ticks_auto = _nice_125_ticks(z_min, z_max, n_target=5)
        if z_lim is None:
            z_lim = (z0_auto, z_top_auto)
        if z_ticks is None:
            z_ticks = list(z_ticks_auto)

    # Remove bottom tick globally
    eps = 1e-12
    z_ticks_clean = [t for t in z_ticks if t > (z_lim[0] + eps)]

    # Formatter: short scientific notation (e.g., 1e-3)
    fmt_func = lambda v, _pos: f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    # Plotting
    fig = plt.figure(figsize=figsize)
    axes: list[Axes] = []

    for i, (X, Y, Z, C) in enumerate(data, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            facecolors=C,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1, 1, 0.7))

        ax.set_xlabel(r"$x_1^{(m)}$", labelpad=10)
        ax.set_ylabel(r"$x_2^{(m)}$", labelpad=10)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])

        # Shared z-axis control (all three) + remove bottom tick
        ax.set_zlim(*z_lim)
        ax.set_zticks(z_ticks_clean)
        ax.zaxis.set_major_formatter(FuncFormatter(fmt_func))
        ax.tick_params(axis="z", pad=10)

        ax.set_zlabel(r"$\sigma_{\text{epis}}^2(x^{(m)})$", labelpad=20)
        ax.set_title(titles[i - 1])
        axes.append(ax)

    # Legend under the row
    import matplotlib.patches as mpatches

    patch_in = mpatches.Patch(
        color=palette["green"],
        label=r"$x_1^{(m)}$ and $x_2^{(m)}$ ID",
    )
    patch_one = mpatches.Patch(
        color=palette["orange"],
        label=r"$x_1^{(m)}$ or $x_2^{(m)}$ OOD",
    )
    patch_both = mpatches.Patch(
        color=palette["red"],
        label=r"$x_1^{(m)}$ and $x_2^{(m)}$ OOD",
    )

    fig.legend(
        handles=[patch_in, patch_one, patch_both],
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=1.5,
        handleheight=1.5,
    )

    plt.tight_layout(rect=[0, 0.15, 0.98, 1])
    return fig, axes
