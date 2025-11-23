import os
from typing import Iterable, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from src.constants import BASE_DIR


def _ensure_image_dir(project_root_dir: str, chapter_id: str) -> str:
    """Create (if needed) and return the images folder for a given chapter id."""
    images_path = os.path.join(project_root_dir, "images", chapter_id)
    os.makedirs(images_path, exist_ok=True)
    return images_path


class NotebookFigureSaver:
    """Lightweight, publication-friendly figure exporter for notebooks & scripts."""

    def __init__(self, chapter_id: str) -> None:
        self.project_root_dir = BASE_DIR
        self.images_path = _ensure_image_dir(self.project_root_dir, chapter_id)
        self.default_dpi = 150

    def save_fig(
        self,
        fig: plt.Figure,
        fig_id: str,
        *,
        leg: Optional[Legend] = None,
        formats: Iterable[str] = ("png", "pdf"),
        dpi: Optional[int] = None,
        resize: Optional[Tuple[float, float]] = None,  # (width, height) in inches
        close: bool = False,
        quiet: bool = False,
        suppress_pgf: Optional[bool] = None,  # only relevant if "pgf" is in formats
        **savefig_kwargs,
    ) -> None:
        """Save a figure (and optionally its legend) to multiple formats (PNG/PDF/PGF).

        If a legend object is provided (e.g., from fig.legend()), it will be added
        to the figure before saving, ensuring the legend is rendered in all outputs.
        PGF will be written only if explicitly requested and configured correctly.
        """
        if resize:
            fig.set_size_inches(resize, forward=True)

        use_dpi = dpi if dpi is not None else self.default_dpi

        # --- Ensure legend is attached to the figure before saving ---
        if leg is not None:
            # Make sure the legend is included in the save (especially for tight_layout)
            fig.add_artist(leg)

        # --- PGF handling ---
        formats = tuple(formats)
        want_pgf = "pgf" in formats
        if suppress_pgf is True and want_pgf:
            formats = tuple(ext for ext in formats if ext != "pgf")
            want_pgf = False

        # --- Save all requested formats ---
        for ext in formats:
            path = os.path.join(self.images_path, f"{fig_id}.{ext}")
            fig.savefig(
                path, dpi=use_dpi, format=ext, bbox_inches="tight", **savefig_kwargs
            )
            if not quiet:
                print(f"Saved: {path}")

        if close:
            plt.close(fig)
