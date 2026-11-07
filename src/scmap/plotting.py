"""Shared figure style, following Nature's figure specifications.

Arial at 5-7 pt, bold lowercase panel labels at 8 pt, no background grid, axis lines and
ticks only, a colour-blind-safe palette (Wong, Nat. Methods 8, 441; 2011), and widths of one
(89 mm) or two (183 mm) journal columns. Messages belong in the caption, not on the figure,
so panels carry letters instead of titles. Every figure is saved as an editable vector PDF
and a 300 dpi PNG for the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4  # inches per millimetre
SINGLE = 89 * MM  # one Nature column
ONE_HALF = 120 * MM
DOUBLE = 183 * MM  # two Nature columns

INK, INK_2, INK_3 = "#000000", "#333333", "#777777"
GRID = "#dddddd"  # kept for reference lines only; no background grid
# Wong palette with the vermillion swapped for a deep red and the amber for a purple (ColorBrewer):
# blue and red stay distinct under common colour-vision deficiencies; red and green differ in
# luminance
BLUE, RED, PURPLE, SKY, GREEN = "#0072B2", "#B2182B", "#7B3294", "#56B4E9", "#009E73"
SERIES = [BLUE, RED, GREEN, PURPLE, "#E69F00"]
CLASSIFIERS = ("logreg", "knn_pca", "scvi_knn", "scanvi", "scanvi_knn")
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#9ecae1", BLUE, "#08306b"])

METHOD_LABELS = {
    "logreg": "Logistic regression",
    "knn_pca": "kNN (PCA)",
    "scvi_knn": "scVI + kNN",
    "scanvi": "scANVI",
    "scanvi_knn": "scANVI + kNN",
    "scanvi_latent_distance": "scANVI (latent distance)",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6.5,
            "axes.titlesize": 6.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.borderaxespad": 0.2,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "errorbar.capsize": 1.5,
            "patch.linewidth": 0.4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays editable
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    """Bold lowercase panel letter at the top left, outside the axes (Nature style)."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig, path: Path) -> None:
    """Write the vector PDF (for print) and a 300 dpi PNG (for the README)."""
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
