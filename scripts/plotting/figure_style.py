from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager

from matplotlib import font_manager

ARIAL_FOUND = any(
    f.name == "Arial"
    for f in font_manager.fontManager.ttflist
)

if not ARIAL_FOUND:
    raise RuntimeError(
        "Arial is not installed or not visible to matplotlib."
    )

FONT = "Arial"

MM = 1 / 25.4

# Color-blind-friendly method encoding.
COLORS = {
    "Top2All": "#0072B2",
    "GlobalMean": "#999999",
    "SameConditionMean": "#009E73",
    "kNN1": "#D55E00",
    "kNN3": "#E69F00",
    "kNN5": "#CC79A7",
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.5,

        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,

        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        "legend.fontsize": 8.0,

        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean_axes(ax):
    """Remove all text/ticks/axes while preserving plot geometry."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def publication_axes(ax):
    """Minimal publication-style axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        pad=2.5,
    )


def save_all(fig, stem, dpi=600):
    """
    Save PNG, PDF, SVG.
    Do not use bbox_inches='tight' so axes/clean canvases retain
    identical physical dimensions.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=dpi,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
    )
    fig.savefig(
        stem.with_suffix(".svg"),
    )
