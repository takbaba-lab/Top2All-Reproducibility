#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import (
    FancyBboxPatch,
    Circle,
    Rectangle,
    FancyArrowPatch,
)

ROOT = Path(__file__).resolve().parents[2]
import os
OUT = Path(os.environ.get("TOP2ALL_PLOT_OUT", "reproduced_figures")) / "Figure1"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)

from figure_style import (
    MM,
    COLORS,
    apply_style,
    save_all,
)

apply_style()

DATA = ROOT / 'metadata/sample_metadata.csv'


meta = pd.read_csv(DATA)

if len(meta) != 24:
    raise RuntimeError(
        f"Expected 24 samples, found {len(meta)}"
    )


# ============================================================
# Colors
# ============================================================

BLUE = COLORS["Top2All"]
GREEN = COLORS["SameConditionMean"]
ORANGE = COLORS["kNN1"]

GREY = "#A7A9AC"
MID_GREY = "#8F969B"
LIGHT_GREY = "#E6E7E8"
VERY_LIGHT_GREY = "#F4F5F6"
DARK = "#222222"

FIG_W = 178 * MM
FIG_H = 118 * MM


# ============================================================
# Canvas
# ============================================================

def setup_canvas():

    fig = plt.figure(
        figsize=(FIG_W, FIG_H)
    )

    # A: framework
    ax_a = fig.add_axes([
        0.05, 0.63,
        0.90, 0.29,
    ])

    # B: compact experimental design
    ax_b = fig.add_axes([
        0.05, 0.28,
        0.30, 0.28,
    ])

    # C: wider evaluation design
    ax_c = fig.add_axes([
        0.39, 0.28,
        0.56, 0.28,
    ])

    # D: evaluation hierarchy
    ax_d = fig.add_axes([
        0.05, 0.05,
        0.90, 0.15,
    ])

    for ax in [
        ax_a,
        ax_b,
        ax_c,
        ax_d,
    ]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    return fig, ax_a, ax_b, ax_c, ax_d


# ============================================================
# Helpers
# ============================================================

def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=DARK,
    lw=1.2,
    mutation=10,
):

    p = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )

    ax.add_patch(p)

    return p


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    facecolor,
    edgecolor=DARK,
    linewidth=1.0,
    radius=0.03,
):

    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            f"round,pad=0.01,"
            f"rounding_size={radius}"
        ),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    ax.add_patch(p)

    return p


# ============================================================
# Panel A
# High-expression Top1000 -> broader transcriptome
# ============================================================

def draw_a(ax, clean=False):

    # --------------------------------------------------------
    # Expression axis
    # --------------------------------------------------------

    arrow(
        ax,
        0.035,
        0.18,
        0.035,
        0.82,
        color="0.35",
        lw=0.9,
        mutation=8,
    )

    if not clean:
        ax.text(
            0.015,
            0.50,
            "Expression",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.35",
        )

    # --------------------------------------------------------
    # INPUT: high-expression genes only
    # --------------------------------------------------------

    x0 = 0.075

    input_heights = np.array([
        0.63,
        0.59,
        0.55,
        0.51,
        0.47,
        0.44,
        0.41,
        0.38,
        0.35,
        0.32,
    ])

    for i, h in enumerate(input_heights):

        ax.add_patch(
            Rectangle(
                (
                    x0 + i * 0.018,
                    0.17,
                ),
                0.012,
                h,
                facecolor=BLUE,
                edgecolor="none",
            )
        )

    if not clean:

        # More space above the bars than v2
        ax.text(
            0.16,
            0.955,
            "Top 1000",
            ha="center",
            va="center",
            fontsize=9,
        )

        ax.text(
            0.16,
            0.885,
            "high-expression genes",
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.35",
        )

    # --------------------------------------------------------
    # Top2All
    # --------------------------------------------------------

    arrow(
        ax,
        0.29,
        0.48,
        0.39,
        0.48,
    )

    rounded_box(
        ax,
        0.40,
        0.29,
        0.17,
        0.38,
        facecolor=BLUE,
        edgecolor=BLUE,
    )

    if not clean:
        ax.text(
            0.485,
            0.48,
            "Top2All",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )

    arrow(
        ax,
        0.59,
        0.48,
        0.69,
        0.48,
    )

    # --------------------------------------------------------
    # OUTPUT: broader transcriptome
    # Includes long low-expression tail
    # --------------------------------------------------------

    n = 42

    xs = np.linspace(
        0.72,
        0.955,
        n,
    )

    heights = (
        0.64
        * np.exp(
            -np.linspace(
                0,
                4.4,
                n,
            )
        )
        + 0.035
    )

    rng = np.random.default_rng(101)

    heights += rng.normal(
        0,
        0.012,
        n,
    )

    heights = np.clip(
        heights,
        0.025,
        None,
    )

    for i, (x, h) in enumerate(
        zip(xs, heights)
    ):

        if i < 10:
            color = MID_GREY
        else:
            color = "#C8CDD1"

        ax.plot(
            [x, x],
            [0.17, 0.17 + h],
            color=color,
            linewidth=1.35,
        )

    if not clean:

        ax.text(
            0.835,
            0.955,
            "Predicted transcriptome",
            ha="center",
            va="center",
            fontsize=9,
        )

        # Align low-expression label and line with
        # the light-gray tail itself.
        tail_start = xs[10]
        tail_end = xs[-1]

        line_y = 0.115

        ax.plot(
            [
                tail_start,
                tail_end,
            ],
            [
                line_y,
                line_y,
            ],
            color="0.68",
            linewidth=0.75,
        )

        ax.text(
            (tail_start + tail_end) / 2,
            0.075,
            "low-expression genes",
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.45",
        )


# ============================================================
# Panel B
# 8 conditions x 3 replicates
# Male upper, Female lower
# ============================================================

B_ROWS = [
    ("Male",   "Sham",     "Control_male"),
    ("Male",   "Cas+Oil",  "Cas_Oil"),
    ("Male",   "Cas+DHT",  "Cas_DHT"),
    ("Male",   "Cas+E2",   "Cas_E2"),

    ("Female", "Sham",     "Control_female"),
    ("Female", "Ovx+Oil",  "Ovx_Oil"),
    ("Female", "Ovx+DHT",  "Ovx_DHT"),
    ("Female", "Ovx+E2",   "Ovx_E2"),
]


def draw_b(ax, clean=False):

    # Slightly more symmetric placement than v2
    y_positions = np.array([
        0.82,
        0.72,
        0.62,
        0.52,
        0.42,
        0.32,
        0.22,
        0.12,
    ])

    x_positions = [
        0.58,
        0.73,
        0.88,
    ]

    # --------------------------------------------------------
    # Male background
    # Corrected vertical position:
    # centered behind four Male rows
    # --------------------------------------------------------

    male_top = 0.87
    male_bottom = 0.47

    ax.add_patch(
        Rectangle(
            (
                0.04,
                male_bottom,
            ),
            0.92,
            male_top - male_bottom,
            facecolor=VERY_LIGHT_GREY,
            edgecolor="none",
            zorder=0,
        )
    )

    # separation between Male and Female
    ax.plot(
        [0.04, 0.96],
        [0.47, 0.47],
        color="0.80",
        linewidth=0.8,
    )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for (
        sex,
        label,
        cell_type
    ), y in zip(
        B_ROWS,
        y_positions,
    ):

        subset = meta[
            meta["cell_type"].astype(str)
            == cell_type
        ]

        if len(subset) != 3:
            raise RuntimeError(
                f"{cell_type}: expected 3 samples, "
                f"found {len(subset)}"
            )

        if not clean:

            ax.text(
                0.25,
                y,
                label,
                ha="left",
                va="center",
                fontsize=7.8,
            )

        # One color only:
        # color has no biological-class meaning here.
        for x in x_positions:

            ax.add_patch(
                Circle(
                    (x, y),
                    0.027,
                    facecolor=BLUE,
                    edgecolor="white",
                    linewidth=0.6,
                )
            )

    if not clean:

        ax.text(
            0.06,
            0.67,
            "Male",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

        ax.text(
            0.06,
            0.27,
            "Female",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

        for j, x in enumerate(
            x_positions,
            start=1,
        ):
            ax.text(
                x,
                0.94,
                str(j),
                ha="center",
                va="center",
                fontsize=8,
            )

        ax.text(
            0.73,
            1.00,
            "Replicate",
            ha="center",
            va="bottom",
            fontsize=8,
        )


# ============================================================
# Panel C
# OOF + condition / OOF - condition / LOCO
# ============================================================

def mini_scheme(
    ax,
    y,
    mode,
    clean=False,
):

    # More horizontal separation from left-hand labels
    xs = np.linspace(
        0.30,
        0.72,
        8,
    )

    # Slightly larger replicate ellipses
    dot_rx = 0.0105
    dot_ry = 0.015

    for i, x in enumerate(xs):

        offsets = [
            -0.016,
            0.0,
            0.016,
        ]

        if mode in [
            "oof_cond",
            "oof_nocond",
        ]:

            colors = [
                GREY,
                GREY,
                ORANGE,
            ]

        elif mode == "loco":

            if i == 6:
                colors = [
                    ORANGE,
                    ORANGE,
                    ORANGE,
                ]
            else:
                colors = [
                    GREY,
                    GREY,
                    GREY,
                ]

        else:
            raise ValueError(mode)

        for off, fc in zip(
            offsets,
            colors,
        ):

            # Use ellipse-like marker via scatter:
            # s is larger than v2 for visibility.
            ax.scatter(
                x + off,
                y,
                s=18,
                marker="o",
                facecolor=fc,
                edgecolor="none",
                zorder=3,
            )

    # --------------------------------------------------------
    # Condition embedding indicator
    # --------------------------------------------------------

    token_x = 0.84
    token_w = 0.038
    token_h = 0.038

    if mode == "oof_cond":

        ax.add_patch(
            Rectangle(
                (
                    token_x,
                    y - token_h / 2,
                ),
                token_w,
                token_h,
                facecolor=GREEN,
                edgecolor=GREEN,
                linewidth=0.8,
            )
        )

    else:

        ax.add_patch(
            Rectangle(
                (
                    token_x,
                    y - token_h / 2,
                ),
                token_w,
                token_h,
                facecolor="white",
                edgecolor=GREEN,
                linewidth=0.9,
            )
        )

        ax.plot(
            [
                token_x - 0.003,
                token_x + token_w + 0.003,
            ],
            [
                y - token_h / 2 - 0.004,
                y + token_h / 2 + 0.004,
            ],
            color=GREEN,
            linewidth=1.0,
        )

    if not clean:

        labels = {
            "oof_cond":
                "OOF + cond.",
            "oof_nocond":
                "OOF − cond.",
            "loco":
                "LOCO",
        }

        # more whitespace before the diagrams
        ax.text(
            0.025,
            y,
            labels[mode],
            ha="left",
            va="center",
            fontsize=8.5,
        )


def draw_c(ax, clean=False):

    mini_scheme(
        ax,
        0.73,
        "oof_cond",
        clean,
    )

    mini_scheme(
        ax,
        0.48,
        "oof_nocond",
        clean,
    )

    mini_scheme(
        ax,
        0.23,
        "loco",
        clean,
    )

    if not clean:

        # ----------------------------------------------------
        # Legend with wider train/test spacing
        # ----------------------------------------------------

        ax.scatter(
            0.52,
            0.94,
            s=18,
            facecolor=GREY,
            edgecolor="none",
        )

        ax.text(
            0.545,
            0.94,
            "train",
            va="center",
            ha="left",
            fontsize=7.5,
        )

        ax.scatter(
            0.68,
            0.94,
            s=18,
            facecolor=ORANGE,
            edgecolor="none",
        )

        ax.text(
            0.705,
            0.94,
            "test",
            va="center",
            ha="left",
            fontsize=7.5,
        )

        ax.add_patch(
            Rectangle(
                (
                    0.825,
                    0.927,
                ),
                0.028,
                0.028,
                facecolor=GREEN,
                edgecolor=GREEN,
            )
        )

        ax.text(
            0.87,
            0.94,
            "cond.",
            va="center",
            ha="left",
            fontsize=7.5,
        )


# ============================================================
# Panel D
# Evaluation hierarchy
#
# Sample-wise:
#   a row of the expression matrix
#
# Gene-wise:
#   a column of the expression matrix
#
# Residual:
#   replicate deviations from condition mean
#
# Condition shift:
#   true and predicted held-out condition shifts
#   from the training/reference centroid
# ============================================================

def draw_matrix_icon(
    ax,
    cx,
    cy,
    highlight,
):

    """
    Mini expression matrix.

    highlight:
        "row"    -> sample-wise evaluation
        "column" -> gene-wise evaluation
    """

    n_rows = 5
    n_cols = 7

    cell_w = 0.014
    cell_h = 0.030

    gap_x = 0.003
    gap_y = 0.009

    total_w = (
        n_cols * cell_w
        + (n_cols - 1) * gap_x
    )

    total_h = (
        n_rows * cell_h
        + (n_rows - 1) * gap_y
    )

    x0 = cx - total_w / 2
    y0 = cy - total_h / 2

    # deterministic gray-scale matrix
    shades = [
        "#E3E6E8",
        "#D1D5D8",
        "#BCC2C6",
        "#AAB1B6",
        "#D9DDE0",
    ]

    highlight_row = 2
    highlight_col = 3

    for r in range(n_rows):

        for c in range(n_cols):

            x = (
                x0
                + c * (cell_w + gap_x)
            )

            y = (
                y0
                + (n_rows - 1 - r)
                * (cell_h + gap_y)
            )

            if (
                highlight == "row"
                and r == highlight_row
            ):
                fc = BLUE

            elif (
                highlight == "column"
                and c == highlight_col
            ):
                fc = BLUE

            else:
                fc = shades[
                    (r + c) % len(shades)
                ]

            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor=fc,
                    edgecolor="white",
                    linewidth=0.25,
                )
            )


def draw_d(ax, clean=False):

    nodes = [
        ("Sample-wise",  0.12, BLUE),
        ("Gene-wise",    0.36, BLUE),
        ("Residual",     0.60, BLUE),
        ("Condition shift", 0.84, ORANGE),
    ]

    box_w = 0.17
    box_h = 0.42

    for i, (label, x, color) in enumerate(
        nodes
    ):

        rounded_box(
            ax,
            x - box_w / 2,
            0.29,
            box_w,
            box_h,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
        )

        # ====================================================
        # SAMPLE-WISE
        #
        # One row across genes:
        # evaluate a transcriptome profile within one sample.
        # ====================================================

        if i == 0:

            draw_matrix_icon(
                ax,
                cx=x,
                cy=0.50,
                highlight="row",
            )

        # ====================================================
        # GENE-WISE
        #
        # One column across samples:
        # evaluate one gene across samples.
        # ====================================================

        elif i == 1:

            draw_matrix_icon(
                ax,
                cx=x,
                cy=0.50,
                highlight="column",
            )

        # ====================================================
        # RESIDUAL
        #
        # Replicate deviation from condition mean.
        # ====================================================

        elif i == 2:

            mean_y = 0.49

            ax.plot(
                [
                    x - 0.060,
                    x + 0.060,
                ],
                [
                    mean_y,
                    mean_y,
                ],
                color="0.45",
                linewidth=1.0,
            )

            sample_x = [
                x - 0.040,
                x,
                x + 0.040,
            ]

            sample_y = [
                0.58,
                0.44,
                0.54,
            ]

            for sx, sy in zip(
                sample_x,
                sample_y,
            ):

                ax.plot(
                    [
                        sx,
                        sx,
                    ],
                    [
                        mean_y,
                        sy,
                    ],
                    color=BLUE,
                    linewidth=1.1,
                )

                ax.scatter(
                    sx,
                    sy,
                    s=14,
                    facecolor=BLUE,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )

        # ====================================================
        # UNSEEN STATE
        #
        # IMPORTANT:
        # This is expression-profile space, not latent space.
        #
        # Gray point:
        #   centroid/reference of training states.
        #
        # Orange arrow:
        #   true held-out-condition shift.
        #
        # Blue arrow:
        #   Top2All-predicted shift.
        #
        # The shorter blue arrow visually represents
        # amplitude compression observed in LOCO.
        # ====================================================

        else:

            origin_x = x - 0.055
            origin_y = 0.43

            # Training/reference centroid
            ax.scatter(
                origin_x,
                origin_y,
                s=26,
                facecolor=GREY,
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

            # True unseen-condition shift:
            # orange, longer
            true_x = x + 0.055
            true_y = 0.59

            arrow(
                ax,
                origin_x + 0.006,
                origin_y + 0.004,
                true_x - 0.008,
                true_y - 0.010,
                color=ORANGE,
                lw=1.5,
                mutation=8,
            )

            ax.scatter(
                true_x,
                true_y,
                s=24,
                facecolor=ORANGE,
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

            # Predicted shift:
            # blue and compressed
            pred_x = x + 0.020
            pred_y = 0.50

            arrow(
                ax,
                origin_x + 0.006,
                origin_y + 0.001,
                pred_x - 0.007,
                pred_y - 0.005,
                color=BLUE,
                lw=1.5,
                mutation=8,
            )

            ax.scatter(
                pred_x,
                pred_y,
                s=24,
                facecolor=BLUE,
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

        if not clean:

            ax.text(
                x,
                0.11,
                label,
                ha="center",
                va="center",
                fontsize=8,
            )

        if i < len(nodes) - 1:

            arrow(
                ax,
                x + 0.10,
                0.50,
                nodes[i + 1][1] - 0.10,
                0.50,
                color="0.55",
                lw=0.9,
                mutation=7,
            )


# ============================================================
# Build figure
# ============================================================

def make_figure(clean=False):

    (
        fig,
        ax_a,
        ax_b,
        ax_c,
        ax_d,
    ) = setup_canvas()

    draw_a(
        ax_a,
        clean=clean,
    )

    draw_b(
        ax_b,
        clean=clean,
    )

    draw_c(
        ax_c,
        clean=clean,
    )

    draw_d(
        ax_d,
        clean=clean,
    )

    if not clean:

        panel_axes = [
            ax_a,
            ax_b,
            ax_c,
            ax_d,
        ]

        labels = [
            "A",
            "B",
            "C",
            "D",
        ]

        for ax, label in zip(
            panel_axes,
            labels,
        ):

            ax.text(
                -0.02,
                1.04,
                label,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

    return fig


# ============================================================
# Save
# ============================================================

fig = make_figure(
    clean=False
)

save_all(
    fig,
    OUT / "Figure1",
)

plt.close(fig)




# ============================================================
# Report
# ============================================================

print("=" * 72)
print("FIGURE 1")
print("=" * 72)

print(
    "Samples:",
    len(meta),
)

print(
    "Conditions:",
    meta["cell_type"].nunique(),
)

print(
    "Replicates/condition:",
    meta.groupby("cell_type")
    .size()
    .unique()
    .tolist()
)

print()
print(
    "Panel D unseen-condition icon:"
)
print(
    "gray = training/reference centroid"
)
print(
    "orange = true held-out-condition shift"
)
print(
    "blue = Top2All-predicted shift"
)
print(
    "space = transcriptomic expression-profile space, not latent space"
)

print()
print(
    "Output:",
    OUT,
)

print("FIGURE 1 BUILD: PASS")