import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

INPUT = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis"
    r"\design_rule_book\lib1R4_triplicate_stride1\all_rg_values.csv"
)

OUTPUT = INPUT.parent / "VGIG_R4_central50_overlay.pdf"

CENTER_ID = "107_VGIG_R4_IGcenter_VGends"
ENDS_ID   = "108_VGIG_R4_VGcenter_IGends"

N_BINS = 140
SMOOTH_WINDOW = 5


# ============================================================
# COLORS
# ============================================================

# Motif identity — only used in sequence cartoons
VG_COLOR = "#4C78A8"       # VPGVG — blue
IG_COLOR = "#59A14F"       # VPGIG — green

# Architecture / distributions
# Cool vs warm, deliberately different from motif colors
CENTER_COLOR = "#287C8E"   # deep teal
ENDS_COLOR   = "#C96A4A"   # warm terracotta

# Neutral full-distribution outline
OUTLINE_COLOR = "#30343B"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("\nLoaded:")
print(INPUT)


# ============================================================
# SELECT EXACT ARCHITECTURES
# ============================================================

center_df = df[
    df["id"].astype(str) == CENTER_ID
].copy()

ends_df = df[
    df["id"].astype(str) == ENDS_ID
].copy()


if center_df.empty:
    raise ValueError(
        f"Could not find:\n{CENTER_ID}"
    )

if ends_df.empty:
    raise ValueError(
        f"Could not find:\n{ENDS_ID}"
    )


# ============================================================
# EXTRACT RAW Rg
# ============================================================

rg_center = pd.to_numeric(
    center_df["rg_nm"],
    errors="coerce"
).dropna().to_numpy()

rg_ends = pd.to_numeric(
    ends_df["rg_nm"],
    errors="coerce"
).dropna().to_numpy()


print("\n============================================================")
print("SEQUENCES SELECTED")
print("============================================================")

print(f"\n{CENTER_ID}")
print(
    "Replicates:",
    sorted(
        center_df["replicate"]
        .unique()
        .tolist()
    )
)

print(f"\n{ENDS_ID}")
print(
    "Replicates:",
    sorted(
        ends_df["replicate"]
        .unique()
        .tolist()
    )
)

print("\nFrames:")
print(f"VPGIG center : {len(rg_center):,}")
print(f"VPGIG ends   : {len(rg_ends):,}")


# ============================================================
# CENTRAL 50% Rg RANGE = IQR
# ============================================================

def central50(rg):
    """
    Return the 25th percentile, 75th percentile,
    and central 50% Rg range (IQR).
    """

    p25 = np.percentile(rg, 25)
    p75 = np.percentile(rg, 75)

    iqr = p75 - p25

    return p25, p75, iqr


p25_center, p75_center, iqr_center = central50(
    rg_center
)

p25_ends, p75_ends, iqr_ends = central50(
    rg_ends
)


print("\n============================================================")
print("CENTRAL 50% Rg RANGE (IQR)")
print("============================================================")

print(
    f"VPGIG center : "
    f"{p25_center:.3f}–{p75_center:.3f} nm "
    f"(IQR = {iqr_center:.3f} nm)"
)

print(
    f"VPGIG ends   : "
    f"{p25_ends:.3f}–{p75_ends:.3f} nm "
    f"(IQR = {iqr_ends:.3f} nm)"
)


# ============================================================
# ADDITIONAL COMPARISON
# ============================================================

absolute_difference = (
    iqr_ends
    -
    iqr_center
)

relative_difference = (
    absolute_difference
    /
    iqr_center
    *
    100.0
)


print("\nDifference:")
print(
    f"VPGIG ends - VPGIG center = "
    f"{absolute_difference:+.3f} nm"
)

print(
    f"Relative to VPGIG center = "
    f"{relative_difference:+.1f}%"
)


# ============================================================
# HISTOGRAM
# ============================================================

all_rg = np.concatenate([
    rg_center,
    rg_ends
])

bins = np.linspace(
    all_rg.min(),
    all_rg.max(),
    N_BINS + 1
)


counts_center, edges = np.histogram(
    rg_center,
    bins=bins,
    density=True
)

counts_ends, _ = np.histogram(
    rg_ends,
    bins=bins,
    density=True
)


centers = 0.5 * (
    edges[:-1]
    +
    edges[1:]
)


# ============================================================
# LIGHT SMOOTHING
# ============================================================

def smooth(y, window=5):

    kernel = (
        np.ones(window)
        /
        window
    )

    return np.convolve(
        y,
        kernel,
        mode="same"
    )


counts_center = smooth(
    counts_center,
    SMOOTH_WINDOW
)

counts_ends = smooth(
    counts_ends,
    SMOOTH_WINDOW
)


# ============================================================
# CENTRAL-50% MASKS
# ============================================================

mask_center = (
    (centers >= p25_center)
    &
    (centers <= p75_center)
)

mask_ends = (
    (centers >= p25_ends)
    &
    (centers <= p75_ends)
)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(7.2, 5.0)
)

gs = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[5.0, 1.0],
    hspace=0.02
)

ax = fig.add_subplot(
    gs[0]
)

ax_int = fig.add_subplot(
    gs[1],
    sharex=ax
)


# ============================================================
# ONLY FILL CENTRAL 50%
# ============================================================

# VPGIG CENTER
ax.fill_between(
    centers,
    0,
    counts_center,
    where=mask_center,
    interpolate=True,
    color=CENTER_COLOR,
    alpha=0.22,
    linewidth=0,
    zorder=1
)

# VPGIG ENDS
ax.fill_between(
    centers,
    0,
    counts_ends,
    where=mask_ends,
    interpolate=True,
    color=ENDS_COLOR,
    alpha=0.20,
    linewidth=0,
    zorder=2
)


# ============================================================
# COMPLETE DISTRIBUTION OUTLINES
# ============================================================

ax.plot(
    centers,
    counts_center,
    color=CENTER_COLOR,
    linewidth=2.2,
    zorder=4
)

ax.plot(
    centers,
    counts_ends,
    color=ENDS_COLOR,
    linewidth=2.2,
    zorder=5
)


# ============================================================
# SUBTLE P25 / P75 INDICATORS
# ============================================================

# These connect the shaded central 50% region
# to the interval shown below.

for x in [
    p25_center,
    p75_center
]:

    ax.axvline(
        x,
        color=CENTER_COLOR,
        linewidth=0.7,
        alpha=0.20,
        linestyle=":",
        zorder=0
    )


for x in [
    p25_ends,
    p75_ends
]:

    ax.axvline(
        x,
        color=ENDS_COLOR,
        linewidth=0.7,
        alpha=0.20,
        linestyle=":",
        zorder=0
    )


# ============================================================
# MAIN AXIS FORMATTING
# ============================================================

ymax = max(
    counts_center.max(),
    counts_ends.max()
)

ax.set_ylim(
    0,
    ymax * 1.06
)

ax.set_ylabel(
    "Probability density",
    fontsize=10
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

ax.spines["bottom"].set_visible(
    False
)

ax.tick_params(
    axis="y",
    direction="out",
    length=3.5,
    width=0.8,
    labelsize=9
)

ax.tick_params(
    axis="x",
    bottom=False,
    labelbottom=False
)


# ============================================================
# CENTRAL 50% INTERVAL PANEL
# ============================================================

ax_int.set_ylim(
    -0.05,
    1.55
)

y_center = 1.05
y_ends = 0.40


# ------------------------------------------------------------
# CENTER INTERVAL
# ------------------------------------------------------------

ax_int.hlines(
    y_center,
    p25_center,
    p75_center,
    color=CENTER_COLOR,
    linewidth=3.0
)

ax_int.vlines(
    [
        p25_center,
        p75_center
    ],
    y_center - 0.10,
    y_center + 0.10,
    color=CENTER_COLOR,
    linewidth=1.4
)

ax_int.text(
    p75_center + 0.18,
    y_center,
    rf"$\mathrm{{IQR}} = {iqr_center:.2f}$ nm",
    color=CENTER_COLOR,
    fontsize=9,
    va="center",
    ha="left"
)


# ------------------------------------------------------------
# ENDS INTERVAL
# ------------------------------------------------------------

ax_int.hlines(
    y_ends,
    p25_ends,
    p75_ends,
    color=ENDS_COLOR,
    linewidth=3.0
)

ax_int.vlines(
    [
        p25_ends,
        p75_ends
    ],
    y_ends - 0.10,
    y_ends + 0.10,
    color=ENDS_COLOR,
    linewidth=1.4
)

ax_int.text(
    p75_ends + 0.18,
    y_ends,
    rf"$\mathrm{{IQR}} = {iqr_ends:.2f}$ nm",
    color=ENDS_COLOR,
    fontsize=9,
    va="center",
    ha="left"
)


# ============================================================
# INTERVAL AXIS FORMATTING
# ============================================================

ax_int.set_xlabel(
    r"$R_g$ (nm)",
    fontsize=10
)

ax_int.set_yticks([])

ax_int.spines["top"].set_visible(
    False
)

ax_int.spines["right"].set_visible(
    False
)

ax_int.spines["left"].set_visible(
    False
)

ax_int.tick_params(
    axis="x",
    direction="out",
    length=3.5,
    width=0.8,
    labelsize=9
)


# ============================================================
# COMMON X RANGE
# ============================================================

xrange = (
    all_rg.max()
    -
    all_rg.min()
)

xmin = (
    all_rg.min()
    -
    0.03 * xrange
)

# Give slightly more room on right for IQR labels
xmax = (
    all_rg.max()
    +
    0.08 * xrange
)

ax.set_xlim(
    xmin,
    xmax
)


# ============================================================
# SEQUENCE CARTOON FUNCTION
# ============================================================

def draw_sequence_bar(
    ax,
    x,
    y,
    width,
    height,
    pattern,
    architecture_color,
    label
):

    # --------------------------------------------------------
    # 1:1 composition
    #
    # IG center:
    # VG21 | IG42 | VG21
    #
    # IG ends:
    # IG21 | VG42 | IG21
    # --------------------------------------------------------

    if pattern == "IG_center":

        segments = [
            (0.00, 0.25, VG_COLOR),
            (0.25, 0.50, IG_COLOR),
            (0.75, 0.25, VG_COLOR)
        ]

    elif pattern == "IG_ends":

        segments = [
            (0.00, 0.25, IG_COLOR),
            (0.25, 0.50, VG_COLOR),
            (0.75, 0.25, IG_COLOR)
        ]

    else:

        raise ValueError(
            f"Unknown pattern: {pattern}"
        )


    # --------------------------------------------------------
    # Architecture-color line
    # --------------------------------------------------------

    ax.plot(
        [
            x,
            x + width
        ],
        [
            y + height + 0.026,
            y + height + 0.026
        ],
        transform=ax.transAxes,
        color=architecture_color,
        linewidth=2.7,
        solid_capstyle="round",
        clip_on=False
    )


    # --------------------------------------------------------
    # Motif blocks
    # --------------------------------------------------------

    for start, fraction, color in segments:

        rect = Rectangle(
            (
                x + start * width,
                y
            ),
            fraction * width,
            height,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False
        )

        ax.add_patch(
            rect
        )


    # --------------------------------------------------------
    # Outline
    # --------------------------------------------------------

    outline = Rectangle(
        (
            x,
            y
        ),
        width,
        height,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="#3B3B3B",
        linewidth=0.65,
        clip_on=False
    )

    ax.add_patch(
        outline
    )


    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    ax.text(
        x + width / 2,
        y - 0.035,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#303030"
    )


# ============================================================
# DRAW ARCHITECTURES
# ============================================================

draw_sequence_bar(
    ax=ax,
    x=0.18,
    y=1.105,
    width=0.27,
    height=0.045,
    pattern="IG_center",
    architecture_color=CENTER_COLOR,
    label="VPGIG center"
)

draw_sequence_bar(
    ax=ax,
    x=0.58,
    y=1.105,
    width=0.27,
    height=0.045,
    pattern="IG_ends",
    architecture_color=ENDS_COLOR,
    label="VPGIG ends"
)


# ============================================================
# MOTIF LEGEND
# ============================================================

legend_y = 1.245


# VPGVG
vg_patch = Rectangle(
    (
        0.37,
        legend_y
    ),
    0.025,
    0.025,
    transform=ax.transAxes,
    facecolor=VG_COLOR,
    edgecolor="none",
    clip_on=False
)

ax.add_patch(
    vg_patch
)

ax.text(
    0.405,
    legend_y + 0.0125,
    "VPGVG",
    transform=ax.transAxes,
    va="center",
    ha="left",
    fontsize=9
)


# VPGIG
ig_patch = Rectangle(
    (
        0.53,
        legend_y
    ),
    0.025,
    0.025,
    transform=ax.transAxes,
    facecolor=IG_COLOR,
    edgecolor="none",
    clip_on=False
)

ax.add_patch(
    ig_patch
)

ax.text(
    0.565,
    legend_y + 0.0125,
    "VPGIG",
    transform=ax.transAxes,
    va="center",
    ha="left",
    fontsize=9
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(
    top=0.68,
    left=0.12,
    right=0.97,
    bottom=0.12
)


# ============================================================
# SAVE PDF + PNG + SVG
# ============================================================

pdf_output = OUTPUT
png_output = OUTPUT.with_suffix(
    ".png"
)
svg_output = OUTPUT.with_suffix(
    ".svg"
)


fig.savefig(
    pdf_output,
    bbox_inches="tight"
)

fig.savefig(
    png_output,
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    svg_output,
    format="svg",
    bbox_inches="tight"
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n============================================================")
print("CENTRAL 50% Rg SUMMARY")
print("============================================================")

print(
    f"VPGIG center IQR : "
    f"{iqr_center:.3f} nm"
)

print(
    f"VPGIG ends IQR   : "
    f"{iqr_ends:.3f} nm"
)

print(
    f"Absolute difference: "
    f"{absolute_difference:+.3f} nm"
)

print(
    f"Relative difference: "
    f"{relative_difference:+.1f}%"
)


print("\n============================================================")
print("SAVED")
print("============================================================")

print(
    f"PDF: {pdf_output}"
)

print(
    f"PNG: {png_output}"
)

print(
    f"SVG: {svg_output}"
)


plt.show()