import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# DESIGN RULE 4 — REGIME OVERVIEW
#
# Recreates the original 2-panel thesis figure:
#   A  Mean Rg
#   B  Collapsed fraction
#
# For each library/composition:
#   solid line = mean across architectures
#   translucent band = min-max across architectures
#
# Correct composition labels:
#   R1 6:1 (14%)
#   R2 3:1 (25%)
#   R3 2:1 (33%)
#   R4 1:1 (50%)
#   R5 1:2 (67%)
#   R6 1:5 (83%)
# ============================================================

RATIOS = ["R1", "R2", "R3", "R4", "R5", "R6"]

RATIO_LABELS = {
    "R1": "6:1\n(14%)",
    "R2": "3:1\n(25%)",
    "R3": "2:1\n(33%)",
    "R4": "1:1\n(50%)",
    "R5": "1:2\n(67%)",
    "R6": "1:5\n(83%)",
}

LIBRARIES = ["VGEG", "VGKG", "IGKG"]

LIBRARY_LABELS = {
    "VGEG": "VPGVG + VPGEG",
    "VGKG": "VPGVG + VPGKG",
    "IGKG": "VPGIG + VPGKG",
}

COLORS = {
    "VGEG": "#4472C4",
    "VGKG": "#ED7D31",
    "IGKG": "#70AD47",
}

MARKERS = {
    "VGEG": "o",
    "VGKG": "s",
    "IGKG": "^",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def normalize_ratio(value):
    s = str(value).upper().strip()
    for r in RATIOS:
        if r in s:
            return r
    return s


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.arange(len(RATIOS)))
    ax.set_xticklabels([RATIO_LABELS[r] for r in RATIOS])
    ax.set_xlabel("Motif composition A:C")


def panel_label(ax, letter):
    ax.text(
        -0.13, 1.08, letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
    )


def summarize(df):
    rows = []

    for library in LIBRARIES:
        for ratio in RATIOS:
            sub = df[
                (df["library"] == library)
                & (df["ratio_norm"] == ratio)
            ]

            if sub.empty:
                raise ValueError(f"No rows found for {library}, {ratio}")

            rows.append({
                "library": library,
                "ratio": ratio,
                "n_architectures": len(sub),

                "mean_Rg_mean": sub["mean_Rg"].mean(),
                "mean_Rg_min": sub["mean_Rg"].min(),
                "mean_Rg_max": sub["mean_Rg"].max(),

                "collapsed_mean": sub["collapsed_percent"].mean(),
                "collapsed_min": sub["collapsed_percent"].min(),
                "collapsed_max": sub["collapsed_percent"].max(),
            })

    return pd.DataFrame(rows)


def get_library_summary(summary, library):
    sub = summary[summary["library"] == library].copy()
    sub["ratio"] = pd.Categorical(
        sub["ratio"],
        categories=RATIOS,
        ordered=True,
    )
    return sub.sort_values("ratio")


def main():
    parser = argparse.ArgumentParser(
        description="Recreate the original DR4 2-panel regime overview."
    )
    parser.add_argument(
        "--indir",
        required=True,
        help="Folder containing rg_summary_VGEG.csv, rg_summary_VGKG.csv, rg_summary_IGKG.csv",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output folder",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=4.61,
        help="Collapse threshold in nm (default 4.61)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("DESIGN RULE 4 — REGIME OVERVIEW")
    print("=" * 78)
    print(f"Input : {args.indir}")
    print(f"Output: {args.outdir}")
    print()

    frames = []

    for library in LIBRARIES:
        path = os.path.join(args.indir, f"rg_summary_{library}.csv")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing input file:\n{path}")

        temp = pd.read_csv(path)

        required = ["ratio", "architecture", "mean_Rg", "collapsed_percent"]
        missing = [c for c in required if c not in temp.columns]

        if missing:
            raise KeyError(
                f"{library}: missing columns {missing}\n\n"
                "Available columns:\n"
                + "\n".join(temp.columns)
            )

        temp["library"] = library
        temp["ratio_norm"] = temp["ratio"].apply(normalize_ratio)

        frames.append(temp)
        print(f"Loaded {library}: {len(temp)} rows")

    df = pd.concat(frames, ignore_index=True)
    summary = summarize(df)

    print("\nArchitectures contributing to each point:")
    print(
        summary.pivot(
            index="library",
            columns="ratio",
            values="n_architectures",
        )
    )

    summary_path = os.path.join(
        args.outdir,
        "01_regime_overview_summary.csv",
    )
    summary.to_csv(summary_path, index=False)

    # ========================================================
    # FIGURE
    # ========================================================

    x = np.arange(len(RATIOS))

    fig, axes = plt.subplots(
        1, 2,
        figsize=(9.6, 4.2),
    )

    # --------------------------------------------------------
    # A — Mean Rg
    # --------------------------------------------------------
    ax = axes[0]

    for library in LIBRARIES:
        sub = get_library_summary(summary, library)

        y = sub["mean_Rg_mean"].to_numpy(float)
        low = sub["mean_Rg_min"].to_numpy(float)
        high = sub["mean_Rg_max"].to_numpy(float)

        ax.fill_between(
            x,
            low,
            high,
            color=COLORS[library],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            y,
            color=COLORS[library],
            marker=MARKERS[library],
            markersize=7,
            linewidth=2.5,
            markeredgecolor=COLORS[library],
            markerfacecolor=COLORS[library],
            zorder=3,
        )

    ax.axhline(
        args.threshold,
        color="#666666",
        linestyle="--",
        linewidth=1.3,
        zorder=0,
    )

    style_axis(ax)
    ax.set_ylabel(r"Mean $R_g$ (nm)")
    panel_label(ax, "A")

    # --------------------------------------------------------
    # B — Collapsed fraction
    # --------------------------------------------------------
    ax = axes[1]

    for library in LIBRARIES:
        sub = get_library_summary(summary, library)

        y = sub["collapsed_mean"].to_numpy(float)
        low = sub["collapsed_min"].to_numpy(float)
        high = sub["collapsed_max"].to_numpy(float)

        ax.fill_between(
            x,
            low,
            high,
            color=COLORS[library],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

        ax.plot(
            x,
            y,
            color=COLORS[library],
            marker=MARKERS[library],
            markersize=7,
            linewidth=2.5,
            markeredgecolor=COLORS[library],
            markerfacecolor=COLORS[library],
            zorder=3,
        )

    style_axis(ax)
    ax.set_ylabel("Collapsed fraction (%)")
    panel_label(ax, "B")

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS[library],
            marker=MARKERS[library],
            linewidth=2.5,
            markersize=8,
            label=LIBRARY_LABELS[library],
        )
        for library in LIBRARIES
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.2,
    )

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.25,
        wspace=0.30,
    )

    png_path = os.path.join(
        args.outdir,
        "01_regime_overview.png",
    )
    pdf_path = os.path.join(
        args.outdir,
        "01_regime_overview.pdf",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
