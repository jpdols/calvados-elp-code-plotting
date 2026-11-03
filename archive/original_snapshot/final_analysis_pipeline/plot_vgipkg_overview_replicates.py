import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


# ============================================================
# VGIPKG ARCHITECTURE CLASSIFICATION
# ============================================================

CONTROLS = {833, 834, 835}


# ------------------------------------------------------------
# COLOUR = KG POSITION
# ------------------------------------------------------------

KG_CLASS = {
    833: "control",
    834: "control",
    835: "control",

    836: "peripheral",
    837: "internal",
    838: "internal",
    839: "internal",
    840: "internal",

    841: "peripheral",
    842: "internal",
    843: "internal",

    844: "internal",
    845: "peripheral",
    846: "internal",

    847: "peripheral",
    848: "internal",
    849: "internal",

    850: "peripheral",

    851: "peripheral",
    852: "internal",
    853: "peripheral",

    854: "peripheral",
    855: "internal",
    856: "peripheral",
    857: "peripheral",
    858: "peripheral",

    859: "internal",
    860: "internal",
    861: "internal",
    862: "internal",
}


# ------------------------------------------------------------
# SHAPE = IP DOMAIN SIZE / ORGANIZATION
# ------------------------------------------------------------

IP_CLASS = {
    833: "control",
    834: "control",
    835: "control",

    # Large IP domains
    836: "large",
    837: "large",
    838: "large",
    839: "large",
    840: "large",

    841: "large",
    842: "large",
    843: "large",

    844: "large",
    845: "large",
    846: "large",

    # Three/four smaller IP domains
    847: "small",
    848: "small",
    849: "small",
    850: "small",

    # Large / polarized domains
    851: "large",
    852: "large",
    853: "large",

    854: "large",
    855: "large",
    856: "large",

    # Gradient designs
    857: "small",
    858: "small",

    # Large central IP region
    859: "large",
    860: "large",
    861: "large",

    # Four smaller IP domains
    862: "small",
}


# ============================================================
# VISUAL SETTINGS
# ============================================================

COLORS = {
    "peripheral": "C0",
    "internal": "C1",
    "control": "0.45",
}

MARKERS = {
    "large": "o",
    "small": "^",
    "control": "D",
}


# ============================================================
# RAW METRICS REQUIRED FROM EACH REPLICATE
# ============================================================

RAW_METRICS = [
    "mean_rg_nm_370K",
    "delta_mean_rg_370_minus_300_nm",
    "k_collapse_per_ns_370K",
    "k_escape_per_ns_370K",
    "mean_collapsed_dwell_ns_370K",
]


# ============================================================
# HELPERS
# ============================================================

def extract_numeric_id(value):
    """
    Convert:
        833_VGIPKG_R2_random_shuffle
    to:
        833
    """
    return int(str(value).split("_")[0])


def load_replicates(files):
    """
    Load any number of replicate plotting-summary CSVs.

    Each file represents one independent replicate.
    """

    all_reps = []

    for rep_number, file in enumerate(files, start=1):

        file = Path(file)

        print(f"Loading replicate {rep_number}:")
        print(f"  {file}")

        df = pd.read_csv(file)

        if "id" not in df.columns:
            raise ValueError(
                f"'id' column not found in:\n{file}"
            )

        missing = [
            metric
            for metric in RAW_METRICS
            if metric not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns in {file}:\n"
                f"{missing}"
            )

        df["architecture_id"] = (
            df["id"].apply(extract_numeric_id)
        )

        df["replicate"] = rep_number

        # ----------------------------------------------------
        # Calculate kinetic ratio within each replicate first.
        # ----------------------------------------------------

        df["k_ratio_collapse_escape_370K"] = (
            df["k_collapse_per_ns_370K"]
            /
            df["k_escape_per_ns_370K"]
        )

        keep_columns = [
            "architecture_id",
            "replicate",
        ]

        if "architecture_type" in df.columns:
            keep_columns.append(
                "architecture_type"
            )

        keep_columns += RAW_METRICS

        keep_columns.append(
            "k_ratio_collapse_escape_370K"
        )

        all_reps.append(
            df[keep_columns].copy()
        )

    combined = pd.concat(
        all_reps,
        ignore_index=True
    )

    return combined


def aggregate_replicates(df):
    """
    Calculate replicate mean, SD and n for every architecture.
    """

    metrics = RAW_METRICS + [
        "k_ratio_collapse_escape_370K"
    ]

    agg_dict = {}

    for metric in metrics:

        agg_dict[f"{metric}_mean"] = (
            metric,
            "mean"
        )

        agg_dict[f"{metric}_sd"] = (
            metric,
            "std"
        )

        agg_dict[f"{metric}_n"] = (
            metric,
            "count"
        )

    summary = (
        df
        .groupby("architecture_id")
        .agg(**agg_dict)
        .reset_index()
    )

    # Add architecture name
    if "architecture_type" in df.columns:

        names = (
            df
            .groupby("architecture_id")[
                "architecture_type"
            ]
            .first()
            .reset_index()
        )

        summary = summary.merge(
            names,
            on="architecture_id",
            how="left"
        )

    # Add classifications
    summary["kg_class"] = (
        summary["architecture_id"]
        .map(KG_CLASS)
    )

    summary["ip_class"] = (
        summary["architecture_id"]
        .map(IP_CLASS)
    )

    missing_kg = summary.loc[
        summary["kg_class"].isna(),
        "architecture_id"
    ].tolist()

    missing_ip = summary.loc[
        summary["ip_class"].isna(),
        "architecture_id"
    ].tolist()

    if missing_kg:
        print(
            "WARNING: missing KG classification:",
            missing_kg
        )

    if missing_ip:
        print(
            "WARNING: missing IP classification:",
            missing_ip
        )

    return summary


# ============================================================
# LEGEND
# ============================================================

def make_legend_items():

    return [

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLORS["peripheral"],
            markeredgecolor="black",
            label="Peripheral / outward KG"
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=COLORS["internal"],
            markeredgecolor="black",
            label="Internal KG"
        ),

        Line2D(
            [0], [0],
            marker="D",
            linestyle="None",
            markerfacecolor=COLORS["control"],
            markeredgecolor="black",
            label="Control / reference"
        ),

        Line2D(
            [0], [0],
            marker=MARKERS["large"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Large IP domains"
        ),

        Line2D(
            [0], [0],
            marker=MARKERS["small"],
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Small / multiple IP domains"
        ),
    ]


# ============================================================
# 6-PANEL OVERVIEW
# ============================================================

def make_overview_plot(summary, outdir):

    outdir = Path(outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Sort ONCE using replicate-mean Rg at 370 K.
    # --------------------------------------------------------

    summary = (
        summary
        .sort_values(
            "mean_rg_nm_370K_mean",
            ascending=True
        )
        .reset_index(drop=True)
    )

    x = np.arange(
        len(summary)
    )

    metrics = [

        {
            "column": "mean_rg_nm_370K",
            "title": "A   Compaction at 370 K",
            "ylabel": r"Mean $R_g$ at 370 K (nm)",
        },

        {
            "column":
                "delta_mean_rg_370_minus_300_nm",
            "title": "B   Temperature response",
            "ylabel":
                r"$\Delta R_g$ (370 K - 300 K) (nm)",
        },

        {
            "column":
                "k_collapse_per_ns_370K",
            "title":
                "C   Rate of entering collapsed state",
            "ylabel":
                r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
        },

        {
            "column":
                "k_escape_per_ns_370K",
            "title":
                "D   Rate of leaving collapsed state",
            "ylabel":
                r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
        },

        {
            "column":
                "k_ratio_collapse_escape_370K",
            "title":
                "E   Collapse / escape rate ratio",
            "ylabel":
                r"$k_{\mathrm{collapse}}"
                r"/k_{\mathrm{escape}}$",
        },

        {
            "column":
                "mean_collapsed_dwell_ns_370K",
            "title":
                "F   Mean collapsed-state dwell time",
            "ylabel":
                r"Mean collapsed dwell time (ns)",
        },
    ]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(15, 22),
        sharex=True
    )

    for ax, metric_info in zip(
        axes,
        metrics
    ):

        metric = metric_info["column"]

        mean_col = f"{metric}_mean"
        sd_col = f"{metric}_sd"

        for i, row in summary.iterrows():

            kg_class = row["kg_class"]
            ip_class = row["ip_class"]

            color = COLORS.get(
                kg_class,
                "0.5"
            )

            marker = MARKERS.get(
                ip_class,
                "o"
            )

            mean = row[mean_col]
            sd = row[sd_col]

            # n=1 -> SD is NaN
            if pd.isna(sd):
                sd = 0

            ax.errorbar(
                i,
                mean,
                yerr=sd,
                fmt=marker,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.45,
                markersize=7,
                capsize=3,
                elinewidth=1.0,
                alpha=0.95,
            )

        ax.set_ylabel(
            metric_info["ylabel"]
        )

        ax.set_title(
            metric_info["title"],
            loc="left",
            fontsize=12,
            fontweight="bold"
        )

        ax.grid(
            axis="y",
            alpha=0.20
        )

    # --------------------------------------------------------
    # Reference lines
    # --------------------------------------------------------

    # Rg collapse threshold
    axes[0].axhline(
        4.30,
        linestyle="--",
        linewidth=1.1,
        color="0.4",
        alpha=0.8
    )

    # Delta Rg = zero
    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        color="0.45",
        alpha=0.7
    )

    # kcollapse/kescape = 1
    axes[4].axhline(
        1,
        linestyle="--",
        linewidth=1.0,
        color="0.45",
        alpha=0.7
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    axes[-1].set_xticks(x)

    axes[-1].set_xticklabels(
        summary[
            "architecture_id"
        ].astype(str),
        rotation=90
    )

    axes[-1].set_xlabel(
        "Architecture ID "
        "(ordered by replicate-mean "
        r"$R_g$ at 370 K)"
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    axes[0].legend(
        handles=make_legend_items(),
        loc="best",
        frameon=True,
        ncol=2
    )

    # --------------------------------------------------------
    # Figure title
    # --------------------------------------------------------

    n_reps = int(
        summary[
            "mean_rg_nm_370K_n"
        ].max()
    )

    fig.suptitle(
        "VGIPKG architecture-collapse overview\n"
        f"Points = replicate mean; "
        f"error bars = SD; "
        f"up to n={n_reps} independent simulations",
        fontsize=15,
        y=0.995
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.975]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png = (
        outdir
        /
        "VGIPKG_architecture_overview_6panel.png"
    )

    pdf = (
        outdir
        /
        "VGIPKG_architecture_overview_6panel.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Saved overview:")
    print(png)
    print(pdf)

    return summary


# ============================================================
# SEPARATE SCATTER:
# Rg370 vs kcollapse/kescape
# ============================================================

def make_ratio_scatter(summary, outdir):

    outdir = Path(outdir)

    xcol = "k_ratio_collapse_escape_370K_mean"
    ycol = "mean_rg_nm_370K_mean"

    clean = (
        summary[
            [
                "architecture_id",
                "architecture_type",
                "kg_class",
                "ip_class",
                xcol,
                ycol,
            ]
        ]
        .dropna()
        .copy()
    )

    x = clean[xcol].to_numpy()
    y = clean[ycol].to_numpy()

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    pearson_r, pearson_p = pearsonr(
        x,
        y
    )

    spearman_rho, spearman_p = spearmanr(
        x,
        y
    )

    print()
    print(
        "Rg370 vs kcollapse/kescape:"
    )

    print(
        f"Pearson r = {pearson_r:.4f} "
        f"(p={pearson_p:.4g})"
    )

    print(
        f"Spearman rho = "
        f"{spearman_rho:.4f} "
        f"(p={spearman_p:.4g})"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for _, row in clean.iterrows():

        color = COLORS.get(
            row["kg_class"],
            "0.5"
        )

        marker = MARKERS.get(
            row["ip_class"],
            "o"
        )

        ax.scatter(
            row[xcol],
            row[ycol],
            s=70,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.95,
            zorder=3
        )

        ax.annotate(
            str(
                row["architecture_id"]
            ),
            (
                row[xcol],
                row[ycol]
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8
        )

    # --------------------------------------------------------
    # Linear fit
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        200
    )

    yy = (
        slope * xx
        + intercept
    )

    ax.plot(
        xx,
        yy,
        linestyle="--",
        linewidth=1.2,
        color="0.35"
    )

    # --------------------------------------------------------
    # Correlation text
    # --------------------------------------------------------

    stats_text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman rho = {spearman_rho:.3f}"
    )

    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.9
        )
    )

    ax.set_xlabel(
        r"$k_{\mathrm{collapse}}"
        r"/k_{\mathrm{escape}}$ at 370 K"
    )

    ax.set_ylabel(
        r"Mean $R_g$ at 370 K (nm)"
    )

    ax.set_title(
        "VGIPKG compaction versus kinetic state preference"
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        handles=make_legend_items(),
        loc="best",
        frameon=True
    )

    fig.tight_layout()

        # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png = outdir / "rg370_vs_rate_ratio.png"
    pdf = outdir / "rg370_vs_rate_ratio.pdf"

    print()
    print("Saving scatter to:")
    print(repr(str(png)))
    print(repr(str(pdf)))

    fig.savefig(
        str(png),
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        str(pdf),
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Saved scatter:")
    print(png)
    print(pdf)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Replicate-aware VGIPKG overview "
            "and kinetic analysis."
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more "
            "plotting_summary_cleaned.csv "
            "replicate files."
        )
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    print(
        f"Replicate files supplied: "
        f"{len(args.inputs)}"
    )

    # Load
    combined = load_replicates(
        args.inputs
    )

    # Aggregate
    summary = aggregate_replicates(
        combined
    )

    print(
        f"Architectures found: "
        f"{len(summary)}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save aggregated data
    # --------------------------------------------------------

    summary_csv = (
        outdir
        /
        "VGIPKG_architecture_overview_data.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False
    )

    print(
        f"Saved aggregated data:\n"
        f"{summary_csv}"
    )

    # --------------------------------------------------------
    # Main overview
    # --------------------------------------------------------

    ordered_summary = make_overview_plot(
        summary,
        outdir
    )

    # --------------------------------------------------------
    # Separate kinetic scatter
    # --------------------------------------------------------

    make_ratio_scatter(
        ordered_summary,
        outdir
    )


if __name__ == "__main__":
    main()