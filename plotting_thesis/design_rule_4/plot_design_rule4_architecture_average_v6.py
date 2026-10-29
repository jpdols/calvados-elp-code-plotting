#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_design_rule4_architecture_average_v6.py

Design Rule 4 summary in the requested "one point per architecture" style.

IMPORTANT:
- Ratios R1-R6 are NOT connected on the x-axis.
- For every library and ratio, each architecture is first compared with the
  mean across the architectures included in that figure.
- The six ratio-specific effects are then averaged to ONE point per architecture.
- The light vertical line behind each point shows the full R1-R6 spread
  (minimum to maximum), not a confidence interval.
- All points are CIRCLES. Architecture is encoded by COLOUR ONLY.
- Kinetics use:
      log2(k_arch / mean(k across architectures at the same ratio))
  so there is NO division by "even".

Outputs:
01_VGEG_VGKG_all_architectures_average.*
    VGEG and VGKG are pooled after within-ratio normalization.
    One point per architecture + light min-max ratio spread.

02_IGKG_all_architectures_average.*
    IGKG alone in exactly the same style, for direct visual comparison.

03_block_size_comparison.*
    Only Even / Small / Medium / Large.
    Two horizontally offset points per architecture:
      - VGEG + VGKG combined
      - IGKG alone
    Both are ratio-averaged with light min-max ratio spread.
    All markers remain circles.

Usage:
python plot_design_rule4_architecture_average_v6.py --indir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\charge_distribution_rg_analysis_v3" --outdir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\architecture_average_v6"
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


LIBRARIES = ["VGEG", "VGKG", "IGKG"]

ARCH_ORDER = [
    "randomshuffle",
    "even",
    "smallblocks",
    "mediumblocks",
    "largeblocks",
    "charge_center",
    "charge_ends",
    "diblock",
]

BLOCK_ARCHS = [
    "even",
    "smallblocks",
    "mediumblocks",
    "largeblocks",
]

ARCH_LABELS = {
    "randomshuffle": "Random shuffle",
    "even": "Even",
    "smallblocks": "Small blocks",
    "mediumblocks": "Medium blocks",
    "largeblocks": "Large blocks",
    "charge_center": "Charge centre",
    "charge_ends": "Charge ends",
    "diblock": "Diblock",
}

ARCH_COLORS = {
    "randomshuffle": "#A6A6A6",
    "even": "#4472C4",
    "smallblocks": "#70AD47",
    "mediumblocks": "#8064A2",
    "largeblocks": "#ED7D31",
    "charge_center": "#E02F2F",
    "charge_ends": "#22B5C7",
    "diblock": "#8C564B",
}

RATIO_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6"]


# ============================================================
# HELPERS
# ============================================================

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9)


def panel_letter(ax, letter):
    ax.text(
        -0.13, 1.04, letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig, outdir, basename):
    png = outdir / f"{basename}.png"
    pdf = outdir / f"{basename}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def first_existing(df, candidates, description):
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    raise ValueError(
        f"Could not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def canonical_ratio(value):
    if pd.isna(value):
        return np.nan
    s = str(value).upper().strip()
    m = re.search(r"R\s*([1-6])", s)
    if m:
        return "R" + m.group(1)
    s2 = s.replace(" ", "")
    return {
        "6:1": "R1",
        "3:1": "R2",
        "2:1": "R3",
        "1:1": "R4",
        "1:2": "R5",
        "1:5": "R6",
    }.get(s2, s)


def canonical_architecture(value):
    if pd.isna(value):
        return np.nan

    s = str(value).lower().strip()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)

    if "random" in s or "shuffle" in s:
        return "randomshuffle"
    if "diblock" in s:
        return "diblock"
    if "small" in s:
        return "smallblocks"
    if "medium" in s:
        return "mediumblocks"
    if "large" in s:
        return "largeblocks"
    if "even" in s or "alternat" in s:
        return "even"
    if "center" in s or "centre" in s:
        return "charge_center"
    if (
        "endcap" in s
        or "charge_ends" in s
        or "charged_ends" in s
        or s.endswith("_ends")
        or "egends" in s
        or "kgends" in s
        or "eg_ends" in s
        or "kg_ends" in s
    ):
        return "charge_ends"

    return s


# ============================================================
# LOAD DATA
# ============================================================

def standardize_summary(raw, library):
    ratio_col = first_existing(
        raw,
        ["ratio", "ratio_norm", "composition_ratio", "ratio_label"],
        "ratio column",
    )
    arch_col = first_existing(
        raw,
        ["architecture", "architecture_type", "architecture_class", "arch_norm"],
        "architecture column",
    )
    mean_rg_col = first_existing(
        raw,
        ["mean_Rg", "mean_rg_nm", "mean_rg", "rg_mean_nm"],
        "mean Rg column",
    )
    range90_col = first_existing(
        raw,
        ["Rg_range90", "rg_range90_nm", "rg_range90"],
        "central 90% Rg range column",
    )
    collapse_col = first_existing(
        raw,
        [
            "collapsed_fraction",
            "collapse_fraction",
            "fraction_collapsed",
            "collapsed_percent",
            "collapse_percent",
        ],
        "collapsed fraction column",
    )
    kcollapse_col = first_existing(
        raw,
        ["k_collapse_per_ns", "k_collapse", "collapse_rate_per_ns"],
        "k_collapse column",
    )
    kescape_col = first_existing(
        raw,
        ["k_escape_per_ns", "k_escape", "escape_rate_per_ns"],
        "k_escape column",
    )

    out = pd.DataFrame({
        "library": library,
        "ratio": raw[ratio_col].apply(canonical_ratio),
        "architecture": raw[arch_col].apply(canonical_architecture),
        "mean_rg_nm": pd.to_numeric(raw[mean_rg_col], errors="coerce"),
        "rg_range90_nm": pd.to_numeric(raw[range90_col], errors="coerce"),
        "collapsed_raw": pd.to_numeric(raw[collapse_col], errors="coerce"),
        "k_collapse_per_ns": pd.to_numeric(raw[kcollapse_col], errors="coerce"),
        "k_escape_per_ns": pd.to_numeric(raw[kescape_col], errors="coerce"),
    })

    finite = out["collapsed_raw"].dropna()
    if len(finite) and finite.quantile(0.95) <= 1.01:
        out["collapsed_pct"] = 100.0 * out["collapsed_raw"]
    else:
        out["collapsed_pct"] = out["collapsed_raw"]

    return out.drop(columns="collapsed_raw")


def load_data(indir):
    frames = []

    for library in LIBRARIES:
        path = indir / f"rg_summary_{library}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        raw = pd.read_csv(path)
        df = standardize_summary(raw, library)
        frames.append(df)
        print(f"Loaded {library}: {len(df)} rows")

    data = pd.concat(frames, ignore_index=True)

    data = data[
        data["ratio"].isin(RATIO_ORDER)
        & data["architecture"].isin(ARCH_ORDER)
    ].copy()

    value_cols = [
        "mean_rg_nm",
        "rg_range90_nm",
        "collapsed_pct",
        "k_collapse_per_ns",
        "k_escape_per_ns",
    ]

    # If the CSV contains replicate rows, average them first.
    data = (
        data.groupby(
            ["library", "ratio", "architecture"],
            as_index=False,
        )[value_cols]
        .mean()
    )

    return data


# ============================================================
# NORMALIZE TO THE ARCHITECTURE MEAN AT EACH RATIO
# ============================================================

def effects_against_architecture_average(data, architectures):
    """
    For each library × ratio:
      linear metrics:
          architecture value - mean across requested architectures

      kinetic metrics:
          log2(architecture k / mean k across requested architectures)

    Thus ZERO means "same as the architecture average at this ratio".
    Nothing is referenced to EVEN.
    """

    d = data[data["architecture"].isin(architectures)].copy()
    rows = []

    for (library, ratio), group in d.groupby(["library", "ratio"]):

        mean_rg_ref = group["mean_rg_nm"].mean()
        range_ref = group["rg_range90_nm"].mean()
        collapse_ref = group["collapsed_pct"].mean()
        kcollapse_ref = group["k_collapse_per_ns"].mean()
        kescape_ref = group["k_escape_per_ns"].mean()

        for _, row in group.iterrows():

            if (
                pd.notna(row["k_collapse_per_ns"])
                and pd.notna(kcollapse_ref)
                and row["k_collapse_per_ns"] > 0
                and kcollapse_ref > 0
            ):
                log2_kcollapse = np.log2(
                    row["k_collapse_per_ns"] / kcollapse_ref
                )
            else:
                log2_kcollapse = np.nan

            if (
                pd.notna(row["k_escape_per_ns"])
                and pd.notna(kescape_ref)
                and row["k_escape_per_ns"] > 0
                and kescape_ref > 0
            ):
                log2_kescape = np.log2(
                    row["k_escape_per_ns"] / kescape_ref
                )
            else:
                log2_kescape = np.nan

            rows.append({
                "library": library,
                "ratio": ratio,
                "architecture": row["architecture"],

                "delta_mean_rg_nm":
                    row["mean_rg_nm"] - mean_rg_ref,

                "delta_rg_range90_nm":
                    row["rg_range90_nm"] - range_ref,

                "delta_collapsed_pp":
                    row["collapsed_pct"] - collapse_ref,

                "log2_kcollapse_vs_average":
                    log2_kcollapse,

                "log2_kescape_vs_average":
                    log2_kescape,
            })

    return pd.DataFrame(rows)


# ============================================================
# COLLAPSE RATIOS TO ONE POINT PER ARCHITECTURE
# ============================================================

METRICS = [
    "delta_mean_rg_nm",
    "delta_rg_range90_nm",
    "delta_collapsed_pp",
    "log2_kcollapse_vs_average",
    "log2_kescape_vs_average",
]


def ratio_average_summary(effects, libraries, architectures):
    """
    Pool the requested libraries and all six ratio-specific effects.

    For each architecture and metric:
      point = arithmetic mean across all library × ratio values
      light line = full minimum-to-maximum spread

    For VGEG+VGKG this means 12 normalized observations per architecture
    when all six ratios are present.
    For IGKG this means 6 observations per architecture.
    """

    d = effects[
        effects["library"].isin(libraries)
        & effects["architecture"].isin(architectures)
    ].copy()

    rows = []

    for arch in architectures:
        sub = d[d["architecture"] == arch]

        record = {
            "architecture": arch,
            "n_rows": len(sub),
        }

        for metric in METRICS:
            vals = sub[metric].dropna().to_numpy(dtype=float)

            record[f"{metric}_mean"] = (
                float(np.mean(vals)) if len(vals) else np.nan
            )
            record[f"{metric}_min"] = (
                float(np.min(vals)) if len(vals) else np.nan
            )
            record[f"{metric}_max"] = (
                float(np.max(vals)) if len(vals) else np.nan
            )
            record[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            )

        rows.append(record)

    return pd.DataFrame(rows)


# ============================================================
# DRAW ONE SUMMARY PANEL
# ============================================================

def draw_architecture_summary_panel(
    ax,
    summary,
    architectures,
    metric,
    ylabel,
    title,
    letter,
):
    x = np.arange(len(architectures))

    ax.axhline(
        0,
        color="0.62",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    for i, arch in enumerate(architectures):
        row = summary[summary["architecture"] == arch].iloc[0]

        mean = row[f"{metric}_mean"]
        lo = row[f"{metric}_min"]
        hi = row[f"{metric}_max"]

        # Light line behind the point = FULL R1-R6 spread.
        ax.vlines(
            i,
            lo,
            hi,
            color=ARCH_COLORS[arch],
            alpha=0.24,
            linewidth=3.0,
            zorder=1,
        )

        # Small end caps make the spread readable without looking like
        # statistical error bars.
        cap = 0.075
        ax.hlines(
            [lo, hi],
            i - cap,
            i + cap,
            color=ARCH_COLORS[arch],
            alpha=0.24,
            linewidth=1.4,
            zorder=1,
        )

        # ONE marker shape everywhere: circle.
        ax.scatter(
            i,
            mean,
            s=48,
            marker="o",
            color=ARCH_COLORS[arch],
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ARCH_LABELS[a] for a in architectures],
        rotation=28,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)
    panel_letter(ax, letter)
    clean_axis(ax)


# ============================================================
# FIVE-PANEL SINGLE-GROUP FIGURE
# ============================================================

def make_five_panel_summary(
    summary,
    architectures,
    outdir,
    basename,
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.6, 7.6),
    )

    panels = [
        (
            "delta_rg_range90_nm",
            r"$\Delta$ central 90% $R_g$ range vs architecture mean (nm)",
            "Conformational range",
        ),
        (
            "log2_kcollapse_vs_average",
            r"$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$",
            "Compact-state entry",
        ),
        (
            "log2_kescape_vs_average",
            r"$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$",
            "Compact-state escape",
        ),
        (
            "delta_mean_rg_nm",
            r"$\Delta$ mean $R_g$ vs architecture mean (nm)",
            "Mean dimensions",
        ),
        (
            "delta_collapsed_pp",
            r"$\Delta$ collapsed fraction vs architecture mean (pp)",
            "Collapsed population",
        ),
    ]

    flat = axes.ravel()

    for i, (metric, ylabel, title) in enumerate(panels):
        draw_architecture_summary_panel(
            flat[i],
            summary,
            architectures,
            metric,
            ylabel,
            title,
            chr(ord("A") + i),
        )

    flat[5].axis("off")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ARCH_COLORS[a],
            markeredgecolor="white",
            markersize=7,
            label=ARCH_LABELS[a],
        )
        for a in architectures
    ]

    flat[5].legend(
        handles=handles,
        loc="center left",
        frameon=False,
        title="Architecture",
        ncol=2 if len(architectures) > 4 else 1,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.12,
        wspace=0.36,
        hspace=0.46,
    )

    save_figure(fig, outdir, basename)


# ============================================================
# BLOCK-SIZE DIRECT COMPARISON:
# VGEG+VGKG COMBINED VERSUS IGKG
# ============================================================

def draw_block_comparison_panel(
    ax,
    combined,
    igkg,
    metric,
    ylabel,
    title,
    letter,
):
    x = np.arange(len(BLOCK_ARCHS))
    offset = 0.105

    ax.axhline(
        0,
        color="0.62",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    # Both groups use circles.
    # Architecture = colour.
    # The only distinction between combined and IGKG is horizontal position
    # plus filled versus open circle, NOT marker shape.
    for i, arch in enumerate(BLOCK_ARCHS):

        for summary, dx, filled in [
            (combined, -offset, True),
            (igkg, +offset, False),
        ]:
            row = summary[
                summary["architecture"] == arch
            ].iloc[0]

            mean = row[f"{metric}_mean"]
            lo = row[f"{metric}_min"]
            hi = row[f"{metric}_max"]

            xpos = i + dx

            ax.vlines(
                xpos,
                lo,
                hi,
                color=ARCH_COLORS[arch],
                alpha=0.22,
                linewidth=2.7,
                zorder=1,
            )

            cap = 0.055
            ax.hlines(
                [lo, hi],
                xpos - cap,
                xpos + cap,
                color=ARCH_COLORS[arch],
                alpha=0.22,
                linewidth=1.2,
                zorder=1,
            )

            if filled:
                face = ARCH_COLORS[arch]
                edge = "white"
                lw = 0.65
            else:
                face = "white"
                edge = ARCH_COLORS[arch]
                lw = 1.8

            ax.scatter(
                xpos,
                mean,
                s=50,
                marker="o",
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ARCH_LABELS[a] for a in BLOCK_ARCHS],
        rotation=22,
        ha="right",
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)
    panel_letter(ax, letter)
    clean_axis(ax)


def make_block_comparison(combined, igkg, outdir):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.6, 7.6),
    )

    panels = [
        (
            "delta_rg_range90_nm",
            r"$\Delta$ central 90% $R_g$ range vs block-size mean (nm)",
            "Conformational range",
        ),
        (
            "log2_kcollapse_vs_average",
            r"$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$",
            "Compact-state entry",
        ),
        (
            "log2_kescape_vs_average",
            r"$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$",
            "Compact-state escape",
        ),
        (
            "delta_mean_rg_nm",
            r"$\Delta$ mean $R_g$ vs block-size mean (nm)",
            "Mean dimensions",
        ),
        (
            "delta_collapsed_pp",
            r"$\Delta$ collapsed fraction vs block-size mean (pp)",
            "Collapsed population",
        ),
    ]

    flat = axes.ravel()

    for i, (metric, ylabel, title) in enumerate(panels):
        draw_block_comparison_panel(
            flat[i],
            combined,
            igkg,
            metric,
            ylabel,
            title,
            chr(ord("A") + i),
        )

    flat[5].axis("off")

    # Architecture legend: colour only, every marker is a circle.
    architecture_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=ARCH_COLORS[a],
            markeredgecolor="white",
            markersize=7,
            label=ARCH_LABELS[a],
        )
        for a in BLOCK_ARCHS
    ]

    # Dataset distinction uses fill, still the SAME circle marker.
    group_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.35",
            markeredgecolor="0.35",
            markersize=7,
            label="VGEG + VGKG",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="0.35",
            markeredgewidth=1.6,
            markersize=7,
            label="IGKG",
        ),
    ]

    leg1 = flat[5].legend(
        handles=architecture_handles,
        loc="upper left",
        frameon=False,
        title="Architecture",
    )
    flat[5].add_artist(leg1)

    flat[5].legend(
        handles=group_handles,
        loc="lower left",
        frameon=False,
        title="Library summary",
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.12,
        wspace=0.36,
        hspace=0.46,
    )

    save_figure(
        fig,
        outdir,
        "03_block_size_combined_vs_IGKG",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        required=True,
        type=Path,
        help="Folder containing rg_summary_VGEG.csv, rg_summary_VGKG.csv and rg_summary_IGKG.csv",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DESIGN RULE 4 — RATIO-AVERAGED ARCHITECTURE SUMMARY")
    print("=" * 78)
    print(f"Input : {args.indir}")
    print(f"Output: {args.outdir}")
    print()

    data = load_data(args.indir)

    # --------------------------------------------------------
    # ALL 8 ARCHITECTURES:
    # normalize against the 8-architecture mean at each ratio
    # --------------------------------------------------------
    effects_all = effects_against_architecture_average(
        data,
        ARCH_ORDER,
    )

    combined_all = ratio_average_summary(
        effects_all,
        libraries=["VGEG", "VGKG"],
        architectures=ARCH_ORDER,
    )

    igkg_all = ratio_average_summary(
        effects_all,
        libraries=["IGKG"],
        architectures=ARCH_ORDER,
    )

    combined_all.to_csv(
        args.outdir / "01_VGEG_VGKG_all_architectures_average.csv",
        index=False,
    )
    igkg_all.to_csv(
        args.outdir / "02_IGKG_all_architectures_average.csv",
        index=False,
    )

    make_five_panel_summary(
        combined_all,
        ARCH_ORDER,
        args.outdir,
        "01_VGEG_VGKG_all_architectures_average",
    )

    make_five_panel_summary(
        igkg_all,
        ARCH_ORDER,
        args.outdir,
        "02_IGKG_all_architectures_average",
    )

    # --------------------------------------------------------
    # BLOCK SIZE ONLY:
    # IMPORTANT: normalize again, now against the mean of ONLY
    # even/small/medium/large at each ratio.
    # --------------------------------------------------------
    effects_blocks = effects_against_architecture_average(
        data,
        BLOCK_ARCHS,
    )

    combined_blocks = ratio_average_summary(
        effects_blocks,
        libraries=["VGEG", "VGKG"],
        architectures=BLOCK_ARCHS,
    )

    igkg_blocks = ratio_average_summary(
        effects_blocks,
        libraries=["IGKG"],
        architectures=BLOCK_ARCHS,
    )

    combined_blocks.to_csv(
        args.outdir / "03_block_size_VGEG_VGKG_average.csv",
        index=False,
    )
    igkg_blocks.to_csv(
        args.outdir / "03_block_size_IGKG_average.csv",
        index=False,
    )

    make_block_comparison(
        combined_blocks,
        igkg_blocks,
        args.outdir,
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print("Interpretation of each point:")
    print("  dot        = mean effect across R1-R6")
    print("  light line = full min-max spread across R1-R6")
    print()
    print("Kinetics:")
    print("  log2(k_arch / mean k at the same ratio)")
    print("  0 = architecture-average kinetics")
    print("  +1 = 2x the architecture-average rate")
    print("  -1 = 0.5x the architecture-average rate")
    print()
    print("NO metric is referenced to EVEN.")
    print("ALL marker shapes are circles; architecture is encoded by colour.")


if __name__ == "__main__":
    main()
