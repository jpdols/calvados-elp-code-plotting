#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_design_rule4_thesis_v7.py

DESIGN RULE 4 — DISTRIBUTING CHARGE

Creates THREE thesis figures, preserving the Version-5 visual style.

FIGURE 1 — VGEG + VGKG combined, all 8 architectures
    Same four ratio-trend panels as V5.
    Effects remain relative to EVEN within each library × ratio:
      A  Δ mean Rg vs even
      B  Δ central 90% Rg range vs even
      C  Δ collapsed fraction vs even
      D  log2(k_collapse / k_collapse,even)

    The solid point/line is the mean of VGEG and VGKG.
    A faint vertical line behind every point spans the TWO underlying
    VGEG and VGKG values. It is a min-max chemistry spread, NOT a CI.

FIGURE 2 — Block-size test, averaged across VGEG + VGKG + IGKG
    Only even / small / medium / large blocks.

    IMPORTANT: this figure is NOT relative to EVEN.
    For each library × ratio, the reference is the mean across the four
    block architectures.

      A  Δ mean Rg vs four-block mean
      B  Δ central 90% Rg range vs four-block mean
      C  Δ collapsed fraction vs four-block mean
      D  log2(k_collapse / mean k_collapse of four blocks)
      E  log2(k_escape   / mean k_escape   of four blocks)

    The solid point/line is the mean across all three libraries.
    A faint vertical line spans the three underlying library values.

FIGURE 3 — IGKG alone, all 8 architectures
    Same four-panel FORMAT and SAME metrics as Figure 1 so the two figures
    can be compared directly:
      A  Δ mean Rg vs even
      B  Δ central 90% Rg range vs even
      C  Δ collapsed fraction vs even
      D  log2(k_collapse / k_collapse,even)

STYLE
    - architecture = colour only
    - every marker = circle
    - no shaded regions
    - faint vertical spread lines behind mean points
    - ratio labels R1-R6:
        6:1 (14%), 3:1 (25%), 2:1 (33%),
        1:1 (50%), 1:2 (67%), 1:5 (83%)

Usage:
python plot_design_rule4_thesis_v7.py --indir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\charge_distribution_rg_analysis_v3" --outdir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\thesis_v7"
"""

import argparse
from pathlib import Path
import re

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
    "lines.linewidth": 1.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# ============================================================
# CONSTANTS
# ============================================================

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
    "randomshuffle": "#BDBDBD",
    "even": "#5B9BD5",
    "smallblocks": "#70AD47",
    "mediumblocks": "#8064A2",
    "largeblocks": "#ED7D31",
    "charge_center": "#E74C3C",
    "charge_ends": "#2CB1C9",
    "diblock": "#8C564B",
}

RATIO_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6"]

RATIO_LABELS = {
    "R1": "6:1\n(14%)",
    "R2": "3:1\n(25%)",
    "R3": "2:1\n(33%)",
    "R4": "1:1\n(50%)",
    "R5": "1:2\n(67%)",
    "R6": "1:5\n(83%)",
}


# ============================================================
# BASIC HELPERS
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
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def safe_log2_ratio(value, reference):
    if (
        pd.isna(value)
        or pd.isna(reference)
        or value <= 0
        or reference <= 0
    ):
        return np.nan
    return float(np.log2(value / reference))


# ============================================================
# NORMALIZATION OF LABELS
# ============================================================

def canonical_ratio(value):
    if pd.isna(value):
        return np.nan

    s = str(value).upper().strip()
    match = re.search(r"R\s*([1-6])", s)
    if match:
        return "R" + match.group(1)

    compact = s.replace(" ", "")
    return {
        "6:1": "R1",
        "3:1": "R2",
        "2:1": "R3",
        "1:1": "R4",
        "1:2": "R5",
        "1:5": "R6",
    }.get(compact, s)


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
# LOAD V3 SUMMARIES
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
    if len(finite) > 0 and finite.quantile(0.95) <= 1.01:
        out["collapsed_pct"] = 100.0 * out["collapsed_raw"]
    else:
        out["collapsed_pct"] = out["collapsed_raw"]

    return out.drop(columns="collapsed_raw")


def load_data(indir):
    frames = []

    for library in LIBRARIES:
        path = indir / f"rg_summary_{library}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file:\n{path}")

        raw = pd.read_csv(path)
        standardized = standardize_summary(raw, library)
        frames.append(standardized)

        print(
            f"Loaded {library}: "
            f"{len(standardized)} rows from {path.name}"
        )

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

    # If replicate rows exist, reduce to one summary row before comparisons.
    data = (
        data.groupby(
            ["library", "ratio", "architecture"],
            as_index=False,
        )[value_cols]
        .mean()
    )

    return data


# ============================================================
# EFFECTS RELATIVE TO EVEN
# Used ONLY for Figures 1 and 3
# ============================================================

def calculate_effects_vs_even(data):
    rows = []

    for (library, ratio), group in data.groupby(["library", "ratio"]):
        even_rows = group[group["architecture"] == "even"]

        if len(even_rows) != 1:
            raise ValueError(
                f"{library} {ratio}: expected one even row, "
                f"found {len(even_rows)}"
            )

        even = even_rows.iloc[0]

        for _, row in group.iterrows():
            rows.append({
                "library": library,
                "ratio": ratio,
                "architecture": row["architecture"],

                "delta_mean_rg_nm":
                    row["mean_rg_nm"] - even["mean_rg_nm"],

                "delta_rg_range90_nm":
                    row["rg_range90_nm"] - even["rg_range90_nm"],

                "delta_collapsed_pp":
                    row["collapsed_pct"] - even["collapsed_pct"],

                "log2_kcollapse_vs_even":
                    safe_log2_ratio(
                        row["k_collapse_per_ns"],
                        even["k_collapse_per_ns"],
                    ),
            })

    return pd.DataFrame(rows)


# ============================================================
# BLOCK-SIZE EFFECTS RELATIVE TO THE FOUR-BLOCK AVERAGE
# Used ONLY for Figure 2
# ============================================================

def calculate_block_effects_vs_average(data):
    """
    For each library × ratio, use only:
        even / small / medium / large

    Linear metrics:
        architecture value - mean of the four block architectures

    Kinetic metrics:
        log2(architecture rate / mean rate of the four block architectures)

    Therefore zero means the FOUR-BLOCK AVERAGE, not EVEN.
    """
    subset = data[data["architecture"].isin(BLOCK_ARCHS)].copy()
    rows = []

    for (library, ratio), group in subset.groupby(["library", "ratio"]):
        refs = {
            "mean_rg_nm": group["mean_rg_nm"].mean(),
            "rg_range90_nm": group["rg_range90_nm"].mean(),
            "collapsed_pct": group["collapsed_pct"].mean(),
            "k_collapse_per_ns": group["k_collapse_per_ns"].mean(),
            "k_escape_per_ns": group["k_escape_per_ns"].mean(),
        }

        for _, row in group.iterrows():
            rows.append({
                "library": library,
                "ratio": ratio,
                "architecture": row["architecture"],

                "delta_mean_rg_vs_block_average":
                    row["mean_rg_nm"] - refs["mean_rg_nm"],

                "delta_range90_vs_block_average":
                    row["rg_range90_nm"] - refs["rg_range90_nm"],

                "delta_collapsed_vs_block_average":
                    row["collapsed_pct"] - refs["collapsed_pct"],

                "log2_kcollapse_vs_block_average":
                    safe_log2_ratio(
                        row["k_collapse_per_ns"],
                        refs["k_collapse_per_ns"],
                    ),

                "log2_kescape_vs_block_average":
                    safe_log2_ratio(
                        row["k_escape_per_ns"],
                        refs["k_escape_per_ns"],
                    ),
            })

    return pd.DataFrame(rows)


# ============================================================
# MEAN + MIN/MAX SPREAD ACROSS LIBRARIES
# ============================================================

def summarize_across_libraries(
    table,
    libraries,
    architectures,
    metrics,
):
    """
    For each ratio × architecture:
      mean = arithmetic mean across requested libraries
      min/max = spread of the underlying library values

    Figure 1:
      two values -> VGEG and VGKG

    Figure 2:
      three values -> VGEG, VGKG and IGKG
    """
    subset = table[
        table["library"].isin(libraries)
        & table["architecture"].isin(architectures)
    ].copy()

    rows = []

    for ratio in RATIO_ORDER:
        for arch in architectures:
            sub = subset[
                (subset["ratio"] == ratio)
                & (subset["architecture"] == arch)
            ]

            if sub.empty:
                continue

            rec = {
                "ratio": ratio,
                "architecture": arch,
                "n_libraries": sub["library"].nunique(),
            }

            for metric in metrics:
                vals = sub[metric].dropna().to_numpy(dtype=float)

                if len(vals) == 0:
                    rec[f"{metric}_mean"] = np.nan
                    rec[f"{metric}_min"] = np.nan
                    rec[f"{metric}_max"] = np.nan
                else:
                    rec[f"{metric}_mean"] = float(np.mean(vals))
                    rec[f"{metric}_min"] = float(np.min(vals))
                    rec[f"{metric}_max"] = float(np.max(vals))

            rows.append(rec)

    return pd.DataFrame(rows)


# ============================================================
# RATIO-TREND PANEL WITH FAINT VERTICAL SPREAD
# ============================================================

def plot_ratio_trend_with_spread(
    ax,
    summary,
    architectures,
    metric,
    ylabel,
    heading,
    letter,
):
    x = np.arange(len(RATIO_ORDER))

    ax.axhline(
        0,
        color="0.62",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    for arch in architectures:
        sub = (
            summary[summary["architecture"] == arch]
            .set_index("ratio")
            .reindex(RATIO_ORDER)
        )

        y = sub[f"{metric}_mean"].to_numpy(dtype=float)
        lo = sub[f"{metric}_min"].to_numpy(dtype=float)
        hi = sub[f"{metric}_max"].to_numpy(dtype=float)

        # Faint vertical line BEHIND each mean point.
        # This is the full library min-max spread, not a CI/error estimate.
        for xi, low, high in zip(x, lo, hi):
            if np.isfinite(low) and np.isfinite(high):
                ax.vlines(
                    xi,
                    low,
                    high,
                    color=ARCH_COLORS[arch],
                    alpha=0.24,
                    linewidth=1.5,
                    zorder=1,
                )

        # Same V5 solid line + circle style.
        ax.plot(
            x,
            y,
            color=ARCH_COLORS[arch],
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            label=ARCH_LABELS[arch],
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(
        heading,
        fontweight="bold",
        pad=8,
    )

    panel_letter(ax, letter)

    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABELS[r] for r in RATIO_ORDER])
    ax.set_xlabel("Motif composition (A:C)")

    clean_axis(ax)


def plot_ratio_trend_no_spread(
    ax,
    table,
    architectures,
    metric,
    ylabel,
    heading,
    letter,
):
    """Used for IGKG alone: same visual style, one library so no chemistry spread."""
    x = np.arange(len(RATIO_ORDER))

    ax.axhline(
        0,
        color="0.62",
        linestyle="--",
        linewidth=0.9,
        zorder=0,
    )

    for arch in architectures:
        sub = (
            table[table["architecture"] == arch]
            .set_index("ratio")
            .reindex(RATIO_ORDER)
        )

        y = sub[metric].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=ARCH_COLORS[arch],
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            label=ARCH_LABELS[arch],
            zorder=3,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(
        heading,
        fontweight="bold",
        pad=8,
    )

    panel_letter(ax, letter)

    ax.set_xticks(x)
    ax.set_xticklabels([RATIO_LABELS[r] for r in RATIO_ORDER])
    ax.set_xlabel("Motif composition (A:C)")

    clean_axis(ax)


def architecture_handles(architectures):
    return [
        Line2D(
            [0], [0],
            color=ARCH_COLORS[a],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            markersize=5.5,
            label=ARCH_LABELS[a],
        )
        for a in architectures
    ]


# ============================================================
# FIGURE 1 — VGEG + VGKG COMBINED
# V5 structure + faint chemistry spread
# ============================================================

def make_combined_vgeg_vgkg(effects, outdir):
    metrics = [
        "delta_mean_rg_nm",
        "delta_rg_range90_nm",
        "delta_collapsed_pp",
        "log2_kcollapse_vs_even",
    ]

    summary = summarize_across_libraries(
        effects,
        libraries=["VGEG", "VGKG"],
        architectures=ARCH_ORDER,
        metrics=metrics,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.3, 7.5),
    )

    panels = [
        (
            "delta_mean_rg_nm",
            r"$\Delta$ mean $R_g$ vs even (nm)",
            "Conformational dimensions",
        ),
        (
            "delta_rg_range90_nm",
            r"$\Delta$ central 90% $R_g$ range vs even (nm)",
            "Conformational range",
        ),
        (
            "delta_collapsed_pp",
            r"$\Delta$ collapsed fraction vs even (pp)",
            "Collapsed population",
        ),
        (
            "log2_kcollapse_vs_even",
            r"$\log_2(k_{\mathrm{collapse}}/k_{\mathrm{collapse,even}})$",
            "Compact-state entry",
        ),
    ]

    for i, (metric, ylabel, heading) in enumerate(panels):
        plot_ratio_trend_with_spread(
            axes.ravel()[i],
            summary,
            ARCH_ORDER,
            metric,
            ylabel,
            heading,
            chr(ord("A") + i),
        )

    fig.legend(
        handles=architecture_handles(ARCH_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        title="Architecture",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.93,
        bottom=0.18,
        wspace=0.30,
        hspace=0.38,
    )

    save_figure(
        fig,
        outdir,
        "01_VGEG_VGKG_combined_effects_with_spread",
    )

    summary.to_csv(
        outdir / "01_VGEG_VGKG_combined_effects_with_spread.csv",
        index=False,
    )


# ============================================================
# FIGURE 2 — BLOCK SIZE, AVERAGED ACROSS ALL THREE LIBRARIES
# Reference = average of the four block architectures, NOT EVEN
# ============================================================

def make_block_size_average(block_effects, outdir):
    metrics = [
        "delta_mean_rg_vs_block_average",
        "delta_range90_vs_block_average",
        "delta_collapsed_vs_block_average",
        "log2_kcollapse_vs_block_average",
        "log2_kescape_vs_block_average",
    ]

    summary = summarize_across_libraries(
        block_effects,
        libraries=["VGEG", "VGKG", "IGKG"],
        architectures=BLOCK_ARCHS,
        metrics=metrics,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 7.6),
    )

    panels = [
        (
            "delta_range90_vs_block_average",
            r"$\Delta$ central 90% $R_g$ range vs block-size mean (nm)",
            "Conformational range",
        ),
        (
            "log2_kcollapse_vs_block_average",
            r"$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$",
            "Compact-state entry",
        ),
        (
            "log2_kescape_vs_block_average",
            r"$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$",
            "Compact-state escape",
        ),
        (
            "delta_mean_rg_vs_block_average",
            r"$\Delta$ mean $R_g$ vs block-size mean (nm)",
            "Mean dimensions",
        ),
        (
            "delta_collapsed_vs_block_average",
            r"$\Delta$ collapsed fraction vs block-size mean (pp)",
            "Collapsed population",
        ),
    ]

    flat = axes.ravel()

    for i, (metric, ylabel, heading) in enumerate(panels):
        plot_ratio_trend_with_spread(
            flat[i],
            summary,
            BLOCK_ARCHS,
            metric,
            ylabel,
            heading,
            chr(ord("A") + i),
        )

    flat[5].axis("off")

    flat[5].legend(
        handles=architecture_handles(BLOCK_ARCHS),
        loc="center left",
        frameon=False,
        title="Architecture",
        fontsize=10,
        title_fontsize=10,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.10,
        wspace=0.34,
        hspace=0.38,
    )

    save_figure(
        fig,
        outdir,
        "02_block_size_average_reference",
    )

    summary.to_csv(
        outdir / "02_block_size_average_reference.csv",
        index=False,
    )


# ============================================================
# FIGURE 3 — IGKG ALONE, SAME EFFECT PLOT AS FIGURE 1
# ============================================================

def make_igkg_effects(effects, outdir):
    igkg = effects[
        effects["library"] == "IGKG"
    ].copy()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.3, 7.5),
    )

    panels = [
        (
            "delta_mean_rg_nm",
            r"$\Delta$ mean $R_g$ vs even (nm)",
            "Conformational dimensions",
        ),
        (
            "delta_rg_range90_nm",
            r"$\Delta$ central 90% $R_g$ range vs even (nm)",
            "Conformational range",
        ),
        (
            "delta_collapsed_pp",
            r"$\Delta$ collapsed fraction vs even (pp)",
            "Collapsed population",
        ),
        (
            "log2_kcollapse_vs_even",
            r"$\log_2(k_{\mathrm{collapse}}/k_{\mathrm{collapse,even}})$",
            "Compact-state entry",
        ),
    ]

    for i, (metric, ylabel, heading) in enumerate(panels):
        plot_ratio_trend_no_spread(
            axes.ravel()[i],
            igkg,
            ARCH_ORDER,
            metric,
            ylabel,
            heading,
            chr(ord("A") + i),
        )

    fig.legend(
        handles=architecture_handles(ARCH_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        title="Architecture",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.93,
        bottom=0.18,
        wspace=0.30,
        hspace=0.38,
    )

    save_figure(
        fig,
        outdir,
        "03_IGKG_effects_same_style_as_combined",
    )

    igkg.to_csv(
        outdir / "03_IGKG_effects_same_style_as_combined.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Design Rule 4 thesis figures — V7 — DR3-style block-size analysis."
        )
    )

    parser.add_argument(
        "--indir",
        required=True,
        type=Path,
        help=(
            "Folder containing rg_summary_VGEG.csv, "
            "rg_summary_VGKG.csv and rg_summary_IGKG.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output folder; created automatically if needed.",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DESIGN RULE 4 — DISTRIBUTING CHARGE — V7")
    print("=" * 78)
    print(f"Input : {args.indir}")
    print(f"Output: {args.outdir}")
    print()

    data = load_data(args.indir)

    print()
    print("Rows after standardization:", len(data))

    counts = (
        data.groupby(["library", "ratio"])["architecture"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=LIBRARIES, columns=RATIO_ORDER)
    )

    print()
    print("Architectures per library / ratio:")
    print(counts.to_string())

    # Figures 1 and 3: same V5 relative-to-even logic.
    effects_even = calculate_effects_vs_even(data)

    effects_even.to_csv(
        args.outdir / "00_effects_relative_to_even_figures_1_and_3.csv",
        index=False,
    )

    # Figure 2: NEW reference = average of even/small/medium/large.
    block_effects = calculate_block_effects_vs_average(data)

    block_effects.to_csv(
        args.outdir / "00_block_effects_relative_to_block_average.csv",
        index=False,
    )

    print()
    print("Creating Figure 1...")
    make_combined_vgeg_vgkg(
        effects_even,
        args.outdir,
    )

    print()
    print("Creating Figure 2...")
    make_block_size_average(
        block_effects,
        args.outdir,
    )

    print()
    print("Creating Figure 3...")
    make_igkg_effects(
        effects_even,
        args.outdir,
    )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    print(
        "Figure 1: same V5 VGEG+VGKG effect plot; "
        "faint lines show the two underlying chemistry values."
    )
    print(
        "Figure 2: block-size effects referenced to the mean of "
        "even/small/medium/large, NOT to even."
    )
    print(
        "Figure 2 kinetics: log2(k / mean k) calculated separately "
        "for k_collapse and k_escape."
    )
    print(
        "Figure 3: IGKG alone in the SAME four-panel effect format "
        "as Figure 1."
    )
    print(
        "All architecture markers are circles; architecture is encoded "
        "by colour only."
    )


if __name__ == "__main__":
    main()
