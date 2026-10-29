#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_design_rule4_thesis_v5.py

DESIGN RULE 4 — DISTRIBUTING CHARGE

Creates THREE thesis figures.

1) VGEG + VGKG COMBINED — all 8 architectures
   The two libraries are averaged directly at every architecture × ratio.
   NO shaded region.

   A  Δ mean Rg vs even
   B  Δ central 90% Rg range vs even
   C  Δ collapsed fraction vs even (percentage points)
   D  log2(k_collapse / k_collapse,even)

2) BLOCK-SIZE TEST — VGEG + VGKG + IGKG AVERAGED
   Only even / small / medium / large blocks.
   The three libraries are averaged directly at every architecture × ratio.
   NO shading and NO separate chemistry traces.

   A  Δ mean Rg vs even
   B  Δ central 90% Rg range vs even
   C  Δ collapsed fraction vs even (percentage points)
   D  log2(k_collapse / k_collapse,even)
   E  log2(k_escape / k_escape,even)

3) IGKG ABSOLUTE METRICS — all 8 architectures
   IGKG shown on its own, matching the previous absolute-metric figure style.

   A  mean Rg
   B  central 90% Rg range
   C  collapsed fraction
   D  k_collapse
   E  k_escape

All ratio labels:
   R1 = 6:1 = 14%
   R2 = 3:1 = 25%
   R3 = 2:1 = 33%
   R4 = 1:1 = 50%
   R5 = 1:2 = 67%
   R6 = 1:5 = 83%

Usage:
python plot_design_rule4_thesis_v5.py --indir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\charge_distribution_rg_analysis_v3" --outdir "E:\\Master thesis\\calvados_tea\\plotting_thesis\\design_rule_4\\thesis_v5"
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

# Distinct architecture markers for the absolute IGKG figure,
# matching the style of the earlier absolute-metric plot.
ARCH_MARKERS = {
    "randomshuffle": "D",
    "even": "o",
    "smallblocks": "s",
    "mediumblocks": "^",
    "largeblocks": "v",
    "charge_center": "o",
    "charge_ends": "^",
    "diblock": "D",
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
        -0.13,
        1.04,
        letter,
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

    # Safety: if replicate rows exist, collapse them to one summary row
    # before matched comparisons.
    value_cols = [
        "mean_rg_nm",
        "rg_range90_nm",
        "collapsed_pct",
        "k_collapse_per_ns",
        "k_escape_per_ns",
    ]

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
# ============================================================

def calculate_effects(data):
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

                "log2_kescape_vs_even":
                    safe_log2_ratio(
                        row["k_escape_per_ns"],
                        even["k_escape_per_ns"],
                    ),
            })

    return pd.DataFrame(rows)


# ============================================================
# SIMPLE AVERAGING ACROSS LIBRARIES
# ============================================================

def average_effects(effects, libraries, architectures, metrics):
    """
    Arithmetic mean across the requested libraries for every
    ratio × architecture. No band, no CI, no separate traces.
    """
    subset = effects[
        effects["library"].isin(libraries)
        & effects["architecture"].isin(architectures)
    ].copy()

    grouped = (
        subset.groupby(
            ["ratio", "architecture"],
            as_index=False,
        )[metrics]
        .mean()
    )

    return grouped


# ============================================================
# GENERIC RATIO-TREND PLOT
# ============================================================

def plot_ratio_trend_panel(
    ax,
    table,
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
    ax.set_xlabel("Motif composition A:C")

    clean_axis(ax)


# ============================================================
# FIGURE 1 — VGEG + VGKG COMBINED
# ============================================================

def make_combined_vgeg_vgkg(effects, outdir):
    metrics = [
        "delta_mean_rg_nm",
        "delta_rg_range90_nm",
        "delta_collapsed_pp",
        "log2_kcollapse_vs_even",
    ]

    avg = average_effects(
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
        plot_ratio_trend_panel(
            axes.ravel()[i],
            avg,
            ARCH_ORDER,
            metric,
            ylabel,
            heading,
            chr(ord("A") + i),
        )

    handles = [
        Line2D(
            [0], [0],
            color=ARCH_COLORS[a],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=ARCH_LABELS[a],
        )
        for a in ARCH_ORDER
    ]

    fig.legend(
        handles=handles,
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
        "01_VGEG_VGKG_combined_effects",
    )

    avg.to_csv(
        outdir / "01_VGEG_VGKG_combined_effects.csv",
        index=False,
    )


# ============================================================
# FIGURE 2 — BLOCK SIZE, AVERAGED ACROSS ALL THREE LIBRARIES
# ============================================================

def make_block_size_average(effects, outdir):
    metrics = [
        "delta_mean_rg_nm",
        "delta_rg_range90_nm",
        "delta_collapsed_pp",
        "log2_kcollapse_vs_even",
        "log2_kescape_vs_even",
    ]

    avg = average_effects(
        effects,
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
        (
            "log2_kescape_vs_even",
            r"$\log_2(k_{\mathrm{escape}}/k_{\mathrm{escape,even}})$",
            "Compact-state escape",
        ),
    ]

    flat = axes.ravel()

    for i, (metric, ylabel, heading) in enumerate(panels):
        plot_ratio_trend_panel(
            flat[i],
            avg,
            BLOCK_ARCHS,
            metric,
            ylabel,
            heading,
            chr(ord("A") + i),
        )

    # Last cell used for the legend, exactly to keep the 5-panel layout clean.
    flat[5].axis("off")

    handles = [
        Line2D(
            [0], [0],
            color=ARCH_COLORS[a],
            marker="o",
            linewidth=1.8,
            markersize=6,
            label=ARCH_LABELS[a],
        )
        for a in BLOCK_ARCHS
    ]

    flat[5].legend(
        handles=handles,
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
        "02_block_size_average_VGEG_VGKG_IGKG",
    )

    avg.to_csv(
        outdir / "02_block_size_average_VGEG_VGKG_IGKG.csv",
        index=False,
    )


# ============================================================
# FIGURE 3 — IGKG ABSOLUTE METRICS
# ============================================================

def make_igkg_absolute(data, outdir):
    igkg = data[data["library"] == "IGKG"].copy()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 7.6),
    )

    flat = axes.ravel()
    x = np.arange(len(RATIO_ORDER))

    panels = [
        (
            "mean_rg_nm",
            r"Mean $R_g$ (nm)",
            "Mean dimensions",
        ),
        (
            "rg_range90_nm",
            r"Central 90% $R_g$ range (nm)",
            "Conformational range",
        ),
        (
            "collapsed_pct",
            "Collapsed fraction (%)",
            "Collapsed population",
        ),
        (
            "k_collapse_per_ns",
            r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
            "Compact-state entry",
        ),
        (
            "k_escape_per_ns",
            r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
            "Compact-state escape",
        ),
    ]

    for i, (metric, ylabel, heading) in enumerate(panels):
        ax = flat[i]

        for arch in ARCH_ORDER:
            sub = (
                igkg[igkg["architecture"] == arch]
                .set_index("ratio")
                .reindex(RATIO_ORDER)
            )

            y = sub[metric].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                color=ARCH_COLORS[arch],
                marker=ARCH_MARKERS[arch],
                markersize=5.3,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=1.7,
                label=ARCH_LABELS[arch],
            )

        ax.set_ylabel(ylabel)
        ax.set_title(
            heading,
            fontweight="bold",
            pad=8,
        )

        panel_letter(
            ax,
            chr(ord("A") + i),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [RATIO_LABELS[r] for r in RATIO_ORDER]
        )
        ax.set_xlabel("Motif composition A:C")

        clean_axis(ax)

    flat[5].axis("off")

    handles = [
        Line2D(
            [0], [0],
            color=ARCH_COLORS[a],
            marker=ARCH_MARKERS[a],
            linewidth=1.7,
            markersize=6,
            label=ARCH_LABELS[a],
        )
        for a in ARCH_ORDER
    ]

    flat[5].legend(
        handles=handles,
        loc="center left",
        frameon=False,
        title="Architecture",
        ncol=2,
        fontsize=9,
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
        "03_IGKG_absolute_metrics",
    )

    igkg.to_csv(
        outdir / "03_IGKG_absolute_metrics.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Design Rule 4 thesis figures: "
            "combined VGEG/VGKG, averaged block-size test, "
            "and IGKG absolute metrics."
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

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("DESIGN RULE 4 — DISTRIBUTING CHARGE")
    print("=" * 76)
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

    effects = calculate_effects(data)

    effects.to_csv(
        args.outdir / "00_effects_relative_to_even.csv",
        index=False,
    )

    print()
    print("Creating Figure 1...")
    make_combined_vgeg_vgkg(
        effects,
        args.outdir,
    )

    print()
    print("Creating Figure 2...")
    make_block_size_average(
        effects,
        args.outdir,
    )

    print()
    print("Creating Figure 3...")
    make_igkg_absolute(
        data,
        args.outdir,
    )

    print()
    print("=" * 76)
    print("DONE")
    print("=" * 76)
    print(
        "Figure 1: VGEG + VGKG arithmetic mean, no shading."
    )
    print(
        "Figure 2: VGEG + VGKG + IGKG arithmetic mean for "
        "even/small/medium/large, no shading."
    )
    print(
        "Figure 3: IGKG absolute metrics retained as a separate figure."
    )
    print(
        "Kinetic effects are log2 fold changes relative to EVEN "
        "within the same library and ratio."
    )


if __name__ == "__main__":
    main()
