#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_architecture_thesis_figures_v2.py

Updated DR3 thesis plotting script.

PURPOSE
-------
Additional ratio-similarity analysis:
    Compare every R1-R6 pair for each of the four properties using the
    eight architecture-response values averaged across chemistry.
    Correlation tests pattern similarity; RMSE to y=x tests absolute overlap.

Figure 1:
    Test whether architecture effects are conserved across the four
    motif chemistries.

    For each property, architecture effects are first averaged across
    R1-R6 within each chemistry. The resulting 8 architecture values
    are compared pairwise between chemistries.

    Each point = one architecture.

    Four chemistries:
        VGIP
        VGIG
        VAIP
        VAIG

    Six pairwise chemistry comparisons:
        VGIP vs VGIG
        VGIP vs VAIP
        VGIP vs VAIG
        VGIG vs VAIP
        VGIG vs VAIG
        VAIP vs VAIG

    Four properties:
        central 90% Rg range
        k_collapse
        k_escape
        compact-state Rg

    Output:
        4 rows x 6 columns correlation matrix.

    This figure establishes whether architectures tend to produce
    similar directional effects across motif chemistries.


Figure 2:
    Architecture effects shown separately for all six compositions:

        R1 R2 R3 R4 R5 R6

    Rows:
        central 90% Rg range
        k_collapse
        k_escape
        compact-state Rg

    Columns:
        R1-R6

    Within each panel:
        x = architecture
        faint points = individual chemistry observations
        large point = mean across chemistries
        error bar = 95% CI across chemistry-level values

    R4 replicates:
        If a replicate column exists, architecture effects are calculated
        relative to the 8-architecture mean within:
            chemistry + ratio + replicate

        Replicates are then averaged within chemistry before the final
        across-chemistry mean is calculated.

        Therefore:
            - all three R4 replicates are used
            - R4 does NOT receive 3x the statistical weight
            - the four chemistry-level values remain the four comparable
              observations in the main plot

    R1/R2/R3/R5/R6:
        If only one replicate exists, they naturally contribute one
        value per chemistry.


IMPORTANT
---------
Architecture effects are always defined relative to the mean of the
eight architectures at identical chemistry + ratio + replicate.

For rates:
    effect = log2(rate / condition mean rate)

For Rg observables:
    effect = value - condition mean value

This script requires measured k_escape.
It does NOT derive k_escape from dwell time.
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


# ============================================================
# GLOBAL SETTINGS
# ============================================================

ARCH_ORDER = [
    "random shuffle",
    "even",
    "small blocks",
    "medium blocks",
    "large blocks",
    "higher-lambda center",
    "higher-lambda endcaps",
    "diblock",
]


ARCH_LABELS = {
    "even": "Even",
    "small blocks": "Small\nblocks",
    "medium blocks": "Medium\nblocks",
    "large blocks": "Large\nblocks",
    "higher-lambda center": r"High-$\lambda$" + "\ncenter",
    "higher-lambda endcaps": r"High-$\lambda$" + "\nendcaps",
    "diblock": "Diblock",
    "random shuffle": "Random\nshuffle",
}


# ============================================================
# THESIS-WIDE ARCHITECTURE COLOURS
# ============================================================
# Keep these identical across all DR2/DR3 figures.
ARCH_COLORS = {
    "random shuffle": "#7f7f7f",       # grey
    "even": "#1f77b4",                 # blue
    "small blocks": "#2ca02c",         # green
    "medium blocks": "#9467bd",        # purple
    "large blocks": "#ff7f0e",         # orange
    "higher-lambda center": "#d62728", # red
    "higher-lambda endcaps": "#17becf",# cyan
    "diblock": "#8c564b",              # brown
}

# One marker convention for architecture-coded correlation plots.
ARCH_MARKERS = {
    "random shuffle": "o",
    "even": "o",
    "small blocks": "o",
    "medium blocks": "o",
    "large blocks": "o",
    "higher-lambda center": "o",
    "higher-lambda endcaps": "o",
    "diblock": "o",
}


FAMILY_ORDER = [
    "VGIP",
    "VGIG",
    "VAIP",
    "VAIG",
]


FAMILY_LABELS = {
    "VGIP": "VPGVG + IPAVG",
    "VGIG": "VPGVG + VPGIG",
    "VAIP": "VPAVG + IPAVG",
    "VAIG": "VPAVG + VPGIG",
}


FAMILY_MARKERS = {
    "VGIP": "o",
    "VGIG": "s",
    "VAIP": "^",
    "VAIG": "D",
}


RATIO_ORDER = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
]


RATIO_LABELS = {
    "R1": "6:1",
    "R2": "3:1",
    "R3": "2:1",
    "R4": "1:1",
    "R5": "1:2",
    "R6": "1:5",
}

# Fraction of the higher-stickiness motif represented by each ratio.
# These values are used as the actual numeric x-coordinate in the two
# headline composition-dependent DR3 figures, so spacing reflects composition.
HIGH_STICKINESS_FRACTION = {
    "R1": 1 / 7,
    "R2": 1 / 4,
    "R3": 1 / 3,
    "R4": 1 / 2,
    "R5": 2 / 3,
    "R6": 5 / 6,
}

RATIO_TICK_LABELS = {
    ratio: f"{RATIO_LABELS[ratio]}\n({100 * HIGH_STICKINESS_FRACTION[ratio]:.0f}%)"
    for ratio in RATIO_ORDER
}


# ============================================================
# METRICS
# ============================================================

METRICS = [
    {
        "raw": "rg_range90_nm",
        "effect": "effect_rg90_nm",
        "ylabel": (
            r"$\Delta$ central 90% "
            r"$R_g$ range (nm)"
        ),
        "short": (
            r"Central 90% $R_g$ range"
        ),
        "panel_title": (
            "Conformational range"
        ),
        "raw_ylabel": (
            r"Central 90% $R_g$ range (nm)"
        ),
    },

    {
        "raw": "k_collapse_per_ns",
        "effect": "log2_fold_kcollapse",
        "ylabel": (
            r"$\log_2$ fold change in "
            r"$k_{\mathrm{collapse}}$"
        ),
        "short": (
            r"$k_{\mathrm{collapse}}$"
        ),
        "panel_title": (
            "Compact-state entry"
        ),
        "raw_ylabel": (
            r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)"
        ),
    },

    {
        "raw": "k_escape_per_ns",
        "effect": "log2_fold_kescape",
        "ylabel": (
            r"$\log_2$ fold change in "
            r"$k_{\mathrm{escape}}$"
        ),
        "short": (
            r"$k_{\mathrm{escape}}$"
        ),
        "panel_title": (
            "Compact-state escape"
        ),
        "raw_ylabel": (
            r"$k_{\mathrm{escape}}$ (ns$^{-1}$)"
        ),
    },

    {
        "raw": "mean_rg_collapsed_nm",
        "effect": "effect_collapsed_rg_nm",
        "ylabel": (
            r"$\Delta$ mean compact-state "
            r"$R_g$ (nm)"
        ),
        "short": (
            r"Compact-state $R_g$"
        ),
        "panel_title": (
            "Compact-state dimensions"
        ),
        "raw_ylabel": (
            r"Mean compact-state $R_g$ (nm)"
        ),
    },
]


# ============================================================
# STYLE
# ============================================================

def set_style():

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def clean_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# ARCHITECTURE NAME NORMALIZATION
# ============================================================

def canonical_architecture_name(name):

    if pd.isna(name):
        return np.nan

    s = str(name).lower().strip()
    s = s.replace("_", " ")
    s = s.replace("-", " ")

    if "random" in s:
        return "random shuffle"

    if "diblock" in s:
        return "diblock"

    if "small" in s:
        return "small blocks"

    if "medium" in s:
        return "medium blocks"

    if "large" in s:
        return "large blocks"

    if "even" in s or "alternating" in s:
        return "even"

    if "center" in s or "centre" in s:
        return "higher-lambda center"

    if "endcap" in s or "ends" in s:
        return "higher-lambda endcaps"

    return str(name)


def find_architecture_column(df):

    if "architecture_class" in df.columns:
        return "architecture_class"

    if "architecture_type" in df.columns:
        return "architecture_type"

    raise ValueError(
        "Could not find architecture_class "
        "or architecture_type."
    )


# ============================================================
# REPLICATE COLUMN
# ============================================================

def find_replicate_column(df):

    candidates = [
        "replicate",
        "rep",
        "replicate_id",
        "rep_id",
        "run_rep",
    ]

    for column in candidates:

        if column in df.columns:

            print(
                f"Using replicate column: {column}"
            )

            return column

    print()
    print(
        "No replicate column found."
    )

    print(
        "Treating every row as replicate 1 "
        "(valid for the current single-replicate R1-R6 comparison)."
    )

    df["_replicate"] = 1

    return "_replicate"


# ============================================================
# FIND REAL K_ESCAPE
# ============================================================

def find_escape_column(df):

    candidates = [
        "k_escape_per_ns",
        "k_escape",
        "escape_rate_per_ns",
        "escape_rate",
    ]

    for column in candidates:

        if column in df.columns:

            print(
                f"Using measured escape-rate column: "
                f"{column}"
            )

            if column != "k_escape_per_ns":

                df["k_escape_per_ns"] = (
                    df[column]
                )

            return df

    raise ValueError(
        "\nNo measured k_escape column was found.\n\n"
        "Expected one of:\n"
        "    k_escape_per_ns\n"
        "    k_escape\n"
        "    escape_rate_per_ns\n"
        "    escape_rate\n\n"
        "This script intentionally does NOT calculate "
        "k_escape from dwell time."
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if "temperature" in df.columns:

        df = df[
            df["temperature"]
            .astype(str)
            .str.contains("370")
        ].copy()

    elif "temp" in df.columns:

        df = df[
            df["temp"]
            .astype(str)
            .str.contains("370")
        ].copy()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    arch_col = find_architecture_column(df)

    df["architecture_plot"] = (
        df[arch_col]
        .apply(canonical_architecture_name)
    )

    # Resolve centre/endcap semantics using fixed library design ID offsets.
    # E.g. VGcenter_IPends denotes higher-lambda IP at the ENDS.
    id_col = next((c for c in ('id','sequence_id','sequence_name','name')
                   if c in df.columns), None)
    if id_col is not None:
        ids = df[id_col].astype(str).str.extract(r'(?<!\d)(\d{3})(?!\d)', expand=False)
        ids = pd.to_numeric(ids, errors='coerce')
        bases = {'VGIP':31, 'VGIG':79, 'VAIP':127, 'VAIG':175}
        arch_by_offset = {0:'even',1:'small blocks',2:'medium blocks',
                          3:'large blocks',4:'higher-lambda center',
                          5:'higher-lambda endcaps',6:'diblock',7:'random shuffle'}
        for fam,base in bases.items():
            mask = df['family'].astype(str).str.upper().eq(fam) & ids.between(base,base+47)
            if mask.any():
                offsets = ((ids[mask]-base).astype(int) % 8)
                df.loc[mask,'architecture_plot'] = offsets.map(arch_by_offset).to_numpy()
        print('Architecture mapping: using numeric sequence IDs where available.')
    else:
        print('WARNING: no numeric sequence ID column; centre/endcap mapping remains name-based.')

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    df["ratio"] = (
        df["ratio"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["ratio"].isin(RATIO_ORDER)
    ].copy()

    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    df["family"] = (
        df["family"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["family"].isin(FAMILY_ORDER)
    ].copy()

    # --------------------------------------------------------
    # Escape rate
    # --------------------------------------------------------

    df = find_escape_column(df)

    # --------------------------------------------------------
    # Replicate
    # --------------------------------------------------------

    replicate_col = find_replicate_column(df)

    df["_replicate_plot"] = (
        df[replicate_col]
        .astype(str)
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "family",
        "ratio",
        "architecture_plot",
        "rg_range90_nm",
        "k_collapse_per_ns",
        "k_escape_per_ns",
        "mean_rg_collapsed_nm",
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Restrict architecture classes
    # --------------------------------------------------------

    df = df[
        df["architecture_plot"].isin(
            ARCH_ORDER
        )
    ].copy()

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DATA CHECK")
    print("=" * 72)

    print()
    print(
        f"Rows after filtering: {len(df)}"
    )

    print()
    print("Families:")

    print(
        df["family"]
        .value_counts()
        .reindex(FAMILY_ORDER)
    )

    print()
    print("Ratios:")

    print(
        df["ratio"]
        .value_counts()
        .reindex(RATIO_ORDER)
    )

    print()
    print("Architectures:")

    print(
        df["architecture_plot"]
        .value_counts()
        .reindex(ARCH_ORDER)
    )

    # --------------------------------------------------------
    # Replicate counts per family x ratio
    # --------------------------------------------------------

    replicate_counts = (
        df.groupby(
            [
                "family",
                "ratio",
            ]
        )["_replicate_plot"]
        .nunique()
        .unstack("ratio")
        .reindex(
            index=FAMILY_ORDER,
            columns=RATIO_ORDER,
        )
    )

    print()
    print(
        "Number of replicates per family x ratio:"
    )

    print(replicate_counts)

    # --------------------------------------------------------
    # Architecture completeness
    # --------------------------------------------------------

    condition_counts = (
        df.groupby(
            [
                "family",
                "ratio",
                "_replicate_plot",
            ]
        )["architecture_plot"]
        .nunique()
    )

    print()
    print(
        "Number of architectures per "
        "family x ratio x replicate:"
    )

    print(
        condition_counts
        .value_counts()
        .sort_index()
    )

    incomplete = (
        condition_counts[
            condition_counts != 8
        ]
    )

    if len(incomplete) > 0:

        print()
        print(
            "WARNING: incomplete replicate conditions:"
        )

        print(incomplete)

    return df


# ============================================================
# ARCHITECTURE EFFECTS
# ============================================================

def calculate_architecture_effects(df):

    df = df.copy()

    # IMPORTANT:
    #
    # Architecture effects are calculated within a replicate.
    #
    # Therefore R4 replicate 1 is normalized against the
    # 8 architectures from R4 replicate 1, etc.
    #
    # This prevents replicate-to-replicate shifts from being
    # interpreted as architecture effects.

    condition_cols = [
        "family",
        "ratio",
        "_replicate_plot",
    ]

    raw_metrics = [
        "rg_range90_nm",
        "k_collapse_per_ns",
        "k_escape_per_ns",
        "mean_rg_collapsed_nm",
    ]

    for metric in raw_metrics:

        df[
            metric + "_condition_mean"
        ] = (
            df
            .groupby(
                condition_cols
            )[metric]
            .transform("mean")
        )

    # --------------------------------------------------------
    # Absolute Rg effects
    # --------------------------------------------------------

    df["effect_rg90_nm"] = (
        df["rg_range90_nm"]
        -
        df[
            "rg_range90_nm_condition_mean"
        ]
    )

    df["effect_collapsed_rg_nm"] = (
        df["mean_rg_collapsed_nm"]
        -
        df[
            "mean_rg_collapsed_nm_condition_mean"
        ]
    )

    # --------------------------------------------------------
    # Rate effects
    # --------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        df["log2_fold_kcollapse"] = (
            np.log2(
                df["k_collapse_per_ns"]
                /
                df[
                    "k_collapse_per_ns_condition_mean"
                ]
            )
        )

        df["log2_fold_kescape"] = (
            np.log2(
                df["k_escape_per_ns"]
                /
                df[
                    "k_escape_per_ns_condition_mean"
                ]
            )
        )

    return df


# ============================================================
# COLLAPSE REPLICATES TO CHEMISTRY-LEVEL VALUES
# ============================================================

def make_chemistry_level_table(df):

    """
    Average replicate-level architecture effects within:

        family + ratio + architecture

    This is crucial for R4.

    Example:
        VGIG, R4, high-lambda center

    If three replicate values exist:
        replicate 1 effect
        replicate 2 effect
        replicate 3 effect

    they become ONE chemistry-level value.

    Thus every chemistry gets equal weight in the final R4
    architecture mean.
    """

    grouping = [
        "family",
        "ratio",
        "architecture_plot",
    ]

    agg_dict = {}

    for metric in METRICS:

        # Architecture effect used in the main DR3 figure.
        agg_dict[
            metric["effect"]
        ] = "mean"

        # Absolute/raw observable used in the supplementary figure.
        # Replicates are averaged first within each
        # family + ratio + architecture, exactly as for the effects.
        agg_dict[
            metric["raw"]
        ] = "mean"

    chemistry_df = (
        df.groupby(
            grouping,
            as_index=False,
        )
        .agg(agg_dict)
    )

    # Number of replicates used
    replicate_n = (
        df.groupby(
            grouping
        )["_replicate_plot"]
        .nunique()
        .rename("n_replicates")
        .reset_index()
    )

    chemistry_df = (
        chemistry_df.merge(
            replicate_n,
            on=grouping,
            how="left",
        )
    )

    return chemistry_df


# ============================================================
# MEAN + 95% CI
# ============================================================

def mean_ci95(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:

        return (
            np.nan,
            np.nan,
            0,
        )

    mean = np.mean(values)

    if n == 1:

        return (
            mean,
            np.nan,
            1,
        )

    sd = np.std(
        values,
        ddof=1,
    )

    ci = (
        1.96
        *
        sd
        /
        np.sqrt(n)
    )

    return (
        mean,
        ci,
        n,
    )


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(
    fig,
    outdir,
    basename,
):

    png = (
        outdir
        /
        f"{basename}.png"
    )

    pdf = (
        outdir
        /
        f"{basename}.pdf"
    )

    svg = (
        outdir
        /
        f"{basename}.svg"
    )

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {svg}")


# ============================================================
# CORRELATION HELPERS
# ============================================================

def correlation_values(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < 3:

        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_r, pearson_p = (
        pearsonr(
            x,
            y,
        )
    )

    spearman_rho, spearman_p = (
        spearmanr(
            x,
            y,
        )
    )

    return {
        "n": n,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }



# ============================================================
# RATIO-TO-RATIO SIMILARITY
# ============================================================

def make_ratio_architecture_profiles(chemistry_df):
    """
    Build the architecture-response profile used to compare ratios.

    For each:
        ratio + architecture

    average the chemistry-level architecture effects across the four
    chemistries. Because make_chemistry_level_table() has already averaged
    replicates within chemistry, R4 does not receive extra statistical weight.

    Each ratio therefore becomes an 8-value architecture-response vector
    for each property.
    """
    agg_dict = {
        metric["effect"]: "mean"
        for metric in METRICS
    }

    profiles = (
        chemistry_df
        .groupby(
            ["ratio", "architecture_plot"],
            as_index=False,
        )
        .agg(agg_dict)
    )

    profiles["architecture_plot"] = pd.Categorical(
        profiles["architecture_plot"],
        categories=ARCH_ORDER,
        ordered=True,
    )

    return profiles.sort_values(
        ["ratio", "architecture_plot"]
    ).reset_index(drop=True)


def calculate_ratio_pair_correlations(chemistry_df):
    """
    Compare every pair of composition ratios separately for each property.

    Primary comparison:
        the 8 architecture effects averaged across chemistry.

    Thus R1 vs R2 asks:
        do the same eight architectures produce the same relative response
        pattern at R1 and R2?

    Outputs Pearson r, Spearman rho, RMSE, MAE, slope, intercept and n.

    Correlation measures similarity of the PATTERN.
    RMSE/MAE and the y=x comparison measure absolute overlap.
    """
    profiles = make_ratio_architecture_profiles(chemistry_df)
    ratio_pairs = list(combinations(RATIO_ORDER, 2))
    rows = []

    for metric in METRICS:
        effect = metric["effect"]

        for ratio_x, ratio_y in ratio_pairs:
            x = (
                profiles.loc[
                    profiles["ratio"] == ratio_x,
                    ["architecture_plot", effect],
                ]
                .rename(columns={effect: "x"})
            )

            y = (
                profiles.loc[
                    profiles["ratio"] == ratio_y,
                    ["architecture_plot", effect],
                ]
                .rename(columns={effect: "y"})
            )

            merged = x.merge(
                y,
                on="architecture_plot",
                how="inner",
            )

            xv = merged["x"].to_numpy(dtype=float)
            yv = merged["y"].to_numpy(dtype=float)
            valid = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[valid]
            yv = yv[valid]

            stats = correlation_values(xv, yv)

            if len(xv) > 0:
                rmse = float(np.sqrt(np.mean((yv - xv) ** 2)))
                mae = float(np.mean(np.abs(yv - xv)))
                mean_abs_effect = float(
                    np.mean(np.abs(np.concatenate([xv, yv])))
                )
                normalized_rmse = (
                    rmse / mean_abs_effect
                    if mean_abs_effect > 0
                    else np.nan
                )
            else:
                rmse = np.nan
                mae = np.nan
                normalized_rmse = np.nan

            if len(xv) >= 2 and np.std(xv) > 0:
                slope, intercept = np.polyfit(xv, yv, 1)
            else:
                slope = np.nan
                intercept = np.nan

            rows.append({
                "metric": metric["short"],
                "effect_column": effect,
                "ratio_x": ratio_x,
                "ratio_y": ratio_y,
                **stats,
                "rmse_to_identity": rmse,
                "mae_to_identity": mae,
                "normalized_rmse_to_identity": normalized_rmse,
                "identity_slope": slope,
                "identity_intercept": intercept,
            })

    return pd.DataFrame(rows), profiles


def figure_ratio_pair_correlations(
    chemistry_df,
    outdir,
):
    """
    4 rows x 15 columns.

    Rows:
        the four DR3 properties

    Columns:
        every pair among R1-R6

    Each point = one architecture, after averaging its chemistry-level
    architecture effect across the four chemistries.

    Dashed diagonal = exact y=x overlap.

    This is the most direct visual test of whether two ratios encode the
    same architecture-response profile.
    """
    correlation_df, profiles = (
        calculate_ratio_pair_correlations(
            chemistry_df
        )
    )

    ratio_pairs = list(
        combinations(
            RATIO_ORDER,
            2,
        )
    )


    fig, axes = plt.subplots(
        len(METRICS),
        len(ratio_pairs),
        figsize=(31, 10.5),
        squeeze=False,
    )

    for row, metric in enumerate(METRICS):
        effect = metric["effect"]

        for col, (ratio_x, ratio_y) in enumerate(ratio_pairs):
            ax = axes[row, col]

            x = (
                profiles.loc[
                    profiles["ratio"] == ratio_x,
                    ["architecture_plot", effect],
                ]
                .rename(columns={effect: "x"})
            )

            y = (
                profiles.loc[
                    profiles["ratio"] == ratio_y,
                    ["architecture_plot", effect],
                ]
                .rename(columns={effect: "y"})
            )

            merged = x.merge(
                y,
                on="architecture_plot",
                how="inner",
            )

            merged["architecture_plot"] = pd.Categorical(
                merged["architecture_plot"],
                categories=ARCH_ORDER,
                ordered=True,
            )
            merged = merged.sort_values("architecture_plot")

            for _, point in merged.iterrows():
                architecture = point["architecture_plot"]

                if (
                    not np.isfinite(point["x"])
                    or not np.isfinite(point["y"])
                ):
                    continue

                ax.scatter(
                    point["x"],
                    point["y"],
                    marker=ARCH_MARKERS[architecture],
                    s=48,
                    color=ARCH_COLORS[architecture],
                    alpha=0.90,
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )

            values = np.concatenate([
                merged["x"].to_numpy(dtype=float),
                merged["y"].to_numpy(dtype=float),
            ])
            values = values[np.isfinite(values)]

            if len(values) > 0:
                spread = max(
                    values.max() - values.min(),
                    1e-6,
                )
                padding = 0.10 * spread
                lower = values.min() - padding
                upper = values.max() + padding

                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    linestyle="--",
                    linewidth=0.9,
                    color="0.65",
                    zorder=0,
                )
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)

            ax.axhline(
                0,
                linewidth=0.6,
                color="0.88",
                zorder=0,
            )
            ax.axvline(
                0,
                linewidth=0.6,
                color="0.88",
                zorder=0,
            )

            stats_row = correlation_df[
                (correlation_df["effect_column"] == effect)
                & (correlation_df["ratio_x"] == ratio_x)
                & (correlation_df["ratio_y"] == ratio_y)
            ].iloc[0]

            rho = stats_row["spearman_rho"]
            r = stats_row["pearson_r"]
            rmse = stats_row["rmse_to_identity"]

            annotation = (
                rf"$r$={r:.2f}" + "\n"
                rf"$\rho_s$={rho:.2f}" + "\n"
                f"RMSE={rmse:.3g}"
            )

            ax.text(
                0.05,
                0.95,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
            )

            if row == 0:
                ax.set_title(
                    f"{ratio_x} vs {ratio_y}",
                    fontsize=8,
                    fontweight="bold",
                )

            if row == len(METRICS) - 1:
                ax.set_xlabel(ratio_x)
            else:
                ax.set_xlabel("")

            if col == 0:
                ax.set_ylabel(
                    metric["short"] + "\n" + ratio_y
                )
            else:
                ax.set_ylabel("")

            clean_axis(ax)

    legend_handles = []
    for architecture in ARCH_ORDER:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=ARCH_MARKERS[architecture],
                linestyle="none",
                markersize=6,
                markerfacecolor=ARCH_COLORS[architecture],
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=ARCH_LABELS[architecture].replace("\n", " "),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(
        h_pad=1.2,
        w_pad=0.65,
        rect=[0, 0.06, 1, 1],
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_ratio_pair_correlations_all_pairs",
    )

    correlation_df.to_csv(
        outdir / "DR3_ratio_pair_correlations_all_pairs.csv",
        index=False,
    )

    profiles.to_csv(
        outdir / "DR3_ratio_architecture_profiles.csv",
        index=False,
    )

    print(
        "Saved all pairwise ratio correlation statistics."
    )

    return correlation_df, profiles


def figure_ratio_similarity_heatmaps(
    correlation_df,
    outdir,
):
    """
    Compact 4 x 3 summary figure.

    For each property:
        Pearson r
        Spearman rho
        RMSE to y=x

    The correlation panels test similarity of architecture ordering/pattern.
    RMSE tests absolute overlap.
    """
    fig, axes = plt.subplots(
        len(METRICS),
        3,
        figsize=(11.5, 12.5),
        squeeze=False,
    )

    stat_columns = [
        (
            "pearson_r",
            "Pearson r",
            -1.0,
            1.0,
            "coolwarm",
        ),
        (
            "spearman_rho",
            r"Spearman $\rho_s$",
            -1.0,
            1.0,
            "coolwarm",
        ),
        (
            "rmse_to_identity",
            "RMSE to y = x",
            None,
            None,
            "viridis_r",
        ),
    ]

    for row, metric in enumerate(METRICS):
        metric_df = correlation_df[
            correlation_df["effect_column"]
            == metric["effect"]
        ]

        for col, (
            stat_col,
            stat_label,
            vmin,
            vmax,
            cmap,
        ) in enumerate(stat_columns):

            ax = axes[row, col]

            matrix = np.full(
                (len(RATIO_ORDER), len(RATIO_ORDER)),
                np.nan,
                dtype=float,
            )

            if stat_col in ("pearson_r", "spearman_rho"):
                np.fill_diagonal(matrix, 1.0)
            else:
                np.fill_diagonal(matrix, 0.0)

            for _, result in metric_df.iterrows():
                i = RATIO_ORDER.index(result["ratio_x"])
                j = RATIO_ORDER.index(result["ratio_y"])
                value = result[stat_col]
                matrix[i, j] = value
                matrix[j, i] = value

            image = ax.imshow(
                matrix,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect="equal",
            )

            ax.set_xticks(
                np.arange(len(RATIO_ORDER))
            )
            ax.set_xticklabels(RATIO_ORDER)

            ax.set_yticks(
                np.arange(len(RATIO_ORDER))
            )

            if col == 0:
                ax.set_yticklabels(RATIO_ORDER)
                ax.set_ylabel(metric["short"])
            else:
                ax.set_yticklabels([])

            if row == 0:
                ax.set_title(
                    stat_label,
                    fontweight="bold",
                )

            for i in range(len(RATIO_ORDER)):
                for j in range(len(RATIO_ORDER)):
                    value = matrix[i, j]
                    if not np.isfinite(value):
                        continue

                    if stat_col == "rmse_to_identity":
                        label = f"{value:.3g}"
                    else:
                        label = f"{value:.2f}"

                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

            fig.colorbar(
                image,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )

    fig.tight_layout(
        h_pad=1.8,
        w_pad=1.2,
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_ratio_similarity_heatmaps",
    )


def figure_ratio_similarity_overview(
    correlation_df,
    outdir,
):
    """
    One compact heatmap: rows = four properties, columns = 15 ratio pairs.

    Cell value = Spearman rho.

    This is the quickest view for spotting pairs such as R1/R2 that show
    the same architecture-response pattern across ALL four properties.
    """
    ratio_pairs = list(
        combinations(
            RATIO_ORDER,
            2,
        )
    )

    matrix = np.full(
        (len(METRICS), len(ratio_pairs)),
        np.nan,
        dtype=float,
    )

    for row, metric in enumerate(METRICS):
        for col, (ratio_x, ratio_y) in enumerate(ratio_pairs):
            d = correlation_df[
                (correlation_df["effect_column"] == metric["effect"])
                & (correlation_df["ratio_x"] == ratio_x)
                & (correlation_df["ratio_y"] == ratio_y)
            ]

            if len(d):
                matrix[row, col] = d["spearman_rho"].iloc[0]

    fig, ax = plt.subplots(
        figsize=(13.5, 4.2)
    )

    image = ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(ratio_pairs))
    )
    ax.set_xticklabels(
        [
            f"{a}\n{b}"
            for a, b in ratio_pairs
        ],
        fontsize=8,
    )

    ax.set_yticks(
        np.arange(len(METRICS))
    )
    ax.set_yticklabels(
        [
            metric["short"]
            for metric in METRICS
        ]
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(
        r"Spearman $\rho_s$"
    )

    fig.tight_layout()

    save_figure(
        fig,
        outdir,
        "figure_DR3_ratio_similarity_overview",
    )

# ============================================================
# FIGURE 1
# CHEMISTRY CONSERVATION
# ============================================================

def make_family_architecture_means(
    chemistry_df,
):

    """
    Average architecture effects across R1-R6 for each chemistry.

    Output:
        family
        architecture
        metric effects

    This produces 8 architecture values per chemistry.

    These are the values used in Figure 1.
    """

    grouping = [
        "family",
        "architecture_plot",
    ]

    agg_dict = {
        metric["effect"]: "mean"
        for metric in METRICS
    }

    result = (
        chemistry_df
        .groupby(
            grouping,
            as_index=False,
        )
        .agg(agg_dict)
    )

    return result


def figure_chemistry_correlations(
    chemistry_df,
    outdir,
):

    family_arch = (
        make_family_architecture_means(
            chemistry_df
        )
    )

    chemistry_pairs = list(
        combinations(
            FAMILY_ORDER,
            2,
        )
    )

    n_rows = len(METRICS)
    n_cols = len(chemistry_pairs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(17, 11),
    )

    correlation_rows = []

    # Architecture markers so each of the 8 points
    # can be identified visually.


    for row, metric in enumerate(
        METRICS
    ):

        effect = metric["effect"]

        for col, (
            family_x,
            family_y,
        ) in enumerate(
            chemistry_pairs
        ):

            ax = axes[
                row,
                col,
            ]

            x_table = (
                family_arch[
                    family_arch["family"]
                    == family_x
                ][
                    [
                        "architecture_plot",
                        effect,
                    ]
                ]
                .rename(
                    columns={
                        effect: "x"
                    }
                )
            )

            y_table = (
                family_arch[
                    family_arch["family"]
                    == family_y
                ][
                    [
                        "architecture_plot",
                        effect,
                    ]
                ]
                .rename(
                    columns={
                        effect: "y"
                    }
                )
            )

            merged = (
                x_table.merge(
                    y_table,
                    on="architecture_plot",
                    how="inner",
                )
            )

            merged = (
                merged[
                    merged[
                        "architecture_plot"
                    ].isin(
                        ARCH_ORDER
                    )
                ]
                .copy()
            )

            merged[
                "architecture_plot"
            ] = pd.Categorical(
                merged[
                    "architecture_plot"
                ],
                categories=ARCH_ORDER,
                ordered=True,
            )

            merged = (
                merged.sort_values(
                    "architecture_plot"
                )
            )

            # -----------------------------------------------
            # Plot one point per architecture
            # -----------------------------------------------

            for _, point in (
                merged.iterrows()
            ):

                architecture = (
                    point[
                        "architecture_plot"
                    ]
                )

                ax.scatter(
                    point["x"],
                    point["y"],
                    marker=ARCH_MARKERS[architecture],
                    s=58,
                    color=ARCH_COLORS[architecture],
                    alpha=0.90,
                    edgecolor="white",
                    linewidth=0.5,
                )

            # -----------------------------------------------
            # Correlation
            # -----------------------------------------------

            stats = correlation_values(
                merged["x"],
                merged["y"],
            )

            correlation_rows.append({
                "metric": metric["short"],
                "family_x": family_x,
                "family_y": family_y,
                **stats,
            })

            # -----------------------------------------------
            # y = x reference line
            # -----------------------------------------------

            values = np.concatenate([
                merged["x"]
                .to_numpy(
                    dtype=float
                ),
                merged["y"]
                .to_numpy(
                    dtype=float
                ),
            ])

            values = values[
                np.isfinite(values)
            ]

            if len(values) > 0:

                padding = (
                    0.08
                    *
                    max(
                        values.max()
                        -
                        values.min(),
                        1e-6,
                    )
                )

                lower = (
                    values.min()
                    -
                    padding
                )

                upper = (
                    values.max()
                    +
                    padding
                )

                ax.plot(
                    [
                        lower,
                        upper,
                    ],
                    [
                        lower,
                        upper,
                    ],
                    linestyle="--",
                    linewidth=1,
                    color="0.65",
                    zorder=0,
                )

                ax.set_xlim(
                    lower,
                    upper,
                )

                ax.set_ylim(
                    lower,
                    upper,
                )

            # -----------------------------------------------
            # Zero lines
            # -----------------------------------------------

            ax.axhline(
                0,
                linewidth=0.7,
                color="0.85",
                zorder=0,
            )

            ax.axvline(
                0,
                linewidth=0.7,
                color="0.85",
                zorder=0,
            )

            # -----------------------------------------------
            # Correlation annotation
            # -----------------------------------------------

            if np.isfinite(
                stats[
                    "spearman_rho"
                ]
            ):

                text = (
                    rf"$\rho_s$ = "
                    f"{stats['spearman_rho']:.2f}"
                    "\n"
                    f"p = "
                    f"{stats['spearman_p']:.3f}"
                )

            else:

                text = "n/a"

            ax.text(
                0.05,
                0.95,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

            # -----------------------------------------------
            # Column title
            # -----------------------------------------------

            if row == 0:

                ax.set_title(
                    f"{family_x} vs {family_y}",
                    fontweight="bold",
                )

            # -----------------------------------------------
            # Axis labels
            # -----------------------------------------------

            if row == n_rows - 1:

                ax.set_xlabel(
                    FAMILY_LABELS[
                        family_x
                    ]
                )

            else:

                ax.set_xlabel("")

            if col == 0:

                ax.set_ylabel(
                    metric["short"]
                    + "\n"
                    + FAMILY_LABELS[
                        family_y
                    ]
                )

            else:

                ax.set_ylabel("")

            clean_axis(ax)

    # --------------------------------------------------------
    # Architecture legend
    # --------------------------------------------------------

    legend_handles = []

    for architecture in ARCH_ORDER:

        handle = plt.Line2D(
            [0],
            [0],
            marker=ARCH_MARKERS[architecture],
            linestyle="none",
            markersize=7,
            markerfacecolor=ARCH_COLORS[architecture],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=(
                ARCH_LABELS[
                    architecture
                ]
                .replace(
                    "\n",
                    " ",
                )
            ),
        )

        legend_handles.append(
            handle
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            -0.015,
        ),
    )

    fig.tight_layout(
        h_pad=1.8,
        w_pad=1.2,
        rect=[
            0,
            0.05,
            1,
            1,
        ],
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_chemistry_architecture_correlations",
    )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_df.to_csv(
        outdir
        /
        "DR3_chemistry_architecture_correlations.csv",
        index=False,
    )

    print(
        "Saved chemistry correlation table."
    )


# ============================================================
# FIGURE 2
# ALL SIX RATIOS SEPARATELY
# ============================================================

def figure_ratio_matrix(
    chemistry_df,
    replicate_df,
    outdir,
):

    n_rows = len(METRICS)
    n_cols = len(RATIO_ORDER)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(23, 14),
        sharey="row",
    )

    # Chemistry offsets.
    #
    # These four points are chemistry-level values.
    # For R4 each chemistry value is already the mean
    # of its replicate-level architecture effects.

    family_offsets = {
        "VGIP": -0.12,
        "VGIG": -0.04,
        "VAIP": +0.04,
        "VAIG": +0.12,
    }

    panel_letters = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    panel_counter = 0

    for row, metric in enumerate(
        METRICS
    ):

        effect = metric["effect"]

        for col, ratio in enumerate(
            RATIO_ORDER
        ):

            ax = axes[
                row,
                col,
            ]

            ratio_df = chemistry_df[
                chemistry_df["ratio"]
                == ratio
            ]

            # ===============================================
            # Architecture
            # ===============================================

            for i, architecture in enumerate(
                ARCH_ORDER
            ):

                arch = ratio_df[
                    ratio_df[
                        "architecture_plot"
                    ]
                    == architecture
                ]

                # -------------------------------------------
                # Individual chemistry-level points
                # -------------------------------------------

                for family in FAMILY_ORDER:

                    point = arch[
                        arch["family"]
                        == family
                    ]

                    if len(point) == 0:
                        continue

                    y = (
                        point[
                            effect
                        ]
                        .iloc[0]
                    )

                    if not np.isfinite(y):
                        continue

                    ax.scatter(
                        (
                            i
                            +
                            family_offsets[
                                family
                            ]
                        ),
                        y,
                        marker=FAMILY_MARKERS[family],
                        s=31,
                        color=ARCH_COLORS[architecture],
                        alpha=0.50,
                        edgecolor="none",
                        zorder=3,
                    )

                    # ---------------------------------------
                    # R4 replicate-level values
                    #
                    # These are shown as tiny open points
                    # around the chemistry-level point.
                    #
                    # This makes the triplicate information
                    # visible without giving R4 3x weight.
                    # ---------------------------------------

                    if ratio == "R4":

                        rep_points = (
                            replicate_df[
                                (
                                    replicate_df[
                                        "ratio"
                                    ]
                                    == ratio
                                )
                                &
                                (
                                    replicate_df[
                                        "family"
                                    ]
                                    == family
                                )
                                &
                                (
                                    replicate_df[
                                        "architecture_plot"
                                    ]
                                    == architecture
                                )
                            ]
                        )

                        rep_values = (
                            rep_points[
                                effect
                            ]
                            .replace(
                                [
                                    np.inf,
                                    -np.inf,
                                ],
                                np.nan,
                            )
                            .dropna()
                            .to_numpy()
                        )

                        if (
                            len(rep_values)
                            > 1
                        ):

                            rep_jitter = (
                                np.linspace(
                                    -0.025,
                                    +0.025,
                                    len(
                                        rep_values
                                    ),
                                )
                            )

                            for (
                                jitter,
                                rep_value,
                            ) in zip(
                                rep_jitter,
                                rep_values,
                            ):

                                ax.scatter(
                                    (
                                        i
                                        +
                                        family_offsets[
                                            family
                                        ]
                                        +
                                        jitter
                                    ),
                                    rep_value,
                                    marker=FAMILY_MARKERS[family],
                                    s=14,
                                    facecolors="none",
                                    edgecolors=ARCH_COLORS[architecture],
                                    linewidths=0.75,
                                    alpha=0.75,
                                    zorder=2,
                                )

                # -------------------------------------------
                # Mean across FOUR chemistries
                # -------------------------------------------

                values = (
                    arch[
                        effect
                    ]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                )

                mean, ci, n = (
                    mean_ci95(
                        values
                    )
                )

                if np.isfinite(mean):

                    ax.errorbar(
                        i,
                        mean,
                        yerr=ci,
                        fmt="o",
                        markersize=7,
                        color=ARCH_COLORS[architecture],
                        ecolor=ARCH_COLORS[architecture],
                        markeredgecolor="white",
                        markeredgewidth=0.55,
                        capsize=3,
                        linewidth=1.45,
                        zorder=10,
                    )

            # ===============================================
            # ZERO REFERENCE
            # ===============================================

            ax.axhline(
                0,
                linestyle="--",
                linewidth=1,
                color="0.45",
            )

            # ===============================================
            # X AXIS
            # ===============================================

            ax.set_xticks(
                np.arange(
                    len(
                        ARCH_ORDER
                    )
                )
            )

            # Only show architecture labels on bottom row.
            # Otherwise 24 repeated x labels becomes messy.

            if row == (
                n_rows - 1
            ):

                ax.set_xticklabels(
                    [
                        ARCH_LABELS[x]
                        for x in ARCH_ORDER
                    ],
                    rotation=48,
                    ha="right",
                    fontsize=7.5,
                )

            else:

                ax.set_xticklabels(
                    []
                )

            # ===============================================
            # RATIO TITLES
            # ===============================================

            if row == 0:

                ax.set_title(
                    RATIO_LABELS[
                        ratio
                    ],
                    fontweight="bold",
                    fontsize=11,
                )

            # ===============================================
            # Y LABELS
            # ===============================================

            if col == 0:

                ax.set_ylabel(
                    metric[
                        "ylabel"
                    ]
                )

            # ===============================================
            # PANEL LETTER
            # ===============================================

            ax.text(
                -0.12,
                1.03,
                panel_letters[
                    panel_counter
                ],
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                va="bottom",
            )

            panel_counter += 1

            clean_axis(ax)

    # ========================================================
    # FAMILY LEGEND
    # ========================================================

    family_handles = []

    for family in FAMILY_ORDER:

        handle = plt.Line2D(
            [0],
            [0],
            marker=(
                FAMILY_MARKERS[
                    family
                ]
            ),
            linestyle="none",
            markersize=7,
            label=(
                FAMILY_LABELS[
                    family
                ]
            ),
        )

        family_handles.append(
            handle
        )

    # Architecture colour legend.
    architecture_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=ARCH_COLORS[architecture],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=ARCH_LABELS[architecture].replace("\n", " "),
        )
        for architecture in ARCH_ORDER
    ]

    # Chemistry is encoded by marker shape, architecture by colour.
    fig.legend(
        handles=architecture_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        title="Architecture (colour)",
    )

    fig.legend(
        handles=family_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.012),
        title="Chemistry (shape)",
    )

    fig.tight_layout(
        h_pad=1.3,
        w_pad=0.8,
        rect=[
            0,
            0.12,
            1,
            1,
        ],
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_architecture_effects_R1_R6",
    )




# ============================================================
# DESIGN-RULE SUMMARY FIGURES
# Architecture sensitivity and architecture trajectories
# ============================================================

def _architecture_sensitivity_table(chemistry_df):
    """
    Quantify how strongly architecture matters at each composition ratio.

    For each ratio/property:
      1. chemistry-level architecture effects are already normalized relative
         to the 8-architecture mean at identical chemistry + ratio;
      2. average each architecture effect across the four chemistries;
      3. calculate the SD across the 8 architecture means.

    SD is preferred over max-min range because it is less dominated by a
    single extreme architecture.
    """
    rows = []

    for metric in METRICS:
        effect = metric["effect"]

        for ratio in RATIO_ORDER:
            d = chemistry_df[
                chemistry_df["ratio"] == ratio
            ].copy()

            arch_means = (
                d.groupby("architecture_plot", observed=True)[effect]
                .mean()
                .reindex(ARCH_ORDER)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if len(arch_means) >= 2:
                sensitivity_sd = float(arch_means.std(ddof=1))
                sensitivity_range = float(
                    arch_means.max() - arch_means.min()
                )
                mean_abs_effect = float(
                    np.abs(arch_means).mean()
                )
            else:
                sensitivity_sd = np.nan
                sensitivity_range = np.nan
                mean_abs_effect = np.nan

            rows.append(
                {
                    "metric": metric["short"],
                    "effect_column": effect,
                    "ratio": ratio,
                    "n_architectures": int(len(arch_means)),
                    "architecture_effect_sd": sensitivity_sd,
                    "architecture_effect_range": sensitivity_range,
                    "mean_absolute_architecture_effect": mean_abs_effect,
                }
            )

    return pd.DataFrame(rows)


def figure_architecture_sensitivity_across_ratios(
    chemistry_df,
    outdir,
):
    """
    Headline design-rule summary:
    how strongly architecture matters as motif composition changes.

    x-position is the actual fraction of higher-stickiness motif.
    Tick labels additionally show the lower-stickiness:higher-stickiness ratio.
    """
    sensitivity = _architecture_sensitivity_table(
        chemistry_df
    )

    sensitivity.to_csv(
        Path(outdir) / "DR3_architecture_sensitivity_across_ratios.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.4, 7.0),
        squeeze=False,
    )
    axes = axes.ravel()

    x = np.array(
        [
            100 * HIGH_STICKINESS_FRACTION[r]
            for r in RATIO_ORDER
        ],
        dtype=float,
    )

    for panel_idx, (ax, metric) in enumerate(
        zip(axes, METRICS)
    ):
        d = (
            sensitivity[
                sensitivity["metric"] == metric["short"]
            ]
            .set_index("ratio")
            .reindex(RATIO_ORDER)
        )

        y = d["architecture_effect_sd"].to_numpy(
            dtype=float
        )

        ax.plot(
            x,
            y,
            color="0.20",
            linewidth=1.8,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=52,
            color="0.20",
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [RATIO_TICK_LABELS[r] for r in RATIO_ORDER],
            fontsize=8.5,
            linespacing=1.25,
        )
        ax.set_xlabel(
            "Motif composition (A:B)",
            fontsize=9,
        )

        if metric["raw"] in {"rg_range90_nm", "mean_rg_collapsed_nm"}:
            unit = "nm"
        else:
            unit = r"$\log_2$ fold change"

        ax.set_ylabel(
            f"Architecture sensitivity, SD ({unit})",
            fontsize=9,
        )

        ax.set_title(
            metric["panel_title"],
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="bottom",
        )

        ax.set_xlim(10, 87)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelsize=8.5, pad=5)
        ax.tick_params(axis="y", labelsize=8.5)
        clean_axis(ax)

    fig.tight_layout(
        h_pad=1.45,
        w_pad=1.55,
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_architecture_sensitivity_across_ratios_main",
    )


def figure_architecture_trajectories_across_ratios(
    chemistry_df,
    outdir,
):
    """
    Main DR3 architecture-response figure.

    Improvements for thesis readability:
      - actual composition spacing on x-axis;
      - actual motif ratios rather than R1-R6;
      - percentage higher-stickiness motif shown under each ratio;
      - architecture trajectories are visually dominant;
      - chemistry-derived 95% CI bars are thinner, capless and translucent;
      - fixed thesis-wide architecture colours.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.6, 7.6),
        squeeze=False,
    )
    axes = axes.ravel()

    # Use physical composition as the x coordinate rather than six equally
    # spaced categorical positions.
    x = np.array(
        [
            100 * HIGH_STICKINESS_FRACTION[r]
            for r in RATIO_ORDER
        ],
        dtype=float,
    )

    for panel_idx, (ax, metric) in enumerate(
        zip(axes, METRICS)
    ):
        effect = metric["effect"]

        for architecture in ARCH_ORDER:
            means = []
            cis = []

            for ratio in RATIO_ORDER:
                values = (
                    chemistry_df[
                        (chemistry_df["ratio"] == ratio)
                        & (
                            chemistry_df["architecture_plot"]
                            == architecture
                        )
                    ][effect]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                mean, ci, n = mean_ci95(values)
                means.append(mean)
                cis.append(ci)

            means = np.asarray(means, dtype=float)
            cis = np.asarray(cis, dtype=float)

            # Uncertainty first, deliberately de-emphasized.
            ax.errorbar(
                x,
                means,
                yerr=cis,
                fmt="none",
                ecolor=ARCH_COLORS[architecture],
                elinewidth=0.8,
                capsize=0,
                alpha=0.32,
                zorder=1,
            )

            # Architecture trajectory is the visual focus.
            ax.plot(
                x,
                means,
                color=ARCH_COLORS[architecture],
                linewidth=2.05,
                alpha=0.96,
                zorder=3,
            )

            ax.scatter(
                x,
                means,
                s=42,
                color=ARCH_COLORS[architecture],
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=0.9,
            color="0.58",
            zorder=0,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [RATIO_TICK_LABELS[r] for r in RATIO_ORDER],
            fontsize=8.5,
            linespacing=1.25,
        )
        ax.set_xlim(10, 87)

        ax.set_xlabel(
            "Motif composition (A:B)",
            fontsize=9,
        )
        ax.set_ylabel(
            metric["ylabel"],
            fontsize=9.5,
        )
        ax.set_title(
            metric["panel_title"],
            fontsize=11,
            fontweight="bold",
            pad=7,
        )

        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="bottom",
        )

        ax.tick_params(axis="x", labelsize=8.5, pad=5)
        ax.tick_params(axis="y", labelsize=8.5)
        clean_axis(ax)

    # Keep the established thesis architecture encoding.
    architecture_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            color=ARCH_COLORS[architecture],
            markerfacecolor=ARCH_COLORS[architecture],
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=ARCH_LABELS[architecture].replace("\n", " "),
        )
        for architecture in ARCH_ORDER
    ]

    fig.legend(
        handles=architecture_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
        title="Architecture",
        fontsize=8.7,
        title_fontsize=9.3,
        columnspacing=1.6,
        handlelength=2.4,
        handletextpad=0.6,
    )

    fig.tight_layout(
        h_pad=2.05,
        w_pad=1.75,
        rect=[0, 0.125, 1, 1],
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_architecture_trajectories_composition_main",
    )



# ============================================================
# SUPPLEMENTARY: ABSOLUTE OBSERVABLES ACROSS COMPOSITION
# ============================================================

def figure_absolute_observables_across_ratios(
    chemistry_df,
    outdir,
):
    """
    Supplementary counterpart to the architecture-effect figure.

    Unlike the main figure, these panels show the actual observable values,
    without subtraction/division by the mean across architectures.

    For every architecture and composition:
      - coloured line/large point = mean across the four chemistries;
      - thin translucent error bar = 95% CI across chemistry-level values;
      - small chemistry-shaped points = the four actual chemistry values.

    This makes the absolute scale and chemistry-to-chemistry spread visible
    while retaining the thesis-wide architecture colour encoding.
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 7.8),
        squeeze=False,
    )
    axes = axes.ravel()

    x = np.array(
        [
            100 * HIGH_STICKINESS_FRACTION[r]
            for r in RATIO_ORDER
        ],
        dtype=float,
    )

    # Small horizontal offsets stop the four chemistry markers from sitting
    # exactly on top of one another, while preserving the composition position.
    family_offsets = {
        family: offset
        for family, offset in zip(
            FAMILY_ORDER,
            [-1.15, -0.38, 0.38, 1.15],
        )
    }

    for panel_idx, (ax, metric) in enumerate(
        zip(axes, METRICS)
    ):
        raw = metric["raw"]

        for architecture in ARCH_ORDER:
            means = []
            cis = []

            for ratio in RATIO_ORDER:
                values = (
                    chemistry_df[
                        (chemistry_df["ratio"] == ratio)
                        & (
                            chemistry_df["architecture_plot"]
                            == architecture
                        )
                    ][raw]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                mean, ci, n = mean_ci95(values)
                means.append(mean)
                cis.append(ci)

            means = np.asarray(means, dtype=float)
            cis = np.asarray(cis, dtype=float)

            # Chemistry-to-chemistry uncertainty around the architecture mean.
            ax.errorbar(
                x,
                means,
                yerr=cis,
                fmt="none",
                ecolor=ARCH_COLORS[architecture],
                elinewidth=0.75,
                capsize=0,
                alpha=0.25,
                zorder=1,
            )

            # Mean architecture trajectory across chemistries.
            ax.plot(
                x,
                means,
                color=ARCH_COLORS[architecture],
                linewidth=1.75,
                alpha=0.92,
                zorder=2,
            )

            ax.scatter(
                x,
                means,
                s=38,
                color=ARCH_COLORS[architecture],
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )

            # Actual chemistry-level values.
            for family in FAMILY_ORDER:
                d = chemistry_df[
                    (chemistry_df["family"] == family)
                    & (chemistry_df["architecture_plot"] == architecture)
                ]

                for ratio in RATIO_ORDER:
                    q = d[d["ratio"] == ratio]

                    if q.empty:
                        continue

                    y = q[raw].iloc[0]

                    if not np.isfinite(y):
                        continue

                    ax.scatter(
                        100 * HIGH_STICKINESS_FRACTION[ratio]
                        + family_offsets[family],
                        y,
                        s=19,
                        marker=FAMILY_MARKERS[family],
                        facecolor=ARCH_COLORS[architecture],
                        edgecolor="white",
                        linewidth=0.35,
                        alpha=0.58,
                        zorder=3,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [RATIO_TICK_LABELS[r] for r in RATIO_ORDER],
            fontsize=8.5,
            linespacing=1.25,
        )
        ax.set_xlim(10, 87)

        ax.set_xlabel(
            "Motif composition (A:B)",
            fontsize=9.5,
            labelpad=8,
        )
        ax.set_ylabel(
            metric["raw_ylabel"],
            fontsize=9.5,
        )
        ax.set_title(
            metric["panel_title"],
            fontsize=11,
            fontweight="bold",
            pad=7,
        )

        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="bottom",
        )

        ax.tick_params(
            axis="x",
            labelsize=8.5,
            pad=5,
        )
        ax.tick_params(
            axis="y",
            labelsize=8.5,
        )
        clean_axis(ax)

    architecture_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5.8,
            color=ARCH_COLORS[architecture],
            markerfacecolor=ARCH_COLORS[architecture],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=ARCH_LABELS[architecture].replace("\\n", " "),
        )
        for architecture in ARCH_ORDER
    ]

    chemistry_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="none",
            markersize=5.8,
            markerfacecolor="0.45",
            markeredgecolor="white",
            markeredgewidth=0.45,
            color="0.45",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]

    legend_arch = fig.legend(
        handles=architecture_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        title="Architecture (colour)",
        fontsize=8.4,
        title_fontsize=9,
        columnspacing=1.5,
        handlelength=2.2,
        handletextpad=0.55,
    )

    fig.add_artist(legend_arch)

    fig.legend(
        handles=chemistry_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        title="Chemistry (shape)",
        fontsize=8.4,
        title_fontsize=9,
        columnspacing=1.7,
        handletextpad=0.55,
    )

    fig.tight_layout(
        h_pad=2.05,
        w_pad=1.8,
        rect=[0, 0.165, 1, 1],
    )

    save_figure(
        fig,
        outdir,
        "figure_S_DR3_absolute_observables_across_composition",
    )


# ============================================================
# MAIN-THESIS SPLIT RATIO FIGURES
# R1-R3 and R4-R6
# ============================================================

def figure_ratio_matrix_subset(
    chemistry_df,
    replicate_df,
    outdir,
    ratios,
    filename,
):
    """
    Main-thesis architecture-effect figure for a subset of ratios.

    Recommended use:
        R1-R3 -> first composition-regime figure
        R4-R6 -> second composition-regime figure

    Visual encoding is thesis-wide and fixed:
        colour = architecture
        shape  = chemistry

    Large coloured points/error bars are means across the four chemistries.
    Smaller semi-transparent points are the four chemistry-level observations.
    For R4, tiny open points additionally show the underlying replicate-level
    architecture effects without giving R4 extra statistical weight.
    """

    n_rows = len(METRICS)
    n_cols = len(ratios)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12.4, 12.0),
        sharey="row",
        squeeze=False,
    )

    family_offsets = {
        "VGIP": -0.12,
        "VGIG": -0.04,
        "VAIP": +0.04,
        "VAIG": +0.12,
    }

    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    panel_counter = 0

    for row, metric in enumerate(METRICS):
        effect = metric["effect"]

        for col, ratio in enumerate(ratios):
            ax = axes[row, col]

            ratio_df = chemistry_df[
                chemistry_df["ratio"] == ratio
            ]

            for i, architecture in enumerate(ARCH_ORDER):
                arch = ratio_df[
                    ratio_df["architecture_plot"] == architecture
                ]

                # Chemistry-level observations: shape = chemistry,
                # colour = architecture.
                for family in FAMILY_ORDER:
                    point = arch[arch["family"] == family]

                    if len(point) == 0:
                        continue

                    y = point[effect].iloc[0]
                    if not np.isfinite(y):
                        continue

                    ax.scatter(
                        i + family_offsets[family],
                        y,
                        marker=FAMILY_MARKERS[family],
                        s=34,
                        color=ARCH_COLORS[architecture],
                        alpha=0.48,
                        edgecolor="none",
                        zorder=3,
                    )

                    # R4 has triplicates. Show them only as small open
                    # architecture-coloured chemistry markers.
                    if ratio == "R4":
                        rep_points = replicate_df[
                            (replicate_df["ratio"] == ratio)
                            & (replicate_df["family"] == family)
                            & (
                                replicate_df["architecture_plot"]
                                == architecture
                            )
                        ]

                        rep_values = (
                            rep_points[effect]
                            .replace([np.inf, -np.inf], np.nan)
                            .dropna()
                            .to_numpy()
                        )

                        if len(rep_values) > 1:
                            rep_jitter = np.linspace(
                                -0.025,
                                +0.025,
                                len(rep_values),
                            )

                            for jitter, rep_value in zip(
                                rep_jitter,
                                rep_values,
                            ):
                                ax.scatter(
                                    i
                                    + family_offsets[family]
                                    + jitter,
                                    rep_value,
                                    marker=FAMILY_MARKERS[family],
                                    s=14,
                                    facecolors="none",
                                    edgecolors=ARCH_COLORS[architecture],
                                    linewidths=0.75,
                                    alpha=0.72,
                                    zorder=2,
                                )

                # Architecture mean across the four chemistries.
                values = (
                    arch[effect]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                mean, ci, n = mean_ci95(values)

                if np.isfinite(mean):
                    ax.errorbar(
                        i,
                        mean,
                        yerr=ci,
                        fmt="o",
                        markersize=7.5,
                        color=ARCH_COLORS[architecture],
                        ecolor=ARCH_COLORS[architecture],
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        capsize=3,
                        linewidth=1.5,
                        zorder=10,
                    )

            ax.axhline(
                0,
                linestyle="--",
                linewidth=0.9,
                color="0.55",
                zorder=0,
            )

            ax.set_xticks(np.arange(len(ARCH_ORDER)))

            if row == n_rows - 1:
                ax.set_xticklabels(
                    [ARCH_LABELS[x] for x in ARCH_ORDER],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
            else:
                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(
                    RATIO_LABELS[ratio],
                    fontweight="bold",
                    fontsize=12,
                    pad=8,
                )

            if col == 0:
                ax.set_ylabel(
                    metric["ylabel"],
                    fontsize=10,
                )

            ax.text(
                -0.10,
                1.035,
                panel_letters[panel_counter],
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )
            panel_counter += 1

            clean_axis(ax)

    # Architecture = colour.
    architecture_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=ARCH_COLORS[architecture],
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=ARCH_LABELS[architecture].replace("\n", " "),
        )
        for architecture in ARCH_ORDER
    ]

    # Chemistry = shape only. Neutral grey prevents the legend from
    # implying that chemistry has its own colour.
    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="none",
            markersize=7,
            markerfacecolor="0.45",
            markeredgecolor="0.45",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]

    fig.legend(
        handles=architecture_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        title="Architecture (colour)",
        fontsize=8.5,
        title_fontsize=9,
    )

    fig.legend(
        handles=family_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        title="Chemistry (shape)",
        fontsize=8.5,
        title_fontsize=9,
    )

    fig.tight_layout(
        h_pad=1.15,
        w_pad=0.85,
        rect=[0.02, 0.115, 1, 1],
    )

    save_figure(
        fig,
        outdir,
        filename,
    )


def figure_ratio_matrix_split_main_thesis(
    chemistry_df,
    replicate_df,
    outdir,
):
    """Create the two recommended main-thesis DR3 ratio figures."""

    figure_ratio_matrix_subset(
        chemistry_df,
        replicate_df,
        outdir,
        ratios=["R1", "R2", "R3"],
        filename="figure_DR3_architecture_effects_R1_R3_main",
    )

    figure_ratio_matrix_subset(
        chemistry_df,
        replicate_df,
        outdir,
        ratios=["R4", "R5", "R6"],
        filename="figure_DR3_architecture_effects_R4_R6_main",
    )


# ============================================================
# FIGURE 3
# OPTIONAL R4 DETAIL
# ============================================================

def figure_R4_replicates(
    replicate_df,
    outdir,
):

    """
    Dedicated R4 figure.

    Here we explicitly show all replicate-level observations.

    For each architecture:
        small points = individual replicate effects
        large point = mean across all chemistry-level means

    This figure is useful for checking whether the R4 architecture
    effects are reproducible across the triplicate simulations.
    """

    ratio = "R4"

    df = replicate_df[
        replicate_df["ratio"]
        == ratio
    ].copy()

    if len(df) == 0:

        print(
            "No R4 data available; "
            "skipping R4 replicate figure."
        )

        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes = axes.flatten()

    family_offsets = {
        "VGIP": -0.18,
        "VGIG": -0.06,
        "VAIP": +0.06,
        "VAIG": +0.18,
    }

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    for (
        ax,
        metric,
        letter,
    ) in zip(
        axes,
        METRICS,
        letters,
    ):

        effect = metric["effect"]

        for i, architecture in enumerate(
            ARCH_ORDER
        ):

            arch = df[
                df[
                    "architecture_plot"
                ]
                == architecture
            ]

            chemistry_means = []

            for family in FAMILY_ORDER:

                family_data = arch[
                    arch["family"]
                    == family
                ]

                values = (
                    family_data[
                        effect
                    ]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .dropna()
                    .to_numpy()
                )

                if len(values) == 0:
                    continue

                x_center = (
                    i
                    +
                    family_offsets[
                        family
                    ]
                )

                jitter = np.linspace(
                    -0.025,
                    +0.025,
                    len(values),
                )

                for j, value in enumerate(
                    values
                ):

                    ax.scatter(
                        x_center
                        +
                        jitter[j],
                        value,
                        marker=FAMILY_MARKERS[family],
                        s=24,
                        color=ARCH_COLORS[architecture],
                        alpha=0.45,
                        edgecolors="none",
                    )

                chemistry_mean = (
                    np.mean(values)
                )

                chemistry_means.append(
                    chemistry_mean
                )

                ax.scatter(
                    x_center,
                    chemistry_mean,
                    marker=FAMILY_MARKERS[family],
                    s=55,
                    facecolors="none",
                    edgecolors=ARCH_COLORS[architecture],
                    linewidths=1.1,
                    zorder=6,
                )

            # -----------------------------------------------
            # Final mean across chemistry means
            # -----------------------------------------------

            mean, ci, n = mean_ci95(
                chemistry_means
            )

            if np.isfinite(mean):

                ax.errorbar(
                    i,
                    mean,
                    yerr=ci,
                    fmt="o",
                    markersize=8,
                    color=ARCH_COLORS[architecture],
                    ecolor=ARCH_COLORS[architecture],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    capsize=4,
                    linewidth=1.5,
                    zorder=10,
                )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="0.45",
        )

        ax.set_xticks(
            np.arange(
                len(
                    ARCH_ORDER
                )
            )
        )

        ax.set_xticklabels(
            [
                ARCH_LABELS[x]
                for x in ARCH_ORDER
            ],
            rotation=42,
            ha="right",
        )

        ax.set_ylabel(
            metric[
                "ylabel"
            ]
        )

        ax.set_title(
            (
                f"{letter}   "
                f"{metric['panel_title']}"
            ),
            loc="left",
            fontweight="bold",
        )

        clean_axis(ax)

    fig.tight_layout(
        h_pad=2.3,
        w_pad=2.0,
    )

    save_figure(
        fig,
        outdir,
        "figure_DR3_R4_triplicate_detail",
    )



# ============================================================
# EXTERNAL R4 REPLICATE DATA
# ============================================================

def prepare_external_R4_data(csv_path, temperature="370K", chunksize=500_000):
    """
    Build the dedicated R4 replicate table directly from frame-level
    all_rg_values.csv. The original R1-R6 analysis is not changed.

    Only the 32 DR3 R4 designs are retained:
        VGIP  055-062
        VGIG  103-110
        VAIP  151-158
        VAIG  199-206

    Metrics are calculated independently for each id x replicate:
        central 90% Rg range = p95 - p05
        fraction collapsed
        mean compact-state Rg
        k_collapse = N(expanded -> compact) / total expanded time
        k_escape   = N(compact -> expanded) / total compact time

    The large CSV is read in chunks so the complete 9.6-million-row file
    does not have to be held in memory before filtering.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dedicated R4 all-Rg file:\n{csv_path}")

    required_cols = [
        "id", "temp", "replicate", "analysed_index", "rg_nm",
        "collapsed", "threshold_nm", "frame_dt_ns",
    ]

    family_ranges = {
        "VGIP": (55, 62),
        "VGIG": (103, 110),
        "VAIP": (151, 158),
        "VAIG": (199, 206),
    }

    # The eight IDs in each family are deliberately ordered identically.
    # Mapping by numeric offset avoids ambiguity in names such as
    # VGcenter_IPends, which means the higher-lambda IP motif is at the ends.
    architecture_by_offset = {
        0: "even",
        1: "small blocks",
        2: "medium blocks",
        3: "large blocks",
        4: "higher-lambda center",
        5: "higher-lambda endcaps",
        6: "diblock",
        7: "random shuffle",
    }

    print()
    print("=" * 72)
    print("DEDICATED R4 FRAME-LEVEL ANALYSIS")
    print("=" * 72)
    print(f"Input       : {csv_path}")
    print(f"Temperature : {temperature}")
    print(f"Chunk size  : {chunksize:,}")

    kept_chunks = []
    available_temperatures = set()
    total_rows = 0
    retained_rows = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=lambda c: c in required_cols,
        chunksize=chunksize,
    ):
        total_rows += len(chunk)

        missing = [c for c in required_cols if c not in chunk.columns]
        if missing:
            raise ValueError(
                "all_rg_values.csv is missing required columns: "
                + ", ".join(missing)
            )

        available_temperatures.update(
            chunk["temp"].dropna().astype(str).str.strip().unique().tolist()
        )

        numeric_id = pd.to_numeric(
            chunk["id"].astype(str).str.extract(r"^(\d+)", expand=False),
            errors="coerce",
        )

        r4_mask = pd.Series(False, index=chunk.index)
        for lo, hi in family_ranges.values():
            r4_mask |= numeric_id.between(lo, hi)

        temp_mask = (
            chunk["temp"].astype(str).str.strip().str.upper()
            == str(temperature).strip().upper()
        )

        sub = chunk.loc[r4_mask & temp_mask].copy()
        if not sub.empty:
            kept_chunks.append(sub)
            retained_rows += len(sub)

    if not kept_chunks:
        temps = ", ".join(sorted(available_temperatures)) or "<none>"
        raise ValueError(
            f"No DR3 R4 rows found at {temperature}.\n"
            f"Temperatures present in the file: {temps}\n"
            "Use --r4-temperature to select the intended temperature."
        )

    frames = pd.concat(kept_chunks, ignore_index=True)

    print(f"Rows scanned : {total_rows:,}")
    print(f"Rows retained: {retained_rows:,}")
    print(
        "Temperatures present: "
        + ", ".join(sorted(available_temperatures))
    )

    # Assign family and architecture from the exact R4 numeric ID ranges.
    numeric_id = pd.to_numeric(
        frames["id"].astype(str).str.extract(r"^(\d+)", expand=False),
        errors="coerce",
    )
    frames["family"] = pd.Series(pd.NA, index=frames.index, dtype="string")
    frames["architecture_plot"] = pd.Series(
        pd.NA, index=frames.index, dtype="string"
    )

    for family, (lo, hi) in family_ranges.items():
        mask = numeric_id.between(lo, hi)
        frames.loc[mask, "family"] = family
        offsets = numeric_id.loc[mask].astype(int) - lo
        frames.loc[mask, "architecture_plot"] = offsets.map(
            architecture_by_offset
        ).to_numpy()

    frames["ratio"] = "R4"
    frames["_replicate_plot"] = frames["replicate"].astype(str)
    frames["rg_nm"] = pd.to_numeric(frames["rg_nm"], errors="coerce")
    frames["analysed_index"] = pd.to_numeric(
        frames["analysed_index"], errors="coerce"
    )
    frames["frame_dt_ns"] = pd.to_numeric(
        frames["frame_dt_ns"], errors="coerce"
    )

    # Robust conversion in case pandas reads collapsed as strings.
    if frames["collapsed"].dtype != bool:
        frames["collapsed"] = (
            frames["collapsed"].astype(str).str.strip().str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

    rows = []
    grouping = [
        "family", "id", "architecture_plot", "ratio", "_replicate_plot"
    ]

    for keys, run in frames.groupby(grouping, sort=False, dropna=False):
        family, seq_id, architecture, ratio, replicate = keys
        run = run.sort_values("analysed_index")

        rg = run["rg_nm"].to_numpy(dtype=float)
        compact = run["collapsed"].to_numpy(dtype=bool)
        dt_values = run["frame_dt_ns"].dropna().to_numpy(dtype=float)

        if len(rg) == 0 or len(dt_values) == 0:
            continue

        dt = float(dt_values[0])
        if not np.allclose(dt_values, dt, rtol=0, atol=1e-12):
            raise ValueError(
                f"frame_dt_ns is not constant for {seq_id}, {replicate}."
            )

        p05, p95 = np.percentile(rg, [5, 95])
        fraction_collapsed = float(np.mean(compact))
        mean_rg_collapsed = (
            float(np.mean(rg[compact])) if np.any(compact) else np.nan
        )

        if len(compact) > 1:
            n_collapse = int(np.sum((~compact[:-1]) & compact[1:]))
            n_escape = int(np.sum(compact[:-1] & (~compact[1:])))
        else:
            n_collapse = 0
            n_escape = 0

        total_collapsed_time = float(np.sum(compact) * dt)
        total_expanded_time = float(np.sum(~compact) * dt)

        k_collapse = (
            n_collapse / total_expanded_time
            if total_expanded_time > 0 else np.nan
        )
        k_escape = (
            n_escape / total_collapsed_time
            if total_collapsed_time > 0 else np.nan
        )

        rows.append({
            "family": family,
            "id": seq_id,
            "architecture_plot": architecture,
            "ratio": ratio,
            "_replicate_plot": replicate,
            "n_frames": len(rg),
            "frame_dt_ns": dt,
            "rg_range90_nm": float(p95 - p05),
            "fraction_collapsed": fraction_collapsed,
            "mean_rg_collapsed_nm": mean_rg_collapsed,
            "n_collapse_transitions": n_collapse,
            "n_escape_transitions": n_escape,
            "total_collapsed_time_ns": total_collapsed_time,
            "total_expanded_time_ns": total_expanded_time,
            "k_collapse_per_ns": float(k_collapse),
            "k_escape_per_ns": float(k_escape),
        })

    summary_df = pd.DataFrame(rows)

    print()
    print("Frame-level summary created:")
    print(f"  rows (id x replicate): {len(summary_df)}")
    print(f"  unique designs       : {summary_df['id'].nunique()}")
    print(f"  unique replicates    : {summary_df['_replicate_plot'].nunique()}")
    print()
    print("Runs per family x architecture:")
    print(
        summary_df.groupby(["family", "architecture_plot"])["_replicate_plot"]
        .nunique()
        .unstack(0)
        .reindex(ARCH_ORDER)
    )

    expected_designs = 32
    if summary_df["id"].nunique() != expected_designs:
        print(
            f"WARNING: expected {expected_designs} R4 designs but found "
            f"{summary_df['id'].nunique()}."
        )

    # Architecture effects are normalized within family + replicate.
    effect_df = calculate_architecture_effects(summary_df)
    chemistry_df = make_chemistry_level_table(effect_df)
    return summary_df, effect_df, chemistry_df

def figure_R4_external_replicates(replicate_df, chemistry_df, outdir):
    """
    Dedicated R4-only 2x2 figure derived from frame-level all_rg_values.csv.

    Tiny filled points  = individual independent runs.
    Open shape          = mean of runs within one chemistry.
    Large filled point  = mean across the four chemistry means.
    Error bar           = 95% CI across the four chemistry means.

    Architecture effects are already normalized within chemistry + run.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.3), squeeze=False)
    axes = axes.ravel()

    family_offsets = {
        "VGIP": -0.18,
        "VGIG": -0.06,
        "VAIP": +0.06,
        "VAIG": +0.18,
    }

    for panel_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        effect = metric["effect"]

        for i, architecture in enumerate(ARCH_ORDER):
            arch_rep = replicate_df[
                replicate_df["architecture_plot"] == architecture
            ]
            arch_chem = chemistry_df[
                chemistry_df["architecture_plot"] == architecture
            ]

            # Individual run-level effects, grouped visually by chemistry.
            for family in FAMILY_ORDER:
                run_values = (
                    arch_rep.loc[arch_rep["family"] == family, effect]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if len(run_values):
                    jitter = (
                        np.linspace(-0.028, +0.028, len(run_values))
                        if len(run_values) > 1
                        else np.array([0.0])
                    )
                    for dx, value in zip(jitter, run_values):
                        ax.scatter(
                            i + family_offsets[family] + dx,
                            value,
                            marker=FAMILY_MARKERS[family],
                            s=23,
                            color=ARCH_COLORS[architecture],
                            alpha=0.34,
                            edgecolor="none",
                            zorder=2,
                        )

                # Chemistry-level mean after averaging its runs.
                chem_point = arch_chem[arch_chem["family"] == family]
                if len(chem_point):
                    chem_mean = chem_point[effect].iloc[0]
                    if np.isfinite(chem_mean):
                        ax.scatter(
                            i + family_offsets[family],
                            chem_mean,
                            marker=FAMILY_MARKERS[family],
                            s=58,
                            facecolors="none",
                            edgecolors=ARCH_COLORS[architecture],
                            linewidths=1.25,
                            zorder=6,
                        )

            # Final architecture mean across the four chemistry-level means.
            chemistry_values = (
                arch_chem[effect]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            mean, ci, n = mean_ci95(chemistry_values)

            if np.isfinite(mean):
                ax.errorbar(
                    i,
                    mean,
                    yerr=ci,
                    fmt="o",
                    markersize=8.2,
                    color=ARCH_COLORS[architecture],
                    ecolor=ARCH_COLORS[architecture],
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    capsize=3.5,
                    linewidth=1.5,
                    zorder=10,
                )

        ax.axhline(0, linestyle="--", linewidth=0.9, color="0.52", zorder=0)
        ax.set_xticks(np.arange(len(ARCH_ORDER)))
        ax.set_xticklabels(
            [ARCH_LABELS[a] for a in ARCH_ORDER],
            rotation=43,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(metric["ylabel"], fontsize=9.5)
        ax.set_title(metric["panel_title"], fontsize=11, fontweight="bold", pad=7)
        ax.text(
            -0.11, 1.04, chr(ord("A") + panel_idx),
            transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom",
        )
        clean_axis(ax)

    architecture_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6.5,
            markerfacecolor=ARCH_COLORS[a], markeredgecolor="white",
            markeredgewidth=0.5, label=ARCH_LABELS[a].replace("\n", " ")
        )
        for a in ARCH_ORDER
    ]
    family_handles = [
        plt.Line2D(
            [0], [0], marker=FAMILY_MARKERS[f], linestyle="none", markersize=6.5,
            markerfacecolor="none", markeredgecolor="0.35",
            markeredgewidth=1.1, label=FAMILY_LABELS[f]
        )
        for f in FAMILY_ORDER
    ]

    legend_arch = fig.legend(
        handles=architecture_handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.047), title="Architecture (colour)", fontsize=8.4,
        title_fontsize=9,
    )
    fig.add_artist(legend_arch)
    fig.legend(
        handles=family_handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.002), title="Chemistry (shape)", fontsize=8.4,
        title_fontsize=9,
    )

    fig.tight_layout(h_pad=1.7, w_pad=1.7, rect=[0, 0.145, 1, 1])
    save_figure(fig, outdir, "figure_DR3_R4_independent_runs_detail")


# ============================================================
# SAVE EFFECT TABLES
# ============================================================

def save_tables(
    replicate_df,
    chemistry_df,
    outdir,
):

    replicate_file = (
        outdir
        /
        "DR3_architecture_effects_replicate_level.csv"
    )

    chemistry_file = (
        outdir
        /
        "DR3_architecture_effects_chemistry_level.csv"
    )

    replicate_df.to_csv(
        replicate_file,
        index=False,
    )

    chemistry_df.to_csv(
        chemistry_file,
        index=False,
    )

    print(
        f"Saved: {replicate_file}"
    )

    print(
        f"Saved: {chemistry_file}"
    )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# DR3 PRINT-READY THREE-FIGURE WORKFLOW
# ============================================================
PRINT_METRICS = [m for m in METRICS if m['effect'] in ('log2_fold_kcollapse', 'log2_fold_kescape', 'effect_collapsed_rg_nm')]
PRINT_METRICS = sorted(PRINT_METRICS, key=lambda m: ['log2_fold_kcollapse','log2_fold_kescape','effect_collapsed_rg_nm'].index(m['effect']))
PRINT_METRICS = [dict(m, ylabel=(
    r'$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$' if m['effect']=='log2_fold_kcollapse' else
    r'$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$' if m['effect']=='log2_fold_kescape' else
    r'$\Delta\,\overline{R}_{g,\mathrm{compact}}$ (nm)')) for m in PRINT_METRICS]

def print_style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':18, 'axes.titlesize':19,
      'axes.titleweight':'bold', 'axes.labelsize':17, 'xtick.labelsize':17,
      'ytick.labelsize':17, 'legend.fontsize':17, 'axes.linewidth':1.5,
      'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.bbox':'tight'})

def format_panel(ax, metric, letter, title=None):
    ax.axhline(0, color='#777777', linewidth=1.4, linestyle='--', zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=1.2, length=4)
    ax.set_ylabel(metric['ylabel'], labelpad=8)
    if title: ax.set_title(title, pad=16)
    ax.text(-0.13, 1.055, letter, transform=ax.transAxes, fontsize=24,
            fontweight='bold', va='bottom')

def save_print(fig, outdir, stem, dpi):
    fig.savefig(outdir / (stem+'.pdf'), bbox_inches='tight', pad_inches=.23)
    fig.savefig(outdir / (stem+'.png'), dpi=dpi, bbox_inches='tight', pad_inches=.23)
    fig.savefig(outdir / (stem+'.svg'), bbox_inches='tight', pad_inches=.23)
    plt.close(fig)
    print('Saved', stem)

def print_legend(fig, y=.015):
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=ARCH_COLORS[a], lw=3.0, marker='o', markersize=7,
                      label=ARCH_LABELS[a].replace('\n',' ')) for a in ARCH_ORDER]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,y),
               ncol=4, frameon=False, columnspacing=1.7, handlelength=2.2)

def figure_1_ratios(chemistry_df, outdir, dpi):
    """Four chemistry means per architecture; each pools six ratios equally.

    Pale dots = six ratio effects; outlined marker = chemistry mean;
    faint vertical whisker = min-max across ratios (NOT a CI).
    """
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,1,figsize=(13.2,13.1),sharex=True)
    offsets=np.linspace(-.24,.24,4)
    rows=[]
    for j,(ax,m) in enumerate(zip(axes,PRINT_METRICS)):
        key=m['effect']
        format_panel(ax,m,'ABC'[j],['Compact-state entry','Compact-state escape','Compact-state dimensions'][j])
        for ia,arch in enumerate(ARCH_ORDER):
            for ic,fam in enumerate(FAMILY_ORDER):
                sub=chemistry_df.loc[(chemistry_df.family==fam)&(chemistry_df.architecture_plot==arch)]
                vals=[]
                for ir,ratio in enumerate(RATIO_ORDER):
                    z=pd.to_numeric(sub.loc[sub.ratio==ratio,key],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                    if len(z):
                        v=float(z.mean());vals.append(v)
                        ax.scatter(ia+offsets[ic]+(ir-2.5)*.011,v,s=29,color=ARCH_COLORS[arch],alpha=.26,zorder=2)
                if vals:
                    x=ia+offsets[ic];avg=float(np.mean(vals))
                    ax.vlines(x,min(vals),max(vals),color=ARCH_COLORS[arch],lw=2.0,alpha=.30,zorder=2)
                    ax.scatter(x,avg,s=115,marker=FAMILY_MARKERS[fam],facecolors='white',
                               edgecolors=ARCH_COLORS[arch],linewidths=2.2,zorder=5)
                    rows.append({'family':fam,'architecture':arch,'metric':key,'mean_R1_R6':avg,
                                 'min_ratio':min(vals),'max_ratio':max(vals),'n_ratios':len(vals)})
                    if len(vals)!=6:print('WARNING: missing ratios',fam,arch,key,len(vals))
        ax.set_xlim(-.65,7.65);expand_y(ax,.22)
    thesis_axis_layout(fig,axes,bottom=.26,top=.94)
    legend=[Line2D([0],[0],marker=FAMILY_MARKERS[f],color='0.25',lw=0,
        markerfacecolor='white',markersize=10,label=FAMILY_LABELS[f]) for f in FAMILY_ORDER]
    legend.append(Line2D([0],[0],marker='o',color='0.6',lw=0,alpha=.4,markersize=7,label='Individual ratio (R1–R6)'))
    legend.append(Line2D([0],[0],color='0.6',lw=2.5,alpha=.5,label='Ratio range (min–max)'))
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.018),
               ncol=3,frameon=False,fontsize=13,columnspacing=1.5)
    save_print(fig,outdir,'01_DR3_chemistry_architecture_effects_pooled_ratios',dpi)
    pd.DataFrame(rows).to_csv(outdir/'01_DR3_chemistry_means_spread_R1_R6.csv',index=False)

def thesis_axis_layout(fig, axes, bottom=.21, top=.95):
    """Shared x labels appear only below panel C; reserve space for legend."""
    for ax in axes[:-1]:
        ax.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
    axes[-1].set_xticks(np.arange(len(ARCH_ORDER)))
    axes[-1].set_xticklabels([ARCH_LABELS[a].replace('\n',' ') for a in ARCH_ORDER],
                             rotation=28,ha='right',fontsize=14)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(),fontsize=16,labelpad=11)
        ax.set_title(ax.get_title(),fontsize=19,pad=13)
        ax.tick_params(axis='y',labelsize=14)
    fig.subplots_adjust(left=.19,right=.975,top=top,bottom=bottom,hspace=.44)

def pooled_table(chemistry_df):
    # Every ratio has equal weight WITHIN chemistry; every chemistry equal weight overall.
    keys=['family','architecture_plot']
    agg={m['effect']:'mean' for m in PRINT_METRICS}
    pooled=chemistry_df.groupby(keys,as_index=False).agg(agg)
    counts=chemistry_df.groupby(keys)['ratio'].nunique().rename('n_ratios').reset_index()
    pooled=pooled.merge(counts,on=keys)
    if (pooled.n_ratios!=6).any():
        print('WARNING: some chemistry x architecture groups have fewer than six ratios; see pooled CSV')
    return pooled

def expand_y(ax, fraction=.24):
    """Increase headroom and footroom without clipping points or error bars."""
    lo, hi = ax.get_ylim()
    mid = (lo + hi) / 2
    half = (hi - lo) / 2 * (1 + fraction)
    ax.set_ylim(mid - half, mid + half)


def figure_2_pooled(chemistry_df,outdir,dpi):
    """MAIN PLOT: x = six motif ratios; eight architecture lines, averaged over four chemistries."""
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,1,figsize=(12.5,12.8),sharex=True)
    x=np.arange(6,dtype=float)
    records=[]
    for j,(ax,m) in enumerate(zip(axes,PRINT_METRICS)):
        key=m['effect']
        format_panel(ax,m,'ABC'[j],['Compact-state entry','Compact-state escape','Compact-state dimensions'][j])
        for arch in ARCH_ORDER:
            means=[];cis=[]
            for ir,ratio in enumerate(RATIO_ORDER):
                sub=chemistry_df.loc[(chemistry_df.architecture_plot==arch)&(chemistry_df.ratio==ratio)]
                vals=[]
                for ic,fam in enumerate(FAMILY_ORDER):
                    z=pd.to_numeric(sub.loc[sub.family==fam,key],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                    if len(z):vals.append(float(z.mean()))
                avg,ci,n=mean_ci95(np.asarray(vals,float))
                means.append(avg);cis.append(ci)
                records.append({'architecture':arch,'ratio':ratio,'metric':key,
                    'mean_four_chemistries':avg,'ci95_chemistries':ci,'n_chemistries':n})
                if n!=4:print('WARNING: fewer than four chemistries',arch,ratio,key,n)
            color=ARCH_COLORS[arch]
            ax.errorbar(x,means,yerr=cis,color=color,linewidth=3.1,marker='o',
                        markersize=8.5,markeredgecolor='white',markeredgewidth=.7,
                        elinewidth=1.5,capsize=3,capthick=1.4,zorder=3,label=ARCH_LABELS[arch].replace('\n',' '))
        ax.set_xlim(-.2,5.2);expand_y(ax,.18)
        ax.tick_params(axis='y',labelsize=15)
    for ax in axes[:-1]:ax.tick_params(axis='x',bottom=False,labelbottom=False)
    axes[-1].set_xticks(x,[RATIO_LABELS[r] for r in RATIO_ORDER],fontsize=17)
    axes[-1].set_xlabel('Motif composition (A:B)',fontsize=18,labelpad=11)
    fig.subplots_adjust(left=.17,right=.98,top=.945,bottom=.25,hspace=.49)
    handles=[Line2D([0],[0],color=ARCH_COLORS[a],lw=3.1,marker='o',markersize=8,
      label=ARCH_LABELS[a].replace('\n',' ')) for a in ARCH_ORDER]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=4,
        frameon=False,fontsize=14,title='Architecture',title_fontsize=15,
        columnspacing=1.1,handlelength=1.9)
    save_print(fig,outdir,'02_DR3_MAIN_architecture_trends_across_six_ratios',dpi)
    pd.DataFrame(records).to_csv(outdir/'02_DR3_MAIN_ratio_means_and_CI.csv',index=False)

def figure_3_r4(replicate_df,chemistry_df,outdir,dpi):
    rep=replicate_df.loc[replicate_df.ratio=='R4'].copy() if 'ratio' in replicate_df.columns else replicate_df.copy()
    chem=chemistry_df.loc[chemistry_df.ratio=='R4'].copy() if 'ratio' in chemistry_df.columns else chemistry_df.copy()
    if rep.empty: raise ValueError('No R4 replicate data found.')
    rep_counts=rep.groupby(['family','architecture_plot'])['_replicate_plot'].nunique() if '_replicate_plot' in rep else rep.groupby(['family','architecture_plot']).size()
    if rep_counts.min()<3:
        print('WARNING: fewer than three independent R4 runs for at least one family/architecture.')
    fig,axes=plt.subplots(3,1,figsize=(13.2,13.1),sharex=True)
    offsets={'VGIP':-.18,'VGIG':-.06,'VAIP':.06,'VAIG':.18}
    records=[]
    for j,(ax,m) in enumerate(zip(axes,PRINT_METRICS)):
        key=m['effect']
        for i,a in enumerate(ARCH_ORDER):
            group=chem.loc[chem.architecture_plot==a]
            for family in FAMILY_ORDER:
                raw=rep.loc[(rep.architecture_plot==a)&(rep.family==family),key]
                vals=pd.to_numeric(raw,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
                for dx,v in zip(np.linspace(-.025,.025,len(vals)),vals):
                    ax.scatter(i+offsets[family]+dx,v,marker=FAMILY_MARKERS[family],
                         s=32,color=ARCH_COLORS[a],alpha=.30,edgecolors='none',zorder=2)
                chemvals=group.loc[group.family==family,key].replace([np.inf,-np.inf],np.nan).dropna()
                if len(chemvals):
                    ax.scatter(i+offsets[family],float(chemvals.iloc[0]),marker=FAMILY_MARKERS[family],
                         s=115,facecolor='white',edgecolor=ARCH_COLORS[a],linewidth=2,zorder=4)
            vals=group[key].replace([np.inf,-np.inf],np.nan).dropna().to_numpy()
            mean,ci,_=mean_ci95(vals)
            ax.errorbar(i,mean,yerr=ci,fmt='o',ms=11,color=ARCH_COLORS[a],
                        mec='black',mew=.8,elinewidth=2.3,capsize=5,zorder=6)
            records.append({'architecture':a,'metric':key,'mean':mean,'ci95_chemistry':ci,'n_chemistries':len(vals)})
        format_panel(ax,m,'ABC'[j],['Compact-state entry','Compact-state escape','Compact-state dimensions'][j])
        ax.set_xlim(-.55,7.55)
        expand_y(ax,.20)
    from matplotlib.lines import Line2D
    markers=[Line2D([0],[0],marker=FAMILY_MARKERS[f],color='black',markerfacecolor='none',
                    lw=0,markersize=9,label=FAMILY_LABELS[f]) for f in FAMILY_ORDER]
    markers.extend([Line2D([0],[0],marker='o',color='grey',lw=0,markersize=6,alpha=.5,label='Individual run'),
                    Line2D([0],[0],marker='o',color='black',markerfacecolor='white',lw=0,markersize=9,label='Chemistry mean'),
                    Line2D([0],[0],marker='o',color='black',lw=0,markersize=10,label='Overall mean ± 95% CI')])
    thesis_axis_layout(fig,axes,bottom=.26,top=.94)
    fig.legend(handles=markers,loc='lower center',bbox_to_anchor=(.5,.012),ncol=3,frameon=False,fontsize=12.5)
    save_print(fig,outdir,'03_DR3_R4_independent_replicate_validation',dpi)
    pd.DataFrame(records).to_csv(outdir/'03_DR3_R4_means_and_CI.csv',index=False)

def main():
    p=argparse.ArgumentParser(description='DR3: three large-font print-ready thesis figures.')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--root',help='Parent folder of VGIP/VGIG/VAIP/VAIG_detailed_stride1')
    g.add_argument('--input',help='Combined rg_state_summary_370K.csv')
    p.add_argument('--outdir',required=True)
    p.add_argument('--r4-all-rg',help='Frame-level all_rg_values.csv containing three independent R4 runs')
    p.add_argument('--r4-temperature',default='370K')
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    print_style()
    if args.root:
        folders={'VGIP':'VGIP_detailed_stride1','VGIG':'VGIG_detailed_stride1',
                 'VAIP':'VAIP_detailed_stride1','VAIG':'VAIG_detailed_stride1'}
        dfs=[]
        for fam,folder in folders.items():
            f=Path(args.root)/folder/'rg_state_summary_370K.csv'
            if not f.exists(): raise FileNotFoundError(f)
            tmp=pd.read_csv(f);tmp['family']=fam;dfs.append(tmp)
        df=pd.concat(dfs,ignore_index=True)
    else: df=pd.read_csv(args.input)
    df=prepare_data(df)
    rep=calculate_architecture_effects(df)
    chem=make_chemistry_level_table(rep)
    figure_1_ratios(chem,outdir,args.dpi)
    figure_2_pooled(chem,outdir,args.dpi)
    if args.r4_all_rg and not Path(args.r4_all_rg).is_file():
        print('WARNING: R4 replicate CSV not found:', args.r4_all_rg)
        print('Skipping Figure 3; Figures 1 and 2 are already saved.')
        args.r4_all_rg = None
    if args.r4_all_rg:
        raw_r4,rep_r4,chem_r4=prepare_external_R4_data(args.r4_all_rg,temperature=args.r4_temperature)
        raw_r4.to_csv(outdir/'03_DR3_R4_frame_level_summary.csv',index=False)
        rep_r4.to_csv(outdir/'03_DR3_R4_run_effects.csv',index=False)
        chem_r4.to_csv(outdir/'03_DR3_R4_chemistry_effects.csv',index=False)
        figure_3_r4(rep_r4,chem_r4,outdir,args.dpi)
    else:
        if '_replicate_plot' in rep and rep.loc[rep.ratio=='R4','_replicate_plot'].nunique()>=3:
            figure_3_r4(rep,chem,outdir,args.dpi)
        else:
            print('Figure 3 skipped: provide --r4-all-rg with independent R4 runs.')
    print('Done. Outputs:',outdir)

if __name__=='__main__':
    main()
