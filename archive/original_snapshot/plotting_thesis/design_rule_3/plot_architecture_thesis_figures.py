"""
plot_architecture_thesis_figures.py

Final thesis plotting script for architecture design rules.

Produces four thesis-ready figures:

FIGURE 1 — Overall architecture design effects
    A  Central 90% Rg range
    B  k_collapse
    C  k_escape
    D  Mean compact-state Rg

FIGURE 2 — Architecture effects by composition regime
    Rows:
        Central 90% Rg range
        k_collapse
        k_escape
        Mean compact-state Rg
    Columns:
        Low    R1-R2
        Middle R3-R4
        High   R5-R6

FIGURE 3 — Conservation across motif chemistries
    Rows:
        Central 90% Rg range
        k_collapse
        k_escape
        Mean compact-state Rg
    Each architecture shows:
        VGIP / VGIG / VAIP / VAIG

FIGURE 4 — Ensemble breadth and compact-state population
    A  Raw central 90% Rg range vs compact-state population
       with quadratic fit
    B  Residual central 90% Rg range after accounting for
       compact-state population, grouped by architecture

All figures:
    - 370 K only
    - no overall figure title
    - saved as PNG and PDF
    - architecture effects calculated relative to the
      eight-architecture mean at identical chemistry + composition

IMPORTANT:
    This script requires a measured k_escape column.
    It does NOT derive k_escape from dwell time.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL SETTINGS
# ============================================================

ARCH_ORDER = [
    "even",
    "small blocks",
    "medium blocks",
    "large blocks",
    "higher-lambda center",
    "higher-lambda endcaps",
    "diblock",
    "random shuffle",
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

FAMILY_ORDER = [
    "VGIP",
    "VGIG",
    "VAIP",
    "VAIG",
]

FAMILY_MARKERS = {
    "VGIP": "o",
    "VGIG": "s",
    "VAIP": "^",
    "VAIG": "D",
}

REGIME_ORDER = [
    "Low",
    "Middle",
    "High",
]

REGIME_MAP = {
    "R1": "Low",
    "R2": "Low",
    "R3": "Middle",
    "R4": "Middle",
    "R5": "High",
    "R6": "High",
}

REGIME_RATIOS = {
    "Low": ["R1", "R2"],
    "Middle": ["R3", "R4"],
    "High": ["R5", "R6"],
}

REGIME_LABELS = {
    "Low": "Low higher-interaction motif fraction\nR1-R2",
    "Middle": "Middle higher-interaction motif fraction\nR3-R4",
    "High": "High higher-interaction motif fraction\nR5-R6",
}


# ============================================================
# FINAL METRICS
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
    },
]


# ============================================================
# STYLE
# ============================================================

def set_style():

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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

    if "endcap" in s:
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
                f"Using measured escape-rate column: {column}"
            )

            if column != "k_escape_per_ns":

                df["k_escape_per_ns"] = df[column]

            return df

    raise ValueError(
        "\nNo measured k_escape column was found.\n\n"
        "This final thesis script intentionally does NOT calculate\n"
        "k_escape = 1 / dwell time.\n\n"
        "Expected one of:\n"
        "    k_escape_per_ns\n"
        "    k_escape\n"
        "    escape_rate_per_ns\n"
        "    escape_rate\n\n"
        "Calculate the transition-derived escape rate first:\n"
        "    k_escape = N(C->E) / total_time_in_C\n"
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # 370 K
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

    # --------------------------------------------------------
    # Ratio
    # --------------------------------------------------------

    df["ratio"] = (
        df["ratio"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["regime"] = (
        df["ratio"]
        .map(REGIME_MAP)
    )

    # --------------------------------------------------------
    # Escape rate
    # --------------------------------------------------------

    df = find_escape_column(df)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "family",
        "ratio",
        "architecture_plot",
        "compact_fraction",
        "rg_range90_nm",
        "k_collapse_per_ns",
        "k_escape_per_ns",
        "mean_rg_collapsed_nm",
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Print basic checks
    # --------------------------------------------------------

    print()
    print("Architectures:")

    for x in ARCH_ORDER:

        n = (
            df["architecture_plot"]
            == x
        ).sum()

        print(
            f"  {x:<28s} {n}"
        )

    print()
    print("Families:")

    print(
        df["family"]
        .value_counts()
    )

    print()
    print("Ratios:")

    print(
        df["ratio"]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# ARCHITECTURE EFFECTS
# ============================================================

def calculate_architecture_effects(df):

    condition_cols = [
        "family",
        "ratio",
    ]

    raw_metrics = [
        "rg_range90_nm",
        "compact_fraction",
        "k_collapse_per_ns",
        "k_escape_per_ns",
        "mean_rg_collapsed_nm",
    ]

    # --------------------------------------------------------
    # Check eight architectures
    # --------------------------------------------------------

    counts = (
        df.groupby(condition_cols)
        ["architecture_plot"]
        .nunique()
    )

    incomplete = counts[
        counts != 8
    ]

    print()
    print(
        "Architecture counts per chemistry x ratio:"
    )

    print(
        counts.value_counts()
        .sort_index()
    )

    if len(incomplete) > 0:

        print()
        print("WARNING: incomplete conditions:")

        print(incomplete)

    # --------------------------------------------------------
    # Condition means
    # --------------------------------------------------------

    for metric in raw_metrics:

        df[
            metric + "_condition_mean"
        ] = (
            df
            .groupby(condition_cols)[metric]
            .transform("mean")
        )

    # --------------------------------------------------------
    # Absolute effects
    # --------------------------------------------------------

    df["effect_rg90_nm"] = (
        df["rg_range90_nm"]
        -
        df["rg_range90_nm_condition_mean"]
    )

    df["effect_collapsed_rg_nm"] = (
        df["mean_rg_collapsed_nm"]
        -
        df["mean_rg_collapsed_nm_condition_mean"]
    )

    # --------------------------------------------------------
    # Log2 rate effects
    # --------------------------------------------------------

    df["log2_fold_kcollapse"] = np.log2(
        df["k_collapse_per_ns"]
        /
        df["k_collapse_per_ns_condition_mean"]
    )

    df["log2_fold_kescape"] = np.log2(
        df["k_escape_per_ns"]
        /
        df["k_escape_per_ns_condition_mean"]
    )

    return df


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
        return np.nan, np.nan, 0

    mean = np.mean(values)

    if n == 1:
        return mean, np.nan, 1

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

    return mean, ci, n


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

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ============================================================
# GENERIC ARCHITECTURE EFFECT PANEL
# ============================================================

def plot_architecture_effect_panel(
    ax,
    df,
    metric,
    show_individual=True,
):

    effect = metric["effect"]

    for i, architecture in enumerate(
        ARCH_ORDER
    ):

        sub = df[
            df["architecture_plot"]
            == architecture
        ]

        values = (
            sub[effect]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        # ----------------------------------------------------
        # Individual chemistry x ratio points
        # ----------------------------------------------------

        if show_individual:

            jitter = np.linspace(
                -0.14,
                0.14,
                len(values),
            )

            for j, value in enumerate(values):

                ax.scatter(
                    i + jitter[j],
                    value,
                    s=25,
                    alpha=0.22,
                    edgecolors="none",
                )

        # ----------------------------------------------------
        # Mean ± CI
        # ----------------------------------------------------

        mean, ci, n = mean_ci95(
            values
        )

        if np.isfinite(mean):

            ax.errorbar(
                i,
                mean,
                yerr=ci,
                fmt="o",
                markersize=8,
                capsize=4,
                linewidth=1.6,
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
            len(ARCH_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            ARCH_LABELS[x]
            for x in ARCH_ORDER
        ],
        rotation=40,
        ha="right",
    )

    ax.set_ylabel(
        metric["ylabel"]
    )

    clean_axis(ax)


# ============================================================
# FIGURE 1
# OVERALL DESIGN EFFECTS
# ============================================================

def figure_overall_design_rules(
    df,
    outdir,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    axes = axes.flatten()

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    for ax, metric, letter in zip(
        axes,
        METRICS,
        letters,
    ):

        plot_architecture_effect_panel(
            ax,
            df,
            metric,
        )

        ax.set_title(
            f"{letter}   {metric['panel_title']}",
            loc="left",
            fontweight="bold",
        )

    fig.tight_layout(
        h_pad=2.4,
        w_pad=2.2,
    )

    save_figure(
        fig,
        outdir,
        "figure_architecture_overall_design_rules",
    )


# ============================================================
# FIGURE 2
# LOW / MIDDLE / HIGH REGIME MATRIX
# ============================================================

def figure_regime_matrix(
    df,
    outdir,
):

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 15),
        sharey="row",
    )

    for row, metric in enumerate(
        METRICS
    ):

        for col, regime in enumerate(
            REGIME_ORDER
        ):

            ax = axes[
                row,
                col,
            ]

            sub = df[
                df["regime"]
                == regime
            ]

            ratios = (
                REGIME_RATIOS[
                    regime
                ]
            )

            offsets = {
                ratios[0]: -0.08,
                ratios[1]: +0.08,
            }

            # -----------------------------------------------
            # Architecture
            # -----------------------------------------------

            for i, architecture in enumerate(
                ARCH_ORDER
            ):

                arch = sub[
                    sub["architecture_plot"]
                    == architecture
                ]

                # -------------------------------------------
                # Individual conditions
                # -------------------------------------------

                for family in FAMILY_ORDER:

                    for ratio in ratios:

                        point = arch[
                            (
                                arch["family"]
                                == family
                            )
                            &
                            (
                                arch["ratio"]
                                == ratio
                            )
                        ]

                        if len(point) == 0:
                            continue

                        y = point[
                            metric["effect"]
                        ].iloc[0]

                        if not np.isfinite(y):
                            continue

                        ax.scatter(
                            i + offsets[ratio],
                            y,
                            marker=(
                                FAMILY_MARKERS[
                                    family
                                ]
                            ),
                            s=26,
                            alpha=0.30,
                            edgecolors="none",
                        )

                # -------------------------------------------
                # Mean
                # -------------------------------------------

                values = (
                    arch[
                        metric["effect"]
                    ]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

                mean, ci, n = mean_ci95(
                    values
                )

                if np.isfinite(mean):

                    ax.errorbar(
                        i,
                        mean,
                        yerr=ci,
                        fmt="o",
                        markersize=7,
                        capsize=3,
                        linewidth=1.4,
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
                    len(ARCH_ORDER)
                )
            )

            ax.set_xticklabels(
                [
                    ARCH_LABELS[x]
                    for x in ARCH_ORDER
                ],
                rotation=43,
                ha="right",
                fontsize=7.5,
            )

            clean_axis(ax)

            # -----------------------------------------------
            # Column headings only
            # -----------------------------------------------

            if row == 0:

                ax.set_title(
                    REGIME_LABELS[
                        regime
                    ],
                    fontweight="bold",
                    fontsize=10,
                )

            # -----------------------------------------------
            # Y labels
            # -----------------------------------------------

            if col == 0:

                ax.set_ylabel(
                    metric["ylabel"]
                )

    # --------------------------------------------------------
    # Panel letters
    # --------------------------------------------------------

    letters = list(
        "ABCDEFGHIJKL"
    )

    counter = 0

    for row in range(4):

        for col in range(3):

            axes[row, col].text(
                -0.10,
                1.04,
                letters[counter],
                transform=axes[
                    row,
                    col,
                ].transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
            )

            counter += 1

    fig.tight_layout(
        h_pad=2.2,
        w_pad=1.5,
    )

    save_figure(
        fig,
        outdir,
        "figure_architecture_composition_regimes",
    )


# ============================================================
# FIGURE 3
# CONSERVATION ACROSS CHEMISTRY
# ============================================================

def figure_chemistry_consistency(
    df,
    outdir,
):

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 14),
        sharex=True,
    )

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

    for row, metric in enumerate(
        METRICS
    ):

        ax = axes[row]

        effect = metric["effect"]

        for i, architecture in enumerate(
            ARCH_ORDER
        ):

            arch = df[
                df["architecture_plot"]
                == architecture
            ]

            for family in FAMILY_ORDER:

                sub = arch[
                    arch["family"]
                    == family
                ]

                values = (
                    sub[effect]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

                mean, ci, n = mean_ci95(
                    values
                )

                if not np.isfinite(mean):
                    continue

                ax.errorbar(
                    (
                        i
                        +
                        family_offsets[
                            family
                        ]
                    ),
                    mean,
                    yerr=ci,
                    fmt=(
                        FAMILY_MARKERS[
                            family
                        ]
                    ),
                    markersize=6.5,
                    capsize=3,
                    linewidth=1.2,
                    label=(
                        family
                        if i == 0
                        else None
                    ),
                )

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1,
            color="0.45",
        )

        ax.set_ylabel(
            metric["ylabel"]
        )

        ax.set_title(
            (
                f"{letters[row]}   "
                f"{metric['panel_title']}"
            ),
            loc="left",
            fontweight="bold",
        )

        clean_axis(ax)

    axes[-1].set_xticks(
        np.arange(
            len(ARCH_ORDER)
        )
    )

    axes[-1].set_xticklabels(
        [
            ARCH_LABELS[x]
            for x in ARCH_ORDER
        ],
        rotation=40,
        ha="right",
    )

    axes[0].legend(
        frameon=False,
        ncol=4,
        loc="upper left",
    )

    fig.tight_layout(
        h_pad=2.0,
    )

    save_figure(
        fig,
        outdir,
        "figure_architecture_conservation_across_chemistries",
    )


# ============================================================
# FIGURE 4
# BREADTH VS COMPACT POPULATION
# ============================================================

def figure_breadth_population(
    df,
    outdir,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    ax1 = axes[0]
    ax2 = axes[1]

    # ========================================================
    # PANEL A
    # Raw relationship
    # ========================================================

    for architecture in ARCH_ORDER:

        sub = df[
            df["architecture_plot"]
            == architecture
        ]

        ax1.scatter(
            sub["compact_fraction"],
            sub["rg_range90_nm"],
            s=42,
            alpha=0.65,
            label=(
                ARCH_LABELS[
                    architecture
                ]
                .replace("\n", " ")
            ),
        )

    # --------------------------------------------------------
    # Quadratic fit
    # --------------------------------------------------------

    fit_df = (
        df[
            [
                "compact_fraction",
                "rg_range90_nm",
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    x = fit_df[
        "compact_fraction"
    ].to_numpy()

    y = fit_df[
        "rg_range90_nm"
    ].to_numpy()

    coefficients = np.polyfit(
        x,
        y,
        deg=2,
    )

    polynomial = np.poly1d(
        coefficients
    )

    xx = np.linspace(
        x.min(),
        x.max(),
        300,
    )

    ax1.plot(
        xx,
        polynomial(xx),
        color="black",
        linewidth=2,
        label="Quadratic population trend",
    )

    ax1.set_xlabel(
        "Compact-state population"
    )

    ax1.set_ylabel(
        r"Central 90% $R_g$ range (nm)"
    )

    ax1.set_title(
        "A   Ensemble breadth across the collapse regime",
        loc="left",
        fontweight="bold",
    )

    ax1.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    clean_axis(ax1)

    # ========================================================
    # CALCULATE RESIDUALS
    # ========================================================

    df = df.copy()

    df[
        "predicted_rg_range90_nm"
    ] = polynomial(
        df["compact_fraction"]
    )

    df[
        "rg_range90_population_residual"
    ] = (
        df["rg_range90_nm"]
        -
        df["predicted_rg_range90_nm"]
    )

    # ========================================================
    # PANEL B
    # Residual architecture effect
    # ========================================================

    for i, architecture in enumerate(
        ARCH_ORDER
    ):

        sub = df[
            df["architecture_plot"]
            == architecture
        ]

        values = (
            sub[
                "rg_range90_population_residual"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        # ----------------------------------------------------
        # Individual observations
        # ----------------------------------------------------

        jitter = np.linspace(
            -0.13,
            0.13,
            len(values),
        )

        for j, value in enumerate(values):

            ax2.scatter(
                i + jitter[j],
                value,
                s=25,
                alpha=0.22,
                edgecolors="none",
            )

        # ----------------------------------------------------
        # Mean ± CI
        # ----------------------------------------------------

        mean, ci, n = mean_ci95(
            values
        )

        ax2.errorbar(
            i,
            mean,
            yerr=ci,
            fmt="o",
            markersize=8,
            capsize=4,
            linewidth=1.5,
            zorder=10,
        )

    ax2.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="0.45",
    )

    ax2.set_xticks(
        np.arange(
            len(ARCH_ORDER)
        )
    )

    ax2.set_xticklabels(
        [
            ARCH_LABELS[x]
            for x in ARCH_ORDER
        ],
        rotation=40,
        ha="right",
    )

    ax2.set_ylabel(
        (
            "Residual central 90% "
            r"$R_g$ range (nm)"
        )
    )

    ax2.set_title(
        (
            "B   Architecture effect after accounting "
            "for compact population"
        ),
        loc="left",
        fontweight="bold",
    )

    clean_axis(ax2)

    fig.tight_layout(
        w_pad=2.5,
    )

    save_figure(
        fig,
        outdir,
        "figure_architecture_breadth_population",
    )

    # --------------------------------------------------------
    # Save residual table
    # --------------------------------------------------------

    residual_file = (
        outdir
        /
        "architecture_breadth_population_residuals.csv"
    )

    df.to_csv(
        residual_file,
        index=False,
    )

    print(
        f"Saved: {residual_file}"
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

def create_summary_table(
    df,
    outdir,
):

    rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    for architecture in ARCH_ORDER:

        sub = df[
            df["architecture_plot"]
            == architecture
        ]

        row = {
            "regime": "All",
            "architecture": architecture,
        }

        for metric in METRICS:

            values = (
                sub[
                    metric["effect"]
                ]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            mean, ci, n = mean_ci95(
                values
            )

            row[
                metric["effect"]
                + "_mean"
            ] = mean

            row[
                metric["effect"]
                + "_ci95"
            ] = ci

            row[
                metric["effect"]
                + "_n"
            ] = n

        rows.append(row)

    # --------------------------------------------------------
    # Regimes
    # --------------------------------------------------------

    for regime in REGIME_ORDER:

        for architecture in ARCH_ORDER:

            sub = df[
                (
                    df["regime"]
                    == regime
                )
                &
                (
                    df[
                        "architecture_plot"
                    ]
                    == architecture
                )
            ]

            row = {
                "regime": regime,
                "architecture": architecture,
            }

            for metric in METRICS:

                values = (
                    sub[
                        metric["effect"]
                    ]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .dropna()
                )

                mean, ci, n = mean_ci95(
                    values
                )

                row[
                    metric["effect"]
                    + "_mean"
                ] = mean

                row[
                    metric["effect"]
                    + "_ci95"
                ] = ci

                row[
                    metric["effect"]
                    + "_n"
                ] = n

            rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    output = (
        outdir
        /
        "architecture_thesis_design_rule_summary.csv"
    )

    summary.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV containing architecture metrics at 370 K"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    set_style()

    input_path = Path(
        args.input
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("ARCHITECTURE THESIS FIGURES")
    print("=" * 72)

    print(
        f"Input : {input_path}"
    )

    print(
        f"Output: {outdir}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_csv(
        input_path
    )

    print()
    print(
        f"Loaded {len(df)} rows"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    df = prepare_data(
        df
    )

    df = calculate_architecture_effects(
        df
    )

    # ========================================================
    # SAVE ANALYSIS TABLE
    # ========================================================

    values_file = (
        outdir
        /
        "architecture_thesis_effect_values.csv"
    )

    df.to_csv(
        values_file,
        index=False,
    )

    print(
        f"Saved: {values_file}"
    )

    # ========================================================
    # FIGURE 1
    # ========================================================

    print()
    print(
        "Creating Figure 1: overall design rules..."
    )

    figure_overall_design_rules(
        df,
        outdir,
    )

    # ========================================================
    # FIGURE 2
    # ========================================================

    print()
    print(
        "Creating Figure 2: composition regimes..."
    )

    figure_regime_matrix(
        df,
        outdir,
    )

    # ========================================================
    # FIGURE 3
    # ========================================================

    print()
    print(
        "Creating Figure 3: chemistry consistency..."
    )

    figure_chemistry_consistency(
        df,
        outdir,
    )

    # ========================================================
    # FIGURE 4
    # ========================================================

    print()
    print(
        "Creating Figure 4: breadth vs population..."
    )

    figure_breadth_population(
        df,
        outdir,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    create_summary_table(
        df,
        outdir,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()