from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ============================================================
# SETTINGS
# ============================================================

MOTIF_LAMBDA = {
    "AVG": 0.6294581781012,   # VPAVG
    "AG":  0.6568240708464,   # VPGAG
    "VG":  0.6850471057402,   # VPGVG
    "IP":  0.7430614622074,   # IPAVG
    "IG":  0.7986503898464,   # VPGIG
    "LG":  0.8147436293176,   # VPGLG
    "FG":  0.8506676135856,   # VPGFG
}


# Supports BOTH datasets:
#
# Original R4 triplicates:
#   VGAVG, IGIP, VGIP, VGIG, VAIP, VAIG
#
# New 907-930 triplicates:
#   AVGAG, IGLG, AVGFG

PAIR_DEFINITIONS = {
    "VGAVG": ("VG", "AVG"),
    "IGIP":  ("IG", "IP"),
    "VGIP":  ("VG", "IP"),
    "VGIG":  ("VG", "IG"),
    "VAIP":  ("AVG", "IP"),
    "VAIG":  ("AVG", "IG"),
    "AVGAG": ("AVG", "AG"),
    "IGLG":  ("IG", "LG"),
    "AVGFG": ("AVG", "FG"),
}


PAIR_DISPLAY = {
    "VGAVG": "VPGVG–VPAVG",
    "IGIP":  "VPGIG–IPAVG",
    "VGIP":  "VPGVG–IPAVG",
    "VGIG":  "VPGVG–VPGIG",
    "VAIP":  "VPAVG–IPAVG",
    "VAIG":  "VPAVG–VPGIG",
    "AVGAG": "VPAVG–VPGAG",
    "IGLG":  "VPGIG–VPGLG",
    "AVGFG": "VPAVG–VPGFG",
}


PAIR_COLORS = {
    "VGAVG": "#4C78A8",
    "IGIP":  "#F58518",
    "VGIP":  "#54A24B",
    "VGIG":  "#B279A2",
    "VAIP":  "#E45756",
    "VAIG":  "#9D755D",
    "AVGAG": "#76B7B2",
    "IGLG":  "#EDC948",
    "AVGFG": "#FF9DA7",
}


METRICS = {
    "rg_range_90_nm": {
        "short": "Central 90% Rg range",
        "ylabel": r"Central 90% $R_g$ range (nm)",
    },

    "mean_collapse_depth_nm": {
        "short": "Mean collapse depth",
        "ylabel": "Mean collapse depth (nm)",
    },

    "k_collapse_per_ns": {
        "short": "Collapse rate",
        "ylabel": r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
    },

    "k_escape_per_ns": {
        "short": "Escape rate",
        "ylabel": r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
    },
}


ARCH_LABELS = {
    "alternating": "Alternating",
    "smallblocks_3": "Small blocks",
    "smallblocks": "Small blocks",
    "mediumblocks_7": "Medium blocks",
    "mediumblocks": "Medium blocks",
    "largeblocks_14": "Large blocks",
    "largeblocks": "Large blocks",
    "diblock": "Diblock",
    "randomshuffle": "Random shuffle",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def delta_lambda(group):
    a, b = PAIR_DEFINITIONS[group]
    return abs(
        MOTIF_LAMBDA[a] - MOTIF_LAMBDA[b]
    )


def mean_lambda(group):
    a, b = PAIR_DEFINITIONS[group]

    return (
        MOTIF_LAMBDA[a] +
        MOTIF_LAMBDA[b]
    ) / 2.0


def clean_arch_label(x):
    x = str(x)

    if x in ARCH_LABELS:
        return ARCH_LABELS[x]

    x = x.replace("_", " ")

    return x


def pooled_sd(groups):
    """
    Pooled within-architecture replicate SD.

    This represents normal simulation-to-simulation variability
    within the same architecture.
    """

    numerator = 0.0
    denominator = 0

    for g in groups:

        g = np.asarray(
            g,
            dtype=float
        )

        g = g[
            np.isfinite(g)
        ]

        if len(g) >= 2:

            numerator += (
                (len(g) - 1) *
                np.var(
                    g,
                    ddof=1
                )
            )

            denominator += (
                len(g) - 1
            )

    if denominator == 0:
        return np.nan

    return np.sqrt(
        numerator /
        denominator
    )


# ============================================================
# CLASSICAL ANOVA + EFFECT SIZES
# ============================================================

def classical_anova_effect_sizes(groups):

    groups = [
        np.asarray(
            g,
            dtype=float
        )
        for g in groups
    ]

    groups = [
        g[np.isfinite(g)]
        for g in groups
    ]

    groups = [
        g
        for g in groups
        if len(g) > 0
    ]

    all_values = np.concatenate(
        groups
    )

    grand_mean = np.mean(
        all_values
    )

    ss_between = sum(
        len(g) *
        (
            np.mean(g) -
            grand_mean
        ) ** 2
        for g in groups
    )

    ss_within = sum(
        np.sum(
            (
                g -
                np.mean(g)
            ) ** 2
        )
        for g in groups
    )

    ss_total = (
        ss_between +
        ss_within
    )

    k = len(groups)
    n = len(all_values)

    df_between = k - 1
    df_within = n - k

    ms_between = (
        ss_between /
        df_between
        if df_between > 0
        else np.nan
    )

    ms_within = (
        ss_within /
        df_within
        if df_within > 0
        else np.nan
    )

    if (
        np.isfinite(ms_within)
        and
        ms_within > 0
    ):

        F = (
            ms_between /
            ms_within
        )

        p = stats.f.sf(
            F,
            df_between,
            df_within
        )

    else:

        F = np.inf
        p = 0.0

    eta2 = (
        ss_between /
        ss_total
        if ss_total > 0
        else np.nan
    )

    omega_num = (
        ss_between -
        df_between *
        ms_within
    )

    omega_den = (
        ss_total +
        ms_within
    )

    omega2_raw = (
        omega_num /
        omega_den
        if omega_den > 0
        else np.nan
    )

    omega2 = (
        max(
            0.0,
            omega2_raw
        )
        if np.isfinite(
            omega2_raw
        )
        else np.nan
    )

    return {

        "anova_F":
            F,

        "anova_df_between":
            df_between,

        "anova_df_within":
            df_within,

        "anova_p":
            p,

        "eta_squared":
            eta2,

        "omega_squared_raw":
            omega2_raw,

        "omega_squared":
            omega2,

        "ss_between":
            ss_between,

        "ss_within":
            ss_within,

        "ss_total":
            ss_total,
    }


# ============================================================
# WELCH ANOVA
# ============================================================

def welch_anova(groups):
    """
    Welch one-way ANOVA.

    More robust than ordinary ANOVA when replicate
    variances differ between architectures.
    """

    groups = [
        np.asarray(
            g,
            dtype=float
        )
        for g in groups
    ]

    groups = [
        g[np.isfinite(g)]
        for g in groups
    ]

    groups = [
        g
        for g in groups
        if len(g) >= 2
    ]

    k = len(groups)

    if k < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    n = np.array(
        [
            len(g)
            for g in groups
        ],
        dtype=float
    )

    means = np.array(
        [
            np.mean(g)
            for g in groups
        ],
        dtype=float
    )

    variances = np.array(
        [
            np.var(
                g,
                ddof=1
            )
            for g in groups
        ],
        dtype=float
    )

    # Prevent numerical problems when
    # replicate variance is exactly zero.
    eps = np.finfo(
        float
    ).eps

    variances = np.maximum(
        variances,
        eps
    )

    weights = (
        n /
        variances
    )

    weight_sum = np.sum(
        weights
    )

    weighted_mean = (
        np.sum(
            weights *
            means
        )
        /
        weight_sum
    )

    numerator = (
        np.sum(
            weights *
            (
                means -
                weighted_mean
            ) ** 2
        )
        /
        (k - 1)
    )

    correction_term = np.sum(

        (
            (
                1.0 -
                weights /
                weight_sum
            ) ** 2
        )
        /
        (
            n -
            1.0
        )

    )

    correction = (

        1.0 +

        (
            2.0 *
            (k - 2.0)
            /
            (
                k ** 2 -
                1.0
            )
        )
        *
        correction_term

    )

    F = (
        numerator /
        correction
    )

    df1 = (
        k -
        1.0
    )

    df2 = (

        (
            k ** 2 -
            1.0
        )
        /
        (
            3.0 *
            correction_term
        )

    )

    p = stats.f.sf(
        F,
        df1,
        df2
    )

    return (
        F,
        df1,
        df2,
        p
    )


# ============================================================
# LABEL HELPERS
# ============================================================

def p_to_label(p):

    if not np.isfinite(p):
        return "NA"

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return "ns"


def omega_label(x):

    if not np.isfinite(x):
        return "NA"

    if x < 0.01:
        return "negligible"

    if x < 0.06:
        return "small"

    if x < 0.14:
        return "moderate"

    return "large"


# ============================================================
# SINGLE METRIC / PAIR ANALYSIS
# ============================================================

def analyse_metric_pair(
    sub,
    metric,
    group_name
):

    sub = sub.dropna(
        subset=[metric]
    ).copy()

    arch_order = (

        sub[
            [
                "id",
                "architecture_type"
            ]
        ]

        .drop_duplicates()

        .sort_values(
            "id"
        )

    )

    groups = []

    for _, row in arch_order.iterrows():

        values = sub.loc[
            sub["id"] == row["id"],
            metric
        ].to_numpy(
            float
        )

        if len(values) > 0:
            groups.append(
                values
            )

    classical = (
        classical_anova_effect_sizes(
            groups
        )
    )

    (
        welch_F,
        welch_df1,
        welch_df2,
        welch_p
    ) = welch_anova(
        groups
    )

    # ----------------------------------------------
    # Levene test
    # ----------------------------------------------

    try:

        (
            levene_stat,
            levene_p
        ) = stats.levene(
            *groups,
            center="median"
        )

    except Exception:

        levene_stat = np.nan
        levene_p = np.nan

    # ----------------------------------------------
    # Architecture-level summary
    # ----------------------------------------------

    arch_stats = (

        sub

        .groupby(
            [
                "id",
                "architecture_type"
            ]
        )[metric]

        .agg(
            [
                "mean",
                "std",
                "count"
            ]
        )

        .reset_index()

    )

    architecture_spread = (

        arch_stats[
            "mean"
        ].max()

        -

        arch_stats[
            "mean"
        ].min()

    )

    within_sd = pooled_sd(
        groups
    )

    if (
        np.isfinite(within_sd)
        and
        within_sd > 0
    ):

        spread_to_noise = (
            architecture_spread /
            within_sd
        )

    else:

        spread_to_noise = np.nan

    min_row = arch_stats.loc[
        arch_stats[
            "mean"
        ].idxmin()
    ]

    max_row = arch_stats.loc[
        arch_stats[
            "mean"
        ].idxmax()
    ]

    result = {

        "metric":
            metric,

        "metric_label":
            METRICS[
                metric
            ]["short"],

        "group":
            group_name,

        "pair":
            PAIR_DISPLAY[
                group_name
            ],

        "delta_lambda":
            delta_lambda(
                group_name
            ),

        "mean_lambda":
            mean_lambda(
                group_name
            ),

        "n_architectures":
            len(
                arch_stats
            ),

        "n_runs":
            len(
                sub
            ),

        "architecture_spread":
            architecture_spread,

        "pooled_within_arch_sd":
            within_sd,

        "spread_to_noise":
            spread_to_noise,

        "min_architecture":
            min_row[
                "architecture_type"
            ],

        "min_architecture_mean":
            min_row[
                "mean"
            ],

        "max_architecture":
            max_row[
                "architecture_type"
            ],

        "max_architecture_mean":
            max_row[
                "mean"
            ],

        "levene_stat":
            levene_stat,

        "levene_p":
            levene_p,

        "welch_F":
            welch_F,

        "welch_df1":
            welch_df1,

        "welch_df2":
            welch_df2,

        "welch_p":
            welch_p,

        **classical,
    }

    return (
        result,
        arch_stats
    )


# ============================================================
# COMPLETE STATISTICAL ANALYSIS
# ============================================================

def run_statistics(df):

    global_rows = []
    architecture_rows = []
    tukey_rows = []

    groups_present = [

        group

        for group in PAIR_DEFINITIONS

        if group in set(
            df["group"]
        )

    ]

    for metric in METRICS:

        for group_name in groups_present:

            sub = df[
                df["group"] ==
                group_name
            ].copy()

            if sub.empty:
                continue

            (
                result,
                arch_stats
            ) = analyse_metric_pair(
                sub,
                metric,
                group_name
            )

            global_rows.append(
                result
            )

            arch_stats[
                "metric"
            ] = metric

            arch_stats[
                "group"
            ] = group_name

            arch_stats[
                "pair"
            ] = PAIR_DISPLAY[
                group_name
            ]

            arch_stats[
                "delta_lambda"
            ] = delta_lambda(
                group_name
            )

            architecture_rows.append(
                arch_stats
            )

            # ------------------------------------------
            # Tukey HSD
            # ------------------------------------------

            tukey_sub = sub.dropna(
                subset=[
                    metric,
                    "architecture_type"
                ]
            ).copy()

            if (
                tukey_sub[
                    "architecture_type"
                ].nunique()
                >= 2
            ):

                try:

                    tuk = (
                        pairwise_tukeyhsd(

                            endog=
                            tukey_sub[
                                metric
                            ].astype(
                                float
                            ),

                            groups=
                            tukey_sub[
                                "architecture_type"
                            ].astype(
                                str
                            ),

                            alpha=0.05
                        )
                    )

                    table = pd.DataFrame(

                        data=
                        tuk._results_table.data[
                            1:
                        ],

                        columns=
                        tuk._results_table.data[
                            0
                        ]

                    )

                    table[
                        "metric"
                    ] = metric

                    table[
                        "group"
                    ] = group_name

                    table[
                        "pair"
                    ] = PAIR_DISPLAY[
                        group_name
                    ]

                    table[
                        "delta_lambda"
                    ] = delta_lambda(
                        group_name
                    )

                    tukey_rows.append(
                        table
                    )

                except Exception as exc:

                    warnings.warn(
                        f"Tukey failed for "
                        f"{group_name}, "
                        f"{metric}: "
                        f"{exc}"
                    )

    global_df = pd.DataFrame(
        global_rows
    )

    if architecture_rows:

        architecture_df = pd.concat(
            architecture_rows,
            ignore_index=True
        )

    else:

        architecture_df = pd.DataFrame()

    if tukey_rows:

        tukey_df = pd.concat(
            tukey_rows,
            ignore_index=True
        )

    else:

        tukey_df = pd.DataFrame()

    # ========================================================
    # FDR CORRECTION
    #
    # IMPORTANT:
    # Correction is applied across all motif-pair tests
    # PRESENT IN THIS INPUT FILE for each metric.
    # ========================================================

    global_df[
        "welch_p_fdr"
    ] = np.nan

    global_df[
        "anova_p_fdr"
    ] = np.nan

    for metric in global_df[
        "metric"
    ].unique():

        mask = (
            global_df[
                "metric"
            ] == metric
        )

        p_welch = (
            global_df.loc[
                mask,
                "welch_p"
            ]
            .to_numpy(
                float
            )
        )

        p_anova = (
            global_df.loc[
                mask,
                "anova_p"
            ]
            .to_numpy(
                float
            )
        )

        valid_w = np.isfinite(
            p_welch
        )

        valid_a = np.isfinite(
            p_anova
        )

        corrected_w = np.full(
            p_welch.shape,
            np.nan
        )

        corrected_a = np.full(
            p_anova.shape,
            np.nan
        )

        if valid_w.any():

            corrected_w[
                valid_w
            ] = multipletests(

                p_welch[
                    valid_w
                ],

                alpha=0.05,

                method="fdr_bh"

            )[1]

        if valid_a.any():

            corrected_a[
                valid_a
            ] = multipletests(

                p_anova[
                    valid_a
                ],

                alpha=0.05,

                method="fdr_bh"

            )[1]

        global_df.loc[
            mask,
            "welch_p_fdr"
        ] = corrected_w

        global_df.loc[
            mask,
            "anova_p_fdr"
        ] = corrected_a

    global_df[
        "welch_significance"
    ] = global_df[
        "welch_p_fdr"
    ].apply(
        p_to_label
    )

    global_df[
        "omega_interpretation"
    ] = global_df[
        "omega_squared"
    ].apply(
        omega_label
    )

    return (
        global_df,
        architecture_df,
        tukey_df
    )


# ============================================================
# PLOT SETTINGS
# ============================================================

def setup_matplotlib():

    plt.rcParams.update({

        "font.size":
            9,

        "axes.labelsize":
            10,

        "axes.titlesize":
            10,

        "xtick.labelsize":
            8,

        "ytick.labelsize":
            8,

        "legend.fontsize":
            8,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False,

        "figure.dpi":
            150,

        "savefig.dpi":
            400,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    })


def save_figure(
    fig,
    outbase
):

    fig.savefig(
        str(
            outbase.with_suffix(
                ".png"
            )
        ),
        bbox_inches="tight",
        dpi=400
    )

    fig.savefig(
        str(
            outbase.with_suffix(
                ".pdf"
            )
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        str(
            outbase.with_suffix(
                ".svg"
            )
        ),
        bbox_inches="tight"
    )


# ============================================================
# PLOT 1:
# INDIVIDUAL TRIPLICATES + MEAN +/- SD
# ============================================================

def plot_triplicates(
    df,
    global_df,
    metric,
    outdir
):

    groups_present = [

        g

        for g in PAIR_DEFINITIONS

        if g in set(
            df["group"]
        )

    ]

    n_groups = len(
        groups_present
    )

    fig_width = max(
        9,
        3.0 *
        n_groups
    )

    fig, axes = plt.subplots(

        1,
        n_groups,

        figsize=(
            fig_width,
            4.7
        ),

        sharey=False

    )

    if n_groups == 1:
        axes = [axes]

    for (
        ax,
        group_name
    ) in zip(
        axes,
        groups_present
    ):

        sub = df[
            df["group"] ==
            group_name
        ].copy()

        arch_order = (

            sub[
                [
                    "id",
                    "architecture_type"
                ]
            ]

            .drop_duplicates()

            .sort_values(
                "id"
            )

            .reset_index(
                drop=True
            )

        )

        xs = np.arange(
            len(
                arch_order
            )
        )

        for (
            x,
            row
        ) in zip(
            xs,
            arch_order.itertuples()
        ):

            vals = sub.loc[
                sub["id"] ==
                row.id,
                metric
            ].dropna().to_numpy(
                float
            )

            if len(vals) > 1:

                offsets = np.linspace(
                    -0.11,
                    0.11,
                    len(vals)
                )

            else:

                offsets = np.array(
                    [0.0]
                )

            ax.scatter(

                x +
                offsets,

                vals,

                s=28,

                alpha=0.60,

                color=
                PAIR_COLORS[
                    group_name
                ],

                edgecolor="none",

                zorder=2

            )

            if len(vals) > 0:

                mean = np.mean(
                    vals
                )

                if len(vals) > 1:

                    sd = np.std(
                        vals,
                        ddof=1
                    )

                else:

                    sd = np.nan

                ax.errorbar(

                    x,
                    mean,

                    yerr=sd,

                    fmt="o",

                    color="black",

                    markersize=4.5,

                    capsize=3,

                    elinewidth=1.0,

                    zorder=3

                )

        statrow = global_df[
            (
                global_df[
                    "metric"
                ] == metric
            )
            &
            (
                global_df[
                    "group"
                ] == group_name
            )
        ].iloc[0]

        ax.text(

            0.03,
            0.97,

            (
                rf"$\Delta\lambda$ = "
                f"{statrow['delta_lambda']:.3f}"
                "\n"
                rf"Welch $q$ = "
                f"{statrow['welch_p_fdr']:.3g}"
                "\n"
                rf"$\omega^2$ = "
                f"{statrow['omega_squared']:.2f}"
                "\n"
                "spread/noise = "
                f"{statrow['spread_to_noise']:.1f}"
            ),

            transform=
            ax.transAxes,

            va="top",

            ha="left",

            fontsize=8

        )

        ax.set_xticks(
            xs
        )

        ax.set_xticklabels(

            [
                clean_arch_label(
                    x
                )
                for x in
                arch_order[
                    "architecture_type"
                ]
            ],

            rotation=60,

            ha="right"

        )

        ax.set_title(
            PAIR_DISPLAY[
                group_name
            ]
        )

        ax.set_ylabel(
            METRICS[
                metric
            ]["ylabel"]
        )

        ax.grid(
            axis="y",
            alpha=0.18
        )

    fig.tight_layout()

    save_figure(

        fig,

        outdir /
        f"triplicates_{metric}"

    )

    plt.close(
        fig
    )


# ============================================================
# PLOT 2:
# SIGNAL/NOISE + OMEGA² VS DELTA LAMBDA
# ============================================================

def plot_signal_vs_delta_lambda(
    global_df,
    metric,
    outdir
):

    d = global_df[
        global_df[
            "metric"
        ] == metric
    ].copy()

    d = d.sort_values(
        [
            "delta_lambda",
            "mean_lambda"
        ]
    )

    fig, axes = plt.subplots(

        1,
        2,

        figsize=(
            9.5,
            4.2
        )

    )

    # ----------------------------------------------
    # Spread / replicate noise
    # ----------------------------------------------

    ax = axes[0]

    for _, r in d.iterrows():

        significant = (
            r[
                "welch_p_fdr"
            ] < 0.05
        )

        ax.scatter(

            r[
                "delta_lambda"
            ],

            r[
                "spread_to_noise"
            ],

            s=75,

            facecolor=(
                PAIR_COLORS[
                    r["group"]
                ]
                if significant
                else "white"
            ),

            edgecolor=
            PAIR_COLORS[
                r["group"]
            ],

            linewidth=1.5,

            zorder=3

        )

        ax.annotate(

            r["group"],

            (
                r[
                    "delta_lambda"
                ],
                r[
                    "spread_to_noise"
                ]
            ),

            xytext=(
                4,
                4
            ),

            textcoords=
            "offset points",

            fontsize=7.5

        )

    ax.axhline(

        1.0,

        color="0.5",

        linestyle="--",

        linewidth=1

    )

    ax.set_xlabel(
        r"Motif interaction contrast, $\Delta\lambda$"
    )

    ax.set_ylabel(
        "Architecture spread / pooled replicate SD"
    )

    ax.grid(
        alpha=0.18
    )

    # ----------------------------------------------
    # Omega squared
    # ----------------------------------------------

    ax = axes[1]

    for _, r in d.iterrows():

        significant = (
            r[
                "welch_p_fdr"
            ] < 0.05
        )

        ax.scatter(

            r[
                "delta_lambda"
            ],

            r[
                "omega_squared"
            ],

            s=75,

            facecolor=(
                PAIR_COLORS[
                    r["group"]
                ]
                if significant
                else "white"
            ),

            edgecolor=
            PAIR_COLORS[
                r["group"]
            ],

            linewidth=1.5,

            zorder=3

        )

        ax.annotate(

            r["group"],

            (
                r[
                    "delta_lambda"
                ],
                r[
                    "omega_squared"
                ]
            ),

            xytext=(
                4,
                4
            ),

            textcoords=
            "offset points",

            fontsize=7.5

        )

    ax.axhline(
        0.01,
        color="0.80",
        linestyle=":",
        linewidth=0.8
    )

    ax.axhline(
        0.06,
        color="0.75",
        linestyle=":",
        linewidth=0.8
    )

    ax.axhline(
        0.14,
        color="0.70",
        linestyle=":",
        linewidth=0.8
    )

    ax.set_xlabel(
        r"Motif interaction contrast, $\Delta\lambda$"
    )

    ax.set_ylabel(
        r"Architecture effect size, $\omega^2$"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.grid(
        alpha=0.18
    )

    fig.tight_layout()

    save_figure(

        fig,

        outdir /
        (
            "architecture_signal_vs_delta_lambda_"
            f"{metric}"
        )

    )

    plt.close(
        fig
    )


# ============================================================
# PLOT 3:
# ALL FOUR METRICS - SIGNAL / NOISE
# ============================================================

def plot_all_metrics_summary(
    global_df,
    outdir
):

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            9.5,
            8.0
        )

    )

    panel_labels = [
        "A",
        "B",
        "C",
        "D"
    ]

    for (
        ax,
        metric,
        panel
    ) in zip(

        axes.flat,
        METRICS.keys(),
        panel_labels

    ):

        d = global_df[
            global_df[
                "metric"
            ] == metric
        ].copy()

        d = d.sort_values(
            [
                "delta_lambda",
                "mean_lambda"
            ]
        )

        for _, r in d.iterrows():

            significant = (
                r[
                    "welch_p_fdr"
                ] < 0.05
            )

            ax.scatter(

                r[
                    "delta_lambda"
                ],

                r[
                    "spread_to_noise"
                ],

                s=75,

                facecolor=(
                    PAIR_COLORS[
                        r["group"]
                    ]
                    if significant
                    else "white"
                ),

                edgecolor=
                PAIR_COLORS[
                    r["group"]
                ],

                linewidth=1.5,

                zorder=3

            )

            ax.annotate(

                r["group"],

                (
                    r[
                        "delta_lambda"
                    ],
                    r[
                        "spread_to_noise"
                    ]
                ),

                xytext=(
                    4,
                    4
                ),

                textcoords=
                "offset points",

                fontsize=7

            )

        ax.axhline(

            1,

            color="0.55",

            linestyle="--",

            linewidth=0.9

        )

        ax.set_xlabel(
            r"Motif interaction contrast, $\Delta\lambda$"
        )

        ax.set_ylabel(
            "Architecture spread / pooled SD"
        )

        ax.text(

            -0.12,
            1.04,

            panel,

            transform=
            ax.transAxes,

            fontsize=12,

            fontweight="bold"

        )

        ax.text(

            0.03,
            0.96,

            METRICS[
                metric
            ]["short"],

            transform=
            ax.transAxes,

            va="top",

            fontsize=9

        )

        ax.grid(
            alpha=0.18
        )

    fig.tight_layout()

    save_figure(

        fig,

        outdir /
        "architecture_signal_to_noise_all_metrics"

    )

    plt.close(
        fig
    )


# ============================================================
# PLOT 4:
# ALL FOUR METRICS - EFFECT SIZE
# ============================================================

def plot_effect_size_summary(
    global_df,
    outdir
):

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            9.5,
            8.0
        )

    )

    panel_labels = [
        "A",
        "B",
        "C",
        "D"
    ]

    for (
        ax,
        metric,
        panel
    ) in zip(

        axes.flat,
        METRICS.keys(),
        panel_labels

    ):

        d = global_df[
            global_df[
                "metric"
            ] == metric
        ].copy()

        for _, r in d.iterrows():

            significant = (
                r[
                    "welch_p_fdr"
                ] < 0.05
            )

            ax.scatter(

                r[
                    "delta_lambda"
                ],

                r[
                    "omega_squared"
                ],

                s=75,

                facecolor=(
                    PAIR_COLORS[
                        r["group"]
                    ]
                    if significant
                    else "white"
                ),

                edgecolor=
                PAIR_COLORS[
                    r["group"]
                ],

                linewidth=1.5,

                zorder=3

            )

            ax.annotate(

                r["group"],

                (
                    r[
                        "delta_lambda"
                    ],
                    r[
                        "omega_squared"
                    ]
                ),

                xytext=(
                    4,
                    4
                ),

                textcoords=
                "offset points",

                fontsize=7

            )

        ax.axhline(
            0.01,
            color="0.80",
            linestyle=":",
            linewidth=0.8
        )

        ax.axhline(
            0.06,
            color="0.75",
            linestyle=":",
            linewidth=0.8
        )

        ax.axhline(
            0.14,
            color="0.70",
            linestyle=":",
            linewidth=0.8
        )

        ax.set_xlabel(
            r"Motif interaction contrast, $\Delta\lambda$"
        )

        ax.set_ylabel(
            r"$\omega^2$"
        )

        ax.set_ylim(
            bottom=0
        )

        ax.text(

            -0.12,
            1.04,

            panel,

            transform=
            ax.transAxes,

            fontsize=12,

            fontweight="bold"

        )

        ax.text(

            0.03,
            0.96,

            METRICS[
                metric
            ]["short"],

            transform=
            ax.transAxes,

            va="top",

            fontsize=9

        )

        ax.grid(
            alpha=0.18
        )

    fig.tight_layout()

    save_figure(

        fig,

        outdir /
        "architecture_effect_size_all_metrics"

    )

    plt.close(
        fig
    )


# ============================================================
# PLOT 5:
# REPLICATE VARIABILITY ITSELF
# ============================================================

def plot_replicate_variability(
    global_df,
    outdir
):

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            9.5,
            8.0
        )

    )

    panel_labels = [
        "A",
        "B",
        "C",
        "D"
    ]

    for (
        ax,
        metric,
        panel
    ) in zip(

        axes.flat,
        METRICS.keys(),
        panel_labels

    ):

        d = global_df[
            global_df[
                "metric"
            ] == metric
        ].copy()

        d = d.sort_values(
            [
                "delta_lambda",
                "mean_lambda"
            ]
        )

        x = np.arange(
            len(d)
        )

        ax.bar(

            x,

            d[
                "pooled_within_arch_sd"
            ],

            color=[
                PAIR_COLORS[
                    g
                ]
                for g in
                d[
                    "group"
                ]
            ],

            edgecolor="black",

            linewidth=0.5

        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(

            d[
                "group"
            ],

            rotation=45,

            ha="right"

        )

        ax.set_ylabel(
            "Pooled within-architecture SD"
        )

        ax.text(

            -0.12,
            1.04,

            panel,

            transform=
            ax.transAxes,

            fontsize=12,

            fontweight="bold"

        )

        ax.text(

            0.03,
            0.96,

            METRICS[
                metric
            ]["short"],

            transform=
            ax.transAxes,

            va="top",

            fontsize=9

        )

        ax.grid(
            axis="y",
            alpha=0.18
        )

    fig.tight_layout()

    save_figure(

        fig,

        outdir /
        "replicate_variability_by_composition"

    )

    plt.close(
        fig
    )


# ============================================================
# TEXT REPORT
# ============================================================

def print_report(
    global_df
):

    print()

    print(
        "=" * 100
    )

    print(
        "GLOBAL ARCHITECTURE TESTS"
    )

    print(
        "=" * 100
    )

    for metric in METRICS:

        print()

        print(
            METRICS[
                metric
            ]["short"]
        )

        print(
            "-" * 100
        )

        d = global_df[
            global_df[
                "metric"
            ] == metric
        ].sort_values(
            [
                "delta_lambda",
                "mean_lambda"
            ]
        )

        cols = [

            "group",

            "delta_lambda",

            "mean_lambda",

            "architecture_spread",

            "pooled_within_arch_sd",

            "spread_to_noise",

            "welch_p",

            "welch_p_fdr",

            "omega_squared",

            "omega_interpretation",

            "levene_p",

        ]

        print(

            d[
                cols
            ].to_string(

                index=False,

                float_format=
                lambda x:
                f"{x:.4g}"

            )

        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Replicate-aware statistical analysis "
            "of CALVADOS architecture effects."
        )

    )

    parser.add_argument(

        "--input",

        required=True,

        help=
        "Input per_run_summary.csv"

    )

    parser.add_argument(

        "--outdir",

        required=True,

        help=
        "Output directory"

    )

    parser.add_argument(

        "--temp",

        default="370K",

        help=
        "Temperature to analyse "
        "(default: 370K)"

    )

    args = parser.parse_args()

    infile = Path(
        args.input
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    setup_matplotlib()

    df = pd.read_csv(
        infile
    )

    required = {

        "id",
        "temp",
        "replicate",
        "group",
        "architecture_type",

    } | set(
        METRICS.keys()
    )

    missing = (
        required -
        set(
            df.columns
        )
    )

    if missing:

        raise ValueError(

            "Missing required columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )

        )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    df = df[
        df[
            "temp"
        ].astype(
            str
        ) == args.temp
    ].copy()

    # --------------------------------------------------------
    # Supported families
    # --------------------------------------------------------

    found_groups = sorted(
        df[
            "group"
        ].dropna().unique()
    )

    supported_groups = [

        g

        for g in found_groups

        if g in
        PAIR_DEFINITIONS

    ]

    unsupported_groups = [

        g

        for g in found_groups

        if g not in
        PAIR_DEFINITIONS

    ]

    if unsupported_groups:

        print(
            "WARNING: unsupported groups "
            "will be skipped:"
        )

        print(
            unsupported_groups
        )

        print()

    df = df[
        df[
            "group"
        ].isin(
            supported_groups
        )
    ].copy()

    if df.empty:

        raise ValueError(

            f"No supported rows found "
            f"for temperature "
            f"{args.temp}"

        )

    print(
        "=" * 100
    )

    print(
        "CALVADOS TRIPLICATE STATISTICS"
    )

    print(
        "=" * 100
    )

    print(
        f"Input:         {infile}"
    )

    print(
        f"Temperature:   {args.temp}"
    )

    print(
        f"Runs:          {len(df)}"
    )

    print(
        f"Architectures: {df['id'].nunique()}"
    )

    print(
        f"Motif pairs:   {df['group'].nunique()}"
    )

    print(
        "Groups:        "
        +
        ", ".join(
            sorted(
                df[
                    "group"
                ].unique()
            )
        )
    )

    print()

    # --------------------------------------------------------
    # Replicate count check
    # --------------------------------------------------------

    counts = (

        df

        .groupby(
            [
                "group",
                "id"
            ]
        )[
            "replicate"
        ]

        .nunique()

    )

    print(
        "Replicates per architecture:"
    )

    print(
        counts
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    if not (
        counts == 3
    ).all():

        warnings.warn(
            "Not every architecture has "
            "exactly three replicates."
        )

    # ========================================================
    # RUN STATISTICS
    # ========================================================

    (
        global_df,
        architecture_df,
        tukey_df
    ) = run_statistics(
        df
    )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    global_df.to_csv(

        outdir /
        "global_architecture_statistics.csv",

        index=False

    )

    architecture_df.to_csv(

        outdir /
        "architecture_mean_sd.csv",

        index=False

    )

    tukey_df.to_csv(

        outdir /
        "tukey_pairwise_statistics.csv",

        index=False

    )

    # --------------------------------------------------------
    # Significant Tukey comparisons
    # --------------------------------------------------------

    if (
        not tukey_df.empty
        and
        "reject" in
        tukey_df.columns
    ):

        reject_bool = (

            tukey_df[
                "reject"
            ]

            .astype(
                str
            )

            .str.lower()

            .isin(
                [
                    "true",
                    "1"
                ]
            )

        )

        sig_tukey = tukey_df[
            reject_bool
        ].copy()

        sig_tukey.to_csv(

            outdir /
            "tukey_significant_only.csv",

            index=False

        )

    # ========================================================
    # PLOTS
    # ========================================================

    for metric in METRICS:

        plot_triplicates(

            df,
            global_df,
            metric,
            outdir

        )

        plot_signal_vs_delta_lambda(

            global_df,
            metric,
            outdir

        )

    plot_all_metrics_summary(

        global_df,
        outdir

    )

    plot_effect_size_summary(

        global_df,
        outdir

    )

    plot_replicate_variability(

        global_df,
        outdir

    )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print_report(
        global_df
    )

    print()

    print(
        "=" * 100
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Output: {outdir}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Welch q < 0.05:"
    )

    print(
        "      evidence for a global architecture effect"
    )

    print()

    print(
        "  omega^2:"
    )

    print(
        "      magnitude of architecture-associated variation"
    )

    print()

    print(
        "  spread/noise:"
    )

    print(
        "      max-min architecture means / pooled replicate SD"
    )

    print()

    print(
        "  Tukey HSD:"
    )

    print(
        "      identifies specific architecture pairs "
        "that differ"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "FDR correction is performed across the motif pairs "
        "present in THIS input file for each metric."
    )


if __name__ == "__main__":
    main()