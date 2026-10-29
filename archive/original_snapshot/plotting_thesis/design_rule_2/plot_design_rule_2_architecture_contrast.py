from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis\design_rule_book"
)

PER_RUN_FILES = [
    ROOT / "907_930_3reps_stride1" / "per_run_summary.csv",
    ROOT / "lib1R4_triplicate_stride1" / "per_run_summary.csv",
]

OUTDIR = ROOT / "design_rule_2_architecture_contrast"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

PRIMARY_METRIC = "iqr_rg_nm"

ALPHA = 0.05


# ============================================================
# MOTIF STICKINESS
# ============================================================

LAMBDA = {
    "VPAVG": 0.6294581781012,
    "VPGAG": 0.6568240708464,
    "VPGVG": 0.6850471057402,
    "IPAVG": 0.7430614622074,
    "VPGIG": 0.7986503898464,
    "VPGLG": 0.8147436293176,
    "VPGFG": 0.8506676135856,
}


# ============================================================
# MOTIF PAIRS
# ============================================================

PAIR_INFO = {
    "IGLG": ("VPGIG", "VPGLG"),
    "AVGAG": ("VPAVG", "VPGAG"),
    "IGIP": ("VPGIG", "IPAVG"),
    "VGAVG": ("VPGVG", "VPAVG"),
    "VGIP": ("VPGVG", "IPAVG"),
    "VGIG": ("VPGVG", "VPGIG"),
    "VAIP": ("VPAVG", "IPAVG"),
    "VAIG": ("VPAVG", "VPGIG"),
    "AVGFG": ("VPAVG", "VPGFG"),
}


PAIR_META = {}

for pair, (motif_a, motif_b) in PAIR_INFO.items():

    la = LAMBDA[motif_a]
    lb = LAMBDA[motif_b]

    PAIR_META[pair] = {
        "motif_a": motif_a,
        "motif_b": motif_b,
        "lambda_a": la,
        "lambda_b": lb,
        "delta_lambda": abs(la - lb),
        "mean_lambda": (la + lb) / 2,
    }


# ============================================================
# ORDER
#
# IGIP deliberately before VGAVG
# ============================================================

PAIR_ORDER = [
    "IGLG",
    "AVGAG",
    "IGIP",
    "VGAVG",
    "VGIP",
    "VGIG",
    "VAIP",
    "VAIG",
    "AVGFG",
]


PAIR_DISPLAY = {
    "IGLG": "VPGIG–VPGLG",
    "AVGAG": "VPAVG–VPGAG",
    "IGIP": "VPGIG–IPAVG",
    "VGAVG": "VPGVG–VPAVG",
    "VGIP": "VPGVG–IPAVG",
    "VGIG": "VPGVG–VPGIG",
    "VAIP": "VPAVG–IPAVG",
    "VAIG": "VPAVG–VPGIG",
    "AVGFG": "VPAVG–VPGFG",
}


# ============================================================
# METRICS
# ============================================================

METRIC_LABELS = {

    "mean_rg_nm":
        r"Mean $R_g$",

    "median_rg_nm":
        r"Median $R_g$",

    "std_rg_nm":
        r"SD $R_g$",

    "iqr_rg_nm":
        r"Central 50% $R_g$ range",

    "rg_range_90_nm":
        r"Central 90% $R_g$ range",

    "p05_rg_nm":
        r"$R_{g,5}$",

    "p95_rg_nm":
        r"$R_{g,95}$",

    "fraction_collapsed":
        "Compact-state population",

    "mean_collapse_depth_nm":
        "Mean collapse depth",

    "mean_collapsed_dwell_ns":
        "Collapsed-state dwell time",

    "mean_expanded_dwell_ns":
        "Expanded-state dwell time",

    "k_collapse_per_ns":
        r"$k_{\mathrm{collapse}}$",

    "k_escape_per_ns":
        r"$k_{\mathrm{escape}}$",
}


MAIN_METRICS = [
    "iqr_rg_nm",
    "rg_range_90_nm",
    "fraction_collapsed",
    "k_collapse_per_ns",
]


# ============================================================
# ARCHITECTURES
# ============================================================

ARCH_ORDER = [
    "Alternating",
    "Small blocks",
    "Medium blocks",
    "Large blocks",
    "Diblock",
    "Random shuffle",
    r"Higher-$\lambda$ center",
    r"Higher-$\lambda$ ends",
]


ARCH_MARKERS = {
    "Alternating": "o",
    "Small blocks": "s",
    "Medium blocks": "^",
    "Large blocks": "v",
    "Diblock": "D",
    "Random shuffle": "P",
    r"Higher-$\lambda$ center": "<",
    r"Higher-$\lambda$ ends": ">",
}


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


DEFAULT_COLORS = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()["color"]
)


# ============================================================
# ARCHITECTURE COLORS
#
# Architecture identity remains consistent across motif pairs.
# ============================================================

ARCH_COLORS = {
    arch: DEFAULT_COLORS[
        i % len(DEFAULT_COLORS)
    ]
    for i, arch in enumerate(ARCH_ORDER)
}


# ============================================================
# PAIR / VIOLIN COLORS
#
# One distinct color per chemistry.
# ============================================================

PAIR_CMAP = plt.get_cmap("viridis")

PAIR_COLORS = {
    pair: PAIR_CMAP(
        0.08 + 0.84 * i / (len(PAIR_ORDER) - 1)
    )
    for i, pair in enumerate(PAIR_ORDER)
}


# ============================================================
# HELPERS
# ============================================================

def detect_pair_from_text(value):

    text = str(value).upper()

    for pair in sorted(
        PAIR_INFO,
        key=len,
        reverse=True
    ):

        if pair.upper() in text:
            return pair

    return None


def add_pair_column(df):

    out = df.copy()

    if "group" in out.columns:

        out["pair"] = (
            out["group"]
            .astype(str)
            .apply(detect_pair_from_text)
        )

        return out


    candidate_cols = [
        "id",
        "architecture",
        "architecture_type",
        "sequence",
        "name",
        "design",
        "library",
        "system",
        "pair",
        "motif_pair",
    ]


    existing = [
        c
        for c in candidate_cols
        if c in out.columns
    ]


    def detect_row(row):

        text = " ".join(
            str(row[c])
            for c in existing
        )

        return detect_pair_from_text(
            text
        )


    out["pair"] = out.apply(
        detect_row,
        axis=1
    )

    return out


# ============================================================
# LOAD PER-RUN DATA
# ============================================================

frames = []


for file in PER_RUN_FILES:

    print("\nReading:")
    print(file)

    temp = pd.read_csv(file)

    temp["source_file"] = str(file)

    temp = add_pair_column(
        temp
    )

    frames.append(
        temp
    )


runs = pd.concat(
    frames,
    ignore_index=True
)


# ============================================================
# TEMPERATURE FILTER
# ============================================================

if "temp" in runs.columns:

    runs = runs[
        runs["temp"]
        .astype(str)
        .str.contains(
            "370",
            regex=False
        )
    ].copy()


runs = runs[
    runs["pair"].isin(
        PAIR_INFO
    )
].copy()


# ============================================================
# ARCHITECTURE NORMALISATION
# ============================================================

def architecture_text(row):

    columns = [
        "architecture_type",
        "architecture",
        "id",
        "name",
    ]

    values = []

    for column in columns:

        if column in row.index:
            values.append(
                str(row[column])
            )

    return " ".join(
        values
    ).lower()


def architecture_class(row):

    # Use architecture_type first because it explicitly contains
    # the architecture definition.
    arch_type = str(
        row.get("architecture_type", "")
    ).strip()

    text = arch_type.lower()


    # ========================================================
    # STANDARD ARCHITECTURES
    # ========================================================

    if (
        "alternating" in text
        or text == "even"
    ):
        return "Alternating"

    if "smallblock" in text:
        return "Small blocks"

    if "mediumblock" in text:
        return "Medium blocks"

    if "largeblock" in text:
        return "Large blocks"

    if "diblock" in text:
        return "Diblock"

    if (
        "randomshuffle" in text
        or "random_shuffle" in text
        or "random shuffle" in text
        or text == "shuffle"
    ):
        return "Random shuffle"


    # ========================================================
    # DETERMINE HIGHER-LAMBDA MOTIF
    # ========================================================

    pair = row["pair"]

    motif_a, motif_b = PAIR_INFO[pair]

    if LAMBDA[motif_a] > LAMBDA[motif_b]:
        high_motif = motif_a

    elif LAMBDA[motif_b] > LAMBDA[motif_a]:
        high_motif = motif_b

    else:
        warnings.warn(
            f"{pair}: motifs have identical lambda."
        )
        return f"Unknown: {arch_type}"


    # ========================================================
    # LIBRARY CODE -> FULL MOTIF NAME
    # ========================================================

    CODE_TO_MOTIF = {
        "VG": "VPGVG",
        "IG": "VPGIG",
        "LG": "VPGLG",
        "FG": "VPGFG",
        "AVG": "VPAVG",
        "VA": "VPAVG",      # used in VAIP / VAIG architecture names
        "AG": "VPGAG",
        "IP": "IPAVG",
    }


    # ========================================================
    # DIRECTIONAL: Xcenter_Yends
    #
    # Examples:
    # IGcenter_VGends
    # VGcenter_IGends
    # IPcenter_VAends
    # ========================================================

    if "center" in text:

        center_raw = (
            text.split("center")[0]
            .replace("_", "")
            .replace("-", "")
            .strip()
            .upper()
        )

        center_motif = CODE_TO_MOTIF.get(
            center_raw
        )

        if center_motif is None:
            warnings.warn(
                f"Could not identify center motif from "
                f"architecture_type='{arch_type}'"
            )
            return f"Unknown: {arch_type}"

        if center_motif == high_motif:
            return r"Higher-$\lambda$ center"

        return r"Higher-$\lambda$ ends"


    # ========================================================
    # DIRECTIONAL: Xendcaps
    #
    # Examples:
    # IPendcaps
    # IGendcaps
    #
    # The motif before "endcaps" is explicitly the motif at
    # the chain termini.
    # ========================================================

    if "endcaps" in text:

        end_raw = (
            text.split("endcaps")[0]
            .replace("_", "")
            .replace("-", "")
            .strip()
            .upper()
        )

        end_motif = CODE_TO_MOTIF.get(
            end_raw
        )

        if end_motif is None:
            warnings.warn(
                f"Could not identify endcap motif from "
                f"architecture_type='{arch_type}'"
            )
            return f"Unknown: {arch_type}"

        if end_motif == high_motif:
            return r"Higher-$\lambda$ ends"

        return r"Higher-$\lambda$ center"


    # ========================================================
    # UNKNOWN
    # ========================================================

    warnings.warn(
        f"Unrecognised architecture_type: '{arch_type}'"
    )

    return f"Unknown: {arch_type}"

# ============================================================
# APPLY ARCHITECTURE CLASSIFICATION
# ============================================================

runs["architecture_class"] = runs.apply(
    architecture_class,
    axis=1
)

# Keep only recognised architectures
runs = runs[
    runs["architecture_class"].isin(ARCH_ORDER)
].copy()


# ============================================================
# NUMERIC PRIMARY METRIC
# ============================================================

runs[
    PRIMARY_METRIC
] = pd.to_numeric(
    runs[
        PRIMARY_METRIC
    ],
    errors="coerce"
)


# ============================================================
# VALIDATE LIBRARY
# ============================================================

print("\n")
print("=" * 80)
print("LIBRARY VALIDATION")
print("=" * 80)


for pair in PAIR_ORDER:

    sub = runs[
        runs["pair"]
        == pair
    ]


    counts = (
        sub.groupby(
            "architecture_class"
        )
        .size()
    )


    print(
        f"\n{pair} "
        f"Δλ="
        f"{PAIR_META[pair]['delta_lambda']:.3f}"
    )


    print(
        counts.to_string()
    )


    valid_archs = [
        arch
        for arch in ARCH_ORDER
        if arch in counts.index
    ]


    if len(valid_archs) != 8:

        warnings.warn(
            f"{pair}: expected 8 recognised "
            f"architectures, found "
            f"{len(valid_archs)}."
        )


# ============================================================
# SAVE VALIDATION TABLE
# ============================================================

validation_records = []


for pair in PAIR_ORDER:

    sub = runs[
        runs["pair"]
        == pair
    ]


    for arch in ARCH_ORDER:

        arch_sub = sub[
            sub[
                "architecture_class"
            ]
            == arch
        ]


        validation_records.append({
            "pair": pair,
            "architecture": arch,
            "n_replicates": len(
                arch_sub
            ),
        })


validation = pd.DataFrame(
    validation_records
)


validation.to_csv(
    OUTDIR
    / "DR2_library_validation.csv",
    index=False
)


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def get_architecture_groups(
    data,
    metric
):

    groups = []

    labels = []


    for arch in ARCH_ORDER:

        sub = data[
            data[
                "architecture_class"
            ]
            == arch
        ]


        values = pd.to_numeric(
            sub[
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy()


        if len(values) >= 2:

            groups.append(
                values
            )

            labels.append(
                arch
            )


    return groups, labels


def architecture_means(
    data,
    metric
):

    records = []


    for arch in ARCH_ORDER:

        sub = data[
            data[
                "architecture_class"
            ]
            == arch
        ]


        values = pd.to_numeric(
            sub[
                metric
            ],
            errors="coerce"
        ).dropna()


        if len(values) == 0:
            continue


        records.append({
            "architecture": arch,
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "n": len(values),
        })


    return pd.DataFrame(
        records
    )


def pooled_within_architecture_sd(
    data,
    metric
):

    numerator = 0.0

    denominator = 0


    for arch in ARCH_ORDER:

        sub = data[
            data[
                "architecture_class"
            ]
            == arch
        ]


        values = pd.to_numeric(
            sub[
                metric
            ],
            errors="coerce"
        ).dropna().to_numpy()


        n = len(
            values
        )


        if n < 2:
            continue


        variance = np.var(
            values,
            ddof=1
        )


        numerator += (
            (n - 1)
            * variance
        )


        denominator += (
            n - 1
        )


    if denominator == 0:

        return np.nan


    return np.sqrt(
        numerator
        / denominator
    )


def architecture_spread(
    data,
    metric
):

    means = architecture_means(
        data,
        metric
    )


    if len(
        means
    ) < 2:

        return np.nan


    return (
        means[
            "mean"
        ].max()
        -
        means[
            "mean"
        ].min()
    )


def architecture_signal_noise(
    data,
    metric
):

    signal = architecture_spread(
        data,
        metric
    )


    noise = pooled_within_architecture_sd(
        data,
        metric
    )


    if (
        not np.isfinite(
            signal
        )
        or
        not np.isfinite(
            noise
        )
        or
        noise == 0
    ):

        return np.nan


    return (
        signal
        / noise
    )


def omega_squared(
    data,
    metric
):

    groups, labels = (
        get_architecture_groups(
            data,
            metric
        )
    )


    if len(
        groups
    ) < 2:

        return np.nan


    all_values = np.concatenate(
        groups
    )


    grand_mean = np.mean(
        all_values
    )


    ss_between = sum(

        len(group)
        *
        (
            np.mean(group)
            - grand_mean
        ) ** 2

        for group
        in groups
    )


    ss_within = sum(

        np.sum(
            (
                group
                - np.mean(group)
            ) ** 2
        )

        for group
        in groups
    )


    k = len(
        groups
    )


    n = len(
        all_values
    )


    df_between = (
        k - 1
    )


    df_within = (
        n - k
    )


    if df_within <= 0:

        return np.nan


    ms_within = (
        ss_within
        / df_within
    )


    ss_total = (
        ss_between
        + ss_within
    )


    denominator = (
        ss_total
        + ms_within
    )


    if denominator == 0:

        return 0.0


    omega2 = (

        ss_between
        -
        df_between
        * ms_within

    ) / denominator


    return max(
        0.0,
        omega2
    )


def welch_anova(
    data,
    metric
):

    groups, labels = (
        get_architecture_groups(
            data,
            metric
        )
    )


    if len(
        groups
    ) < 2:

        return (
            np.nan,
            np.nan,
            np.nan
        )


    try:

        result = anova_oneway(
            groups,
            use_var="unequal",
            welch_correction=True
        )


        return (
            float(
                result.statistic
            ),
            float(
                result.pvalue
            ),
            float(
                result.df_denom
            )
        )


    except Exception as error:

        warnings.warn(
            f"Welch ANOVA failed: {error}"
        )


        return (
            np.nan,
            np.nan,
            np.nan
        )


# ============================================================
# CALCULATE STATISTICS FOR ALL AVAILABLE METRICS
# ============================================================

available_metrics = [
    metric
    for metric in METRIC_LABELS
    if metric in runs.columns
]


metric_records = []


for metric in available_metrics:

    runs[
        metric
    ] = pd.to_numeric(
        runs[
            metric
        ],
        errors="coerce"
    )


    for pair in PAIR_ORDER:

        sub = runs[
            runs[
                "pair"
            ]
            == pair
        ].copy()


        sub = sub.dropna(
            subset=[
                metric
            ]
        )


        if len(
            sub
        ) == 0:

            continue


        metric_records.append({

            "pair":
                pair,

            "metric":
                metric,

            "delta_lambda":
                PAIR_META[
                    pair
                ][
                    "delta_lambda"
                ],

            "mean_lambda":
                PAIR_META[
                    pair
                ][
                    "mean_lambda"
                ],

            "architecture_spread":
                architecture_spread(
                    sub,
                    metric
                ),

            "pooled_replicate_sd":
                pooled_within_architecture_sd(
                    sub,
                    metric
                ),

            "signal_noise":
                architecture_signal_noise(
                    sub,
                    metric
                ),

            "omega2":
                omega_squared(
                    sub,
                    metric
                ),
        })


all_metric_stats = pd.DataFrame(
    metric_records
)


# ============================================================
# SPEARMAN CORRELATIONS FOR SNR
# ============================================================

correlation_records = []


for metric in available_metrics:

    sub = all_metric_stats[
        all_metric_stats[
            "metric"
        ]
        == metric
    ].dropna(
        subset=[
            "delta_lambda",
            "signal_noise"
        ]
    )


    if len(
        sub
    ) < 3:

        continue


    rho, p = spearmanr(
        sub[
            "delta_lambda"
        ],
        sub[
            "signal_noise"
        ]
    )


    correlation_records.append({

        "metric":
            metric,

        "metric_label":
            METRIC_LABELS[
                metric
            ],

        "spearman_rho":
            rho,

        "spearman_p":
            p,
    })


correlations = pd.DataFrame(
    correlation_records
)


# ============================================================
# IQR-SPECIFIC WELCH ANOVA
# ============================================================

iqr_records = []


for pair in PAIR_ORDER:

    sub = runs[
        runs[
            "pair"
        ]
        == pair
    ].copy()


    sub = sub.dropna(
        subset=[
            PRIMARY_METRIC
        ]
    )


    F_welch, p_welch, df_denom = (
        welch_anova(
            sub,
            PRIMARY_METRIC
        )
    )


    iqr_records.append({

        "pair":
            pair,

        "motif_pair":
            PAIR_DISPLAY[
                pair
            ],

        "delta_lambda":
            PAIR_META[
                pair
            ][
                "delta_lambda"
            ],

        "mean_lambda":
            PAIR_META[
                pair
            ][
                "mean_lambda"
            ],

        "architecture_spread":
            architecture_spread(
                sub,
                PRIMARY_METRIC
            ),

        "pooled_replicate_sd":
            pooled_within_architecture_sd(
                sub,
                PRIMARY_METRIC
            ),

        "signal_noise":
            architecture_signal_noise(
                sub,
                PRIMARY_METRIC
            ),

        "omega2":
            omega_squared(
                sub,
                PRIMARY_METRIC
            ),

        "welch_F":
            F_welch,

        "welch_df_denom":
            df_denom,

        "welch_p":
            p_welch,
    })


iqr_stats = pd.DataFrame(
    iqr_records
)


# ============================================================
# BH-FDR CORRECTION ACROSS THE NINE IQR TESTS
# ============================================================

valid = iqr_stats[
    "welch_p"
].notna()


reject = np.full(
    len(
        iqr_stats
    ),
    False
)


q_values = np.full(
    len(
        iqr_stats
    ),
    np.nan
)


if valid.sum() > 0:

    reject_valid, q_valid, _, _ = (
        multipletests(
            iqr_stats.loc[
                valid,
                "welch_p"
            ].to_numpy(),
            alpha=ALPHA,
            method="fdr_bh"
        )
    )


    reject[
        np.where(
            valid
        )[0]
    ] = reject_valid


    q_values[
        np.where(
            valid
        )[0]
    ] = q_valid


iqr_stats[
    "BH_FDR_q"
] = q_values


iqr_stats[
    "BH_FDR_significant"
] = reject


# ============================================================
# TUKEY HSD FOR IQR
# ============================================================

tukey_records = []

tukey_summary_records = []


for pair in PAIR_ORDER:

    sub = runs[
        runs[
            "pair"
        ]
        == pair
    ].copy()


    sub = sub.dropna(
        subset=[
            PRIMARY_METRIC,
            "architecture_class"
        ]
    )


    sub = sub[
        sub[
            "architecture_class"
        ].isin(
            ARCH_ORDER
        )
    ]


    n_architectures = (
        sub[
            "architecture_class"
        ]
        .nunique()
    )


    if n_architectures < 2:

        continue


    tukey = pairwise_tukeyhsd(

        endog=sub[
            PRIMARY_METRIC
        ].to_numpy(),

        groups=sub[
            "architecture_class"
        ].astype(str).to_numpy(),

        alpha=ALPHA
    )


    table = pd.DataFrame(

        data=tukey
        ._results_table
        .data[
            1:
        ],

        columns=tukey
        ._results_table
        .data[
            0
        ]
    )


    for _, row in table.iterrows():

        tukey_records.append({

            "pair":
                pair,

            "delta_lambda":
                PAIR_META[
                    pair
                ][
                    "delta_lambda"
                ],

            "architecture_1":
                row[
                    "group1"
                ],

            "architecture_2":
                row[
                    "group2"
                ],

            "mean_difference":
                float(
                    row[
                        "meandiff"
                    ]
                ),

            "p_adjusted":
                float(
                    row[
                        "p-adj"
                    ]
                ),

            "ci_low":
                float(
                    row[
                        "lower"
                    ]
                ),

            "ci_high":
                float(
                    row[
                        "upper"
                    ]
                ),

            "significant":
                bool(
                    row[
                        "reject"
                    ]
                ),
        })


    n_comparisons = len(
        table
    )


    n_significant = int(
        table[
            "reject"
        ]
        .astype(bool)
        .sum()
    )


    tukey_summary_records.append({

        "pair":
            pair,

        "n_architectures":
            n_architectures,

        "n_pairwise_comparisons":
            n_comparisons,

        "n_significant":
            n_significant,

        "fraction_significant":
            (
                n_significant
                / n_comparisons
            ),
    })


tukey_all = pd.DataFrame(
    tukey_records
)


tukey_summary = pd.DataFrame(
    tukey_summary_records
)


# ============================================================
# MERGE COMPLETE IQR TABLE
# ============================================================

dr2_table = iqr_stats.merge(

    tukey_summary,

    on="pair",

    how="left"
)


# ============================================================
# ARCHITECTURE MEAN TABLE
#
# Useful for checking exactly what Panel A contains.
# ============================================================

architecture_mean_records = []


for pair in PAIR_ORDER:

    sub = runs[
        runs[
            "pair"
        ]
        == pair
    ]


    means = architecture_means(
        sub,
        PRIMARY_METRIC
    )


    if len(
        means
    ) == 0:

        continue


    chemistry_mean = (
        means[
            "mean"
        ].mean()
    )


    for _, row in means.iterrows():

        architecture_mean_records.append({

            "pair":
                pair,

            "motif_pair":
                PAIR_DISPLAY[
                    pair
                ],

            "delta_lambda":
                PAIR_META[
                    pair
                ][
                    "delta_lambda"
                ],

            "architecture":
                row[
                    "architecture"
                ],

            "architecture_mean_IQR_nm":
                row[
                    "mean"
                ],

            "architecture_sd_IQR_nm":
                row[
                    "sd"
                ],

            "n_replicates":
                row[
                    "n"
                ],

            "chemistry_mean_of_architecture_means_nm":
                chemistry_mean,

            "architecture_deviation_nm":
                (
                    row[
                        "mean"
                    ]
                    -
                    chemistry_mean
                ),
        })


architecture_means_table = pd.DataFrame(
    architecture_mean_records
)


# ============================================================
# SAVE TABLES
# ============================================================

dr2_table.to_csv(

    OUTDIR
    / "DR2_IQR_complete_statistics_table.csv",

    index=False
)


tukey_all.to_csv(

    OUTDIR
    / "DR2_IQR_Tukey_pairwise.csv",

    index=False
)


architecture_means_table.to_csv(

    OUTDIR
    / "DR2_IQR_architecture_means.csv",

    index=False
)


all_metric_stats.to_csv(

    OUTDIR
    / "DR2_all_metric_SNR_effect_sizes.csv",

    index=False
)


correlations.to_csv(

    OUTDIR
    / "DR2_all_metric_Spearman_correlations.csv",

    index=False
)


print("\n")
print("=" * 80)
print("IQR DESIGN RULE 2 STATISTICS")
print("=" * 80)

print(
    dr2_table.to_string(
        index=False
    )
)


# ============================================================
# FIGURE SAVE HELPER
# ============================================================

def save_figure(
    fig,
    basename
):

    fig.savefig(
        OUTDIR
        / f"{basename}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUTDIR
        / f"{basename}.png",
        dpi=600,
        bbox_inches="tight"
    )

    fig.savefig(
        OUTDIR
        / f"{basename}.svg",
        bbox_inches="tight"
    )


# ============================================================
# SMOOTH GUIDE
#
# Descriptive guide only, not inferential fit.
# ============================================================

def quadratic_guide(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )


    valid = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )


    x = x[
        valid
    ]

    y = y[
        valid
    ]


    coefficients = np.polyfit(
        x,
        y,
        deg=2
    )


    xfit = np.linspace(
        x.min(),
        x.max(),
        300
    )


    yfit = np.polyval(
        coefficients,
        xfit
    )


    return (
        xfit,
        yfit
    )


# ============================================================
# MAIN FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        11.3,
        8.0
    )
)


gs = fig.add_gridspec(

    2,
    3,

    height_ratios=[
        1.20,
        1.0
    ],

    width_ratios=[
        1.08,
        1.08,
        1.0
    ],

    hspace=0.42,

    wspace=0.46
)


# ============================================================
# PANEL A
#
# VIOLIN:
# exactly ONE value per architecture:
# mean of its three independent replicates.
#
# Therefore each violin contains eight values.
#
# PALE POINTS:
# individual replicate deviations.
#
# OPAQUE POINTS:
# architecture means.
# ============================================================

axA = fig.add_subplot(
    gs[
        0,
        :
    ]
)


arch_offsets = np.linspace(
    -0.27,
    0.27,
    len(
        ARCH_ORDER
    )
)


for pair_index, pair in enumerate(
    PAIR_ORDER
):

    pair_data = runs[
        runs[
            "pair"
        ]
        == pair
    ].copy()


    pair_data = pair_data.dropna(
        subset=[
            PRIMARY_METRIC
        ]
    )


    # --------------------------------------------------------
    # Architecture means
    # --------------------------------------------------------

    means = architecture_means(
        pair_data,
        PRIMARY_METRIC
    )


    # Only recognised architectures
    means = means[
        means[
            "architecture"
        ].isin(
            ARCH_ORDER
        )
    ].copy()


    # --------------------------------------------------------
    # Centre chemistry using equal weighting of architectures
    # --------------------------------------------------------

    chemistry_center = (
        means[
            "mean"
        ].mean()
    )


    means[
        "deviation"
    ] = (
        means[
            "mean"
        ]
        -
        chemistry_center
    )


    # ========================================================
    # VIOLIN
    #
    # IMPORTANT:
    # ONLY THE EIGHT ARCHITECTURE MEANS GO IN HERE.
    # ========================================================

    violin_values = (
        means[
            "deviation"
        ]
        .dropna()
        .to_numpy()
    )


    if len(
        violin_values
    ) >= 3:

        violin = axA.violinplot(

            [
                violin_values
            ],

            positions=[
                pair_index
            ],

            widths=0.82,

            showmeans=False,

            showmedians=False,

            showextrema=False,

            bw_method=0.45
        )


        for body in violin[
            "bodies"
        ]:

            body.set_facecolor(
                PAIR_COLORS[
                    pair
                ]
            )

            body.set_edgecolor(
                PAIR_COLORS[
                    pair
                ]
            )

            body.set_alpha(
                0.14
            )

            body.set_linewidth(
                0.9
            )


    # ========================================================
    # ARCHITECTURE POINTS
    # ========================================================

    for arch_index, arch in enumerate(
        ARCH_ORDER
    ):

        arch_data = pair_data[
            pair_data[
                "architecture_class"
            ]
            == arch
        ].copy()


        values = pd.to_numeric(
            arch_data[
                PRIMARY_METRIC
            ],
            errors="coerce"
        ).dropna().to_numpy()


        if len(
            values
        ) == 0:

            continue


        x = (
            pair_index
            +
            arch_offsets[
                arch_index
            ]
        )


        # ----------------------------------------------------
        # Replicates centred relative to same chemistry mean
        # ----------------------------------------------------

        replicate_deviation = (
            values
            -
            chemistry_center
        )


        replicate_jitter = np.linspace(
            -0.010,
            0.010,
            len(
                values
            )
        )


        axA.scatter(

            np.full(
                len(
                    values
                ),
                x
            )
            +
            replicate_jitter,

            replicate_deviation,

            marker=ARCH_MARKERS[
                arch
            ],

            color=ARCH_COLORS[
                arch
            ],

            s=19,

            alpha=0.18,

            linewidths=0.45,

            zorder=3
        )


        # ----------------------------------------------------
        # Architecture mean
        # ----------------------------------------------------

        architecture_mean = np.mean(
            values
        )


        architecture_deviation = (
            architecture_mean
            -
            chemistry_center
        )


        axA.scatter(

            x,

            architecture_deviation,

            marker=ARCH_MARKERS[
                arch
            ],

            color=ARCH_COLORS[
                arch
            ],

            s=43,

            alpha=0.98,

            edgecolors="white",

            linewidths=0.55,

            zorder=5
        )


# ------------------------------------------------------------
# Zero reference
# ------------------------------------------------------------

axA.axhline(

    0,

    linestyle="--",

    linewidth=0.9,

    alpha=0.40,

    zorder=1
)


# ------------------------------------------------------------
# X labels
# ------------------------------------------------------------

axA.set_xticks(
    np.arange(
        len(
            PAIR_ORDER
        )
    )
)


axA.set_xticklabels(

    [

        (
            f"{PAIR_DISPLAY[pair]}\n"
            +
            rf"$\Delta\bar{{\lambda}}$="
            +
            f"{PAIR_META[pair]['delta_lambda']:.3f}"
        )

        for pair
        in PAIR_ORDER
    ],

    rotation=31,

    ha="right"
)


axA.set_ylabel(

    "Architecture deviation in\n"
    r"central 50% $R_g$ range (nm)"
)


# ------------------------------------------------------------
# Architecture legend
# ------------------------------------------------------------

legend_handles = []


for arch in ARCH_ORDER:

    legend_handles.append(

        plt.Line2D(

            [0],
            [0],

            marker=ARCH_MARKERS[
                arch
            ],

            linestyle="None",

            markerfacecolor=ARCH_COLORS[
                arch
            ],

            markeredgecolor=ARCH_COLORS[
                arch
            ],

            markersize=6,

            label=arch
        )
    )


axA.legend(

    handles=legend_handles,

    frameon=False,

    ncol=4,

    loc="upper left",

    bbox_to_anchor=(
        0.0,
        1.01
    ),

    columnspacing=1.3,

    handletextpad=0.4,

    fontsize=7.3
)


axA.text(

    -0.055,

    1.04,

    "A",

    transform=axA.transAxes,

    fontsize=13,

    fontweight="bold"
)


for spine in [
    "top",
    "right"
]:

    axA.spines[
        spine
    ].set_visible(
        False
    )


# ============================================================
# PANEL B
# SIGNAL-TO-NOISE
# ============================================================

axB = fig.add_subplot(
    gs[
        1,
        0
    ]
)


METRIC_MARKERS = {
    "iqr_rg_nm": "o",
    "rg_range_90_nm": "s",
    "fraction_collapsed": "^",
    "k_collapse_per_ns": "D",
}


METRIC_SHORT_LABELS = {

    "iqr_rg_nm":
        r"Central 50% $R_g$",

    "rg_range_90_nm":
        r"Central 90% $R_g$",

    "fraction_collapsed":
        "% collapsed",

    "k_collapse_per_ns":
        r"$k_{\mathrm{collapse}}$",
}


metric_colors = {

    metric:
        DEFAULT_COLORS[
            i
        ]

    for i, metric
    in enumerate(
        MAIN_METRICS
    )
}


DISPLAY_X_OFFSETS = {

    "iqr_rg_nm":
        -0.0027,

    "rg_range_90_nm":
        -0.0009,

    "fraction_collapsed":
        0.0009,

    "k_collapse_per_ns":
        0.0027,
}


for metric in MAIN_METRICS:

    if metric not in available_metrics:
        continue


    sub = all_metric_stats[
        all_metric_stats[
            "metric"
        ]
        == metric
    ].sort_values(
        "delta_lambda"
    )


    x = sub[
        "delta_lambda"
    ].to_numpy()


    y = sub[
        "signal_noise"
    ].to_numpy()


    color = metric_colors[
        metric
    ]


    xfit, yfit = quadratic_guide(
        x,
        y
    )


    axB.plot(

        xfit,

        yfit,

        linestyle="--",

        linewidth=1.0,

        alpha=0.43,

        color=color,

        zorder=1
    )


    display_x = (
        x
        +
        DISPLAY_X_OFFSETS[
            metric
        ]
    )


    axB.scatter(

        display_x,

        y,

        marker=METRIC_MARKERS[
            metric
        ],

        color=color,

        s=42,

        alpha=0.92,

        edgecolors="white",

        linewidths=0.45,

        label=METRIC_SHORT_LABELS[
            metric
        ],

        zorder=4
    )


axB.set_xlabel(

    r"Motif stickiness contrast, "
    r"$\Delta\bar{\lambda}$"
)


axB.set_ylabel(
    "Signal-to-noise ratio"
)


axB.legend(

    frameon=False,

    loc="upper left",

    fontsize=7.1,

    handletextpad=0.5,

    borderaxespad=0.3
)


axB.text(

    -0.18,

    1.04,

    "B",

    transform=axB.transAxes,

    fontsize=13,

    fontweight="bold"
)


for spine in [
    "top",
    "right"
]:

    axB.spines[
        spine
    ].set_visible(
        False
    )


# ============================================================
# PANEL C
#
# Omega squared for IQR.
#
# Filled:
# BH-FDR q < 0.05
#
# Open:
# q >= 0.05
# ============================================================

axC = fig.add_subplot(
    gs[
        1,
        1
    ]
)


x = dr2_table[
    "delta_lambda"
].to_numpy()


y = dr2_table[
    "omega2"
].to_numpy()


# ------------------------------------------------------------
# Descriptive guide
# ------------------------------------------------------------

xfit, yfit = quadratic_guide(
    x,
    y
)


yfit = np.clip(
    yfit,
    0,
    1
)


axC.plot(

    xfit,

    yfit,

    linestyle="--",

    linewidth=1.25,

    alpha=0.48,

    color=DEFAULT_COLORS[
        0
    ],

    zorder=1
)


# ------------------------------------------------------------
# Points
# ------------------------------------------------------------

for _, row in dr2_table.iterrows():

    significant = bool(
        row[
            "BH_FDR_significant"
        ]
    )


    if significant:

        facecolor = DEFAULT_COLORS[
            0
        ]

    else:

        facecolor = "white"


    axC.scatter(

        row[
            "delta_lambda"
        ],

        row[
            "omega2"
        ],

        s=52,

        facecolor=facecolor,

        edgecolor=DEFAULT_COLORS[
            0
        ],

        linewidth=1.25,

        zorder=4
    )


# ------------------------------------------------------------
# Label positions
# ------------------------------------------------------------

LABEL_OFFSETS = {

    "IGLG":
        (-5, 10),

    "AVGAG":
        (8, -2),

    "IGIP":
        (8, 3),

    "VGAVG":
        (8, -10),

    "VGIP":
        (8, 8),

    "VGIG":
        (9, -17),

    "VAIP":
        (9, 9),

    "VAIG":
        (8, 8),

    "AVGFG":
        (8, 8),
}


for _, row in dr2_table.iterrows():

    dx, dy = LABEL_OFFSETS[
        row[
            "pair"
        ]
    ]


    axC.annotate(

        row[
            "pair"
        ],

        (
            row[
                "delta_lambda"
            ],

            row[
                "omega2"
            ]
        ),

        xytext=(
            dx,
            dy
        ),

        textcoords="offset points",

        fontsize=7,

        zorder=6
    )


axC.set_xlabel(

    r"Motif stickiness contrast, "
    r"$\Delta\bar{\lambda}$"
)


axC.set_ylabel(

    r"Architecture effect size, "
    r"$\omega^2$"
)


axC.set_ylim(
    -0.03,
    1.0
)


# ------------------------------------------------------------
# q-value key
# ------------------------------------------------------------

sig_handles = [

    plt.Line2D(

        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor=
            DEFAULT_COLORS[
                0
            ],

        markeredgecolor=
            DEFAULT_COLORS[
                0
            ],

        markersize=5.5,

        label=r"$q<0.05$"
    ),

    plt.Line2D(

        [0],
        [0],

        marker="o",

        linestyle="None",

        markerfacecolor="white",

        markeredgecolor=
            DEFAULT_COLORS[
                0
            ],

        markersize=5.5,

        label=r"$q\geq0.05$"
    ),
]


axC.legend(

    handles=sig_handles,

    frameon=False,

    loc="upper left",

    fontsize=7,

    handletextpad=0.4
)


axC.text(

    -0.18,

    1.04,

    "C",

    transform=axC.transAxes,

    fontsize=13,

    fontweight="bold"
)


for spine in [
    "top",
    "right"
]:

    axC.spines[
        spine
    ].set_visible(
        False
    )


# ============================================================
# PANEL D
# TUKEY COUNTS
# ============================================================

axD = fig.add_subplot(
    gs[
        1,
        2
    ]
)


tukey_plot = tukey_summary.copy()


tukey_plot[
    "pair_order"
] = tukey_plot[
    "pair"
].map(
    {
        pair: i

        for i, pair
        in enumerate(
            PAIR_ORDER
        )
    }
)


tukey_plot = tukey_plot.sort_values(
    "pair_order"
)


bar_x = np.arange(
    len(
        tukey_plot
    )
)


bars = axD.bar(

    bar_x,

    tukey_plot[
        "n_significant"
    ],

    width=0.70,

    alpha=0.80
)


for bar, (_, row) in zip(
    bars,
    tukey_plot.iterrows()
):

    value = int(
        row[
            "n_significant"
        ]
    )


    axD.text(

        bar.get_x()
        +
        bar.get_width()
        / 2,

        value
        + 0.45,

        f"{value}",

        ha="center",

        va="bottom",

        fontsize=7
    )


axD.set_xticks(
    bar_x
)


axD.set_xticklabels(

    tukey_plot[
        "pair"
    ],

    rotation=48,

    ha="right"
)


axD.set_ylabel(

    "Significant Tukey HSD\n"
    "comparisons"
)


axD.set_xlabel(
    "Motif pair"
)


axD.set_ylim(
    0,
    28
)


axD.set_yticks(
    [
        0,
        7,
        14,
        21,
        28
    ]
)


axD.text(

    0.98,

    0.96,

    "28 possible pairs",

    transform=axD.transAxes,

    ha="right",

    va="top",

    fontsize=7,

    alpha=0.65
)


axD.text(

    -0.18,

    1.04,

    "D",

    transform=axD.transAxes,

    fontsize=13,

    fontweight="bold"
)


for spine in [
    "top",
    "right"
]:

    axD.spines[
        spine
    ].set_visible(
        False
    )


# ============================================================
# SAVE MAIN FIGURE
# ============================================================

save_figure(

    fig,

    "figure_DR2_architecture_sensitivity_IQR_final"
)


plt.close(
    fig
)


# ============================================================
# SUPPLEMENTARY SNR HEATMAP
# ============================================================

HEATMAP_METRICS = [

    "iqr_rg_nm",

    "fraction_collapsed",

    "k_escape_per_ns",

    "k_collapse_per_ns",

    "mean_collapsed_dwell_ns",

    "std_rg_nm",

    "mean_expanded_dwell_ns",

    "mean_rg_nm",

    "rg_range_90_nm",

    "p95_rg_nm",

    "p05_rg_nm",

    "mean_collapse_depth_nm",

    "median_rg_nm",
]


HEATMAP_METRICS = [

    metric

    for metric
    in HEATMAP_METRICS

    if metric
    in available_metrics
]


heat_matrix = np.full(

    (
        len(
            HEATMAP_METRICS
        ),

        len(
            PAIR_ORDER
        )
    ),

    np.nan
)


for i, metric in enumerate(
    HEATMAP_METRICS
):

    for j, pair in enumerate(
        PAIR_ORDER
    ):

        row = all_metric_stats[
            (
                all_metric_stats[
                    "metric"
                ]
                == metric
            )
            &
            (
                all_metric_stats[
                    "pair"
                ]
                == pair
            )
        ]


        if len(
            row
        ):

            heat_matrix[
                i,
                j
            ] = row.iloc[
                0
            ][
                "signal_noise"
            ]


fig, ax = plt.subplots(
    figsize=(
        9.2,
        5.8
    )
)


im = ax.imshow(

    heat_matrix,

    aspect="auto",

    interpolation="nearest",

    cmap="viridis"
)


ax.set_xticks(
    np.arange(
        len(
            PAIR_ORDER
        )
    )
)


ax.set_xticklabels(

    [

        (
            f"{PAIR_DISPLAY[pair]}\n"
            +
            rf"$\Delta\bar{{\lambda}}$="
            +
            f"{PAIR_META[pair]['delta_lambda']:.3f}"
        )

        for pair
        in PAIR_ORDER
    ],

    rotation=45,

    ha="right"
)


ax.set_yticks(
    np.arange(
        len(
            HEATMAP_METRICS
        )
    )
)


ax.set_yticklabels(

    [

        METRIC_LABELS[
            metric
        ]

        for metric
        in HEATMAP_METRICS
    ]
)


ax.set_xlabel(

    "Motif pair, ordered by "
    r"stickiness contrast "
    r"$\Delta\bar{\lambda}$"
)


cbar = fig.colorbar(

    im,

    ax=ax,

    pad=0.02
)


cbar.set_label(
    "Signal-to-noise ratio"
)


# ------------------------------------------------------------
# Spearman values on supplementary heatmap
# ------------------------------------------------------------

for i, metric in enumerate(
    HEATMAP_METRICS
):

    row = correlations[
        correlations[
            "metric"
        ]
        == metric
    ]


    if len(
        row
    ) == 0:

        continue


    rho = row.iloc[
        0
    ][
        "spearman_rho"
    ]


    p = row.iloc[
        0
    ][
        "spearman_p"
    ]


    ax.text(

        len(
            PAIR_ORDER
        )
        - 0.02,

        i,

        (
            rf"  $\rho$={rho:.2f}, "
            rf"$p$={p:.1e}"
        ),

        ha="left",

        va="center",

        fontsize=7.3,

        clip_on=False
    )


ax.set_xlim(

    -0.5,

    len(
        PAIR_ORDER
    )
    + 2.2
)


fig.tight_layout()


save_figure(

    fig,

    "figure_S_DR2_SNR_heatmap"
)


plt.close(
    fig
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("DESIGN RULE 2 ANALYSIS COMPLETE")
print("=" * 80)


print(
    "\nPrimary statistical unit:"
)

print(
    "One independent simulation replicate "
    "from per_run_summary.csv"
)


print(
    "\nPanel A violin input:"
)

print(
    "ONE mean per architecture "
    "(mean of three replicates); "
    "8 values per motif pair."
)


print(
    "\nWelch ANOVA:"
)

print(
    "Calculated directly from replicate-level "
    "IQR values."
)


print(
    "\nBH-FDR:"
)

print(
    "Applied across the nine global "
    "IQR architecture tests."
)


print(
    "\nComplete IQR statistics:"
)

print(
    OUTDIR
    / "DR2_IQR_complete_statistics_table.csv"
)


print(
    "\nArchitecture means:"
)

print(
    OUTDIR
    / "DR2_IQR_architecture_means.csv"
)


print(
    "\nMain figure:"
)

print(
    OUTDIR
    / "figure_DR2_architecture_sensitivity_IQR_final.pdf"
)


print(
    "\nDone."
)