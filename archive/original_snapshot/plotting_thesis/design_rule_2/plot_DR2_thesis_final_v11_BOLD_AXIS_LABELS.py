from pathlib import Path
import argparse
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

parser = argparse.ArgumentParser(description="DR2 print-ready thesis figures")
parser.add_argument("--root", type=Path, default=Path(r"E:\Master thesis\calvados_tea\plotting_thesis\design_rule_2"))
parser.add_argument("--outdir", type=Path, default=None)
args = parser.parse_args()
ROOT = args.root

PER_RUN_FILES = [
    ROOT / "907_930_3reps_stride1" / "per_run_summary.csv",
    ROOT / "931_954_triplicate_stride1" / "per_run_summary.csv",
    ROOT / "lib1R4_triplicate_stride1" / "per_run_summary.csv",
]

OUTDIR = args.outdir or (ROOT / "design_rule_2_architecture_contrast" / "thesis_final")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

PRIMARY_METRIC = "rg_range_90_nm"

ALPHA = 0.05


# ============================================================
# MOTIF STICKINESS
# ============================================================

LAMBDA = {
    "VPAVG": 0.6294581781012,
    "VPGAG": 0.6568240708464,
    "VPGNG": 0.6527600406134,
    "VPGGG": 0.7124129984854,
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
    "GGNG": ("VPGGG", "VPGNG"),
    "LGIP": ("VPGLG", "IPAVG"),
    "GGAVG": ("VPGGG", "VPAVG"),
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

PAIR_ORDER = sorted(PAIR_INFO, key=lambda p: (PAIR_META[p]["delta_lambda"], p))
PAIR_DISPLAY = {p: f"{m1}–{m2}" for p, (m1, m2) in PAIR_INFO.items()}


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
    "Alternating": "#5B9BD5",
    "Small blocks": "#70AD47",
    "Medium blocks": "#8064A2",
    "Large blocks": "#ED7D31",
    "Diblock": "#8C564B",
    "Random shuffle": "#BDBDBD",
    r"Higher-$\lambda$ center": "#E53935",
    r"Higher-$\lambda$ ends": "#20B2C9",
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

    if not file.exists():
        raise FileNotFoundError(f"Missing required replicate summary: {file}")
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
        "GG": "VPGGG",
        "NG": "VPGNG",
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
# PRINT-READY FIGURES: main C, supplement B+D, two violins
# ============================================================
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 21,
    "axes.labelsize": 25, "axes.titlesize": 26,
    "xtick.labelsize": 21, "ytick.labelsize": 21,
    "legend.fontsize": 23, "axes.linewidth": 1.8,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

# Proteot-inspired colors: green = FDR significant; blue = nonsignificant.
PROTEOT_GREEN = "#98F522"  # Exact bright lime green sampled from DR5 figure (RGB 152,245,34)
PROTEOT_BLUE = "#245A81"

FIGDIR = OUTDIR / "thesis_figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def save_final(fig, name):
    for ext in ("pdf", "svg", "png"):
        opts = {"dpi": 400} if ext == "png" else {}
        dest = FIGDIR / (name + "." + ext)
        fig.savefig(dest, bbox_inches="tight", pad_inches=0.24, **opts)
        print("Saved", dest)
    plt.close(fig)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.grid(axis="y", color="0.85", lw=0.65, alpha=0.6)
    ax.set_axisbelow(True)


# MAIN FIGURE: ORIGINAL PANEL C, enlarged and collision-free labels.
# Use labels in a right-hand key with matching numbers, avoiding any
# ambiguous or overlapping point annotations.
fig = plt.figure(figsize=(21, 12.5))
gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.75], wspace=0.13)
ax = fig.add_subplot(gs[0, 0]); keyax = fig.add_subplot(gs[0, 1]); keyax.axis("off")
main = dr2_table.sort_values(["delta_lambda", "pair"]).reset_index(drop=True)
x = main["delta_lambda"].to_numpy(dtype=float)
y = main["omega2"].to_numpy(dtype=float)
if np.isfinite(x).sum() >= 3:
    xp, yp = quadratic_guide(x, y)
    ax.plot(xp, np.clip(yp, 0, 1), "--", color="0.65", lw=2.6,
            alpha=0.8, label="Quadratic visual guide", zorder=1)
for i, row in main.iterrows():
    significant = str(row["BH_FDR_significant"]).lower() in ("true", "1")
    ax.scatter(row["delta_lambda"], row["omega2"], s=1850,
               facecolor=PROTEOT_GREEN if significant else PROTEOT_BLUE,
               edgecolor="white", lw=3.0, zorder=4)
    ax.text(row["delta_lambda"], row["omega2"], str(i+1),
            ha="center", va="center", fontsize=19,
            fontweight="bold", color="white", zorder=5)
ax.set_xlabel(r"Motif stickiness contrast, $\Delta\bar{\lambda}$", labelpad=12, fontweight="bold")
ax.set_ylabel(r"Architecture effect size, $\omega^2$", labelpad=12, fontweight="bold")
ax.set_ylim(-0.065, 1.055)
ax.set_xlim(min(x)-0.020, max(x)+0.020)
clean_axes(ax)
keyax.text(0.0, 0.99, "Motif pairs", transform=keyax.transAxes,
           ha="left", va="top", fontsize=28, fontweight="bold")
# Each motif-pair entry inherits the significance color of its point.
# The right-aligned third column gives the actual motif-stickiness contrast.
keyax.text(0.98, 0.99, r"$\Delta\bar{\lambda}$", transform=keyax.transAxes,
           ha="right", va="top", fontsize=25, fontweight="bold")
for i, row in main.iterrows():
    significant = str(row["BH_FDR_significant"]).lower() in ("true", "1")
    entry_color = PROTEOT_GREEN if significant else PROTEOT_BLUE
    y_entry = 0.91 - i * 0.057
    keyax.text(0.00, y_entry, f"{i+1:>2}", transform=keyax.transAxes,
               fontsize=22, va="top", ha="left", fontweight="bold", color=entry_color)
    keyax.text(0.13, y_entry, PAIR_DISPLAY[row["pair"]],
               transform=keyax.transAxes, fontsize=22, va="top",
               ha="left", fontweight="semibold", color=entry_color)
    keyax.text(0.98, y_entry, f"{row['delta_lambda']:.3f}",
               transform=keyax.transAxes, fontsize=22, va="top",
               ha="right", fontweight="semibold", color=entry_color)

legend = [
    Line2D([], [], marker="o", linestyle="None", markersize=19,
           markerfacecolor=PROTEOT_GREEN, markeredgecolor="white", label=r"BH–FDR $q<0.05$"),
    Line2D([], [], marker="o", linestyle="None", markersize=19,
           markerfacecolor=PROTEOT_BLUE, markeredgecolor="white", label=r"BH–FDR $q\geq0.05$"),
    Line2D([], [], color="0.65", linestyle="--", lw=2.6,
           label="Descriptive guide"),
]
keyax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-0.01, 0.0),
             frameon=False, fontsize=21, borderaxespad=0, labelspacing=0.85)
save_final(fig, "DR2_MAIN_architecture_effect_size_omega2")


# SUPPLEMENT: original B and D, each expanded, stacked vertically.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 19),
                               gridspec_kw={"height_ratios": [1, 1.03]})
metrics = [
    ("iqr_rg_nm", r"Central 50% $R_g$", "o", "#285E88"),
    ("rg_range_90_nm", r"Central 90% $R_g$", "s", "#D38A27"),
    ("fraction_collapsed", "Compact-state population", "^", "#358A70"),
    ("k_collapse_per_ns", r"$k_{\mathrm{collapse}}$", "D", "#A3548A"),
]
for j, (metric, label, marker, color) in enumerate(metrics):
    sub = all_metric_stats.loc[all_metric_stats["metric"] == metric].sort_values("delta_lambda")
    if sub.empty:
        continue
    xx = sub["delta_lambda"].to_numpy(dtype=float)
    yy = sub["signal_noise"].to_numpy(dtype=float)
    if np.isfinite(xx).sum() >= 3 and np.isfinite(yy).sum() >= 3:
        xxfit, yyfit = quadratic_guide(xx, yy)
        ax1.plot(xxfit, yyfit, ls="--", lw=2.3, color=color, alpha=0.52)
    ax1.scatter(xx + (j-1.5)*0.00085, yy, marker=marker, s=195,
                facecolor=color, edgecolor="white", lw=0.7, zorder=4, label=label)
ax1.set_xlabel(r"Motif stickiness contrast, $\Delta\bar{\lambda}$", fontweight="bold")
ax1.set_ylabel("Architecture signal-to-noise ratio", fontweight="bold")
ax1.legend(frameon=False, ncol=2, loc="upper left", fontsize=23, markerscale=1.25, labelspacing=0.7)
ax1.text(-0.055, 1.025, "A", transform=ax1.transAxes, fontsize=26,
         fontweight="bold", va="bottom")
clean_axes(ax1)

bars_df = tukey_summary.set_index("pair").reindex(PAIR_ORDER).reset_index()
xs = np.arange(len(bars_df))
bars = ax2.bar(xs, bars_df["n_significant"].to_numpy(dtype=float),
               color=PROTEOT_BLUE, edgecolor="white", width=0.68)
for bar, count in zip(bars, bars_df["n_significant"]):
    if np.isfinite(count):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.45,
                 str(int(count)), ha="center", va="bottom", fontsize=21)
ax2.set_xticks(xs, [PAIR_DISPLAY[p] for p in PAIR_ORDER], rotation=42, ha="right", fontsize=19)
ax2.set_ylim(0, 30)
ax2.set_yticks([0, 7, 14, 21, 28])
ax2.set_ylabel("Significant Tukey HSD comparisons", fontweight="bold")
ax2.set_xlabel("Motif chemistry", fontweight="bold")
ax2.text(0.99, 0.96, "28 possible comparisons per chemistry",
         transform=ax2.transAxes, ha="right", va="top", fontsize=18, color="0.4")
ax2.text(-0.055, 1.025, "B", transform=ax2.transAxes, fontsize=26,
         fontweight="bold", va="bottom")
clean_axes(ax2)
fig.subplots_adjust(left=0.105, right=0.97, top=0.965, bottom=0.23, hspace=0.68)
save_final(fig, "DR2_SUPPLEMENT_signal_noise_and_Tukey")


# TWO STACKED VIOLINS:
# A: original DR2 centered IQR architecture deviations (nine pairs).
# B: conformational-regime central90 values, chemistry sorted by
#    mean compact-state population, with absolute Rg90 values.
# Distinct x-axes are intentional because the sort order differs.
fig, axes = plt.subplots(2, 1, figsize=(28, 25))
# Taller plotting areas: reduced unused space below legend and between panels.
arch_offsets = np.linspace(-0.275, 0.275, len(ARCH_ORDER))

# A — centered central90, 8 architecture means per violin.
ax = axes[0]
for pair_index, pair in enumerate(PAIR_ORDER):
    sub = runs.loc[runs["pair"] == pair].copy()
    means = architecture_means(sub, "rg_range_90_nm")
    means = means.loc[means["architecture"].isin(ARCH_ORDER)]
    if means.empty:
        continue
    center = means["mean"].mean()
    deviations = (means["mean"] - center).dropna().to_numpy(dtype=float)
    if len(deviations) >= 3 and np.ptp(deviations) > 1e-12:
        violin = ax.violinplot([deviations], positions=[pair_index],
                               widths=0.81, showmeans=False,
                               showmedians=False, showextrema=False,
                               bw_method=0.45)
        for body in violin["bodies"]:
            body.set_facecolor(PAIR_COLORS[pair]); body.set_edgecolor(PAIR_COLORS[pair])
            body.set_alpha(0.16); body.set_linewidth(1.0)
    for j, arch in enumerate(ARCH_ORDER):
        vals = pd.to_numeric(sub.loc[sub["architecture_class"] == arch, "rg_range_90_nm"],
                             errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        xpos = pair_index + arch_offsets[j]
        jit = np.linspace(-0.014, 0.014, len(vals))
        ax.scatter(xpos+jit, vals-center, marker=ARCH_MARKERS[arch],
                   s=85, facecolor=ARCH_COLORS[arch], alpha=0.24,
                   edgecolor="none", zorder=3)
        ax.scatter(xpos, np.mean(vals)-center, marker=ARCH_MARKERS[arch],
                   s=210, facecolor=ARCH_COLORS[arch], edgecolor="white",
                   linewidth=1.5, zorder=5)
ax.axhline(0, ls="--", lw=1.1, color="0.45")
ax.set_ylabel(r"$\Delta$(central 90% $R_g$ range) (nm)", fontweight="bold")
ax.set_xticks(np.arange(len(PAIR_ORDER)))
ax.set_xticklabels([PAIR_DISPLAY[p] + "\n" + r"$\Delta\bar{\lambda}=$" +
                    f"{PAIR_META[p]['delta_lambda']:.3f}" for p in PAIR_ORDER],
                   rotation=32, ha="right", fontsize=19)
ax.set_xlabel("Motif pair, ordered by stickiness contrast", fontweight="bold")
ax.text(-0.035, 1.025, "A", transform=ax.transAxes, fontsize=28,
        fontweight="bold", va="bottom")
clean_axes(ax)

# B — absolute central90 values, chemistry sorted by f_compact.
# First average replicates within each architecture; then average the
# eight architecture means for the chemistry's sorting statistic.
regime = []
for pair in PAIR_ORDER:
    sub = runs.loc[runs["pair"] == pair].copy()
    if "fraction_collapsed" not in sub:
        raise KeyError("fraction_collapsed is required for regime ordering")
    by_arch = sub.groupby("architecture_class")["fraction_collapsed"].mean()
    by_arch = by_arch.reindex(ARCH_ORDER).dropna()
    regime.append((pair, float(by_arch.mean())))
regime.sort(key=lambda item: item[1])
regime_pairs = [p for p, _ in regime]
regime_fraction = {p: f for p, f in regime}
ax = axes[1]
for pair_index, pair in enumerate(regime_pairs):
    sub = runs.loc[runs["pair"] == pair].copy()
    means = architecture_means(sub, "rg_range_90_nm")
    means = means.loc[means["architecture"].isin(ARCH_ORDER)]
    vals_violin = means["mean"].dropna().to_numpy(dtype=float)
    if len(vals_violin) >= 3 and np.ptp(vals_violin) > 1e-12:
        violin = ax.violinplot([vals_violin], positions=[pair_index],
                               widths=0.80, showmeans=False,
                               showmedians=False, showextrema=False,
                               bw_method=0.40)
        for body in violin["bodies"]:
            body.set_facecolor("0.76"); body.set_edgecolor("0.48")
            body.set_alpha(0.28); body.set_linewidth(1.0)
    for j, arch in enumerate(ARCH_ORDER):
        vals = pd.to_numeric(sub.loc[sub["architecture_class"] == arch, "rg_range_90_nm"],
                             errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        xpos = pair_index + arch_offsets[j]
        ax.scatter(xpos+np.linspace(-0.013, 0.013, len(vals)), vals,
                   marker=ARCH_MARKERS[arch], s=85,
                   facecolor=ARCH_COLORS[arch], alpha=0.23,
                   edgecolor="none", zorder=3)
        ax.errorbar(xpos, np.mean(vals),
                    yerr=np.std(vals, ddof=1) if len(vals)>1 else None,
                    fmt=ARCH_MARKERS[arch], ms=12.5,
                    color=ARCH_COLORS[arch], markeredgecolor="white",
                    markeredgewidth=1.0, elinewidth=2.1,
                    capsize=4, zorder=5)
ax.set_ylabel(r"Central 90% $R_g$ range (nm)", fontweight="bold")
ax.set_xticks(np.arange(len(regime_pairs)))
ax.set_xticklabels([PAIR_DISPLAY[p] + "\n" +
                    rf"$f_{{\mathrm{{compact}}}}={regime_fraction[p]:.2f}$"
                    for p in regime_pairs], rotation=32, ha="right", fontsize=19)
ax.set_xlabel("Motif pair, ordered by mean compact-state population", fontweight="bold")
ax.text(-0.035, 1.025, "B", transform=ax.transAxes, fontsize=28,
        fontweight="bold", va="bottom")
clean_axes(ax)
handles = [Line2D([], [], marker=ARCH_MARKERS[a], linestyle="None", markersize=16,
                  markerfacecolor=ARCH_COLORS[a], markeredgecolor="white", label=a)
           for a in ARCH_ORDER]
fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.52, 0.985),
           frameon=False, prop={"size": 30, "weight": "bold"},
           columnspacing=1.6, markerscale=1.35, handletextpad=0.55, labelspacing=0.9)
fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.115, hspace=0.53)
save_final(fig, "DR2_two_violin_panels_central90_and_conformational_regime")

print("\nCompleted DR2 thesis figures in", FIGDIR)
