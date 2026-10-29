# ============================================================
# DESIGN RULE 2 — SUPPLEMENTARY MULTI-METRIC ANALYSIS
# INCLUDING Rg DISTRIBUTION-OVERLAP METRICS
# ============================================================
#
# STANDARD SCALAR METRICS
# -----------------------
# Statistical unit:
#     one independent simulation replicate
#
# Design:
#     9 motif pairs
#     8 architectures
#     3 replicates
#     216 simulations
#
# For each metric × motif pair:
#     architecture spread =
#         max(architecture mean) - min(architecture mean)
#
#     pooled replicate SD =
#         pooled within-architecture SD
#
#     S/N =
#         architecture spread / pooled replicate SD
#
#     Welch ANOVA
#     omega^2
#     Tukey HSD
#     BH-FDR across 9 chemistries per metric
#
#
# NEW DISTRIBUTION-OVERLAP METRICS
# --------------------------------
#
# Input:
#     DR2_Rg_all_pairwise_overlap_metrics.csv
#
# Three pairwise distance measures:
#
#     1. Full Rg distribution distance
#     2. Central50 interval distance
#     3. Central90 interval distance
#
# To make their S/N NUMERICALLY COMPARABLE to the old S/N,
# each trajectory is converted to a replicate-level scalar:
#
#     mean distance from that trajectory to all 21 trajectories
#     belonging to the OTHER seven architectures.
#
# Thus:
#
#     24 trajectory scores per chemistry
#     = 8 architectures × 3 replicates
#
# We then apply EXACTLY the same S/N definition:
#
#     architecture spread / pooled replicate SD
#
# as for IQR, Rg90, kcollapse, etc.
#
#
# IMPORTANT STATISTICAL NOTE
# --------------------------
#
# The original scalar metrics are genuinely independent
# replicate-level observables.
#
# The overlap-derived scores are trajectory-level scores, but
# comparison partners are reused between scores. Therefore
# Welch/Tukey results for overlap metrics should be interpreted
# as supplementary/descriptive architecture-separation tests,
# not as perfectly independent replicate-level inferential tests.
#
# The S/N calculation itself is intentionally kept identical
# to the old definition so sensitivity magnitudes can be
# compared across all observables.
#
#
# HEATMAP
# -------
#
# A — Architecture S/N
# B — omega^2
# C — significant Tukey comparisons
#
# ============================================================


from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# ============================================================
# INPUT / OUTPUT
# ============================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis"
    r"\design_rule_book"
)


RUN_INPUT_FILES = [

    ROOT
    / "907_930_3reps_stride1"
    / "per_run_summary.csv",

    ROOT
    / "lib1R4_triplicate_stride1"
    / "per_run_summary.csv",
]


OVERLAP_DIR = (
    ROOT
    / "design_rule_2_architecture_contrast"
    / "distribution_overlap"
)


OVERLAP_PAIRWISE_FILE = (
    OVERLAP_DIR
    / "DR2_Rg_all_pairwise_overlap_metrics.csv"
)


OVERLAP_WITHIN_BETWEEN_FILE = (
    OVERLAP_DIR
    / "DR2_Rg_within_between_overlap_summary.csv"
)


OUTDIR = (
    ROOT
    / "design_rule_2_architecture_contrast"
    / "supplement_multimetric_with_overlap"
)


OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TEMPERATURE = 370.0

ALPHA = 0.05


# ============================================================
# MOTIF STICKINESS VALUES
# 370 K, gamma = 6
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

    "IGLG": (
        "VPGIG",
        "VPGLG",
    ),

    "AVGAG": (
        "VPAVG",
        "VPGAG",
    ),

    "VGAVG": (
        "VPGVG",
        "VPAVG",
    ),

    "IGIP": (
        "VPGIG",
        "IPAVG",
    ),

    "VGIP": (
        "VPGVG",
        "IPAVG",
    ),

    "VGIG": (
        "VPGVG",
        "VPGIG",
    ),

    "VAIP": (
        "VPAVG",
        "IPAVG",
    ),

    "VAIG": (
        "VPAVG",
        "VPGIG",
    ),

    "AVGFG": (
        "VPAVG",
        "VPGFG",
    ),
}


# ============================================================
# PAIR METADATA
# ============================================================

PAIR_META = {}


for pair, (
    motif_a,
    motif_b,
) in PAIR_INFO.items():

    lambda_a = LAMBDA[
        motif_a
    ]

    lambda_b = LAMBDA[
        motif_b
    ]

    delta_lambda = abs(
        lambda_a
        -
        lambda_b
    )

    mean_lambda = (
        lambda_a
        +
        lambda_b
    ) / 2.0


    PAIR_META[
        pair
    ] = {

        "motif_a":
            motif_a,

        "motif_b":
            motif_b,

        "lambda_a":
            lambda_a,

        "lambda_b":
            lambda_b,

        "delta_lambda":
            delta_lambda,

        "mean_lambda":
            mean_lambda,
    }


PAIR_ORDER = sorted(

    PAIR_INFO.keys(),

    key=lambda p:
        PAIR_META[p][
            "delta_lambda"
        ]
)


# ============================================================
# ARCHITECTURE ORDER
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


# ============================================================
# ARCHITECTURE CLASSIFICATION
# ============================================================

CODE_TO_MOTIF = {

    "VG": "VPGVG",

    "IG": "VPGIG",

    "LG": "VPGLG",

    "FG": "VPGFG",

    "AVG": "VPAVG",

    "VA": "VPAVG",

    "AG": "VPGAG",

    "IP": "IPAVG",
}


def get_high_lambda_motif(
    pair
):

    motif_a, motif_b = (
        PAIR_INFO[
            pair
        ]
    )


    if (
        LAMBDA[
            motif_a
        ]
        >
        LAMBDA[
            motif_b
        ]
    ):

        return motif_a


    return motif_b


def architecture_class(
    row
):

    arch_type = str(
        row.get(
            "architecture_type",
            ""
        )
    )


    sequence_id = str(
        row.get(
            "id",
            ""
        )
    )


    text = (
        arch_type
        +
        " "
        +
        sequence_id
    )


    text = (
        text
        .lower()
        .replace(
            " ",
            ""
        )
        .replace(
            "-",
            ""
        )
    )


    # --------------------------------------------------------
    # STANDARD ARCHITECTURES
    # --------------------------------------------------------

    if (
        "alternating" in text
        or "_even" in text
        or text.endswith(
            "even"
        )
    ):

        return "Alternating"


    if (
        "smallblocks" in text
        or "smallblock" in text
    ):

        return "Small blocks"


    if (
        "mediumblocks" in text
        or "mediumblock" in text
    ):

        return "Medium blocks"


    if (
        "largeblocks" in text
        or "largeblock" in text
    ):

        return "Large blocks"


    if "diblock" in text:

        return "Diblock"


    if (
        "randomshuffle" in text
        or "random_shuffle" in text
        or "shuffle" in text
    ):

        return "Random shuffle"


    # --------------------------------------------------------
    # DIRECTIONAL ARCHITECTURES
    # --------------------------------------------------------

    pair = row[
        "pair"
    ]


    high_motif = (
        get_high_lambda_motif(
            pair
        )
    )


    # Xcenter_Yends

    center_match = re.search(

        r"(AVG|VG|IG|LG|FG|VA|AG|IP)"
        r"center",

        text,

        flags=re.IGNORECASE
    )


    if center_match:

        center_code = (
            center_match
            .group(
                1
            )
            .upper()
        )


        center_motif = (
            CODE_TO_MOTIF.get(
                center_code
            )
        )


        if center_motif is None:

            return (
                "Unknown: "
                +
                arch_type
            )


        if (
            center_motif
            ==
            high_motif
        ):

            return (
                r"Higher-$\lambda$ center"
            )


        return (
            r"Higher-$\lambda$ ends"
        )


    # Xendcaps

    endcap_match = re.search(

        r"(AVG|VG|IG|LG|FG|VA|AG|IP)"
        r"endcaps",

        text,

        flags=re.IGNORECASE
    )


    if endcap_match:

        end_code = (
            endcap_match
            .group(
                1
            )
            .upper()
        )


        end_motif = (
            CODE_TO_MOTIF.get(
                end_code
            )
        )


        if end_motif is None:

            return (
                "Unknown: "
                +
                arch_type
            )


        if (
            end_motif
            ==
            high_motif
        ):

            return (
                r"Higher-$\lambda$ ends"
            )


        return (
            r"Higher-$\lambda$ center"
        )


    return (
        "Unknown: "
        +
        arch_type
    )


# ============================================================
# IDENTIFY MOTIF PAIR
# ============================================================

def identify_pair(
    row
):

    text = " ".join(
        [

            str(
                row.get(
                    "id",
                    ""
                )
            ),

            str(
                row.get(
                    "group",
                    ""
                )
            ),

            str(
                row.get(
                    "architecture_type",
                    ""
                )
            ),
        ]
    ).upper()


    pair_patterns = [

        "AVGFG",

        "AVGAG",

        "VGAVG",

        "IGIP",

        "IGLG",

        "VAIP",

        "VAIG",

        "VGIP",

        "VGIG",
    ]


    for pair in pair_patterns:

        if pair in text:

            return pair


    return np.nan


# ============================================================
# LOAD STANDARD PER-RUN DATA
# ============================================================

frames = []


for path in RUN_INPUT_FILES:

    if not path.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{path}"
        )


    print(
        "\nLoading:"
    )

    print(
        path
    )


    df = pd.read_csv(
        path
    )


    df[
        "source_file"
    ] = str(
        path
    )


    frames.append(
        df
    )


runs = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)


print(
    "\n"
    + "=" * 80
)


print(
    "STANDARD PER-RUN DATA LOADED"
)


print(
    "=" * 80
)


print(
    f"\nTotal rows loaded: "
    f"{len(runs)}"
)


# ============================================================
# TEMPERATURE FILTER
# ============================================================

if "temp" not in runs.columns:

    raise KeyError(
        "Column 'temp' was not found."
    )


runs[
    "temp_numeric"
] = pd.to_numeric(

    runs[
        "temp"
    ]
    .astype(
        str
    )
    .str.extract(
        r"(\d+(?:\.\d+)?)",
        expand=False
    ),

    errors="coerce"
)


runs = runs[

    np.isclose(

        runs[
            "temp_numeric"
        ],

        TEMPERATURE
    )

].copy()


print(
    f"\nRows remaining at "
    f"{TEMPERATURE:.0f} K: "
    f"{len(runs)}"
)


if runs.empty:

    raise RuntimeError(
        "No rows remain after temperature filtering."
    )


# ============================================================
# MOTIF PAIR IDENTIFICATION
# ============================================================

runs[
    "pair"
] = runs.apply(
    identify_pair,
    axis=1
)


runs = runs[
    runs[
        "pair"
    ].isin(
        PAIR_INFO.keys()
    )
].copy()


# ============================================================
# ARCHITECTURE CLASSIFICATION
# ============================================================

runs[
    "architecture_class"
] = [

    architecture_class(
        row
    )

    for _, row in runs.iterrows()
]


unknown_mask = ~runs[
    "architecture_class"
].isin(
    ARCH_ORDER
)


if unknown_mask.any():

    print(
        "\nWARNING — unknown architectures:"
    )


    print(
        runs.loc[
            unknown_mask,
            [
                "id",
                "pair",
                "architecture_type",
                "architecture_class",
            ]
        ].to_string(
            index=False
        )
    )


runs = runs[
    runs[
        "architecture_class"
    ].isin(
        ARCH_ORDER
    )
].copy()


# ============================================================
# STANDARD METRICS
# ============================================================

STANDARD_METRICS = {

    "iqr_rg_nm": {

        "column":
            "iqr_rg_nm",

        "label":
            r"Central 50% $R_g$ range (IQR)",
    },


    "rg_range_90_nm": {

        "column":
            "rg_range_90_nm",

        "label":
            r"Central 90% $R_g$ range",
    },


    "std_rg_nm": {

        "column":
            "std_rg_nm",

        "label":
            r"SD $R_g$",
    },


    "mean_rg_nm": {

        "column":
            "mean_rg_nm",

        "label":
            r"Mean $R_g$",
    },


    "median_rg_nm": {

        "column":
            "median_rg_nm",

        "label":
            r"Median $R_g$",
    },


    "p05_rg_nm": {

        "column":
            "p05_rg_nm",

        "label":
            r"$R_{g,5}$",
    },


    "p95_rg_nm": {

        "column":
            "p95_rg_nm",

        "label":
            r"$R_{g,95}$",
    },


    "fraction_collapsed": {

        "column":
            "fraction_collapsed",

        "label":
            r"Collapsed population",
    },


    "mean_collapse_depth_nm": {

        "column":
            "mean_collapse_depth_nm",

        "label":
            r"Mean collapse depth",
    },


    "k_collapse_per_ns": {

        "column":
            "k_collapse_per_ns",

        "label":
            r"$k_{\mathrm{collapse}}$",
    },


    "k_escape_per_ns": {

        "column":
            "k_escape_per_ns",

        "label":
            r"$k_{\mathrm{escape}}$",
    },


    "mean_collapsed_dwell_ns": {

        "column":
            "mean_collapsed_dwell_ns",

        "label":
            r"Collapsed-state dwell time",
    },


    "mean_expanded_dwell_ns": {

        "column":
            "mean_expanded_dwell_ns",

        "label":
            r"Expanded-state dwell time",
    },
}


# ============================================================
# VERIFY STANDARD METRICS
# ============================================================

missing_metrics = []


for metric, info in STANDARD_METRICS.items():

    column = info[
        "column"
    ]


    if column not in runs.columns:

        missing_metrics.append(
            column
        )


if missing_metrics:

    raise KeyError(
        "\nMissing standard metric columns:\n"
        +
        "\n".join(
            missing_metrics
        )
    )


for metric, info in STANDARD_METRICS.items():

    column = info[
        "column"
    ]


    runs[
        column
    ] = pd.to_numeric(
        runs[
            column
        ],
        errors="coerce"
    )


# ============================================================
# LOAD PAIRWISE OVERLAP DATA
# ============================================================

if not OVERLAP_PAIRWISE_FILE.exists():

    raise FileNotFoundError(
        "\nOverlap pairwise file not found:\n"
        f"{OVERLAP_PAIRWISE_FILE}"
    )


print(
    "\nLoading overlap data:"
)


print(
    OVERLAP_PAIRWISE_FILE
)


overlap_pairs = pd.read_csv(
    OVERLAP_PAIRWISE_FILE
)


print(
    f"\nPairwise overlap rows: "
    f"{len(overlap_pairs):,}"
)


# ============================================================
# VERIFY OVERLAP COLUMNS
# ============================================================

OVERLAP_DISTANCE_COLUMNS = {

    "full_distribution_distance": {

        "column":
            "full_distance",

        "label":
            r"Full $R_g$ distribution separation",
    },


    "central50_interval_distance": {

        "column":
            "central50_interval_distance",

        "label":
            r"Central 50% interval separation",
    },


    "central90_interval_distance": {

        "column":
            "central90_interval_distance",

        "label":
            r"Central 90% interval separation",
    },
}


required_overlap_columns = [

    "pair",

    "distribution_A",

    "distribution_B",

    "architecture_A",

    "architecture_B",

    "replicate_A",

    "replicate_B",

    "comparison_type",
]


for info in OVERLAP_DISTANCE_COLUMNS.values():

    required_overlap_columns.append(
        info[
            "column"
        ]
    )


missing_overlap_columns = [

    column

    for column in required_overlap_columns

    if column not in overlap_pairs.columns
]


if missing_overlap_columns:

    raise KeyError(
        "\nMissing overlap columns:\n"
        +
        "\n".join(
            missing_overlap_columns
        )
    )


# ============================================================
# CREATE TRAJECTORY-LEVEL OVERLAP SCORES
# ============================================================
#
# For each individual trajectory:
#
#     mean distance to all trajectories belonging to the
#     OTHER seven architectures.
#
# Complete design:
#
#     7 other architectures × 3 replicates = 21 distances
#
# This creates exactly one scalar per trajectory.
#
# We deliberately exclude same-architecture comparisons from
# the trajectory score because they represent replicate noise,
# not architecture separation.
#
# ============================================================

overlap_score_rows = []


for pair in PAIR_ORDER:

    pair_df = overlap_pairs[
        overlap_pairs[
            "pair"
        ]
        ==
        pair
    ].copy()


    distributions = sorted(
        set(
            pair_df[
                "distribution_A"
            ].dropna()
        )
        |
        set(
            pair_df[
                "distribution_B"
            ].dropna()
        )
    )


    print(
        f"\n{pair}: "
        f"{len(distributions)} overlap distributions"
    )


    for distribution_id in distributions:

        involving = pair_df[

            (
                pair_df[
                    "distribution_A"
                ]
                ==
                distribution_id
            )

            |

            (
                pair_df[
                    "distribution_B"
                ]
                ==
                distribution_id
            )
        ].copy()


        if involving.empty:

            continue


        # ----------------------------------------------------
        # Determine architecture and replicate for this
        # trajectory from whichever side it occurs on.
        # ----------------------------------------------------

        side_a = involving[
            involving[
                "distribution_A"
            ]
            ==
            distribution_id
        ]


        if not side_a.empty:

            architecture = (
                side_a[
                    "architecture_A"
                ]
                .iloc[
                    0
                ]
            )

            replicate = (
                side_a[
                    "replicate_A"
                ]
                .iloc[
                    0
                ]
            )

        else:

            side_b = involving[
                involving[
                    "distribution_B"
                ]
                ==
                distribution_id
            ]

            architecture = (
                side_b[
                    "architecture_B"
                ]
                .iloc[
                    0
                ]
            )

            replicate = (
                side_b[
                    "replicate_B"
                ]
                .iloc[
                    0
                ]
            )


        # ----------------------------------------------------
        # BETWEEN-ARCHITECTURE comparisons only
        # ----------------------------------------------------

        between = involving[
            involving[
                "comparison_type"
            ]
            ==
            "between_architecture"
        ].copy()


        within = involving[
            involving[
                "comparison_type"
            ]
            ==
            "within_architecture"
        ].copy()


        row = {

            "pair":
                pair,

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

            "distribution_id":
                distribution_id,

            "architecture_class":
                architecture,

            "replicate":
                replicate,

            "n_between_distances":
                len(
                    between
                ),

            "n_within_distances":
                len(
                    within
                ),
        }


        # ----------------------------------------------------
        # Mean separation score for each overlap metric
        # ----------------------------------------------------

        for metric, info in (
            OVERLAP_DISTANCE_COLUMNS.items()
        ):

            column = info[
                "column"
            ]


            values_between = pd.to_numeric(
                between[
                    column
                ],
                errors="coerce"
            )


            values_within = pd.to_numeric(
                within[
                    column
                ],
                errors="coerce"
            )


            row[
                metric
            ] = (
                values_between.mean()
            )


            row[
                metric
                +
                "_within_mean"
            ] = (
                values_within.mean()
            )


        overlap_score_rows.append(
            row
        )


overlap_scores = pd.DataFrame(
    overlap_score_rows
)


# ============================================================
# VALIDATE OVERLAP SCORE DESIGN
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "OVERLAP TRAJECTORY-SCORE VALIDATION"
)


print(
    "=" * 80
)


overlap_validation = (

    overlap_scores

    .groupby(
        [
            "pair",
            "architecture_class",
        ]
    )

    .size()

    .unstack(
        fill_value=0
    )

    .reindex(
        index=PAIR_ORDER,
        columns=ARCH_ORDER,
        fill_value=0
    )
)


print(
    overlap_validation.to_string()
)


print(
    "\nExpected: 3 scores per architecture."
)


print(
    "Expected between distances per trajectory: 21."
)


print(
    "Expected within distances per trajectory: 2."
)


if len(
    overlap_scores
) != 216:

    warnings.warn(
        f"Found {len(overlap_scores)} overlap trajectory "
        "scores; expected 216."
    )


bad_between = overlap_scores[
    overlap_scores[
        "n_between_distances"
    ]
    !=
    21
]


bad_within = overlap_scores[
    overlap_scores[
        "n_within_distances"
    ]
    !=
    2
]


if not bad_between.empty:

    warnings.warn(
        "Some overlap trajectory scores do not have "
        "21 between-architecture comparisons."
    )


if not bad_within.empty:

    warnings.warn(
        "Some overlap trajectory scores do not have "
        "2 within-architecture comparisons."
    )


OVERLAP_SCORE_CSV = (
    OUTDIR
    / "DR2_overlap_trajectory_level_scores.csv"
)


overlap_scores.to_csv(
    OVERLAP_SCORE_CSV,
    index=False
)


# ============================================================
# OPTIONAL: LOAD ORIGINAL BETWEEN/WITHIN RATIOS
# ============================================================
#
# These are saved separately for interpretation.
# They are NOT used as the S/N values in the heatmap.
#
# ============================================================

if OVERLAP_WITHIN_BETWEEN_FILE.exists():

    original_overlap_ratio_df = pd.read_csv(
        OVERLAP_WITHIN_BETWEEN_FILE
    )

    ORIGINAL_RATIO_OUT = (
        OUTDIR
        / "DR2_overlap_original_between_within_ratios.csv"
    )

    original_overlap_ratio_df.to_csv(
        ORIGINAL_RATIO_OUT,
        index=False
    )

else:

    original_overlap_ratio_df = None

    ORIGINAL_RATIO_OUT = None


# ============================================================
# METRIC DEFINITIONS
# ============================================================

METRICS = {}


# Standard metrics

for metric, info in STANDARD_METRICS.items():

    METRICS[
        metric
    ] = {

        "column":
            info[
                "column"
            ],

        "label":
            info[
                "label"
            ],

        "source":
            "standard",
    }


# Overlap-derived metrics

for metric, info in OVERLAP_DISTANCE_COLUMNS.items():

    METRICS[
        metric
    ] = {

        "column":
            metric,

        "label":
            info[
                "label"
            ],

        "source":
            "overlap",
    }


# ============================================================
# WELCH ONE-WAY ANOVA
# ============================================================

def welch_anova(
    groups
):

    clean_groups = []


    for g in groups:

        g = np.asarray(
            g,
            dtype=float
        )


        g = g[
            np.isfinite(
                g
            )
        ]


        if len(
            g
        ) >= 2:

            clean_groups.append(
                g
            )


    groups = clean_groups


    k = len(
        groups
    )


    if k < 2:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    n = np.array(
        [
            len(
                g
            )
            for g in groups
        ],
        dtype=float
    )


    means = np.array(
        [
            np.mean(
                g
            )
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


    variances = np.where(
        variances <= 0,
        1e-15,
        variances
    )


    weights = (
        n
        /
        variances
    )


    weight_sum = np.sum(
        weights
    )


    weighted_mean = np.sum(
        weights
        *
        means
    ) / weight_sum


    numerator = np.sum(
        weights
        *
        (
            means
            -
            weighted_mean
        ) ** 2
    ) / (
        k
        -
        1
    )


    term = np.sum(
        (
            1.0
            /
            (
                n
                -
                1.0
            )
        )
        *
        (
            1.0
            -
            weights
            /
            weight_sum
        ) ** 2
    )


    correction = (
        1.0
        +
        (
            2.0
            *
            (
                k
                -
                2.0
            )
            /
            (
                k ** 2
                -
                1.0
            )
        )
        *
        term
    )


    F = (
        numerator
        /
        correction
    )


    df_num = (
        k
        -
        1
    )


    if term <= 0:

        return (
            F,
            np.inf,
            0.0,
        )


    df_den = (
        (
            k ** 2
            -
            1.0
        )
        /
        (
            3.0
            *
            term
        )
    )


    p = stats.f.sf(
        F,
        df_num,
        df_den
    )


    return (
        float(
            F
        ),
        float(
            df_den
        ),
        float(
            p
        ),
    )


# ============================================================
# OMEGA-SQUARED
# ============================================================

def omega_squared(
    groups
):

    clean_groups = []


    for g in groups:

        g = np.asarray(
            g,
            dtype=float
        )


        g = g[
            np.isfinite(
                g
            )
        ]


        if len(
            g
        ) > 0:

            clean_groups.append(
                g
            )


    groups = clean_groups


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


    ss_between = 0.0


    for g in groups:

        ss_between += (
            len(
                g
            )
            *
            (
                np.mean(
                    g
                )
                -
                grand_mean
            ) ** 2
        )


    ss_within = 0.0


    for g in groups:

        group_mean = np.mean(
            g
        )


        ss_within += np.sum(
            (
                g
                -
                group_mean
            ) ** 2
        )


    k = len(
        groups
    )


    N = len(
        all_values
    )


    if N <= k:

        return np.nan


    ms_within = (
        ss_within
        /
        (
            N
            -
            k
        )
    )


    ss_total = (
        ss_between
        +
        ss_within
    )


    denominator = (
        ss_total
        +
        ms_within
    )


    if denominator <= 0:

        return 0.0


    omega2 = (
        ss_between
        -
        (
            k
            -
            1
        )
        *
        ms_within
    ) / denominator


    return max(
        0.0,
        float(
            omega2
        )
    )


# ============================================================
# POOLED WITHIN-ARCHITECTURE SD
# ============================================================

def pooled_sd(
    groups
):

    numerator = 0.0

    denominator = 0


    for g in groups:

        g = np.asarray(
            g,
            dtype=float
        )


        g = g[
            np.isfinite(
                g
            )
        ]


        if len(
            g
        ) < 2:

            continue


        variance = np.var(
            g,
            ddof=1
        )


        numerator += (
            (
                len(
                    g
                )
                -
                1
            )
            *
            variance
        )


        denominator += (
            len(
                g
            )
            -
            1
        )


    if denominator == 0:

        return np.nan


    return np.sqrt(
        numerator
        /
        denominator
    )


# ============================================================
# GET DATASET FOR METRIC
# ============================================================

def get_metric_dataset(
    metric
):

    info = METRICS[
        metric
    ]


    if (
        info[
            "source"
        ]
        ==
        "standard"
    ):

        return (
            runs,
            info[
                "column"
            ]
        )


    elif (
        info[
            "source"
        ]
        ==
        "overlap"
    ):

        return (
            overlap_scores,
            info[
                "column"
            ]
        )


    else:

        raise ValueError(
            f"Unknown metric source: "
            f"{info['source']}"
        )


# ============================================================
# MAIN STATISTICAL ANALYSIS
# ============================================================

global_rows = []

tukey_rows = []


for metric, info in METRICS.items():

    source_df, column = (
        get_metric_dataset(
            metric
        )
    )


    label = info[
        "label"
    ]


    source_type = info[
        "source"
    ]


    print(
        "\n"
        + "=" * 80
    )


    print(
        f"ANALYSING: {label}"
    )


    print(
        f"Source: {source_type}"
    )


    print(
        "=" * 80
    )


    for pair in PAIR_ORDER:

        sub = source_df[
            source_df[
                "pair"
            ]
            ==
            pair
        ].copy()


        sub[
            column
        ] = pd.to_numeric(
            sub[
                column
            ],
            errors="coerce"
        )


        sub = sub[
            np.isfinite(
                sub[
                    column
                ]
            )
        ].copy()


        groups = []

        architecture_means = []


        for arch in ARCH_ORDER:

            values = (
                sub.loc[
                    sub[
                        "architecture_class"
                    ]
                    ==
                    arch,
                    column
                ]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )


            if len(
                values
            ) > 0:

                groups.append(
                    values
                )


                architecture_means.append(
                    np.mean(
                        values
                    )
                )


        n_arch = len(
            groups
        )


        if n_arch < 2:

            warnings.warn(
                f"Skipping {metric} / {pair}: "
                "fewer than 2 architectures."
            )

            continue


        # ----------------------------------------------------
        # EXACT SAME S/N DEFINITION FOR ALL 16 METRICS
        # ----------------------------------------------------

        architecture_spread = (
            np.max(
                architecture_means
            )
            -
            np.min(
                architecture_means
            )
        )


        pooled_rep_sd = pooled_sd(
            groups
        )


        if (
            np.isfinite(
                pooled_rep_sd
            )
            and
            pooled_rep_sd > 0
        ):

            signal_noise = (
                architecture_spread
                /
                pooled_rep_sd
            )

        else:

            signal_noise = np.nan


        # ----------------------------------------------------
        # EFFECT SIZE
        # ----------------------------------------------------

        omega2 = omega_squared(
            groups
        )


        # ----------------------------------------------------
        # WELCH ANOVA
        # ----------------------------------------------------

        (
            welch_F,
            welch_df,
            welch_p,
        ) = welch_anova(
            groups
        )


        # ----------------------------------------------------
        # TUKEY HSD
        # ----------------------------------------------------

        tukey_sub = sub[
            [
                column,
                "architecture_class",
            ]
        ].dropna()


        n_sig = 0

        n_pairs = 0


        if (
            tukey_sub[
                "architecture_class"
            ].nunique()
            >=
            2
        ):

            tukey = pairwise_tukeyhsd(

                endog=(
                    tukey_sub[
                        column
                    ].to_numpy()
                ),

                groups=(
                    tukey_sub[
                        "architecture_class"
                    ].to_numpy()
                ),

                alpha=ALPHA,
            )


            table = (
                tukey
                .summary()
                .data
            )


            tukey_df = pd.DataFrame(
                table[
                    1:
                ],
                columns=table[
                    0
                ]
            )


            n_pairs = len(
                tukey_df
            )


            reject_values = (
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
                        "1",
                    ]
                )
            )


            n_sig = int(
                reject_values.sum()
            )


            for _, tr in tukey_df.iterrows():

                reject = str(
                    tr[
                        "reject"
                    ]
                ).lower() in [
                    "true",
                    "1",
                ]


                tukey_rows.append(
                    {

                        "metric":
                            metric,

                        "metric_label":
                            label,

                        "metric_source":
                            source_type,

                        "pair":
                            pair,

                        "motif_pair":
                            (
                                f"{PAIR_INFO[pair][0]}"
                                "–"
                                f"{PAIR_INFO[pair][1]}"
                            ),

                        "delta_lambda":
                            PAIR_META[
                                pair
                            ][
                                "delta_lambda"
                            ],

                        "architecture_1":
                            tr[
                                "group1"
                            ],

                        "architecture_2":
                            tr[
                                "group2"
                            ],

                        "mean_difference":
                            float(
                                tr[
                                    "meandiff"
                                ]
                            ),

                        "p_adjusted_Tukey":
                            float(
                                tr[
                                    "p-adj"
                                ]
                            ),

                        "lower":
                            float(
                                tr[
                                    "lower"
                                ]
                            ),

                        "upper":
                            float(
                                tr[
                                    "upper"
                                ]
                            ),

                        "significant":
                            reject,
                    }
                )


            if n_pairs > 0:

                fraction_sig = (
                    n_sig
                    /
                    n_pairs
                )

            else:

                fraction_sig = np.nan

        else:

            fraction_sig = np.nan


        # ----------------------------------------------------
        # SAVE GLOBAL RESULT
        # ----------------------------------------------------

        global_rows.append(
            {

                "metric":
                    metric,

                "metric_label":
                    label,

                "metric_source":
                    source_type,

                "metric_column":
                    column,

                "pair":
                    pair,

                "motif_pair":
                    (
                        f"{PAIR_INFO[pair][0]}"
                        "–"
                        f"{PAIR_INFO[pair][1]}"
                    ),

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

                "n_architectures":
                    n_arch,

                "architecture_spread":
                    architecture_spread,

                "pooled_replicate_sd":
                    pooled_rep_sd,

                "signal_noise":
                    signal_noise,

                "omega2":
                    omega2,

                "welch_F":
                    welch_F,

                "welch_df_denom":
                    welch_df,

                "welch_p":
                    welch_p,

                "n_pairwise_comparisons":
                    n_pairs,

                "n_significant_Tukey":
                    n_sig,

                "fraction_significant_Tukey":
                    fraction_sig,

                "percent_significant_Tukey":
                    (
                        100.0
                        *
                        fraction_sig
                        if np.isfinite(
                            fraction_sig
                        )
                        else np.nan
                    ),
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

global_df = pd.DataFrame(
    global_rows
)


tukey_all_df = pd.DataFrame(
    tukey_rows
)


# ============================================================
# BH-FDR
#
# Separately for each observable across 9 motif chemistries.
# ============================================================

global_df[
    "BH_FDR_q"
] = np.nan


global_df[
    "BH_FDR_significant"
] = False


for metric in global_df[
    "metric"
].unique():

    metric_mask = (
        global_df[
            "metric"
        ]
        ==
        metric
    )


    valid_mask = (
        metric_mask
        &
        global_df[
            "welch_p"
        ].notna()
    )


    p_values = (
        global_df.loc[
            valid_mask,
            "welch_p"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if len(
        p_values
    ) == 0:

        continue


    (
        reject,
        q_values,
        _,
        _,
    ) = multipletests(
        p_values,
        alpha=ALPHA,
        method="fdr_bh",
    )


    global_df.loc[
        valid_mask,
        "BH_FDR_q"
    ] = q_values


    global_df.loc[
        valid_mask,
        "BH_FDR_significant"
    ] = reject


# ============================================================
# TUKEY SUMMARY
# ============================================================

tukey_summary_df = (

    global_df[
        [
            "metric",
            "metric_label",
            "metric_source",
            "pair",
            "motif_pair",
            "delta_lambda",
            "n_pairwise_comparisons",
            "n_significant_Tukey",
            "fraction_significant_Tukey",
            "percent_significant_Tukey",
            "welch_p",
            "BH_FDR_q",
            "BH_FDR_significant",
        ]
    ]

    .copy()
)


# ============================================================
# METRIC SENSITIVITY RANKING
# ============================================================

metric_summary_rows = []


for metric, sub in global_df.groupby(
    "metric"
):

    label = (
        sub[
            "metric_label"
        ]
        .iloc[
            0
        ]
    )


    source_type = (
        sub[
            "metric_source"
        ]
        .iloc[
            0
        ]
    )


    # --------------------------------------------------------
    # Delta lambda vs S/N
    # --------------------------------------------------------

    valid_sn = (
        sub[
            [
                "delta_lambda",
                "signal_noise",
            ]
        ]
        .dropna()
    )


    if len(
        valid_sn
    ) >= 3:

        rho_sn, p_sn = stats.spearmanr(

            valid_sn[
                "delta_lambda"
            ],

            valid_sn[
                "signal_noise"
            ]
        )

    else:

        rho_sn = np.nan

        p_sn = np.nan


    # --------------------------------------------------------
    # Delta lambda vs omega squared
    # --------------------------------------------------------

    valid_omega = (
        sub[
            [
                "delta_lambda",
                "omega2",
            ]
        ]
        .dropna()
    )


    if len(
        valid_omega
    ) >= 3:

        (
            rho_omega,
            p_omega,
        ) = stats.spearmanr(

            valid_omega[
                "delta_lambda"
            ],

            valid_omega[
                "omega2"
            ]
        )

    else:

        rho_omega = np.nan

        p_omega = np.nan


    # --------------------------------------------------------
    # Delta lambda vs Tukey fraction
    # --------------------------------------------------------

    valid_tukey = (
        sub[
            [
                "delta_lambda",
                "fraction_significant_Tukey",
            ]
        ]
        .dropna()
    )


    if len(
        valid_tukey
    ) >= 3:

        (
            rho_tukey,
            p_tukey,
        ) = stats.spearmanr(

            valid_tukey[
                "delta_lambda"
            ],

            valid_tukey[
                "fraction_significant_Tukey"
            ]
        )

    else:

        rho_tukey = np.nan

        p_tukey = np.nan


    total_sig = (
        sub[
            "n_significant_Tukey"
        ].sum()
    )


    total_comparisons = (
        sub[
            "n_pairwise_comparisons"
        ].sum()
    )


    if total_comparisons > 0:

        overall_tukey_fraction = (
            total_sig
            /
            total_comparisons
        )

    else:

        overall_tukey_fraction = np.nan


    metric_summary_rows.append(
        {

            "metric":
                metric,

            "metric_label":
                label,

            "metric_source":
                source_type,

            "spearman_deltaLambda_vs_signal_noise":
                rho_sn,

            "p_deltaLambda_vs_signal_noise":
                p_sn,

            "spearman_deltaLambda_vs_omega2":
                rho_omega,

            "p_deltaLambda_vs_omega2":
                p_omega,

            "spearman_deltaLambda_vs_fraction_Tukey":
                rho_tukey,

            "p_deltaLambda_vs_fraction_Tukey":
                p_tukey,

            "mean_signal_noise":
                sub[
                    "signal_noise"
                ].mean(),

            "median_signal_noise":
                sub[
                    "signal_noise"
                ].median(),

            "mean_omega2":
                sub[
                    "omega2"
                ].mean(),

            "total_significant_Tukey":
                int(
                    total_sig
                ),

            "total_Tukey_comparisons":
                int(
                    total_comparisons
                ),

            "overall_fraction_significant_Tukey":
                overall_tukey_fraction,

            "overall_percent_significant_Tukey":
                (
                    100.0
                    *
                    overall_tukey_fraction
                    if np.isfinite(
                        overall_tukey_fraction
                    )
                    else np.nan
                ),

            "n_global_FDR_significant":
                int(
                    sub[
                        "BH_FDR_significant"
                    ].sum()
                ),
        }
    )


metric_summary_df = pd.DataFrame(
    metric_summary_rows
)


metric_summary_df = (

    metric_summary_df

    .sort_values(
        [
            "spearman_deltaLambda_vs_signal_noise",
            "total_significant_Tukey",
        ],
        ascending=[
            False,
            False,
        ]
    )

    .reset_index(
        drop=True
    )
)


metric_summary_df[
    "sensitivity_rank"
] = (
    np.arange(
        len(
            metric_summary_df
        )
    )
    +
    1
)


# ============================================================
# SAVE STATISTICS
# ============================================================

GLOBAL_CSV = (
    OUTDIR
    /
    "DR2_all_16_metrics_global_statistics.csv"
)


TUKEY_ALL_CSV = (
    OUTDIR
    /
    "DR2_all_16_metrics_Tukey_all_comparisons.csv"
)


TUKEY_SUMMARY_CSV = (
    OUTDIR
    /
    "DR2_all_16_metrics_Tukey_summary.csv"
)


METRIC_RANKING_CSV = (
    OUTDIR
    /
    "DR2_all_16_metrics_sensitivity_ranking.csv"
)


global_df.to_csv(
    GLOBAL_CSV,
    index=False
)


tukey_all_df.to_csv(
    TUKEY_ALL_CSV,
    index=False
)


tukey_summary_df.to_csv(
    TUKEY_SUMMARY_CSV,
    index=False
)


metric_summary_df.to_csv(
    METRIC_RANKING_CSV,
    index=False
)


# ============================================================
# HEATMAP ORDER
# ============================================================

METRIC_ORDER = [

    # Ensemble breadth
    "iqr_rg_nm",
    "rg_range_90_nm",
    "std_rg_nm",

    # Full conformational distribution
    "full_distribution_distance",

    # Ensemble position
    "mean_rg_nm",
    "median_rg_nm",
    "p05_rg_nm",
    "p95_rg_nm",

    # Collapse equilibrium
    "fraction_collapsed",
    "mean_collapse_depth_nm",

    # Dynamics
    "k_collapse_per_ns",
    "k_escape_per_ns",
    "mean_collapsed_dwell_ns",
    "mean_expanded_dwell_ns",
]


METRIC_LABELS = [

    METRICS[
        metric
    ][
        "label"
    ]

    for metric in METRIC_ORDER
]


# ============================================================
# PAIR LABELS
# ============================================================

PAIR_LABELS = []


for pair in PAIR_ORDER:

    delta_lambda = (
        PAIR_META[
            pair
        ][
            "delta_lambda"
        ]
    )


    PAIR_LABELS.append(
        (
            f"{pair}\n"
            r"$\Delta\bar{\lambda}$="
            f"{delta_lambda:.3f}"
        )
    )


# ============================================================
# MATRIX CREATION
# ============================================================

def make_matrix(
    value_column
):

    matrix = np.full(
        (
            len(
                METRIC_ORDER
            ),
            len(
                PAIR_ORDER
            ),
        ),
        np.nan,
    )


    for i, metric in enumerate(
        METRIC_ORDER
    ):

        for j, pair in enumerate(
            PAIR_ORDER
        ):

            row = global_df[
                (
                    global_df[
                        "metric"
                    ]
                    ==
                    metric
                )
                &
                (
                    global_df[
                        "pair"
                    ]
                    ==
                    pair
                )
            ]


            if len(
                row
            ) == 1:

                matrix[
                    i,
                    j
                ] = row[
                    value_column
                ].iloc[
                    0
                ]


    return matrix


sn_matrix = make_matrix(
    "signal_noise"
)


omega_matrix = make_matrix(
    "omega2"
)


tukey_matrix = make_matrix(
    "percent_significant_Tukey"
)


tukey_n_sig_matrix = make_matrix(
    "n_significant_Tukey"
)


tukey_n_total_matrix = make_matrix(
    "n_pairwise_comparisons"
)


# ============================================================
# GLOBAL BH-FDR SIGNIFICANCE MATRIX
# ============================================================

q_sig_matrix = np.zeros(
    (
        len(
            METRIC_ORDER
        ),
        len(
            PAIR_ORDER
        ),
    ),
    dtype=bool
)


for i, metric in enumerate(
    METRIC_ORDER
):

    for j, pair in enumerate(
        PAIR_ORDER
    ):

        row = global_df[
            (
                global_df[
                    "metric"
                ]
                ==
                metric
            )
            &
            (
                global_df[
                    "pair"
                ]
                ==
                pair
            )
        ]


        if len(
            row
        ) == 1:

            q_sig_matrix[
                i,
                j
            ] = bool(
                row[
                    "BH_FDR_significant"
                ].iloc[
                    0
                ]
            )


# ============================================================
# FIGURE STYLE
# ============================================================

plt.rcParams.update(
    {

        "font.size":
            9,

        "axes.labelsize":
            10,

        "xtick.labelsize":
            8,

        "ytick.labelsize":
            8.5,

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    }
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(
        10.8,
        14.5
    ),
    constrained_layout=True
)


# ============================================================
# PANEL A — SIGNAL TO NOISE
# ============================================================

ax = axes[
    0
]


im_a = ax.imshow(
    sn_matrix,
    aspect="auto",
    interpolation="nearest"
)


ax.text(
    -0.08,
    1.04,
    "A",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    va="bottom",
    ha="left",
)


ax.set_yticks(
    np.arange(
        len(
            METRIC_LABELS
        )
    )
)


ax.set_yticklabels(
    METRIC_LABELS
)


ax.set_xticks(
    np.arange(
        len(
            PAIR_LABELS
        )
    )
)


ax.set_xticklabels(
    PAIR_LABELS,
    rotation=45,
    ha="right"
)


cbar_a = fig.colorbar(
    im_a,
    ax=ax,
    pad=0.015
)


cbar_a.set_label(
    "Architecture signal-to-noise ratio"
)


for i in range(
    len(
        METRIC_ORDER
    )
):

    for j in range(
        len(
            PAIR_ORDER
        )
    ):

        value = sn_matrix[
            i,
            j
        ]


        if not np.isfinite(
            value
        ):

            continue


        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=6.1
        )


# ============================================================
# PANEL B — OMEGA SQUARED
# ============================================================

ax = axes[
    1
]


im_b = ax.imshow(
    omega_matrix,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=1
)


ax.text(
    -0.08,
    1.04,
    "B",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    va="bottom",
    ha="left",
)


ax.set_yticks(
    np.arange(
        len(
            METRIC_LABELS
        )
    )
)


ax.set_yticklabels(
    METRIC_LABELS
)


ax.set_xticks(
    np.arange(
        len(
            PAIR_LABELS
        )
    )
)


ax.set_xticklabels(
    PAIR_LABELS,
    rotation=45,
    ha="right"
)


cbar_b = fig.colorbar(
    im_b,
    ax=ax,
    pad=0.015
)


cbar_b.set_label(
    r"Architecture effect size, $\omega^2$"
)


for i in range(
    len(
        METRIC_ORDER
    )
):

    for j in range(
        len(
            PAIR_ORDER
        )
    ):

        value = omega_matrix[
            i,
            j
        ]


        if not np.isfinite(
            value
        ):

            continue


        if q_sig_matrix[
            i,
            j
        ]:

            label = (
                f"{value:.2f}*"
            )

        else:

            label = (
                f"{value:.2f}"
            )


        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=6.0
        )


ax.text(
    1.0,
    -0.17,
    r"$*$ BH-FDR $q<0.05$",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=8
)


# ============================================================
# PANEL C — TUKEY HSD
# ============================================================

ax = axes[
    2
]


observed_tukey_max = np.nanmax(
    tukey_matrix
)


tukey_vmax = (
    np.ceil(
        observed_tukey_max
        /
        10.0
    )
    *
    10.0
)


tukey_vmax = max(
    10.0,
    tukey_vmax
)


im_c = ax.imshow(
    tukey_matrix,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=tukey_vmax
)


ax.text(
    -0.08,
    1.04,
    "C",
    transform=ax.transAxes,
    fontsize=14,
    fontweight="bold",
    va="bottom",
    ha="left",
)


ax.set_yticks(
    np.arange(
        len(
            METRIC_LABELS
        )
    )
)


ax.set_yticklabels(
    METRIC_LABELS
)


ax.set_xticks(
    np.arange(
        len(
            PAIR_LABELS
        )
    )
)


ax.set_xticklabels(
    PAIR_LABELS,
    rotation=45,
    ha="right"
)


cbar_c = fig.colorbar(
    im_c,
    ax=ax,
    pad=0.015
)


cbar_c.set_label(
    "Significant Tukey HSD comparisons (%)"
)


for i in range(
    len(
        METRIC_ORDER
    )
):

    for j in range(
        len(
            PAIR_ORDER
        )
    ):

        n_sig = tukey_n_sig_matrix[
            i,
            j
        ]


        n_total = tukey_n_total_matrix[
            i,
            j
        ]


        if (
            not np.isfinite(
                n_sig
            )
            or
            not np.isfinite(
                n_total
            )
        ):

            continue


        n_sig = int(
            n_sig
        )


        n_total = int(
            n_total
        )


        ax.text(
            j,
            i,
            f"{n_sig}/{n_total}",
            ha="center",
            va="center",
            fontsize=6.1
        )


# ============================================================
# COMMON AXIS FORMATTING
# ============================================================

axes[
    2
].set_xlabel(
    "Motif pair, ordered by increasing stickiness contrast"
)


for ax in axes:

    ax.set_xlim(
        -0.5,
        len(
            PAIR_ORDER
        )
        -
        0.5
    )


    ax.set_ylim(
        len(
            METRIC_ORDER
        )
        -
        0.5,
        -0.5
    )


# ============================================================
# SAVE HEATMAP
# ============================================================

PDF_OUT = (
    OUTDIR
    /
    "figure_DR2_supplement_16metric_heatmaps.pdf"
)


PNG_OUT = (
    OUTDIR
    /
    "figure_DR2_supplement_16metric_heatmaps.png"
)


SVG_OUT = (
    OUTDIR
    /
    "figure_DR2_supplement_16metric_heatmaps.svg"
)


fig.savefig(
    PDF_OUT,
    bbox_inches="tight"
)


fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    SVG_OUT,
    bbox_inches="tight"
)


plt.close(
    fig
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "MULTI-METRIC + DISTRIBUTION OVERLAP ANALYSIS COMPLETE"
)


print(
    "=" * 100
)


DISPLAY_COLUMNS = [

    "metric_label",

    "metric_source",

    "mean_signal_noise",

    "spearman_deltaLambda_vs_signal_noise",

    "spearman_deltaLambda_vs_omega2",

    "spearman_deltaLambda_vs_fraction_Tukey",

    "total_significant_Tukey",

    "n_global_FDR_significant",
]


print(
    "\nOBSERVABLE SENSITIVITY SUMMARY\n"
)


print(
    metric_summary_df[
        DISPLAY_COLUMNS
    ].to_string(
        index=False
    )
)


# ============================================================
# RANK ALL METRICS BY MEAN S/N
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "ALL OBSERVABLES RANKED BY MEAN ARCHITECTURE S/N"
)


print(
    "=" * 100
)


sn_rank = (

    metric_summary_df

    .sort_values(
        "mean_signal_noise",
        ascending=False
    )
)


print(
    sn_rank[
        [
            "metric_label",
            "metric_source",
            "mean_signal_noise",
            "median_signal_noise",
            "spearman_deltaLambda_vs_signal_noise",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# RANK BY TUKEY SIGNIFICANCE
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "OBSERVABLES RANKED BY TOTAL SIGNIFICANT TUKEY COMPARISONS"
)


print(
    "=" * 100
)


tukey_rank = (

    metric_summary_df

    .sort_values(
        "total_significant_Tukey",
        ascending=False
    )
)


print(
    tukey_rank[
        [
            "metric_label",
            "metric_source",
            "total_significant_Tukey",
            "total_Tukey_comparisons",
            "overall_percent_significant_Tukey",
            "n_global_FDR_significant",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# OVERLAP-ONLY SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "OVERLAP-DERIVED METRICS ONLY"
)


print(
    "=" * 100
)


overlap_only_summary = metric_summary_df[
    metric_summary_df[
        "metric_source"
    ]
    ==
    "overlap"
].copy()


print(
    overlap_only_summary[
        [
            "metric_label",
            "mean_signal_noise",
            "spearman_deltaLambda_vs_signal_noise",
            "mean_omega2",
            "total_significant_Tukey",
            "n_global_FDR_significant",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# OUTPUT LOCATIONS
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "OUTPUT FILES"
)


print(
    "=" * 100
)


OUTPUTS = [

    OVERLAP_SCORE_CSV,

    GLOBAL_CSV,

    TUKEY_ALL_CSV,

    TUKEY_SUMMARY_CSV,

    METRIC_RANKING_CSV,

    PDF_OUT,

    PNG_OUT,

    SVG_OUT,
]


if ORIGINAL_RATIO_OUT is not None:

    OUTPUTS.append(
        ORIGINAL_RATIO_OUT
    )


for path in OUTPUTS:

    print(
        path
    )


print(
    "\n"
    + "=" * 100
)


print(
    "IMPORTANT INTERPRETATION"
)


print(
    "=" * 100
)


print(
    "\nAll heatmap S/N values use:"
)


print(
    "    architecture spread / pooled within-architecture SD"
)


print(
    "\nTherefore the S/N values use the same mathematical "
    "definition for all 16 rows."
)


print(
    "\nFor overlap rows, each replicate-level observable is "
    "the mean distance to the 21 trajectories belonging "
    "to the other seven architectures."
)


print(
    "\nThe original between/within overlap-distance ratios "
    "are retained separately and are NOT mixed with the "
    "heatmap S/N values."
)


print(
    "\nOverlap-derived Welch/Tukey results should be treated "
    "as supplementary because trajectory scores share "
    "comparison partners."
)