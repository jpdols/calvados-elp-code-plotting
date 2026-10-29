# ============================================================
# DESIGN RULE 2 — Rg DISTRIBUTION AND INTERVAL OVERLAP
# ============================================================
#
# For each motif pair:
#
#     8 architectures × 3 replicates = 24 distributions
#
# Every replicate distribution is compared with every other
# replicate distribution:
#
#     C(24,2) = 276 unique comparisons per motif pair
#
# Across 9 motif pairs:
#
#     9 × 276 = 2484 comparisons
#
#
# THREE OVERLAP MEASURES
# ------------------------------------------------------------
#
# 1. FULL DISTRIBUTION OVERLAP
#
#       OVL = sum_i min(p_i, q_i)
#
#    where p and q are normalized histogram probability
#    masses on a common histogram grid.
#
#
# 2. CENTRAL 50% INTERVAL OVERLAP
#
#    Each replicate has:
#
#       I50 = [P25, P75]
#
#    Similarity is quantified using intersection-over-union:
#
#       IoU50 = |I50_A ∩ I50_B| / |I50_A ∪ I50_B|
#
#
# 3. CENTRAL 90% INTERVAL OVERLAP
#
#       I90 = [P05, P95]
#
#       IoU90 = |I90_A ∩ I90_B| / |I90_A ∪ I90_B|
#
#
# All overlap measures:
#
#       1.0 = identical / complete overlap
#       0.0 = no overlap
#
# Corresponding distances:
#
#       D = 1 - overlap
#
# Therefore:
#
#       larger D = stronger distributional separation
#
#
# COMPARISON TYPES
# ------------------------------------------------------------
#
# WITHIN architecture:
#     same architecture, different independent replicates
#
# BETWEEN architecture:
#     different sequence architectures
#
#
# EXPECTED COUNTS PER MOTIF PAIR
# ------------------------------------------------------------
#
# Within:
#
#     8 × C(3,2) = 24
#
# Between:
#
#     C(8,2) × 3 × 3 = 252
#
# Total:
#
#     24 + 252 = 276
#
#
# IMPORTANT
# ------------------------------------------------------------
#
# The frame-level Rg values are used to estimate each replicate
# distribution, but each replicate remains a distinct unit.
#
# We do NOT treat individual frames as independent replicates.
#
# ============================================================


from pathlib import Path
from itertools import combinations
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# INPUT / OUTPUT
# ============================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis"
    r"\design_rule_book"
)


INPUT_FILES = [

    ROOT
    / "907_930_3reps_stride1"
    / "all_rg_values.csv",

    ROOT
    / "lib1R4_triplicate_stride1"
    / "all_rg_values.csv",
]


OUTDIR = (
    ROOT
    / "design_rule_2_architecture_contrast"
    / "distribution_overlap"
)


OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TEMPERATURE = 370.0

# Common histogram resolution within each motif pair.
N_BINS = 150


# ============================================================
# MOTIF STICKINESS
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
            abs(
                lambda_a
                -
                lambda_b
            ),

        "mean_lambda":
            (
                lambda_a
                +
                lambda_b
            )
            /
            2.0,
    }


PAIR_ORDER = sorted(

    PAIR_INFO.keys(),

    key=lambda p:
        PAIR_META[
            p
        ][
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
# SHORT MOTIF CODES
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


# ============================================================
# LOAD FRAME-LEVEL Rg DATA
# ============================================================

frames = []


for path in INPUT_FILES:

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


data = pd.concat(
    frames,
    ignore_index=True,
    sort=False
)


print(
    "\n"
    + "=" * 80
)


print(
    "DATA LOADED"
)


print(
    "=" * 80
)


print(
    f"\nTotal Rg observations: "
    f"{len(data):,}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "id",

    "temp",

    "replicate",

    "architecture_type",

    "rg_nm",
]


missing_columns = [

    col

    for col in REQUIRED_COLUMNS

    if col not in data.columns
]


if missing_columns:

    raise KeyError(
        "\nMissing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ============================================================
# TEMPERATURE FILTER
# ============================================================

data[
    "temp_numeric"
] = pd.to_numeric(

    data[
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


data = data[

    np.isclose(

        data[
            "temp_numeric"
        ],

        TEMPERATURE
    )

].copy()


print(
    f"\nRg observations at "
    f"{TEMPERATURE:.0f} K: "
    f"{len(data):,}"
)


if data.empty:

    raise RuntimeError(
        "No data remain after temperature filtering."
    )


# ============================================================
# NUMERIC Rg
# ============================================================

data[
    "rg_nm"
] = pd.to_numeric(
    data[
        "rg_nm"
    ],
    errors="coerce"
)


data = data[
    np.isfinite(
        data[
            "rg_nm"
        ]
    )
].copy()


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


data[
    "pair"
] = data.apply(
    identify_pair,
    axis=1
)


print(
    "\nMotif-pair identification:"
)


print(
    data[
        "pair"
    ].value_counts(
        dropna=False
    )
)


data = data[
    data[
        "pair"
    ].isin(
        PAIR_INFO.keys()
    )
].copy()


# ============================================================
# HIGHER-LAMBDA MOTIF
# ============================================================

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


# ============================================================
# ARCHITECTURE CLASSIFICATION
# ============================================================

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
# CLASSIFY UNIQUE SIMULATIONS
# ============================================================

simulation_metadata = (

    data[
        [
            "id",
            "replicate",
            "pair",
            "architecture_type",
        ]
    ]

    .drop_duplicates()

    .copy()
)


simulation_metadata[
    "architecture_class"
] = [

    architecture_class(
        row
    )

    for _, row in simulation_metadata.iterrows()
]


# ============================================================
# CHECK CLASSIFICATION
# ============================================================

unknown = simulation_metadata[

    ~simulation_metadata[
        "architecture_class"
    ].isin(
        ARCH_ORDER
    )

]


if not unknown.empty:

    print(
        "\nWARNING — unrecognised architectures:"
    )


    print(
        unknown.to_string(
            index=False
        )
    )


# ============================================================
# MERGE CLASSIFICATION
# ============================================================

data = data.merge(

    simulation_metadata[
        [
            "id",
            "replicate",
            "pair",
            "architecture_class",
        ]
    ],

    on=[
        "id",
        "replicate",
        "pair",
    ],

    how="left"
)


data = data[
    data[
        "architecture_class"
    ].isin(
        ARCH_ORDER
    )
].copy()


# ============================================================
# DISTRIBUTION ID
# ============================================================

distribution_metadata = (

    data[
        [
            "pair",
            "architecture_class",
            "replicate",
            "id",
        ]
    ]

    .drop_duplicates()

    .copy()
)


distribution_metadata[
    "distribution_id"
] = (

    distribution_metadata[
        "pair"
    ].astype(
        str
    )

    +

    "__"

    +

    distribution_metadata[
        "architecture_class"
    ]
    .astype(
        str
    )
    .str.replace(
        r"[^A-Za-z0-9]+",
        "_",
        regex=True
    )
    .str.strip(
        "_"
    )

    +

    "__rep"

    +

    distribution_metadata[
        "replicate"
    ].astype(
        str
    )
)


data = data.merge(

    distribution_metadata[
        [
            "pair",
            "architecture_class",
            "replicate",
            "id",
            "distribution_id",
        ]
    ],

    on=[
        "pair",
        "architecture_class",
        "replicate",
        "id",
    ],

    how="left"
)


# ============================================================
# CALCULATE PERCENTILES FOR EVERY DISTRIBUTION
# ============================================================

percentile_rows = []


for distribution_id, sub in data.groupby(
    "distribution_id"
):

    rg = sub[
        "rg_nm"
    ].to_numpy(
        dtype=float
    )


    rg = rg[
        np.isfinite(
            rg
        )
    ]


    if len(
        rg
    ) == 0:

        continue


    percentile_rows.append(
        {

            "distribution_id":
                distribution_id,

            "p05_rg_nm":
                np.percentile(
                    rg,
                    5
                ),

            "p25_rg_nm":
                np.percentile(
                    rg,
                    25
                ),

            "p50_rg_nm":
                np.percentile(
                    rg,
                    50
                ),

            "p75_rg_nm":
                np.percentile(
                    rg,
                    75
                ),

            "p95_rg_nm":
                np.percentile(
                    rg,
                    95
                ),

            "iqr_rg_nm":
                (
                    np.percentile(
                        rg,
                        75
                    )
                    -
                    np.percentile(
                        rg,
                        25
                    )
                ),

            "rg_range_90_nm":
                (
                    np.percentile(
                        rg,
                        95
                    )
                    -
                    np.percentile(
                        rg,
                        5
                    )
                ),

            "n_rg_frames":
                len(
                    rg
                ),
        }
    )


percentile_df = pd.DataFrame(
    percentile_rows
)


distribution_metadata = (
    distribution_metadata.merge(

        percentile_df,

        on="distribution_id",

        how="left"
    )
)


# ============================================================
# ADD PAIR METADATA
# ============================================================

distribution_metadata[
    "delta_lambda"
] = distribution_metadata[
    "pair"
].map(

    lambda p:
        PAIR_META[
            p
        ][
            "delta_lambda"
        ]
)


distribution_metadata[
    "mean_lambda"
] = distribution_metadata[
    "pair"
].map(

    lambda p:
        PAIR_META[
            p
        ][
            "mean_lambda"
        ]
)


# ============================================================
# VALIDATION
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "DISTRIBUTION COUNTS"
)


print(
    "=" * 80
)


distribution_counts = (

    distribution_metadata
    .groupby(
        "pair"
    )[
        "distribution_id"
    ]
    .nunique()

    .reindex(
        PAIR_ORDER
    )
)


print(
    distribution_counts.to_string()
)


for pair, count in distribution_counts.items():

    if count != 24:

        warnings.warn(

            f"{pair}: found {count} distributions; "
            "expected 24."
        )


# ============================================================
# SAVE DISTRIBUTION METADATA
# ============================================================

METADATA_OUT = (

    OUTDIR
    /
    "DR2_Rg_distribution_metadata.csv"
)


distribution_metadata.to_csv(
    METADATA_OUT,
    index=False
)


# ============================================================
# FULL DISTRIBUTION OVERLAP
# ============================================================

def overlap_coefficient(
    probability_a,
    probability_b
):

    """
    Overlap coefficient between two discrete probability
    distributions on identical histogram bins.

    OVL = sum(min(p_i, q_i))
    """

    probability_a = np.asarray(
        probability_a,
        dtype=float
    )


    probability_b = np.asarray(
        probability_b,
        dtype=float
    )


    if (
        probability_a.shape
        !=
        probability_b.shape
    ):

        raise ValueError(
            "Probability arrays have different shapes."
        )


    overlap = np.sum(

        np.minimum(
            probability_a,
            probability_b
        )
    )


    return float(
        np.clip(
            overlap,
            0.0,
            1.0
        )
    )


# ============================================================
# INTERVAL INTERSECTION-OVER-UNION
# ============================================================

def interval_iou(
    lower_a,
    upper_a,
    lower_b,
    upper_b
):

    """
    Intersection-over-union for two 1D intervals.

    1 = identical intervals
    0 = no overlap
    """

    lower_a = float(
        lower_a
    )

    upper_a = float(
        upper_a
    )

    lower_b = float(
        lower_b
    )

    upper_b = float(
        upper_b
    )


    intersection = max(

        0.0,

        min(
            upper_a,
            upper_b
        )

        -

        max(
            lower_a,
            lower_b
        )
    )


    union = (

        max(
            upper_a,
            upper_b
        )

        -

        min(
            lower_a,
            lower_b
        )
    )


    if union <= 0:

        return np.nan


    return float(
        intersection
        /
        union
    )


# ============================================================
# ALL PAIRWISE COMPARISONS
# ============================================================

comparison_rows = []


print(
    "\n"
    + "=" * 80
)


print(
    "CALCULATING OVERLAPS"
)


print(
    "=" * 80
)


for pair in PAIR_ORDER:

    print(
        f"\nProcessing {pair} "
        f"(Δλ = "
        f"{PAIR_META[pair]['delta_lambda']:.3f})"
    )


    pair_data = data[
        data[
            "pair"
        ]
        ==
        pair
    ].copy()


    pair_metadata = distribution_metadata[
        distribution_metadata[
            "pair"
        ]
        ==
        pair
    ].copy()


    # --------------------------------------------------------
    # COMMON HISTOGRAM GRID
    # --------------------------------------------------------

    rg_min = pair_data[
        "rg_nm"
    ].min()


    rg_max = pair_data[
        "rg_nm"
    ].max()


    if (
        not np.isfinite(
            rg_min
        )
        or
        not np.isfinite(
            rg_max
        )
        or
        rg_max <= rg_min
    ):

        raise RuntimeError(
            f"Invalid Rg range for {pair}."
        )


    bin_edges = np.linspace(

        rg_min,

        rg_max,

        N_BINS
        +
        1
    )


    # --------------------------------------------------------
    # CALCULATE HISTOGRAM DISTRIBUTIONS ONCE
    # --------------------------------------------------------

    distributions = {}

    metadata_lookup = {}


    for _, meta in pair_metadata.iterrows():

        distribution_id = meta[
            "distribution_id"
        ]


        rg = pair_data.loc[

            pair_data[
                "distribution_id"
            ]
            ==
            distribution_id,

            "rg_nm"

        ].to_numpy(
            dtype=float
        )


        rg = rg[
            np.isfinite(
                rg
            )
        ]


        counts, _ = np.histogram(

            rg,

            bins=bin_edges
        )


        total = counts.sum()


        if total == 0:

            warnings.warn(
                f"Empty histogram: {distribution_id}"
            )

            continue


        probability = (

            counts.astype(
                float
            )

            /

            float(
                total
            )
        )


        distributions[
            distribution_id
        ] = probability


        metadata_lookup[
            distribution_id
        ] = meta.to_dict()


    distribution_ids = list(
        distributions.keys()
    )


    print(
        f"  distributions: "
        f"{len(distribution_ids)}"
    )


    print(
        f"  comparisons: "
        f"{len(distribution_ids) * (len(distribution_ids)-1) // 2}"
    )


    # --------------------------------------------------------
    # UNIQUE PAIRWISE COMPARISONS
    # --------------------------------------------------------

    for (
        distribution_a,
        distribution_b,
    ) in combinations(
        distribution_ids,
        2
    ):

        meta_a = metadata_lookup[
            distribution_a
        ]


        meta_b = metadata_lookup[
            distribution_b
        ]


        # ====================================================
        # 1. FULL DISTRIBUTION OVERLAP
        # ====================================================

        full_overlap = overlap_coefficient(

            distributions[
                distribution_a
            ],

            distributions[
                distribution_b
            ]
        )


        full_distance = (
            1.0
            -
            full_overlap
        )


        # ====================================================
        # 2. CENTRAL 50% INTERVAL OVERLAP
        # ====================================================

        iou_50 = interval_iou(

            meta_a[
                "p25_rg_nm"
            ],

            meta_a[
                "p75_rg_nm"
            ],

            meta_b[
                "p25_rg_nm"
            ],

            meta_b[
                "p75_rg_nm"
            ]
        )


        distance_50 = (
            1.0
            -
            iou_50
        )


        # ====================================================
        # 3. CENTRAL 90% INTERVAL OVERLAP
        # ====================================================

        iou_90 = interval_iou(

            meta_a[
                "p05_rg_nm"
            ],

            meta_a[
                "p95_rg_nm"
            ],

            meta_b[
                "p05_rg_nm"
            ],

            meta_b[
                "p95_rg_nm"
            ]
        )


        distance_90 = (
            1.0
            -
            iou_90
        )


        # ====================================================
        # COMPARISON TYPE
        # ====================================================

        same_architecture = (

            meta_a[
                "architecture_class"
            ]

            ==

            meta_b[
                "architecture_class"
            ]
        )


        if same_architecture:

            comparison_type = (
                "within_architecture"
            )

        else:

            comparison_type = (
                "between_architecture"
            )


        # ====================================================
        # SAVE
        # ====================================================

        comparison_rows.append(
            {

                # --------------------------------------------
                # Chemistry
                # --------------------------------------------

                "pair":
                    pair,

                "motif_a":
                    PAIR_META[
                        pair
                    ][
                        "motif_a"
                    ],

                "motif_b":
                    PAIR_META[
                        pair
                    ][
                        "motif_b"
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

                # --------------------------------------------
                # Distribution A
                # --------------------------------------------

                "distribution_A":
                    distribution_a,

                "sequence_id_A":
                    meta_a[
                        "id"
                    ],

                "architecture_A":
                    meta_a[
                        "architecture_class"
                    ],

                "replicate_A":
                    meta_a[
                        "replicate"
                    ],

                "p05_A":
                    meta_a[
                        "p05_rg_nm"
                    ],

                "p25_A":
                    meta_a[
                        "p25_rg_nm"
                    ],

                "p75_A":
                    meta_a[
                        "p75_rg_nm"
                    ],

                "p95_A":
                    meta_a[
                        "p95_rg_nm"
                    ],

                # --------------------------------------------
                # Distribution B
                # --------------------------------------------

                "distribution_B":
                    distribution_b,

                "sequence_id_B":
                    meta_b[
                        "id"
                    ],

                "architecture_B":
                    meta_b[
                        "architecture_class"
                    ],

                "replicate_B":
                    meta_b[
                        "replicate"
                    ],

                "p05_B":
                    meta_b[
                        "p05_rg_nm"
                    ],

                "p25_B":
                    meta_b[
                        "p25_rg_nm"
                    ],

                "p75_B":
                    meta_b[
                        "p75_rg_nm"
                    ],

                "p95_B":
                    meta_b[
                        "p95_rg_nm"
                    ],

                # --------------------------------------------
                # Classification
                # --------------------------------------------

                "same_architecture":
                    same_architecture,

                "comparison_type":
                    comparison_type,

                # --------------------------------------------
                # FULL DISTRIBUTION
                # --------------------------------------------

                "full_overlap_fraction":
                    full_overlap,

                "full_overlap_percent":
                    100.0
                    *
                    full_overlap,

                "full_distance":
                    full_distance,

                "full_distance_percent":
                    100.0
                    *
                    full_distance,

                # --------------------------------------------
                # CENTRAL 50%
                # --------------------------------------------

                "central50_interval_overlap_fraction":
                    iou_50,

                "central50_interval_overlap_percent":
                    100.0
                    *
                    iou_50,

                "central50_interval_distance":
                    distance_50,

                "central50_interval_distance_percent":
                    100.0
                    *
                    distance_50,

                # --------------------------------------------
                # CENTRAL 90%
                # --------------------------------------------

                "central90_interval_overlap_fraction":
                    iou_90,

                "central90_interval_overlap_percent":
                    100.0
                    *
                    iou_90,

                "central90_interval_distance":
                    distance_90,

                "central90_interval_distance_percent":
                    100.0
                    *
                    distance_90,

                # --------------------------------------------
                # Histogram settings
                # --------------------------------------------

                "n_bins":
                    N_BINS,

                "histogram_rg_min_nm":
                    rg_min,

                "histogram_rg_max_nm":
                    rg_max,
            }
        )


# ============================================================
# PAIRWISE DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)


PAIRWISE_OUT = (

    OUTDIR
    /
    "DR2_Rg_all_pairwise_overlap_metrics.csv"
)


comparison_df.to_csv(
    PAIRWISE_OUT,
    index=False
)


# ============================================================
# VALIDATE COMPARISON COUNTS
# ============================================================

print(
    "\n"
    + "=" * 80
)


print(
    "PAIRWISE COMPARISON COUNTS"
)


print(
    "=" * 80
)


comparison_counts = (

    comparison_df
    .groupby(
        [
            "pair",
            "comparison_type",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reindex(
        PAIR_ORDER
    )
)


print(
    comparison_counts.to_string()
)


print(
    "\nExpected per complete motif pair:"
)


print(
    "  within architecture  = 24"
)


print(
    "  between architecture = 252"
)


print(
    "  total                = 276"
)


# ============================================================
# ARCHITECTURE-PAIR SUMMARY
#
# Same architecture:
#     C(3,2) = 3 replicate comparisons
#
# Different architectures:
#     3 × 3 = 9 replicate comparisons
# ============================================================

architecture_pair_rows = []


for pair in PAIR_ORDER:

    pair_sub = comparison_df[
        comparison_df[
            "pair"
        ]
        ==
        pair
    ]


    # --------------------------------------------------------
    # WITHIN ARCHITECTURE
    # --------------------------------------------------------

    for arch in ARCH_ORDER:

        sub = pair_sub[

            (
                pair_sub[
                    "architecture_A"
                ]
                ==
                arch
            )

            &

            (
                pair_sub[
                    "architecture_B"
                ]
                ==
                arch
            )
        ]


        if len(
            sub
        ) == 0:

            continue


        architecture_pair_rows.append(
            {

                "pair":
                    pair,

                "delta_lambda":
                    PAIR_META[
                        pair
                    ][
                        "delta_lambda"
                    ],

                "architecture_A":
                    arch,

                "architecture_B":
                    arch,

                "comparison_type":
                    "within_architecture",

                "n_replicate_comparisons":
                    len(
                        sub
                    ),

                "mean_full_overlap_percent":
                    sub[
                        "full_overlap_percent"
                    ].mean(),

                "std_full_overlap_percent":
                    sub[
                        "full_overlap_percent"
                    ].std(
                        ddof=1
                    ),

                "mean_central50_overlap_percent":
                    sub[
                        "central50_interval_overlap_percent"
                    ].mean(),

                "std_central50_overlap_percent":
                    sub[
                        "central50_interval_overlap_percent"
                    ].std(
                        ddof=1
                    ),

                "mean_central90_overlap_percent":
                    sub[
                        "central90_interval_overlap_percent"
                    ].mean(),

                "std_central90_overlap_percent":
                    sub[
                        "central90_interval_overlap_percent"
                    ].std(
                        ddof=1
                    ),
            }
        )


    # --------------------------------------------------------
    # BETWEEN ARCHITECTURES
    # --------------------------------------------------------

    for (
        arch_a,
        arch_b,
    ) in combinations(
        ARCH_ORDER,
        2
    ):

        mask = (

            (
                (
                    pair_sub[
                        "architecture_A"
                    ]
                    ==
                    arch_a
                )

                &

                (
                    pair_sub[
                        "architecture_B"
                    ]
                    ==
                    arch_b
                )
            )

            |

            (
                (
                    pair_sub[
                        "architecture_A"
                    ]
                    ==
                    arch_b
                )

                &

                (
                    pair_sub[
                        "architecture_B"
                    ]
                    ==
                    arch_a
                )
            )
        )


        sub = pair_sub[
            mask
        ]


        if len(
            sub
        ) == 0:

            continue


        architecture_pair_rows.append(
            {

                "pair":
                    pair,

                "delta_lambda":
                    PAIR_META[
                        pair
                    ][
                        "delta_lambda"
                    ],

                "architecture_A":
                    arch_a,

                "architecture_B":
                    arch_b,

                "comparison_type":
                    "between_architecture",

                "n_replicate_comparisons":
                    len(
                        sub
                    ),

                "mean_full_overlap_percent":
                    sub[
                        "full_overlap_percent"
                    ].mean(),

                "std_full_overlap_percent":
                    sub[
                        "full_overlap_percent"
                    ].std(
                        ddof=1
                    ),

                "mean_central50_overlap_percent":
                    sub[
                        "central50_interval_overlap_percent"
                    ].mean(),

                "std_central50_overlap_percent":
                    sub[
                        "central50_interval_overlap_percent"
                    ].std(
                        ddof=1
                    ),

                "mean_central90_overlap_percent":
                    sub[
                        "central90_interval_overlap_percent"
                    ].mean(),

                "std_central90_overlap_percent":
                    sub[
                        "central90_interval_overlap_percent"
                    ].std(
                        ddof=1
                    ),
            }
        )


architecture_pair_df = pd.DataFrame(
    architecture_pair_rows
)


ARCH_PAIR_OUT = (

    OUTDIR
    /
    "DR2_Rg_architecture_pair_overlap_summary.csv"
)


architecture_pair_df.to_csv(
    ARCH_PAIR_OUT,
    index=False
)


# ============================================================
# WITHIN VS BETWEEN SUMMARY
# ============================================================

summary_rows = []


OVERLAP_METRICS = {

    "full": {
        "overlap":
            "full_overlap_fraction",

        "distance":
            "full_distance",
    },

    "central50": {
        "overlap":
            "central50_interval_overlap_fraction",

        "distance":
            "central50_interval_distance",
    },

    "central90": {
        "overlap":
            "central90_interval_overlap_fraction",

        "distance":
            "central90_interval_distance",
    },
}


for pair in PAIR_ORDER:

    sub = comparison_df[
        comparison_df[
            "pair"
        ]
        ==
        pair
    ]


    within = sub[
        sub[
            "comparison_type"
        ]
        ==
        "within_architecture"
    ]


    between = sub[
        sub[
            "comparison_type"
        ]
        ==
        "between_architecture"
    ]


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

        "n_within_comparisons":
            len(
                within
            ),

        "n_between_comparisons":
            len(
                between
            ),
    }


    for metric_name, columns in OVERLAP_METRICS.items():

        overlap_col = columns[
            "overlap"
        ]

        distance_col = columns[
            "distance"
        ]


        mean_within_overlap = (
            within[
                overlap_col
            ].mean()
        )


        mean_between_overlap = (
            between[
                overlap_col
            ].mean()
        )


        mean_within_distance = (
            within[
                distance_col
            ].mean()
        )


        mean_between_distance = (
            between[
                distance_col
            ].mean()
        )


        row[
            f"{metric_name}_mean_within_overlap_percent"
        ] = (
            100.0
            *
            mean_within_overlap
        )


        row[
            f"{metric_name}_mean_between_overlap_percent"
        ] = (
            100.0
            *
            mean_between_overlap
        )


        row[
            f"{metric_name}_mean_within_distance"
        ] = (
            mean_within_distance
        )


        row[
            f"{metric_name}_mean_between_distance"
        ] = (
            mean_between_distance
        )


        # Distributional architecture signal/noise:
        #
        # between-architecture distance
        # -----------------------------
        # within-architecture distance

        if (
            np.isfinite(
                mean_within_distance
            )
            and
            mean_within_distance > 0
        ):

            ratio = (

                mean_between_distance

                /

                mean_within_distance
            )

        else:

            ratio = np.nan


        row[
            f"{metric_name}_between_within_distance_ratio"
        ] = ratio


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


SUMMARY_OUT = (

    OUTDIR
    /
    "DR2_Rg_within_between_overlap_summary.csv"
)


summary_df.to_csv(
    SUMMARY_OUT,
    index=False
)


# ============================================================
# CORRELATION WITH MOTIF STICKINESS CONTRAST
# ============================================================

correlation_rows = []


for metric_name in OVERLAP_METRICS.keys():

    y_column = (
        f"{metric_name}_between_within_distance_ratio"
    )


    valid = summary_df[
        [
            "delta_lambda",
            y_column,
        ]
    ].dropna()


    if len(
        valid
    ) >= 3:

        rho, p = stats.spearmanr(

            valid[
                "delta_lambda"
            ],

            valid[
                y_column
            ]
        )

    else:

        rho = np.nan

        p = np.nan


    correlation_rows.append(
        {

            "overlap_metric":
                metric_name,

            "n_motif_pairs":
                len(
                    valid
                ),

            "spearman_rho_deltaLambda_vs_betweenWithinRatio":
                rho,

            "spearman_p":
                p,

            "mean_between_within_distance_ratio":
                valid[
                    y_column
                ].mean(),
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


CORRELATION_OUT = (

    OUTDIR
    /
    "DR2_Rg_overlap_metric_correlations.csv"
)


correlation_df.to_csv(
    CORRELATION_OUT,
    index=False
)


# ============================================================
# INDIVIDUAL REPLICATE SUMMARY
#
# Each distribution should have:
#
#     23 comparisons total
#      2 same-architecture comparisons
#     21 other-architecture comparisons
# ============================================================

individual_rows = []


for pair in PAIR_ORDER:

    pair_metadata = distribution_metadata[
        distribution_metadata[
            "pair"
        ]
        ==
        pair
    ]


    pair_comparisons = comparison_df[
        comparison_df[
            "pair"
        ]
        ==
        pair
    ]


    for _, meta in pair_metadata.iterrows():

        distribution_id = meta[
            "distribution_id"
        ]


        sub = pair_comparisons[

            (
                pair_comparisons[
                    "distribution_A"
                ]
                ==
                distribution_id
            )

            |

            (
                pair_comparisons[
                    "distribution_B"
                ]
                ==
                distribution_id
            )
        ]


        within = sub[
            sub[
                "comparison_type"
            ]
            ==
            "within_architecture"
        ]


        between = sub[
            sub[
                "comparison_type"
            ]
            ==
            "between_architecture"
        ]


        individual_rows.append(
            {

                "pair":
                    pair,

                "delta_lambda":
                    PAIR_META[
                        pair
                    ][
                        "delta_lambda"
                    ],

                "distribution_id":
                    distribution_id,

                "sequence_id":
                    meta[
                        "id"
                    ],

                "architecture":
                    meta[
                        "architecture_class"
                    ],

                "replicate":
                    meta[
                        "replicate"
                    ],

                "n_total_comparisons":
                    len(
                        sub
                    ),

                "n_within_architecture_comparisons":
                    len(
                        within
                    ),

                "n_between_architecture_comparisons":
                    len(
                        between
                    ),

                # --------------------------------------------
                # FULL
                # --------------------------------------------

                "mean_full_overlap_all_percent":
                    sub[
                        "full_overlap_percent"
                    ].mean(),

                "mean_full_overlap_within_percent":
                    within[
                        "full_overlap_percent"
                    ].mean(),

                "mean_full_overlap_between_percent":
                    between[
                        "full_overlap_percent"
                    ].mean(),

                # --------------------------------------------
                # CENTRAL 50
                # --------------------------------------------

                "mean_central50_overlap_all_percent":
                    sub[
                        "central50_interval_overlap_percent"
                    ].mean(),

                "mean_central50_overlap_within_percent":
                    within[
                        "central50_interval_overlap_percent"
                    ].mean(),

                "mean_central50_overlap_between_percent":
                    between[
                        "central50_interval_overlap_percent"
                    ].mean(),

                # --------------------------------------------
                # CENTRAL 90
                # --------------------------------------------

                "mean_central90_overlap_all_percent":
                    sub[
                        "central90_interval_overlap_percent"
                    ].mean(),

                "mean_central90_overlap_within_percent":
                    within[
                        "central90_interval_overlap_percent"
                    ].mean(),

                "mean_central90_overlap_between_percent":
                    between[
                        "central90_interval_overlap_percent"
                    ].mean(),
            }
        )


individual_df = pd.DataFrame(
    individual_rows
)


INDIVIDUAL_OUT = (

    OUTDIR
    /
    "DR2_Rg_individual_replicate_overlap_summary.csv"
)


individual_df.to_csv(
    INDIVIDUAL_OUT,
    index=False
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)


print(
    "Rg OVERLAP ANALYSIS COMPLETE"
)


print(
    "=" * 100
)


print(
    "\nWITHIN VS BETWEEN ARCHITECTURE SUMMARY\n"
)


DISPLAY_COLUMNS = [

    "pair",

    "delta_lambda",

    "full_mean_within_overlap_percent",

    "full_mean_between_overlap_percent",

    "full_between_within_distance_ratio",

    "central50_between_within_distance_ratio",

    "central90_between_within_distance_ratio",
]


print(
    summary_df[
        DISPLAY_COLUMNS
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


print(
    "\n"
    + "=" * 100
)


print(
    "CORRELATION WITH MOTIF STICKINESS CONTRAST"
)


print(
    "=" * 100
)


print(
    correlation_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# OUTPUT FILES
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


for path in [

    PAIRWISE_OUT,

    ARCH_PAIR_OUT,

    SUMMARY_OUT,

    CORRELATION_OUT,

    INDIVIDUAL_OUT,

    METADATA_OUT,

]:

    print(
        path
    )


print(
    "\nExpected complete design:"
)


print(
    "  24 distributions per motif pair"
)


print(
    "  23 comparisons involving each individual replicate"
)


print(
    "  24 within-architecture comparisons per motif pair"
)


print(
    "  252 between-architecture comparisons per motif pair"
)


print(
    "  276 unique comparisons per motif pair"
)


print(
    "  2484 unique comparisons across 9 motif pairs"
)


print(
    "\nOverlap interpretation:"
)


print(
    "  100% = complete overlap"
)


print(
    "    0% = no overlap"
)


print(
    "\nDistance interpretation:"
)


print(
    "  distance = 1 - overlap"
)


print(
    "  larger distance = stronger separation"
)


print(
    "\nCentral50:"
)


print(
    "  overlap of P25-P75 intervals using intersection-over-union"
)


print(
    "\nCentral90:"
)


print(
    "  overlap of P05-P95 intervals using intersection-over-union"
)