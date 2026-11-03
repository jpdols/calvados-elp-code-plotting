from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis\design_rule_book"
)

PER_RUN_FILES = [
    ROOT / "907_930_3reps_stride1" / "per_run_summary.csv",
    ROOT / "931_954_triplicate_stride1" / "per_run_summary.csv",
    ROOT / "lib1R4_triplicate_stride1" / "per_run_summary.csv",
]

OUTDIR = (
    ROOT
    / "design_rule_2_architecture_contrast"
    / "supplement_conformational_regime"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TEMPERATURE = 370

# Choose:
#
# "rg_range_90_nm"
#     central 90% Rg range
#
# "iqr_rg_nm"
#     central 50% Rg range / IQR
#
METRIC = "rg_range_90_nm"


if METRIC == "rg_range_90_nm":

    YLABEL = (
        r"Central 90% $R_g$ range, "
        r"$\Delta R_g^{90}$ (nm)"
    )

    FILETAG = "central90"

elif METRIC == "iqr_rg_nm":

    YLABEL = (
        r"Central 50% $R_g$ range, "
        r"IQR$_{R_g}$ (nm)"
    )

    FILETAG = "IQR"

else:

    YLABEL = METRIC
    FILETAG = METRIC


# ============================================================
# MOTIF STICKINESS
# ============================================================

LAMBDA = {

    "VPAVG":
        0.6294581781012,

    "VPGNG":
        0.6527600406134,

    "VPGAG":
        0.6568240708464,

    "VPGVG":
        0.6850471057402,

    "VPGGG":
        0.7124129984854,

    "IPAVG":
        0.7430614622074,

    "VPGIG":
        0.7986503898464,

    "VPGLG":
        0.8147436293176,

    "VPGFG":
        0.8506676135856,
}


# ============================================================
# PAIR DEFINITIONS
# ============================================================

PAIR_INFO = {

    "IGLG":
        ("VPGIG", "VPGLG"),

    "AVGAG":
        ("VPAVG", "VPGAG"),

    "IGIP":
        ("VPGIG", "IPAVG"),

    "VGAVG":
        ("VPGVG", "VPAVG"),

    "VGIP":
        ("VPGVG", "IPAVG"),

    "GGNG":
        ("VPGGG", "VPGNG"),

    "LGIP":
        ("VPGLG", "IPAVG"),

    "GGAVG":
        ("VPGGG", "VPAVG"),

    "VGIG":
        ("VPGVG", "VPGIG"),

    "VAIP":
        ("VPAVG", "IPAVG"),

    "VAIG":
        ("VPAVG", "VPGIG"),

    "AVGFG":
        ("VPAVG", "VPGFG"),
}


PAIR_DISPLAY = {

    "IGLG":
        "VPGIG–VPGLG",

    "AVGAG":
        "VPAVG–VPGAG",

    "IGIP":
        "VPGIG–IPAVG",

    "VGAVG":
        "VPGVG–VPAVG",

    "VGIP":
        "VPGVG–IPAVG",

    "GGNG":
        "VPGGG–VPGNG",

    "LGIP":
        "VPGLG–IPAVG",

    "GGAVG":
        "VPGGG–VPAVG",

    "VGIG":
        "VPGVG–VPGIG",

    "VAIP":
        "VPAVG–IPAVG",

    "VAIG":
        "VPAVG–VPGIG",

    "AVGFG":
        "VPAVG–VPGFG",
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

        "mean_lambda":
            (
                lambda_a
                + lambda_b
            )
            / 2,

        "delta_lambda":
            abs(
                lambda_a
                - lambda_b
            ),
    }


# ============================================================
# ID -> CHEMISTRY
#
# Original DR2 libraries + new intermediate-contrast systems.
# ============================================================

#
# New systems:
#
# 907–914  AVGFG
# 915–922  AVGAG
# 923–930  IGLG
#
# 931–938  LGIP
# 939–946  GGAVG
# 947–954  GGNG
#
# The remaining original R4 systems are normally detected
# from the "group" / naming information in the CSV.
#

ID_PAIR_RANGES = [

    (907, 914, "AVGFG"),
    (915, 922, "AVGAG"),
    (923, 930, "IGLG"),

    (931, 938, "LGIP"),
    (939, 946, "GGAVG"),
    (947, 954, "GGNG"),
]


# ============================================================
# ARCHITECTURES
# ============================================================

ARCH_ORDER = [

    "Random shuffle",

    "Alternating",

    "Diblock",

    "Small blocks",

    "Medium blocks",

    "Large blocks",

    r"Higher $\lambda$ center",

    r"Higher $\lambda$ ends",
]


# ============================================================
# ARCHITECTURE COLORS
#
# Matches the general palette from the existing architecture
# figure.
# ============================================================

ARCH_COLORS = {

    "Random shuffle":
        "#1f77b4",

    "Alternating":
        "#2ca02c",

    "Diblock":
        "#17becf",

    "Small blocks":
        "#ff7f0e",

    "Medium blocks":
        "#9467bd",

    "Large blocks":
        "#8c564b",

    r"Higher $\lambda$ center":
        "#d62728",

    r"Higher $\lambda$ ends":
        "#e377c2",
}


# ============================================================
# SMALL X OFFSETS FOR ARCHITECTURE POINTS
# ============================================================

ARCH_OFFSETS = {

    "Random shuffle":
        -0.175,

    "Alternating":
        -0.125,

    "Diblock":
        -0.075,

    "Small blocks":
        -0.025,

    "Medium blocks":
        0.025,

    "Large blocks":
        0.075,

    r"Higher $\lambda$ center":
        0.125,

    r"Higher $\lambda$ ends":
        0.175,
}


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({

    "font.size":
        9,

    "axes.labelsize":
        10,

    "xtick.labelsize":
        8,

    "ytick.labelsize":
        8,

    "legend.fontsize":
        8,

    "axes.linewidth":
        0.8,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ============================================================
# HELPERS
# ============================================================

def extract_numeric_id(value):

    """
    Extract the first three-digit sequence ID from strings such
    as:

        907_AVGFG_R4_alternating
        931_LGIP_R4_even

    Returns None if no suitable ID is found.
    """

    if pd.isna(
        value
    ):
        return None

    match = re.search(
        r"(?<!\d)(\d{3})(?!\d)",
        str(value)
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def pair_from_id(
    seq_id
):

    if seq_id is None:
        return None

    for (
        start,
        end,
        pair,
    ) in ID_PAIR_RANGES:

        if (
            start
            <= seq_id
            <= end
        ):

            return pair

    return None


def detect_pair_from_text(
    value
):

    if pd.isna(
        value
    ):
        return None

    text = (
        str(value)
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    # Explicit pair codes first.
    for pair in sorted(
        PAIR_INFO,
        key=len,
        reverse=True
    ):

        if pair.upper() in text:
            return pair

    # Also try the full motif names.
    for pair, (
        motif_a,
        motif_b,
    ) in PAIR_INFO.items():

        motif_a_clean = (
            motif_a
            .upper()
            .replace("-", "")
            .replace("_", "")
        )

        motif_b_clean = (
            motif_b
            .upper()
            .replace("-", "")
            .replace("_", "")
        )

        if (
            motif_a_clean in text
            and motif_b_clean in text
        ):

            return pair

    return None


def detect_pair_row(
    row
):

    """
    Robust pair detection.

    Priority:
    1. Explicit group/pair/name columns.
    2. ID range for new systems.
    3. Search all useful text columns.
    """

    # --------------------------------------------------------
    # First try likely text columns
    # --------------------------------------------------------

    preferred_columns = [

        "group",

        "pair",

        "motif_pair",

        "architecture_type",

        "name",

        "sequence",

        "design",

        "system",
    ]

    for column in preferred_columns:

        if column not in row.index:
            continue

        pair = detect_pair_from_text(
            row[column]
        )

        if pair is not None:
            return pair

    # --------------------------------------------------------
    # Try sequence ID
    # --------------------------------------------------------

    id_columns = [

        "id",

        "name",

        "system",

        "sequence",
    ]

    for column in id_columns:

        if column not in row.index:
            continue

        seq_id = extract_numeric_id(
            row[column]
        )

        pair = pair_from_id(
            seq_id
        )

        if pair is not None:
            return pair

    # --------------------------------------------------------
    # Last resort: concatenate row text
    # --------------------------------------------------------

    combined = " ".join(
        str(value)
        for value in row.values
    )

    return detect_pair_from_text(
        combined
    )


# ============================================================
# MOTIF CODE MAPPING
# ============================================================

CODE_TO_MOTIF = {

    "VG":
        "VPGVG",

    "IG":
        "VPGIG",

    "LG":
        "VPGLG",

    "FG":
        "VPGFG",

    "GG":
        "VPGGG",

    "NG":
        "VPGNG",

    "AVG":
        "VPAVG",

    "VA":
        "VPAVG",

    "AG":
        "VPGAG",

    "IP":
        "IPAVG",
}


def clean_architecture_text(
    value
):

    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def motif_code_from_directional_name(
    raw,
    keyword
):

    """
    Extract motif code before 'center' or 'endcaps'.

    Example:

        IGcenter_LGends -> IG
        AVGcenter_FGends -> AVG
    """

    split = re.split(
        keyword,
        str(raw),
        flags=re.IGNORECASE
    )

    if len(split) < 2:
        return None

    prefix = clean_architecture_text(
        split[0]
    ).upper()

    # Sometimes the architecture string can contain a prefix.
    # Test longest motif codes first.
    possible_codes = sorted(
        CODE_TO_MOTIF.keys(),
        key=len,
        reverse=True
    )

    for code in possible_codes:

        if prefix.endswith(
            code
        ):
            return code

    return None


def architecture_class(
    row
):

    raw = str(
        row.get(
            "architecture_type",
            ""
        )
    )

    text = (
        raw
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )

    # --------------------------------------------------------
    # Standard architectures
    # --------------------------------------------------------

    if (
        "randomshuffle"
        in text
        or text == "shuffle"
        or text == "random"
    ):

        return "Random shuffle"

    if (
        "alternating"
        in text
        or text == "even"
    ):

        return "Alternating"

    if (
        "smallblocks"
        in text
        or "smallblock"
        in text
    ):

        return "Small blocks"

    if (
        "mediumblocks"
        in text
        or "mediumblock"
        in text
    ):

        return "Medium blocks"

    if (
        "largeblocks"
        in text
        or "largeblock"
        in text
    ):

        return "Large blocks"

    if "diblock" in text:

        return "Diblock"

    # --------------------------------------------------------
    # Directional architecture
    # --------------------------------------------------------

    pair = row.get(
        "pair",
        None
    )

    if pair not in PAIR_INFO:
        return None

    motif_a, motif_b = PAIR_INFO[
        pair
    ]

    if (
        LAMBDA[motif_a]
        >
        LAMBDA[motif_b]
    ):

        high_motif = motif_a

    else:

        high_motif = motif_b

    # --------------------------------------------------------
    # Xcenter_Yends
    # --------------------------------------------------------

    if "center" in text:

        center_code = (
            motif_code_from_directional_name(
                raw,
                "center"
            )
        )

        if center_code is None:

            warnings.warn(
                "Could not identify center motif "
                f"from architecture '{raw}'"
            )

            return None

        center_motif = CODE_TO_MOTIF.get(
            center_code
        )

        if center_motif == high_motif:

            return (
                r"Higher $\lambda$ center"
            )

        return (
            r"Higher $\lambda$ ends"
        )

    # --------------------------------------------------------
    # Xendcaps
    # --------------------------------------------------------

    if "endcaps" in text:

        end_code = (
            motif_code_from_directional_name(
                raw,
                "endcaps"
            )
        )

        if end_code is None:

            warnings.warn(
                "Could not identify end motif "
                f"from architecture '{raw}'"
            )

            return None

        end_motif = CODE_TO_MOTIF.get(
            end_code
        )

        if end_motif == high_motif:

            return (
                r"Higher $\lambda$ ends"
            )

        return (
            r"Higher $\lambda$ center"
        )

    return None


# ============================================================
# LOAD ALL INPUT FILES
# ============================================================

frames = []


print("\n")
print("=" * 100)
print("LOADING INPUT FILES")
print("=" * 100)


for path in PER_RUN_FILES:

    print(
        f"\nReading:\n{path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{path}"
        )

    temp = pd.read_csv(
        path
    )

    print(
        f"Rows: {len(temp)}"
    )

    print(
        "Columns:"
    )

    print(
        ", ".join(
            temp.columns
        )
    )

    temp[
        "source_file"
    ] = str(
        path
    )

    frames.append(
        temp
    )


df = pd.concat(
    frames,
    ignore_index=True
)


print(
    f"\nTotal rows loaded: {len(df)}"
)


# ============================================================
# TEMPERATURE FILTER
#
# Robust to:
#
# 370
# 370.0
# "370"
# "370K"
# "370 K"
# ============================================================

print("\n")
print("=" * 100)
print("TEMPERATURE FILTER")
print("=" * 100)


if "temp" in df.columns:

    print(
        "\nTemperature values found:"
    )

    print(
        df[
            "temp"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    temp_string = (
        df[
            "temp"
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    # Extract the first numerical temperature value.
    temp_extracted = pd.to_numeric(
        temp_string.str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )

    df = df[
        np.isclose(
            temp_extracted,
            TEMPERATURE,
            atol=0.01,
            equal_nan=False
        )
    ].copy()

else:

    print(
        "\nNo 'temp' column found."
    )

    print(
        "Assuming input files already contain only "
        f"{TEMPERATURE} K data."
    )


print(
    f"\nRows after {TEMPERATURE} K filtering: "
    f"{len(df)}"
)


if len(
    df
) == 0:

    raise RuntimeError(
        "\nNo rows remain after temperature filtering."
    )


# ============================================================
# DETECT CHEMISTRY PAIR
# ============================================================

print("\n")
print("=" * 100)
print("PAIR DETECTION")
print("=" * 100)


df[
    "pair"
] = df.apply(
    detect_pair_row,
    axis=1
)


print(
    "\nDetected pair counts:"
)


print(
    df[
        "pair"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


unmatched_pairs = df[
    df[
        "pair"
    ].isna()
]


if len(
    unmatched_pairs
) > 0:

    print(
        "\nWARNING:"
        f" {len(unmatched_pairs)} rows "
        "could not be assigned to a DR2 chemistry."
    )

    useful = [
        column
        for column in [
            "id",
            "group",
            "architecture_type",
            "name",
        ]
        if column in unmatched_pairs.columns
    ]

    if useful:

        print(
            unmatched_pairs[
                useful
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )


df = df[
    df[
        "pair"
    ].isin(
        PAIR_INFO
    )
].copy()


print(
    f"\nRows after pair filtering: {len(df)}"
)


print(
    f"Chemistries detected: "
    f"{df['pair'].nunique()}"
)


if len(
    df
) == 0:

    raise RuntimeError(
        "\nNo rows remain after pair detection."
    )


# ============================================================
# ARCHITECTURE CLASSIFICATION
# ============================================================

print("\n")
print("=" * 100)
print("ARCHITECTURE CLASSIFICATION")
print("=" * 100)


if (
    "architecture_type"
    not in df.columns
):

    raise KeyError(
        "\nColumn 'architecture_type' was not found."
    )


df[
    "architecture"
] = df.apply(
    architecture_class,
    axis=1
)


print(
    "\nArchitecture counts before filtering:"
)


print(
    df[
        "architecture"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


unknown = df[
    df[
        "architecture"
    ].isna()
]


if len(
    unknown
) > 0:

    print(
        "\nWARNING: unclassified architecture types:"
    )

    print(
        unknown[
            [
                "pair",
                "architecture_type",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )


df = df[
    df[
        "architecture"
    ].isin(
        ARCH_ORDER
    )
].copy()


print(
    f"\nRows after architecture filtering: "
    f"{len(df)}"
)


if len(
    df
) == 0:

    raise RuntimeError(
        "\nNo rows remain after architecture "
        "classification.\n"
        "Inspect the architecture names printed above."
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

print("\n")
print("=" * 100)
print("COLUMN VALIDATION")
print("=" * 100)


required_columns = [

    METRIC,

    "fraction_collapsed",
]


for column in required_columns:

    if column not in df.columns:

        raise KeyError(
            f"\nRequired column '{column}' "
            "not found.\n\nAvailable columns:\n"
            + "\n".join(
                df.columns
            )
        )

    df[
        column
    ] = pd.to_numeric(
        df[
            column
        ],
        errors="coerce"
    )


before_dropna = len(
    df
)


df = df.dropna(
    subset=[
        METRIC,
        "fraction_collapsed",
    ]
).copy()


print(
    f"\nRows before NaN filtering: "
    f"{before_dropna}"
)


print(
    f"Rows after NaN filtering: "
    f"{len(df)}"
)


if len(
    df
) == 0:

    raise RuntimeError(
        "\nAll rows were lost after numeric conversion."
    )


# ============================================================
# ARCHITECTURE-LEVEL SUMMARY
#
# Each point:
# mean of independent replicate values
#
# Error bar:
# SD across independent replicates
# ============================================================

arch_summary = (

    df

    .groupby(
        [
            "pair",
            "architecture",
        ],
        observed=True
    )

    .agg(

        metric_mean=(
            METRIC,
            "mean"
        ),

        metric_sd=(
            METRIC,
            "std"
        ),

        fraction_collapsed_mean=(
            "fraction_collapsed",
            "mean"
        ),

        fraction_collapsed_sd=(
            "fraction_collapsed",
            "std"
        ),

        n_replicates=(
            METRIC,
            "count"
        ),
    )

    .reset_index()
)


# ============================================================
# VALIDATE ARCHITECTURES / REPLICATES
# ============================================================

print("\n")
print("=" * 100)
print("REPLICATES PER ARCHITECTURE")
print("=" * 100)


print(
    arch_summary[
        [
            "pair",
            "architecture",
            "n_replicates",
        ]
    ]
    .sort_values(
        [
            "pair",
            "architecture",
        ]
    )
    .to_string(
        index=False
    )
)


if len(
    arch_summary
) == 0:

    raise RuntimeError(
        "\nArchitecture summary is empty."
    )


# ============================================================
# CHEMISTRY-LEVEL SUMMARY
#
# IMPORTANT:
#
# Compact-state population is first averaged across replicates
# within each architecture and then equally across the eight
# architectures.
#
# Thus each architecture contributes equally.
# ============================================================

chemistry_summary = (

    arch_summary

    .groupby(
        "pair",
        observed=True
    )

    .agg(

        mean_fraction_collapsed=(
            "fraction_collapsed_mean",
            "mean"
        ),

        min_fraction_collapsed=(
            "fraction_collapsed_mean",
            "min"
        ),

        max_fraction_collapsed=(
            "fraction_collapsed_mean",
            "max"
        ),

        n_architectures=(
            "architecture",
            "nunique"
        ),
    )

    .reset_index()
)


chemistry_summary[
    "mean_lambda"
] = chemistry_summary[
    "pair"
].map(
    lambda pair:
        PAIR_META[
            pair
        ][
            "mean_lambda"
        ]
)


chemistry_summary[
    "delta_lambda"
] = chemistry_summary[
    "pair"
].map(
    lambda pair:
        PAIR_META[
            pair
        ][
            "delta_lambda"
        ]
)


chemistry_summary[
    "motif_pair"
] = chemistry_summary[
    "pair"
].map(
    PAIR_DISPLAY
)


# ============================================================
# VALIDATE NUMBER OF ARCHITECTURES
# ============================================================

print("\n")
print("=" * 100)
print("CHEMISTRY VALIDATION")
print("=" * 100)


print(
    chemistry_summary[
        [
            "pair",
            "n_architectures",
        ]
    ]
    .sort_values(
        "pair"
    )
    .to_string(
        index=False
    )
)


incomplete = chemistry_summary[
    chemistry_summary[
        "n_architectures"
    ] != 8
]


if len(
    incomplete
) > 0:

    print(
        "\nWARNING:"
        " The following chemistries do not contain "
        "exactly 8 classified architectures:"
    )

    print(
        incomplete[
            [
                "pair",
                "n_architectures",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# SORT BY OBSERVED COMPACT-STATE POPULATION
# ============================================================

chemistry_summary = (

    chemistry_summary

    .sort_values(
        "mean_fraction_collapsed"
    )

    .reset_index(
        drop=True
    )
)


PAIR_ORDER = chemistry_summary[
    "pair"
].tolist()


PAIR_TO_X = {

    pair:
        index

    for index, pair
    in enumerate(
        PAIR_ORDER
    )
}


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print(
    "CHEMISTRY SUMMARY — ORDERED BY "
    "MEAN COMPACT-STATE POPULATION"
)
print("=" * 110)


print(
    chemistry_summary[
        [
            "pair",
            "motif_pair",
            "mean_lambda",
            "delta_lambda",
            "mean_fraction_collapsed",
            "min_fraction_collapsed",
            "max_fraction_collapsed",
            "n_architectures",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# SAVE TABLES
# ============================================================

chemistry_summary.to_csv(

    OUTDIR
    / "DR2_supplement_chemistry_conformational_regime.csv",

    index=False
)


arch_summary.to_csv(

    OUTDIR
    / "DR2_supplement_architecture_summary.csv",

    index=False
)


# ============================================================
# FINAL SAFETY CHECK BEFORE PLOTTING
# ============================================================

if len(
    PAIR_ORDER
) == 0:

    raise RuntimeError(
        "\nNo chemistries available for plotting."
    )


# ============================================================
# PREPARE VIOLIN DATA
#
# IMPORTANT:
#
# Each violin contains the architecture-level means.
#
# It does NOT contain individual trajectory frames.
# ============================================================

violin_data = []

valid_pair_order = []


for pair in PAIR_ORDER:

    values = (

        arch_summary.loc[
            arch_summary[
                "pair"
            ] == pair,
            "metric_mean"
        ]

        .dropna()

        .to_numpy(
            dtype=float
        )
    )

    if len(
        values
    ) == 0:

        warnings.warn(
            f"No metric values for {pair}; "
            "skipping this chemistry."
        )

        continue

    violin_data.append(
        values
    )

    valid_pair_order.append(
        pair
    )


PAIR_ORDER = valid_pair_order


PAIR_TO_X = {

    pair:
        index

    for index, pair
    in enumerate(
        PAIR_ORDER
    )
}


if len(
    violin_data
) == 0:

    raise RuntimeError(
        "\nNo non-empty datasets are available "
        "for the violin plot."
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        14.5,
        6.8
    )
)


x_positions = np.arange(
    len(
        PAIR_ORDER
    )
)


# ============================================================
# VIOLINS
# ============================================================

violins = ax.violinplot(

    violin_data,

    positions=x_positions,

    widths=0.72,

    showmeans=False,

    showmedians=False,

    showextrema=False,

    bw_method=0.35,
)


for body in violins[
    "bodies"
]:

    body.set_facecolor(
        "0.82"
    )

    body.set_edgecolor(
        "0.55"
    )

    body.set_alpha(
        0.32
    )

    body.set_linewidth(
        0.7
    )


# ============================================================
# ARCHITECTURE MEANS ± REPLICATE SD
# ============================================================

for pair in PAIR_ORDER:

    x_center = PAIR_TO_X[
        pair
    ]

    pair_data = arch_summary[
        arch_summary[
            "pair"
        ] == pair
    ]

    for architecture in ARCH_ORDER:

        row = pair_data[
            pair_data[
                "architecture"
            ] == architecture
        ]

        if len(
            row
        ) == 0:

            continue

        row = row.iloc[
            0
        ]

        x = (
            x_center
            + ARCH_OFFSETS[
                architecture
            ]
        )

        yerr = row[
            "metric_sd"
        ]

        if not np.isfinite(
            yerr
        ):

            yerr = None

        ax.errorbar(

            x,

            row[
                "metric_mean"
            ],

            yerr=yerr,

            fmt="o",

            markersize=6.5,

            markerfacecolor=(
                ARCH_COLORS[
                    architecture
                ]
            ),

            markeredgecolor="white",

            markeredgewidth=0.65,

            ecolor=(
                ARCH_COLORS[
                    architecture
                ]
            ),

            elinewidth=1.0,

            capsize=2.5,

            alpha=0.95,

            zorder=4,
        )


# ============================================================
# X LABELS
#
# Main chemistry name
# observed compact population
# motif stickiness contrast
# ============================================================

xticklabels = []


for pair in PAIR_ORDER:

    row = chemistry_summary[
        chemistry_summary[
            "pair"
        ] == pair
    ].iloc[
        0
    ]

    fcompact = row[
        "mean_fraction_collapsed"
    ]

    delta_lambda = row[
        "delta_lambda"
    ]

    label = (

        f"{PAIR_DISPLAY[pair]}\n"

        + r"$f_{\mathrm{compact}}$"
        + f" = {fcompact:.2f}\n"

        + r"$\Delta\bar{\lambda}$"
        + f" = {delta_lambda:.3f}"
    )

    xticklabels.append(
        label
    )


ax.set_xticks(
    x_positions
)


ax.set_xticklabels(

    xticklabels,

    rotation=32,

    ha="right",

    rotation_mode="anchor"
)


# ============================================================
# AXES
# ============================================================

ax.set_ylabel(
    YLABEL
)


ax.set_xlabel(
    "Motif chemistry, ordered by mean compact-state population",
    labelpad=12
)


ax.set_xlim(
    -0.65,
    len(
        PAIR_ORDER
    ) - 0.35
)


# ============================================================
# GRID / SPINES
# ============================================================

ax.yaxis.grid(

    True,

    linestyle="--",

    linewidth=0.5,

    alpha=0.18
)


ax.xaxis.grid(
    False
)


ax.spines[
    "top"
].set_visible(
    False
)


ax.spines[
    "right"
].set_visible(
    False
)


# ============================================================
# COMPACT-STATE POPULATION ABOVE EACH CHEMISTRY
#
# This makes the expanded -> compact progression immediately
# visible without introducing a second y-axis.
# ============================================================

ymin, ymax = ax.get_ylim()

yrange = (
    ymax
    - ymin
)


for pair in PAIR_ORDER:

    x = PAIR_TO_X[
        pair
    ]

    fcompact = chemistry_summary.loc[

        chemistry_summary[
            "pair"
        ] == pair,

        "mean_fraction_collapsed"

    ].iloc[
        0
    ]

    ax.text(

        x,

        ymax
        + 0.025
        * yrange,

        f"{100 * fcompact:.0f}%",

        ha="center",

        va="bottom",

        fontsize=8,

        fontweight="medium",

        clip_on=False,
    )


ax.text(

    -0.62,

    ymax
    + 0.025
    * yrange,

    "Compact:",

    ha="right",

    va="bottom",

    fontsize=8,

    fontweight="medium",

    clip_on=False,
)


# ============================================================
# LEGEND
# ============================================================

legend_handles = []


for architecture in ARCH_ORDER:

    handle = Line2D(

        [0],

        [0],

        marker="o",

        linestyle="None",

        markerfacecolor=(
            ARCH_COLORS[
                architecture
            ]
        ),

        markeredgecolor="white",

        markeredgewidth=0.6,

        markersize=7,

        label=architecture,
    )

    legend_handles.append(
        handle
    )


ax.legend(

    handles=legend_handles,

    title="Architecture",

    frameon=False,

    bbox_to_anchor=(
        1.01,
        1.0
    ),

    loc="upper left",

    borderaxespad=0.0,
)


# ============================================================
# LAYOUT
# ============================================================

fig.subplots_adjust(

    left=0.075,

    right=0.82,

    bottom=0.34,

    top=0.88
)


# ============================================================
# SAVE
# ============================================================

basename = (

    f"DR2_supplement_{FILETAG}_"
    f"ordered_fraction_collapsed"
)


for extension in [

    "pdf",

    "png",

    "svg",
]:

    save_kwargs = {}

    if extension == "png":

        save_kwargs[
            "dpi"
        ] = 600

    fig.savefig(

        OUTDIR
        / f"{basename}.{extension}",

        bbox_inches="tight",

        **save_kwargs
    )


plt.close(
    fig
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)


print(
    f"\nRuns used: "
    f"{len(df)}"
)


print(
    f"Chemistries plotted: "
    f"{len(PAIR_ORDER)}"
)


print(
    f"Architectures expected per chemistry: "
    f"{len(ARCH_ORDER)}"
)


print(
    "\nOutput directory:"
)


print(
    OUTDIR
)


print(
    "\nFigure:"
)


print(
    OUTDIR
    / f"{basename}.pdf"
)


print(
    "\nPNG:"
)


print(
    OUTDIR
    / f"{basename}.png"
)


print(
    "\nTables:"
)


print(
    OUTDIR
    / "DR2_supplement_chemistry_conformational_regime.csv"
)


print(
    OUTDIR
    / "DR2_supplement_architecture_summary.csv"
)


print(
    "\nNOTE:"
)


print(
    "The x-axis is ordered by the observed mean "
    "compact-state population."
)


print(
    "Each violin contains the eight architecture-level means."
)


print(
    "Each colored point is the architecture mean across "
    "independent replicates; error bars are replicate SD."
)


print(
    "\nTo make the thesis-primary IQR version, change:"
)


print(
    'METRIC = "rg_range_90_nm"'
)


print(
    "to:"
)


print(
    'METRIC = "iqr_rg_nm"'
)