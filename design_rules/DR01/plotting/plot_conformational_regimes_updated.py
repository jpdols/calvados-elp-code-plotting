"""
plot_conformational_regimes.py

DESIGN RULE ANALYSIS
====================

Question:
How does motif composition determine the conformational regime of ELP-like
sequences, and how much variation can sequence architecture introduce within
that regime?

DATA SOURCES
============

1. MIXED LIBRARIES
------------------
VGIP, VGIG, VAIP and VAIG are read ONLY from their existing:

    all_rg_values.csv

No DCD files are opened for these libraries.

Expected folders:

    analysis_library_1/
        VGIP_detailed_stride1/all_rg_values.csv
        VGIG_detailed_stride1/all_rg_values.csv
        VAIP_detailed_stride1/all_rg_values.csv
        VAIG_detailed_stride1/all_rg_values.csv


2. PURE-MOTIF BASELINES
-----------------------
The pure controls are calculated directly from DCD + top.pdb using MDAnalysis:

    001_VPGVG
    002_VPGIG
    005_IPAVG
    006_VPAVG

Expected structure:

    analysis_library_1/
        baseline/
            001_VPGVG/
                300K/
                    001_VPGVG/
                        top.pdb
                        001_VPGVG.dcd
                370K/
                    001_VPGVG/
                        top.pdb
                        001_VPGVG.dcd

            ...

TEMPERATURES
============

    300 K
    370 K

COMPACTION DEFINITION
=====================

    compact if Rg <= 4.61 nm

OUTPUTS
=======

CSV:
    pure_baseline_statistics.csv
    architecture_level_statistics.csv
    composition_level_statistics.csv
    temperature_response_by_architecture.csv

Plots:
    mean Rg vs composition at 300 K
    mean Rg vs composition at 370 K
    compact population vs composition at 300 K
    compact population vs composition at 370 K
    architecture sensitivity vs composition at 300 K
    architecture sensitivity vs composition at 370 K

For mixed compositions:
    one small point = one architecture trajectory mean

At 0% and 100%:
    one diamond = one pure-motif trajectory

The 50,000 trajectory frames are NOT treated as independent sequence-level
observations.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\analysis_library_1"
)

BASELINE_ROOT = (
    ROOT
    / "baseline"
)

OUTDIR = Path(
    r"E:\Master thesis\calvados_tea\plotting_thesis"
    r"\Design_rule_1\composition_regime_plots"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


TEMPERATURES = [
    300,
    370,
]


RG_THRESHOLD = 4.61


# =============================================================================
# FAMILY DEFINITIONS
# =============================================================================

FAMILIES = {

    "VGIP": {
        "folder": "VGIP_detailed_stride1",
        "label": "VPGVG + IPAVG",
        "weak_motif": "VPGVG",
        "strong_motif": "IPAVG",
        "weak_endpoint": "001_VPGVG",
        "strong_endpoint": "005_IPAVG",
    },

    "VGIG": {
        "folder": "VGIG_detailed_stride1",
        "label": "VPGVG + VPGIG",
        "weak_motif": "VPGVG",
        "strong_motif": "VPGIG",
        "weak_endpoint": "001_VPGVG",
        "strong_endpoint": "002_VPGIG",
    },

    "VAIP": {
        "folder": "VAIP_detailed_stride1",
        "label": "VPAVG + IPAVG",
        "weak_motif": "VPAVG",
        "strong_motif": "IPAVG",
        "weak_endpoint": "006_VPAVG",
        "strong_endpoint": "005_IPAVG",
    },

    "VAIG": {
        "folder": "VAIG_detailed_stride1",
        "label": "VPAVG + VPGIG",
        "weak_motif": "VPAVG",
        "strong_motif": "VPGIG",
        "weak_endpoint": "006_VPAVG",
        "strong_endpoint": "002_VPGIG",
    },
}


# =============================================================================
# LIBRARY ID RANGES
# =============================================================================
#
# Each ratio contains 8 deliberately designed architectures.
#
# =============================================================================

FAMILY_ID_RANGES = {

    "VGIP": {
        "R1": (31, 38),
        "R2": (39, 46),
        "R3": (47, 54),
        "R4": (55, 62),
        "R5": (63, 70),
        "R6": (71, 78),
    },

    "VGIG": {
        "R1": (79, 86),
        "R2": (87, 94),
        "R3": (95, 102),
        "R4": (103, 110),
        "R5": (111, 118),
        "R6": (119, 126),
    },

    "VAIP": {
        "R1": (127, 134),
        "R2": (135, 142),
        "R3": (143, 150),
        "R4": (151, 158),
        "R5": (159, 166),
        "R6": (167, 174),
    },

    "VAIG": {
        "R1": (175, 182),
        "R2": (183, 190),
        "R3": (191, 198),
        "R4": (199, 206),
        "R5": (207, 214),
        "R6": (215, 222),
    },
}


# =============================================================================
# COMPOSITION SERIES
# =============================================================================
#
# 84 pentapeptides = 420 residues.
#
# strong_fraction = fraction of the more strongly interacting motif
#
# =============================================================================

RATIO_COMPOSITION = {

    "R1": (72, 12),

    "R2": (63, 21),

    "R3": (56, 28),

    "R4": (42, 42),

    "R5": (28, 56),

    "R6": (14, 70),
}


COMPOSITION_INFO = {}

for ratio, (
    weak_count,
    strong_count,
) in RATIO_COMPOSITION.items():

    total = (
        weak_count
        + strong_count
    )

    COMPOSITION_INFO[
        ratio
    ] = {

        "weak_count":
            weak_count,

        "strong_count":
            strong_count,

        "strong_fraction":
            strong_count / total,

        "strong_percent":
            100.0
            * strong_count
            / total,
    }


# =============================================================================
# BASELINES REQUIRED
# =============================================================================

PURE_IDS = [

    "001_VPGVG",

    "002_VPGIG",

    "005_IPAVG",

    "006_VPAVG",
]


# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (
    11.5,
    6.7,
)

POINT_SIZE = 38

MEAN_POINT_SIZE = 90

ENDPOINT_SIZE = 110

POINT_ALPHA = 0.45

LINEWIDTH = 2.2

FAMILY_OFFSET = 0.006

ARCH_JITTER = 0.002

RNG = np.random.default_rng(
    42
)


# =============================================================================
# HELPER: EXTRACT NUMERIC ID
# =============================================================================

def extract_numeric_id(
    value,
):

    text = str(
        value
    ).strip()

    match = re.search(
        r"(\d+)",
        text,
    )

    if match is None:

        return np.nan

    return int(
        match.group(1)
    )


# =============================================================================
# HELPER: DETERMINE R1-R6 FROM ID
# =============================================================================

def ratio_from_numeric_id(
    value,
    family_key,
):

    numeric_id = extract_numeric_id(
        value
    )

    if pd.isna(
        numeric_id
    ):

        return None

    numeric_id = int(
        numeric_id
    )

    for ratio, (
        first_id,
        last_id,
    ) in FAMILY_ID_RANGES[
        family_key
    ].items():

        if (
            first_id
            <= numeric_id
            <= last_id
        ):

            return ratio

    return None


# =============================================================================
# HELPER: TRAJECTORY STATISTICS
# =============================================================================

def summarize_rg(
    rg_values,
):

    values = pd.to_numeric(
        pd.Series(
            rg_values
        ),
        errors="coerce",
    )

    values = (
        values
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(
        values
    ) == 0:

        raise ValueError(
            "Cannot summarize empty Rg array."
        )

    p05 = np.percentile(
        values,
        5,
    )

    p95 = np.percentile(
        values,
        95,
    )

    compact_population = np.mean(
        values
        <= RG_THRESHOLD
    )

    return {

        "mean_rg":
            np.mean(
                values
            ),

        "median_rg":
            np.median(
                values
            ),

        "sd_rg_within":
            np.std(
                values,
                ddof=1,
            ),

        "rg_p05":
            p05,

        "rg_p95":
            p95,

        "central90_rg":
            p95 - p05,

        "compact_population":
            compact_population,

        "expanded_population":
            1.0
            - compact_population,

        "n_frames":
            len(
                values
            ),
    }


# =============================================================================
# LOAD MIXED FAMILY FROM all_rg_values.csv
# =============================================================================

def load_mixed_family(
    family_key,
    info,
):

    csv_path = (
        ROOT
        / info["folder"]
        / "all_rg_values.csv"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            "\nCould not find:\n"
            f"{csv_path}"
        )


    print()
    print(
        "=" * 78
    )

    print(
        f"LOADING MIXED FAMILY "
        f"{family_key}"
    )

    print(
        csv_path
    )


    # -------------------------------------------------------------------------
    # Check columns without loading file
    # -------------------------------------------------------------------------

    header = pd.read_csv(
        csv_path,
        nrows=0,
    )

    available = list(
        header.columns
    )


    required = [

        "id",

        "temp",

        "group",

        "architecture_type",

        "rg_nm",
    ]


    missing = [

        column

        for column
        in required

        if column
        not in available
    ]


    if missing:

        raise ValueError(

            "\nMissing required columns:\n"
            f"{missing}\n\n"

            "Available columns:\n"
            f"{available}"
        )


    # -------------------------------------------------------------------------
    # Only read required columns
    # -------------------------------------------------------------------------

    df = pd.read_csv(

        csv_path,

        usecols=required,
    )


    print(
        f"Raw rows: "
        f"{len(df):,}"
    )


    # -------------------------------------------------------------------------
    # Temperature parsing
    # -------------------------------------------------------------------------

    print()
    print(
        "Raw temperature values:"
    )

    print(
        df[
            "temp"
        ]
        .drop_duplicates()
        .tolist()
    )


    df[
        "temp"
    ] = (

        df[
            "temp"
        ]

        .astype(
            str
        )

        .str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False,
        )
    )


    df[
        "temp"
    ] = pd.to_numeric(

        df[
            "temp"
        ],

        errors="coerce",
    )


    print(
        "Parsed temperatures:"
    )

    print(
        sorted(
            df[
                "temp"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    )


    df = df[

        df[
            "temp"
        ].isin(
            TEMPERATURES
        )

    ].copy()


    if df.empty:

        raise ValueError(

            "\nNo rows remain after "
            "temperature filtering."
        )


    # -------------------------------------------------------------------------
    # Rg
    # -------------------------------------------------------------------------

    df[
        "rg_nm"
    ] = pd.to_numeric(

        df[
            "rg_nm"
        ],

        errors="coerce",
    )


    df = df.dropna(
        subset=[
            "rg_nm"
        ]
    )


    # -------------------------------------------------------------------------
    # Determine numeric ID and ratio
    # -------------------------------------------------------------------------

    df[
        "numeric_id"
    ] = (

        df[
            "id"
        ]

        .apply(
            extract_numeric_id
        )
    )


    df[
        "ratio"
    ] = (

        df[
            "id"
        ]

        .apply(

            lambda value:

            ratio_from_numeric_id(
                value,
                family_key,
            )
        )
    )


    # -------------------------------------------------------------------------
    # ID diagnostics
    # -------------------------------------------------------------------------

    id_check = (

        df[
            [
                "id",
                "numeric_id",
                "group",
                "architecture_type",
                "ratio",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            "numeric_id"
        )
    )


    print()
    print(
        "Detected architectures:"
    )

    print(
        id_check.to_string(
            index=False
        )
    )


    unresolved = id_check[
        id_check[
            "ratio"
        ].isna()
    ]


    if not unresolved.empty:

        print()
        print(
            "WARNING: unresolved IDs:"
        )

        print(
            unresolved.to_string(
                index=False
            )
        )


    df = df.dropna(
        subset=[
            "ratio"
        ]
    ).copy()


    if df.empty:

        raise ValueError(

            "\nNo rows remain after "
            "R1-R6 assignment."
        )


    # -------------------------------------------------------------------------
    # Summarize one trajectory at a time
    # -------------------------------------------------------------------------

    rows = []


    group_columns = [

        "id",

        "numeric_id",

        "temp",

        "group",

        "architecture_type",

        "ratio",
    ]


    for keys, trajectory in df.groupby(

        group_columns,

        sort=False,
    ):

        (
            sequence_id,
            numeric_id,
            temperature,
            group,
            architecture_type,
            ratio,
        ) = keys


        stats = summarize_rg(
            trajectory[
                "rg_nm"
            ]
        )


        composition = (
            COMPOSITION_INFO[
                ratio
            ]
        )


        row = {

            "family":
                family_key,

            "family_label":
                info[
                    "label"
                ],

            "id":
                sequence_id,

            "numeric_id":
                numeric_id,

            "temp":
                int(
                    temperature
                ),

            "group":
                group,

            "architecture_type":
                architecture_type,

            "ratio":
                ratio,

            "weak_motif":
                info[
                    "weak_motif"
                ],

            "strong_motif":
                info[
                    "strong_motif"
                ],

            "weak_count":
                composition[
                    "weak_count"
                ],

            "strong_count":
                composition[
                    "strong_count"
                ],

            "strong_fraction":
                composition[
                    "strong_fraction"
                ],

            "strong_percent":
                composition[
                    "strong_percent"
                ],

            "is_endpoint":
                False,

            "data_source":
                "all_rg_values.csv",
        }


        row.update(
            stats
        )


        rows.append(
            row
        )


    summary = pd.DataFrame(
        rows
    )


    if summary.empty:

        raise ValueError(

            "\nTrajectory summary is empty "
            f"for {family_key}."
        )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print()
    print(
        "Architectures per temperature / ratio:"
    )


    validation = (

        summary

        .groupby(
            [
                "temp",
                "ratio",
            ]
        )[
            "id"
        ]

        .nunique()

        .unstack(
            fill_value=0
        )

        .reindex(
            columns=[
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
            ]
        )
    )


    print(
        validation.to_string()
    )


    bad = (
        validation
        != 8
    )


    if bad.any().any():

        print()
        print(
            "WARNING:"
        )

        print(
            "Not every temperature/ratio "
            "contains exactly 8 architectures."
        )

    else:

        print()
        print(
            "PASS: every ratio contains "
            "8 architectures at each temperature."
        )


    return summary


# =============================================================================
# BASELINE: FIND TOPOLOGY
# =============================================================================

def find_topology(
    run_dir,
    baseline_id,
):

    preferred = [

        run_dir
        / "top.pdb",

        run_dir
        / f"{baseline_id}.pdb",
    ]


    for path in preferred:

        if path.exists():

            return path


    pdb_files = sorted(
        run_dir.glob(
            "*.pdb"
        )
    )


    if pdb_files:

        return pdb_files[
            0
        ]


    raise FileNotFoundError(

        "\nNo topology PDB found in:\n"
        f"{run_dir}"
    )


# =============================================================================
# BASELINE: CALCULATE Rg FROM DCD
# =============================================================================

def calculate_rg_from_dcd(
    topology_path,
    dcd_path,
):

    try:

        import MDAnalysis as mda

    except ImportError as exc:

        raise ImportError(

            "\nMDAnalysis is required ONLY "
            "for the pure baseline DCD files.\n"

            "The mixed libraries do not use "
            "MDAnalysis."
        ) from exc


    universe = mda.Universe(

        str(
            topology_path
        ),

        str(
            dcd_path
        ),
    )


    atoms = (
        universe.atoms
    )


    n_frames = len(
        universe.trajectory
    )


    n_beads = len(
        atoms
    )


    print()
    print(
        f"Topology: {topology_path}"
    )

    print(
        f"DCD:      {dcd_path}"
    )

    print(
        f"Beads:    {n_beads}"
    )

    print(
        f"Frames:   {n_frames:,}"
    )


    rg_values = np.empty(
        n_frames,
        dtype=np.float64,
    )


    # -------------------------------------------------------------------------
    # Equal-bead-weight geometric radius of gyration
    #
    # MDAnalysis coordinates are in Angstrom.
    # Output is converted to nm.
    # -------------------------------------------------------------------------

    for frame_index, _ in enumerate(
        universe.trajectory
    ):

        positions = (
            atoms.positions
            .astype(
                np.float64,
                copy=False,
            )
        )


        center = np.mean(
            positions,
            axis=0,
        )


        displacement = (
            positions
            - center
        )


        rg_squared = np.mean(

            np.sum(

                displacement
                * displacement,

                axis=1,
            )
        )


        rg_angstrom = np.sqrt(
            rg_squared
        )


        rg_values[
            frame_index
        ] = (
            rg_angstrom
            / 10.0
        )


    print(
        f"Mean Rg:  "
        f"{np.mean(rg_values):.4f} nm"
    )


    return rg_values


# =============================================================================
# LOAD PURE BASELINES
# =============================================================================

def load_pure_baselines():

    print()
    print(
        "=" * 78
    )

    print(
        "LOADING PURE-MOTIF BASELINES "
        "FROM DCD"
    )

    print(
        "=" * 78
    )


    rows = []


    for baseline_id in PURE_IDS:

        for temperature in TEMPERATURES:


            run_dir = (

                BASELINE_ROOT

                / baseline_id

                / f"{temperature}K"

                / baseline_id
            )


            print()
            print(
                "-" * 78
            )

            print(
                f"{baseline_id} "
                f"{temperature} K"
            )


            if not run_dir.exists():

                print(
                    "WARNING: directory missing:"
                )

                print(
                    run_dir
                )

                continue


            # -----------------------------------------------------------------
            # DCD
            # -----------------------------------------------------------------

            expected_dcd = (

                run_dir

                / f"{baseline_id}.dcd"
            )


            if expected_dcd.exists():

                dcd_path = (
                    expected_dcd
                )

            else:

                dcd_candidates = sorted(
                    run_dir.glob(
                        "*.dcd"
                    )
                )


                if not dcd_candidates:

                    print(
                        "WARNING: no DCD found."
                    )

                    continue


                dcd_path = (
                    dcd_candidates[
                        0
                    ]
                )


            # -----------------------------------------------------------------
            # topology
            # -----------------------------------------------------------------

            topology_path = (
                find_topology(
                    run_dir,
                    baseline_id,
                )
            )


            # -----------------------------------------------------------------
            # Rg
            # -----------------------------------------------------------------

            rg_values = (
                calculate_rg_from_dcd(
                    topology_path,
                    dcd_path,
                )
            )


            stats = summarize_rg(
                rg_values
            )


            row = {

                "id":
                    baseline_id,

                "numeric_id":
                    extract_numeric_id(
                        baseline_id
                    ),

                "temp":
                    temperature,

                "group":
                    "pure",

                "architecture_type":
                    "pure motif",

                "ratio":
                    "endpoint",

                "is_endpoint":
                    True,

                "data_source":
                    "DCD",

                "topology_file":
                    str(
                        topology_path
                    ),

                "trajectory_file":
                    str(
                        dcd_path
                    ),
            }


            row.update(
                stats
            )


            rows.append(
                row
            )


    baselines = pd.DataFrame(
        rows
    )


    if baselines.empty:

        raise RuntimeError(
            "\nNo baseline trajectories "
            "could be loaded."
        )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    expected = {

        (
            baseline_id,
            temperature,
        )

        for baseline_id
        in PURE_IDS

        for temperature
        in TEMPERATURES
    }


    observed = {

        (
            row[
                "id"
            ],

            int(
                row[
                    "temp"
                ]
            ),
        )

        for _, row
        in baselines.iterrows()
    }


    missing = (
        expected
        - observed
    )


    print()
    print(
        "=" * 78
    )

    print(
        "PURE BASELINE SUMMARY"
    )

    print(
        "=" * 78
    )


    print(

        baselines[
            [
                "id",
                "temp",
                "mean_rg",
                "median_rg",
                "central90_rg",
                "compact_population",
                "n_frames",
            ]
        ]

        .sort_values(
            [
                "id",
                "temp",
            ]
        )

        .round(
            4
        )

        .to_string(
            index=False
        )
    )


    if missing:

        print()
        print(
            "WARNING: missing baselines:"
        )

        for item in sorted(
            missing
        ):

            print(
                item
            )

    else:

        print()
        print(
            "PASS: all 8 baseline trajectories found."
        )


    return baselines


# =============================================================================
# LOAD ALL MIXED LIBRARIES
# =============================================================================

mixed_frames = []


for family_key, info in FAMILIES.items():

    family_data = load_mixed_family(
        family_key,
        info,
    )

    mixed_frames.append(
        family_data
    )


mixed = pd.concat(

    mixed_frames,

    ignore_index=True,
)


print()
print(
    "=" * 78
)

print(
    "MIXED LIBRARY SUMMARY"
)

print(
    "=" * 78
)


print(
    f"Trajectory-level observations: "
    f"{len(mixed):,}"
)


print(

    mixed

    .groupby(
        [
            "temp",
            "family",
        ]
    )[
        "id"
    ]

    .nunique()

    .to_string()
)


# =============================================================================
# LOAD PURE BASELINES FROM DCD
# =============================================================================

pure_baselines = (
    load_pure_baselines()
)


# =============================================================================
# SAVE UNIQUE PURE BASELINES
# =============================================================================

pure_baseline_file = (

    OUTDIR

    / "pure_baseline_statistics.csv"
)


pure_baselines.to_csv(

    pure_baseline_file,

    index=False,
)


print()
print(
    f"Saved:\n"
    f"{pure_baseline_file}"
)


# =============================================================================
# ADD PURE ENDPOINTS TO EACH FAMILY
# =============================================================================

endpoint_rows = []


for family_key, info in FAMILIES.items():

    for side in [
        "weak",
        "strong",
    ]:


        if side == "weak":

            endpoint_id = info[
                "weak_endpoint"
            ]

            strong_fraction = 0.0

            weak_count = 84

            strong_count = 0


        else:

            endpoint_id = info[
                "strong_endpoint"
            ]

            strong_fraction = 1.0

            weak_count = 0

            strong_count = 84


        for temperature in TEMPERATURES:


            match = pure_baselines[

                (
                    pure_baselines[
                        "id"
                    ]
                    == endpoint_id
                )

                &

                (
                    pure_baselines[
                        "temp"
                    ]
                    == temperature
                )
            ]


            if match.empty:

                print()
                print(
                    "WARNING: missing endpoint:"
                )

                print(
                    endpoint_id,
                    temperature,
                )

                continue


            row = (
                match
                .iloc[0]
                .to_dict()
            )


            row.update({

                "family":
                    family_key,

                "family_label":
                    info[
                        "label"
                    ],

                "weak_motif":
                    info[
                        "weak_motif"
                    ],

                "strong_motif":
                    info[
                        "strong_motif"
                    ],

                "ratio":
                    (
                        "pure_weak"
                        if side == "weak"
                        else "pure_strong"
                    ),

                "weak_count":
                    weak_count,

                "strong_count":
                    strong_count,

                "strong_fraction":
                    strong_fraction,

                "strong_percent":
                    100.0
                    * strong_fraction,

                "is_endpoint":
                    True,
            })


            endpoint_rows.append(
                row
            )


family_endpoints = pd.DataFrame(
    endpoint_rows
)


# =============================================================================
# COMPLETE ANALYSIS TABLE
# =============================================================================

data = pd.concat(

    [
        mixed,
        family_endpoints,
    ],

    ignore_index=True,

    sort=False,
)


# =============================================================================
# SAVE ARCHITECTURE-LEVEL MASTER CSV
# =============================================================================

architecture_columns = [

    "family",

    "family_label",

    "id",

    "numeric_id",

    "temp",

    "ratio",

    "architecture_type",

    "weak_motif",

    "strong_motif",

    "weak_count",

    "strong_count",

    "strong_fraction",

    "strong_percent",

    "is_endpoint",

    "data_source",

    "mean_rg",

    "median_rg",

    "sd_rg_within",

    "rg_p05",

    "rg_p95",

    "central90_rg",

    "compact_population",

    "expanded_population",

    "n_frames",
]


architecture_columns = [

    column

    for column
    in architecture_columns

    if column
    in data.columns
]


architecture_file = (

    OUTDIR

    / "architecture_level_statistics.csv"
)


data[
    architecture_columns
].sort_values(

    [
        "temp",
        "family",
        "strong_fraction",
        "numeric_id",
    ]

).to_csv(

    architecture_file,

    index=False,
)


print()
print(
    f"Saved:\n"
    f"{architecture_file}"
)


# =============================================================================
# COMPOSITION-LEVEL STATISTICS
# =============================================================================
#
# IMPORTANT:
#
# For R1-R6:
#     statistics describe variation across the 8 architectures.
#
# For pure endpoints:
#     n_architectures = 1.
#
# Therefore architecture spread at the endpoints is not interpreted.
#
# =============================================================================

composition_summary = (

    data

    .groupby(
        [
            "family",
            "family_label",
            "temp",
            "weak_motif",
            "strong_motif",
            "strong_fraction",
            "strong_percent",
        ],

        as_index=False,
    )

    .agg(

        n_architectures=(
            "id",
            "nunique",
        ),

        mean_rg=(
            "mean_rg",
            "mean",
        ),

        median_architecture_mean_rg=(
            "mean_rg",
            "median",
        ),

        sd_between_architectures=(
            "mean_rg",
            "std",
        ),

        min_architecture_mean_rg=(
            "mean_rg",
            "min",
        ),

        max_architecture_mean_rg=(
            "mean_rg",
            "max",
        ),

        mean_central90_rg=(
            "central90_rg",
            "mean",
        ),

        sd_central90_rg=(
            "central90_rg",
            "std",
        ),

        mean_compact_population=(
            "compact_population",
            "mean",
        ),

        sd_compact_population=(
            "compact_population",
            "std",
        ),

        min_compact_population=(
            "compact_population",
            "min",
        ),

        max_compact_population=(
            "compact_population",
            "max",
        ),
    )
)


composition_summary[
    "architecture_rg_range"
] = (

    composition_summary[
        "max_architecture_mean_rg"
    ]

    -

    composition_summary[
        "min_architecture_mean_rg"
    ]
)


composition_summary[
    "compact_population_range"
] = (

    composition_summary[
        "max_compact_population"
    ]

    -

    composition_summary[
        "min_compact_population"
    ]
)


# Endpoints do not have architecture variation.
# Set these values to NaN instead of misleadingly showing zero.

endpoint_mask = (

    composition_summary[
        "n_architectures"
    ]
    == 1
)


composition_summary.loc[

    endpoint_mask,

    [
        "sd_between_architectures",
        "architecture_rg_range",
        "sd_compact_population",
        "compact_population_range",
    ],

] = np.nan


composition_file = (

    OUTDIR

    / "composition_level_statistics.csv"
)


composition_summary.sort_values(

    [
        "temp",
        "family",
        "strong_fraction",
    ]

).to_csv(

    composition_file,

    index=False,
)


print(
    f"Saved:\n"
    f"{composition_file}"
)


# =============================================================================
# TEMPERATURE RESPONSE PER ARCHITECTURE
# =============================================================================

temperature_source = data[
    [
        "family",
        "id",
        "numeric_id",
        "ratio",
        "architecture_type",
        "strong_fraction",
        "temp",
        "mean_rg",
        "median_rg",
        "central90_rg",
        "compact_population",
    ]
].copy()


temperature_wide = temperature_source.pivot_table(

    index=[
        "family",
        "id",
        "numeric_id",
        "ratio",
        "architecture_type",
        "strong_fraction",
    ],

    columns="temp",

    values=[
        "mean_rg",
        "median_rg",
        "central90_rg",
        "compact_population",
    ],

    aggfunc="first",
)


temperature_wide.columns = [

    f"{metric}_{int(temp)}K"

    for metric, temp
    in temperature_wide.columns
]


temperature_wide = (
    temperature_wide
    .reset_index()
)


# -------------------------------------------------------------------------
# Delta Rg
# -------------------------------------------------------------------------

if (
    "mean_rg_300K"
    in temperature_wide.columns

    and

    "mean_rg_370K"
    in temperature_wide.columns
):

    temperature_wide[
        "delta_mean_rg_370_minus_300"
    ] = (

        temperature_wide[
            "mean_rg_370K"
        ]

        -

        temperature_wide[
            "mean_rg_300K"
        ]
    )


# -------------------------------------------------------------------------
# Relative Rg change
# -------------------------------------------------------------------------

if (
    "mean_rg_300K"
    in temperature_wide.columns

    and

    "mean_rg_370K"
    in temperature_wide.columns
):

    temperature_wide[
        "relative_mean_rg_change"
    ] = (

        (
            temperature_wide[
                "mean_rg_370K"
            ]

            -

            temperature_wide[
                "mean_rg_300K"
            ]
        )

        /

        temperature_wide[
            "mean_rg_300K"
        ]
    )


    temperature_wide[
        "percent_mean_rg_change"
    ] = (

        100.0

        * temperature_wide[
            "relative_mean_rg_change"
        ]
    )


# -------------------------------------------------------------------------
# Compact population change
# -------------------------------------------------------------------------

if (
    "compact_population_300K"
    in temperature_wide.columns

    and

    "compact_population_370K"
    in temperature_wide.columns
):

    temperature_wide[
        "delta_compact_population_370_minus_300"
    ] = (

        temperature_wide[
            "compact_population_370K"
        ]

        -

        temperature_wide[
            "compact_population_300K"
        ]
    )


# -------------------------------------------------------------------------
# Central 90% range change
# -------------------------------------------------------------------------

if (
    "central90_rg_300K"
    in temperature_wide.columns

    and

    "central90_rg_370K"
    in temperature_wide.columns
):

    temperature_wide[
        "delta_central90_rg_370_minus_300"
    ] = (

        temperature_wide[
            "central90_rg_370K"
        ]

        -

        temperature_wide[
            "central90_rg_300K"
        ]
    )


temperature_file = (

    OUTDIR

    / "temperature_response_by_architecture.csv"
)


temperature_wide.to_csv(

    temperature_file,

    index=False,
)


print(
    f"Saved:\n"
    f"{temperature_file}"
)


# =============================================================================
# VALIDATION
# =============================================================================

print()
print(
    "=" * 78
)

print(
    "FINAL VALIDATION"
)

print(
    "=" * 78
)


validation = (

    data

    .groupby(
        [
            "temp",
            "family",
            "strong_fraction",
        ]
    )[
        "id"
    ]

    .nunique()
)


print(
    validation.to_string()
)


print()
print(
    "Expected:"
)

print(
    "  0% stronger motif   -> 1 pure sequence"
)

print(
    "  R1-R6               -> 8 architectures"
)

print(
    "  100% stronger motif -> 1 pure sequence"
)


# =============================================================================
# FINAL THESIS PLOTTING
# =============================================================================
#
# DESIGN RULE 1
# -------------
# Question:
#     Does motif composition mainly determine the conformational regime?
#
# This final figure uses ONLY 370 K and shows the same data in two ways:
#
#   A. x = fraction of the higher-lambda motif
#   B. x = sequence-average CALVADOS interaction parameter per residue
#
# For each motif pair:
#     solid line  = mean Rg across the 8 architectures
#     shaded area = full min-max range of the 8 architecture mean Rg values
#
# Pure endpoints are deliberately omitted because they contain only one sequence
# and therefore cannot define an architecture envelope.
# =============================================================================

PLOT_TEMPERATURE = 370

# Exact motif-averaged CALVADOS residue interaction parameters at 370 K,
# gamma = 6.0. Because every motif is five residues long, the sequence-average
# lambda for a mixed composition is the composition-weighted motif mean.
MOTIF_LAMBDA = {
    "VPGVG": 0.6850471057402,
    "IPAVG": 0.7430614622074,
    "VPGIG": 0.7986503898464,
    "VPAVG": 0.6294581781012,
}

# Use matplotlib's default color cycle so this remains consistent with the
# rest of the analysis without hard-coding colors.
family_keys = list(FAMILIES.keys())
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
family_colors = {
    family: color_cycle[i % len(color_cycle)]
    for i, family in enumerate(family_keys)
}

# -----------------------------------------------------------------------------
# Build plotting table from the already-loaded architecture-level data.
# Only R1-R6 mixed libraries are used here; pure endpoints are excluded.
# -----------------------------------------------------------------------------
plot_data = mixed[
    mixed["temp"] == PLOT_TEMPERATURE
].copy()

plot_data["lambda_weak"] = plot_data["weak_motif"].map(MOTIF_LAMBDA)
plot_data["lambda_strong"] = plot_data["strong_motif"].map(MOTIF_LAMBDA)

if plot_data[["lambda_weak", "lambda_strong"]].isna().any().any():
    missing_motifs = sorted(
        set(plot_data.loc[plot_data["lambda_weak"].isna(), "weak_motif"].dropna())
        | set(plot_data.loc[plot_data["lambda_strong"].isna(), "strong_motif"].dropna())
    )
    raise ValueError(
        "Missing MOTIF_LAMBDA values for: " + ", ".join(missing_motifs)
    )

plot_data["lambda_sequence"] = (
    (1.0 - plot_data["strong_fraction"]) * plot_data["lambda_weak"]
    + plot_data["strong_fraction"] * plot_data["lambda_strong"]
)

# One row per motif pair and composition. The min/max values are the outermost
# architecture-level mean Rg values, so the shaded band directly visualizes the
# conformational range accessible through architecture at fixed composition.
regime_summary = (
    plot_data
    .groupby(
        [
            "family",
            "family_label",
            "weak_motif",
            "strong_motif",
            "strong_fraction",
            "strong_percent",
            "lambda_sequence",
        ],
        as_index=False,
    )
    .agg(
        n_architectures=("id", "nunique"),
        mean_rg=("mean_rg", "mean"),
        min_rg=("mean_rg", "min"),
        max_rg=("mean_rg", "max"),
        sd_rg=("mean_rg", "std"),
    )
    .sort_values(["family", "strong_fraction"])
)

# Validate that every shaded envelope is based on the intended eight designs.
bad = regime_summary[regime_summary["n_architectures"] != 8]
if not bad.empty:
    print("WARNING: some composition points do not contain exactly 8 architectures:")
    print(
        bad[
            ["family", "strong_percent", "n_architectures"]
        ].to_string(index=False)
    )
else:
    print("PASS: every plotted composition contains exactly 8 architectures.")

# Save the exact values used in the A/B figure.
regime_output = OUTDIR / "composition_regime_370K_statistics.csv"
regime_summary.to_csv(regime_output, index=False)
print(f"Saved: {regime_output}")

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
def plot_regime_panel(ax, x_column, xlabel):
    for family_key in family_keys:
        family = regime_summary[
            regime_summary["family"] == family_key
        ].copy()

        family = family.sort_values(x_column)
        color = family_colors[family_key]

        x = family[x_column].to_numpy(dtype=float)
        y = family["mean_rg"].to_numpy(dtype=float)
        y_min = family["min_rg"].to_numpy(dtype=float)
        y_max = family["max_rg"].to_numpy(dtype=float)

        # Architecture envelope: full range across the 8 architecture means.
        ax.fill_between(
            x,
            y_min,
            y_max,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        # Composition-defined central trajectory.
        ax.plot(
            x,
            y,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=6.5,
            label=FAMILIES[family_key]["label"],
            zorder=3,
        )

    ax.axhline(
        RG_THRESHOLD,
        linestyle="--",
        linewidth=1.1,
        color="0.40",
        alpha=0.8,
        zorder=0,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mean $R_g$ (nm)")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -----------------------------------------------------------------------------
# A/B FIGURE
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.8, 5.4),
    sharey=True,
)

# A — direct composition axis
plot_regime_panel(
    axes[0],
    x_column="strong_fraction",
    xlabel=r"Fraction of higher-$\bar{\lambda}$ motif",
)

composition_ticks = [
    COMPOSITION_INFO[ratio]["strong_fraction"]
    for ratio in ["R1", "R2", "R3", "R4", "R5", "R6"]
]
axes[0].set_xticks(composition_ticks)
axes[0].set_xticklabels([f"{100*x:.0f}%" for x in composition_ticks])
axes[0].set_xlim(min(composition_ticks) - 0.045, max(composition_ticks) + 0.045)
axes[0].set_title("Composition")

# B — same observations, but x is the actual mean lambda per residue of the
# complete sequence composition.
plot_regime_panel(
    axes[1],
    x_column="lambda_sequence",
    xlabel=r"Sequence-average interaction parameter $\bar{\lambda}_{seq}$",
)
axes[1].set_title("Sequence-average interaction parameter")

# Shared y range, based only on the plotted 370 K architecture means.
ymin = min(regime_summary["min_rg"].min(), RG_THRESHOLD) - 0.12
ymax = regime_summary["max_rg"].max() + 0.12
axes[0].set_ylim(ymin, ymax)

# Theta annotation only once to avoid clutter.
axes[1].text(
    0.99,
    RG_THRESHOLD + 0.03,
    rf"$R_{{g,\Theta}}$ = {RG_THRESHOLD:.2f} nm",
    transform=axes[1].get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=9,
    color="0.40",
)

# Panel labels.
for label, ax in zip(["A", "B"], axes):
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

# One shared motif-pair legend.
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

fig.legend(
    handles,
    labels,
    title="Motif pair",
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=4,
)

fig.tight_layout()
fig.subplots_adjust(top=0.82)

output_base = OUTDIR / "composition_regime_AB_370K"

for extension in ["png", "pdf", "svg"]:
    output_path = Path(f"{output_base}.{extension}")
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_path}")

plt.close(fig)

print()
print("=" * 78)
print("DESIGN RULE 1 A/B FIGURE COMPLETE")
print("=" * 78)
print(f"Figure: {output_base}.pdf")
print(f"Data:   {regime_output}")
# =============================================================================
# SUPPLEMENTARY / ARCHITECTURE-RESOLVED COMPOSITION FIGURE
# =============================================================================
# Same Design Rule 1 relationship as the main composition plot, but resolved by
# architecture. Composition remains on the x-axis throughout the thesis.
#
# x = fraction of the higher-stickiness motif
# y = architecture-level mean Rg
# small coloured points = individual architecture trajectories
# large black-edged point = mean across the 8 architectures at that composition
# line = composition-level mean across architectures
# dashed horizontal line = ideal-chain reference Rg,Theta
#
# IMPORTANT: x positions are the actual numerical motif fractions, not equally
# spaced categorical positions.
# =============================================================================

ARCHITECTURE_PLOT_TEMPERATURE = 370

# Architecture display order and labels. The matching function below is deliberately
# tolerant of the naming variants used in the library CSV files.
ARCHITECTURE_ORDER = [
    "random shuffle",
    "alternating",
    "diblock",
    "small blocks",
    "medium blocks",
    "large blocks",
    "higher stickiness center",
    "higher stickiness ends",
]

ARCHITECTURE_DISPLAY = {
    "random shuffle": "Random shuffle",
    "alternating": "Alternating",
    "diblock": "Diblock",
    "small blocks": "Small blocks",
    "medium blocks": "Medium blocks",
    "large blocks": "Large blocks",
    "higher stickiness center": r"Higher $\lambda$ center",
    "higher stickiness ends": r"Higher $\lambda$ ends",
}


def canonical_architecture_name(value):
    """Map architecture_type strings from the library to eight thesis labels."""
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)

    if "random" in text or "shuffle" in text:
        return "random shuffle"
    if "altern" in text or compact == "even":
        return "alternating"
    if "diblock" in text:
        return "diblock"
    if "small" in text:
        return "small blocks"
    if "medium" in text:
        return "medium blocks"
    if "large" in text:
        return "large blocks"

    # Directional architectures. Existing analysis names can contain either
    # explicit higher/lower wording or motif-specific center/end labels.
    if "higher" in text and ("center" in text or "centre" in text):
        return "higher stickiness center"
    if "higher" in text and ("end" in text or "terminal" in text):
        return "higher stickiness ends"

    return str(value).strip()


arch_plot_data = mixed[
    mixed["temp"] == ARCHITECTURE_PLOT_TEMPERATURE
].copy()

arch_plot_data["architecture_plot"] = (
    arch_plot_data["architecture_type"].apply(canonical_architecture_name)
)

# Diagnostics: make unexpected architecture names visible rather than silently
# dropping them.
observed_architectures = sorted(
    arch_plot_data["architecture_plot"].dropna().unique().tolist()
)
print()
print("=" * 78)
print("ARCHITECTURE-RESOLVED COMPOSITION FIGURE")
print("=" * 78)
print("Detected architecture labels:")
for name in observed_architectures:
    print(f"  {name}")

# Use the matplotlib default cycle, consistent with the rest of this script.
arch_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
architecture_colors = {
    architecture: arch_color_cycle[i % len(arch_color_cycle)]
    for i, architecture in enumerate(ARCHITECTURE_ORDER)
}

# Any unexpected labels receive subsequent default-cycle colours so they remain
# visible in the figure and legend.
for name in observed_architectures:
    if name not in architecture_colors:
        architecture_colors[name] = arch_color_cycle[
            len(architecture_colors) % len(arch_color_cycle)
        ]

# Small deterministic x offsets prevent eight architecture means at the same
# composition from sitting directly on top of one another. Offsets are tiny in
# composition units and do not alter the numerical x-axis interpretation.
all_architectures_for_offsets = [
    name for name in ARCHITECTURE_ORDER if name in observed_architectures
] + [
    name for name in observed_architectures if name not in ARCHITECTURE_ORDER
]

if len(all_architectures_for_offsets) > 1:
    offset_values = np.linspace(-0.010, 0.010, len(all_architectures_for_offsets))
else:
    offset_values = np.array([0.0])

architecture_x_offsets = dict(zip(all_architectures_for_offsets, offset_values))

# Composition-level means are used only for the black trajectory through each
# panel. The coloured points remain the individual architecture-level means.
arch_composition_summary = (
    arch_plot_data
    .groupby(
        ["family", "family_label", "strong_fraction", "strong_percent"],
        as_index=False,
    )
    .agg(
        n_architectures=("id", "nunique"),
        mean_rg=("mean_rg", "mean"),
        sd_between_architectures=("mean_rg", "std"),
        min_rg=("mean_rg", "min"),
        max_rg=("mean_rg", "max"),
    )
    .sort_values(["family", "strong_fraction"])
)

bad_arch_counts = arch_composition_summary[
    arch_composition_summary["n_architectures"] != 8
]
if not bad_arch_counts.empty:
    print("WARNING: some plotted composition points do not contain 8 architectures:")
    print(
        bad_arch_counts[
            ["family", "strong_percent", "n_architectures"]
        ].to_string(index=False)
    )
else:
    print("PASS: every architecture-resolved composition contains 8 architectures.")

# Save exactly what is plotted.
arch_plot_csv = OUTDIR / "composition_architecture_resolved_370K.csv"
arch_plot_data[
    [
        "family",
        "family_label",
        "id",
        "numeric_id",
        "ratio",
        "architecture_type",
        "architecture_plot",
        "strong_fraction",
        "strong_percent",
        "mean_rg",
        "central90_rg",
        "compact_population",
    ]
].sort_values(
    ["family", "strong_fraction", "numeric_id"]
).to_csv(arch_plot_csv, index=False)
print(f"Saved: {arch_plot_csv}")

# -----------------------------------------------------------------------------
# 2 x 2 architecture-resolved figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12.6, 9.0),
    sharex=True,
    sharey=True,
)
axes = axes.ravel()

composition_ticks = np.array([
    COMPOSITION_INFO[ratio]["strong_fraction"]
    for ratio in ["R1", "R2", "R3", "R4", "R5", "R6"]
], dtype=float)

# Shared y limits make family-to-family differences directly comparable.
plot_ymin = min(arch_plot_data["mean_rg"].min(), RG_THRESHOLD) - 0.15
plot_ymax = max(arch_plot_data["mean_rg"].max(), RG_THRESHOLD) + 0.15

for ax, family_key in zip(axes, family_keys):
    family_points = arch_plot_data[
        arch_plot_data["family"] == family_key
    ].copy()

    family_summary = arch_composition_summary[
        arch_composition_summary["family"] == family_key
    ].sort_values("strong_fraction")

    # Architecture-level means.
    for architecture in all_architectures_for_offsets:
        subset = family_points[
            family_points["architecture_plot"] == architecture
        ].copy()

        if subset.empty:
            continue

        x = (
            subset["strong_fraction"].to_numpy(dtype=float)
            + architecture_x_offsets.get(architecture, 0.0)
        )
        y = subset["mean_rg"].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=34,
            color=architecture_colors[architecture],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )

    # Composition-level mean trajectory across architectures.
    x_mean = family_summary["strong_fraction"].to_numpy(dtype=float)
    y_mean = family_summary["mean_rg"].to_numpy(dtype=float)

    ax.plot(
        x_mean,
        y_mean,
        color="0.18",
        linewidth=2.0,
        zorder=2,
    )

    ax.scatter(
        x_mean,
        y_mean,
        s=72,
        facecolor="white",
        edgecolor="0.15",
        linewidth=1.5,
        zorder=5,
    )

    # Theta reference.
    ax.axhline(
        RG_THRESHOLD,
        linestyle="--",
        linewidth=1.1,
        color="0.42",
        alpha=0.85,
        zorder=0,
    )

    ax.set_title(
        FAMILIES[family_key]["label"],
        fontsize=13,
        pad=8,
    )

    ax.set_xlim(
        composition_ticks.min() - 0.055,
        composition_ticks.max() + 0.055,
    )
    ax.set_ylim(plot_ymin, plot_ymax)

    ax.set_xticks(composition_ticks)
    ax.set_xticklabels([
        f"{100*x:.0f}%" for x in composition_ticks
    ])

    ax.grid(axis="y", alpha=0.16, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Shared labels keep the four panels visually clean.
fig.supxlabel(
    "Fraction of higher-stickiness motif",
    fontsize=13,
    y=0.045,
)
fig.supylabel(
    r"Mean $R_g$ (nm)",
    fontsize=13,
    x=0.045,
)

# Theta annotation once, in the upper-right panel.
axes[1].text(
    0.985,
    RG_THRESHOLD + 0.035,
    rf"$R_{{g,\Theta}}={RG_THRESHOLD:.2f}$ nm",
    transform=axes[1].get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=9,
    color="0.42",
)

# Architecture legend. Add a separate symbol for the composition mean.
legend_handles = []
for architecture in all_architectures_for_offsets:
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=architecture_colors[architecture],
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7,
            label=ARCHITECTURE_DISPLAY.get(architecture, architecture),
        )
    )

legend_handles.append(
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="-",
        color="0.18",
        markerfacecolor="white",
        markeredgecolor="0.15",
        markeredgewidth=1.2,
        markersize=7,
        linewidth=1.8,
        label="Mean across architectures",
    )
)

fig.legend(
    handles=legend_handles,
    title="Architecture",
    frameon=False,
    loc="center left",
    bbox_to_anchor=(0.84, 0.52),
    fontsize=9.5,
    title_fontsize=10.5,
)

fig.tight_layout(rect=[0.06, 0.07, 0.83, 0.98])

arch_output_base = OUTDIR / "composition_architecture_resolved_vertical_370K"
for extension in ["png", "pdf", "svg"]:
    output_path = Path(f"{arch_output_base}.{extension}")
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {output_path}")

plt.close(fig)

print()
print("=" * 78)
print("ARCHITECTURE-RESOLVED VERTICAL COMPOSITION FIGURE COMPLETE")
print("=" * 78)
print(f"Figure: {arch_output_base}.pdf")
print(f"Data:   {arch_plot_csv}")
