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
    r"\Design_rule_1\composition_regime_plots_DR1ab_v7_300K_SUPPLEMENT"
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
# DESIGN RULE 1 — SUPPLEMENTARY FIGURES (300 K)
# New output directory: composition_regime_plots_DR1ab_v7_300K_SUPPLEMENT
# DR1a: architecture-resolved 2x2 composition panels
# DR1b: sequence-average stickiness does not uniquely determine mean Rg
# =============================================================================
from matplotlib.lines import Line2D

PLOT_TEMPERATURE = 300
# Same fixed CALVADOS-TEA stickiness values as in the 370 K figure.
# Verify parameter temperature dependence before interpreting the x-axis physically.
MOTIF_LAMBDA = {
    "VPGVG": 0.6850471057402,
    "IPAVG": 0.7430614622074,
    "VPGIG": 0.7986503898464,
    "VPAVG": 0.6294581781012,
}

# Deliberately large typography for figures reduced on an A4 thesis page.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

# Consistent chemistry colours in DR1a and DR1b, deliberately distinct
# from the eight architecture-category colours used in DR2.
FAMILY_COLORS = {
    "VGIP": "#273F73",  # deep navy
    "VGIG": "#B07D22",  # ochre
    "VAIP": "#6D4B80",  # muted plum
    "VAIG": "#28665C",  # deep forest teal
}
ARCHITECTURE_ORDER = [
    "alternating", "small blocks", "medium blocks", "large blocks",
    "diblock", "random shuffle", "higher stickiness center",
    "higher stickiness ends",
]
ARCHITECTURE_COLORS = {
    "alternating": "#5599D0",
    "small blocks": "#72AF4D",
    "medium blocks": "#8064A2",
    "large blocks": "#F07B2D",
    "diblock": "#8C564B",
    "random shuffle": "#A7A7A7",
    "higher stickiness center": "#E63946",
    "higher stickiness ends": "#16A9C3",
}
ARCHITECTURE_MARKERS = {
    "alternating": "o", "small blocks": "s", "medium blocks": "^",
    "large blocks": "v", "diblock": "D", "random shuffle": "P",
    "higher stickiness center": "<", "higher stickiness ends": ">",
}
ARCHITECTURE_LABELS = {
    "alternating": "Alternating", "small blocks": "Small blocks",
    "medium blocks": "Medium blocks", "large blocks": "Large blocks",
    "diblock": "Diblock", "random shuffle": "Random shuffle",
    "higher stickiness center": r"Higher-$\lambda$ center",
    "higher stickiness ends": r"Higher-$\lambda$ ends",
}

def canonical_architecture_name(value):
    """Map motif-specific library names to the intended architecture categories.

    'Higher' refers to the higher-lambda motif of the current motif pair.
    For example, VGcenter_IPends means higher-stickiness motifs at ends,
    while IPcenter_VGends means higher-stickiness motifs at center.
    """
    name = str(value).strip()
    text = name.lower()
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

    # Explicit higher-stickiness architecture names, if present.
    if "higher" in text:
        if "center" in text or "centre" in text:
            return "higher stickiness center"
        if "end" in text or "terminal" in text:
            return "higher stickiness ends"

    # The library uses IP/IG as the stronger motifs in all four families.
    # These are *motif-specific* names, not generic higher/lower names.
    if re.match(r"^(ip|ig)center_", text):
        return "higher stickiness center"
    if re.match(r"^(ip|ig)endcaps$", text):
        return "higher stickiness ends"
    if re.match(r"^(vg|va)center_(ip|ig)ends$", text):
        return "higher stickiness ends"

    return name

plot_data = mixed.loc[mixed["temp"] == PLOT_TEMPERATURE].copy()
plot_data["architecture_plot"] = plot_data["architecture_type"].apply(canonical_architecture_name)
plot_data["lambda_weak"] = plot_data["weak_motif"].map(MOTIF_LAMBDA)
plot_data["lambda_strong"] = plot_data["strong_motif"].map(MOTIF_LAMBDA)
if plot_data[["lambda_weak", "lambda_strong"]].isna().any().any():
    raise ValueError("Missing motif stickiness values in MOTIF_LAMBDA.")
plot_data["lambda_sequence"] = (
    (1 - plot_data["strong_fraction"]) * plot_data["lambda_weak"]
    + plot_data["strong_fraction"] * plot_data["lambda_strong"]
)

observed = set(plot_data["architecture_plot"].unique())
unknown = observed - set(ARCHITECTURE_ORDER)
if unknown:
    raise ValueError(f"Unrecognized architecture types: {sorted(unknown)}")

summary = (
    plot_data.groupby(
        ["family", "family_label", "strong_fraction", "strong_percent",
         "lambda_sequence"], as_index=False
    ).agg(
        n_architectures=("id", "nunique"),
        mean_rg=("mean_rg", "mean"),
        min_rg=("mean_rg", "min"),
        max_rg=("mean_rg", "max"),
        sd_between_architectures=("mean_rg", "std"),
    ).sort_values(["family", "strong_fraction"])
)
bad = summary.loc[summary["n_architectures"] != 8]
if not bad.empty:
    raise ValueError(
        "Expected exactly eight architectures per composition:\n"
        + bad[["family", "strong_percent", "n_architectures"]].to_string(index=False)
    )

summary.to_csv(OUTDIR / "DR1_composition_summary_300K.csv", index=False)
plot_data.to_csv(OUTDIR / "DR1_architecture_level_300K.csv", index=False)

def style_axis(ax):
    ax.grid(axis="y", alpha=0.18, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=5, width=1.1, pad=6)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

def save_all(fig, basename):
    for ext in ("pdf", "png", "svg"):
        dest = OUTDIR / f"{basename}.{ext}"
        fig.savefig(dest, dpi=350, bbox_inches="tight")
        print("Saved:", dest)
    plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE DR1a — architecture-spread boxes for each composition, four panels
# Each box summarizes the eight *architecture mean Rg* values, not frames.
# The whiskers span the minimum and maximum architecture means.
# All eight points share the same chemistry colour: identity is irrelevant.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16.2, 12.4), sharex=True, sharey=True)
axes = axes.ravel()
family_keys = list(FAMILIES)
composition_ticks = np.array([
    COMPOSITION_INFO[k]["strong_fraction"]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6")
])
ymin = min(plot_data["mean_rg"].min(), RG_THRESHOLD) - .20
ymax = max(plot_data["mean_rg"].max(), RG_THRESHOLD) + .20

for i, (ax, key) in enumerate(zip(axes, family_keys)):
    points = plot_data.loc[plot_data["family"] == key]
    comp = summary.loc[summary["family"] == key].sort_values("strong_fraction")
    color = FAMILY_COLORS[key]
    groups = [
        points.loc[np.isclose(points["strong_fraction"], x), "mean_rg"].to_numpy()
        for x in composition_ticks
    ]
    if any(len(group) != 8 for group in groups):
        raise ValueError(f"Expected 8 architecture means at every composition for {key}")

    # Full-range colored rectangle = minimum to maximum of the eight
    # architecture-level mean Rg values (NOT a trajectory-frame distribution).
    # A short dark horizontal bar marks the median of the eight architecture means.
    # The connected open circles mark the arithmetic composition mean.
    from matplotlib.patches import Rectangle
    box_width = .039
    for x, group in zip(composition_ticks, groups):
        low, high = float(np.min(group)), float(np.max(group))
        median = float(np.median(group))
        ax.add_patch(Rectangle(
            (x - box_width / 2, low), box_width, high - low,
            facecolor=color, edgecolor=color, alpha=.23,
            linewidth=1.8, zorder=3,
        ))
        ax.plot([x - box_width / 2, x + box_width / 2],
                [median, median], color="#242424", lw=2.1, zorder=5)
        # Identical subtle dots; the identity of each architecture is omitted.
        jitter = np.linspace(-.011, .011, len(group))
        ax.scatter(x + jitter, group, s=32, color=color, alpha=.65,
                   edgecolors="white", linewidths=.45, zorder=4)

    # Composition mean: chemistry-colored connecting line and solid circles.
    ax.plot(comp["strong_fraction"], comp["mean_rg"],
            color=color, linewidth=2.8, zorder=6)
    ax.scatter(comp["strong_fraction"], comp["mean_rg"],
               s=88, facecolors=color, edgecolors="none",
               linewidths=0, zorder=7)
    ax.axhline(RG_THRESHOLD, color="0.47", ls="--", lw=1.3, zorder=1)
    ax.set_title(FAMILIES[key]["label"], pad=16, color=color)
    ax.set_xlim(composition_ticks.min() - .055, composition_ticks.max() + .055)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(composition_ticks)
    ax.set_xticklabels([f"{x:.0%}" for x in composition_ticks])
    style_axis(ax)
    ax.text(-.075, 1.08, "ABCD"[i], transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top")

fig.supxlabel("Fraction of higher-stickiness motif", fontsize=20,
              fontweight="bold", y=.025)
fig.supylabel(r"Mean $R_g$ (nm)", fontsize=20, fontweight="bold", x=.025)
fig.subplots_adjust(left=.095, right=.985, top=.935, bottom=.11,
                    hspace=.30, wspace=.13)
save_all(fig, "DR1a_composition_full_range_solid_means_300K")

# ---------------------------------------------------------------------------
# FIGURE DR1b — mean Rg versus sequence-average stickiness
# A different x-axis for each chemistry; identical lambda values do not
# automatically imply identical Rg. Points are observed composition means.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 8.0))
for key in family_keys:
    comp = summary.loc[summary["family"] == key].sort_values("lambda_sequence")
    x = comp["lambda_sequence"].to_numpy()
    y = comp["mean_rg"].to_numpy()
    ax.fill_between(
        x, comp["min_rg"].to_numpy(), comp["max_rg"].to_numpy(),
        color=FAMILY_COLORS[key], alpha=.13, lw=0
    )
    ax.plot(x, y, marker="o", ms=10, lw=3.0,
            color=FAMILY_COLORS[key], label=FAMILIES[key]["label"])
ax.axhline(RG_THRESHOLD, color="0.43", ls="--", lw=1.4)
ax.set_xlabel(r"Sequence-average stickiness $\bar{\lambda}_{\mathrm{seq}}$",
              labelpad=14)
ax.set_ylabel(r"Mean $R_g$ (nm)", labelpad=13)
ax.set_ylim(ymin, ymax)
style_axis(ax)
ax.legend(loc="best", frameon=False, prop={"size": 15, "weight": "bold"})
fig.subplots_adjust(left=.12, right=.98, bottom=.17, top=.97)
save_all(fig, "DR1b_sequence_average_stickiness_300K")

print("\nDR1a and DR1b thesis figures completed.")
print("Output directory:", OUTDIR)
