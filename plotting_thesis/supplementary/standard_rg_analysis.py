#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
summarize_all_rg_values.py

Standard CALVADOS Rg + two-state analysis.

Input:
    one all_rg_values.csv
or:
    a root directory containing multiple all_rg_values.csv files

Output:
    one row per sequence × temperature × source file.

Calculates:
    - full Rg distribution statistics
    - IQR / central 50% Rg range
    - central 80% and 90% Rg ranges
    - compact/expanded populations
    - compact-state and expanded-state Rg
    - collapse and escape transitions
    - collapse and escape rates
    - compact and expanded dwell times

State definitions are taken DIRECTLY from the `collapsed` column
already present in all_rg_values.csv.

Rates:
    k_collapse = N(E -> C) / total expanded time
    k_escape   = N(C -> E) / total collapsed time
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================

def extract_ratio(sequence_id):

    match = re.search(
        r"(?:^|_)R(\d+)(?:_|$)",
        str(sequence_id),
        flags=re.IGNORECASE,
    )

    if match is None:
        return np.nan

    return f"R{match.group(1)}"


def extract_numeric_id(sequence_id):

    match = re.match(
        r"(\d+)",
        str(sequence_id),
    )

    if match is None:
        return np.nan

    return int(match.group(1))


def first_value(df, column):

    if column not in df.columns:
        return np.nan

    values = df[column].dropna()

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


def sample_std(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    return np.std(
        values,
        ddof=1,
    )


def distribution_summary(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "iqr": np.nan,
            "range80": np.nan,
            "range90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    p05 = np.percentile(values, 5)
    p10 = np.percentile(values, 10)
    p25 = np.percentile(values, 25)
    p50 = np.percentile(values, 50)
    p75 = np.percentile(values, 75)
    p90 = np.percentile(values, 90)
    p95 = np.percentile(values, 95)

    return {
        "n": len(values),

        "mean":
            np.mean(values),

        "std":
            sample_std(values),

        "median":
            p50,

        "p05":
            p05,

        "p10":
            p10,

        "p25":
            p25,

        "p50":
            p50,

        "p75":
            p75,

        "p90":
            p90,

        "p95":
            p95,

        "iqr":
            p75 - p25,

        "range80":
            p90 - p10,

        "range90":
            p95 - p05,

        "min":
            np.min(values),

        "max":
            np.max(values),
    }


# ============================================================
# FRAME TIME
# ============================================================

def determine_frame_spacing_ns(df):
    """
    Determine physical time represented by one row.

    Prefer time_ns because it already contains the actual
    analysed-frame timestamps.

    Fallback:
        frame_dt_ns * stride
    """

    if "time_ns" in df.columns:

        times = pd.to_numeric(
            df["time_ns"],
            errors="coerce",
        ).dropna().to_numpy()

        if len(times) > 1:

            differences = np.diff(
                times
            )

            differences = differences[
                differences > 0
            ]

            if len(differences) > 0:

                return float(
                    np.median(
                        differences
                    )
                )

    frame_dt = first_value(
        df,
        "frame_dt_ns",
    )

    stride = first_value(
        df,
        "stride",
    )

    try:
        frame_dt = float(frame_dt)
    except (TypeError, ValueError):
        frame_dt = np.nan

    try:
        stride = float(stride)
    except (TypeError, ValueError):
        stride = 1.0

    if np.isfinite(frame_dt):

        return (
            frame_dt
            * stride
        )

    return np.nan


# ============================================================
# DWELL EVENTS
# ============================================================

def calculate_dwell_times(
    states,
    frame_spacing_ns,
):
    """
    Calculate contiguous state dwell lengths.

    states:
        True  = collapsed
        False = expanded

    Returns:
        collapsed_dwells
        expanded_dwells

    Each dwell duration is:
        number of frames in dwell × frame spacing
    """

    states = np.asarray(
        states,
        dtype=bool,
    )

    if len(states) == 0:

        return (
            np.array([]),
            np.array([]),
        )

    collapsed_dwells = []
    expanded_dwells = []

    current_state = states[0]
    current_length = 1

    for state in states[1:]:

        if state == current_state:

            current_length += 1

        else:

            dwell_time = (
                current_length
                * frame_spacing_ns
            )

            if current_state:

                collapsed_dwells.append(
                    dwell_time
                )

            else:

                expanded_dwells.append(
                    dwell_time
                )

            current_state = state
            current_length = 1

    # Final dwell

    dwell_time = (
        current_length
        * frame_spacing_ns
    )

    if current_state:

        collapsed_dwells.append(
            dwell_time
        )

    else:

        expanded_dwells.append(
            dwell_time
        )

    return (
        np.asarray(
            collapsed_dwells,
            dtype=float,
        ),
        np.asarray(
            expanded_dwells,
            dtype=float,
        ),
    )


# ============================================================
# DWELL SUMMARY
# ============================================================

def dwell_summary(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "max": np.nan,
        }

    return {
        "n":
            len(values),

        "mean":
            np.mean(values),

        "median":
            np.median(values),

        "std":
            sample_std(values),

        "max":
            np.max(values),
    }


# ============================================================
# ANALYSE ONE SEQUENCE
# ============================================================

def analyse_sequence(
    df,
    source_file,
):

    # Sort trajectory chronologically

    if "analysed_index" in df.columns:

        df = df.sort_values(
            "analysed_index"
        ).copy()

    elif "time_ns" in df.columns:

        df = df.sort_values(
            "time_ns"
        ).copy()

    else:

        df = df.copy()

    sequence_id = str(
        df["id"].iloc[0]
    )

    temperature = str(
        df["temp"].iloc[0]
    )

    # --------------------------------------------------------
    # Rg
    # --------------------------------------------------------

    rg = pd.to_numeric(
        df["rg_nm"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = np.isfinite(rg)

    df = df.loc[
        valid
    ].copy()

    rg = rg[
        valid
    ]

    if len(rg) == 0:
        return None

    full = distribution_summary(
        rg
    )

    # --------------------------------------------------------
    # State assignment
    # --------------------------------------------------------

    if "collapsed" not in df.columns:

        raise ValueError(
            "'collapsed' column is required "
            "for state and kinetic analysis."
        )

    collapsed = (
        df["collapsed"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if collapsed.isna().any():

        raise ValueError(
            f"Could not interpret all values in "
            f"'collapsed' for {sequence_id}"
        )

    states = collapsed.to_numpy(
        dtype=bool
    )

    # --------------------------------------------------------
    # State-specific Rg
    # --------------------------------------------------------

    rg_collapsed = rg[
        states
    ]

    rg_expanded = rg[
        ~states
    ]

    compact_stats = distribution_summary(
        rg_collapsed
    )

    expanded_stats = distribution_summary(
        rg_expanded
    )

    # --------------------------------------------------------
    # Frame spacing
    # --------------------------------------------------------

    frame_spacing_ns = (
        determine_frame_spacing_ns(
            df
        )
    )

    if not np.isfinite(
        frame_spacing_ns
    ):

        raise ValueError(
            f"Could not determine frame spacing "
            f"for {sequence_id}"
        )

    # --------------------------------------------------------
    # Transitions
    # --------------------------------------------------------

    previous = states[:-1]
    current = states[1:]

    collapse_transitions = int(
        np.sum(
            (~previous)
            &
            current
        )
    )

    escape_transitions = int(
        np.sum(
            previous
            &
            (~current)
        )
    )

    # --------------------------------------------------------
    # Total state time
    # --------------------------------------------------------

    total_collapsed_time_ns = (
        np.sum(states)
        *
        frame_spacing_ns
    )

    total_expanded_time_ns = (
        np.sum(~states)
        *
        frame_spacing_ns
    )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    if total_expanded_time_ns > 0:

        k_collapse = (
            collapse_transitions
            /
            total_expanded_time_ns
        )

    else:

        k_collapse = np.nan

    if total_collapsed_time_ns > 0:

        k_escape = (
            escape_transitions
            /
            total_collapsed_time_ns
        )

    else:

        k_escape = np.nan

    # --------------------------------------------------------
    # Dwell times
    # --------------------------------------------------------

    (
        collapsed_dwells,
        expanded_dwells,
    ) = calculate_dwell_times(
        states,
        frame_spacing_ns,
    )

    collapsed_dwell_stats = (
        dwell_summary(
            collapsed_dwells
        )
    )

    expanded_dwell_stats = (
        dwell_summary(
            expanded_dwells
        )
    )

    # ========================================================
    # OUTPUT ROW
    # ========================================================

    row = {

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "id":
            sequence_id,

        "numeric_id":
            extract_numeric_id(
                sequence_id
            ),

        "temp":
            temperature,

        "group":
            first_value(
                df,
                "group",
            ),

        "ratio":
            extract_ratio(
                sequence_id
            ),

        "architecture_type":
            first_value(
                df,
                "architecture_type",
            ),

        "source_file":
            str(source_file),

        # ----------------------------------------------------
        # Analysis settings
        # ----------------------------------------------------

        "threshold_nm":
            first_value(
                df,
                "threshold_nm",
            ),

        "stride":
            first_value(
                df,
                "stride",
            ),

        "frame_dt_ns":
            first_value(
                df,
                "frame_dt_ns",
            ),

        "analysed_frame_spacing_ns":
            frame_spacing_ns,

        # ----------------------------------------------------
        # Full distribution
        # ----------------------------------------------------

        "n_frames":
            len(rg),

        "mean_rg_nm":
            full["mean"],

        "std_rg_nm":
            full["std"],

        "median_rg_nm":
            full["median"],

        "min_rg_nm":
            full["min"],

        "max_rg_nm":
            full["max"],

        "rg_p05_nm":
            full["p05"],

        "rg_p10_nm":
            full["p10"],

        "rg_p25_nm":
            full["p25"],

        "rg_p50_nm":
            full["p50"],

        "rg_p75_nm":
            full["p75"],

        "rg_p90_nm":
            full["p90"],

        "rg_p95_nm":
            full["p95"],

        # PRIMARY breadth metric
        "iqr_rg_nm":
            full["iqr"],

        "rg_range80_nm":
            full["range80"],

        # SECONDARY breadth metric
        "rg_range90_nm":
            full["range90"],

        # ----------------------------------------------------
        # State populations
        # ----------------------------------------------------

        "n_collapsed_frames":
            int(np.sum(states)),

        "n_expanded_frames":
            int(np.sum(~states)),

        "fraction_collapsed":
            np.mean(states),

        "fraction_expanded":
            np.mean(~states),

        # ----------------------------------------------------
        # Compact-state Rg
        # ----------------------------------------------------

        "mean_rg_collapsed_nm":
            compact_stats["mean"],

        "std_rg_collapsed_nm":
            compact_stats["std"],

        "median_rg_collapsed_nm":
            compact_stats["median"],

        "rg_collapsed_p25_nm":
            compact_stats["p25"],

        "rg_collapsed_p75_nm":
            compact_stats["p75"],

        "iqr_rg_collapsed_nm":
            compact_stats["iqr"],

        # ----------------------------------------------------
        # Expanded-state Rg
        # ----------------------------------------------------

        "mean_rg_expanded_nm":
            expanded_stats["mean"],

        "std_rg_expanded_nm":
            expanded_stats["std"],

        "median_rg_expanded_nm":
            expanded_stats["median"],

        "rg_expanded_p25_nm":
            expanded_stats["p25"],

        "rg_expanded_p75_nm":
            expanded_stats["p75"],

        "iqr_rg_expanded_nm":
            expanded_stats["iqr"],

        # ----------------------------------------------------
        # Transitions
        # ----------------------------------------------------

        "n_collapse_transitions":
            collapse_transitions,

        "n_escape_transitions":
            escape_transitions,

        # ----------------------------------------------------
        # State times
        # ----------------------------------------------------

        "total_collapsed_time_ns":
            total_collapsed_time_ns,

        "total_expanded_time_ns":
            total_expanded_time_ns,

        # ----------------------------------------------------
        # Kinetics
        # ----------------------------------------------------

        "k_collapse_per_ns":
            k_collapse,

        "k_escape_per_ns":
            k_escape,

        # ----------------------------------------------------
        # Collapsed dwell
        # ----------------------------------------------------

        "n_collapsed_dwells":
            collapsed_dwell_stats["n"],

        "mean_collapsed_dwell_ns":
            collapsed_dwell_stats["mean"],

        "median_collapsed_dwell_ns":
            collapsed_dwell_stats["median"],

        "std_collapsed_dwell_ns":
            collapsed_dwell_stats["std"],

        "max_collapsed_dwell_ns":
            collapsed_dwell_stats["max"],

        # ----------------------------------------------------
        # Expanded dwell
        # ----------------------------------------------------

        "n_expanded_dwells":
            expanded_dwell_stats["n"],

        "mean_expanded_dwell_ns":
            expanded_dwell_stats["mean"],

        "median_expanded_dwell_ns":
            expanded_dwell_stats["median"],

        "std_expanded_dwell_ns":
            expanded_dwell_stats["std"],

        "max_expanded_dwell_ns":
            expanded_dwell_stats["max"],
    }

    return row


# ============================================================
# FIND FILES
# ============================================================

def find_input_files(
    input_file,
    root,
):

    files = []

    if input_file is not None:

        file = Path(
            input_file
        )

        if not file.exists():

            raise FileNotFoundError(
                file
            )

        files.append(
            file
        )

    if root is not None:

        root = Path(
            root
        )

        if not root.exists():

            raise FileNotFoundError(
                root
            )

        files.extend(
            sorted(
                root.rglob(
                    "all_rg_values.csv"
                )
            )
        )

    # Remove duplicates

    unique = []

    seen = set()

    for file in files:

        key = str(
            file.resolve()
        )

        if key not in seen:

            seen.add(key)
            unique.append(file)

    if len(unique) == 0:

        raise RuntimeError(
            "No all_rg_values.csv files found."
        )

    return unique


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Summarize CALVADOS all_rg_values.csv "
            "files."
        )
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Single all_rg_values.csv file."
        ),
    )

    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Root directory. All files named "
            "all_rg_values.csv are found recursively."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV.",
    )

    parser.add_argument(
        "--temperature",
        default=None,
        help=(
            "Optional temperature filter, "
            "for example 370K."
        ),
    )

    args = parser.parse_args()

    if (
        args.input is None
        and
        args.root is None
    ):

        parser.error(
            "Provide --input or --root."
        )

    files = find_input_files(
        args.input,
        args.root,
    )

    print()
    print("=" * 72)
    print("CALVADOS RG + STATE SUMMARY")
    print("=" * 72)

    print(
        f"Found {len(files)} "
        f"all_rg_values.csv file(s)"
    )

    rows = []

    for file in files:

        print()
        print(
            f"Reading: {file}"
        )

        df = pd.read_csv(
            file
        )

        required = [
            "id",
            "temp",
            "rg_nm",
            "collapsed",
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            print(
                f"  SKIP: missing {missing}"
            )

            continue

        # ----------------------------------------------------
        # Optional temperature filter
        # ----------------------------------------------------

        if args.temperature is not None:

            target = (
                str(args.temperature)
                .strip()
                .upper()
            )

            temp = (
                df["temp"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            df = df[
                temp == target
            ].copy()

        if df.empty:

            print(
                "  No rows after filtering."
            )

            continue

        # ----------------------------------------------------
        # One analysis per sequence × temperature
        # ----------------------------------------------------

        grouped = df.groupby(
            [
                "id",
                "temp",
            ],
            sort=False,
        )

        print(
            f"  Simulations: "
            f"{len(grouped)}"
        )

        for (
            sequence_id,
            temperature,
        ), sub in grouped:

            result = analyse_sequence(
                sub,
                file,
            )

            if result is not None:

                rows.append(
                    result
                )

    # ========================================================
    # COMBINE
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    if results.empty:

        raise RuntimeError(
            "No simulations were successfully analysed."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        [
            "numeric_id",
            "temp",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Simulations analysed: "
        f"{len(results)}"
    )

    print(
        f"Output:\n{output}"
    )

    print()
    print(
        "Key output metrics:"
    )

    print(
        "  iqr_rg_nm"
    )

    print(
        "  rg_range90_nm"
    )

    print(
        "  fraction_collapsed"
    )

    print(
        "  mean_rg_collapsed_nm"
    )

    print(
        "  k_collapse_per_ns"
    )

    print(
        "  k_escape_per_ns"
    )

    print(
        "  mean_collapsed_dwell_ns"
    )

    print(
        "  mean_expanded_dwell_ns"
    )


if __name__ == "__main__":
    main()