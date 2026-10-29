#!/usr/bin/env python3
"""
Analyse collapsed-state dwell distributions directly from frame-by-frame
all_rg_values.csv files.

Key features
------------
1. Reads one all_rg_values.csv per independent replicate.
2. Reconstructs BOTH:
      a) the original single-threshold state definition
      b) a hysteresis state definition that suppresses rapid threshold recrossing
3. Excludes left/right-censored dwell events by default.
4. Uses linear dwell-time bins aligned to the saved-frame resolution instead of
   logarithmic histogram bins.
5. Generates matched-architecture count and normalized-shape comparisons.
6. Calculates sustained-collapse metrics (>1 ns by default).
7. Optionally correlates hysteresis dwell metrics with replicate-mean Rg370.
8. Exports all derived dwell events and summaries.

Recommended VGIPKG defaults
---------------------------
Original collapse threshold: 4.30 nm
Hysteresis collapse-entry threshold: 4.25 nm
Hysteresis escape threshold: 4.35 nm
Saved-frame spacing: 0.01 ns

Interpretation of hysteresis
----------------------------
Expanded -> collapsed only when Rg <= collapse_enter_nm
Collapsed -> expanded only when Rg >= escape_exit_nm

While Rg lies between the two thresholds, the previous state is retained.
This suppresses rapid C/E/C/E recrossings caused by fluctuations around 4.30 nm.
"""

import argparse
import io
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


# ============================================================
# SETTINGS / ARCHITECTURE CLASSIFICATION
# ============================================================

TEMPERATURE = 370

KG_CLASS = {
    833: "control",
    834: "control",
    835: "control",

    836: "peripheral",
    837: "internal",
    838: "internal",
    839: "internal",
    840: "internal",

    841: "peripheral",
    842: "internal",
    843: "internal",

    844: "internal",
    845: "peripheral",
    846: "internal",

    847: "peripheral",
    848: "internal",
    849: "internal",

    850: "peripheral",

    851: "peripheral",
    852: "internal",
    853: "peripheral",

    854: "peripheral",
    855: "internal",
    856: "peripheral",
    857: "peripheral",
    858: "peripheral",

    859: "internal",
    860: "internal",
    861: "internal",
    862: "internal",
}

MATCHED_GROUPS = [
    [836, 837],
    [841, 842, 843],
    [844, 845, 846],
    [847, 848, 849],
    [850, 862],
    [851, 852, 853],
    [854, 855],
    [841, 856],
]


# ============================================================
# BASIC HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    """Return the first available column from candidates."""
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        match = lower_map.get(candidate.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            "Could not find any of these columns:\n"
            f"{candidates}\n\n"
            "Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


def detect_rg_columns(df):
    """
    Detect columns in all_rg_values.csv.

    Supports small naming differences between analysis-script versions.
    """
    return {
        "id": find_column(
            df,
            [
                "id",
                "architecture_id",
                "_architecture_id",
                "architecture",
                "name",
            ],
        ),
        "temp": find_column(
            df,
            [
                "temp",
                "temperature",
                "temperature_K",
                "temp_K",
            ],
        ),
        "rg": find_column(
            df,
            [
                "rg_nm",
                "Rg_nm",
                "mean_rg_nm",
                "radius_of_gyration_nm",
                "rg",
                "Rg",
            ],
        ),
        "time": find_column(
            df,
            [
                "time_ns",
                "simulation_time_ns",
                "t_ns",
                "time",
            ],
            required=False,
        ),
        "frame": find_column(
            df,
            [
                "frame",
                "frame_index",
                "global_frame",
                "frame_number",
            ],
            required=False,
        ),
    }


def extract_numeric_id(value):
    """833_VGIPKG_R2_random_shuffle -> 833"""
    text = str(value).strip()

    match = re.search(r"(\d{3})", text)

    if match is None:
        raise ValueError(
            f"Could not extract a 3-digit architecture ID from: {value}"
        )

    return int(match.group(1))


def temperature_to_numeric(series):
    """Convert values such as 370, 370.0, '370K', '370 K' to numeric."""
    text = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return pd.to_numeric(
        text.str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        ),
        errors="coerce"
    )


def sanitize_filename(name):
    """Make a filename conservative for Windows."""
    name = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        str(name)
    )
    return name.rstrip(" .")


def safe_savefig(fig, filepath, dpi=None, bbox_inches="tight"):
    """
    Save through a short temporary file, then copy to destination.

    This avoids occasional Pillow/Windows problems with direct matplotlib
    writes to nested paths.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filepath = (
        filepath.parent
        / sanitize_filename(filepath.name)
    )

    suffix = filepath.suffix.lower()

    if suffix == ".png":
        fmt = "png"
    elif suffix == ".pdf":
        fmt = "pdf"
    elif suffix == ".svg":
        fmt = "svg"
    else:
        fmt = suffix.lstrip(".")

    buffer = io.BytesIO()

    kwargs = {
        "format": fmt,
        "bbox_inches": bbox_inches,
    }

    if dpi is not None:
        kwargs["dpi"] = dpi

    fig.savefig(
        buffer,
        **kwargs
    )

    buffer.seek(0)

    fd, temp_name = tempfile.mkstemp(
        suffix=suffix,
        prefix="mepfig_"
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        with open(temp_path, "wb") as handle:
            handle.write(
                buffer.getvalue()
            )

        shutil.copyfile(
            str(temp_path),
            str(filepath)
        )

    finally:
        buffer.close()

        try:
            temp_path.unlink()
        except OSError:
            pass


# ============================================================
# LOAD FRAME-BY-FRAME Rg DATA
# ============================================================

def load_rg_files(
    files,
    temperature=TEMPERATURE,
    fallback_frame_time_ns=0.01,
):
    """
    Load one all_rg_values.csv per independent replicate.

    Returns standardized columns:
        architecture_id
        replicate
        rg_nm
        time_ns
        frame_index
    """
    all_data = []

    for replicate, file in enumerate(
        files,
        start=1
    ):
        file = Path(file)

        print()
        print(
            f"Loading replicate {replicate}:"
        )
        print(file)

        df = pd.read_csv(file)

        cols = detect_rg_columns(df)

        print("Detected columns:")
        for key, value in cols.items():
            print(
                f"  {key}: {value}"
            )

        df["architecture_id"] = (
            df[cols["id"]]
            .apply(extract_numeric_id)
        )

        temp_numeric = temperature_to_numeric(
            df[cols["temp"]]
        )

        print(
            "Temperatures found:",
            sorted(
                temp_numeric
                .dropna()
                .unique()
                .tolist()
            )
        )

        df = df[
            np.isclose(
                temp_numeric,
                temperature
            )
        ].copy()

        print(
            f"Frames after {temperature} K filter:",
            len(df)
        )

        df["rg_nm"] = pd.to_numeric(
            df[cols["rg"]],
            errors="coerce"
        )

        df = df[
            np.isfinite(
                df["rg_nm"]
            )
        ].copy()

        # ---------------------------
        # Frame / time standardization
        # ---------------------------
        if cols["frame"] is not None:
            df["frame_index"] = pd.to_numeric(
                df[cols["frame"]],
                errors="coerce"
            )
        else:
            df["frame_index"] = (
                df.groupby("architecture_id")
                .cumcount()
            )

        if cols["time"] is not None:
            df["time_ns"] = pd.to_numeric(
                df[cols["time"]],
                errors="coerce"
            )
        else:
            df["time_ns"] = (
                df["frame_index"]
                * fallback_frame_time_ns
            )

        df["replicate"] = replicate

        keep = df[
            [
                "architecture_id",
                "replicate",
                "frame_index",
                "time_ns",
                "rg_nm",
            ]
        ].copy()

        all_data.append(keep)

    if not all_data:
        raise ValueError(
            "No all_rg_values.csv inputs were loaded."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    combined = (
        combined
        .sort_values(
            [
                "architecture_id",
                "replicate",
                "time_ns",
                "frame_index",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


# ============================================================
# FRAME-TIME ESTIMATION
# ============================================================

def estimate_frame_dt_ns(
    sub,
    fallback_frame_time_ns=0.01
):
    """
    Estimate saved-frame spacing from time_ns.

    Falls back to --frame-time-ns if a reliable time step cannot be inferred.
    """
    times = (
        pd.to_numeric(
            sub["time_ns"],
            errors="coerce"
        )
        .dropna()
        .to_numpy()
    )

    if len(times) >= 2:
        diffs = np.diff(times)
        diffs = diffs[
            np.isfinite(diffs)
            & (diffs > 0)
        ]

        if len(diffs) > 0:
            return float(
                np.median(diffs)
            )

    return float(
        fallback_frame_time_ns
    )


# ============================================================
# STATE ASSIGNMENT
# ============================================================

def assign_single_threshold_states(
    rg_values,
    threshold_nm=4.30
):
    """
    Original two-state definition.

    collapsed: Rg < threshold
    expanded:  Rg >= threshold
    """
    rg_values = np.asarray(
        rg_values,
        dtype=float
    )

    return np.where(
        rg_values < threshold_nm,
        "collapsed",
        "expanded"
    )


def assign_hysteresis_states(
    rg_values,
    collapse_enter_nm=4.25,
    escape_exit_nm=4.35,
):
    """
    Hysteresis state definition.

    Expanded -> collapsed only when Rg <= collapse_enter_nm.
    Collapsed -> expanded only when Rg >= escape_exit_nm.
    Between thresholds, retain the previous state.

    For a trajectory beginning inside the hysteresis band, the midpoint is
    used only to initialize the first state.
    """
    if collapse_enter_nm >= escape_exit_nm:
        raise ValueError(
            "collapse_enter_nm must be smaller than escape_exit_nm."
        )

    rg_values = np.asarray(
        rg_values,
        dtype=float
    )

    if len(rg_values) == 0:
        return np.array(
            [],
            dtype=object
        )

    midpoint = (
        collapse_enter_nm
        + escape_exit_nm
    ) / 2.0

    states = np.empty(
        len(rg_values),
        dtype=object
    )

    first = rg_values[0]

    if first <= collapse_enter_nm:
        current = "collapsed"
    elif first >= escape_exit_nm:
        current = "expanded"
    else:
        current = (
            "collapsed"
            if first < midpoint
            else "expanded"
        )

    states[0] = current

    for i in range(
        1,
        len(rg_values)
    ):
        rg = rg_values[i]

        if current == "expanded":
            if rg <= collapse_enter_nm:
                current = "collapsed"

        else:  # currently collapsed
            if rg >= escape_exit_nm:
                current = "expanded"

        states[i] = current

    return states


# ============================================================
# DWELL EXTRACTION
# ============================================================

def states_to_dwells(
    states,
    architecture_id,
    replicate,
    frame_dt_ns,
    state_method,
):
    """
    Convert a sequence of discrete states into dwell events.

    Dwell duration is number_of_saved_frames * frame_dt_ns.

    First event is left-censored.
    Last event is right-censored.
    """
    states = np.asarray(
        states,
        dtype=object
    )

    if len(states) == 0:
        return []

    rows = []

    start = 0
    event_index = 0

    for i in range(
        1,
        len(states) + 1
    ):
        at_end = (
            i == len(states)
        )

        changed = (
            not at_end
            and states[i] != states[start]
        )

        if at_end or changed:
            n_frames = i - start

            rows.append({
                "architecture_id":
                    int(architecture_id),

                "replicate":
                    int(replicate),

                "event_index":
                    event_index,

                "state":
                    states[start],

                "start_frame_in_series":
                    start,

                "end_frame_in_series":
                    i - 1,

                "n_frames":
                    n_frames,

                "dwell_time_ns":
                    n_frames * frame_dt_ns,

                "left_censored":
                    start == 0,

                "right_censored":
                    i == len(states),

                "censored":
                    (
                        start == 0
                        or i == len(states)
                    ),

                "state_method":
                    state_method,

                "frame_dt_ns":
                    frame_dt_ns,
            })

            event_index += 1
            start = i

    return rows


def derive_dwell_events(
    rg_df,
    single_threshold_nm=4.30,
    collapse_enter_nm=4.25,
    escape_exit_nm=4.35,
    fallback_frame_time_ns=0.01,
):
    """
    Reconstruct single-threshold and hysteresis dwell events for every
    architecture x replicate trajectory.
    """
    single_rows = []
    hysteresis_rows = []

    grouped = rg_df.groupby(
        [
            "architecture_id",
            "replicate"
        ],
        sort=True
    )

    for (
        architecture_id,
        replicate
    ), sub in grouped:

        sub = (
            sub
            .sort_values(
                [
                    "time_ns",
                    "frame_index"
                ]
            )
            .reset_index(drop=True)
        )

        rg = (
            sub["rg_nm"]
            .to_numpy()
        )

        dt = estimate_frame_dt_ns(
            sub,
            fallback_frame_time_ns=
                fallback_frame_time_ns
        )

        single_states = (
            assign_single_threshold_states(
                rg,
                threshold_nm=
                    single_threshold_nm
            )
        )

        hyst_states = (
            assign_hysteresis_states(
                rg,
                collapse_enter_nm=
                    collapse_enter_nm,
                escape_exit_nm=
                    escape_exit_nm,
            )
        )

        single_rows.extend(
            states_to_dwells(
                single_states,
                architecture_id,
                replicate,
                dt,
                state_method="single_threshold",
            )
        )

        hysteresis_rows.extend(
            states_to_dwells(
                hyst_states,
                architecture_id,
                replicate,
                dt,
                state_method="hysteresis",
            )
        )

    return (
        pd.DataFrame(single_rows),
        pd.DataFrame(hysteresis_rows),
    )


def get_complete_collapsed_events(
    dwell_df,
    include_censored=False
):
    """Select collapsed dwell events and optionally remove censored ones."""
    df = dwell_df[
        dwell_df["state"]
        == "collapsed"
    ].copy()

    if not include_censored:
        df = df[
            ~df["censored"]
        ].copy()

    return df


# ============================================================
# KINETIC / DWELL SUMMARY
# ============================================================

def calculate_metrics_per_replicate(
    dwell_df,
    sustained_threshold_ns=1.0,
    tail_threshold_ns=2.0,
    include_censored=False,
):
    """
    Calculate per-architecture/per-replicate collapsed-state metrics.
    """
    rows = []

    for (
        architecture_id,
        replicate
    ), all_events in dwell_df.groupby(
        [
            "architecture_id",
            "replicate"
        ]
    ):

        collapsed = all_events[
            all_events["state"]
            == "collapsed"
        ].copy()

        expanded = all_events[
            all_events["state"]
            == "expanded"
        ].copy()

        if not include_censored:
            collapsed_complete = collapsed[
                ~collapsed["censored"]
            ].copy()

            expanded_complete = expanded[
                ~expanded["censored"]
            ].copy()
        else:
            collapsed_complete = collapsed
            expanded_complete = expanded

        c = (
            collapsed_complete["dwell_time_ns"]
            .to_numpy()
        )

        sustained = c[
            c > sustained_threshold_ns
        ]

        tail = c[
            c > tail_threshold_ns
        ]

        total_collapsed_time = (
            collapsed["dwell_time_ns"]
            .sum()
        )

        total_expanded_time = (
            expanded["dwell_time_ns"]
            .sum()
        )

        total_time = (
            total_collapsed_time
            + total_expanded_time
        )

        rows.append({
            "architecture_id":
                architecture_id,

            "replicate":
                replicate,

            "state_method":
                all_events["state_method"].iloc[0],

            "frame_dt_ns":
                all_events["frame_dt_ns"].iloc[0],

            "n_collapsed_events":
                len(collapsed_complete),

            "n_expanded_events":
                len(expanded_complete),

            "n_sustained_gt_threshold":
                len(sustained),

            "n_tail_gt_threshold":
                len(tail),

            "fraction_sustained":
                (
                    len(sustained) / len(c)
                    if len(c) > 0
                    else np.nan
                ),

            "fraction_tail_given_sustained":
                (
                    len(tail) / len(sustained)
                    if len(sustained) > 0
                    else np.nan
                ),

            "mean_collapsed_dwell_ns":
                (
                    np.mean(c)
                    if len(c) > 0
                    else np.nan
                ),

            "median_collapsed_dwell_ns":
                (
                    np.median(c)
                    if len(c) > 0
                    else np.nan
                ),

            "mean_sustained_dwell_ns":
                (
                    np.mean(sustained)
                    if len(sustained) > 0
                    else np.nan
                ),

            "median_sustained_dwell_ns":
                (
                    np.median(sustained)
                    if len(sustained) > 0
                    else np.nan
                ),

            "collapsed_time_ns":
                total_collapsed_time,

            "expanded_time_ns":
                total_expanded_time,

            "total_time_ns":
                total_time,

            "collapsed_fraction":
                (
                    total_collapsed_time / total_time
                    if total_time > 0
                    else np.nan
                ),

            # For an alternating two-state series, the number of complete
            # collapsed events is a close proxy for collapse entries.
            "collapse_entries":
                len(collapsed_complete),

            "k_collapse_per_ns":
                (
                    len(collapsed_complete)
                    / total_expanded_time
                    if total_expanded_time > 0
                    else np.nan
                ),

            "k_escape_per_ns":
                (
                    len(expanded_complete)
                    / total_collapsed_time
                    if total_collapsed_time > 0
                    else np.nan
                ),
        })

    return pd.DataFrame(rows)


def summarize_metrics(per_rep):
    """
    Mean/SD across arbitrary replicate number.
    """
    if per_rep.empty:
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in per_rep.columns
        if c not in {
            "architecture_id",
            "replicate",
            "state_method",
        }
        and pd.api.types.is_numeric_dtype(
            per_rep[c]
        )
    ]

    rows = []

    for architecture_id, sub in per_rep.groupby(
        "architecture_id"
    ):
        row = {
            "architecture_id":
                architecture_id,

            "n_replicates":
                sub["replicate"].nunique(),

            "kg_class":
                KG_CLASS.get(
                    int(architecture_id),
                    "unknown"
                ),
        }

        for col in numeric_cols:
            row[f"{col}_mean"] = (
                sub[col].mean()
            )

            row[f"{col}_sd"] = (
                sub[col].std(ddof=1)
                if len(sub) > 1
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# STANDARD VS HYSTERESIS COMPARISON
# ============================================================

def compare_state_methods(
    single_summary,
    hyst_summary
):
    """
    Merge architecture-level summaries for sensitivity comparison.
    """
    keep_metrics = [
        "n_collapsed_events_mean",
        "n_sustained_gt_threshold_mean",
        "mean_collapsed_dwell_ns_mean",
        "median_collapsed_dwell_ns_mean",
        "collapsed_fraction_mean",
        "k_collapse_per_ns_mean",
        "k_escape_per_ns_mean",
    ]

    a_cols = [
        "architecture_id"
    ] + [
        c
        for c in keep_metrics
        if c in single_summary.columns
    ]

    b_cols = [
        "architecture_id"
    ] + [
        c
        for c in keep_metrics
        if c in hyst_summary.columns
    ]

    a = single_summary[
        a_cols
    ].copy()

    b = hyst_summary[
        b_cols
    ].copy()

    a = a.rename(
        columns={
            c: f"{c}_single"
            for c in a.columns
            if c != "architecture_id"
        }
    )

    b = b.rename(
        columns={
            c: f"{c}_hysteresis"
            for c in b.columns
            if c != "architecture_id"
        }
    )

    merged = a.merge(
        b,
        on="architecture_id",
        how="outer"
    )

    for metric in keep_metrics:
        s = f"{metric}_single"
        h = f"{metric}_hysteresis"

        if s in merged.columns and h in merged.columns:
            merged[
                f"{metric}_hysteresis_over_single"
            ] = (
                merged[h]
                / merged[s]
            )

    return merged


# ============================================================
# LINEAR-BIN DISTRIBUTIONS
# ============================================================

def linear_bin_edges(
    start_ns,
    end_ns,
    width_ns
):
    """
    Build equal-width bins.

    Example:
        start=0, end=3, width=0.05
    """
    if width_ns <= 0:
        raise ValueError(
            "Bin width must be > 0."
        )

    if end_ns <= start_ns:
        raise ValueError(
            "end_ns must be > start_ns."
        )

    n = int(
        np.ceil(
            (end_ns - start_ns)
            / width_ns
        )
    )

    edges = (
        start_ns
        + np.arange(
            n + 1
        )
        * width_ns
    )

    # Guarantee coverage of end_ns
    if edges[-1] < end_ns:
        edges = np.append(
            edges,
            edges[-1]
            + width_ns
        )

    return edges


def histogram_per_replicate(
    collapsed_df,
    architecture_id,
    bin_edges,
):
    """
    Calculate a histogram separately for each replicate.

    Returns
    -------
    mean_counts
    sd_counts
    mean_fraction
    sd_fraction
    n_replicates
    """
    sub = collapsed_df[
        collapsed_df["architecture_id"]
        == architecture_id
    ]

    reps = sorted(
        sub["replicate"].unique()
    )

    if len(reps) == 0:
        return (
            None,
            None,
            None,
            None,
            0
        )

    count_rows = []
    fraction_rows = []

    for rep in reps:
        values = sub.loc[
            sub["replicate"] == rep,
            "dwell_time_ns"
        ].to_numpy()

        counts, _ = np.histogram(
            values,
            bins=bin_edges
        )

        counts = counts.astype(float)

        count_rows.append(counts)

        total = counts.sum()

        if total > 0:
            fraction_rows.append(
                counts / total
            )
        else:
            fraction_rows.append(
                np.zeros_like(counts)
            )

    count_rows = np.vstack(
        count_rows
    )

    fraction_rows = np.vstack(
        fraction_rows
    )

    mean_counts = count_rows.mean(
        axis=0
    )

    mean_fraction = fraction_rows.mean(
        axis=0
    )

    if len(reps) > 1:
        sd_counts = count_rows.std(
            axis=0,
            ddof=1
        )

        sd_fraction = fraction_rows.std(
            axis=0,
            ddof=1
        )
    else:
        sd_counts = np.zeros_like(
            mean_counts
        )

        sd_fraction = np.zeros_like(
            mean_fraction
        )

    return (
        mean_counts,
        sd_counts,
        mean_fraction,
        sd_fraction,
        len(reps),
    )


def export_linear_binned_distributions(
    collapsed_df,
    architecture_ids,
    outdir,
    start_ns,
    end_ns,
    width_ns,
    tag,
):
    """Export the exact plotted binned values."""
    edges = linear_bin_edges(
        start_ns,
        end_ns,
        width_ns
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    rows = []

    for architecture_id in architecture_ids:
        (
            mean_counts,
            sd_counts,
            mean_fraction,
            sd_fraction,
            n_reps,
        ) = histogram_per_replicate(
            collapsed_df,
            architecture_id,
            edges
        )

        if mean_counts is None:
            continue

        for i, center in enumerate(
            centers
        ):
            rows.append({
                "architecture_id":
                    architecture_id,

                "n_replicates":
                    n_reps,

                "bin_left_ns":
                    edges[i],

                "bin_right_ns":
                    edges[i + 1],

                "bin_center_ns":
                    center,

                "mean_count_per_replicate":
                    mean_counts[i],

                "sd_count_per_replicate":
                    sd_counts[i],

                "mean_fraction_per_replicate":
                    mean_fraction[i],

                "sd_fraction_per_replicate":
                    sd_fraction[i],
            })

    out = pd.DataFrame(rows)

    outfile = (
        outdir
        / f"binned_dwell_distribution_{tag}.csv"
    )

    out.to_csv(
        outfile,
        index=False
    )

    return outfile


def plot_matched_linear_distributions(
    collapsed_df,
    selected_architecture_ids,
    outdir,
    start_ns,
    end_ns,
    width_ns,
    sustained_threshold_ns,
    tail_threshold_ns,
    tag,
):
    """
    For each matched architecture group, generate:

      1. mean event count per replicate per equal-width bin
      2. mean fraction of events per replicate per equal-width bin

    Equal linear bins avoid the artificial short-time peaks produced by
    logarithmic binning of 0.01-ns-quantized dwell times.
    """
    plot_dir = (
        outdir
        / f"matched_{tag}"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = set(
        int(x)
        for x in selected_architecture_ids
    )

    edges = linear_bin_edges(
        start_ns,
        end_ns,
        width_ns
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    for group in MATCHED_GROUPS:
        ids = [
            x
            for x in group
            if x in selected
        ]

        if len(ids) < 2:
            continue

        # ---------------------------
        # Count plot
        # ---------------------------
        fig, ax = plt.subplots(
            figsize=(10, 6.5)
        )

        for architecture_id in ids:
            (
                mean_counts,
                sd_counts,
                _,
                _,
                n_reps,
            ) = histogram_per_replicate(
                collapsed_df,
                architecture_id,
                edges
            )

            if mean_counts is None:
                continue

            ax.plot(
                centers,
                mean_counts,
                linewidth=1.8,
                label=str(architecture_id)
            )

            if n_reps > 1:
                ax.fill_between(
                    centers,
                    np.maximum(
                        mean_counts
                        - sd_counts,
                        0
                    ),
                    mean_counts
                    + sd_counts,
                    alpha=0.12
                )

        ax.axvline(
            sustained_threshold_ns,
            linestyle="--",
            linewidth=1.0,
            color="0.35"
        )

        ax.axvline(
            tail_threshold_ns,
            linestyle=":",
            linewidth=1.0,
            color="0.35"
        )

        ax.set_xlim(
            start_ns,
            end_ns
        )

        ax.set_xlabel(
            "Collapsed-state dwell time (ns)"
        )

        ax.set_ylabel(
            f"Mean event count / replicate / {width_ns:g}-ns bin"
        )

        ax.set_title(
            "Hysteresis dwell counts: "
            + " vs ".join(
                str(x)
                for x in ids
            )
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            frameon=True
        )

        fig.tight_layout()

        name = (
            "counts_"
            + "_".join(
                str(x)
                for x in ids
            )
            + f"_{tag}.png"
        )

        safe_savefig(
            fig,
            plot_dir / name,
            dpi=300
        )

        plt.close(fig)

        # ---------------------------
        # Normalized shape plot
        # ---------------------------
        fig, ax = plt.subplots(
            figsize=(10, 6.5)
        )

        for architecture_id in ids:
            (
                _,
                _,
                mean_fraction,
                sd_fraction,
                n_reps,
            ) = histogram_per_replicate(
                collapsed_df,
                architecture_id,
                edges
            )

            if mean_fraction is None:
                continue

            ax.plot(
                centers,
                mean_fraction,
                linewidth=1.8,
                label=str(architecture_id)
            )

            if n_reps > 1:
                ax.fill_between(
                    centers,
                    np.maximum(
                        mean_fraction
                        - sd_fraction,
                        0
                    ),
                    mean_fraction
                    + sd_fraction,
                    alpha=0.12
                )

        ax.axvline(
            sustained_threshold_ns,
            linestyle="--",
            linewidth=1.0,
            color="0.35"
        )

        ax.axvline(
            tail_threshold_ns,
            linestyle=":",
            linewidth=1.0,
            color="0.35"
        )

        ax.set_xlim(
            start_ns,
            end_ns
        )

        ax.set_xlabel(
            "Collapsed-state dwell time (ns)"
        )

        ax.set_ylabel(
            f"Fraction of events / {width_ns:g}-ns bin"
        )

        ax.set_title(
            "Hysteresis dwell-time shape: "
            + " vs ".join(
                str(x)
                for x in ids
            )
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            frameon=True
        )

        fig.tight_layout()

        name = (
            "shape_"
            + "_".join(
                str(x)
                for x in ids
            )
            + f"_{tag}.png"
        )

        safe_savefig(
            fig,
            plot_dir / name,
            dpi=300
        )

        plt.close(fig)


# ============================================================
# ALL-ARCHITECTURE HEATMAP
# ============================================================

def plot_count_heatmap(
    collapsed_df,
    architecture_ids,
    outdir,
    start_ns,
    end_ns,
    width_ns,
    rg_order=None,
):
    """
    Heatmap of mean event count per replicate.

    This is more readable than 30 overlapping curves.
    """
    edges = linear_bin_edges(
        start_ns,
        end_ns,
        width_ns
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    ids = list(
        architecture_ids
    )

    if rg_order is not None:
        rank = {
            int(a): i
            for i, a in enumerate(
                rg_order
            )
        }

        ids = sorted(
            ids,
            key=lambda x: rank.get(
                int(x),
                10**9
            )
        )

    rows = []

    valid_ids = []

    for architecture_id in ids:
        (
            mean_counts,
            _,
            _,
            _,
            _,
        ) = histogram_per_replicate(
            collapsed_df,
            architecture_id,
            edges
        )

        if mean_counts is None:
            continue

        rows.append(
            mean_counts
        )
        valid_ids.append(
            architecture_id
        )

    if not rows:
        return

    matrix = np.vstack(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[
            edges[0],
            edges[-1],
            len(valid_ids) - 0.5,
            -0.5,
        ],
    )

    ax.set_yticks(
        np.arange(
            len(valid_ids)
        )
    )

    ax.set_yticklabels(
        [
            str(x)
            for x in valid_ids
        ]
    )

    ax.set_xlabel(
        "Collapsed-state dwell time (ns)"
    )

    ax.set_ylabel(
        "Architecture ID"
    )

    ax.set_title(
        "Hysteresis collapsed-dwell counts "
        "(mean events per replicate)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        f"Mean count / replicate / {width_ns:g}-ns bin"
    )

    fig.tight_layout()

    safe_savefig(
        fig,
        outdir / "hysteresis_dwell_count_heatmap.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================
# DYNAMIC PROPERTY OVERVIEW
# ============================================================

def calculate_relative_variation(summary):
    """
    Quantify how strongly each architecture-level property varies across
    the library.

    Three simple descriptors are reported:
      - relative_range = (max - min) / mean
      - CV = SD / mean
      - fold_range = max / min

    These are descriptive across architectures, not uncertainty estimates.
    """
    metric_info = [
        (
            "n_collapsed_events_mean",
            "Collapsed event count",
        ),
        (
            "n_sustained_gt_threshold_mean",
            "Sustained event count",
        ),
        (
            "fraction_sustained_mean",
            "Fraction sustained",
        ),
        (
            "mean_collapsed_dwell_ns_mean",
            "Mean collapsed dwell",
        ),
        (
            "median_collapsed_dwell_ns_mean",
            "Median collapsed dwell",
        ),
        (
            "mean_sustained_dwell_ns_mean",
            "Mean sustained dwell",
        ),
        (
            "median_sustained_dwell_ns_mean",
            "Median sustained dwell",
        ),
        (
            "collapsed_fraction_mean",
            "Collapsed fraction",
        ),
        (
            "k_collapse_per_ns_mean",
            "Collapse-entry rate",
        ),
        (
            "k_escape_per_ns_mean",
            "Escape rate",
        ),
    ]

    rows = []

    for col, label in metric_info:
        if col not in summary.columns:
            continue

        values = pd.to_numeric(
            summary[col],
            errors="coerce"
        ).dropna().to_numpy()

        if len(values) == 0:
            continue

        mean = float(np.mean(values))
        sd = float(
            np.std(values, ddof=1)
            if len(values) > 1
            else np.nan
        )
        vmin = float(np.min(values))
        vmax = float(np.max(values))

        rows.append({
            "metric_column": col,
            "metric": label,
            "mean_across_architectures": mean,
            "sd_across_architectures": sd,
            "min": vmin,
            "max": vmax,
            "absolute_range": vmax - vmin,
            "relative_range": (
                (vmax - vmin) / abs(mean)
                if mean != 0
                else np.nan
            ),
            "cv_across_architectures": (
                sd / abs(mean)
                if mean != 0
                else np.nan
            ),
            "fold_range": (
                vmax / vmin
                if vmin > 0
                else np.nan
            ),
            "n_architectures": len(values),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = (
            result
            .sort_values(
                "relative_range",
                ascending=False
            )
            .reset_index(drop=True)
        )

    return result


def plot_dynamic_property_overview(
    summary,
    outdir,
    rg_order=None,
):
    """
    Main architecture-by-architecture overview of dynamic properties.

    Mean +/- replicate SD is shown for each architecture. If Rg ordering is
    available, architectures are kept in the same compact-to-expanded order
    as the original Rg overview so the figures can be compared directly.
    """
    if summary.empty:
        return

    plot_df = summary.copy()

    if rg_order is not None:
        order_map = {
            int(arch): i
            for i, arch in enumerate(rg_order)
        }

        plot_df["plot_order"] = (
            plot_df["architecture_id"]
            .map(order_map)
        )

        plot_df = (
            plot_df
            .dropna(subset=["plot_order"])
            .sort_values("plot_order")
            .reset_index(drop=True)
        )

    else:
        plot_df = (
            plot_df
            .sort_values("architecture_id")
            .reset_index(drop=True)
        )

    metrics = [
        (
            "k_collapse_per_ns_mean",
            "k_collapse_per_ns_sd",
            "A   Collapse-entry rate",
            r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
        ),
        (
            "k_escape_per_ns_mean",
            "k_escape_per_ns_sd",
            "B   Escape rate",
            r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
        ),
        (
            "n_collapsed_events_mean",
            "n_collapsed_events_sd",
            "C   Collapsed event count",
            "Events / replicate",
        ),
        (
            "n_sustained_gt_threshold_mean",
            "n_sustained_gt_threshold_sd",
            "D   Sustained event count",
            "Events > 1 ns / replicate",
        ),
        (
            "fraction_sustained_mean",
            "fraction_sustained_sd",
            "E   Fraction of collapse events sustained",
            "Fraction > 1 ns",
        ),
        (
            "median_sustained_dwell_ns_mean",
            "median_sustained_dwell_ns_sd",
            "F   Median sustained dwell time",
            "Median dwell > 1 ns (ns)",
        ),
        (
            "collapsed_fraction_mean",
            "collapsed_fraction_sd",
            "G   Collapsed-state occupancy",
            "Collapsed fraction",
        ),
    ]

    metrics = [
        x
        for x in metrics
        if x[0] in plot_df.columns
    ]

    n_panels = len(metrics)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(16, 3.1 * n_panels),
        sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    x = np.arange(
        len(plot_df)
    )

    for ax, (
        mean_col,
        sd_col,
        title,
        ylabel
    ) in zip(
        axes,
        metrics
    ):
        for i, row in plot_df.iterrows():

            kg_class = KG_CLASS.get(
                int(row["architecture_id"]),
                "unknown"
            )

            color = {
                "peripheral": "C0",
                "internal": "C1",
                "control": "0.45",
            }.get(
                kg_class,
                "0.55"
            )

            y = row[mean_col]

            if pd.isna(y):
                continue

            sd = (
                row[sd_col]
                if sd_col in plot_df.columns
                else np.nan
            )

            if pd.isna(sd):
                sd = 0.0

            ax.errorbar(
                i,
                y,
                yerr=sd,
                fmt="o",
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                capsize=3,
                elinewidth=1.0,
                zorder=3
            )

        ax.set_title(
            title,
            loc="left",
            fontweight="bold"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            axis="y",
            alpha=0.20
        )

    axes[-1].set_xticks(
        x
    )

    axes[-1].set_xticklabels(
        plot_df["architecture_id"]
        .astype(str),
        rotation=90
    )

    axes[-1].set_xlabel(
        "Architecture ID"
        + (
            " (ordered by mean Rg370, compact to expanded)"
            if rg_order is not None
            else ""
        )
    )

    fig.suptitle(
        "VGIPKG dynamic-property overview from hysteresis analysis",
        fontsize=16,
        y=0.997
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.985]
    )

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_property_overview_7panel.png",
        dpi=300
    )

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_property_overview_7panel.pdf"
    )

    plt.close(fig)


def plot_relative_dynamic_profiles(
    summary,
    outdir,
    rg_order=None,
):
    """
    Plot all major dynamic observables on a common relative scale.

    Each metric is converted to percent deviation from the library mean:

        100 * (value / library_mean - 1)

    This makes it easy to see which properties show the largest architecture-
    dependent spread despite having different physical units.
    """
    if summary.empty:
        return

    plot_df = summary.copy()

    if rg_order is not None:
        order_map = {
            int(arch): i
            for i, arch in enumerate(rg_order)
        }

        plot_df["plot_order"] = (
            plot_df["architecture_id"]
            .map(order_map)
        )

        plot_df = (
            plot_df
            .dropna(subset=["plot_order"])
            .sort_values("plot_order")
            .reset_index(drop=True)
        )

    else:
        plot_df = (
            plot_df
            .sort_values("architecture_id")
            .reset_index(drop=True)
        )

    metrics = [
        (
            "k_collapse_per_ns_mean",
            "Collapse-entry rate"
        ),
        (
            "k_escape_per_ns_mean",
            "Escape rate"
        ),
        (
            "n_collapsed_events_mean",
            "Collapsed event count"
        ),
        (
            "n_sustained_gt_threshold_mean",
            "Sustained event count"
        ),
        (
            "fraction_sustained_mean",
            "Fraction sustained"
        ),
        (
            "median_sustained_dwell_ns_mean",
            "Median sustained dwell"
        ),
        (
            "collapsed_fraction_mean",
            "Collapsed fraction"
        ),
    ]

    metrics = [
        (col, label)
        for col, label in metrics
        if col in plot_df.columns
    ]

    x = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    export_rows = []

    for col, label in metrics:

        values = pd.to_numeric(
            plot_df[col],
            errors="coerce"
        ).to_numpy(
            dtype=float
        )

        library_mean = np.nanmean(
            values
        )

        if (
            not np.isfinite(library_mean)
            or library_mean == 0
        ):
            continue

        relative = (
            100.0
            * (
                values
                / library_mean
                - 1.0
            )
        )

        ax.plot(
            x,
            relative,
            marker="o",
            linewidth=1.25,
            markersize=4,
            label=label
        )

        for architecture_id, value in zip(
            plot_df["architecture_id"],
            relative
        ):
            export_rows.append({
                "architecture_id":
                    int(architecture_id),

                "metric":
                    label,

                "percent_deviation_from_library_mean":
                    value,
            })

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
        color="0.35"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_df["architecture_id"]
        .astype(str),
        rotation=90
    )

    ax.set_xlabel(
        "Architecture ID"
        + (
            " (ordered by mean Rg370, compact to expanded)"
            if rg_order is not None
            else ""
        )
    )

    ax.set_ylabel(
        "Deviation from library mean (%)"
    )

    ax.set_title(
        "Relative architecture-dependent variation of dynamic properties"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.legend(
        frameon=True,
        fontsize=9,
        ncol=2
    )

    fig.tight_layout()

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_properties_relative_variation.png",
        dpi=300
    )

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_properties_relative_variation.pdf"
    )

    plt.close(fig)

    pd.DataFrame(
        export_rows
    ).to_csv(
        outdir
        / "hysteresis_dynamic_properties_relative_values.csv",
        index=False
    )


def plot_relative_variation_ranking(
    variation_df,
    outdir,
):
    """
    Simple ranked bar plot showing which metrics vary most across architectures.
    """
    if variation_df.empty:
        return

    plot_df = (
        variation_df
        .dropna(
            subset=["relative_range"]
        )
        .sort_values(
            "relative_range",
            ascending=True
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["metric"],
        100.0
        * plot_df["relative_range"]
    )

    ax.set_xlabel(
        "Architecture range / library mean (%)"
    )

    ax.set_ylabel(
        "Dynamic property"
    )

    ax.set_title(
        "Which dynamic properties vary most across VGIPKG architectures?"
    )

    ax.grid(
        axis="x",
        alpha=0.20
    )

    fig.tight_layout()

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_property_relative_range_ranking.png",
        dpi=300
    )

    safe_savefig(
        fig,
        outdir
        / "hysteresis_dynamic_property_relative_range_ranking.pdf"
    )

    plt.close(fig)


# ============================================================
# Rg370 CORRELATIONS
# ============================================================

def load_overview_data(
    overview_file
):
    if overview_file is None:
        return None, None, None

    overview = pd.read_csv(
        overview_file
    )

    if "architecture_id" not in overview.columns:
        raise ValueError(
            "--overview-data must contain architecture_id."
        )

    rg_col = None

    for candidate in [
        "mean_rg_nm_370K_mean",
        "mean_rg_nm_370K",
    ]:
        if candidate in overview.columns:
            rg_col = candidate
            break

    if rg_col is None:
        raise ValueError(
            "Could not find a recognized Rg370 column "
            "in --overview-data."
        )

    rg_order = (
        overview
        .sort_values(rg_col)
        ["architecture_id"]
        .astype(int)
        .tolist()
    )

    return (
        overview,
        rg_col,
        rg_order
    )


def plot_rg_correlations(
    summary,
    overview,
    rg_col,
    outdir,
):
    """
    Correlate replicate-mean Rg370 with key hysteresis metrics.
    """
    if (
        overview is None
        or summary.empty
    ):
        return

    merged = summary.merge(
        overview[
            [
                "architecture_id",
                rg_col
            ]
        ],
        on="architecture_id",
        how="inner"
    )

    metrics = [
        (
            "n_collapsed_events_mean",
            "Collapsed events per replicate",
            "rg_vs_hyst_collapsed_event_count.png",
        ),
        (
            "n_sustained_gt_threshold_mean",
            "Sustained collapsed events per replicate",
            "rg_vs_hyst_sustained_event_count.png",
        ),
        (
            "fraction_sustained_mean",
            "Fraction of collapsed events sustained",
            "rg_vs_hyst_fraction_sustained.png",
        ),
        (
            "median_sustained_dwell_ns_mean",
            "Median sustained dwell time (ns)",
            "rg_vs_hyst_median_sustained.png",
        ),
        (
            "k_collapse_per_ns_mean",
            "Hysteresis collapse-entry rate (ns$^{-1}$)",
            "rg_vs_hyst_kcollapse.png",
        ),
        (
            "k_escape_per_ns_mean",
            "Hysteresis escape rate (ns$^{-1}$)",
            "rg_vs_hyst_kescape.png",
        ),
        (
            "collapsed_fraction_mean",
            "Hysteresis collapsed fraction",
            "rg_vs_hyst_collapsed_fraction.png",
        ),
    ]

    stat_rows = []

    for (
        metric,
        xlabel,
        filename
    ) in metrics:

        if metric not in merged.columns:
            continue

        clean = merged[
            [
                "architecture_id",
                metric,
                rg_col
            ]
        ].dropna()

        if len(clean) < 3:
            continue

        x = clean[
            metric
        ].to_numpy()

        y = clean[
            rg_col
        ].to_numpy()

        pearson_r, pearson_p = (
            pearsonr(
                x,
                y
            )
        )

        spearman_rho, spearman_p = (
            spearmanr(
                x,
                y
            )
        )

        stat_rows.append({
            "metric":
                metric,

            "pearson_r":
                pearson_r,

            "pearson_p":
                pearson_p,

            "spearman_rho":
                spearman_rho,

            "spearman_p":
                spearman_p,

            "n_architectures":
                len(clean),
        })

        fig, ax = plt.subplots(
            figsize=(8, 6.5)
        )

        ax.scatter(
            x,
            y,
            s=55
        )

        for _, row in clean.iterrows():
            ax.annotate(
                str(
                    int(
                        row["architecture_id"]
                    )
                ),
                (
                    row[metric],
                    row[rg_col]
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8
            )

        if (
            np.nanstd(x) > 0
            and np.nanstd(y) > 0
        ):
            slope, intercept = (
                np.polyfit(
                    x,
                    y,
                    1
                )
            )

            xx = np.linspace(
                np.min(x),
                np.max(x),
                200
            )

            ax.plot(
                xx,
                slope * xx
                + intercept,
                linestyle="--",
                linewidth=1.1
            )

        ax.text(
            0.03,
            0.97,
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman rho = {spearman_rho:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="0.75",
                alpha=0.9
            )
        )

        ax.set_xlabel(
            xlabel
        )

        ax.set_ylabel(
            r"Mean $R_g$ at 370 K (nm)"
        )

        ax.set_title(
            f"VGIPKG: Rg370 vs {xlabel}"
        )

        ax.grid(
            alpha=0.20
        )

        fig.tight_layout()

        safe_savefig(
            fig,
            outdir / filename,
            dpi=300
        )

        plt.close(fig)

    pd.DataFrame(
        stat_rows
    ).to_csv(
        outdir
        / "rg370_hysteresis_correlations.csv",
        index=False
    )


def plot_kcollapse_vs_sustained(
    summary,
    outdir,
):
    """
    Test whether sustained-event count mostly inherits collapse-entry frequency.
    """
    xcol = "k_collapse_per_ns_mean"
    ycol = "n_sustained_gt_threshold_mean"

    clean = summary[
        [
            "architecture_id",
            xcol,
            ycol
        ]
    ].dropna()

    if len(clean) < 3:
        return

    x = clean[xcol].to_numpy()
    y = clean[ycol].to_numpy()

    pr, pp = pearsonr(
        x,
        y
    )

    sr, sp = spearmanr(
        x,
        y
    )

    fig, ax = plt.subplots(
        figsize=(8, 6.5)
    )

    ax.scatter(
        x,
        y,
        s=55
    )

    for _, row in clean.iterrows():
        ax.annotate(
            str(
                int(
                    row["architecture_id"]
                )
            ),
            (
                row[xcol],
                row[ycol]
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8
        )

    if np.nanstd(x) > 0:
        slope, intercept = np.polyfit(
            x,
            y,
            1
        )

        xx = np.linspace(
            x.min(),
            x.max(),
            200
        )

        ax.plot(
            xx,
            slope * xx + intercept,
            linestyle="--",
            linewidth=1.1
        )

    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pr:.3f}\n"
        f"Spearman rho = {sr:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.9
        )
    )

    ax.set_xlabel(
        "Hysteresis collapse-entry rate (ns$^{-1}$)"
    )

    ax.set_ylabel(
        "Sustained collapsed events per replicate"
    )

    ax.set_title(
        "Does sustained-event count mainly reflect collapse-entry frequency?"
    )

    ax.grid(
        alpha=0.20
    )

    fig.tight_layout()

    safe_savefig(
        fig,
        outdir
        / "hysteresis_kcollapse_vs_sustained_count.png",
        dpi=300
    )

    plt.close(fig)

    pd.DataFrame(
        [{
            "pearson_r":
                pr,

            "pearson_p":
                pp,

            "spearman_rho":
                sr,

            "spearman_p":
                sp,

            "n_architectures":
                len(clean),
        }]
    ).to_csv(
        outdir
        / "hysteresis_kcollapse_vs_sustained_count_stats.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct collapsed-state dwell events directly from "
            "all_rg_values.csv files using both the original Rg threshold "
            "and a hysteresis state definition."
        )
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "One or more all_rg_values.csv files, "
            "one per independent replicate."
        )
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--architectures",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional architecture IDs. "
            "If omitted, all detected architectures are used."
        )
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=370,
        help="Temperature to analyse. Default = 370 K."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.30,
        help=(
            "Original single collapse threshold in nm. "
            "Default = 4.30."
        )
    )

    parser.add_argument(
        "--collapse-enter-nm",
        type=float,
        default=4.25,
        help=(
            "Hysteresis threshold for entering collapsed state. "
            "Default = 4.25 nm."
        )
    )

    parser.add_argument(
        "--escape-exit-nm",
        type=float,
        default=4.35,
        help=(
            "Hysteresis threshold for leaving collapsed state. "
            "Default = 4.35 nm."
        )
    )

    parser.add_argument(
        "--frame-time-ns",
        type=float,
        default=0.01,
        help=(
            "Fallback saved-frame spacing if time cannot be inferred "
            "from all_rg_values.csv. Default = 0.01 ns."
        )
    )

    parser.add_argument(
        "--include-censored",
        action="store_true",
        help=(
            "Include first/last trajectory-boundary dwell events "
            "in distribution summaries. Default = exclude."
        )
    )

    parser.add_argument(
        "--sustained-threshold-ns",
        type=float,
        default=1.0,
        help=(
            "Sustained collapse threshold. Default = 1.0 ns."
        )
    )

    parser.add_argument(
        "--tail-threshold-ns",
        type=float,
        default=2.0,
        help=(
            "Long-tail threshold. Default = 2.0 ns."
        )
    )

    parser.add_argument(
        "--short-max-ns",
        type=float,
        default=3.0,
        help=(
            "Maximum dwell time for short-range matched plots. "
            "Default = 3.0 ns."
        )
    )

    parser.add_argument(
        "--short-bin-ns",
        type=float,
        default=0.05,
        help=(
            "Linear bin width for 0-short_max dwell plots. "
            "Default = 0.05 ns (= 5 frames at 0.01 ns/frame)."
        )
    )

    parser.add_argument(
        "--tail-max-ns",
        type=float,
        default=8.0,
        help=(
            "Maximum dwell time for broader tail plots. "
            "Default = 8.0 ns."
        )
    )

    parser.add_argument(
        "--tail-bin-ns",
        type=float,
        default=0.20,
        help=(
            "Linear bin width for broader tail plots. "
            "Default = 0.20 ns."
        )
    )

    parser.add_argument(
        "--overview-data",
        default=None,
        help=(
            "Optional VGIPKG architecture overview CSV containing "
            "replicate-mean Rg370."
        )
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("============================================================")
    print("VGIPKG dwell analysis from original all_rg_values.csv")
    print("============================================================")
    print(
        f"Temperature: {args.temperature:g} K"
    )
    print(
        f"Original threshold: Rg < {args.threshold:g} nm"
    )
    print(
        "Hysteresis: "
        f"expanded -> collapsed at <= {args.collapse_enter_nm:g} nm; "
        f"collapsed -> expanded at >= {args.escape_exit_nm:g} nm"
    )
    print(
        f"Sustained threshold: > {args.sustained_threshold_ns:g} ns"
    )
    print(
        f"Tail threshold: > {args.tail_threshold_ns:g} ns"
    )

    # --------------------------------------------------------
    # Load raw per-frame Rg
    # --------------------------------------------------------
    rg_df = load_rg_files(
        args.inputs,
        temperature=args.temperature,
        fallback_frame_time_ns=
            args.frame_time_ns,
    )

    # --------------------------------------------------------
    # Architecture selection
    # --------------------------------------------------------
    if args.architectures:
        architecture_ids = list(
            args.architectures
        )

        rg_df = rg_df[
            rg_df["architecture_id"]
            .isin(architecture_ids)
        ].copy()

    else:
        architecture_ids = sorted(
            rg_df["architecture_id"]
            .unique()
            .tolist()
        )

    if rg_df.empty:
        raise ValueError(
            "No Rg frames remain after filtering."
        )

    print()
    print(
        "Architectures selected:",
        architecture_ids
    )

    print(
        "Total Rg frames:",
        len(rg_df)
    )

    # --------------------------------------------------------
    # Reconstruct dwell events
    # --------------------------------------------------------
    single_dwells, hyst_dwells = (
        derive_dwell_events(
            rg_df,
            single_threshold_nm=
                args.threshold,
            collapse_enter_nm=
                args.collapse_enter_nm,
            escape_exit_nm=
                args.escape_exit_nm,
            fallback_frame_time_ns=
                args.frame_time_ns,
        )
    )

    single_dwells.to_csv(
        outdir
        / "derived_dwells_single_threshold.csv",
        index=False
    )

    hyst_dwells.to_csv(
        outdir
        / "derived_dwells_hysteresis.csv",
        index=False
    )

    # Complete collapsed dwell sets used for distributions
    single_collapsed = (
        get_complete_collapsed_events(
            single_dwells,
            include_censored=
                args.include_censored
        )
    )

    hyst_collapsed = (
        get_complete_collapsed_events(
            hyst_dwells,
            include_censored=
                args.include_censored
        )
    )

    print()
    print(
        "Complete collapsed events "
        f"(single threshold): {len(single_collapsed)}"
    )

    print(
        "Complete collapsed events "
        f"(hysteresis): {len(hyst_collapsed)}"
    )

    if len(single_collapsed) > 0:
        print(
            "Fraction of collapsed events retained by hysteresis:",
            f"{len(hyst_collapsed) / len(single_collapsed):.4f}"
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    single_per_rep = (
        calculate_metrics_per_replicate(
            single_dwells,
            sustained_threshold_ns=
                args.sustained_threshold_ns,
            tail_threshold_ns=
                args.tail_threshold_ns,
            include_censored=
                args.include_censored,
        )
    )

    hyst_per_rep = (
        calculate_metrics_per_replicate(
            hyst_dwells,
            sustained_threshold_ns=
                args.sustained_threshold_ns,
            tail_threshold_ns=
                args.tail_threshold_ns,
            include_censored=
                args.include_censored,
        )
    )

    single_summary = summarize_metrics(
        single_per_rep
    )

    hyst_summary = summarize_metrics(
        hyst_per_rep
    )

    single_per_rep.to_csv(
        outdir
        / "single_threshold_metrics_per_replicate.csv",
        index=False
    )

    hyst_per_rep.to_csv(
        outdir
        / "hysteresis_metrics_per_replicate.csv",
        index=False
    )

    single_summary.to_csv(
        outdir
        / "single_threshold_metrics_summary.csv",
        index=False
    )

    hyst_summary.to_csv(
        outdir
        / "hysteresis_metrics_summary.csv",
        index=False
    )

    method_comparison = (
        compare_state_methods(
            single_summary,
            hyst_summary
        )
    )

    method_comparison.to_csv(
        outdir
        / "single_threshold_vs_hysteresis.csv",
        index=False
    )

    # --------------------------------------------------------
    # Overview / Rg order
    # --------------------------------------------------------
    overview, rg_col, rg_order = (
        load_overview_data(
            args.overview_data
        )
    )

    # --------------------------------------------------------
    # Architecture-by-architecture dynamic-property overview
    # --------------------------------------------------------
    variation_df = calculate_relative_variation(
        hyst_summary
    )

    variation_file = (
        outdir
        / "hysteresis_dynamic_property_variation_summary.csv"
    )

    variation_df.to_csv(
        variation_file,
        index=False
    )

    plot_dynamic_property_overview(
        hyst_summary,
        outdir,
        rg_order=rg_order,
    )

    plot_relative_dynamic_profiles(
        hyst_summary,
        outdir,
        rg_order=rg_order,
    )

    plot_relative_variation_ranking(
        variation_df,
        outdir,
    )

    # --------------------------------------------------------
    # Hysteresis distributions: short region
    # --------------------------------------------------------
    short_csv = (
        export_linear_binned_distributions(
            hyst_collapsed,
            architecture_ids,
            outdir,
            start_ns=0.0,
            end_ns=
                args.short_max_ns,
            width_ns=
                args.short_bin_ns,
            tag="hysteresis_short",
        )
    )

    plot_matched_linear_distributions(
        hyst_collapsed,
        architecture_ids,
        outdir,
        start_ns=0.0,
        end_ns=
            args.short_max_ns,
        width_ns=
            args.short_bin_ns,
        sustained_threshold_ns=
            args.sustained_threshold_ns,
        tail_threshold_ns=
            args.tail_threshold_ns,
        tag="hysteresis_short",
    )

    # --------------------------------------------------------
    # Hysteresis distributions: broader tail
    # --------------------------------------------------------
    tail_csv = (
        export_linear_binned_distributions(
            hyst_collapsed,
            architecture_ids,
            outdir,
            start_ns=0.0,
            end_ns=
                args.tail_max_ns,
            width_ns=
                args.tail_bin_ns,
            tag="hysteresis_tail",
        )
    )

    plot_matched_linear_distributions(
        hyst_collapsed,
        architecture_ids,
        outdir,
        start_ns=0.0,
        end_ns=
            args.tail_max_ns,
        width_ns=
            args.tail_bin_ns,
        sustained_threshold_ns=
            args.sustained_threshold_ns,
        tail_threshold_ns=
            args.tail_threshold_ns,
        tag="hysteresis_tail",
    )

    # --------------------------------------------------------
    # All-architecture count heatmap
    # --------------------------------------------------------
    plot_count_heatmap(
        hyst_collapsed,
        architecture_ids,
        outdir,
        start_ns=0.0,
        end_ns=
            args.short_max_ns,
        width_ns=
            args.short_bin_ns,
        rg_order=rg_order,
    )

    # --------------------------------------------------------
    # Rg / kinetic relationships
    # --------------------------------------------------------
    plot_rg_correlations(
        hyst_summary,
        overview,
        rg_col,
        outdir,
    )

    plot_kcollapse_vs_sustained(
        hyst_summary,
        outdir,
    )

    print()
    print("============================================================")
    print("Finished")
    print("============================================================")
    print(
        "Saved outputs to:"
    )
    print(outdir)

    print()
    print("Important outputs:")
    print(
        outdir
        / "derived_dwells_hysteresis.csv"
    )
    print(
        outdir
        / "hysteresis_metrics_summary.csv"
    )
    print(
        outdir
        / "single_threshold_vs_hysteresis.csv"
    )
    print(short_csv)
    print(tail_csv)
    print(
        outdir
        / "matched_hysteresis_short"
    )
    print(
        outdir
        / "matched_hysteresis_tail"
    )
    print(
        outdir
        / "hysteresis_dwell_count_heatmap.png"
    )
    print(
        outdir
        / "hysteresis_dynamic_property_overview_7panel.png"
    )
    print(
        outdir
        / "hysteresis_dynamic_properties_relative_variation.png"
    )
    print(
        outdir
        / "hysteresis_dynamic_property_relative_range_ranking.png"
    )
    print(
        variation_file
    )


if __name__ == "__main__":
    main()
