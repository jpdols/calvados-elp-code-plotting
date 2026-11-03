#!/usr/bin/env python
"""
investigate_vgipkg_properties.py

Exploratory architecture-level analysis for VGIPKG replicate simulations.

Inputs
------
For each replicate:
    - all_rg_values.csv
    - per_run_summary.csv

The script extracts a broad set of properties from frame-by-frame Rg trajectories,
aggregates them across replicates, quantifies architecture-to-architecture variation,
computes pairwise correlations, and writes an interactive HTML dashboard in which
architectures can be sorted by any property.

Designed for 1, 2, 3, ... replicates.

Main outputs
------------
architecture_properties_per_replicate.csv
architecture_properties_summary.csv
property_variation_summary.csv
property_correlations_pearson.csv
property_correlations_spearman.csv
property_pairwise_long.csv
property_overview_sorted_by_rg370.png
property_relative_variation.png
property_variation_ranking.png
property_correlation_heatmap.png
property_scatter_matrix_selected.png
interactive_architecture_property_explorer.html
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def architecture_id_from_text(value):
    m = re.search(r"(?<!\d)(\d{3})(?!\d)", str(value))
    return int(m.group(1)) if m else np.nan


def first_existing_column(df, candidates, required=False):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def infer_dt_ns(group, fallback=0.01):
    time_col = first_existing_column(
        group,
        ["time_ns", "time", "t_ns", "Time_ns", "Time (ns)"],
        required=False,
    )
    if time_col is None or len(group) < 2:
        return fallback

    t = pd.to_numeric(group[time_col], errors="coerce").dropna().to_numpy()
    if len(t) < 2:
        return fallback

    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return fallback

    return float(np.median(diffs))


def runs_from_frame_file(path, replicate_label):
    df = pd.read_csv(path)

    id_col = first_existing_column(
        df,
        ["id", "architecture", "architecture_id", "name", "run"],
        required=True,
    )
    temp_col = first_existing_column(
        df,
        ["temp", "temperature", "temperature_K", "T"],
        required=True,
    )
    rg_col = first_existing_column(
        df,
        ["rg_nm", "Rg_nm", "rg", "Rg", "radius_of_gyration_nm"],
        required=True,
    )

    df = df.copy()
    df["architecture_id"] = df[id_col].apply(architecture_id_from_text)
    df["temperature_K"] = (
        df[temp_col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    )
    df["rg_nm_value"] = pd.to_numeric(df[rg_col], errors="coerce")
    df["replicate"] = replicate_label

    frame_col = first_existing_column(
        df,
        ["frame", "frame_index", "ordered_frame", "frame_idx"],
        required=False,
    )
    time_col = first_existing_column(
        df,
        ["time_ns", "time", "t_ns", "Time_ns", "Time (ns)"],
        required=False,
    )

    sort_cols = ["architecture_id", "temperature_K"]
    if time_col is not None:
        sort_cols.append(time_col)
    elif frame_col is not None:
        sort_cols.append(frame_col)

    return df.sort_values(sort_cols).reset_index(drop=True)


def derive_hysteresis_state(rg, enter=4.25, exit_=4.35, midpoint=4.30):
    rg = np.asarray(rg, dtype=float)
    state = np.zeros(len(rg), dtype=np.int8)
    if len(rg) == 0:
        return state

    if rg[0] <= enter:
        state[0] = 1
    elif rg[0] >= exit_:
        state[0] = 0
    else:
        state[0] = 1 if rg[0] < midpoint else 0

    for i in range(1, len(rg)):
        if state[i - 1] == 0:
            state[i] = 1 if rg[i] <= enter else 0
        else:
            state[i] = 0 if rg[i] >= exit_ else 1
    return state


def run_length_events(state, dt):
    state = np.asarray(state, dtype=np.int8)
    if len(state) == 0:
        return []

    events = []
    start = 0
    current = int(state[0])

    for i in range(1, len(state)):
        if int(state[i]) != current:
            events.append(
                {
                    "state": current,
                    "start_idx": start,
                    "end_idx": i - 1,
                    "frames": i - start,
                    "dwell_ns": (i - start) * dt,
                    "left_censored": start == 0,
                    "right_censored": False,
                }
            )
            start = i
            current = int(state[i])

    events.append(
        {
            "state": current,
            "start_idx": start,
            "end_idx": len(state) - 1,
            "frames": len(state) - start,
            "dwell_ns": (len(state) - start) * dt,
            "left_censored": start == 0,
            "right_censored": True,
        }
    )
    return events


def safe_div(a, b):
    if b is None or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def autocorrelation_time_ns(x, dt, max_lag_ns=10.0):
    """
    Integrated autocorrelation time using the initial positive sequence.
    Useful as a simple measure of how slowly Rg decorrelates.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan

    x = x - np.mean(x)
    var = np.dot(x, x) / n
    if var <= 0:
        return 0.0

    max_lag = min(n - 1, max(1, int(round(max_lag_ns / dt))))

    # FFT autocorrelation
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(f * np.conjugate(f), n=size)[:n]
    acf = acf / np.arange(n, 0, -1)
    acf = acf / acf[0]

    tau = 0.5
    for lag in range(1, max_lag + 1):
        if not np.isfinite(acf[lag]) or acf[lag] <= 0:
            break
        tau += acf[lag]
    return float(tau * dt)


def quantile_span(rg, low=0.10, high=0.90):
    if len(rg) == 0:
        return np.nan
    return float(np.quantile(rg, high) - np.quantile(rg, low))


def extract_run_properties(
    group,
    threshold=4.30,
    enter=4.25,
    exit_=4.35,
    sustained_ns=1.0,
    long_ns=2.0,
    fallback_dt=0.01,
):
    rg = pd.to_numeric(group["rg_nm_value"], errors="coerce").dropna().to_numpy()
    if len(rg) < 2:
        return {}

    dt = infer_dt_ns(group, fallback=fallback_dt)
    hstate = derive_hysteresis_state(rg, enter=enter, exit_=exit_, midpoint=threshold)
    sstate = (rg < threshold).astype(np.int8)

    events_h = run_length_events(hstate, dt)
    complete_h = [
        e for e in events_h if not e["left_censored"] and not e["right_censored"]
    ]
    collapsed_h = [e for e in complete_h if e["state"] == 1]
    expanded_h = [e for e in complete_h if e["state"] == 0]

    cd = np.array([e["dwell_ns"] for e in collapsed_h], dtype=float)
    ed = np.array([e["dwell_ns"] for e in expanded_h], dtype=float)
    sustained = cd[cd > sustained_ns]
    long_events = cd[cd > long_ns]

    # transition counts from hysteresis state
    diff = np.diff(hstate)
    n_collapse = int(np.sum(diff == 1))
    n_escape = int(np.sum(diff == -1))
    expanded_time = float(np.sum(hstate == 0) * dt)
    collapsed_time = float(np.sum(hstate == 1) * dt)

    # Rg increments / dynamics
    drg = np.diff(rg)
    abs_drg = np.abs(drg)

    # threshold neighbourhood occupancy
    near_threshold = np.mean((rg >= enter) & (rg <= exit_))

    return {
        "n_frames": len(rg),
        "trajectory_ns": len(rg) * dt,
        "frame_dt_ns": dt,

        # distribution / structure
        "rg_mean_nm": float(np.mean(rg)),
        "rg_median_nm": float(np.median(rg)),
        "rg_sd_nm": float(np.std(rg, ddof=1)),
        "rg_cv": safe_div(float(np.std(rg, ddof=1)), float(np.mean(rg))),
        "rg_q05_nm": float(np.quantile(rg, 0.05)),
        "rg_q10_nm": float(np.quantile(rg, 0.10)),
        "rg_q25_nm": float(np.quantile(rg, 0.25)),
        "rg_q75_nm": float(np.quantile(rg, 0.75)),
        "rg_q90_nm": float(np.quantile(rg, 0.90)),
        "rg_q95_nm": float(np.quantile(rg, 0.95)),
        "rg_iqr_nm": float(np.quantile(rg, 0.75) - np.quantile(rg, 0.25)),
        "rg_q90_q10_span_nm": quantile_span(rg, 0.10, 0.90),
        "rg_range_nm": float(np.max(rg) - np.min(rg)),
        "fraction_below_single_threshold": float(np.mean(sstate)),
        "collapsed_fraction_hysteresis": float(np.mean(hstate)),
        "near_threshold_fraction": float(near_threshold),

        # frame-to-frame mobility
        "mean_abs_dRg_per_frame_nm": float(np.mean(abs_drg)),
        "rms_dRg_per_frame_nm": float(np.sqrt(np.mean(drg ** 2))),
        "sd_dRg_per_frame_nm": float(np.std(drg, ddof=1)),
        "p95_abs_dRg_per_frame_nm": float(np.quantile(abs_drg, 0.95)),
        "mean_abs_dRg_per_ns_nm": safe_div(float(np.mean(abs_drg)), dt),
        "rg_autocorr_time_ns": autocorrelation_time_ns(rg, dt),

        # event counts
        "n_collapsed_events": len(collapsed_h),
        "n_expanded_events": len(expanded_h),
        "n_collapse_transitions": n_collapse,
        "n_escape_transitions": n_escape,
        "collapsed_events_per_100ns": safe_div(len(collapsed_h), len(rg) * dt) * 100,
        "sustained_events_gt1ns": int(np.sum(cd > sustained_ns)),
        "sustained_events_per_100ns": safe_div(int(np.sum(cd > sustained_ns)), len(rg) * dt) * 100,
        "long_events_gt2ns": int(np.sum(cd > long_ns)),
        "long_events_per_100ns": safe_div(int(np.sum(cd > long_ns)), len(rg) * dt) * 100,

        # dwell statistics collapsed
        "collapsed_dwell_mean_ns": float(np.mean(cd)) if len(cd) else np.nan,
        "collapsed_dwell_median_ns": float(np.median(cd)) if len(cd) else np.nan,
        "collapsed_dwell_sd_ns": float(np.std(cd, ddof=1)) if len(cd) > 1 else np.nan,
        "collapsed_dwell_q90_ns": float(np.quantile(cd, 0.90)) if len(cd) else np.nan,
        "collapsed_dwell_q95_ns": float(np.quantile(cd, 0.95)) if len(cd) else np.nan,
        "collapsed_dwell_max_ns": float(np.max(cd)) if len(cd) else np.nan,

        # dwell statistics expanded
        "expanded_dwell_mean_ns": float(np.mean(ed)) if len(ed) else np.nan,
        "expanded_dwell_median_ns": float(np.median(ed)) if len(ed) else np.nan,
        "expanded_dwell_q90_ns": float(np.quantile(ed, 0.90)) if len(ed) else np.nan,

        # persistence
        "fraction_collapses_sustained_gt1ns": safe_div(int(np.sum(cd > sustained_ns)), len(cd)),
        "fraction_collapses_long_gt2ns": safe_div(int(np.sum(cd > long_ns)), len(cd)),
        "long_given_sustained": safe_div(int(np.sum(cd > long_ns)), int(np.sum(cd > sustained_ns))),
        "sustained_dwell_mean_ns": float(np.mean(sustained)) if len(sustained) else np.nan,
        "sustained_dwell_median_ns": float(np.median(sustained)) if len(sustained) else np.nan,
        "sustained_dwell_q90_ns": float(np.quantile(sustained, 0.90)) if len(sustained) else np.nan,

        # rates
        "k_collapse_per_ns": safe_div(n_collapse, expanded_time),
        "k_escape_per_ns": safe_div(n_escape, collapsed_time),
        "k_ratio": safe_div(
            safe_div(n_collapse, expanded_time),
            safe_div(n_escape, collapsed_time),
        ),
    }


# ---------------------------------------------------------------------
# Per-replicate and architecture summaries
# ---------------------------------------------------------------------

def load_per_run_summary(path, replicate_label):
    df = pd.read_csv(path)
    id_col = first_existing_column(
        df,
        ["id", "architecture", "architecture_id", "name", "run"],
        required=True,
    )
    temp_col = first_existing_column(
        df,
        ["temp", "temperature", "temperature_K", "T"],
        required=True,
    )

    df = df.copy()
    df["architecture_id"] = df[id_col].apply(architecture_id_from_text)
    df["temperature_K"] = (
        df[temp_col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    )
    df["replicate"] = replicate_label
    return df


def find_summary_rg_column(df):
    candidates = [
        "mean_rg_nm", "rg_mean_nm", "mean_rg", "Rg_mean_nm",
        "mean_Rg_nm", "avg_rg_nm", "average_rg_nm", "Rg_mean"
    ]
    return first_existing_column(df, candidates, required=False)


def build_per_replicate_table(
    frame_files,
    summary_files,
    threshold,
    enter,
    exit_,
    sustained_ns,
    long_ns,
    fallback_dt,
):
    records = []
    summary_records = []

    for i, frame_path in enumerate(frame_files, start=1):
        rep = f"rep{i:02d}"
        frame_df = runs_from_frame_file(frame_path, rep)

        for (arch, temp), group in frame_df.groupby(
            ["architecture_id", "temperature_K"], sort=True
        ):
            if not np.isfinite(arch):
                continue
            props = extract_run_properties(
                group,
                threshold=threshold,
                enter=enter,
                exit_=exit_,
                sustained_ns=sustained_ns,
                long_ns=long_ns,
                fallback_dt=fallback_dt,
            )
            props.update(
                {
                    "architecture_id": int(arch),
                    "temperature_K": float(temp),
                    "replicate": rep,
                }
            )
            records.append(props)

    per_rep_temp = pd.DataFrame(records)

    # Summaries are optional but useful for cross-checking mean Rg.
    for i, summary_path in enumerate(summary_files, start=1):
        rep = f"rep{i:02d}"
        sdf = load_per_run_summary(summary_path, rep)
        rgcol = find_summary_rg_column(sdf)
        if rgcol is not None:
            tmp = sdf[
                ["architecture_id", "temperature_K", "replicate", rgcol]
            ].copy()
            tmp = tmp.rename(columns={rgcol: "rg_from_per_run_summary_nm"})
            summary_records.append(tmp)

    if summary_records:
        ext_summary = pd.concat(summary_records, ignore_index=True)
        per_rep_temp = per_rep_temp.merge(
            ext_summary,
            on=["architecture_id", "temperature_K", "replicate"],
            how="left",
        )

    # Put 300K and 370K in one architecture/replicate row.
    meta_cols = ["architecture_id", "replicate"]
    value_cols = [
        c for c in per_rep_temp.columns
        if c not in meta_cols + ["temperature_K"]
    ]

    rows = []
    for (arch, rep), g in per_rep_temp.groupby(meta_cols, sort=True):
        row = {"architecture_id": int(arch), "replicate": rep}

        for _, r in g.iterrows():
            t = int(round(float(r["temperature_K"])))
            for col in value_cols:
                row[f"{col}_{t}K"] = r[col]

        # Derived temperature-response quantities
        if "rg_mean_nm_300K" in row and "rg_mean_nm_370K" in row:
            row["delta_rg_370_minus_300_nm"] = (
                row["rg_mean_nm_370K"] - row["rg_mean_nm_300K"]
            )
            row["rg_ratio_370_over_300"] = safe_div(
                row["rg_mean_nm_370K"], row["rg_mean_nm_300K"]
            )
            row["relative_rg_change_percent"] = (
                safe_div(
                    row["rg_mean_nm_370K"] - row["rg_mean_nm_300K"],
                    row["rg_mean_nm_300K"],
                ) * 100.0
            )

        if (
            "collapsed_fraction_hysteresis_300K" in row
            and "collapsed_fraction_hysteresis_370K" in row
        ):
            row["delta_collapsed_fraction_370_minus_300"] = (
                row["collapsed_fraction_hysteresis_370K"]
                - row["collapsed_fraction_hysteresis_300K"]
            )

        if "rg_sd_nm_300K" in row and "rg_sd_nm_370K" in row:
            row["delta_rg_sd_370_minus_300_nm"] = (
                row["rg_sd_nm_370K"] - row["rg_sd_nm_300K"]
            )

        if "rg_autocorr_time_ns_300K" in row and "rg_autocorr_time_ns_370K" in row:
            row["delta_rg_autocorr_time_ns"] = (
                row["rg_autocorr_time_ns_370K"]
                - row["rg_autocorr_time_ns_300K"]
            )

        rows.append(row)

    return per_rep_temp, pd.DataFrame(rows)


def aggregate_architectures(per_rep):
    numeric_cols = [
        c for c in per_rep.columns
        if c not in ["architecture_id", "replicate"]
        and pd.api.types.is_numeric_dtype(per_rep[c])
    ]

    out = []
    for arch, g in per_rep.groupby("architecture_id", sort=True):
        row = {
            "architecture_id": int(arch),
            "n_replicates": int(g["replicate"].nunique()),
        }
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[col + "_mean"] = vals.mean()
            row[col + "_sd"] = vals.std(ddof=1)
            row[col + "_min"] = vals.min()
            row[col + "_max"] = vals.max()
        out.append(row)

    return pd.DataFrame(out)


def property_mean_columns(summary):
    exclude_fragments = [
        "n_frames",
        "trajectory_ns",
        "frame_dt_ns",
        "rg_from_per_run_summary",
    ]
    cols = []
    for c in summary.columns:
        if not c.endswith("_mean"):
            continue
        if any(x in c for x in exclude_fragments):
            continue
        cols.append(c)
    return cols


# ---------------------------------------------------------------------
# Variation / statistics
# ---------------------------------------------------------------------

def variation_summary(summary, mean_cols):
    rows = []
    for col in mean_cols:
        x = pd.to_numeric(summary[col], errors="coerce").dropna()
        if len(x) < 2:
            continue

        mean = float(x.mean())
        sd_between = float(x.std(ddof=1))
        vmin = float(x.min())
        vmax = float(x.max())

        # Mean replicate SD corresponding to this property
        sd_col = col[:-5] + "_sd"
        within_sd = (
            pd.to_numeric(summary[sd_col], errors="coerce").mean()
            if sd_col in summary.columns else np.nan
        )

        rows.append(
            {
                "property": col[:-5],
                "architecture_mean": mean,
                "architecture_sd": sd_between,
                "min": vmin,
                "max": vmax,
                "absolute_range": vmax - vmin,
                "relative_range_percent": (
                    100 * (vmax - vmin) / abs(mean) if mean != 0 else np.nan
                ),
                "cv_between_architectures_percent": (
                    100 * sd_between / abs(mean) if mean != 0 else np.nan
                ),
                "mean_within_architecture_replicate_sd": within_sd,
                "between_sd_over_mean_within_sd": (
                    sd_between / within_sd
                    if pd.notna(within_sd) and within_sd > 0
                    else np.nan
                ),
                "n_architectures": int(len(x)),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["between_sd_over_mean_within_sd", "relative_range_percent"],
            ascending=False,
        ).reset_index(drop=True)
    return df


def correlation_matrices(summary, mean_cols):
    """
    Compute correlations only for properties that actually vary across
    architectures. Constant columns have undefined Pearson/Spearman
    correlations and otherwise generate a very large number of warnings.
    """
    data = summary[mean_cols].copy()
    data.columns = [c[:-5] for c in mean_cols]

    keep = []
    for col in data.columns:
        x = pd.to_numeric(data[col], errors="coerce").dropna()
        if len(x) >= 3 and x.nunique() > 1 and np.nanstd(x.to_numpy(dtype=float)) > 0:
            keep.append(col)

    data = data[keep]
    return data.corr(method="pearson"), data.corr(method="spearman")


def pairwise_long(summary, mean_cols):
    data = summary[["architecture_id"] + mean_cols].copy()
    rename = {c: c[:-5] for c in mean_cols}
    data = data.rename(columns=rename)

    props = []
    for p in rename.values():
        x = pd.to_numeric(data[p], errors="coerce").dropna()
        if len(x) >= 3 and x.nunique() > 1 and np.nanstd(x.to_numpy(dtype=float)) > 0:
            props.append(p)

    rows = []
    for i, p1 in enumerate(props):
        for p2 in props[i + 1:]:
            tmp = data[["architecture_id", p1, p2]].dropna()
            if len(tmp) < 3:
                continue

            x = pd.to_numeric(tmp[p1], errors="coerce")
            y = pd.to_numeric(tmp[p2], errors="coerce")

            if x.nunique() <= 1 or y.nunique() <= 1:
                continue

            pearson = x.corr(y, method="pearson")
            spearman = x.corr(y, method="spearman")

            rows.append(
                {
                    "property_x": p1,
                    "property_y": p2,
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                    "n": len(tmp),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------

def compact_label(name):
    replacements = {
        "_370K": " 370K",
        "_300K": " 300K",
        "collapsed_fraction_hysteresis": "Collapsed fraction",
        "fraction_collapses_sustained_gt1ns": "Fraction sustained >1 ns",
        "sustained_events_per_100ns": "Sustained events / 100 ns",
        "collapsed_events_per_100ns": "Collapsed events / 100 ns",
        "k_collapse_per_ns": "Collapse-entry rate",
        "k_escape_per_ns": "Escape rate",
        "rg_autocorr_time_ns": "Rg autocorr. time",
        "rg_q90_q10_span_nm": "Rg 10–90% span",
        "rg_sd_nm": "Rg SD",
        "rg_iqr_nm": "Rg IQR",
        "delta_rg_370_minus_300_nm": "Delta Rg (370-300)",
        "relative_rg_change_percent": "Relative Rg change",
        "mean_abs_dRg_per_frame_nm": "|dRg| / frame",
        "near_threshold_fraction": "Threshold-band occupancy",
        "sustained_dwell_median_ns": "Median sustained dwell",
    }
    label = name
    for k, v in replacements.items():
        label = label.replace(k, v)
    return label.replace("_", " ")


def select_key_properties(summary):
    preferred = [
        "rg_mean_nm_370K_mean",
        "delta_rg_370_minus_300_nm_mean",
        "rg_sd_nm_370K_mean",
        "rg_q90_q10_span_nm_370K_mean",
        "rg_autocorr_time_ns_370K_mean",
        "collapsed_fraction_hysteresis_370K_mean",
        "k_collapse_per_ns_370K_mean",
        "k_escape_per_ns_370K_mean",
        "collapsed_events_per_100ns_370K_mean",
        "sustained_events_per_100ns_370K_mean",
        "fraction_collapses_sustained_gt1ns_370K_mean",
        "sustained_dwell_median_ns_370K_mean",
        "mean_abs_dRg_per_frame_nm_370K_mean",
        "near_threshold_fraction_370K_mean",
    ]
    return [x for x in preferred if x in summary.columns]


def plot_overview(summary, outdir):
    key = select_key_properties(summary)
    if not key:
        return

    sort_col = (
        "rg_mean_nm_370K_mean"
        if "rg_mean_nm_370K_mean" in summary.columns
        else key[0]
    )
    df = summary.sort_values(sort_col).reset_index(drop=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(
        len(key), 1, figsize=(16, max(10, 2.6 * len(key))), sharex=True
    )
    if len(key) == 1:
        axes = [axes]

    for ax, col in zip(axes, key):
        sdcol = col[:-5] + "_sd"
        y = df[col].to_numpy(float)
        yerr = (
            df[sdcol].to_numpy(float)
            if sdcol in df.columns else np.zeros(len(df))
        )
        ax.errorbar(
            x, y, yerr=yerr,
            fmt="o", markersize=5,
            capsize=2.5, elinewidth=1
        )
        ax.set_ylabel(compact_label(col[:-5]))
        ax.grid(axis="y", alpha=0.2)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df["architecture_id"].astype(str), rotation=90)
    axes[-1].set_xlabel("Architecture ID (ordered by mean Rg at 370 K)")
    fig.suptitle("VGIPKG architecture-property overview — 3-replicate analysis")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(outdir / "property_overview_sorted_by_rg370.png", dpi=250)
    plt.close(fig)


def plot_variation_ranking(var_df, outdir, top_n=25):
    if var_df.empty:
        return

    # Signal relative to replicate noise is the more useful first ranking.
    d = var_df.dropna(subset=["between_sd_over_mean_within_sd"]).copy()
    d = d.sort_values("between_sd_over_mean_within_sd").tail(top_n)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.36 * len(d))))
    ax.barh(
        [compact_label(x) for x in d["property"]],
        d["between_sd_over_mean_within_sd"],
    )
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Between-architecture SD / mean within-architecture replicate SD")
    ax.set_title("Properties showing architecture signal relative to replicate variation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "property_variation_ranking.png", dpi=250)
    plt.close(fig)

    d2 = var_df.dropna(subset=["relative_range_percent"]).copy()
    d2 = d2.sort_values("relative_range_percent").tail(top_n)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.36 * len(d2))))
    ax.barh(
        [compact_label(x) for x in d2["property"]],
        d2["relative_range_percent"],
    )
    ax.set_xlabel("(max - min) / library mean (%)")
    ax.set_title("Relative architecture-to-architecture property range")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "property_relative_variation.png", dpi=250)
    plt.close(fig)


def plot_corr_heatmap(corr, outdir):
    if corr.empty:
        return

    # limit to a useful subset, otherwise the labels become unreadable
    preferred = [
        "rg_mean_nm_370K",
        "delta_rg_370_minus_300_nm",
        "rg_sd_nm_370K",
        "rg_q90_q10_span_nm_370K",
        "rg_autocorr_time_ns_370K",
        "collapsed_fraction_hysteresis_370K",
        "k_collapse_per_ns_370K",
        "k_escape_per_ns_370K",
        "collapsed_events_per_100ns_370K",
        "sustained_events_per_100ns_370K",
        "fraction_collapses_sustained_gt1ns_370K",
        "sustained_dwell_median_ns_370K",
        "mean_abs_dRg_per_frame_nm_370K",
        "near_threshold_fraction_370K",
    ]
    cols = [c for c in preferred if c in corr.columns]
    c = corr.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels([compact_label(x) for x in cols], rotation=90)
    ax.set_yticklabels([compact_label(x) for x in cols])
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation between selected architecture properties")
    fig.tight_layout()
    fig.savefig(outdir / "property_correlation_heatmap.png", dpi=250)
    plt.close(fig)


def plot_scatter_matrix(summary, outdir):
    cols = select_key_properties(summary)
    # Use a smaller subset for a readable matrix
    wanted = [
        "rg_mean_nm_370K_mean",
        "delta_rg_370_minus_300_nm_mean",
        "rg_sd_nm_370K_mean",
        "rg_autocorr_time_ns_370K_mean",
        "k_collapse_per_ns_370K_mean",
        "k_escape_per_ns_370K_mean",
        "fraction_collapses_sustained_gt1ns_370K_mean",
        "sustained_dwell_median_ns_370K_mean",
    ]
    cols = [c for c in wanted if c in summary.columns]
    if len(cols) < 2:
        return

    d = summary[["architecture_id"] + cols].copy()
    labels = [compact_label(c[:-5]) for c in cols]
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x = pd.to_numeric(d[cols[j]], errors="coerce")
            y = pd.to_numeric(d[cols[i]], errors="coerce")

            if i == j:
                ax.hist(x.dropna(), bins=10)
            else:
                ax.scatter(x, y, s=20)
                valid = x.notna() & y.notna()
                if valid.sum() >= 3:
                    r = x[valid].corr(y[valid])
                    ax.text(
                        0.04, 0.92, f"r={r:.2f}",
                        transform=ax.transAxes,
                        va="top", fontsize=8
                    )
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    fig.suptitle("Selected property scatter matrix", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "property_scatter_matrix_selected.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Interactive HTML
# ---------------------------------------------------------------------

def make_interactive_html(summary, variation, outpath):
    key_cols = property_mean_columns(summary)
    display_cols = ["architecture_id", "n_replicates"] + key_cols

    # Long but still manageable (~30 architectures x many properties).
    data = summary[display_cols].copy()
    records_json = data.round(8).to_json(orient="records")

    labels = {
        c: compact_label(c[:-5]) if c.endswith("_mean") else c
        for c in key_cols
    }
    labels_json = pd.Series(labels).to_json()

    var_map = {}
    if not variation.empty:
        for _, r in variation.iterrows():
            var_map[str(r["property"])] = {
                "relative_range_percent": (
                    None if pd.isna(r["relative_range_percent"])
                    else round(float(r["relative_range_percent"]), 3)
                ),
                "signal_to_replicate_sd": (
                    None if pd.isna(r["between_sd_over_mean_within_sd"])
                    else round(float(r["between_sd_over_mean_within_sd"]), 3)
                ),
            }

    import json
    var_json = json.dumps(var_map)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>VGIPKG Architecture Property Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 22px;
    color: #222;
}}
.controls {{
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    align-items: end;
    margin-bottom: 16px;
}}
label {{
    font-size: 13px;
    font-weight: 600;
}}
select {{
    min-width: 280px;
    padding: 7px;
    margin-top: 4px;
}}
#bar, #scatter {{
    width: 100%;
    min-height: 580px;
}}
.info {{
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 10px 13px;
    margin-bottom: 14px;
    font-size: 13px;
}}
</style>
</head>
<body>
<h2>VGIPKG architecture property explorer</h2>
<div class="info" id="info">
Choose a property. Architectures will be sorted from low to high. Error bars are replicate SD.
</div>

<div class="controls">
  <div>
    <label for="sortProp">Sort / bar property</label><br>
    <select id="sortProp"></select>
  </div>
  <div>
    <label for="scatterX">Scatter x property</label><br>
    <select id="scatterX"></select>
  </div>
  <div>
    <label for="scatterY">Scatter y property</label><br>
    <select id="scatterY"></select>
  </div>
</div>

<div id="bar"></div>
<hr>
<div id="scatter"></div>

<script>
const rows = {records_json};
const labels = {labels_json};
const variation = {var_json};
const properties = Object.keys(labels);

function sdColumn(meanCol) {{
    return meanCol.endsWith("_mean")
        ? meanCol.slice(0, -5) + "_sd"
        : null;
}}

function labelFor(c) {{
    return labels[c] || c;
}}

function fillSelect(el, selected) {{
    properties.forEach(p => {{
        const o = document.createElement("option");
        o.value = p;
        o.textContent = labelFor(p);
        if (p === selected) o.selected = true;
        el.appendChild(o);
    }});
}}

const sortProp = document.getElementById("sortProp");
const scatterX = document.getElementById("scatterX");
const scatterY = document.getElementById("scatterY");

const defaultSort = properties.includes("rg_mean_nm_370K_mean")
    ? "rg_mean_nm_370K_mean"
    : properties[0];
const defaultX = properties.includes("rg_mean_nm_370K_mean")
    ? "rg_mean_nm_370K_mean"
    : properties[0];
const defaultY = properties.includes("k_collapse_per_ns_370K_mean")
    ? "k_collapse_per_ns_370K_mean"
    : properties[Math.min(1, properties.length - 1)];

fillSelect(sortProp, defaultSort);
fillSelect(scatterX, defaultX);
fillSelect(scatterY, defaultY);

function finiteRows(prop) {{
    return rows.filter(r => Number.isFinite(Number(r[prop])));
}}

function updateBar() {{
    const p = sortProp.value;
    const sd = sdColumn(p);
    const sorted = finiteRows(p)
        .slice()
        .sort((a,b) => Number(a[p]) - Number(b[p]));

    const x = sorted.map(r => String(r.architecture_id));
    const y = sorted.map(r => Number(r[p]));
    const err = sd
        ? sorted.map(r => Number.isFinite(Number(r[sd])) ? Number(r[sd]) : 0)
        : sorted.map(() => 0);

    Plotly.react("bar", [{{
        type: "bar",
        x: x,
        y: y,
        error_y: {{
            type: "data",
            array: err,
            visible: true
        }},
        customdata: sorted.map(r => r.n_replicates),
        hovertemplate:
            "Architecture %{{x}}<br>" +
            labelFor(p) + ": %{{y:.5g}}<br>" +
            "n replicates: %{{customdata}}<extra></extra>"
    }}], {{
        title: labelFor(p) + " — architectures sorted low → high",
        xaxis: {{title: "Architecture ID"}},
        yaxis: {{title: labelFor(p)}},
        margin: {{t: 60, b: 90}}
    }}, {{responsive: true}});

    const base = p.endsWith("_mean") ? p.slice(0,-5) : p;
    const v = variation[base];
    document.getElementById("info").textContent = v
      ? `Property: ${{labelFor(p)}} | relative library range: ${{v.relative_range_percent ?? "NA"}}% | between-architecture SD / mean replicate SD: ${{v.signal_to_replicate_sd ?? "NA"}}`
      : `Property: ${{labelFor(p)}}`;
}}

function updateScatter() {{
    const px = scatterX.value;
    const py = scatterY.value;
    const valid = rows.filter(r =>
        Number.isFinite(Number(r[px])) &&
        Number.isFinite(Number(r[py]))
    );

    const x = valid.map(r => Number(r[px]));
    const y = valid.map(r => Number(r[py]));
    const text = valid.map(r => String(r.architecture_id));

    let r = NaN;
    if (x.length >= 3) {{
        const mx = x.reduce((a,b)=>a+b,0)/x.length;
        const my = y.reduce((a,b)=>a+b,0)/y.length;
        let num=0, dx=0, dy=0;
        for (let i=0;i<x.length;i++) {{
            num += (x[i]-mx)*(y[i]-my);
            dx += (x[i]-mx)*(x[i]-mx);
            dy += (y[i]-my)*(y[i]-my);
        }}
        r = num / Math.sqrt(dx*dy);
    }}

    Plotly.react("scatter", [{{
        type: "scatter",
        mode: "markers+text",
        x: x,
        y: y,
        text: text,
        textposition: "top center",
        marker: {{size: 9}},
        hovertemplate:
            "Architecture %{{text}}<br>" +
            labelFor(px) + ": %{{x:.5g}}<br>" +
            labelFor(py) + ": %{{y:.5g}}<extra></extra>"
    }}], {{
        title: labelFor(px) + " vs " + labelFor(py) +
               (Number.isFinite(r) ? ` (Pearson r=${{r.toFixed(3)}})` : ""),
        xaxis: {{title: labelFor(px)}},
        yaxis: {{title: labelFor(py)}},
        margin: {{t: 70, b: 80}}
    }}, {{responsive: true}});
}}

sortProp.addEventListener("change", updateBar);
scatterX.addEventListener("change", updateScatter);
scatterY.addEventListener("change", updateScatter);

updateBar();
updateScatter();
</script>
</body>
</html>
"""
    outpath.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--rg-files", nargs="+", required=True,
        help="all_rg_values.csv files, one per replicate"
    )
    ap.add_argument(
        "--summary-files", nargs="*", default=[],
        help="per_run_summary.csv files, same replicate order as rg-files"
    )
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--threshold", type=float, default=4.30)
    ap.add_argument("--collapse-enter-nm", type=float, default=4.25)
    ap.add_argument("--escape-exit-nm", type=float, default=4.35)
    ap.add_argument("--frame-time-ns", type=float, default=0.01)
    ap.add_argument("--sustained-threshold-ns", type=float, default=1.0)
    ap.add_argument("--long-threshold-ns", type=float, default=2.0)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.summary_files and len(args.summary_files) != len(args.rg_files):
        raise ValueError(
            "--summary-files must contain the same number of files as --rg-files "
            "(or omit --summary-files entirely)."
        )

    per_temp, per_rep = build_per_replicate_table(
        args.rg_files,
        args.summary_files,
        threshold=args.threshold,
        enter=args.collapse_enter_nm,
        exit_=args.escape_exit_nm,
        sustained_ns=args.sustained_threshold_ns,
        long_ns=args.long_threshold_ns,
        fallback_dt=args.frame_time_ns,
    )

    summary = aggregate_architectures(per_rep)
    mean_cols = property_mean_columns(summary)

    variation = variation_summary(summary, mean_cols)
    pearson, spearman = correlation_matrices(summary, mean_cols)
    pairwise = pairwise_long(summary, mean_cols)

    per_temp.to_csv(
        outdir / "properties_per_replicate_per_temperature.csv",
        index=False
    )
    per_rep.to_csv(
        outdir / "architecture_properties_per_replicate.csv",
        index=False
    )
    summary.to_csv(
        outdir / "architecture_properties_summary.csv",
        index=False
    )
    variation.to_csv(
        outdir / "property_variation_summary.csv",
        index=False
    )
    pearson.to_csv(
        outdir / "property_correlations_pearson.csv"
    )
    spearman.to_csv(
        outdir / "property_correlations_spearman.csv"
    )
    pairwise.to_csv(
        outdir / "property_pairwise_long.csv",
        index=False
    )

    plot_overview(summary, outdir)
    plot_variation_ranking(variation, outdir)
    plot_corr_heatmap(pearson, outdir)
    plot_scatter_matrix(summary, outdir)

    make_interactive_html(
        summary,
        variation,
        outdir / "interactive_architecture_property_explorer.html",
    )

    print("\nAnalysis complete.")
    print(f"Architectures: {summary['architecture_id'].nunique()}")
    print(f"Replicates found: {per_rep['replicate'].nunique()}")
    print(f"Properties summarized: {len(mean_cols)}")

    print("\nTop architecture-sensitive properties "
          "(between-architecture SD / mean replicate SD):")
    if not variation.empty:
        cols = [
            "property",
            "relative_range_percent",
            "between_sd_over_mean_within_sd",
        ]
        print(variation[cols].head(20).to_string(index=False))

    print("\nInteractive explorer:")
    print(outdir / "interactive_architecture_property_explorer.html")


if __name__ == "__main__":
    main()
