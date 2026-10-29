#!/usr/bin/env python
"""
analyse_charge_distribution_rg_v2.py

Design Rule 4 — Distributing Charge
Fast first-pass Rg and collapse analysis for:
    VGEG = VPGVG + VPGEG
    VGKG = VPGVG + VPGKG
    IGKG = VPGIG + VPGKG

Key features
------------
- Uses ALL frames for numerical statistics.
- Collapse threshold configurable; default = 4.61 nm.
- Standardizes architecture names into:
    even
    smallblocks
    mediumblocks
    largeblocks
    charge_center
    charge_ends
    diblock
    randomshuffle
- Correctly handles naming such as:
    EGcenter_VGends
    EGendcaps
    VGcenter_EGends
  and analogous KG/IG variants.
- Prints progress and elapsed time.
- Reads only columns needed from the large CSV files.
- Does NOT write a huge combined 14.4-million-row CSV.
- Calculates Rg range 90, compact/expanded Rg, transition counts, dwell times, k_collapse and k_escape.\n- Creates compact first-pass heatmaps and trend plots.
- Distribution plots are optional and sampled only for visualization.
- Performs strict sanity checks on design/frame counts.

Expected input
--------------
ROOT/
    VGEG_detailed_stride1/all_rg_values.csv
    VGKG_detailed_stride1/all_rg_values.csv
    IGKG_detailed_stride1/all_rg_values.csv

Example
-------
python analyse_charge_distribution_rg_v2.py --root "E:\\Master thesis\\calvados_tea\\analysis_library_1" --outdir "E:\\Master thesis\\calvados_tea\\analysis_library_1\\charge_distribution_rg_analysis_v2" --threshold 4.61
"""

from pathlib import Path
import argparse
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

LIBRARIES = {
    "VGEG": {
        "folder": "VGEG_detailed_stride1",
        "label": "VPGVG + VPGEG",
    },
    "VGKG": {
        "folder": "VGKG_detailed_stride1",
        "label": "VPGVG + VPGKG",
    },
    "IGKG": {
        "folder": "IGKG_detailed_stride1",
        "label": "VPGIG + VPGKG",
    },
}

RATIO_ORDER = [f"R{i}" for i in range(1, 7)]

ARCHITECTURE_ORDER = [
    "even",
    "smallblocks",
    "mediumblocks",
    "largeblocks",
    "charge_center",
    "charge_ends",
    "diblock",
    "randomshuffle",
]

ARCHITECTURE_LABELS = {
    "even": "Even",
    "smallblocks": "Small blocks",
    "mediumblocks": "Medium blocks",
    "largeblocks": "Large blocks",
    "charge_center": "Charge center",
    "charge_ends": "Charge ends",
    "diblock": "Diblock",
    "randomshuffle": "Random shuffle",
}

START_TIME = time.time()


def progress(message):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:8.1f} s] {message}", flush=True)


# ============================================================
# NAME PARSING
# ============================================================

def parse_design_name(name):
    """
    Parse ID, library, ratio and standardized architecture.

    Examples:
        227_VGEG_R1_EGcenter_VGends  -> charge_center
        228_VGEG_R1_EGendcaps       -> charge_ends
        252_VGEG_R4_VGcenter_EGends -> charge_ends

    General rule:
    EG or KG is the charged motif.
    Therefore:
        charged motif center -> charge_center
        charged motif ends   -> charge_ends
    """
    name = str(name)
    lower = name.lower()

    id_match = re.match(r"^(\d+)_", name)
    design_id = int(id_match.group(1)) if id_match else np.nan

    lib_match = re.search(r"_(VGEG|VGKG|IGKG)_", name, re.IGNORECASE)
    library = lib_match.group(1).upper() if lib_match else None

    ratio_match = re.search(r"_R([1-6])_", name, re.IGNORECASE)
    ratio = f"R{ratio_match.group(1)}" if ratio_match else None

    if "randomshuffle" in lower:
        architecture = "randomshuffle"
    elif "smallblocks" in lower:
        architecture = "smallblocks"
    elif "mediumblocks" in lower:
        architecture = "mediumblocks"
    elif "largeblocks" in lower:
        architecture = "largeblocks"
    elif "diblock" in lower:
        architecture = "diblock"
    elif "_even" in lower or lower.endswith("even"):
        architecture = "even"

    # Charged motif explicitly in center
    elif "egcenter" in lower or "kgcenter" in lower:
        architecture = "charge_center"

    # Charged motif explicitly at ends/endcaps
    elif "egendcaps" in lower or "kgendcaps" in lower:
        architecture = "charge_ends"
    elif "egends" in lower or "kgends" in lower:
        architecture = "charge_ends"

    else:
        architecture = "unknown"

    return design_id, library, ratio, architecture


# ============================================================
# INPUT
# ============================================================

def load_library(csv_path, expected_library, threshold):
    progress(f"Loading {expected_library}: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)

    required = ["id", "rg_nm"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns {missing}. "
            f"Available columns: {columns}"
        )

    # Read only what is useful for this analysis.
    wanted = [
        "id",
        "rg_nm",
        "architecture_type",
        "group",
        "temp",
        "time_ns",
        "collapsed",
        "threshold_nm",
    ]
    usecols = [c for c in wanted if c in columns]

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"id": "string"},
    )

    progress(f"{expected_library}: loaded {len(df):,} frames")

    unique_ids = df["id"].dropna().unique()

    parsed_lookup = {}
    for design_name in unique_ids:
        parsed_lookup[str(design_name)] = parse_design_name(design_name)

    lookup_df = pd.DataFrame(
        [
            (name, *parsed)
            for name, parsed in parsed_lookup.items()
        ],
        columns=[
            "id",
            "design_id",
            "parsed_library",
            "ratio",
            "architecture",
        ],
    )

    # Keep string dtype compatible for merge.
    lookup_df["id"] = lookup_df["id"].astype("string")

    df = df.merge(lookup_df, on="id", how="left", validate="many_to_one")

    df["library"] = df["parsed_library"].fillna(expected_library)
    df["rg_nm"] = pd.to_numeric(df["rg_nm"], errors="coerce")
    df = df.dropna(subset=["rg_nm"]).copy()

    # Always recalculate using requested threshold.
    df["collapsed_461"] = df["rg_nm"] < threshold

    unknown = sorted(
        df.loc[df["architecture"] == "unknown", "id"]
        .dropna()
        .unique()
    )
    if unknown:
        print("\nWARNING: unknown architecture IDs:")
        for x in unknown:
            print(f"  {x}")
        raise ValueError(
            f"{expected_library}: {len(unknown)} designs could not be classified."
        )

    progress(
        f"{expected_library}: parsed {df['id'].nunique()} designs; "
        f"overall collapse = {100 * df['collapsed_461'].mean():.2f}%"
    )

    return df


# ============================================================
# SANITY CHECKS
# ============================================================

def sanity_check(df, library, expected_frames_per_design=100000):
    progress(f"{library}: running sanity checks")

    errors = []

    n_designs = df["id"].nunique()
    if n_designs != 48:
        errors.append(f"Expected 48 unique designs, found {n_designs}.")

    ratio_counts = df.groupby("ratio", observed=True).size()
    for ratio in RATIO_ORDER:
        value = int(ratio_counts.get(ratio, 0))
        expected = expected_frames_per_design * 8
        if value != expected:
            errors.append(
                f"{ratio}: expected {expected:,} frames, found {value:,}."
            )

    arch_counts = df.groupby("architecture", observed=True).size()
    for arch in ARCHITECTURE_ORDER:
        value = int(arch_counts.get(arch, 0))
        expected = expected_frames_per_design * 6
        if value != expected:
            errors.append(
                f"{arch}: expected {expected:,} frames, found {value:,}."
            )

    design_counts = df.groupby("id", observed=True).size()
    bad_designs = design_counts[design_counts != expected_frames_per_design]

    if len(bad_designs):
        errors.append(
            f"{len(bad_designs)} designs do not contain exactly "
            f"{expected_frames_per_design:,} frames."
        )

    combo_counts = (
        df.groupby(["ratio", "architecture"], observed=True)
        .size()
    )
    for ratio in RATIO_ORDER:
        for arch in ARCHITECTURE_ORDER:
            value = int(combo_counts.get((ratio, arch), 0))
            if value != expected_frames_per_design:
                errors.append(
                    f"{ratio}/{arch}: expected "
                    f"{expected_frames_per_design:,}, found {value:,}."
                )

    if errors:
        print(f"\nSANITY CHECK WARNING — {library}")
        for error in errors:
            print(f"  - {error}")
        print()
    else:
        progress(
            f"{library}: sanity check PASSED "
            f"(48 designs, 6 ratios × 8 architectures)"
        )

    # Print useful architecture counts.
    print(f"\n{library} frames by architecture:")
    print(
        arch_counts.reindex(ARCHITECTURE_ORDER, fill_value=0)
        .to_string()
    )
    print()

    return errors


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(df, library):
    progress(f"{library}: calculating exact statistics on all frames")

    group_cols = [
        "library",
        "ratio",
        "architecture",
        "design_id",
        "id",
    ]

    grouped = df.groupby(group_cols, observed=True, sort=False)

    summary = grouped.agg(
        n_frames=("rg_nm", "size"),
        mean_Rg=("rg_nm", "mean"),
        std_Rg=("rg_nm", "std"),
        median_Rg=("rg_nm", "median"),
        min_Rg=("rg_nm", "min"),
        max_Rg=("rg_nm", "max"),
        collapsed_frames=("collapsed_461", "sum"),
        collapsed_fraction=("collapsed_461", "mean"),
    ).reset_index()

    # Quantiles calculated once per design.
    q = grouped["rg_nm"].quantile([0.05, 0.25, 0.75, 0.95]).unstack()

    q = q.rename(
        columns={
            0.05: "q05_Rg",
            0.25: "q25_Rg",
            0.75: "q75_Rg",
            0.95: "q95_Rg",
        }
    ).reset_index()

    summary = summary.merge(q, on=group_cols, how="left")

    summary["IQR_Rg"] = summary["q75_Rg"] - summary["q25_Rg"]
    summary["Rg_range90"] = summary["q95_Rg"] - summary["q05_Rg"]
    summary["collapsed_percent"] = 100.0 * summary["collapsed_fraction"]

    # State-specific conformational sizes.
    compact = (
        df.loc[df["collapsed_461"]]
        .groupby(group_cols, observed=True)["rg_nm"]
        .agg(mean_compact_Rg="mean", median_compact_Rg="median")
        .reset_index()
    )
    expanded = (
        df.loc[~df["collapsed_461"]]
        .groupby(group_cols, observed=True)["rg_nm"]
        .agg(mean_expanded_Rg="mean", median_expanded_Rg="median")
        .reset_index()
    )
    summary = summary.merge(compact, on=group_cols, how="left")
    summary = summary.merge(expanded, on=group_cols, how="left")

    # Dynamics: contiguous state transitions and dwell times.
    # Use time_ns when available; otherwise infer a constant frame interval.
    kinetic_rows = []
    progress(f"{library}: calculating collapse/escape kinetics")

    for keys, g in df.groupby(group_cols, observed=True, sort=False):
        if "time_ns" in g.columns:
            g = g.sort_values("time_ns")
            times = pd.to_numeric(g["time_ns"], errors="coerce").to_numpy(dtype=float)
        else:
            g = g.copy()
            times = np.arange(len(g), dtype=float)

        states = g["collapsed_461"].to_numpy(dtype=bool)
        n = len(states)

        if n < 2:
            row = dict(zip(group_cols, keys))
            row.update({
                "n_collapse_transitions": 0,
                "n_escape_transitions": 0,
                "n_total_transitions": 0,
                "mean_compact_dwell_ns": np.nan,
                "mean_expanded_dwell_ns": np.nan,
                "total_compact_time_ns": np.nan,
                "total_expanded_time_ns": np.nan,
                "k_collapse_per_ns": np.nan,
                "k_escape_per_ns": np.nan,
            })
            kinetic_rows.append(row)
            continue

        # Robust dt estimate.
        dt_candidates = np.diff(times)
        dt_candidates = dt_candidates[np.isfinite(dt_candidates) & (dt_candidates > 0)]
        dt = float(np.median(dt_candidates)) if len(dt_candidates) else 1.0

        changes = np.flatnonzero(states[1:] != states[:-1]) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, n]

        dwell_states = states[starts]
        dwell_frames = ends - starts
        dwell_times = dwell_frames.astype(float) * dt

        collapse_transitions = int(np.sum((~states[:-1]) & states[1:]))
        escape_transitions = int(np.sum(states[:-1] & (~states[1:])))

        compact_dwells = dwell_times[dwell_states]
        expanded_dwells = dwell_times[~dwell_states]

        total_compact_time = float(np.sum(states) * dt)
        total_expanded_time = float(np.sum(~states) * dt)

        # Continuous-time state-exit rate estimates:
        # expanded -> compact events / total expanded exposure time
        # compact -> expanded events / total compact exposure time
        k_collapse = (
            collapse_transitions / total_expanded_time
            if total_expanded_time > 0 else np.nan
        )
        k_escape = (
            escape_transitions / total_compact_time
            if total_compact_time > 0 else np.nan
        )

        row = dict(zip(group_cols, keys))
        row.update({
            "n_collapse_transitions": collapse_transitions,
            "n_escape_transitions": escape_transitions,
            "n_total_transitions": collapse_transitions + escape_transitions,
            "mean_compact_dwell_ns": (
                float(np.mean(compact_dwells)) if len(compact_dwells) else np.nan
            ),
            "mean_expanded_dwell_ns": (
                float(np.mean(expanded_dwells)) if len(expanded_dwells) else np.nan
            ),
            "total_compact_time_ns": total_compact_time,
            "total_expanded_time_ns": total_expanded_time,
            "k_collapse_per_ns": k_collapse,
            "k_escape_per_ns": k_escape,
        })
        kinetic_rows.append(row)

    kinetics = pd.DataFrame(kinetic_rows)
    summary = summary.merge(kinetics, on=group_cols, how="left")

    # Helpful flag: kinetics based on very few events should not be overinterpreted.
    summary["kinetics_well_sampled"] = (
        (summary["n_collapse_transitions"] >= 10)
        & (summary["n_escape_transitions"] >= 10)
    )

    progress(f"{library}: statistics complete")

    return summary


# ============================================================
# PLOTTING HELPERS
# ============================================================

def prepare_pivot(summary, library, value_col):
    subset = summary[summary["library"] == library]

    pivot = subset.pivot_table(
        index="architecture",
        columns="ratio",
        values=value_col,
        aggfunc="mean",
        observed=True,
    )

    return pivot.reindex(
        index=ARCHITECTURE_ORDER,
        columns=RATIO_ORDER,
    )


def plot_heatmap(
    summary,
    library,
    library_label,
    value_col,
    colorbar_label,
    title_suffix,
    output_file,
    decimals=2,
    vmin=None,
    vmax=None,
):
    pivot = prepare_pivot(summary, library, value_col)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    im = ax.imshow(
        pivot.values.astype(float),
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(RATIO_ORDER)))
    ax.set_xticklabels(RATIO_ORDER)

    ax.set_yticks(np.arange(len(ARCHITECTURE_ORDER)))
    ax.set_yticklabels(
        [ARCHITECTURE_LABELS[x] for x in ARCHITECTURE_ORDER]
    )

    ax.set_xlabel("Composition ratio")
    ax.set_ylabel("Charge architecture")
    ax.set_title(f"{library_label}\n{title_suffix}")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    values = pivot.values.astype(float)
    finite = values[np.isfinite(values)]

    if len(finite):
        midpoint = (np.nanmin(finite) + np.nanmax(finite)) / 2
    else:
        midpoint = 0

    for i in range(len(ARCHITECTURE_ORDER)):
        for j in range(len(RATIO_ORDER)):
            value = values[i, j]
            if np.isfinite(value):
                # White/black text improves readability without fixing plot colors.
                text_color = "white" if value < midpoint else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trends(
    summary,
    library,
    library_label,
    value_col,
    ylabel,
    title_suffix,
    output_file,
    threshold=None,
):
    subset = summary[summary["library"] == library].copy()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for architecture in ARCHITECTURE_ORDER:
        arch = subset[subset["architecture"] == architecture]

        y = []
        for ratio in RATIO_ORDER:
            values = arch.loc[arch["ratio"] == ratio, value_col]
            y.append(values.mean() if len(values) else np.nan)

        ax.plot(
            RATIO_ORDER,
            y,
            marker="o",
            linewidth=1.5,
            label=ARCHITECTURE_LABELS[architecture],
        )

    if threshold is not None:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"Collapse threshold ({threshold:.2f} nm)",
        )

    ax.set_xlabel("Composition ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{library_label}\n{title_suffix}")

    if value_col == "collapsed_percent":
        ax.set_ylim(bottom=0)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_library_heatmap(
    summary,
    value_col,
    output_file,
    colorbar_label,
    title,
    decimals=2,
):
    """
    Rows = library × architecture.
    Useful compact overview before detailed interpretation.
    """
    rows = []
    labels = []

    for library in LIBRARIES:
        for architecture in ARCHITECTURE_ORDER:
            values = []
            for ratio in RATIO_ORDER:
                x = summary.loc[
                    (summary["library"] == library)
                    & (summary["architecture"] == architecture)
                    & (summary["ratio"] == ratio),
                    value_col,
                ]
                values.append(x.mean() if len(x) else np.nan)

            rows.append(values)
            labels.append(
                f"{library} | {ARCHITECTURE_LABELS[architecture]}"
            )

    matrix = np.asarray(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 13))

    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(RATIO_ORDER)))
    ax.set_xticklabels(RATIO_ORDER)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Composition ratio")
    ax.set_ylabel("Library | architecture")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    finite = matrix[np.isfinite(matrix)]
    midpoint = (
        (np.nanmin(finite) + np.nanmax(finite)) / 2
        if len(finite)
        else 0
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                text_color = "white" if value < midpoint else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# OPTIONAL DISTRIBUTIONS
# ============================================================

def plot_sampled_distributions(
    df,
    library,
    library_label,
    ratio,
    threshold,
    output_file,
    sample_per_architecture,
):
    subset = df[df["ratio"] == ratio]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    all_sampled = []

    for architecture in ARCHITECTURE_ORDER:
        values = subset.loc[
            subset["architecture"] == architecture,
            "rg_nm",
        ].dropna()

        if len(values) > sample_per_architecture:
            values = values.sample(
                n=sample_per_architecture,
                random_state=42,
            )

        if len(values):
            all_sampled.append(values.to_numpy())

    if not all_sampled:
        plt.close(fig)
        return

    pooled = np.concatenate(all_sampled)
    low = np.quantile(pooled, 0.001)
    high = np.quantile(pooled, 0.999)
    bins = np.linspace(low, high, 100)

    for architecture in ARCHITECTURE_ORDER:
        values = subset.loc[
            subset["architecture"] == architecture,
            "rg_nm",
        ].dropna()

        if len(values) > sample_per_architecture:
            values = values.sample(
                n=sample_per_architecture,
                random_state=42,
            )

        if not len(values):
            continue

        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            label=ARCHITECTURE_LABELS[architecture],
        )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Collapse threshold ({threshold:.2f} nm)",
    )

    ax.set_xlabel("Rg (nm)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{library_label} — {ratio}\nSampled Rg distributions")
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast Rg and collapse analysis for Design Rule 4: "
            "distributing charge."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing the three *_detailed_stride1 folders.",
    )

    parser.add_argument(
        "--outdir",
        default="charge_distribution_rg_analysis_v2",
        help="Output directory.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.61,
        help="Collapse threshold in nm. Default = 4.61.",
    )

    parser.add_argument(
        "--expected-frames",
        type=int,
        default=100000,
        help=(
            "Expected frames per design for sanity checking. "
            "Default = 100000."
        ),
    )

    parser.add_argument(
        "--distributions",
        action="store_true",
        help=(
            "Also create sampled Rg distribution plots. "
            "Off by default for a faster first pass."
        ),
    )

    parser.add_argument(
        "--distribution-sample",
        type=int,
        default=20000,
        help=(
            "Maximum frames sampled per architecture for distribution plots. "
            "Default = 20000."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    plot_dir = outdir / "plots"

    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print("DESIGN RULE 4 — DISTRIBUTING CHARGE — FAST Rg ANALYSIS V2")
    print("=" * 78)
    print(f"Root:             {root}")
    print(f"Output:           {outdir}")
    print(f"Collapse threshold: {args.threshold:.2f} nm")
    print(f"Distributions:    {'ON' if args.distributions else 'OFF'}")
    print("=" * 78)
    print()

    summaries = []
    loaded_for_optional_plots = {}
    sanity_rows = []

    # --------------------------------------------------------
    # LOAD + SUMMARIZE ONE LIBRARY AT A TIME
    # This avoids holding all 14.4M rows simultaneously.
    # --------------------------------------------------------

    for library, info in LIBRARIES.items():
        csv_path = root / info["folder"] / "all_rg_values.csv"

        if not csv_path.exists():
            progress(f"WARNING: missing {library}: {csv_path}")
            continue

        df = load_library(
            csv_path=csv_path,
            expected_library=library,
            threshold=args.threshold,
        )

        errors = sanity_check(
            df,
            library,
            expected_frames_per_design=args.expected_frames,
        )

        sanity_rows.append(
            {
                "library": library,
                "n_frames": len(df),
                "n_designs": df["id"].nunique(),
                "n_errors": len(errors),
                "sanity_passed": len(errors) == 0,
                "overall_mean_Rg": df["rg_nm"].mean(),
                "overall_median_Rg": df["rg_nm"].median(),
                "overall_collapsed_percent": (
                    100 * df["collapsed_461"].mean()
                ),
            }
        )

        summary = calculate_summary(df, library)
        summaries.append(summary)

        # Save each library immediately.
        summary.to_csv(
            outdir / f"rg_summary_{library}.csv",
            index=False,
        )

        # Create the useful first-pass plots immediately.
        progress(f"{library}: creating first-pass plots")

        plot_heatmap(
            summary,
            library,
            info["label"],
            "mean_Rg",
            "Mean Rg (nm)",
            "Mean radius of gyration",
            plot_dir / f"01_mean_rg_heatmap_{library}.png",
            decimals=2,
        )

        plot_heatmap(
            summary,
            library,
            info["label"],
            "collapsed_percent",
            "Collapsed population (%)",
            f"Collapsed population (Rg < {args.threshold:.2f} nm)",
            plot_dir / f"02_collapse_heatmap_{library}.png",
            decimals=1,
            vmin=0,
            vmax=100,
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "mean_Rg",
            "Mean Rg (nm)",
            "Mean Rg across charge architectures",
            plot_dir / f"03_rg_trends_{library}.png",
            threshold=args.threshold,
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "collapsed_percent",
            "Collapsed population (%)",
            f"Collapse across charge architectures (Rg < {args.threshold:.2f} nm)",
            plot_dir / f"04_collapse_trends_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "Rg_range90",
            "Rg range 90 (nm)",
            "Conformational Rg range (Q95 - Q05)",
            plot_dir / f"05_rg_range90_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "mean_compact_Rg",
            "Mean compact Rg (nm)",
            f"Mean Rg within compact state (Rg < {args.threshold:.2f} nm)",
            plot_dir / f"06_mean_compact_rg_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "mean_expanded_Rg",
            "Mean expanded Rg (nm)",
            f"Mean Rg within expanded state (Rg >= {args.threshold:.2f} nm)",
            plot_dir / f"07_mean_expanded_rg_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "k_collapse_per_ns",
            "k collapse (ns^-1)",
            "Expanded to compact transition rate",
            plot_dir / f"08_k_collapse_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "k_escape_per_ns",
            "k escape (ns^-1)",
            "Compact to expanded transition rate",
            plot_dir / f"09_k_escape_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "n_total_transitions",
            "Number of transitions",
            "Observed compact/expanded transitions",
            plot_dir / f"10_transition_count_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "mean_compact_dwell_ns",
            "Mean compact dwell time (ns)",
            "Mean contiguous compact-state dwell time",
            plot_dir / f"11_compact_dwell_{library}.png",
        )

        plot_trends(
            summary,
            library,
            info["label"],
            "mean_expanded_dwell_ns",
            "Mean expanded dwell time (ns)",
            "Mean contiguous expanded-state dwell time",
            plot_dir / f"12_expanded_dwell_{library}.png",
        )

        if args.distributions:
            progress(
                f"{library}: creating sampled distribution plots "
                f"({args.distribution_sample:,} max per architecture)"
            )

            for ratio in RATIO_ORDER:
                plot_sampled_distributions(
                    df,
                    library,
                    info["label"],
                    ratio,
                    args.threshold,
                    plot_dir / f"13_rg_distribution_{library}_{ratio}.png",
                    args.distribution_sample,
                )

        # Free the multi-million-row dataframe before next library.
        del df

        progress(f"{library}: complete")

    if not summaries:
        raise FileNotFoundError(
            "None of the expected all_rg_values.csv files were found."
        )

    # --------------------------------------------------------
    # COMBINED SMALL SUMMARY
    # --------------------------------------------------------

    progress("Combining summary tables")

    combined_summary = pd.concat(summaries, ignore_index=True)

    combined_summary["ratio"] = pd.Categorical(
        combined_summary["ratio"],
        categories=RATIO_ORDER,
        ordered=True,
    )

    combined_summary["architecture"] = pd.Categorical(
        combined_summary["architecture"],
        categories=ARCHITECTURE_ORDER,
        ordered=True,
    )

    combined_summary = combined_summary.sort_values(
        ["library", "ratio", "architecture"]
    )

    combined_summary.to_csv(
        outdir / "rg_summary_all_libraries.csv",
        index=False,
    )

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(
        outdir / "sanity_and_overall_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MATRICES
    # --------------------------------------------------------

    mean_matrix = combined_summary.pivot_table(
        index=["library", "architecture"],
        columns="ratio",
        values="mean_Rg",
        observed=True,
    )

    mean_matrix.to_csv(
        outdir / "mean_rg_matrix.csv"
    )

    collapse_matrix = combined_summary.pivot_table(
        index=["library", "architecture"],
        columns="ratio",
        values="collapsed_percent",
        observed=True,
    )

    collapse_matrix.to_csv(
        outdir / "collapsed_percent_matrix.csv"
    )

    # --------------------------------------------------------
    # RANKINGS
    # --------------------------------------------------------

    ranking = combined_summary.copy()

    ranking["collapse_rank_within_library_ratio"] = (
        ranking.groupby(
            ["library", "ratio"],
            observed=True,
        )["collapsed_percent"]
        .rank(method="min", ascending=False)
    )

    ranking.to_csv(
        outdir / "collapse_ranking.csv",
        index=False,
    )

    # --------------------------------------------------------
    # COMPACT CROSS-LIBRARY OVERVIEWS
    # --------------------------------------------------------

    progress("Creating cross-library overview plots")

    plot_cross_library_heatmap(
        combined_summary,
        "mean_Rg",
        plot_dir / "06_all_libraries_mean_rg_heatmap.png",
        "Mean Rg (nm)",
        "Design Rule 4 — Mean Rg across all libraries",
        decimals=2,
    )

    plot_cross_library_heatmap(
        combined_summary,
        "collapsed_percent",
        plot_dir / "07_all_libraries_collapse_heatmap.png",
        "Collapsed population (%)",
        f"Design Rule 4 — Collapse (Rg < {args.threshold:.2f} nm)",
        decimals=1,
    )

    plot_cross_library_heatmap(
        combined_summary,
        "Rg_range90",
        plot_dir / "14_all_libraries_rg_range90_heatmap.png",
        "Rg range 90 (nm)",
        "Design Rule 4 — Conformational Rg range (Q95 - Q05)",
        decimals=2,
    )

    plot_cross_library_heatmap(
        combined_summary,
        "mean_compact_Rg",
        plot_dir / "15_all_libraries_mean_compact_rg_heatmap.png",
        "Mean compact Rg (nm)",
        f"Design Rule 4 — Compact-state size (Rg < {args.threshold:.2f} nm)",
        decimals=2,
    )

    # --------------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------------

    progress("Analysis complete")

    print()
    print("=" * 78)
    print("OVERALL RESULTS")
    print("=" * 78)

    display_cols = [
        "library",
        "n_frames",
        "n_designs",
        "sanity_passed",
        "overall_mean_Rg",
        "overall_median_Rg",
        "overall_collapsed_percent",
    ]

    print(
        sanity_df[display_cols]
        .round(
            {
                "overall_mean_Rg": 3,
                "overall_median_Rg": 3,
                "overall_collapsed_percent": 2,
            }
        )
        .to_string(index=False)
    )

    print()
    print("Most collapsed designs:")
    print(
        combined_summary.sort_values(
            "collapsed_percent",
            ascending=False,
        )[
            [
                "id",
                "library",
                "ratio",
                "architecture",
                "mean_Rg",
                "collapsed_percent",
            ]
        ]
        .head(15)
        .round(
            {
                "mean_Rg": 3,
                "collapsed_percent": 2,
            }
        )
        .to_string(index=False)
    )

    print()
    print(f"Results saved to:\n{outdir.resolve()}")
    print()
    print(
        "First plots to inspect:\n"
        "  01_mean_rg_heatmap_VGEG.png\n"
        "  01_mean_rg_heatmap_VGKG.png\n"
        "  01_mean_rg_heatmap_IGKG.png\n"
        "  02_collapse_heatmap_VGEG.png\n"
        "  02_collapse_heatmap_VGKG.png\n"
        "  02_collapse_heatmap_IGKG.png\n"
        "  06_all_libraries_mean_rg_heatmap.png\n"
        "  07_all_libraries_collapse_heatmap.png"
    )


if __name__ == "__main__":
    main()
