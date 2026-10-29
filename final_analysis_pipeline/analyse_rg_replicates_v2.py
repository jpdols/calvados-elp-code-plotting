#!/usr/bin/env python
"""
analyse_rg_replicates_v2.py

Analyse THREE replicate libraries where each replicate folder contains:
    - all_rg_values.csv
    - per_run_summary.csv

For every 370 K architecture/run:
1. Load frame-by-frame Rg values from all three replicates.
2. Identify the architecture/run ID automatically.
3. Split frames into:
       collapsed : Rg < threshold
       expanded  : Rg >= threshold
4. Calculate per-replicate state-specific Rg statistics.
5. Merge these extra statistics into each replicate's per_run_summary.csv.
6. Create a combined 3-replicate summary containing:
       mean across replicates
       sample SD across replicates (ddof=1)
   for all numeric metrics, including the original summary metrics and the
   newly calculated collapsed/expanded Rg metrics.
7. Create collapsed vs expanded Rg distribution plots for every architecture.

Designed for e.g.:
    VGIPKG
    VGIPKG_rep02
    VGIPKG_rep03

Example:
python analyse_rg_replicates_v2.py ^
  --rg-file "E:\...\VGIPKG\all_rg_values.csv" ^
  --rg-file "E:\...\VGIPKG_rep02\all_rg_values.csv" ^
  --rg-file "E:\...\VGIPKG_rep03\all_rg_values.csv" ^
  --summary-file "E:\...\VGIPKG\per_run_summary.csv" ^
  --summary-file "E:\...\VGIPKG_rep02\per_run_summary.csv" ^
  --summary-file "E:\...\VGIPKG_rep03\per_run_summary.csv" ^
  --temperature 370K ^
  --threshold 4.30 ^
  --outdir "E:\...\VGIPKG_3rep_analysis"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RG_CANDIDATES = [
    "rg_nm", "Rg_nm", "rg", "Rg",
    "radius_of_gyration", "radius_of_gyration_nm",
    "radius_gyration", "radius_gyration_nm"
]

TEMP_CANDIDATES = [
    "temp", "temperature", "temperature_K", "Temp", "T"
]

ID_CANDIDATES = [
    "id", "run_id", "architecture_id", "architecture",
    "name", "label"
]

ARCH_CANDIDATES = [
    "architecture_type", "architecture", "arch_type", "design"
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse 3 replicate all_rg_values.csv + per_run_summary.csv libraries."
    )

    p.add_argument(
        "--rg-file",
        action="append",
        required=True,
        help="Path to all_rg_values.csv. Supply exactly 3 times."
    )

    p.add_argument(
        "--summary-file",
        action="append",
        required=True,
        help="Path to per_run_summary.csv. Supply exactly 3 times."
    )

    p.add_argument(
        "--temperature",
        default="370K",
        help="Temperature to analyse. Default: 370K"
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=4.30,
        help="Collapse threshold in nm. Default: 4.30"
    )

    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory."
    )

    p.add_argument(
        "--bins",
        type=int,
        default=70,
        help="Histogram bins. Default: 70"
    )

    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip per-architecture Rg distribution plots."
    )

    return p.parse_args()


def normalise_temp(x):
    s = str(x).strip().upper().replace(" ", "")
    if s.endswith(".0K"):
        s = s.replace(".0K", "K")
    elif re.fullmatch(r"\d+(\.\d+)?", s):
        s = f"{float(s):g}K"
    return s


def detect_column(df, candidates, required=True):
    # exact
    for c in candidates:
        if c in df.columns:
            return c

    # case-insensitive
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise ValueError(
            f"Could not identify required column from {candidates}.\n"
            f"Columns present: {list(df.columns)}"
        )

    return None


def detect_rg_column(df):
    try:
        return detect_column(df, RG_CANDIDATES, required=True)
    except ValueError:
        pass

    rg_like = [
        c for c in df.columns
        if "rg" in str(c).lower()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(rg_like) == 1:
        return rg_like[0]

    raise ValueError(
        f"Could not identify Rg column. Columns: {list(df.columns)}"
    )


def detect_id_column(df):
    # Prefer exact run ID
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c

    # case-insensitive fallback
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in ID_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    raise ValueError(
        "Could not identify architecture/run ID column.\n"
        f"Columns: {list(df.columns)}"
    )


def clean_id_series(s):
    return s.astype(str).str.strip()


def filter_temperature(df, target_temp):
    temp_col = detect_column(df, TEMP_CANDIDATES, required=False)

    if temp_col is None:
        return df.copy()

    target = normalise_temp(target_temp)
    mask = df[temp_col].map(normalise_temp) == target

    if not mask.any():
        raise ValueError(
            f"No rows found for temperature {target_temp}. "
            f"Values in {temp_col}: {df[temp_col].dropna().unique()[:20]}"
        )

    return df.loc[mask].copy()


def state_stats(rg_values, threshold):
    rg = pd.to_numeric(rg_values, errors="coerce").dropna()

    collapsed = rg[rg < threshold]
    expanded = rg[rg >= threshold]

    def desc(arr, prefix):
        arr = pd.to_numeric(arr, errors="coerce").dropna()

        if len(arr) == 0:
            return {
                f"{prefix}_n_frames": 0,
                f"{prefix}_mean_rg_nm": np.nan,
                f"{prefix}_sd_rg_nm": np.nan,
                f"{prefix}_median_rg_nm": np.nan,
                f"{prefix}_p25_rg_nm": np.nan,
                f"{prefix}_p75_rg_nm": np.nan,
                f"{prefix}_iqr_rg_nm": np.nan,
            }

        q25, q75 = np.percentile(arr, [25, 75])

        return {
            f"{prefix}_n_frames": int(len(arr)),
            f"{prefix}_mean_rg_nm": float(arr.mean()),
            f"{prefix}_sd_rg_nm": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
            f"{prefix}_median_rg_nm": float(arr.median()),
            f"{prefix}_p25_rg_nm": float(q25),
            f"{prefix}_p75_rg_nm": float(q75),
            f"{prefix}_iqr_rg_nm": float(q75 - q25),
        }

    q25_all, q75_all = np.percentile(rg, [25, 75])

    result = {
        "n_frames_recalc": int(len(rg)),
        "mean_rg_nm_recalc": float(rg.mean()),
        "sd_rg_nm_recalc": float(rg.std(ddof=1)) if len(rg) > 1 else np.nan,
        "median_rg_nm_recalc": float(rg.median()),
        "iqr_rg_nm_recalc": float(q75_all - q25_all),
        "fraction_collapsed_recalc": float((rg < threshold).mean()),
        "fraction_expanded_recalc": float((rg >= threshold).mean()),
        "threshold_nm_recalc": float(threshold),
    }

    result.update(desc(collapsed, "collapsed"))
    result.update(desc(expanded, "expanded"))

    return result


def analyse_rg_file(path, replicate, temperature, threshold):
    df = pd.read_csv(path)

    df = filter_temperature(df, temperature)

    rg_col = detect_rg_column(df)
    id_col = detect_id_column(df)

    df[id_col] = clean_id_series(df[id_col])
    df[rg_col] = pd.to_numeric(df[rg_col], errors="coerce")
    df = df.dropna(subset=[rg_col]).copy()

    rows = []

    for run_id, sub in df.groupby(id_col, sort=False):
        stats = state_stats(sub[rg_col], threshold)
        stats.update({
            "id": str(run_id),
            "replicate": replicate,
            "rg_source_file": str(path),
        })
        rows.append(stats)

    result = pd.DataFrame(rows)

    print(
        f"Replicate {replicate}: {Path(path).name} -> "
        f"{len(df):,} frames across {len(result)} runs at {temperature}"
    )

    return result, df, id_col, rg_col


def load_summary(path, replicate, temperature):
    df = pd.read_csv(path)
    df = filter_temperature(df, temperature)

    id_col = detect_id_column(df)
    df[id_col] = clean_id_series(df[id_col])

    # Standardise ID name internally
    if id_col != "id":
        if "id" in df.columns:
            raise ValueError(
                f"Summary has both '{id_col}' and 'id'; cannot safely standardise."
            )
        df = df.rename(columns={id_col: "id"})

    df["replicate"] = replicate
    df["summary_source_file"] = str(path)

    print(
        f"Replicate {replicate}: {Path(path).name} -> "
        f"{len(df)} summary rows at {temperature}"
    )

    return df


def merge_replicate_summary(summary_df, rg_stats_df):
    before = len(summary_df)

    merged = summary_df.merge(
        rg_stats_df.drop(columns=["rg_source_file"], errors="ignore"),
        on=["id", "replicate"],
        how="left",
        validate="one_to_one"
    )

    missing = merged["mean_rg_nm_recalc"].isna().sum()

    if missing:
        bad = merged.loc[
            merged["mean_rg_nm_recalc"].isna(),
            "id"
        ].astype(str).tolist()

        print(
            f"WARNING: {missing}/{before} summary rows had no matching "
            f"frame-by-frame Rg data.\n"
            f"Examples: {bad[:10]}"
        )

    return merged


def make_three_rep_summary(combined):
    """
    Aggregate matching run IDs across the three replicates.

    Output contains:
      - shared metadata
      - n_replicates
      - metric_mean_3rep
      - metric_sd_3rep
    """
    rows = []

    excluded_numeric = {
        "replicate"
    }

    for run_id, sub in combined.groupby("id", sort=False):
        row = {"id": run_id}

        row["n_replicates"] = int(sub["replicate"].nunique())

        # Preserve shared non-numeric metadata where possible.
        for c in sub.columns:
            if c in {
                "id", "replicate",
                "summary_source_file",
                "rg_source_file"
            }:
                continue

            if not pd.api.types.is_numeric_dtype(sub[c]):
                vals = sub[c].dropna().astype(str).unique()
                if len(vals) == 1:
                    row[c] = vals[0]

        # Every numeric metric gets mean + SD across replicate-level values.
        numeric_cols = sub.select_dtypes(include=[np.number]).columns

        for c in numeric_cols:
            if c in excluded_numeric:
                continue

            vals = pd.to_numeric(sub[c], errors="coerce").dropna()

            if len(vals) == 0:
                continue

            row[f"{c}_mean_3rep"] = float(vals.mean())

            row[f"{c}_sd_3rep"] = (
                float(vals.std(ddof=1))
                if len(vals) > 1
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def safe_filename(text):
    s = re.sub(r'[<>:"/\\|?*]+', "_", str(text))
    s = re.sub(r"\s+", "_", s)
    return s[:180]


def plot_one_architecture(
    run_id,
    replicate_frames,
    id_columns,
    rg_columns,
    threshold,
    bins,
    outdir
):
    all_rg = []

    for rep_i, df in enumerate(replicate_frames, start=1):
        id_col = id_columns[rep_i - 1]
        rg_col = rg_columns[rep_i - 1]

        sub = df[clean_id_series(df[id_col]) == str(run_id)]

        if len(sub):
            temp = pd.DataFrame({
                "replicate": rep_i,
                "rg_nm": pd.to_numeric(sub[rg_col], errors="coerce")
            }).dropna()
            all_rg.append(temp)

    if not all_rg:
        return

    x = pd.concat(all_rg, ignore_index=True)

    collapsed = x.loc[x["rg_nm"] < threshold, "rg_nm"]
    expanded = x.loc[x["rg_nm"] >= threshold, "rg_nm"]

    if collapsed.empty and expanded.empty:
        return

    rg_min = x["rg_nm"].min()
    rg_max = x["rg_nm"].max()

    if rg_min == rg_max:
        rg_min -= 0.1
        rg_max += 0.1

    edges = np.linspace(rg_min, rg_max, bins + 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    if len(collapsed):
        ax.hist(
            collapsed,
            bins=edges,
            density=True,
            alpha=0.55,
            label=f"Collapsed (Rg < {threshold:.2f} nm)"
        )

    if len(expanded):
        ax.hist(
            expanded,
            bins=edges,
            density=True,
            alpha=0.55,
            label=f"Uncollapsed (Rg ≥ {threshold:.2f} nm)"
        )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.4,
        label=f"Threshold = {threshold:.2f} nm"
    )

    ax.set_xlabel(r"$R_g$ (nm)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{run_id} — 370 K, 3 replicates")
    ax.legend(frameon=False)

    fig.tight_layout()

    filename = safe_filename(run_id) + "_rg_collapsed_uncollapsed.png"
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    if len(args.rg_file) != 3:
        raise SystemExit(
            f"ERROR: supply exactly 3 --rg-file arguments; got {len(args.rg_file)}"
        )

    if len(args.summary_file) != 3:
        raise SystemExit(
            f"ERROR: supply exactly 3 --summary-file arguments; got {len(args.summary_file)}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plots_dir = outdir / "rg_distributions_370K"
    if not args.skip_plots:
        plots_dir.mkdir(exist_ok=True)

    all_rg_stats = []
    all_enriched = []

    replicate_frames = []
    id_columns = []
    rg_columns = []

    # ------------------------------------------------------------
    # Analyse each replicate separately
    # ------------------------------------------------------------
    for rep in (1, 2, 3):
        rg_path = Path(args.rg_file[rep - 1])
        summary_path = Path(args.summary_file[rep - 1])

        if not rg_path.exists():
            raise FileNotFoundError(rg_path)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        rg_stats, rg_frames, id_col, rg_col = analyse_rg_file(
            rg_path,
            replicate=rep,
            temperature=args.temperature,
            threshold=args.threshold
        )

        summary = load_summary(
            summary_path,
            replicate=rep,
            temperature=args.temperature
        )

        enriched = merge_replicate_summary(summary, rg_stats)

        all_rg_stats.append(rg_stats)
        all_enriched.append(enriched)

        replicate_frames.append(rg_frames)
        id_columns.append(id_col)
        rg_columns.append(rg_col)

        rg_stats.to_csv(
            outdir / f"rep{rep:02d}_rg_state_statistics.csv",
            index=False
        )

        enriched.to_csv(
            outdir / f"rep{rep:02d}_per_run_summary_enriched.csv",
            index=False
        )

    # ------------------------------------------------------------
    # Combined replicate-level table
    # ------------------------------------------------------------
    combined = pd.concat(
        all_enriched,
        ignore_index=True,
        sort=False
    )

    combined.to_csv(
        outdir / "all_3rep_per_run_summary_enriched.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Mean + SD across THREE replicates
    # ------------------------------------------------------------
    three_rep = make_three_rep_summary(combined)

    three_rep.to_csv(
        outdir / "per_run_summary_3rep_mean_sd.csv",
        index=False
    )

    # Also make a compact table focusing on new Rg-state metrics
    wanted_prefixes = (
        "mean_rg_nm_recalc_",
        "sd_rg_nm_recalc_",
        "iqr_rg_nm_recalc_",
        "fraction_collapsed_recalc_",
        "collapsed_",
        "expanded_",
        "n_frames_recalc_"
    )

    compact_cols = ["id", "n_replicates"]

    for c in three_rep.columns:
        if c.startswith(wanted_prefixes):
            compact_cols.append(c)

    # Include useful metadata if present
    metadata_candidates = [
        "group",
        "architecture_type",
        "temp",
        "temperature"
    ]

    for c in reversed(metadata_candidates):
        if c in three_rep.columns and c not in compact_cols:
            compact_cols.insert(1, c)

    three_rep[
        [c for c in compact_cols if c in three_rep.columns]
    ].to_csv(
        outdir / "rg_state_summary_3rep_mean_sd_compact.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Distribution plots for each run
    # ------------------------------------------------------------
    if not args.skip_plots:
        ids = combined["id"].dropna().astype(str).unique()

        print(f"\nCreating Rg distribution plots for {len(ids)} runs...")

        for i, run_id in enumerate(ids, start=1):
            plot_one_architecture(
                run_id=run_id,
                replicate_frames=replicate_frames,
                id_columns=id_columns,
                rg_columns=rg_columns,
                threshold=args.threshold,
                bins=args.bins,
                outdir=plots_dir
            )

            if i % 25 == 0 or i == len(ids):
                print(f"  plotted {i}/{len(ids)}")

    print("\nDone.")
    print("\nMain outputs:")
    print(" ", outdir / "all_3rep_per_run_summary_enriched.csv")
    print(" ", outdir / "per_run_summary_3rep_mean_sd.csv")
    print(" ", outdir / "rg_state_summary_3rep_mean_sd_compact.csv")

    if not args.skip_plots:
        print(" ", plots_dir)


if __name__ == "__main__":
    main()
