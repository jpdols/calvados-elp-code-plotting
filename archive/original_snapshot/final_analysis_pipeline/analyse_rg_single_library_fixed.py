#!/usr/bin/env python
"""
analyse_rg_single_library_fixed.py

Single-replicate frame-by-frame Rg analysis for one CALVADOS library.

Key fix:
- In your files, `group` is the library name (e.g. IGKGEG), NOT the ratio.
- R1 / R2 are extracted from the architecture/run ID if no dedicated ratio column exists.

Collapsed:   Rg < threshold
Uncollapsed: Rg >= threshold
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

# Important: `group` is intentionally NOT included here, because in your
# current files it contains the library name (e.g. IGKGEG), not R1/R2.
RATIO_CANDIDATES = [
    "ratio", "composition_ratio", "library_ratio"
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rg-file", required=True)
    p.add_argument("--summary-file", required=True)
    p.add_argument("--temperature", default="370K")
    p.add_argument("--ratio", action="append", default=[])
    p.add_argument("--threshold", type=float, default=4.30)
    p.add_argument("--outdir", required=True)
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def normalise_temp(x):
    s = str(x).strip().upper().replace(" ", "")
    if s.endswith(".0K"):
        s = s.replace(".0K", "K")
    elif re.fullmatch(r"\d+(\.\d+)?", s):
        s = f"{float(s):g}K"
    return s


def detect_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise ValueError(
            f"Could not identify required column from {candidates}\n"
            f"Columns present: {list(df.columns)}"
        )
    return None


def detect_id_column(df):
    return detect_column(df, ID_CANDIDATES, required=True)


def detect_rg_column(df):
    for c in RG_CANDIDATES:
        if c in df.columns:
            return c

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


def clean_id_series(s):
    return s.astype(str).str.strip()


def filter_temperature(df, target_temp):
    temp_col = detect_column(df, TEMP_CANDIDATES, required=False)

    if temp_col is None:
        print("WARNING: no temperature column found; temperature filter skipped.", flush=True)
        return df.copy()

    target = normalise_temp(target_temp)
    mask = df[temp_col].map(normalise_temp) == target

    if not mask.any():
        raise ValueError(
            f"No rows found for temperature {target_temp}. "
            f"Values in {temp_col}: {df[temp_col].dropna().unique()[:20]}"
        )

    return df.loc[mask].copy()


def extract_ratio_from_id(series):
    """
    Extract R1/R2/... from IDs such as:
        001_IGKGEG_R1_even
        IGKGEG_R2_largeblocks
    """
    return (
        series.astype(str)
        .str.extract(r"(?:^|_)(R\d+)(?:_|$)", expand=False)
        .str.upper()
    )


def filter_ratios(df, ratios):
    if not ratios:
        return df.copy()

    wanted = {str(x).strip().upper() for x in ratios}

    # First use an actual ratio column if present.
    ratio_col = detect_column(df, RATIO_CANDIDATES, required=False)

    if ratio_col is not None:
        ratio_values = df[ratio_col].astype(str).str.strip().str.upper()
        mask = ratio_values.isin(wanted)

        if mask.any():
            print(
                f"Filtering ratios using column '{ratio_col}': {sorted(wanted)}",
                flush=True
            )
            return df.loc[mask].copy()

    # Otherwise extract R1/R2/etc from the architecture ID.
    id_col = detect_id_column(df)
    extracted = extract_ratio_from_id(df[id_col])

    available = sorted(extracted.dropna().unique().tolist())

    mask = extracted.isin(wanted)

    if not mask.any():
        raise ValueError(
            f"Could not find requested ratios {sorted(wanted)} in either a dedicated "
            f"ratio column or the ID column '{id_col}'. "
            f"Ratios extracted from IDs: {available[:30]}"
        )

    print(
        f"Filtering ratios from ID column '{id_col}': {sorted(wanted)} "
        f"(available: {available})",
        flush=True
    )

    out = df.loc[mask].copy()
    out["_ratio_extracted"] = extracted.loc[mask].values
    return out


def descriptive_stats(arr, prefix):
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


def state_stats(rg_values, threshold):
    rg = pd.to_numeric(rg_values, errors="coerce").dropna()

    collapsed = rg[rg < threshold]
    uncollapsed = rg[rg >= threshold]

    q25, q75 = np.percentile(rg, [25, 75])

    result = {
        "n_frames_recalc": int(len(rg)),
        "mean_rg_nm_recalc": float(rg.mean()),
        "sd_rg_nm_recalc": float(rg.std(ddof=1)) if len(rg) > 1 else np.nan,
        "median_rg_nm_recalc": float(rg.median()),
        "iqr_rg_nm_recalc": float(q75 - q25),
        "fraction_collapsed_recalc": float((rg < threshold).mean()),
        "fraction_uncollapsed_recalc": float((rg >= threshold).mean()),
        "threshold_nm_recalc": float(threshold),
    }

    result.update(descriptive_stats(collapsed, "collapsed"))
    result.update(descriptive_stats(uncollapsed, "uncollapsed"))

    return result


def prepare_summary(summary_df):
    summary_df = summary_df.copy()
    id_col = detect_id_column(summary_df)
    summary_df[id_col] = clean_id_series(summary_df[id_col])

    if id_col != "id":
        if "id" in summary_df.columns:
            raise ValueError(
                f"Summary contains both '{id_col}' and 'id'; cannot safely rename."
            )
        summary_df = summary_df.rename(columns={id_col: "id"})

    return summary_df


def safe_filename(text):
    s = re.sub(r'[<>:"/\\|?*]+', "_", str(text))
    s = re.sub(r"\s+", "_", s)
    return s[:180]


def make_distribution_plot(run_id, sub, rg_col, threshold, bins, outpath, title_extra=""):
    rg = pd.to_numeric(sub[rg_col], errors="coerce").dropna()
    if len(rg) == 0:
        return

    collapsed = rg[rg < threshold]
    uncollapsed = rg[rg >= threshold]

    rg_min, rg_max = rg.min(), rg.max()
    if rg_min == rg_max:
        rg_min -= 0.1
        rg_max += 0.1

    edges = np.linspace(rg_min, rg_max, bins + 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    if len(collapsed):
        ax.hist(collapsed, bins=edges, density=True, alpha=0.55,
                label=f"Collapsed (Rg < {threshold:.2f} nm)")

    if len(uncollapsed):
        ax.hist(uncollapsed, bins=edges, density=True, alpha=0.55,
                label=f"Uncollapsed (Rg ≥ {threshold:.2f} nm)")

    ax.axvline(threshold, linestyle="--", linewidth=1.4,
               label=f"Threshold = {threshold:.2f} nm")

    ax.set_xlabel(r"$R_g$ (nm)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{run_id}{title_extra}")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    rg_path = Path(args.rg_file)
    summary_path = Path(args.summary_file)
    outdir = Path(args.outdir)
    plots_dir = outdir / "rg_distributions"

    outdir.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        plots_dir.mkdir(exist_ok=True)

    print(f"Loading frame-wise Rg data:\n  {rg_path}", flush=True)
    rg_df = pd.read_csv(rg_path)
    print(f"Loaded {len(rg_df):,} frame rows.", flush=True)

    print(f"Filtering to {args.temperature}...", flush=True)
    rg_df = filter_temperature(rg_df, args.temperature)
    print(f"After temperature filter: {len(rg_df):,} rows.", flush=True)

    if args.ratio:
        print(f"Filtering to ratios: {args.ratio}", flush=True)
        rg_df = filter_ratios(rg_df, args.ratio)
        print(f"After ratio filter: {len(rg_df):,} rows.", flush=True)

    rg_col = detect_rg_column(rg_df)
    rg_id_col = detect_id_column(rg_df)

    rg_df[rg_id_col] = clean_id_series(rg_df[rg_id_col])
    rg_df[rg_col] = pd.to_numeric(rg_df[rg_col], errors="coerce")
    rg_df = rg_df.dropna(subset=[rg_col]).copy()

    print("Calculating collapsed/uncollapsed Rg statistics...", flush=True)

    rows = []
    for run_id, sub in rg_df.groupby(rg_id_col, sort=False):
        row = {"id": str(run_id)}
        row.update(state_stats(sub[rg_col], args.threshold))
        rows.append(row)

    state_df = pd.DataFrame(rows)

    print(
        f"Calculated state statistics for {len(state_df)} architectures/runs.",
        flush=True
    )

    print(f"Loading summary:\n  {summary_path}", flush=True)
    summary_df = pd.read_csv(summary_path)
    print(f"Loaded {len(summary_df)} summary rows.", flush=True)

    summary_df = filter_temperature(summary_df, args.temperature)
    if args.ratio:
        summary_df = filter_ratios(summary_df, args.ratio)
    summary_df = prepare_summary(summary_df)

    print(f"Summary rows after filtering: {len(summary_df)}", flush=True)

    print("Merging state statistics into per_run_summary...", flush=True)
    merged = summary_df.merge(
        state_df,
        on="id",
        how="left",
        validate="one_to_one"
    )

    merged.to_csv(
        outdir / "per_run_summary_enriched_rg_states.csv",
        index=False
    )

    state_cols = [
        "id", "group", "ratio", "_ratio_extracted",
        "architecture_type", "architecture",
        "temp", "temperature",
        "mean_rg_nm", "iqr_rg_nm", "fraction_collapsed",
        "k_collapse_per_ns", "k_escape_per_ns",
        "mean_rg_nm_recalc", "iqr_rg_nm_recalc",
        "fraction_collapsed_recalc",
        "collapsed_n_frames", "collapsed_mean_rg_nm",
        "collapsed_sd_rg_nm", "collapsed_median_rg_nm",
        "collapsed_iqr_rg_nm",
        "uncollapsed_n_frames", "uncollapsed_mean_rg_nm",
        "uncollapsed_sd_rg_nm", "uncollapsed_median_rg_nm",
        "uncollapsed_iqr_rg_nm",
        "threshold_nm_recalc"
    ]

    compact_cols = [c for c in state_cols if c in merged.columns]

    merged[compact_cols].to_csv(
        outdir / "rg_state_summary_compact.csv",
        index=False
    )

    if not args.skip_plots:
        ids = merged["id"].dropna().astype(str).tolist()
        print(f"Creating {len(ids)} distribution plots...", flush=True)

        for i, run_id in enumerate(ids, start=1):
            sub = rg_df[rg_df[rg_id_col].astype(str) == str(run_id)]
            if len(sub) == 0:
                continue

            make_distribution_plot(
                run_id,
                sub,
                rg_col,
                args.threshold,
                args.bins,
                plots_dir / (safe_filename(run_id) + "_rg_states.png"),
                title_extra=f" — {args.temperature}"
            )

            print(f"  plotted {i}/{len(ids)}: {run_id}", flush=True)

    print("\nDone.", flush=True)
    print("Main outputs:", flush=True)
    print(" ", outdir / "per_run_summary_enriched_rg_states.csv", flush=True)
    print(" ", outdir / "rg_state_summary_compact.csv", flush=True)
    if not args.skip_plots:
        print(" ", plots_dir, flush=True)


if __name__ == "__main__":
    main()
