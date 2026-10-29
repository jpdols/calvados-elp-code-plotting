#!/usr/bin/env python
"""
plot_vgipkg_top6_370K_properties.py

Focused 370 K overview for the six strongest / most informative VGIPKG properties:

1. Ensemble breadth       -> Rg IQR
2. Collapse accessibility -> k_collapse
3. Switching activity     -> collapsed events / 100 ns
4. Threshold behaviour    -> fraction of frames in hysteresis threshold band
5. Mean compactness       -> mean Rg
6. Collapse stability     -> k_escape

Creates TWO overview plots with identical six panels:
    A) architectures sorted once by Ensemble breadth (Rg IQR)
    B) architectures sorted once by Switching activity (collapsed events / 100 ns)

Also creates a Pearson correlation heatmap for ONLY these six properties.

Input
-----
architecture_properties_summary.csv
(as produced by investigate_vgipkg_properties.py)

Outputs
-------
top6_370K_overview_sorted_by_ensemble_breadth.png
top6_370K_overview_sorted_by_switching_activity.png
top6_370K_property_correlation_heatmap.png
top6_370K_property_correlations.csv
top6_370K_property_values.csv
top6_370K_property_signal_scores.csv

Architecture signal:
    S = SD of architecture means / mean within-architecture replicate SD

S > 1 means between-architecture variation exceeds typical replicate variation.
This is NOT a formal statistical significance test.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Top six selected 370 K properties
# ---------------------------------------------------------------------

PROPERTIES = [
    {
        "category": "Ensemble breadth",
        "label": "Rg IQR",
        "mean_col": "rg_iqr_nm_370K_mean",
        "sd_col": "rg_iqr_nm_370K_sd",
        "ylabel": "Rg IQR (nm)",
    },
    {
        "category": "Collapse accessibility",
        "label": r"$k_{\mathrm{collapse}}$",
        "mean_col": "k_collapse_per_ns_370K_mean",
        "sd_col": "k_collapse_per_ns_370K_sd",
        "ylabel": r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
    },
    {
        "category": "Switching activity",
        "label": "Collapsed events / 100 ns",
        "mean_col": "collapsed_events_per_100ns_370K_mean",
        "sd_col": "collapsed_events_per_100ns_370K_sd",
        "ylabel": "Collapsed events / 100 ns",
    },
    {
        "category": "Threshold behaviour",
        "label": "Threshold-band occupancy",
        "mean_col": "near_threshold_fraction_370K_mean",
        "sd_col": "near_threshold_fraction_370K_sd",
        "ylabel": "Fraction in 4.25–4.35 nm band",
    },
    {
        "category": "Mean compactness",
        "label": "Mean Rg",
        "mean_col": "rg_mean_nm_370K_mean",
        "sd_col": "rg_mean_nm_370K_sd",
        "ylabel": "Mean Rg (nm)",
    },
    {
        "category": "Collapse stability",
        "label": r"$k_{\mathrm{escape}}$",
        "mean_col": "k_escape_per_ns_370K_mean",
        "sd_col": "k_escape_per_ns_370K_sd",
        "ylabel": r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
    },
]


SORTS = [
    {
        "name": "ensemble_breadth",
        "column": "rg_iqr_nm_370K_mean",
        "label": "Ensemble breadth (Rg IQR)",
    },
    {
        "name": "switching_activity",
        "column": "collapsed_events_per_100ns_370K_mean",
        "label": "Switching activity (collapsed events / 100 ns)",
    },
]


def architecture_signal(df, mean_col, sd_col):
    """
    S = between-architecture SD / mean within-architecture replicate SD
    """
    x = pd.to_numeric(df[mean_col], errors="coerce")
    within = pd.to_numeric(df[sd_col], errors="coerce")

    between_sd = x.std(ddof=1)
    mean_within_sd = within.mean()

    if not np.isfinite(mean_within_sd) or mean_within_sd == 0:
        return np.nan, between_sd, mean_within_sd

    return between_sd / mean_within_sd, between_sd, mean_within_sd


def validate_columns(df):
    required = ["architecture_id"]

    for p in PROPERTIES:
        required.extend([p["mean_col"], p["sd_col"]])

    for s in SORTS:
        required.append(s["column"])

    missing = [c for c in sorted(set(required)) if c not in df.columns]

    if missing:
        raise KeyError(
            "Missing required columns:\n  - "
            + "\n  - ".join(missing)
            + "\n\nCheck that this is architecture_properties_summary.csv "
              "from investigate_vgipkg_properties.py."
        )


def make_signal_table(df):
    rows = []

    for p in PROPERTIES:
        S, between_sd, within_sd = architecture_signal(
            df, p["mean_col"], p["sd_col"]
        )

        rows.append(
            {
                "category": p["category"],
                "property": p["label"].replace("$", ""),
                "architecture_signal_S": S,
                "between_architecture_sd": between_sd,
                "mean_within_architecture_replicate_sd": within_sd,
            }
        )

    return pd.DataFrame(rows)


def make_overview(df, outdir, sort_spec, dpi=300):
    """
    Plot all six properties using ONE common architecture order.
    """
    d = (
        df.sort_values(sort_spec["column"], ascending=True)
          .reset_index(drop=True)
    )

    x = np.arange(len(d))
    labels = d["architecture_id"].astype(int).astype(str)

    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 12),
        constrained_layout=True
    )
    axes = axes.flatten()

    panel_letters = list("ABCDEF")

    for i, p in enumerate(PROPERTIES):
        ax = axes[i]

        y = pd.to_numeric(d[p["mean_col"]], errors="coerce").to_numpy()
        yerr = pd.to_numeric(d[p["sd_col"]], errors="coerce").to_numpy()

        S, _, _ = architecture_signal(
            df,
            p["mean_col"],
            p["sd_col"]
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            linewidth=1.0,
            elinewidth=0.9,
        )

        ax.plot(
            x,
            y,
            linewidth=0.8,
            alpha=0.65,
        )

        ax.set_title(
            f"{panel_letters[i]}. {p['category']} — {p['label']}\n"
            f"Architecture signal S = {S:.2f}",
            fontsize=11.5,
            fontweight="bold",
        )

        ax.set_ylabel(p["ylabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=90,
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.25)

        if i >= 4:
            ax.set_xlabel(
                f"Architecture ID "
                f"(ordered by {sort_spec['label']})"
            )

    fig.suptitle(
        "VGIPKG top six architecture-dependent properties at 370 K\n"
        f"All panels ordered by {sort_spec['label']} | "
        "Triplicate mean ± replicate SD",
        fontsize=15,
        fontweight="bold",
    )

    outfile = (
        outdir
        / f"top6_370K_overview_sorted_by_{sort_spec['name']}.png"
    )

    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return outfile


def make_heatmap(df, outdir, dpi=300):
    """
    Pearson correlation across architecture means for ONLY the top six properties.
    """
    nice_names = [p["category"] for p in PROPERTIES]
    mean_cols = [p["mean_col"] for p in PROPERTIES]

    corr_data = (
        df[mean_cols]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )
    corr_data.columns = nice_names

    corr = corr_data.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    im = ax.imshow(
        corr.values,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    ax.set_xticks(np.arange(len(nice_names)))
    ax.set_yticks(np.arange(len(nice_names)))

    ax.set_xticklabels(
        nice_names,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(nice_names)

    # Correlation values inside cells
    for i in range(len(nice_names)):
        for j in range(len(nice_names)):
            value = corr.iloc[i, j]

            if np.isfinite(value):
                text_color = (
                    "white"
                    if abs(value) >= 0.55
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Pearson r")

    ax.set_title(
        "VGIPKG top six 370 K property correlations\n"
        "Pearson r across 30 architecture means",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    outfile = (
        outdir
        / "top6_370K_property_correlation_heatmap.png"
    )

    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    corr.to_csv(
        outdir
        / "top6_370K_property_correlations.csv"
    )

    return outfile


def save_selected_values(df, outdir):
    keep = ["architecture_id"]
    rename = {}

    for p in PROPERTIES:
        keep.extend(
            [p["mean_col"], p["sd_col"]]
        )

        rename[p["mean_col"]] = (
            f"{p['category']}__mean"
        )
        rename[p["sd_col"]] = (
            f"{p['category']}__replicate_sd"
        )

    selected = (
        df[keep]
        .copy()
        .rename(columns=rename)
    )

    selected.to_csv(
        outdir
        / "top6_370K_property_values.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot top six VGIPKG architecture properties "
            "at 370 K in two architecture orderings."
        )
    )

    parser.add_argument(
        "--summary",
        required=True,
        help="Path to architecture_properties_summary.csv",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300)",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(summary_path)

    validate_columns(df)

    df["architecture_id"] = pd.to_numeric(
        df["architecture_id"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["architecture_id"])
        .copy()
    )

    # Signal scores
    signal = make_signal_table(df)

    signal.to_csv(
        outdir
        / "top6_370K_property_signal_scores.csv",
        index=False,
    )

    # Selected data
    save_selected_values(
        df,
        outdir,
    )

    # Two overview plots using two different common orderings
    overview_files = []

    for sort_spec in SORTS:
        overview_files.append(
            make_overview(
                df,
                outdir,
                sort_spec,
                dpi=args.dpi,
            )
        )

    # Six-property heatmap
    heatmap = make_heatmap(
        df,
        outdir,
        dpi=args.dpi,
    )

    print("\nTop six 370 K architecture signal scores")
    print(
        signal[
            [
                "category",
                "property",
                "architecture_signal_S",
            ]
        ].to_string(index=False)
    )

    print("\nWritten:")

    for outfile in overview_files:
        print(f"  {outfile}")

    print(f"  {heatmap}")
    print(
        f"  {outdir / 'top6_370K_property_correlations.csv'}"
    )
    print(
        f"  {outdir / 'top6_370K_property_values.csv'}"
    )
    print(
        f"  {outdir / 'top6_370K_property_signal_scores.csv'}"
    )


if __name__ == "__main__":
    main()
