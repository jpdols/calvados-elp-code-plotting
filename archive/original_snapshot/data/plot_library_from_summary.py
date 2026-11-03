#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


THETA_420 = 4.30
EXPANDED_420 = 7.87


def clean_label(s):
    return s.replace("_", " ")


def add_refs(ax):
    ax.axhline(THETA_420, linestyle=":", linewidth=2, alpha=0.45)
    ax.text(0.99, THETA_420 + 0.05, "Theta state: 4.30 nm",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)

    ax.axhline(EXPANDED_420, linestyle=":", linewidth=2, alpha=0.35)
    ax.text(0.99, EXPANDED_420 + 0.05, "Expanded coil: 7.87 nm",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9)


def plot_rg300_vs_rg370(df, outdir):
    fig, ax = plt.subplots(figsize=(7, 7))

    groups = sorted(df["group"].unique())
    for g in groups:
        sub = df[df["group"] == g]
        ax.scatter(sub["mean_rg_nm_300K"], sub["mean_rg_nm_370K"], s=35, alpha=0.75, label=g)

    lim_min = min(df["mean_rg_nm_300K"].min(), df["mean_rg_nm_370K"].min()) - 0.3
    lim_max = max(df["mean_rg_nm_300K"].max(), df["mean_rg_nm_370K"].max()) + 0.3
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle=":", color="black", alpha=0.6)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Mean Rg at 300 K (nm)")
    ax.set_ylabel("Mean Rg at 370 K (nm)")
    ax.set_title("Temperature response landscape: Rg300 vs Rg370")
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outdir / "scatter_rg300_vs_rg370.png", dpi=300)
    plt.close()


def plot_delta_vs_rg370(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for g in sorted(df["group"].unique()):
        sub = df[df["group"] == g]
        ax.scatter(
            sub["mean_rg_nm_370K"],
            sub["delta_mean_rg_370_minus_300_nm"],
            s=35,
            alpha=0.75,
            label=g,
        )

    ax.axhline(0, linestyle=":", color="black", alpha=0.6)
    ax.axvline(THETA_420, linestyle=":", color="black", alpha=0.35)
    ax.set_xlabel("Mean Rg at 370 K (nm)")
    ax.set_ylabel("ΔRg = Rg370 − Rg300 (nm)")
    ax.set_title("Design landscape: final compactness versus temperature response")
    ax.grid(linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(outdir / "scatter_delta_rg_vs_rg370.png", dpi=300)
    plt.close()


def plot_delta_hist(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["delta_mean_rg_370_minus_300_nm"], bins=35, alpha=0.85)
    ax.axvline(0, linestyle=":", color="black")
    ax.set_xlabel("ΔRg = Rg370 − Rg300 (nm)")
    ax.set_ylabel("Number of architectures")
    ax.set_title("Distribution of temperature response across 366 architectures")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outdir / "hist_delta_rg.png", dpi=300)
    plt.close()


def plot_family_boxplot(df, outdir):
    groups = sorted(df["group"].unique())
    data = [df[df["group"] == g]["delta_mean_rg_370_minus_300_nm"].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 0.7), 6))
    ax.boxplot(data, labels=groups, showfliers=True)
    ax.axhline(0, linestyle=":", color="black", alpha=0.6)
    ax.set_ylabel("ΔRg = Rg370 − Rg300 (nm)")
    ax.set_title("Temperature response per motif family")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(outdir / "boxplot_delta_rg_by_family.png", dpi=300)
    plt.close()


def plot_architecture_type_boxplot(df, outdir):
    order = sorted(df["architecture_type"].unique())
    data = [df[df["architecture_type"] == a]["delta_mean_rg_370_minus_300_nm"].dropna().values for a in order]

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.7), 6))
    ax.boxplot(data, labels=[clean_label(x) for x in order], showfliers=True)
    ax.axhline(0, linestyle=":", color="black", alpha=0.6)
    ax.set_ylabel("ΔRg = Rg370 − Rg300 (nm)")
    ax.set_title("Temperature response per architecture type")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(outdir / "boxplot_delta_rg_by_architecture_type.png", dpi=300)
    plt.close()


def plot_paired_family(df, outdir):
    famdir = outdir / "paired_family_plots"
    famdir.mkdir(exist_ok=True)

    for group, sub in df.groupby("group"):
        sub = sub.sort_values("id").reset_index(drop=True)
        x = np.arange(len(sub))

        fig, ax = plt.subplots(figsize=(max(9, len(sub) * 0.35), 6))

        for i, row in sub.iterrows():
            ax.plot(
                [i, i],
                [row["mean_rg_nm_300K"], row["mean_rg_nm_370K"]],
                color="0.78",
                linewidth=2,
                zorder=1,
            )

        ax.scatter(x, sub["mean_rg_nm_300K"], s=70, label="300 K", zorder=3)
        ax.scatter(x, sub["mean_rg_nm_370K"], s=70, label="370 K", zorder=3)

        add_refs(ax)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["id"], rotation=90, fontsize=7)
        ax.set_ylabel("Mean radius of gyration (nm)")
        ax.set_title(f"{group}: paired Rg response at 300 K and 370 K")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

        fig.tight_layout()
        fig.savefig(famdir / f"{group}_paired_rg.png", dpi=300)
        plt.close()


def plot_heatmaps(df, outdir):
    heatdir = outdir / "heatmaps"
    heatdir.mkdir(exist_ok=True)

    for group, sub in df.groupby("group"):
        # useful mainly for families with enough designs
        if len(sub) < 6:
            continue

        # row key: remove architecture type from end by using first two ID components where possible
        tmp = sub.copy()
        tmp["row_key"] = tmp["id"].apply(lambda x: "_".join(x.split("_")[1:3]) if "_R" in x else x.split("_")[1])
        pivot = tmp.pivot_table(
            index="row_key",
            columns="architecture_type",
            values="delta_mean_rg_370_minus_300_nm",
            aggfunc="mean",
        )

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.45)))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([clean_label(c) for c in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)

        ax.set_title(f"{group}: ΔRg heatmap")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("ΔRg = Rg370 − Rg300 (nm)")

        fig.tight_layout()
        fig.savefig(heatdir / f"{group}_delta_rg_heatmap.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="per_architecture_300_vs_370.csv")
    parser.add_argument("--outdir", default="summary_plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)

    plot_rg300_vs_rg370(df, outdir)
    plot_delta_vs_rg370(df, outdir)
    plot_delta_hist(df, outdir)
    plot_family_boxplot(df, outdir)
    plot_architecture_type_boxplot(df, outdir)
    plot_paired_family(df, outdir)
    plot_heatmaps(df, outdir)

    print(f"Done. Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()