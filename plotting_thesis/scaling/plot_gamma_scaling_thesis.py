import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main():
    parser = argparse.ArgumentParser(
        description="Plot gamma-scaling results for the thesis."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to gamma_scaling_summary.csv",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.61,
        help="Ideal-chain Rg reference in nm (default: 4.61)",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(csv_path)

    required_columns = {
        "gamma",
        "temperature_K",
        "mean_rg_nm",
        "fraction_collapsed",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values(["gamma", "temperature_K"])

    gammas = sorted(df["gamma"].unique())

    # --------------------------------------------------------
    # Colours
    #
    # Same viridis sampling as the lambda-vs-temperature plot:
    # gamma 1 = dark purple
    # gamma 7 = yellow/green
    # --------------------------------------------------------

    cmap = colormaps["viridis"]
    colors = cmap(np.linspace(0.08, 0.88, len(gammas)))

    gamma_colors = {
        gamma: color
        for gamma, color in zip(gammas, colors)
    }

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.3),
        constrained_layout=True,
    )

    ax_rg, ax_compact = axes

    # --------------------------------------------------------
    # Panel A: mean Rg
    # --------------------------------------------------------

    for gamma in gammas:
        sub = df[df["gamma"] == gamma]

        ax_rg.plot(
            sub["temperature_K"],
            sub["mean_rg_nm"],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=gamma_colors[gamma],
            label=rf"$\gamma={int(gamma)}$",
        )

    ax_rg.axhline(
        args.threshold,
        color="0.35",
        linestyle="--",
        linewidth=1.3,
        zorder=0,
    )

    ax_rg.text(
        0.98,
        args.threshold + 0.05,
        rf"$R_{{g,\Theta}}={args.threshold:.2f}$ nm",
        transform=ax_rg.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.3",
    )

    ax_rg.set_xlabel("Temperature (K)")
    ax_rg.set_ylabel(r"Mean $R_g$ (nm)")

    ax_rg.set_xticks(
        sorted(df["temperature_K"].unique())
    )

    ax_rg.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax_rg.text(
        -0.13,
        1.03,
        "A",
        transform=ax_rg.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Panel B: compact-state population
    # --------------------------------------------------------

    for gamma in gammas:
        sub = df[df["gamma"] == gamma]

        ax_compact.plot(
            sub["temperature_K"],
            sub["fraction_collapsed"],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=gamma_colors[gamma],
            label=rf"$\gamma={int(gamma)}$",
        )

    # 50% reference
    ax_compact.axhline(
        0.5,
        color="0.35",
        linestyle="--",
        linewidth=1.3,
        zorder=0,
    )

    ax_compact.text(
        0.98,
        0.52,
        r"$f_{\mathrm{compact}}=0.5$",
        transform=ax_compact.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.3",
    )

    ax_compact.set_xlabel("Temperature (K)")
    ax_compact.set_ylabel(r"Compact-state population, $f_{\mathrm{compact}}$")

    ax_compact.set_xticks(
        sorted(df["temperature_K"].unique())
    )

    ax_compact.set_ylim(0, 1.0)

    ax_compact.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax_compact.text(
        -0.13,
        1.03,
        "B",
        transform=ax_compact.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = ax_rg.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pdf_path = outdir / "gamma_scaling_thesis.pdf"
    png_path = outdir / "gamma_scaling_thesis.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()