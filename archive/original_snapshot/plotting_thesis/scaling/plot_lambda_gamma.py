import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps


def main():

    # ==============================================================
    # Command-line arguments
    # ==============================================================

    parser = argparse.ArgumentParser(
        description=(
            "Plot temperature-dependent CALVADOS interaction parameters "
            "for selected residues at different TEA gamma values."
        )
    )

    parser.add_argument(
        "--residues-root",
        required=True,
        help="Folder containing the TEA residue CSV files."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory in which the figures will be saved."
    )

    args = parser.parse_args()

    residues_root = Path(args.residues_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    # ==============================================================
    # Analysis settings
    # ==============================================================

    # Residues selected from E_AI_triblock:
    # I = strong temperature response
    # V = intermediate temperature response
    # A = relatively weak temperature response
    residues = ["I", "V", "A"]

    residue_names = {
        "I": "Isoleucine (I)",
        "V": "Valine (V)",
        "A": "Alanine (A)",
    }

    gammas = list(range(1, 8))
    temperatures = list(range(300, 371, 10))


    # ==============================================================
    # Thesis plotting style
    # ==============================================================

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,

        # Keep fonts editable when exporting vector graphics
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


    # ==============================================================
    # Colour palette
    # ==============================================================

    # Sequential colour map because gamma is an ordered variable.
    #
    # Avoid the extreme ends of viridis:
    # - very dark purple can become difficult to distinguish
    # - bright yellow can disappear against a white background

    cmap = colormaps["viridis"]

    colors = cmap(
        np.linspace(0.08, 0.88, len(gammas))
    )


    # ==============================================================
    # Load all data
    # ==============================================================

    data = {
        gamma: {
            residue: []
            for residue in residues
        }
        for gamma in gammas
    }

    for gamma in gammas:

        for temperature in temperatures:

            csv_file = (
                residues_root
                / (
                    f"residues_gamma_{gamma}p0_"
                    f"{temperature}K_argdefault_minus.csv"
                )
            )

            if not csv_file.exists():
                raise FileNotFoundError(
                    f"Could not find residue file:\n{csv_file}"
                )

            df = pd.read_csv(csv_file)

            for residue in residues:

                row = df[df["one"] == residue]

                if row.empty:
                    raise ValueError(
                        f"Residue '{residue}' not found in:\n"
                        f"{csv_file}"
                    )

                lambda_value = row.iloc[0]["lambdas"]

                data[gamma][residue].append(lambda_value)


    # ==============================================================
    # Create figure
    # ==============================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.7),
        sharex=True,
        sharey=True
    )


    # ==============================================================
    # Plot curves
    # ==============================================================

    for gamma, color in zip(gammas, colors):

        for ax, residue in zip(axes, residues):

            ax.plot(
                temperatures,
                data[gamma][residue],
                color=color,
                marker="o",
                markeredgewidth=0,
                label=rf"$\gamma = {gamma}$"
            )


    # ==============================================================
    # Format axes
    # ==============================================================

    for ax, residue in zip(axes, residues):

        ax.set_title(
            residue_names[residue],
            pad=8
        )

        # Same axes for every residue
        ax.set_xlim(298, 372)
        ax.set_ylim(0, 1.35)

        ax.set_xticks(
            np.arange(300, 371, 10)
        )

        ax.set_yticks(
            np.arange(0, 1.31, 0.2)
        )

        ax.set_xlabel(
            "Temperature (K)"
        )

        # Subtle grid
        ax.grid(
            True,
            linewidth=0.6,
            alpha=0.15
        )

        ax.tick_params(
            direction="out",
            length=4,
            width=0.8
        )

        # Keep axes visually clean
        ax.spines["top"].set_linewidth(0.8)
        ax.spines["right"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_linewidth(0.8)


    axes[0].set_ylabel(
        r"Interaction parameter, $\lambda$"
    )


    # ==============================================================
    # Shared legend
    # ==============================================================

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.3
    )


    # ==============================================================
    # Figure spacing
    # ==============================================================

    fig.subplots_adjust(
        top=0.80,
        bottom=0.18,
        left=0.075,
        right=0.99,
        wspace=0.10
    )


    # ==============================================================
    # Export
    # ==============================================================

    pdf_file = (
        outdir
        / "lambda_IVA_vs_temperature_gamma.pdf"
    )

    png_file = (
        outdir
        / "lambda_IVA_vs_temperature_gamma.png"
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight"
    )

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("Plotting complete.")
    print(f"PDF: {pdf_file}")
    print(f"PNG: {png_file}")
    print()


if __name__ == "__main__":
    main()