#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Expected replicate-to-replicate SD in mean Rg
# From VGIPKG triplicate dataset
# ============================================================

REPLICATE_NOISE_RG = {
    300: 0.05646,
    370: 0.04207,
}


# ============================================================
# Helpers
# ============================================================

def read_table(path):
    path = Path(path)

    return pd.read_csv(
        path,
        sep=None,
        engine="python"
    )


def normalize_temperature(value):

    if pd.isna(value):
        return np.nan

    text = (
        str(value)
        .strip()
        .upper()
        .replace("K", "")
        .strip()
    )

    return int(
        round(float(text))
    )


def parse_dataset_argument(dataset_string):
    """
    Expected format:

        LABEL::DIRECTORY

    Example:

        VGAVG::E:\\...\\VGAVG_detailed_stride1
    """

    if "::" not in dataset_string:
        raise ValueError(
            "Dataset must have format LABEL::DIRECTORY\n"
            f"Received: {dataset_string}"
        )

    label, folder = dataset_string.split(
        "::",
        1
    )

    return (
        label.strip(),
        Path(folder)
    )


# ============================================================
# Load one library
# ============================================================

def load_library(
    label,
    folder,
    r4_only=False
):

    all_rg_path = (
        folder
        /
        "all_rg_values.csv"
    )

    if not all_rg_path.exists():

        raise FileNotFoundError(
            f"\nCould not find:\n{all_rg_path}"
        )

    print("\n========================================")
    print(f"Loading: {label}")
    print(f"Folder: {folder}")
    print("========================================")

    rg = read_table(
        all_rg_path
    )

    required = [
        "id",
        "temp",
        "rg_nm"
    ]

    for column in required:

        if column not in rg.columns:

            raise ValueError(
                f"{column} missing from {all_rg_path}"
            )

    # --------------------------------------------------------
    # Normalize temperature
    # --------------------------------------------------------

    rg["temp_K"] = (
        rg["temp"]
        .apply(normalize_temperature)
    )

    # --------------------------------------------------------
    # Optional R4 filtering
    # --------------------------------------------------------

    if r4_only:

        ids_before = (
            rg["id"]
            .nunique()
        )

        rg = (
            rg[
                rg["id"]
                .astype(str)
                .str.contains(
                    r"_R4_",
                    case=False,
                    regex=True,
                    na=False
                )
            ]
            .copy()
        )

        ids_after = (
            rg["id"]
            .nunique()
        )

        print(
            f"R4 filter applied: "
            f"{ids_before} -> {ids_after} architectures"
        )

        if ids_after == 0:

            raise ValueError(
                f"{label}: R4 filter returned zero architectures."
            )

    # --------------------------------------------------------
    # Calculate architecture mean Rg
    # --------------------------------------------------------

    result = (
        rg
        .groupby(
            [
                "id",
                "temp_K"
            ],
            as_index=False
        )
        .agg(
            mean_rg_nm=(
                "rg_nm",
                "mean"
            ),

            rg_sd_frames_nm=(
                "rg_nm",
                "std"
            ),

            n_frames=(
                "rg_nm",
                "size"
            )
        )
    )

    result[
        "library"
    ] = label

    print(
        f"Architectures loaded: "
        f"{result['id'].nunique()}"
    )

    print(
        "Temperatures:",
        sorted(
            result[
                "temp_K"
            ]
            .unique()
        )
    )

    return result


# ============================================================
# Architecture-spread summary
# ============================================================

def calculate_library_summary(
    combined
):

    rows = []

    for (
        library,
        temp_K
    ), group in combined.groupby(
        [
            "library",
            "temp_K"
        ]
    ):

        values = (
            group[
                "mean_rg_nm"
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        architecture_sd = (
            values.std(
                ddof=1
            )
        )

        architecture_range = (
            values.max()
            -
            values.min()
        )

        replicate_sd = (
            REPLICATE_NOISE_RG.get(
                int(temp_K),
                np.nan
            )
        )

        if (
            not pd.isna(replicate_sd)
            and
            replicate_sd > 0
        ):

            signal_to_noise = (
                architecture_sd
                /
                replicate_sd
            )

        else:

            signal_to_noise = np.nan

        rows.append({

            "library":
                library,

            "temp_K":
                int(temp_K),

            "n_architectures":
                group["id"].nunique(),

            "mean_rg_library_mean_nm":
                values.mean(),

            "mean_rg_library_median_nm":
                values.median(),

            "mean_rg_architecture_sd_nm":
                architecture_sd,

            "mean_rg_architecture_range_nm":
                architecture_range,

            "mean_rg_min_nm":
                values.min(),

            "mean_rg_max_nm":
                values.max(),

            "expected_replicate_sd_nm":
                replicate_sd,

            "architecture_signal_S":
                signal_to_noise
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Plot architecture distributions
# ============================================================

def plot_rg_distributions(
    combined,
    library_order,
    temp_K,
    outdir
):

    data = (
        combined[
            combined["temp_K"]
            ==
            temp_K
        ]
        .copy()
    )

    if len(data) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    rng = (
        np.random.default_rng(
            12345
        )
    )

    noise = (
        REPLICATE_NOISE_RG.get(
            int(temp_K),
            np.nan
        )
    )

    for index, library in enumerate(
        library_order
    ):

        library_data = (
            data[
                data["library"]
                ==
                library
            ]
        )

        values = (
            library_data[
                "mean_rg_nm"
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        jitter = (
            rng.uniform(
                -0.17,
                0.17,
                size=len(values)
            )
        )

        x = (
            index
            +
            jitter
        )

        # individual architectures
        ax.scatter(
            x,
            values,
            s=50,
            alpha=0.85,
            zorder=3
        )

        # median
        median = (
            values.median()
        )

        ax.plot(
            [
                index - 0.24,
                index + 0.24
            ],
            [
                median,
                median
            ],
            color="black",
            linewidth=2,
            zorder=5
        )

        # expected replicate noise
        if (
            not pd.isna(noise)
        ):

            ax.fill_between(
                [
                    index - 0.32,
                    index + 0.32
                ],
                median - noise,
                median + noise,
                alpha=0.15,
                color="grey",
                zorder=1
            )

    ax.set_xticks(
        range(
            len(library_order)
        )
    )

    ax.set_xticklabels(
        library_order
    )

    ax.set_ylabel(
        "Mean Rg (nm)"
    )

    ax.set_title(
        (
            f"Architecture-dependent mean Rg at {temp_K} K\n"
            "Dots = architectures; black line = library median"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(
        True
    )

    fig.text(
        0.5,
        0.015,
        (
            "Grey region = library median ± expected replicate SD "
            "from VGIPKG triplicate simulations."
        ),
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1
        ]
    )

    output = (
        outdir
        /
        f"mean_rg_architecture_distribution_{temp_K}K.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# Plot architecture sensitivity S
# ============================================================

def plot_signal_summary(
    summary,
    library_order,
    outdir
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    x = np.arange(
        len(library_order)
    )

    offsets = {
        300: -0.12,
        370: 0.12
    }

    for temp_K in [
        300,
        370
    ]:

        subset = (
            summary[
                summary["temp_K"]
                ==
                temp_K
            ]
            .set_index(
                "library"
            )
        )

        y = []

        for library in library_order:

            if library in subset.index:

                y.append(
                    subset.loc[
                        library,
                        "architecture_signal_S"
                    ]
                )

            else:

                y.append(
                    np.nan
                )

        ax.scatter(
            x
            +
            offsets[
                temp_K
            ],
            y,
            s=70,
            label=f"{temp_K} K"
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        color="black",
        alpha=0.7
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        library_order
    )

    ax.set_ylabel(
        (
            r"Architecture sensitivity "
            r"$S = \sigma_{\mathrm{architecture}}/"
            r"\sigma_{\mathrm{replicate}}$"
        )
    )

    ax.set_title(
        "Architecture sensitivity of mean Rg"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(
        True
    )

    fig.text(
        0.5,
        0.015,
        (
            "S > 1: architecture-to-architecture spread exceeds "
            "the expected replicate SD."
        ),
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1
        ]
    )

    output = (
        outdir
        /
        "architecture_sensitivity_mean_rg.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# Plot architecture range
# ============================================================

def plot_architecture_range(
    summary,
    library_order,
    outdir
):

    fig, ax = plt.subplots(
        figsize=(
            10,
            6
        )
    )

    x = np.arange(
        len(library_order)
    )

    offsets = {
        300: -0.12,
        370: 0.12
    }

    for temp_K in [
        300,
        370
    ]:

        subset = (
            summary[
                summary["temp_K"]
                ==
                temp_K
            ]
            .set_index(
                "library"
            )
        )

        y = []

        for library in library_order:

            if library in subset.index:

                y.append(
                    subset.loc[
                        library,
                        "mean_rg_architecture_range_nm"
                    ]
                )

            else:

                y.append(
                    np.nan
                )

        ax.scatter(
            x
            +
            offsets[
                temp_K
            ],
            y,
            s=70,
            label=f"{temp_K} K"
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        library_order
    )

    ax.set_ylabel(
        "Architecture-dependent Rg range (nm)"
    )

    ax.set_title(
        "Total effect of architecture on mean Rg"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(
        True
    )

    plt.tight_layout()

    output = (
        outdir
        /
        "architecture_range_mean_rg.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare architecture sensitivity of mean Rg "
            "across CALVADOS motif libraries."
        )
    )

    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset as LABEL::DIRECTORY. "
            "Supply once per library."
        )
    )

    parser.add_argument(
        "--r4-only",
        action="append",
        default=[],
        help=(
            "Library label for which only IDs containing "
            "'_R4_' should be included. "
            "Can be supplied multiple times."
        )
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    r4_libraries = {
        x.upper()
        for x in args.r4_only
    }

    results = []
    library_order = []

    for dataset_string in args.dataset:

        label, folder = (
            parse_dataset_argument(
                dataset_string
            )
        )

        library_order.append(
            label
        )

        result = (
            load_library(
                label=label,
                folder=folder,
                r4_only=(
                    label.upper()
                    in
                    r4_libraries
                )
            )
        )

        results.append(
            result
        )

    combined = pd.concat(
        results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save architecture-level values
    # --------------------------------------------------------

    combined_output = (
        outdir
        /
        "architecture_mean_rg_values.csv"
    )

    combined.to_csv(
        combined_output,
        index=False
    )

    print(
        f"\nSaved: {combined_output}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        calculate_library_summary(
            combined
        )
    )

    summary_output = (
        outdir
        /
        "architecture_mean_rg_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False
    )

    print(
        f"Saved: {summary_output}"
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "MEAN RG ARCHITECTURE SENSITIVITY"
    )

    print(
        "========================================"
    )

    for _, row in summary.iterrows():

        print(
            f"\n{row['library']} "
            f"{int(row['temp_K'])} K"
        )

        print(
            f"  N architectures = "
            f"{int(row['n_architectures'])}"
        )

        print(
            f"  architecture SD = "
            f"{row['mean_rg_architecture_sd_nm']:.4f} nm"
        )

        print(
            f"  architecture range = "
            f"{row['mean_rg_architecture_range_nm']:.4f} nm"
        )

        print(
            f"  expected replicate SD = "
            f"{row['expected_replicate_sd_nm']:.4f} nm"
        )

        print(
            f"  S = "
            f"{row['architecture_signal_S']:.2f}"
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    for temp_K in [
        300,
        370
    ]:

        plot_rg_distributions(
            combined=combined,
            library_order=library_order,
            temp_K=temp_K,
            outdir=outdir
        )

    plot_signal_summary(
        summary=summary,
        library_order=library_order,
        outdir=outdir
    )

    plot_architecture_range(
        summary=summary,
        library_order=library_order,
        outdir=outdir
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()