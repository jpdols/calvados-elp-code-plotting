#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Expected replicate noise for mean Rg
# Fallback values from VGIPKG triplicate dataset
# ============================================================

FALLBACK_MEAN_RG_NOISE = {
    300: 0.05646,
    370: 0.04207,
}


# ============================================================
# Motif definitions per library
# ============================================================

LIBRARY_MOTIFS = {
    "VGAVG": ("VPGVG", "VPAVG"),
    "VGIP":  ("VPGVG", "IPAVG"),
    "IGIP":  ("VPGIG", "IPAVG"),
    "VGIG":  ("VPGVG", "VPGIG"),
    "VAIP":  ("VPAVG", "IPAVG"),
    "VAIG":  ("VPAVG", "VPGIG"),
}


# ============================================================
# Basic helpers
# ============================================================

def read_csv(path):
    return pd.read_csv(
        Path(path),
        sep=None,
        engine="python"
    )




def save_png(fig, output, dpi=300):
    """Save a PNG robustly on Windows, with a short-name fallback."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output.open("wb") as f:
            fig.savefig(
                f,
                format="png",
                dpi=dpi,
                bbox_inches="tight"
            )
        print(f"Saved: {output}")
        return output

    except OSError as exc:
        # Some Windows/Pillow setups reject particular long filenames.
        # Retry with a shorter filename in the same output directory.
        short_stem = output.stem[-24:]
        fallback = output.parent / f"plot_{short_stem}.png"

        print(f"WARNING: could not save to {output!s}")
        print(f"Reason: {exc}")
        print(f"Retrying with: {fallback!s}")

        with fallback.open("wb") as f:
            fig.savefig(
                f,
                format="png",
                dpi=dpi,
                bbox_inches="tight"
            )

        print(f"Saved: {fallback}")
        return fallback


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

    if "::" not in dataset_string:

        raise ValueError(
            "Dataset must have format:\n"
            "LABEL::DIRECTORY\n\n"
            f"Received:\n{dataset_string}"
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
# Lambda table
# ============================================================

def load_lambda_table(path):

    df = pd.read_csv(
        path
    )

    if "one" not in df.columns:
        raise ValueError(
            "Residue lambda file must contain column 'one'."
        )

    if "lambdas" not in df.columns:
        raise ValueError(
            "Residue lambda file must contain column 'lambdas'."
        )

    lambda_by_residue = dict(
        zip(
            df["one"].astype(str),
            pd.to_numeric(
                df["lambdas"],
                errors="coerce"
            )
        )
    )

    return lambda_by_residue


def motif_lambda(
    sequence,
    lambda_by_residue
):

    values = []

    for residue in sequence:

        if residue not in lambda_by_residue:

            raise KeyError(
                f"Residue '{residue}' from motif "
                f"'{sequence}' not found in lambda table."
            )

        values.append(
            lambda_by_residue[
                residue
            ]
        )

    return float(
        np.mean(
            values
        )
    )


def calculate_library_lambda_properties(
    lambda_by_residue
):

    rows = []

    for library, (
        motif1,
        motif2
    ) in LIBRARY_MOTIFS.items():

        lambda1 = motif_lambda(
            motif1,
            lambda_by_residue
        )

        lambda2 = motif_lambda(
            motif2,
            lambda_by_residue
        )

        rows.append({

            "library":
                library,

            "motif1":
                motif1,

            "motif2":
                motif2,

            "lambda_motif1":
                lambda1,

            "lambda_motif2":
                lambda2,

            "delta_lambda":
                abs(
                    lambda1
                    -
                    lambda2
                ),

            "mean_pair_lambda":
                (
                    lambda1
                    +
                    lambda2
                )
                /
                2
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Load VGIPKG replicate-noise reference
# ============================================================

def load_noise_reference(
    reference_path=None
):

    noise = {
        300: {
            "mean_rg_nm":
                FALLBACK_MEAN_RG_NOISE[300],

            "rg_iqr_nm":
                np.nan
        },

        370: {
            "mean_rg_nm":
                FALLBACK_MEAN_RG_NOISE[370],

            "rg_iqr_nm":
                np.nan
        }
    }

    if reference_path is None:

        print(
            "\nNo VGIPKG reference supplied."
        )

        print(
            "Mean-Rg S values can still be calculated."
        )

        print(
            "Rg-IQR S values will be unavailable."
        )

        return noise

    reference_path = Path(
        reference_path
    )

    if not reference_path.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{reference_path}"
        )

    ref = pd.read_csv(
        reference_path
    )

    print(
        "\n========================================"
    )

    print(
        "VGIPKG REPLICATE-NOISE REFERENCE"
    )

    print(
        "========================================"
    )

    print(
        reference_path
    )

    for temp in [
        300,
        370
    ]:

        # ----------------------------------------------------
        # Mean Rg replicate SD
        # ----------------------------------------------------

        rg_sd_col = (
            f"rg_mean_nm_{temp}K_sd"
        )

        if rg_sd_col in ref.columns:

            values = (
                pd.to_numeric(
                    ref[
                        rg_sd_col
                    ],
                    errors="coerce"
                )
                .dropna()
            )

            if len(values) > 0:

                noise[
                    temp
                ][
                    "mean_rg_nm"
                ] = (
                    values.mean()
                )

        # ----------------------------------------------------
        # Rg IQR replicate SD
        # ----------------------------------------------------

        iqr_candidates = [
            f"rg_iqr_nm_{temp}K_sd",
            f"iqr_rg_nm_{temp}K_sd",
        ]

        for col in iqr_candidates:

            if col in ref.columns:

                values = (
                    pd.to_numeric(
                        ref[
                            col
                        ],
                        errors="coerce"
                    )
                    .dropna()
                )

                if len(values) > 0:

                    noise[
                        temp
                    ][
                        "rg_iqr_nm"
                    ] = (
                        values.mean()
                    )

                    break

        print(
            f"\n{temp} K"
        )

        print(
            "  mean Rg replicate SD = "
            f"{noise[temp]['mean_rg_nm']:.5f}"
        )

        if np.isfinite(
            noise[
                temp
            ][
                "rg_iqr_nm"
            ]
        ):

            print(
                "  Rg IQR replicate SD = "
                f"{noise[temp]['rg_iqr_nm']:.5f}"
            )

        else:

            print(
                "  Rg IQR replicate SD = unavailable"
            )

    return noise


# ============================================================
# Load one motif library
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
            f"Could not find:\n"
            f"{all_rg_path}"
        )

    print(
        "\n========================================"
    )

    print(
        f"Loading: {label}"
    )

    print(
        f"Folder: {folder}"
    )

    print(
        "========================================"
    )

    rg = read_csv(
        all_rg_path
    )

    required = [
        "id",
        "temp",
        "rg_nm"
    ]

    for col in required:

        if col not in rg.columns:

            raise ValueError(
                f"{col} missing from:\n"
                f"{all_rg_path}"
            )

    rg[
        "temp_K"
    ] = (
        rg[
            "temp"
        ]
        .apply(
            normalize_temperature
        )
    )

    rg[
        "rg_nm"
    ] = (
        pd.to_numeric(
            rg[
                "rg_nm"
            ],
            errors="coerce"
        )
    )

    rg = (
        rg
        .dropna(
            subset=[
                "id",
                "temp_K",
                "rg_nm"
            ]
        )
        .copy()
    )

    # ========================================================
    # R4 filtering
    # ========================================================

    if r4_only:

        n_before = (
            rg[
                "id"
            ]
            .nunique()
        )

        rg = (
            rg[
                rg[
                    "id"
                ]
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

        n_after = (
            rg[
                "id"
            ]
            .nunique()
        )

        print(
            f"R4 filter: "
            f"{n_before} -> {n_after} architectures"
        )

        if n_after == 0:

            raise ValueError(
                f"{label}: R4 filter returned "
                "zero architectures."
            )

    # ========================================================
    # Calculate architecture properties
    # ========================================================

    rows = []

    for (
        architecture,
        temp_K
    ), group in rg.groupby(
        [
            "id",
            "temp_K"
        ],
        sort=False
    ):

        values = (
            group[
                "rg_nm"
            ]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        q25 = np.quantile(
            values,
            0.25
        )

        q75 = np.quantile(
            values,
            0.75
        )

        rows.append({

            "library":
                label,

            "id":
                architecture,

            "temp_K":
                int(
                    temp_K
                ),

            "mean_rg_nm":
                np.mean(
                    values
                ),

            "rg_iqr_nm":
                q75
                -
                q25,

            "rg_q25_nm":
                q25,

            "rg_q75_nm":
                q75,

            "rg_frame_sd_nm":
                np.std(
                    values,
                    ddof=1
                ),

            "n_frames":
                len(
                    values
                )
        })

    result = pd.DataFrame(
        rows
    )

    print(
        "Architectures loaded:",
        result[
            "id"
        ].nunique()
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
# Architecture signal S
# ============================================================

def architecture_signal(
    values,
    replicate_noise
):

    values = (
        pd.to_numeric(
            values,
            errors="coerce"
        )
        .dropna()
    )

    if len(values) < 2:

        return (
            np.nan,
            np.nan
        )

    architecture_sd = (
        values.std(
            ddof=1
        )
    )

    if (
        not np.isfinite(
            replicate_noise
        )
        or
        replicate_noise <= 0
    ):

        return (
            architecture_sd,
            np.nan
        )

    S = (
        architecture_sd
        /
        replicate_noise
    )

    return (
        architecture_sd,
        S
    )


# ============================================================
# Library summary
# ============================================================

def calculate_summary(
    combined,
    noise
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

        temp_K = int(
            temp_K
        )

        mean_rg_values = (
            group[
                "mean_rg_nm"
            ]
        )

        iqr_values = (
            group[
                "rg_iqr_nm"
            ]
        )

        mean_rg_noise = (
            noise[
                temp_K
            ][
                "mean_rg_nm"
            ]
        )

        iqr_noise = (
            noise[
                temp_K
            ][
                "rg_iqr_nm"
            ]
        )

        (
            mean_rg_arch_sd,
            mean_rg_S
        ) = architecture_signal(
            mean_rg_values,
            mean_rg_noise
        )

        (
            iqr_arch_sd,
            iqr_S
        ) = architecture_signal(
            iqr_values,
            iqr_noise
        )

        rows.append({

            "library":
                library,

            "temp_K":
                temp_K,

            "n_architectures":
                group[
                    "id"
                ].nunique(),

            # Mean Rg
            "mean_rg_library_mean_nm":
                mean_rg_values.mean(),

            "mean_rg_library_median_nm":
                mean_rg_values.median(),

            "mean_rg_architecture_sd_nm":
                mean_rg_arch_sd,

            "mean_rg_architecture_range_nm":
                (
                    mean_rg_values.max()
                    -
                    mean_rg_values.min()
                ),

            "mean_rg_replicate_noise_sd_nm":
                mean_rg_noise,

            "mean_rg_S":
                mean_rg_S,

            # Rg IQR
            "rg_iqr_library_mean_nm":
                iqr_values.mean(),

            "rg_iqr_library_median_nm":
                iqr_values.median(),

            "rg_iqr_architecture_sd_nm":
                iqr_arch_sd,

            "rg_iqr_architecture_range_nm":
                (
                    iqr_values.max()
                    -
                    iqr_values.min()
                ),

            "rg_iqr_replicate_noise_sd_nm":
                iqr_noise,

            "rg_iqr_S":
                iqr_S
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Distribution plots
# ============================================================

def plot_property_distribution(
    combined,
    library_order,
    temp_K,
    metric,
    ylabel,
    title,
    noise,
    output
):

    data = (
        combined[
            combined[
                "temp_K"
            ]
            ==
            temp_K
        ]
        .copy()
    )

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            11,
            6
        )
    )

    rng = (
        np.random.default_rng(
            12345
        )
    )

    replicate_noise = (
        noise[
            temp_K
        ][
            metric
        ]
    )

    for index, library in enumerate(
        library_order
    ):

        values = (
            data.loc[
                data[
                    "library"
                ]
                ==
                library,
                metric
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        jitter = (
            rng.uniform(
                -0.17,
                0.17,
                size=len(
                    values
                )
            )
        )

        ax.scatter(
            index
            +
            jitter,
            values,
            s=52,
            alpha=0.85,
            zorder=3
        )

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
            linewidth=2.2,
            zorder=5
        )

        if np.isfinite(
            replicate_noise
        ):

            ax.fill_between(
                [
                    index - 0.32,
                    index + 0.32
                ],
                median
                -
                replicate_noise,
                median
                +
                replicate_noise,
                color="grey",
                alpha=0.16,
                zorder=1
            )

    ax.set_xticks(
        range(
            len(
                library_order
            )
        )
    )

    ax.set_xticklabels(
        library_order
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        f"{title} at {temp_K} K\n"
        "Dots = architectures; "
        "black line = library median"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.set_axisbelow(
        True
    )

    plt.tight_layout()

    save_png(
        fig,
        output,
        dpi=300
    )

    plt.close(fig)


# ============================================================
# S comparison by library
# ============================================================

def plot_S_comparison(
    summary,
    library_order,
    temp_K,
    outdir
):

    data = (
        summary[
            summary[
                "temp_K"
            ]
            ==
            temp_K
        ]
        .set_index(
            "library"
        )
    )

    x = np.arange(
        len(
            library_order
        )
    )

    mean_rg_S = []

    iqr_S = []

    for library in library_order:

        if library in data.index:

            row = (
                data.loc[
                    library
                ]
            )

            mean_rg_S.append(
                row[
                    "mean_rg_S"
                ]
            )

            iqr_S.append(
                row[
                    "rg_iqr_S"
                ]
            )

        else:

            mean_rg_S.append(
                np.nan
            )

            iqr_S.append(
                np.nan
            )

    fig, ax = plt.subplots(
        figsize=(
            11,
            6
        )
    )

    ax.scatter(
        x - 0.10,
        mean_rg_S,
        s=80,
        label="Mean Rg"
    )

    ax.scatter(
        x + 0.10,
        iqr_S,
        s=80,
        marker="s",
        label="Rg IQR"
    )

    # Numeric labels
    for i, value in enumerate(
        mean_rg_S
    ):

        if np.isfinite(
            value
        ):

            ax.annotate(
                f"{value:.2f}",
                (
                    x[i] - 0.10,
                    value
                ),
                xytext=(
                    0,
                    7
                ),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

    for i, value in enumerate(
        iqr_S
    ):

        if np.isfinite(
            value
        ):

            ax.annotate(
                f"{value:.2f}",
                (
                    x[i] + 0.10,
                    value
                ),
                xytext=(
                    0,
                    7
                ),
                textcoords="offset points",
                ha="center",
                fontsize=9
            )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        library_order
    )

    ax.set_ylabel(
        "Architecture signal S"
    )

    ax.set_title(
        f"Architecture sensitivity at {temp_K} K"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    output = (
        outdir
        /
        f"architecture_signal_meanRg_IQR_{temp_K}K.png"
    )

    save_png(
        fig,
        output,
        dpi=300
    )

    plt.close(
        fig
    )


# ============================================================
# Correlation helper
# ============================================================

def calculate_correlations(
    df,
    x_col,
    y_col
):

    subset = (
        df[
            [
                x_col,
                y_col
            ]
        ]
        .dropna()
    )

    if len(
        subset
    ) < 3:

        return (
            np.nan,
            np.nan
        )

    pearson = (
        subset[
            x_col
        ]
        .corr(
            subset[
                y_col
            ],
            method="pearson"
        )
    )

    spearman = (
        subset[
            x_col
        ]
        .corr(
            subset[
                y_col
            ],
            method="spearman"
        )
    )

    return (
        pearson,
        spearman
    )


# ============================================================
# Delta lambda vs S plot
# ============================================================

def plot_delta_lambda_vs_S(
    analysis,
    temp_K,
    y_col,
    ylabel,
    title,
    output
):

    data = (
        analysis[
            analysis[
                "temp_K"
            ]
            ==
            temp_K
        ]
        .dropna(
            subset=[
                "delta_lambda",
                y_col
            ]
        )
        .copy()
    )

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            8,
            6
        )
    )

    ax.scatter(
        data[
            "delta_lambda"
        ],
        data[
            y_col
        ],
        s=90,
        zorder=3
    )

    # --------------------------------------------------------
    # Annotate library names
    # --------------------------------------------------------

    for _, row in data.iterrows():

        ax.annotate(
            row[
                "library"
            ],
            (
                row[
                    "delta_lambda"
                ],
                row[
                    y_col
                ]
            ),
            xytext=(
                6,
                6
            ),
            textcoords="offset points",
            fontsize=10
        )

    # --------------------------------------------------------
    # S = 1 reference
    # --------------------------------------------------------

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.7
    )

    # --------------------------------------------------------
    # Linear fit as descriptive guide only
    # --------------------------------------------------------

    if len(
        data
    ) >= 3:

        x = (
            data[
                "delta_lambda"
            ]
            .to_numpy()
        )

        y = (
            data[
                y_col
            ]
            .to_numpy()
        )

        slope, intercept = (
            np.polyfit(
                x,
                y,
                1
            )
        )

        xfit = np.linspace(
            x.min(),
            x.max(),
            100
        )

        yfit = (
            slope
            *
            xfit
            +
            intercept
        )

        ax.plot(
            xfit,
            yfit,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    pearson, spearman = (
        calculate_correlations(
            data,
            "delta_lambda",
            y_col
        )
    )

    ax.set_xlabel(
        r"Motif hydrophobicity contrast "
        r"$|\Delta\lambda|$"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        f"{title} — {temp_K} K\n"
        f"Pearson r = {pearson:.2f}; "
        f"Spearman ρ = {spearman:.2f}"
    )

    ax.grid(
        alpha=0.2
    )

    ax.set_axisbelow(
        True
    )

    plt.tight_layout()

    save_png(
        fig,
        output,
        dpi=300
    )

    plt.close(
        fig
    )


# ============================================================
# Mean lambda vs S diagnostic
# ============================================================

def plot_mean_lambda_vs_S(
    analysis,
    temp_K,
    y_col,
    ylabel,
    title,
    output
):

    data = (
        analysis[
            analysis[
                "temp_K"
            ]
            ==
            temp_K
        ]
        .dropna(
            subset=[
                "mean_pair_lambda",
                y_col
            ]
        )
        .copy()
    )

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            8,
            6
        )
    )

    ax.scatter(
        data[
            "mean_pair_lambda"
        ],
        data[
            y_col
        ],
        s=90
    )

    for _, row in data.iterrows():

        ax.annotate(
            row[
                "library"
            ],
            (
                row[
                    "mean_pair_lambda"
                ],
                row[
                    y_col
                ]
            ),
            xytext=(
                6,
                6
            ),
            textcoords="offset points",
            fontsize=10
        )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.1,
        alpha=0.7
    )

    pearson, spearman = (
        calculate_correlations(
            data,
            "mean_pair_lambda",
            y_col
        )
    )

    ax.set_xlabel(
        r"Mean motif-pair $\lambda$"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        f"{title} — {temp_K} K\n"
        f"Pearson r = {pearson:.2f}; "
        f"Spearman ρ = {spearman:.2f}"
    )

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    save_png(
        fig,
        output,
        dpi=300
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Threshold-independent architecture analysis "
            "using mean Rg, Rg IQR and motif lambda contrast."
        )
    )

    parser.add_argument(
        "--dataset",
        action="append",
        required=True
    )

    parser.add_argument(
        "--r4-only",
        action="append",
        default=[]
    )

    parser.add_argument(
        "--noise-reference",
        required=True,
        help=(
            "VGIPKG architecture_properties_summary.csv"
        )
    )

    parser.add_argument(
        "--lambda-file",
        required=True,
        help=(
            "Residue lambda CSV containing columns "
            "'one' and 'lambdas'."
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

    # ========================================================
    # Noise
    # ========================================================

    noise = load_noise_reference(
        args.noise_reference
    )

    # ========================================================
    # Lambda properties
    # ========================================================

    lambda_by_residue = (
        load_lambda_table(
            args.lambda_file
        )
    )

    lambda_properties = (
        calculate_library_lambda_properties(
            lambda_by_residue
        )
    )

    lambda_properties.to_csv(
        outdir
        /
        "library_lambda_properties.csv",
        index=False
    )

    print(
        "\n========================================"
    )

    print(
        "LIBRARY LAMBDA PROPERTIES"
    )

    print(
        "========================================"
    )

    print(
        lambda_properties.to_string(
            index=False
        )
    )

    # ========================================================
    # Load libraries
    # ========================================================

    r4_libraries = {
        x.upper()
        for x in args.r4_only
    }

    library_order = []

    results = []

    for dataset_string in args.dataset:

        label, folder = (
            parse_dataset_argument(
                dataset_string
            )
        )

        library_order.append(
            label
        )

        result = load_library(
            label=label,
            folder=folder,
            r4_only=(
                label.upper()
                in
                r4_libraries
            )
        )

        results.append(
            result
        )

    combined = pd.concat(
        results,
        ignore_index=True
    )

    combined.to_csv(
        outdir
        /
        "architecture_threshold_independent_values.csv",
        index=False
    )

    # ========================================================
    # S summary
    # ========================================================

    summary = calculate_summary(
        combined,
        noise
    )

    # Add lambda properties
    analysis = summary.merge(
        lambda_properties,
        on="library",
        how="left",
        validate="many_to_one"
    )

    analysis.to_csv(
        outdir
        /
        "architecture_lambda_signal_summary.csv",
        index=False
    )

    # ========================================================
    # Print S values
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "ARCHITECTURE SIGNAL SUMMARY"
    )

    print(
        "========================================"
    )

    for _, row in analysis.iterrows():

        print(
            f"\n{row['library']} — "
            f"{int(row['temp_K'])} K"
        )

        print(
            f"  Δlambda = "
            f"{row['delta_lambda']:.5f}"
        )

        print(
            f"  mean pair lambda = "
            f"{row['mean_pair_lambda']:.5f}"
        )

        print(
            f"  S(mean Rg) = "
            f"{row['mean_rg_S']:.2f}"
        )

        print(
            f"  S(Rg IQR)  = "
            f"{row['rg_iqr_S']:.2f}"
        )

    # ========================================================
    # Property distributions
    # ========================================================

    for temp_K in [
        300,
        370
    ]:

        plot_property_distribution(
            combined,
            library_order,
            temp_K,
            "mean_rg_nm",
            "Mean Rg (nm)",
            "Mean compactness",
            noise,
            outdir
            /
            f"mean_rg_architecture_distribution_{temp_K}K.png"
        )

        plot_property_distribution(
            combined,
            library_order,
            temp_K,
            "rg_iqr_nm",
            "Rg IQR (nm)",
            "Ensemble breadth",
            noise,
            outdir
            /
            f"ensemble_breadth_rg_iqr_{temp_K}K.png"
        )

        plot_S_comparison(
            summary,
            library_order,
            temp_K,
            outdir
        )

        # ====================================================
        # PRIMARY HYPOTHESIS PLOTS
        # ====================================================

        plot_delta_lambda_vs_S(
            analysis=analysis,
            temp_K=temp_K,
            y_col="mean_rg_S",
            ylabel=r"$S_{\mathrm{mean\,Rg}}$",
            title=(
                "Motif hydrophobicity contrast vs "
                "architecture sensitivity of mean Rg"
            ),
            output=(
                outdir
                /
                f"delta_lambda_vs_S_meanRg_{temp_K}K.png"
            )
        )

        plot_delta_lambda_vs_S(
            analysis=analysis,
            temp_K=temp_K,
            y_col="rg_iqr_S",
            ylabel=r"$S_{\mathrm{Rg\,IQR}}$",
            title=(
                "Motif hydrophobicity contrast vs "
                "architecture sensitivity of ensemble breadth"
            ),
            output=(
                outdir
                /
                f"delta_lambda_vs_S_RgIQR_{temp_K}K.png"
            )
        )

        # ====================================================
        # CONTROL / DIAGNOSTIC PLOTS
        # ====================================================

        plot_mean_lambda_vs_S(
            analysis=analysis,
            temp_K=temp_K,
            y_col="mean_rg_S",
            ylabel=r"$S_{\mathrm{mean\,Rg}}$",
            title=(
                "Overall hydrophobicity vs "
                "architecture sensitivity of mean Rg"
            ),
            output=(
                outdir
                /
                f"mean_lambda_vs_S_meanRg_{temp_K}K.png"
            )
        )

        plot_mean_lambda_vs_S(
            analysis=analysis,
            temp_K=temp_K,
            y_col="rg_iqr_S",
            ylabel=r"$S_{\mathrm{Rg\,IQR}}$",
            title=(
                "Overall hydrophobicity vs "
                "architecture sensitivity of ensemble breadth"
            ),
            output=(
                outdir
                /
                f"mean_lambda_vs_S_RgIQR_{temp_K}K.png"
            )
        )

    # ========================================================
    # Save correlation summary
    # ========================================================

    correlation_rows = []

    for temp_K in [
        300,
        370
    ]:

        d = (
            analysis[
                analysis[
                    "temp_K"
                ]
                ==
                temp_K
            ]
        )

        for x_col in [
            "delta_lambda",
            "mean_pair_lambda"
        ]:

            for y_col in [
                "mean_rg_S",
                "rg_iqr_S"
            ]:

                pearson, spearman = (
                    calculate_correlations(
                        d,
                        x_col,
                        y_col
                    )
                )

                correlation_rows.append({

                    "temp_K":
                        temp_K,

                    "x_variable":
                        x_col,

                    "S_metric":
                        y_col,

                    "pearson_r":
                        pearson,

                    "spearman_rho":
                        spearman,

                    "n_libraries":
                        len(
                            d[
                                [
                                    x_col,
                                    y_col
                                ]
                            ]
                            .dropna()
                        )
                })

    correlation_table = pd.DataFrame(
        correlation_rows
    )

    correlation_table.to_csv(
        outdir
        /
        "lambda_vs_architecture_signal_correlations.csv",
        index=False
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()