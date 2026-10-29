#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Expected replicate-to-replicate noise
# Derived from VGIPKG triplicate simulations
# ============================================================

REPLICATE_NOISE = {
    300: {
        "mean_rg_nm": 0.05646,
        "k_collapse_per_ns": 0.02465,
        "k_escape_per_ns": 0.50398,
    },
    370: {
        "mean_rg_nm": 0.04207,
        "k_collapse_per_ns": 0.11850,
        "k_escape_per_ns": 0.09194,
    },
}


# ============================================================
# Helpers
# ============================================================

def read_table(path):
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

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


def convert_collapsed_column(values):
    if values.dtype == bool:
        return values

    if np.issubdtype(values.dtype, np.number):
        return values.astype(bool)

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "collapsed",
            "true",
            "1",
            "yes"
        ])
    )


def get_noise(temp_K, metric):
    return (
        REPLICATE_NOISE
        .get(
            int(temp_K),
            {}
        )
        .get(
            metric,
            np.nan
        )
    )


# ============================================================
# Parse dataset argument
# ============================================================

def parse_dataset_argument(dataset_string):
    """
    Format:
        LABEL::DIRECTORY
    """

    if "::" not in dataset_string:
        raise ValueError(
            "\nDataset must have format:\n"
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
# Load one library
# ============================================================

def load_library(label, folder):

    all_rg_path = (
        folder
        /
        "all_rg_values.csv"
    )

    summary_path = (
        folder
        /
        "per_run_summary.csv"
    )

    if not all_rg_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{all_rg_path}"
        )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{summary_path}"
        )

    print("\n========================================")
    print(f"Loading library: {label}")
    print(f"Folder: {folder}")
    print("========================================")

    rg = read_table(
        all_rg_path
    )

    summary = read_table(
        summary_path
    )

    required_rg = [
        "id",
        "temp",
        "rg_nm",
        "collapsed"
    ]

    required_summary = [
        "id",
        "temp",
        "k_collapse_per_ns",
        "k_escape_per_ns"
    ]

    for column in required_rg:
        if column not in rg.columns:
            raise ValueError(
                f"{column} missing from {all_rg_path}"
            )

    for column in required_summary:
        if column not in summary.columns:
            raise ValueError(
                f"{column} missing from {summary_path}"
            )

    rg["temp_K"] = (
        rg["temp"]
        .apply(normalize_temperature)
    )

    summary["temp_K"] = (
        summary["temp"]
        .apply(normalize_temperature)
    )

    rg["_collapsed"] = (
        convert_collapsed_column(
            rg["collapsed"]
        )
    )

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

        collapsed = (
            group[
                group["_collapsed"]
            ]
        )

        expanded = (
            group[
                ~group["_collapsed"]
            ]
        )

        rows.append({
            "library":
                label,

            "id":
                architecture,

            "temp_K":
                int(temp_K),

            "mean_rg_nm":
                group["rg_nm"].mean(),

            "mean_rg_collapsed_nm":
                (
                    collapsed["rg_nm"].mean()
                    if len(collapsed) > 0
                    else np.nan
                ),

            "mean_rg_uncollapsed_nm":
                (
                    expanded["rg_nm"].mean()
                    if len(expanded) > 0
                    else np.nan
                ),

            "fraction_collapsed":
                group["_collapsed"].mean(),

            "n_frames":
                len(group)
        })

    metrics = pd.DataFrame(
        rows
    )

    duplicate_mask = (
        summary.duplicated(
            subset=[
                "id",
                "temp_K"
            ],
            keep=False
        )
    )

    if duplicate_mask.any():

        print("\nDuplicate rows detected:")

        print(
            summary.loc[
                duplicate_mask,
                [
                    "id",
                    "temp_K"
                ]
            ]
        )

        raise ValueError(
            f"\n{label}: expected exactly one "
            "summary row per architecture + temperature."
        )

    kinetics = (
        summary[
            [
                "id",
                "temp_K",
                "k_collapse_per_ns",
                "k_escape_per_ns"
            ]
        ]
        .copy()
    )

    result = metrics.merge(
        kinetics,
        on=[
            "id",
            "temp_K"
        ],
        how="left",
        validate="one_to_one"
    )

    # ========================================================
    # VGIP-specific filtering:
    # only R4 = 50:50 VG/IP
    # ========================================================

    if label.upper() == "VGIP":

        n_before = result["id"].nunique()

        result = (
            result[
                result["id"]
                .astype(str)
                .str.contains(
                    r"_VGIP_R4_",
                    case=False,
                    regex=True,
                    na=False
                )
            ]
            .copy()
        )

        n_after = result["id"].nunique()

        print(
            "\nVGIP filter applied: "
            "R4 only (50:50 VG/IP)"
        )

        print(
            f"Architectures before filtering: {n_before}"
        )

        print(
            f"Architectures after filtering:  {n_after}"
        )

        if n_after == 0:
            raise ValueError(
                "VGIP R4 filter returned zero architectures. "
                "Check architecture ID naming."
            )

    print(
        f"Architectures loaded: "
        f"{result['id'].nunique()}"
    )

    print(
        "Temperatures:"
    )

    print(
        sorted(
            result["temp_K"].unique()
        )
    )

    return result


# ============================================================
# Draw metric
# ============================================================

def draw_metric(
    ax,
    data,
    libraries,
    metric,
    ylabel,
    title,
    temp_K,
    show_noise=True,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng(
            42
        )

    noise = (
        get_noise(
            temp_K,
            metric
        )
        if show_noise
        else np.nan
    )

    band_width = 0.70

    for index, library in enumerate(
        libraries
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
                metric
            ]
            .dropna()
        )

        if len(values) == 0:
            continue

        # jitter for individual architecture dots
        jitter = (
            rng.uniform(
                -0.18,
                0.18,
                size=len(values)
            )
        )

        x_positions = (
            index
            +
            jitter
        )

        ax.scatter(
            x_positions,
            values,
            s=42,
            alpha=0.85,
            zorder=3
        )

        # library median
        median_value = (
            values.median()
        )

        ax.plot(
            [
                index - 0.24,
                index + 0.24
            ],
            [
                median_value,
                median_value
            ],
            linewidth=2.0,
            color="black",
            zorder=5
        )

        # expected replicate noise band
        if (
            show_noise
            and
            not pd.isna(noise)
        ):

            lower = (
                median_value
                -
                noise
            )

            upper = (
                median_value
                +
                noise
            )

            rectangle = Rectangle(
                (
                    index
                    -
                    band_width / 2,
                    lower
                ),
                band_width,
                2 * noise,
                facecolor="grey",
                alpha=0.18,
                edgecolor="none",
                zorder=1
            )

            ax.add_patch(
                rectangle
            )

    ax.set_xticks(
        range(
            len(libraries)
        )
    )

    ax.set_xticklabels(
        libraries,
        fontsize=10
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.set_axisbelow(
        True
    )


# ============================================================
# Robust figure saver
# ============================================================

def save_figure_robust(
    fig,
    outdir,
    temp_K,
    file_format,
    dpi=None
):

    temp_label = (
        f"{int(temp_K)}K"
    )

    extension = (
        file_format.lower()
    )

    normal_path = (
        outdir
        /
        f"library_comparison_{temp_label}.{extension}"
    )

    fallback_path = (
        outdir
        /
        f"cmp_{temp_label}.{extension}"
    )

    paths_to_try = [
        normal_path,
        fallback_path
    ]

    last_error = None

    for path in paths_to_try:

        print(
            f"Trying to save: {repr(str(path))}"
        )

        try:

            with open(
                path,
                "wb"
            ) as file:

                kwargs = {
                    "format":
                        file_format,

                    "bbox_inches":
                        "tight"
                }

                if dpi is not None:
                    kwargs["dpi"] = dpi

                fig.savefig(
                    file,
                    **kwargs
                )

            return path

        except OSError as exc:

            last_error = exc

            print(
                f"Could not save to "
                f"{repr(str(path))}: {exc}"
            )

    raise OSError(
        f"Could not save {file_format} figure. "
        f"Last error: {last_error}"
    )


# ============================================================
# Temperature figure
# ============================================================

def plot_temperature(
    combined,
    temp_K,
    libraries,
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

        print(
            f"\nNo data at {temp_K} K. "
            "Skipping."
        )

        return

    rng = (
        np.random.default_rng(
            12345
        )
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            10,
            18
        )
    )

    draw_metric(
        axes[0],
        data,
        libraries,
        metric=
            "mean_rg_nm",
        ylabel=
            "Mean Rg (nm)",
        title=
            "Mean radius of gyration",
        temp_K=
            temp_K,
        show_noise=
            True,
        rng=
            rng
    )

    draw_metric(
        axes[1],
        data,
        libraries,
        metric=
            "mean_rg_collapsed_nm",
        ylabel=
            "Collapsed Rg (nm)",
        title=
            "Mean Rg of collapsed ensemble",
        temp_K=
            temp_K,
        show_noise=
            False,
        rng=
            rng
    )

    draw_metric(
        axes[2],
        data,
        libraries,
        metric=
            "mean_rg_uncollapsed_nm",
        ylabel=
            "Uncollapsed Rg (nm)",
        title=
            "Mean Rg of uncollapsed ensemble",
        temp_K=
            temp_K,
        show_noise=
            False,
        rng=
            rng
    )

    draw_metric(
        axes[3],
        data,
        libraries,
        metric=
            "k_collapse_per_ns",
        ylabel=
            r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
        title=
            "Collapse rate",
        temp_K=
            temp_K,
        show_noise=
            True,
        rng=
            rng
    )

    draw_metric(
        axes[4],
        data,
        libraries,
        metric=
            "k_escape_per_ns",
        ylabel=
            r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
        title=
            "Escape rate",
        temp_K=
            temp_K,
        show_noise=
            True,
        rng=
            rng
    )

    fig.suptitle(
        (
            f"Architecture-dependent behaviour at {temp_K} K\n"
            "Dots = individual architectures; "
            "black line = library median"
        ),
        fontsize=14,
        y=0.995
    )

    fig.text(
        0.5,
        0.008,
        (
            "Grey regions show each library median ± 1 expected "
            "replicate SD estimated from the VGIPKG triplicate "
            "dataset. Grey bands are a reference for expected "
            "simulation variability, not confidence intervals "
            "or significance tests."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True
    )

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.97
        ]
    )

    saved_png = (
        save_figure_robust(
            fig=fig,
            outdir=outdir,
            temp_K=temp_K,
            file_format="png",
            dpi=300
        )
    )

    saved_pdf = (
        save_figure_robust(
            fig=fig,
            outdir=outdir,
            temp_K=temp_K,
            file_format="pdf"
        )
    )

    print(
        f"\nSaved {temp_K} K comparison:"
    )

    print(
        saved_png
    )

    print(
        saved_pdf
    )

    plt.close(
        fig
    )


# ============================================================
# Library-level architecture spread
# ============================================================

def calculate_library_summary(
    combined
):

    rows = []

    metrics = [
        "mean_rg_nm",
        "mean_rg_collapsed_nm",
        "mean_rg_uncollapsed_nm",
        "k_collapse_per_ns",
        "k_escape_per_ns"
    ]

    for (
        library,
        temp_K
    ), group in combined.groupby(
        [
            "library",
            "temp_K"
        ]
    ):

        row = {
            "library":
                library,

            "temp_K":
                temp_K,

            "n_architectures":
                group["id"].nunique()
        }

        for metric in metrics:

            values = (
                group[
                    metric
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

            row[
                f"{metric}_mean"
            ] = values.mean()

            row[
                f"{metric}_median"
            ] = values.median()

            row[
                f"{metric}_architecture_sd"
            ] = architecture_sd

            row[
                f"{metric}_min"
            ] = values.min()

            row[
                f"{metric}_max"
            ] = values.max()

            row[
                f"{metric}_range"
            ] = (
                values.max()
                -
                values.min()
            )

            noise = (
                get_noise(
                    temp_K,
                    metric
                )
            )

            row[
                f"{metric}_expected_replicate_sd"
            ] = noise

            if (
                not pd.isna(noise)
                and
                noise > 0
            ):

                row[
                    f"{metric}_signal_to_noise"
                ] = (
                    architecture_sd
                    /
                    noise
                )

            else:

                row[
                    f"{metric}_signal_to_noise"
                ] = np.nan

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare architecture-dependent Rg and kinetics "
            "across CALVADOS libraries."
        )
    )

    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset in format LABEL::DIRECTORY. "
            "Supply multiple times for multiple libraries."
        )
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

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
                label,
                folder
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
    # Save architecture-level data
    # --------------------------------------------------------

    combined_output = (
        outdir
        /
        "combined_architecture_results.csv"
    )

    combined.to_csv(
        combined_output,
        index=False
    )

    print(
        "\nSaved combined architecture table:"
    )

    print(
        combined_output
    )

    # --------------------------------------------------------
    # Library spread summary
    # --------------------------------------------------------

    library_summary = (
        calculate_library_summary(
            combined
        )
    )

    summary_output = (
        outdir
        /
        "library_architecture_spread_summary.csv"
    )

    library_summary.to_csv(
        summary_output,
        index=False
    )

    print(
        "\nSaved library spread summary:"
    )

    print(
        summary_output
    )

    # --------------------------------------------------------
    # Print architecture spread / noise
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "ARCHITECTURE SPREAD / REPLICATE NOISE"
    )

    print(
        "========================================"
    )

    for _, row in library_summary.iterrows():

        print(
            f"\n{row['library']} — "
            f"{int(row['temp_K'])} K"
        )

        for metric in [
            "mean_rg_nm",
            "k_collapse_per_ns",
            "k_escape_per_ns"
        ]:

            arch_sd = (
                row.get(
                    f"{metric}_architecture_sd",
                    np.nan
                )
            )

            rep_sd = (
                row.get(
                    f"{metric}_expected_replicate_sd",
                    np.nan
                )
            )

            signal = (
                row.get(
                    f"{metric}_signal_to_noise",
                    np.nan
                )
            )

            print(
                f"  {metric}:"
            )

            print(
                f"    architecture SD = "
                f"{arch_sd:.5f}"
            )

            print(
                f"    replicate SD    = "
                f"{rep_sd:.5f}"
            )

            print(
                f"    S = "
                f"{signal:.2f}"
            )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    temperatures = (
        sorted(
            combined[
                "temp_K"
            ]
            .dropna()
            .unique()
        )
    )

    for temp_K in temperatures:

        plot_temperature(
            combined=
                combined,

            temp_K=
                int(temp_K),

            libraries=
                library_order,

            outdir=
                outdir
        )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()