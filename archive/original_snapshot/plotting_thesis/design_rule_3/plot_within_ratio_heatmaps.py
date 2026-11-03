"""
plot_within_ratio_heatmaps_column_scaled.py

For every metric and ratio R1-R6:

    rows    = 8 sequence architectures
    columns = 4 motif-pair chemistries
              VGIP, VGIG, VAIP, VAIG

CELL TEXT:
    Actual physical metric value.

CELL COLOR:
    Independently normalized WITHIN EACH CHEMISTRY COLUMN
    across the eight architectures.

For chemistry f:

    C(a,f) =
        [X(a,f) - min_a X(a,f)]
        /
        [max_a X(a,f) - min_a X(a,f)]

Therefore within every chemistry column:

    0 = lowest architecture value
    1 = highest architecture value

Scientific question:

    "At fixed motif composition and chemistry, how does
     architecture affect the property?"

Then, by looking horizontally across chemistry columns:

    "Does the relative effect of a given architecture
     persist when motif chemistry changes?"

IMPORTANT:
    The physical values themselves are NEVER normalized.
    Normalization is used ONLY for cell colouring.

Outputs:
    - one heatmap per ratio × metric
    - one 6-ratio overview per metric
    - architecture ranges within each ratio × chemistry
    - architecture consistency table across chemistries
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

RATIOS = [
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
]

RATIO_LABELS = {
    "R1": "R1 — 14%",
    "R2": "R2 — 25%",
    "R3": "R3 — 33%",
    "R4": "R4 — 50%",
    "R5": "R5 — 67%",
    "R6": "R6 — 83%",
}


FAMILY_ORDER = [
    "VGIP",
    "VGIG",
    "VAIP",
    "VAIG",
]

FAMILY_LABELS = {
    "VGIP": "VPGVG\n+\nIPAVG",
    "VGIG": "VPGVG\n+\nVPGIG",
    "VAIP": "VPAVG\n+\nIPAVG",
    "VAIG": "VPAVG\n+\nVPGIG",
}


ARCH_ORDER = [
    "even",
    "small blocks",
    "medium blocks",
    "large blocks",
    "higher-lambda center",
    "higher-lambda endcaps",
    "diblock",
    "random shuffle",
]


# ============================================================
# METRICS
# ============================================================

METRICS = {

    # --------------------------------------------------------
    # Global conformational behaviour
    # --------------------------------------------------------

    "mean_rg_nm": (
        r"Mean $R_g$",
        r"$R_g$ (nm)",
        ".2f",
    ),

    "rg_range90_nm": (
        r"Central 90% $R_g$ range",
        r"$R_{g,95}-R_{g,5}$ (nm)",
        ".2f",
    ),

    "compact_fraction": (
        "Compact-state population",
        "Compact fraction",
        ".2f",
    ),

    # --------------------------------------------------------
    # Compact-state structure
    # --------------------------------------------------------

    "mean_rg_collapsed_nm": (
        r"Collapsed-state mean $R_g$",
        r"$R_g$ (nm)",
        ".2f",
    ),

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    "k_collapse_per_ns": (
        r"Collapse rate $k_{\mathrm{collapse}}$",
        r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
        ".2f",
    ),

    "k_escape_per_ns": (
        r"Escape rate $k_{\mathrm{escape}}$",
        r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
        ".2f",
    ),

    "mean_collapsed_dwell_ns": (
        "Mean collapsed-state dwell time",
        "Time (ns)",
        ".2f",
    ),

    "mean_expanded_dwell_ns": (
        "Mean expanded-state dwell time",
        "Time (ns)",
        ".2f",
    ),
}


# ============================================================
# ARCHITECTURE NAME NORMALIZATION
# ============================================================

def canonical_architecture_name(name):
    """
    Convert architecture labels from the CSV to the eight
    canonical architecture classes.
    """

    if pd.isna(name):
        return np.nan

    s = str(name).lower().strip()

    s = s.replace("_", " ")
    s = s.replace("-", " ")

    if "random" in s:
        return "random shuffle"

    if "diblock" in s:
        return "diblock"

    if "small" in s:
        return "small blocks"

    if "medium" in s:
        return "medium blocks"

    if "large" in s:
        return "large blocks"

    if (
        "even" in s
        or "alternating" in s
    ):
        return "even"

    if (
        "center" in s
        or "centre" in s
    ):
        return "higher-lambda center"

    if "endcap" in s:
        return "higher-lambda endcaps"

    return str(name)


def get_architecture_column(df):
    """
    Find architecture label column.
    """

    if "architecture_class" in df.columns:
        return "architecture_class"

    if "architecture_type" in df.columns:
        return "architecture_type"

    raise ValueError(
        "Could not find architecture_class "
        "or architecture_type column."
    )


# ============================================================
# COLUMN-WISE NORMALIZATION
# ============================================================

def normalize_columns(values):
    """
    Normalize each CHEMISTRY COLUMN independently across
    the eight architectures.

    Matrix:
        rows    = architectures
        columns = chemistries

    Within every chemistry column:

        lowest architecture  -> 0
        highest architecture -> 1

    Actual physical values are unchanged.
    This is ONLY used for colouring.

    If all architecture values in one chemistry are equal,
    all cells in that column are assigned 0.5.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    normalized = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    # Loop over chemistry columns
    for j in range(
        values.shape[1]
    ):

        column = values[:, j]

        finite_mask = np.isfinite(
            column
        )

        finite_values = column[
            finite_mask
        ]

        if finite_values.size == 0:
            continue

        col_min = np.min(
            finite_values
        )

        col_max = np.max(
            finite_values
        )

        # All architectures essentially identical
        if np.isclose(
            col_min,
            col_max,
        ):

            normalized[
                finite_mask,
                j
            ] = 0.5

        else:

            normalized[
                finite_mask,
                j
            ] = (
                finite_values
                - col_min
            ) / (
                col_max
                - col_min
            )

    return normalized


# ============================================================
# BUILD RAW VALUE MATRIX
# ============================================================

def build_matrix(
    df,
    ratio,
    metric,
):
    """
    Return:

        8 architectures × 4 chemistries

    for one ratio and one metric.
    """

    sub = df[
        df["ratio"] == ratio
    ].copy()

    if sub.empty:
        return None

    matrix = (
        sub.pivot_table(
            index="architecture_plot",
            columns="family",
            values=metric,
            aggfunc="mean",
        )
        .reindex(
            index=ARCH_ORDER,
            columns=FAMILY_ORDER,
        )
    )

    return matrix


# ============================================================
# TEXT COLOR
# ============================================================

def choose_text_color(
    normalized_value
):
    """
    Improve readability on viridis.
    """

    if not np.isfinite(
        normalized_value
    ):
        return "black"

    # Dark purple / blue background
    if normalized_value < 0.35:
        return "white"

    return "black"


# ============================================================
# SINGLE-RATIO HEATMAP
# ============================================================

def make_heatmap(
    df,
    metric,
    metric_title,
    physical_units,
    number_format,
    ratio,
    outdir,
):

    matrix = build_matrix(
        df,
        ratio,
        metric,
    )

    if matrix is None:
        return

    raw_values = matrix.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        raw_values
    ).any():

        print(
            f"WARNING: no finite values "
            f"for {ratio} / {metric}"
        )

        return

    # ========================================================
    # THIS IS THE IMPORTANT STEP
    # ========================================================

    normalized_values = (
        normalize_columns(
            raw_values
        )
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8.8, 7.2)
    )

    masked = np.ma.masked_invalid(
        normalized_values
    )

    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            len(FAMILY_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            FAMILY_LABELS[f]
            for f in FAMILY_ORDER
        ],
        fontsize=11,
    )

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    ax.set_yticks(
        np.arange(
            len(ARCH_ORDER)
        )
    )

    ax.set_yticklabels(
        ARCH_ORDER,
        fontsize=11,
    )

    ax.set_xlabel(
        "Motif-pair chemistry",
        fontsize=12,
    )

    ax.set_ylabel(
        "Sequence architecture",
        fontsize=12,
    )

    ratio_title = RATIO_LABELS.get(
        ratio,
        ratio,
    )

    ax.set_title(
        f"{metric_title}\n"
        f"{ratio_title} higher-interaction motif — 370 K",
        fontsize=14,
        pad=14,
    )

    # --------------------------------------------------------
    # ACTUAL physical values inside cells
    # --------------------------------------------------------

    for i in range(
        raw_values.shape[0]
    ):

        for j in range(
            raw_values.shape[1]
        ):

            raw = raw_values[
                i,
                j
            ]

            norm = normalized_values[
                i,
                j
            ]

            if np.isfinite(raw):

                text = format(
                    raw,
                    number_format,
                )

                text_color = (
                    choose_text_color(
                        norm
                    )
                )

            else:

                text = "NA"
                text_color = "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    # --------------------------------------------------------
    # Cell grid
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            len(FAMILY_ORDER),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(ARCH_ORDER),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_ticks(
        [0, 0.5, 1]
    )

    cbar.set_ticklabels(
        [
            "Lowest",
            "Middle",
            "Highest",
        ]
    )

    cbar.set_label(
        "Value relative to other architectures\n"
        "within the same chemistry",
        fontsize=10,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    ax.text(
        0.5,
        -0.13,
        (
            f"Cell text = actual {physical_units}.  "
            "Colour = architecture comparison within each chemistry "
            "(each column scaled independently)."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metric_dir = (
        outdir
        / metric
    )

    metric_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{ratio}_{metric}_"
        "column_scaled"
    )

    fig.savefig(
        metric_dir
        / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        metric_dir
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SIX-RATIO OVERVIEW
# ============================================================

def make_six_ratio_overview(
    df,
    metric,
    metric_title,
    physical_units,
    number_format,
    outdir,
):
    """
    2 × 3 overview containing R1-R6.

    IMPORTANT:

    Every chemistry column in every ratio panel is
    independently normalized across the eight architectures.
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 11),
        constrained_layout=True,
    )

    axes = axes.flatten()

    last_im = None

    for ax, ratio in zip(
        axes,
        RATIOS,
    ):

        matrix = build_matrix(
            df,
            ratio,
            metric,
        )

        if matrix is None:

            ax.set_visible(
                False
            )

            continue

        raw_values = (
            matrix.to_numpy(
                dtype=float
            )
        )

        normalized_values = (
            normalize_columns(
                raw_values
            )
        )

        masked = np.ma.masked_invalid(
            normalized_values
        )

        im = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        last_im = im

        # ----------------------------------------------------
        # X axis
        # ----------------------------------------------------

        ax.set_xticks(
            np.arange(
                len(FAMILY_ORDER)
            )
        )

        ax.set_xticklabels(
            FAMILY_ORDER,
            fontsize=9,
        )

        # ----------------------------------------------------
        # Y axis
        # ----------------------------------------------------

        ax.set_yticks(
            np.arange(
                len(ARCH_ORDER)
            )
        )

        # Architecture labels only on left-hand panels
        if ax in [
            axes[0],
            axes[3],
        ]:

            ax.set_yticklabels(
                ARCH_ORDER,
                fontsize=8.5,
            )

        else:

            ax.set_yticklabels(
                []
            )

        # ----------------------------------------------------
        # Panel title
        # ----------------------------------------------------

        ax.set_title(
            RATIO_LABELS.get(
                ratio,
                ratio,
            ),
            fontsize=12,
        )

        # ----------------------------------------------------
        # Actual physical values
        # ----------------------------------------------------

        for i in range(
            raw_values.shape[0]
        ):

            for j in range(
                raw_values.shape[1]
            ):

                raw = raw_values[
                    i,
                    j
                ]

                norm = normalized_values[
                    i,
                    j
                ]

                if np.isfinite(raw):

                    text = format(
                        raw,
                        number_format,
                    )

                    text_color = (
                        choose_text_color(
                            norm
                        )
                    )

                else:

                    text = "NA"
                    text_color = "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=text_color,
                )

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.set_xticks(
            np.arange(
                -0.5,
                len(FAMILY_ORDER),
                1,
            ),
            minor=True,
        )

        ax.set_yticks(
            np.arange(
                -0.5,
                len(ARCH_ORDER),
                1,
            ),
            minor=True,
        )

        ax.grid(
            which="minor",
            color="white",
            linewidth=1,
        )

        ax.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"{metric_title} — architecture effects "
        f"within fixed chemistry — 370 K",
        fontsize=17,
    )

    fig.supxlabel(
        "Motif-pair chemistry",
        fontsize=12,
    )

    fig.supylabel(
        "Sequence architecture",
        fontsize=12,
    )

    # --------------------------------------------------------
    # Shared colorbar
    # --------------------------------------------------------

    if last_im is not None:

        cbar = fig.colorbar(
            last_im,
            ax=axes.tolist(),
            fraction=0.025,
            pad=0.02,
        )

        cbar.set_ticks(
            [0, 0.5, 1]
        )

        cbar.set_ticklabels(
            [
                "Lowest",
                "Middle",
                "Highest",
            ]
        )

        cbar.set_label(
            "Value relative to other architectures\n"
            "within the same chemistry",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Figure note
    # --------------------------------------------------------

    fig.text(
        0.5,
        -0.015,
        (
            f"Numbers are actual {physical_units}; "
            "colour is independently normalized across the eight "
            "architectures within each chemistry column."
        ),
        ha="center",
        fontsize=10,
        color="0.35",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    overview_dir = (
        outdir
        / "six_ratio_overviews"
    )

    overview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        overview_dir
        /
        f"{metric}_all_ratios_column_scaled.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        overview_dir
        /
        f"{metric}_all_ratios_column_scaled.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ARCHITECTURE RANGE WITHIN FIXED CHEMISTRY
# ============================================================

def calculate_architecture_ranges(
    df,
    metrics,
):
    """
    For each:

        ratio × chemistry × metric

    calculate how strongly architecture changes the property.

    architecture_range =
        max across architectures
        -
        min across architectures

    This directly quantifies the magnitude of the
    architecture effect at fixed composition and chemistry.
    """

    rows = []

    for ratio in RATIOS:

        for family in FAMILY_ORDER:

            sub = df[
                (
                    df["ratio"]
                    ==
                    ratio
                )
                &
                (
                    df["family"]
                    ==
                    family
                )
            ]

            for metric in metrics:

                if metric not in sub.columns:
                    continue

                values = (
                    pd.to_numeric(
                        sub[metric],
                        errors="coerce",
                    )
                    .dropna()
                )

                if len(values) == 0:
                    continue

                minimum = (
                    values.min()
                )

                maximum = (
                    values.max()
                )

                rows.append(
                    {
                        "ratio":
                            ratio,

                        "family":
                            family,

                        "metric":
                            metric,

                        "n_architectures":
                            len(values),

                        "minimum":
                            minimum,

                        "maximum":
                            maximum,

                        "architecture_range":
                            maximum
                            -
                            minimum,

                        "architecture_mean":
                            values.mean(),

                        "architecture_std":
                            values.std(),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ARCHITECTURE RELATIVE POSITION ACROSS CHEMISTRIES
# ============================================================

def calculate_architecture_consistency(
    df,
    metrics,
):
    """
    Calculate the same column-normalized values used for
    colouring and save them numerically.

    For each:
        ratio × metric × architecture × chemistry

    store:
        raw value
        relative architecture value (0-1)

    Then also calculate, for every architecture:

        mean_relative_value
        std_relative_value
        relative_range

    across the four chemistries.

    Interpretation:

    mean near 0:
        architecture tends to give LOW values across chemistry

    mean near 1:
        architecture tends to give HIGH values across chemistry

    low std:
        this behaviour is consistent across chemistry

    high std:
        architecture effect depends strongly on chemistry
    """

    detailed_rows = []
    summary_rows = []

    for ratio in RATIOS:

        for metric in metrics:

            matrix = build_matrix(
                df,
                ratio,
                metric,
            )

            if matrix is None:
                continue

            raw = matrix.to_numpy(
                dtype=float
            )

            normalized = (
                normalize_columns(
                    raw
                )
            )

            # -----------------------------------------------
            # Detailed table
            # -----------------------------------------------

            for i, architecture in enumerate(
                ARCH_ORDER
            ):

                for j, family in enumerate(
                    FAMILY_ORDER
                ):

                    detailed_rows.append(
                        {
                            "ratio":
                                ratio,

                            "metric":
                                metric,

                            "architecture":
                                architecture,

                            "family":
                                family,

                            "raw_value":
                                raw[i, j],

                            "relative_architecture_value":
                                normalized[i, j],
                        }
                    )

            # -----------------------------------------------
            # Summary across four chemistries
            # -----------------------------------------------

            for i, architecture in enumerate(
                ARCH_ORDER
            ):

                values = (
                    normalized[
                        i,
                        :
                    ]
                )

                finite = values[
                    np.isfinite(
                        values
                    )
                ]

                if len(finite) == 0:
                    continue

                summary_rows.append(
                    {
                        "ratio":
                            ratio,

                        "metric":
                            metric,

                        "architecture":
                            architecture,

                        "n_chemistries":
                            len(finite),

                        "mean_relative_value":
                            np.mean(
                                finite
                            ),

                        "std_relative_value":
                            np.std(
                                finite,
                                ddof=1,
                            )
                            if len(finite) > 1
                            else np.nan,

                        "min_relative_value":
                            np.min(
                                finite
                            ),

                        "max_relative_value":
                            np.max(
                                finite
                            ),

                        "relative_range":
                            np.max(
                                finite
                            )
                            -
                            np.min(
                                finite
                            ),
                    }
                )

    return (
        pd.DataFrame(
            detailed_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to "
            "architecture_metrics_370K.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 74
    )

    print(
        "FIXED-COMPOSITION ARCHITECTURE HEATMAPS"
    )

    print(
        "Colour normalization: within chemistry column"
    )

    print(
        "=" * 74
    )

    print(
        f"Input:  {input_path}"
    )

    print(
        f"Output: {outdir}"
    )

    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        input_path
    )

    print(
        f"Loaded {len(df)} rows"
    )

    print(
        f"Found {len(df.columns)} columns"
    )

    # ========================================================
    # FILTER 370 K
    # ========================================================

    if "temperature" in df.columns:

        temp = (
            df["temperature"]
            .astype(str)
            .str.upper()
        )

        df = df[
            temp.str.contains(
                "370"
            )
        ].copy()

    elif "temp" in df.columns:

        temp = (
            df["temp"]
            .astype(str)
            .str.upper()
        )

        df = df[
            temp.str.contains(
                "370"
            )
        ].copy()

    print(
        f"370 K rows: {len(df)}"
    )

    # ========================================================
    # ARCHITECTURE LABELS
    # ========================================================

    architecture_column = (
        get_architecture_column(
            df
        )
    )

    print(
        "Architecture source column: "
        f"{architecture_column}"
    )

    df[
        "architecture_plot"
    ] = (
        df[
            architecture_column
        ]
        .apply(
            canonical_architecture_name
        )
    )

    print()
    print(
        "Architecture classes:"
    )

    for architecture in ARCH_ORDER:

        count = (
            df[
                "architecture_plot"
            ]
            ==
            architecture
        ).sum()

        print(
            f"  {architecture:<28}"
            f"{count}"
        )

    # ========================================================
    # CHECK DESIGN
    # ========================================================

    print()
    print(
        "Counts per ratio × family:"
    )

    counts = (
        df.groupby(
            [
                "ratio",
                "family",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=RATIOS,
            columns=FAMILY_ORDER,
        )
    )

    print(
        counts
    )

    # ========================================================
    # AVAILABLE METRICS
    # ========================================================

    available_metrics = {}

    print()
    print(
        "Available metrics:"
    )

    for (
        metric,
        settings,
    ) in METRICS.items():

        if metric in df.columns:

            available_metrics[
                metric
            ] = settings

            print(
                f"  YES  {metric}"
            )

        else:

            print(
                f"  ---  {metric} "
                "not found"
            )

    if not available_metrics:

        raise ValueError(
            "None of the requested "
            "metric columns were found."
        )

    # ========================================================
    # MAKE HEATMAPS
    # ========================================================

    for (
        metric,
        settings,
    ) in available_metrics.items():

        (
            title,
            units,
            number_format,
        ) = settings

        print()
        print(
            "-" * 74
        )

        print(
            metric
        )

        # ----------------------------------------------------
        # Individual ratio plots
        # ----------------------------------------------------

        for ratio in RATIOS:

            print(
                f"  {ratio}"
            )

            make_heatmap(
                df=df,
                metric=metric,
                metric_title=title,
                physical_units=units,
                number_format=number_format,
                ratio=ratio,
                outdir=outdir,
            )

        # ----------------------------------------------------
        # 6-ratio overview
        # ----------------------------------------------------

        make_six_ratio_overview(
            df=df,
            metric=metric,
            metric_title=title,
            physical_units=units,
            number_format=number_format,
            outdir=outdir,
        )

    # ========================================================
    # ARCHITECTURE RANGE TABLE
    # ========================================================

    range_df = (
        calculate_architecture_ranges(
            df,
            list(
                available_metrics.keys()
            ),
        )
    )

    range_file = (
        outdir
        /
        "architecture_ranges_within_fixed_chemistry_370K.csv"
    )

    range_df.to_csv(
        range_file,
        index=False,
    )

    # ========================================================
    # CONSISTENCY TABLES
    # ========================================================

    (
        consistency_detailed,
        consistency_summary,
    ) = (
        calculate_architecture_consistency(
            df,
            list(
                available_metrics.keys()
            ),
        )
    )

    detailed_file = (
        outdir
        /
        "architecture_relative_values_across_chemistries_370K.csv"
    )

    summary_file = (
        outdir
        /
        "architecture_consistency_across_chemistries_370K.csv"
    )

    consistency_detailed.to_csv(
        detailed_file,
        index=False,
    )

    consistency_summary.to_csv(
        summary_file,
        index=False,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print(
        "=" * 74
    )

    print(
        "DONE"
    )

    print(
        "=" * 74
    )

    print(
        f"Metrics plotted: "
        f"{len(available_metrics)}"
    )

    print(
        f"Individual heatmaps: "
        f"{len(available_metrics) * 6}"
    )

    print(
        f"Six-ratio overview plots: "
        f"{len(available_metrics)}"
    )

    print()
    print(
        "Architecture ranges:"
    )

    print(
        f"  {range_file}"
    )

    print()
    print(
        "Detailed relative architecture values:"
    )

    print(
        f"  {detailed_file}"
    )

    print()
    print(
        "Architecture consistency across chemistry:"
    )

    print(
        f"  {summary_file}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  Vertical comparison = architecture effect "
        "within fixed chemistry."
    )

    print(
        "  Horizontal comparison = whether the same "
        "architecture effect persists across chemistry."
    )


if __name__ == "__main__":
    main()