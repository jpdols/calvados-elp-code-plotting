from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

REP01 = Path(
    r"E:\Master thesis\calvados_tea\analysis_library_2"
    r"\VGIGIP\all_rg_values.csv"
)

REP02 = Path(
    r"E:\Master thesis\calvados_tea\analysis_library_2"
    r"\VGIGIP_rep02\all_rg_values.csv"
)

# Shorter output path to avoid Windows path-length problems
OUTDIR = Path(
    r"E:\Master thesis\calvados_tea\analysis_library_2"
    r"\VGIGIP_plots"
)

TEMP = "370K"

# Compact-state definition
THRESHOLD_NM = 4.61

# Only used if frame_dt_ns is absent
FALLBACK_FRAME_DT_NS = 0.01


# ============================================================
# VISUAL SETTINGS
# ============================================================

# Bubble sizes deliberately kept fairly small
MIN_SIZE = 25
MAX_SIZE = 180

# Focus compact-fraction colour gradient on useful range
FRACTION_CMAP_MIN = 0.40
FRACTION_CMAP_MAX = 0.80

# Larger figures to reduce overlap
LANDSCAPE_FIGSIZE = (14, 11)
SCATTER_FIGSIZE = (12, 10)
REPRO_FIGSIZE = (11, 10)
RANKED_FIGSIZE = (17, 8)

LABEL_SIZE = 8


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# GENERAL HELPERS
# ============================================================

def architecture_number(arch_id):
    """
    Extract leading architecture number.

    Example:
        884_VGIGIP_something -> 884
    """

    match = re.match(
        r"(\d+)",
        str(arch_id),
    )

    if match:
        return int(
            match.group(1)
        )

    return np.nan


def point_sizes(fraction):
    """
    Convert compact-state population into bubble size.

    Bubble size still carries information, but the range is
    intentionally modest to reduce overlap.
    """

    fraction = np.asarray(
        fraction,
        dtype=float,
    )

    fraction = np.clip(
        fraction,
        0,
        1,
    )

    return (
        MIN_SIZE
        + fraction
        * (MAX_SIZE - MIN_SIZE)
    )


def add_architecture_labels(
    ax,
    df,
    xcol,
    ycol,
):
    """
    Add architecture numbers to scatter plots.
    """

    for _, row in df.iterrows():

        if (
            pd.isna(row[xcol])
            or pd.isna(row[ycol])
        ):
            continue

        ax.annotate(
            str(
                int(
                    row["architecture_number"]
                )
            ),
            (
                row[xcol],
                row[ycol],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            alpha=0.85,
        )


# ============================================================
# LOAD RAW Rg DATA
# ============================================================

def load_raw(
    path,
    replicate,
):
    """
    Load frame-level Rg data.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find {replicate}:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "id",
        "temp",
        "rg_nm",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{replicate} is missing columns: "
            f"{missing}"
        )

    # Keep only requested temperature
    df = df[
        df["temp"].astype(str)
        == TEMP
    ].copy()

    df["replicate"] = replicate

    df["architecture_number"] = (
        df["id"].apply(
            architecture_number
        )
    )

    print()
    print(
        f"{replicate}: "
        f"{len(df):,} Rg frames"
    )

    print(
        f"{replicate}: "
        f"{df['id'].nunique()} architectures"
    )

    print(
        f"{replicate}: "
        f"temperature = {TEMP}"
    )

    return df


# ============================================================
# FRAME TIME
# ============================================================

def get_frame_dt(sub):
    """
    Determine time between analysed frames.

    Prefer frame_dt_ns stored in all_rg_values.csv.
    """

    if "frame_dt_ns" in sub.columns:

        values = (
            sub["frame_dt_ns"]
            .dropna()
            .unique()
        )

        if len(values) > 0:

            if len(values) > 1:
                print(
                    "WARNING: multiple frame_dt_ns "
                    "values found:",
                    values,
                )

            return float(
                values[0]
            )

    print(
        "WARNING: frame_dt_ns not found. "
        f"Using fallback "
        f"{FALLBACK_FRAME_DT_NS} ns."
    )

    return FALLBACK_FRAME_DT_NS


# ============================================================
# ANALYSE ONE ARCHITECTURE
# ============================================================

def analyse_architecture(sub):
    """
    Recalculate all state-dependent quantities directly
    from frame-level Rg values.

    Compact:
        Rg < THRESHOLD_NM

    k_collapse:
        N(expanded -> compact) / expanded time

    k_escape:
        N(compact -> expanded) / compact time
    """

    # ------------------------------------------
    # Chronological ordering
    # ------------------------------------------

    if "analysed_index" in sub.columns:

        sub = sub.sort_values(
            "analysed_index"
        )

    elif "original_frame" in sub.columns:

        sub = sub.sort_values(
            "original_frame"
        )

    # ------------------------------------------
    # Rg array
    # ------------------------------------------

    rg = (
        sub["rg_nm"]
        .to_numpy(
            dtype=float
        )
    )

    frame_dt_ns = get_frame_dt(
        sub
    )

    # ------------------------------------------
    # State assignment
    # ------------------------------------------

    collapsed = (
        rg < THRESHOLD_NM
    )

    expanded = ~collapsed

    # ------------------------------------------
    # Overall Rg statistics
    # ------------------------------------------

    mean_rg = float(
        np.mean(rg)
    )

    std_rg = float(
        np.std(rg)
    )

    median_rg = float(
        np.median(rg)
    )

    fraction_collapsed = float(
        np.mean(collapsed)
    )

    fraction_expanded = float(
        np.mean(expanded)
    )

    # ------------------------------------------
    # Compact-state Rg
    # ------------------------------------------

    if np.any(collapsed):

        collapsed_rg = (
            rg[collapsed]
        )

        mean_collapsed_rg = float(
            np.mean(
                collapsed_rg
            )
        )

        std_collapsed_rg = float(
            np.std(
                collapsed_rg
            )
        )

        median_collapsed_rg = float(
            np.median(
                collapsed_rg
            )
        )

    else:

        mean_collapsed_rg = np.nan
        std_collapsed_rg = np.nan
        median_collapsed_rg = np.nan

    # ------------------------------------------
    # Expanded-state Rg
    # ------------------------------------------

    if np.any(expanded):

        mean_expanded_rg = float(
            np.mean(
                rg[expanded]
            )
        )

    else:

        mean_expanded_rg = np.nan

    # ------------------------------------------
    # State transitions
    # ------------------------------------------

    n_collapse = int(
        np.sum(
            expanded[:-1]
            & collapsed[1:]
        )
    )

    n_escape = int(
        np.sum(
            collapsed[:-1]
            & expanded[1:]
        )
    )

    # ------------------------------------------
    # State residence times
    # ------------------------------------------

    total_collapsed_time = float(
        np.sum(collapsed)
        * frame_dt_ns
    )

    total_expanded_time = float(
        np.sum(expanded)
        * frame_dt_ns
    )

    # ------------------------------------------
    # Crossing rates
    # ------------------------------------------

    if total_expanded_time > 0:

        k_collapse = (
            n_collapse
            / total_expanded_time
        )

    else:

        k_collapse = np.nan

    if total_collapsed_time > 0:

        k_escape = (
            n_escape
            / total_collapsed_time
        )

    else:

        k_escape = np.nan

    # ------------------------------------------
    # Kinetic bias
    # ------------------------------------------

    if (
        np.isfinite(k_collapse)
        and np.isfinite(k_escape)
        and k_escape > 0
    ):

        kinetic_bias = (
            k_collapse
            / k_escape
        )

    else:

        kinetic_bias = np.nan

    # ------------------------------------------
    # Return
    # ------------------------------------------

    return {
        "n_frames":
            len(rg),

        "frame_dt_ns":
            frame_dt_ns,

        "threshold_nm":
            THRESHOLD_NM,

        "mean_rg_nm":
            mean_rg,

        "std_rg_nm":
            std_rg,

        "median_rg_nm":
            median_rg,

        "fraction_collapsed":
            fraction_collapsed,

        "fraction_expanded":
            fraction_expanded,

        "mean_collapsed_rg_nm":
            mean_collapsed_rg,

        "std_collapsed_rg_nm":
            std_collapsed_rg,

        "median_collapsed_rg_nm":
            median_collapsed_rg,

        "mean_expanded_rg_nm":
            mean_expanded_rg,

        "n_collapse_transitions":
            n_collapse,

        "n_escape_transitions":
            n_escape,

        "total_collapsed_time_ns":
            total_collapsed_time,

        "total_expanded_time_ns":
            total_expanded_time,

        "k_collapse_per_ns":
            k_collapse,

        "k_escape_per_ns":
            k_escape,

        "kinetic_bias":
            kinetic_bias,
    }


# ============================================================
# ANALYSE COMPLETE REPLICATE
# ============================================================

def analyse_replicate(df):
    """
    Analyse each architecture independently.
    """

    rows = []

    grouped = df.groupby(
        [
            "replicate",
            "architecture_number",
            "id",
        ]
    )

    for (
        replicate,
        arch_number,
        arch_id,
    ), sub in grouped:

        metrics = analyse_architecture(
            sub
        )

        row = {
            "replicate":
                replicate,

            "architecture_number":
                arch_number,

            "id":
                arch_id,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = result_df.sort_values(
        "architecture_number"
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# KINETIC LANDSCAPE
# ============================================================

def plot_landscape(
    df,
    title,
    outfile,
):
    """
    x = k_escape
    y = k_collapse

    Bubble size + colour = compact-state population.
    """

    valid = df[
        df["k_escape_per_ns"].notna()
        & df["k_collapse_per_ns"].notna()
    ].copy()

    fig, ax = plt.subplots(
        figsize=LANDSCAPE_FIGSIZE
    )

    sc = ax.scatter(
        valid[
            "k_escape_per_ns"
        ],
        valid[
            "k_collapse_per_ns"
        ],
        s=point_sizes(
            valid[
                "fraction_collapsed"
            ]
        ),
        c=valid[
            "fraction_collapsed"
        ],
        cmap="viridis",
        vmin=FRACTION_CMAP_MIN,
        vmax=FRACTION_CMAP_MAX,
        alpha=0.80,
        edgecolors="black",
        linewidths=0.45,
    )

    add_architecture_labels(
        ax,
        valid,
        "k_escape_per_ns",
        "k_collapse_per_ns",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Compact-state population"
    )

    ax.set_xlabel(
        r"$k_{\mathrm{escape}}$ "
        r"(ns$^{-1}$)"
    )

    ax.set_ylabel(
        r"$k_{\mathrm{collapse}}$ "
        r"(ns$^{-1}$)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.13,
        linewidth=0.8,
    )

    ax.margins(
        x=0.08,
        y=0.08,
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# FRACTION COMPACT vs KINETIC BIAS
# ============================================================

def plot_fraction_vs_kinetic_bias(
    df,
    title,
    outfile,
):
    """
    x = compact-state population
    y = k_collapse / k_escape
    colour = mean compact-state Rg
    """

    plot_df = df[
        np.isfinite(
            df["kinetic_bias"]
        )
        &
        df[
            "mean_collapsed_rg_nm"
        ].notna()
    ].copy()

    fig, ax = plt.subplots(
        figsize=SCATTER_FIGSIZE
    )

    sc = ax.scatter(
        plot_df[
            "fraction_collapsed"
        ],
        plot_df[
            "kinetic_bias"
        ],
        s=70,
        c=plot_df[
            "mean_collapsed_rg_nm"
        ],
        cmap="viridis_r",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.45,
    )

    add_architecture_labels(
        ax,
        plot_df,
        "fraction_collapsed",
        "kinetic_bias",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.1,
        color="black",
        alpha=0.45,
    )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=1.1,
        color="black",
        alpha=0.40,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r"Mean compact-state "
        r"$R_g$ (nm)"
    )

    ax.set_xlabel(
        "Compact-state population"
    )

    ax.set_ylabel(
        r"$k_{\mathrm{collapse}}/"
        r"k_{\mathrm{escape}}$"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.13,
        linewidth=0.8,
    )

    ax.margins(
        y=0.08
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# FRACTION COMPACT vs COMPACT-STATE Rg
# ============================================================

def plot_fraction_vs_collapsed_rg(
    df,
    title,
    outfile,
):
    """
    x = compact-state population
    y = mean compact-state Rg
    colour = kinetic bias
    """

    plot_df = df[
        df[
            "mean_collapsed_rg_nm"
        ].notna()
        &
        np.isfinite(
            df["kinetic_bias"]
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=SCATTER_FIGSIZE
    )

    sc = ax.scatter(
        plot_df[
            "fraction_collapsed"
        ],
        plot_df[
            "mean_collapsed_rg_nm"
        ],
        s=70,
        c=plot_df[
            "kinetic_bias"
        ],
        cmap="viridis",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.45,
    )

    add_architecture_labels(
        ax,
        plot_df,
        "fraction_collapsed",
        "mean_collapsed_rg_nm",
    )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=1.1,
        color="black",
        alpha=0.40,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r"$k_{\mathrm{collapse}}/"
        r"k_{\mathrm{escape}}$"
    )

    ax.set_xlabel(
        "Compact-state population"
    )

    ax.set_ylabel(
        r"Mean compact-state "
        r"$R_g$ (nm)"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.13,
        linewidth=0.8,
    )

    ax.margins(
        y=0.08
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# REPLICATE KINETIC COMPARISON
# ============================================================

def plot_replicate_comparison(
    merged,
    outfile,
):
    """
    Rep01 and rep02 positions in the kinetic landscape.
    Lines connect identical architectures.
    """

    fig, ax = plt.subplots(
        figsize=LANDSCAPE_FIGSIZE
    )

    # Connecting lines
    for _, row in merged.iterrows():

        values = [
            row[
                "k_escape_per_ns_rep01"
            ],
            row[
                "k_escape_per_ns_rep02"
            ],
            row[
                "k_collapse_per_ns_rep01"
            ],
            row[
                "k_collapse_per_ns_rep02"
            ],
        ]

        if any(
            pd.isna(v)
            for v in values
        ):
            continue

        ax.plot(
            [
                row[
                    "k_escape_per_ns_rep01"
                ],
                row[
                    "k_escape_per_ns_rep02"
                ],
            ],
            [
                row[
                    "k_collapse_per_ns_rep01"
                ],
                row[
                    "k_collapse_per_ns_rep02"
                ],
            ],
            color="0.75",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
        )

    # Replicate 01
    ax.scatter(
        merged[
            "k_escape_per_ns_rep01"
        ],
        merged[
            "k_collapse_per_ns_rep01"
        ],
        s=65,
        alpha=0.72,
        label="Replicate 01",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    # Replicate 02
    ax.scatter(
        merged[
            "k_escape_per_ns_rep02"
        ],
        merged[
            "k_collapse_per_ns_rep02"
        ],
        s=65,
        alpha=0.72,
        label="Replicate 02",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    # Architecture labels at midpoint
    for _, row in merged.iterrows():

        values = [
            row[
                "k_escape_per_ns_rep01"
            ],
            row[
                "k_escape_per_ns_rep02"
            ],
            row[
                "k_collapse_per_ns_rep01"
            ],
            row[
                "k_collapse_per_ns_rep02"
            ],
        ]

        if any(
            pd.isna(v)
            for v in values
        ):
            continue

        x = np.mean([
            row[
                "k_escape_per_ns_rep01"
            ],
            row[
                "k_escape_per_ns_rep02"
            ],
        ])

        y = np.mean([
            row[
                "k_collapse_per_ns_rep01"
            ],
            row[
                "k_collapse_per_ns_rep02"
            ],
        ])

        ax.annotate(
            str(
                int(
                    row[
                        "architecture_number"
                    ]
                )
            ),
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            alpha=0.82,
        )

    ax.set_xlabel(
        r"$k_{\mathrm{escape}}$ "
        r"(ns$^{-1}$)"
    )

    ax.set_ylabel(
        r"$k_{\mathrm{collapse}}$ "
        r"(ns$^{-1}$)"
    )

    ax.set_title(
        "VGIGIP kinetic landscape reproducibility"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.13
    )

    ax.margins(
        x=0.08,
        y=0.08,
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# COMPACT-STATE Rg REPRODUCIBILITY
# ============================================================

def plot_collapsed_rg_comparison(
    merged,
    outfile,
):
    """
    x = replicate 01 mean compact-state Rg
    y = replicate 02 mean compact-state Rg

    Colour and size = mean compact-state population.
    """

    plot_df = merged[
        merged[
            "mean_collapsed_rg_nm_rep01"
        ].notna()
        &
        merged[
            "mean_collapsed_rg_nm_rep02"
        ].notna()
    ].copy()

    mean_fraction = (
        plot_df[
            "fraction_collapsed_rep01"
        ]
        +
        plot_df[
            "fraction_collapsed_rep02"
        ]
    ) / 2

    fig, ax = plt.subplots(
        figsize=REPRO_FIGSIZE
    )

    sc = ax.scatter(
        plot_df[
            "mean_collapsed_rg_nm_rep01"
        ],
        plot_df[
            "mean_collapsed_rg_nm_rep02"
        ],
        s=point_sizes(
            mean_fraction
        ),
        c=mean_fraction,
        cmap="viridis",
        vmin=FRACTION_CMAP_MIN,
        vmax=FRACTION_CMAP_MAX,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.45,
    )

    # Identity line
    values = np.concatenate([
        plot_df[
            "mean_collapsed_rg_nm_rep01"
        ].to_numpy(),

        plot_df[
            "mean_collapsed_rg_nm_rep02"
        ].to_numpy(),
    ])

    low = values.min()
    high = values.max()

    margin = max(
        (high - low) * 0.10,
        0.02,
    )

    ax.plot(
        [
            low - margin,
            high + margin,
        ],
        [
            low - margin,
            high + margin,
        ],
        linestyle="--",
        color="black",
        linewidth=1.1,
        alpha=0.40,
    )

    # Architecture labels
    for _, row in plot_df.iterrows():

        ax.annotate(
            str(
                int(
                    row[
                        "architecture_number"
                    ]
                )
            ),
            (
                row[
                    "mean_collapsed_rg_nm_rep01"
                ],
                row[
                    "mean_collapsed_rg_nm_rep02"
                ],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=LABEL_SIZE,
            fontweight="bold",
            alpha=0.82,
        )

    ax.set_xlim(
        low - margin,
        high + margin,
    )

    ax.set_ylim(
        low - margin,
        high + margin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Mean compact-state population"
    )

    ax.set_xlabel(
        r"Mean compact-state $R_g$, "
        r"replicate 01 (nm)"
    )

    ax.set_ylabel(
        r"Mean compact-state $R_g$, "
        r"replicate 02 (nm)"
    )

    ax.set_title(
        "VGIGIP compact-state size reproducibility"
    )

    ax.grid(
        alpha=0.13
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# GENERIC SORTED METRIC PLOT
# ============================================================

def plot_sorted_metric(
    mean_df,
    metric,
    error_metric,
    ylabel,
    title,
    outfile,
    ascending=True,
    reference_line=None,
):
    """
    Ranked architecture plot.

    Point = mean across replicates.
    Error bar = SD between replicate-level values.
    """

    plot_df = mean_df[
        mean_df[
            metric
        ].notna()
    ].copy()

    plot_df = plot_df.sort_values(
        metric,
        ascending=ascending,
    ).reset_index(
        drop=True
    )

    x = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=RANKED_FIGSIZE
    )

    # Error bars
    if (
        error_metric is not None
        and error_metric in plot_df.columns
    ):

        errors = (
            plot_df[
                error_metric
            ]
            .fillna(0)
            .to_numpy()
        )

        ax.errorbar(
            x,
            plot_df[
                metric
            ],
            yerr=errors,
            fmt="none",
            ecolor="0.55",
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.75,
            zorder=1,
        )

    # Main points
    ax.scatter(
        x,
        plot_df[
            metric
        ],
        s=45,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    # Reference line
    if reference_line is not None:

        ax.axhline(
            reference_line,
            linestyle="--",
            linewidth=1.1,
            color="black",
            alpha=0.40,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_df[
            "architecture_number"
        ].astype(int),
        rotation=90,
        fontsize=9,
    )

    ax.set_xlabel(
        "Architecture"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.13,
    )

    ax.margins(
        x=0.02
    )

    fig.tight_layout()

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# SORTED COMPACT FRACTION
# ============================================================

def plot_sorted_fraction_collapsed(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "fraction_collapsed"
        ),

        error_metric=(
            "std_between_rep_fraction_collapsed"
        ),

        ylabel=(
            "Compact-state population"
        ),

        title=(
            "VGIGIP compact-state population"
        ),

        outfile=outfile,

        # Highest compact population first
        ascending=False,

        reference_line=0.5,
    )


# ============================================================
# SORTED MEAN Rg
# ============================================================

def plot_sorted_mean_rg(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "mean_rg_nm"
        ),

        error_metric=(
            "std_between_rep_mean_rg_nm"
        ),

        ylabel=(
            r"Mean $R_g$ (nm)"
        ),

        title=(
            r"VGIGIP mean $R_g$"
        ),

        outfile=outfile,

        # Smallest mean Rg first
        ascending=True,

        reference_line=(
            THRESHOLD_NM
        ),
    )


# ============================================================
# SORTED COMPACT-STATE Rg
# ============================================================

def plot_sorted_collapsed_rg(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "mean_collapsed_rg_nm"
        ),

        error_metric=(
            "std_between_rep_collapsed_rg_nm"
        ),

        ylabel=(
            r"Mean compact-state "
            r"$R_g$ (nm)"
        ),

        title=(
            r"VGIGIP mean compact-state $R_g$"
        ),

        outfile=outfile,

        # Smallest compact-state Rg first
        ascending=True,

        reference_line=None,
    )


# ============================================================
# SORTED k_collapse
# ============================================================

def plot_sorted_k_collapse(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "k_collapse_per_ns"
        ),

        error_metric=(
            "std_between_rep_k_collapse"
        ),

        ylabel=(
            r"$k_{\mathrm{collapse}}$ "
            r"(ns$^{-1}$)"
        ),

        title=(
            "VGIGIP compact-state entry rate"
        ),

        outfile=outfile,

        # Fastest compact-state entry first
        ascending=False,

        reference_line=None,
    )


# ============================================================
# SORTED k_escape
# ============================================================

def plot_sorted_k_escape(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "k_escape_per_ns"
        ),

        error_metric=(
            "std_between_rep_k_escape"
        ),

        ylabel=(
            r"$k_{\mathrm{escape}}$ "
            r"(ns$^{-1}$)"
        ),

        title=(
            "VGIGIP compact-state escape rate"
        ),

        outfile=outfile,

        # Slowest escape first
        ascending=True,

        reference_line=None,
    )


# ============================================================
# SORTED KINETIC BIAS
# ============================================================

def plot_sorted_kinetic_bias(
    mean_df,
    outfile,
):

    plot_sorted_metric(
        mean_df=mean_df,

        metric=(
            "kinetic_bias"
        ),

        error_metric=(
            "std_between_rep_kinetic_bias"
        ),

        ylabel=(
            r"$k_{\mathrm{collapse}}/"
            r"k_{\mathrm{escape}}$"
        ),

        title=(
            "VGIGIP kinetic bias"
        ),

        outfile=outfile,

        ascending=False,

        reference_line=1.0,
    )


# ============================================================
# MAIN
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "=" * 65
)

print(
    "VGIGIP replicate analysis"
)

print(
    f"Temperature: {TEMP}"
)

print(
    f"Compact-state threshold: "
    f"{THRESHOLD_NM:.2f} nm"
)

print(
    "State-dependent metrics are recalculated "
    "from all_rg_values.csv"
)

print(
    "=" * 65
)


# ============================================================
# LOAD RAW DATA
# ============================================================

raw01 = load_raw(
    REP01,
    "rep01",
)

raw02 = load_raw(
    REP02,
    "rep02",
)


# ============================================================
# RECALCULATE METRICS
# ============================================================

print()
print(
    "Recalculating replicate 01..."
)

rep01 = analyse_replicate(
    raw01
)

print(
    "Recalculating replicate 02..."
)

rep02 = analyse_replicate(
    raw02
)


# ============================================================
# SAVE INDIVIDUAL RESULTS
# ============================================================

rep01.to_csv(
    OUTDIR
    / "rep01_metrics.csv",
    index=False,
)

rep02.to_csv(
    OUTDIR
    / "rep02_metrics.csv",
    index=False,
)


# ============================================================
# CHECK ARCHITECTURES
# ============================================================

ids1 = set(
    rep01[
        "architecture_number"
    ].astype(int)
)

ids2 = set(
    rep02[
        "architecture_number"
    ].astype(int)
)

only_rep01 = sorted(
    ids1 - ids2
)

only_rep02 = sorted(
    ids2 - ids1
)

common = sorted(
    ids1 & ids2
)

print()
print(
    f"Rep01 architectures: {len(ids1)}"
)

print(
    f"Rep02 architectures: {len(ids2)}"
)

print(
    f"Common architectures: {len(common)}"
)

if only_rep01:

    print(
        "Only in rep01:",
        only_rep01,
    )

if only_rep02:

    print(
        "Only in rep02:",
        only_rep02,
    )


# ============================================================
# INDIVIDUAL KINETIC LANDSCAPES
# ============================================================

plot_landscape(
    rep01,
    "VGIGIP kinetic landscape — replicate 01",
    OUTDIR
    / "01_kinetic_rep01.png",
)

plot_landscape(
    rep02,
    "VGIGIP kinetic landscape — replicate 02",
    OUTDIR
    / "02_kinetic_rep02.png",
)


# ============================================================
# INDIVIDUAL FRACTION vs KINETIC BIAS
# ============================================================

plot_fraction_vs_kinetic_bias(
    rep01,
    "VGIGIP compact-state population vs kinetic bias — replicate 01",
    OUTDIR
    / "03_fraction_bias_rep01.png",
)

plot_fraction_vs_kinetic_bias(
    rep02,
    "VGIGIP compact-state population vs kinetic bias — replicate 02",
    OUTDIR
    / "04_fraction_bias_rep02.png",
)


# ============================================================
# MERGE REPLICATES
# ============================================================

merge_columns = [
    "architecture_number",
    "id",
    "mean_rg_nm",
    "std_rg_nm",
    "median_rg_nm",
    "fraction_collapsed",
    "mean_collapsed_rg_nm",
    "std_collapsed_rg_nm",
    "median_collapsed_rg_nm",
    "mean_expanded_rg_nm",
    "n_collapse_transitions",
    "n_escape_transitions",
    "k_collapse_per_ns",
    "k_escape_per_ns",
    "kinetic_bias",
]

merged = rep01[
    merge_columns
].merge(
    rep02[
        merge_columns
    ],
    on="architecture_number",
    suffixes=(
        "_rep01",
        "_rep02",
    ),
)


# ============================================================
# SAVE MERGED DATA
# ============================================================

merged.to_csv(
    OUTDIR
    / "replicate_comparison.csv",
    index=False,
)


# ============================================================
# REPLICATE KINETIC COMPARISON
# ============================================================

plot_replicate_comparison(
    merged,
    OUTDIR
    / "05_kinetic_replicates.png",
)


# ============================================================
# COMPACT-STATE Rg REPRODUCIBILITY
# ============================================================

plot_collapsed_rg_comparison(
    merged,
    OUTDIR
    / "06_compact_rg_replicates.png",
)


# ============================================================
# COMBINE REPLICATES
# ============================================================

combined = pd.concat(
    [
        rep01,
        rep02,
    ],
    ignore_index=True,
)


# ============================================================
# MEAN ACROSS REPLICATES
# ============================================================

mean_df = (
    combined
    .groupby(
        "architecture_number",
        as_index=False,
    )
    .agg(

        mean_rg_nm=(
            "mean_rg_nm",
            "mean",
        ),

        std_between_rep_mean_rg_nm=(
            "mean_rg_nm",
            "std",
        ),

        fraction_collapsed=(
            "fraction_collapsed",
            "mean",
        ),

        std_between_rep_fraction_collapsed=(
            "fraction_collapsed",
            "std",
        ),

        mean_collapsed_rg_nm=(
            "mean_collapsed_rg_nm",
            "mean",
        ),

        std_between_rep_collapsed_rg_nm=(
            "mean_collapsed_rg_nm",
            "std",
        ),

        k_collapse_per_ns=(
            "k_collapse_per_ns",
            "mean",
        ),

        std_between_rep_k_collapse=(
            "k_collapse_per_ns",
            "std",
        ),

        k_escape_per_ns=(
            "k_escape_per_ns",
            "mean",
        ),

        std_between_rep_k_escape=(
            "k_escape_per_ns",
            "std",
        ),

        kinetic_bias=(
            "kinetic_bias",
            "mean",
        ),

        std_between_rep_kinetic_bias=(
            "kinetic_bias",
            "std",
        ),

        n_replicates=(
            "replicate",
            "nunique",
        ),
    )
)


# ============================================================
# ALSO CALCULATE RATIO OF MEAN RATES
# ============================================================

mean_df[
    "kinetic_bias_from_mean_rates"
] = (
    mean_df[
        "k_collapse_per_ns"
    ]
    /
    mean_df[
        "k_escape_per_ns"
    ]
)


# ============================================================
# SAVE MEAN RESULTS
# ============================================================

mean_df.to_csv(
    OUTDIR
    / "mean_metrics.csv",
    index=False,
)


# ============================================================
# MEAN KINETIC LANDSCAPE
# ============================================================

plot_landscape(
    mean_df,
    "VGIGIP mean kinetic landscape",
    OUTDIR
    / "07_kinetic_mean.png",
)


# ============================================================
# MEAN FRACTION vs KINETIC BIAS
# ============================================================

plot_fraction_vs_kinetic_bias(
    mean_df,
    "VGIGIP compact-state population vs kinetic bias",
    OUTDIR
    / "08_fraction_vs_bias.png",
)


# ============================================================
# FRACTION vs COMPACT-STATE Rg
# ============================================================

plot_fraction_vs_collapsed_rg(
    mean_df,
    "VGIGIP compact-state population vs compact-state size",
    OUTDIR
    / "09_fraction_vs_compact_rg.png",
)


# ============================================================
# SIMPLE SORTED PLOTS
# ============================================================

plot_sorted_fraction_collapsed(
    mean_df,
    OUTDIR
    / "10_sorted_fraction.png",
)

plot_sorted_mean_rg(
    mean_df,
    OUTDIR
    / "11_sorted_mean_rg.png",
)

plot_sorted_collapsed_rg(
    mean_df,
    OUTDIR
    / "12_sorted_compact_rg.png",
)

plot_sorted_k_collapse(
    mean_df,
    OUTDIR
    / "13_sorted_kcollapse.png",
)

plot_sorted_k_escape(
    mean_df,
    OUTDIR
    / "14_sorted_kescape.png",
)

plot_sorted_kinetic_bias(
    mean_df,
    OUTDIR
    / "15_sorted_bias.png",
)


# ============================================================
# FINISHED
# ============================================================

print()
print(
    "=" * 65
)

print(
    "DONE"
)

print(
    "=" * 65
)

print(
    "All state-dependent metrics were recalculated "
    "directly from raw Rg values."
)

print(
    f"Compact state: "
    f"Rg < {THRESHOLD_NM:.2f} nm"
)

print()
print(
    "Output directory:"
)

print(
    OUTDIR
)

print()
print(
    "Files:"
)

print(
    "  rep01_metrics.csv"
)

print(
    "  rep02_metrics.csv"
)

print(
    "  replicate_comparison.csv"
)

print(
    "  mean_metrics.csv"
)

print()
print(
    "Plots:"
)

print(
    "  01 = kinetic landscape rep01"
)

print(
    "  02 = kinetic landscape rep02"
)

print(
    "  03 = fraction vs kinetic bias rep01"
)

print(
    "  04 = fraction vs kinetic bias rep02"
)

print(
    "  05 = replicate kinetic comparison"
)

print(
    "  06 = compact-state Rg reproducibility"
)

print(
    "  07 = mean kinetic landscape"
)

print(
    "  08 = fraction vs kinetic bias"
)

print(
    "  09 = fraction vs compact-state Rg"
)

print(
    "  10 = sorted compact-state population"
)

print(
    "  11 = sorted mean Rg"
)

print(
    "  12 = sorted compact-state Rg"
)

print(
    "  13 = sorted k_collapse"
)

print(
    "  14 = sorted k_escape"
)

print(
    "  15 = sorted kinetic bias"
)