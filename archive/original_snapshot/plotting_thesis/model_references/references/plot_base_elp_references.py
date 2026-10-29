"""
plot_base_elp_references.py

Analyse base/reference ELP simulations at 300 K and 370 K.

Recursively searches:
    E:\\Master thesis\\calvados_tea\\runs\\scale1\\references

Examples:
    references\\VPGEG\\370K\\rep03\\VPGEG_90\\VPGEG_90.dcd
    references\\VPGFG\\300K\\rep01\\VPGFG_90\\VPGFG_90.dcd

Anything containing "aladini" in its path is excluded.

IMPORTANT:
MDAnalysis reports distances in Angstrom for these trajectories.
All Rg values are therefore divided by 10 to obtain nm.

Outputs:
    base_elp_reference_replicates.csv
    base_elp_reference_summary.csv
    base_elp_reference_response.pdf
    base_elp_reference_response.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
from matplotlib import colormaps


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(
    r"E:\Master thesis\calvados_tea\runs\scale1\references"
)

OUTDIR = ROOT / "base_elp_reference_analysis"

TEMPERATURES = [300, 370]

# Hofmann polymer-scaling parameters, in nm
LP_STAR = 0.40
BACKBONE_B = 0.38

# Analyse every frame
STRIDE = 1

# Excluded paths
EXCLUDE = [
    "aladini",
]

DPI = 300


# ============================================================
# POLYMER REFERENCE
# ============================================================

def rg_theta(n_residues):
    """
    Sequence-specific ideal-chain reference Rg in nm.

    For a chain of L amino acids:
        N = L - 1 peptide bonds

    Rg_theta =
        sqrt[(2 * lp* * b / 6) * N]
    """

    n_bonds = n_residues - 1

    return np.sqrt(
        (2.0 * LP_STAR * BACKBONE_B / 6.0)
        * n_bonds
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def excluded(path):

    text = str(path).lower()

    return any(
        word.lower() in text
        for word in EXCLUDE
    )


def find_runs(root):

    runs = []

    for dcd in root.rglob("*.dcd"):

        if excluded(dcd):
            continue

        parts = dcd.parts

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        temperature = None

        for part in parts:

            if part in ("300K", "370K"):

                temperature = int(
                    part.replace("K", "")
                )

                break

        if temperature not in TEMPERATURES:
            continue

        # ----------------------------------------------------
        # Replicate
        # ----------------------------------------------------

        replicate = None

        for part in parts:

            if part.lower().startswith("rep"):

                replicate = part
                break

        if replicate is None:
            continue

        # ----------------------------------------------------
        # Sequence name
        #
        # references / SEQUENCE / 300K / repXX / ...
        # ----------------------------------------------------

        try:

            temp_index = parts.index(
                f"{temperature}K"
            )

            sequence = parts[
                temp_index - 1
            ]

        except (ValueError, IndexError):

            print(
                f"WARNING: could not determine "
                f"sequence for {dcd}"
            )

            continue

        # ----------------------------------------------------
        # Topology
        # ----------------------------------------------------

        top_pdb = dcd.parent / "top.pdb"

        if top_pdb.exists():

            pdb = top_pdb

        else:

            pdb_candidates = list(
                dcd.parent.glob("*.pdb")
            )

            if len(pdb_candidates) == 0:

                print(
                    f"WARNING: no PDB found for "
                    f"{dcd}"
                )

                continue

            pdb = pdb_candidates[0]

        runs.append(
            {
                "sequence": sequence,
                "temperature": temperature,
                "replicate": replicate,
                "dcd": dcd,
                "pdb": pdb,
            }
        )

    return runs


# ============================================================
# TRAJECTORY ANALYSIS
# ============================================================

def analyse_run(run):

    print()
    print("=" * 78)

    print(
        f"{run['sequence']} | "
        f"{run['temperature']} K | "
        f"{run['replicate']}"
    )

    print(f"DCD: {run['dcd']}")
    print(f"PDB: {run['pdb']}")

    # --------------------------------------------------------
    # Load trajectory
    # --------------------------------------------------------

    u = mda.Universe(
        str(run["pdb"]),
        str(run["dcd"]),
    )

    atoms = u.atoms

    # --------------------------------------------------------
    # Chain length
    #
    # CALVADOS:
    # one coarse-grained bead = one amino acid
    # --------------------------------------------------------

    n_beads = len(atoms)
    n_mda_residues = len(u.residues)

    theta_nm = rg_theta(n_beads)

    print()
    print(f"CALVADOS beads:       {n_beads}")
    print(f"MDAnalysis residues:  {n_mda_residues}")
    print(f"Rg_theta:             {theta_nm:.3f} nm")

    # --------------------------------------------------------
    # Radius of gyration
    #
    # IMPORTANT:
    # MDAnalysis returns Angstrom here.
    # Divide by 10 to convert to nm.
    # --------------------------------------------------------

    rg_values_nm = []

    for ts in u.trajectory[::STRIDE]:

        rg_angstrom = (
            atoms.radius_of_gyration()
        )

        rg_nm = rg_angstrom / 10.0

        rg_values_nm.append(rg_nm)

    rg_values_nm = np.asarray(
        rg_values_nm,
        dtype=float,
    )

    # --------------------------------------------------------
    # Compact-state classification
    # --------------------------------------------------------

    compact = (
        rg_values_nm < theta_nm
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mean_rg_nm = np.mean(
        rg_values_nm
    )

    normalized_rg = (
        mean_rg_nm / theta_nm
    )

    compact_fraction = np.mean(
        compact
    )

    if np.any(compact):

        mean_compact_rg_nm = np.mean(
            rg_values_nm[compact]
        )

    else:

        mean_compact_rg_nm = np.nan

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(f"Frames analysed:      {len(rg_values_nm)}")

    print(
        f"Mean Rg:              "
        f"{mean_rg_nm:.3f} nm"
    )

    print(
        f"Mean Rg / Rg_theta:   "
        f"{normalized_rg:.3f}"
    )

    print(
        f"Compact fraction:     "
        f"{compact_fraction:.3f}"
    )

    if np.isnan(mean_compact_rg_nm):

        print(
            "Compact-state Rg:     "
            "N/A (no compact frames)"
        )

    else:

        print(
            f"Compact-state Rg:     "
            f"{mean_compact_rg_nm:.3f} nm"
        )

    return {

        "sequence": run["sequence"],
        "temperature": run["temperature"],
        "replicate": run["replicate"],

        "n_beads": n_beads,
        "n_mda_residues": n_mda_residues,
        "n_frames": len(rg_values_nm),

        "rg_theta_nm": theta_nm,

        "mean_rg_nm": mean_rg_nm,
        "normalized_rg": normalized_rg,

        "compact_fraction": compact_fraction,
        "mean_compact_rg_nm": mean_compact_rg_nm,

        "dcd": str(run["dcd"]),
    }


# ============================================================
# REPLICATE SUMMARY
# ============================================================

def make_summary(df):

    metrics = [
        "mean_rg_nm",
        "normalized_rg",
        "compact_fraction",
        "mean_compact_rg_nm",
    ]

    rows = []

    for (
        sequence,
        temperature,
    ), group in df.groupby(
        ["sequence", "temperature"]
    ):

        row = {

            "sequence": sequence,
            "temperature": temperature,

            "n_replicates": len(group),

            "n_beads":
                group["n_beads"].iloc[0],

            "rg_theta_nm":
                group["rg_theta_nm"].iloc[0],
        }

        for metric in metrics:

            values = (
                group[metric]
                .dropna()
            )

            if len(values) > 0:

                row[
                    f"{metric}_mean"
                ] = values.mean()

            else:

                row[
                    f"{metric}_mean"
                ] = np.nan

            if len(values) > 1:

                row[
                    f"{metric}_std"
                ] = values.std(
                    ddof=1
                )

            else:

                row[
                    f"{metric}_std"
                ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# COLORS
# ============================================================

def make_colors(sequences):

    cmap = colormaps["viridis"]

    positions = np.linspace(
        0.05,
        0.95,
        len(sequences),
    )

    return {
        sequence: cmap(position)
        for sequence, position
        in zip(sequences, positions)
    }


# ============================================================
# PLOT
# ============================================================

def make_figure(
    replicate_df,
    summary_df,
    outdir,
):

    # Preserve alphabetic ordering
    sequences = sorted(
        replicate_df[
            "sequence"
        ].unique()
    )

    colors = make_colors(
        sequences
    )

    # --------------------------------------------------------
    # Two clean panels
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
    )

    x = np.arange(
        len(sequences)
    )

    # Position 300 and 370 slightly left/right
    temp_offsets = {
        300: -0.12,
        370: 0.12,
    }

    temp_markers = {
        300: "o",
        370: "s",
    }

    # ========================================================
    # Plot each metric
    # ========================================================

    plot_settings = [

        (
            axes[0],
            "normalized_rg",
            "normalized_rg_mean",
            "normalized_rg_std",
            r"Mean $R_g/R_{g,\Theta}$",
        ),

        (
            axes[1],
            "compact_fraction",
            "compact_fraction_mean",
            "compact_fraction_std",
            "Compact-state population",
        ),
    ]

    for (
        ax,
        replicate_metric,
        mean_metric,
        std_metric,
        ylabel,
    ) in plot_settings:

        for i, sequence in enumerate(
            sequences
        ):

            color = colors[
                sequence
            ]

            mean_positions = []
            mean_values = []

            # --------------------------------------------
            # 300 K and 370 K
            # --------------------------------------------

            for temperature in TEMPERATURES:

                xpos = (
                    i
                    + temp_offsets[
                        temperature
                    ]
                )

                # ----------------------------------------
                # Replicates
                # ----------------------------------------

                reps = replicate_df[
                    (
                        replicate_df[
                            "sequence"
                        ]
                        == sequence
                    )
                    &
                    (
                        replicate_df[
                            "temperature"
                        ]
                        == temperature
                    )
                ]

                values = (
                    reps[
                        replicate_metric
                    ]
                    .dropna()
                    .values
                )

                if len(values) > 0:

                    if len(values) == 1:

                        jitter = np.array(
                            [0.0]
                        )

                    else:

                        jitter = np.linspace(
                            -0.035,
                            0.035,
                            len(values),
                        )

                    ax.scatter(
                        xpos + jitter,
                        values,
                        s=25,
                        color=color,
                        alpha=0.30,
                        edgecolors="none",
                        zorder=2,
                    )

                # ----------------------------------------
                # Replicate mean + SD
                # ----------------------------------------

                summary_row = summary_df[
                    (
                        summary_df[
                            "sequence"
                        ]
                        == sequence
                    )
                    &
                    (
                        summary_df[
                            "temperature"
                        ]
                        == temperature
                    )
                ]

                if len(
                    summary_row
                ) == 0:

                    continue

                mean_value = (
                    summary_row[
                        mean_metric
                    ].iloc[0]
                )

                std_value = (
                    summary_row[
                        std_metric
                    ].iloc[0]
                )

                if np.isnan(
                    mean_value
                ):

                    continue

                if np.isnan(
                    std_value
                ):

                    std_value = 0.0

                ax.errorbar(
                    xpos,
                    mean_value,
                    yerr=std_value,
                    fmt=temp_markers[
                        temperature
                    ],
                    markersize=7,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    capsize=3,
                    linewidth=1.2,
                    zorder=4,
                )

                mean_positions.append(
                    xpos
                )

                mean_values.append(
                    mean_value
                )

            # --------------------------------------------
            # Connect mean 300 -> 370
            # --------------------------------------------

            if len(
                mean_positions
            ) == 2:

                ax.plot(
                    mean_positions,
                    mean_values,
                    color=color,
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=3,
                )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

    # ========================================================
    # Reference lines
    # ========================================================

    axes[0].axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.55,
    )

    axes[1].axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.55,
    )

    axes[1].set_ylim(
        -0.03,
        1.03,
    )

    # ========================================================
    # X axis
    # ========================================================

    axes[1].set_xticks(
        x
    )

    axes[1].set_xticklabels(
        sequences,
        rotation=0,
    )

    axes[1].set_xlabel(
        "ELP sequence"
    )

    # ========================================================
    # Panel labels
    # ========================================================

    axes[0].text(
        -0.08,
        1.03,
        "A",
        transform=axes[0].transAxes,
        fontsize=13,
        fontweight="bold",
    )

    axes[1].text(
        -0.08,
        1.03,
        "B",
        transform=axes[1].transAxes,
        fontsize=13,
        fontweight="bold",
    )

    # ========================================================
    # Temperature legend
    # ========================================================

    legend_handles = [

        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="300 K",
        ),

        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="370 K",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
    )

    fig.subplots_adjust(
        top=0.92,
        bottom=0.11,
        left=0.12,
        right=0.98,
        hspace=0.12,
    )

    # ========================================================
    # Save
    # ========================================================

    pdf = (
        outdir
        / "base_elp_reference_response.pdf"
    )

    png = (
        outdir
        / "base_elp_reference_response.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved figure:")
    print(pdf)
    print(png)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Searching:")
    print(ROOT)

    # --------------------------------------------------------
    # Discover simulations
    # --------------------------------------------------------

    runs = find_runs(
        ROOT
    )

    runs = sorted(
        runs,
        key=lambda r: (
            r["sequence"],
            r["temperature"],
            r["replicate"],
        ),
    )

    print()
    print(
        f"Found {len(runs)} "
        f"replicate trajectories."
    )

    if len(runs) == 0:

        raise RuntimeError(
            "No matching replicate "
            "trajectories found."
        )

    # --------------------------------------------------------
    # Show detected files
    # --------------------------------------------------------

    print()
    print(
        "Detected simulations:"
    )

    print(
        "-" * 78
    )

    for run in runs:

        print(
            f"{run['sequence']:15s} "
            f"{run['temperature']} K  "
            f"{run['replicate']:8s}  "
            f"{run['dcd'].name}"
        )

    # --------------------------------------------------------
    # Analyse trajectories
    # --------------------------------------------------------

    results = []

    for run in runs:

        result = analyse_run(
            run
        )

        results.append(
            result
        )

    replicate_df = pd.DataFrame(
        results
    )

    replicate_df = (
        replicate_df
        .sort_values(
            [
                "sequence",
                "temperature",
                "replicate",
            ]
        )
    )

    # --------------------------------------------------------
    # Save replicate-level data
    # --------------------------------------------------------

    replicate_csv = (
        OUTDIR
        / "base_elp_reference_replicates.csv"
    )

    replicate_df.to_csv(
        replicate_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = make_summary(
        replicate_df
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "sequence",
                "temperature",
            ]
        )
    )

    summary_csv = (
        OUTDIR
        / "base_elp_reference_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    columns = [
        "sequence",
        "temperature",
        "n_replicates",
        "n_beads",
        "rg_theta_nm",
        "mean_rg_nm_mean",
        "normalized_rg_mean",
        "compact_fraction_mean",
        "mean_compact_rg_nm_mean",
    ]

    print(
        summary_df[
            columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    make_figure(
        replicate_df,
        summary_df,
        OUTDIR,
    )

    print()
    print(
        "Analysis complete."
    )

    print(
        f"Results written to:"
    )

    print(
        OUTDIR
    )


if __name__ == "__main__":
    main()