from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda


# ============================================================
# Helpers
# ============================================================

def gamma_from_folder(path):
    """Read gamma from folders such as 1.0, 3.0, 6.0."""
    try:
        return float(path.name)
    except ValueError:
        return None


def find_run_directory(temp_dir, system):
    """
    Find the trajectory directory.

    If replicate folders exist, use rep01 only.
    Otherwise use the direct system directory.
    """

    rep01 = temp_dir / "rep01" / system
    direct = temp_dir / system

    if rep01.exists():
        return rep01, "rep01"

    if direct.exists():
        return direct, "direct"

    return None, None


def find_topology(run_dir):
    """Find topology PDB."""

    candidates = [
        run_dir / "top.pdb",
        run_dir / "topology.pdb",
    ]

    for path in candidates:
        if path.exists():
            return path

    pdbs = list(run_dir.glob("*.pdb"))

    if len(pdbs) == 1:
        return pdbs[0]

    return None


def find_trajectory(run_dir, system):
    """Find DCD trajectory."""

    candidates = [
        run_dir / f"{system}.dcd",
        run_dir / "traj.dcd",
        run_dir / "trajectory.dcd",
    ]

    for path in candidates:
        if path.exists():
            return path

    dcds = list(run_dir.glob("*.dcd"))

    if len(dcds) == 1:
        return dcds[0]

    return None


# ============================================================
# Rg calculation
# ============================================================

def calculate_rg(topology, trajectory, stride=1):
    """
    Calculate radius of gyration.

    IMPORTANT:
    MDAnalysis returns distances in Angstrom for these
    CALVADOS trajectories.

    Therefore:
        Rg_nm = Rg_Angstrom / 10
    """

    universe = mda.Universe(
        str(topology),
        str(trajectory)
    )

    protein = universe.atoms

    rg_values = []

    for ts in universe.trajectory[::stride]:

        rg_angstrom = protein.radius_of_gyration()

        # Angstrom -> nm
        rg_nm = rg_angstrom / 10.0

        rg_values.append(rg_nm)

    return np.asarray(
        rg_values,
        dtype=float
    )


# ============================================================
# Two-state statistics and kinetics
# ============================================================

def calculate_state_statistics(
    rg,
    threshold,
    frame_time_ns,
    stride=1
):
    """
    Define states using Rg threshold:

        collapsed: Rg < threshold
        expanded:  Rg >= threshold

    Transition rates:

        k_collapse =
            N(expanded -> collapsed) / total expanded time

        k_escape =
            N(collapsed -> expanded) / total collapsed time

    Rates are returned in ns^-1.
    """

    if len(rg) == 0:
        return {}

    collapsed = rg < threshold

    n_frames = len(rg)

    # Actual time between analysed frames
    dt = frame_time_ns * stride

    # --------------------------------------------------------
    # General Rg statistics
    # --------------------------------------------------------

    mean_rg = np.mean(rg)
    std_rg = np.std(rg)

    fraction_collapsed = np.mean(collapsed)

    # --------------------------------------------------------
    # Mean Rg in each state
    # --------------------------------------------------------

    if np.any(collapsed):
        mean_rg_collapsed = np.mean(
            rg[collapsed]
        )
    else:
        mean_rg_collapsed = np.nan

    if np.any(~collapsed):
        mean_rg_expanded = np.mean(
            rg[~collapsed]
        )
    else:
        mean_rg_expanded = np.nan

    # --------------------------------------------------------
    # Count transitions
    # --------------------------------------------------------

    previous = collapsed[:-1]
    current = collapsed[1:]

    # expanded -> collapsed
    n_collapse = int(
        np.sum(
            (~previous) & current
        )
    )

    # collapsed -> expanded
    n_escape = int(
        np.sum(
            previous & (~current)
        )
    )

    # --------------------------------------------------------
    # Residence times
    # --------------------------------------------------------

    n_collapsed_frames = int(
        np.sum(collapsed)
    )

    n_expanded_frames = int(
        np.sum(~collapsed)
    )

    time_collapsed = (
        n_collapsed_frames * dt
    )

    time_expanded = (
        n_expanded_frames * dt
    )

    # --------------------------------------------------------
    # Transition rates
    # --------------------------------------------------------

    if time_expanded > 0:
        k_collapse = (
            n_collapse /
            time_expanded
        )
    else:
        k_collapse = np.nan

    if time_collapsed > 0:
        k_escape = (
            n_escape /
            time_collapsed
        )
    else:
        k_escape = np.nan

    return {

        "n_frames":
            n_frames,

        "mean_rg_nm":
            mean_rg,

        "std_rg_nm":
            std_rg,

        "fraction_collapsed":
            fraction_collapsed,

        "mean_rg_collapsed_nm":
            mean_rg_collapsed,

        "mean_rg_expanded_nm":
            mean_rg_expanded,

        "n_collapsed_frames":
            n_collapsed_frames,

        "n_expanded_frames":
            n_expanded_frames,

        "n_collapse_events":
            n_collapse,

        "n_escape_events":
            n_escape,

        "time_collapsed_ns":
            time_collapsed,

        "time_expanded_ns":
            time_expanded,

        "k_collapse_per_ns":
            k_collapse,

        "k_escape_per_ns":
            k_escape,
    }


# ============================================================
# Simulation analysis
# ============================================================

def analyse_simulations(
    root,
    system,
    temperatures,
    threshold,
    frame_time_ns,
    stride
):

    rows = []

    gamma_dirs = []

    for path in root.iterdir():

        if not path.is_dir():
            continue

        gamma = gamma_from_folder(path)

        if gamma is not None:
            gamma_dirs.append(
                (gamma, path)
            )

    gamma_dirs.sort()

    print()
    print("==============================")
    print("FOUND GAMMA DIRECTORIES")
    print("==============================")

    for gamma, path in gamma_dirs:
        print(
            f"gamma = {gamma:g}: {path}"
        )

    print()

    # --------------------------------------------------------
    # Analyse every available gamma / temperature
    # --------------------------------------------------------

    for gamma, gamma_dir in gamma_dirs:

        for temperature in temperatures:

            temp_dir = (
                gamma_dir /
                f"{temperature}K"
            )

            if not temp_dir.exists():

                print(
                    f"[SKIP] gamma={gamma:g}, "
                    f"T={temperature} K: "
                    f"temperature folder missing"
                )

                continue

            run_dir, layout = (
                find_run_directory(
                    temp_dir,
                    system
                )
            )

            if run_dir is None:

                print(
                    f"[SKIP] gamma={gamma:g}, "
                    f"T={temperature} K: "
                    f"run directory missing"
                )

                continue

            topology = find_topology(
                run_dir
            )

            trajectory = find_trajectory(
                run_dir,
                system
            )

            if topology is None:

                print(
                    f"[SKIP] gamma={gamma:g}, "
                    f"T={temperature} K: "
                    f"topology missing"
                )

                continue

            if trajectory is None:

                print(
                    f"[SKIP] gamma={gamma:g}, "
                    f"T={temperature} K: "
                    f"DCD missing"
                )

                continue

            print()
            print(
                f"[ANALYSE] gamma={gamma:g} | "
                f"T={temperature} K | "
                f"{layout}"
            )

            print(
                f"  topology:   {topology}"
            )

            print(
                f"  trajectory: {trajectory}"
            )

            # ------------------------------------------------
            # Calculate Rg
            # ------------------------------------------------

            rg = calculate_rg(
                topology,
                trajectory,
                stride=stride
            )

            # ------------------------------------------------
            # Calculate state statistics
            # ------------------------------------------------

            stats = (
                calculate_state_statistics(
                    rg,
                    threshold=threshold,
                    frame_time_ns=frame_time_ns,
                    stride=stride
                )
            )

            print(
                f"  Mean Rg: "
                f"{stats['mean_rg_nm']:.3f} nm"
            )

            print(
                f"  Collapsed fraction: "
                f"{stats['fraction_collapsed']:.4f}"
            )

            print(
                f"  Collapse events: "
                f"{stats['n_collapse_events']}"
            )

            print(
                f"  Escape events: "
                f"{stats['n_escape_events']}"
            )

            print(
                f"  k_collapse: "
                f"{stats['k_collapse_per_ns']:.6g} ns^-1"
            )

            print(
                f"  k_escape: "
                f"{stats['k_escape_per_ns']:.6g} ns^-1"
            )

            row = {

                "gamma":
                    gamma,

                "temperature_K":
                    temperature,

                "layout":
                    layout,

                "run_directory":
                    str(run_dir),

                "topology":
                    str(topology),

                "trajectory":
                    str(trajectory),

                **stats,
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Generic plotting
# ============================================================

def plot_metric(
    df,
    column,
    ylabel,
    title,
    output,
    temperatures,
    threshold_line=None
):

    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    plotted_anything = False

    for gamma in sorted(
        df["gamma"].unique()
    ):

        sub = (
            df[
                df["gamma"] == gamma
            ]
            .sort_values(
                "temperature_K"
            )
        )

        # Remove NaN values for this metric
        valid = sub[
            np.isfinite(sub[column])
        ]

        if valid.empty:
            continue

        plotted_anything = True

        ax.plot(
            valid["temperature_K"],
            valid[column],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=rf"$\gamma$ = {gamma:g}"
        )

    # --------------------------------------------------------
    # Optional horizontal threshold
    # --------------------------------------------------------

    if threshold_line is not None:

        ax.axhline(
            threshold_line,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=(
                rf"$R_g$ threshold = "
                rf"{threshold_line:.2f} nm"
            )
        )

    ax.set_xlabel(
        "Temperature (K)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.set_xlim(
        min(temperatures) - 2,
        max(temperatures) + 2
    )

    ax.set_xticks(
        temperatures
    )

    ax.grid(
        alpha=0.2,
        linewidth=0.7
    )

    if plotted_anything:

        ax.legend(
            frameon=False,
            ncol=2
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No defined values",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes
        )

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Residue lambda analysis
# ============================================================

def find_column(
    df,
    candidates
):

    lower = {
        col.lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    return None


def load_lambda_data(
    residues_root,
    gammas,
    temperatures,
    residues
):

    rows = []

    for gamma in gammas:

        gamma_file = (
            f"{int(gamma)}p0"
        )

        for temperature in temperatures:

            filename = (
                f"residues_gamma_"
                f"{gamma_file}_"
                f"{temperature}K_"
                f"argdefault_minus.csv"
            )

            path = (
                residues_root /
                filename
            )

            if not path.exists():

                print(
                    f"[LAMBDA SKIP] "
                    f"Missing: {filename}"
                )

                continue

            df = pd.read_csv(path)

            # -----------------------------------------------
            # Identify residue column
            # -----------------------------------------------

            residue_col = find_column(
                df,
                [
                    "one",
                    "residue",
                    "aa",
                    "amino_acid",
                    "aminoacid",
                ]
            )

            # -----------------------------------------------
            # Identify lambda column
            # -----------------------------------------------

            lambda_col = find_column(
                df,
                [
                    "lambdas",
                    "lambda_T",
                    "lambda",
                    "lambda_t",
                ]
            )

            if residue_col is None:

                raise ValueError(
                    "\nCould not identify "
                    "residue column in:\n"
                    f"{path}\n\n"
                    f"Columns:\n"
                    f"{list(df.columns)}"
                )

            if lambda_col is None:

                raise ValueError(
                    "\nCould not identify "
                    "lambda column in:\n"
                    f"{path}\n\n"
                    f"Columns:\n"
                    f"{list(df.columns)}"
                )

            # -----------------------------------------------
            # Extract V, I, P
            # -----------------------------------------------

            for residue in residues:

                selection = df[
                    df[residue_col]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    == residue
                ]

                if selection.empty:

                    raise ValueError(
                        f"Residue {residue} "
                        f"not found in:\n"
                        f"{path}"
                    )

                value = float(
                    selection.iloc[0][
                        lambda_col
                    ]
                )

                rows.append({

                    "gamma":
                        float(gamma),

                    "temperature_K":
                        temperature,

                    "residue":
                        residue,

                    "lambda":
                        value,

                    "source_file":
                        str(path),
                })

    return pd.DataFrame(rows)


# ============================================================
# Plot V / I / P lambda(T)
# ============================================================

def plot_lambda_vip(
    lambda_df,
    output,
    temperatures
):

    residues = [
        "V",
        "I",
        "P"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 4.2),
        sharex=True,
        sharey=False
    )

    for ax, residue in zip(
        axes,
        residues
    ):

        residue_df = lambda_df[
            lambda_df["residue"]
            == residue
        ]

        for gamma in sorted(
            residue_df[
                "gamma"
            ].unique()
        ):

            sub = (
                residue_df[
                    residue_df[
                        "gamma"
                    ] == gamma
                ]
                .sort_values(
                    "temperature_K"
                )
            )

            ax.plot(
                sub["temperature_K"],
                sub["lambda"],
                marker="o",
                linewidth=1.6,
                markersize=4,
                label=(
                    rf"$\gamma$ = "
                    rf"{gamma:g}"
                )
            )

        ax.set_title(
            residue
        )

        ax.set_xlabel(
            "Temperature (K)"
        )

        ax.set_xticks(
            temperatures
        )

        ax.grid(
            alpha=0.2,
            linewidth=0.7
        )

    axes[0].set_ylabel(
        r"CALVADOS interaction parameter $\lambda$"
    )

    handles, labels = (
        axes[-1]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.04
        ),
        ncol=7,
        frameon=False
    )

    fig.suptitle(
        "Temperature dependence of V, I and P "
        "interaction parameters",
        y=1.12
    )

    fig.tight_layout()

    fig.savefig(
        output.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root directory containing "
            "gamma folders"
        )
    )

    parser.add_argument(
        "--residues-root",
        type=Path,
        required=True,
        help=(
            "Directory containing "
            "TEA residue CSV files"
        )
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--system",
        default="E_AI_triblock"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.30,
        help=(
            "Collapsed-state Rg "
            "threshold in nm"
        )
    )

    parser.add_argument(
        "--frame-time-ns",
        type=float,
        default=0.01,
        help=(
            "Time between saved trajectory "
            "frames in ns"
        )
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Temperature series
    # --------------------------------------------------------

    temperatures = [
        300,
        310,
        320,
        330,
        340,
        350,
        360,
        370,
    ]

    args.outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("======================================")
    print("GAMMA SCALING ANALYSIS")
    print("======================================")

    print(
        f"System: {args.system}"
    )

    print(
        f"Collapsed threshold: "
        f"{args.threshold:.2f} nm"
    )

    print(
        f"Frame time: "
        f"{args.frame_time_ns} ns"
    )

    print(
        f"Stride: {args.stride}"
    )

    # ========================================================
    # Simulation analysis
    # ========================================================

    results = analyse_simulations(
        root=args.root,
        system=args.system,
        temperatures=temperatures,
        threshold=args.threshold,
        frame_time_ns=args.frame_time_ns,
        stride=args.stride
    )

    if results.empty:

        print()
        print(
            "WARNING: No simulation "
            "trajectories were found."
        )

    else:

        results = (
            results
            .sort_values(
                [
                    "gamma",
                    "temperature_K"
                ]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Save master CSV
        # ----------------------------------------------------

        csv_path = (
            args.outdir /
            "gamma_scaling_summary.csv"
        )

        results.to_csv(
            csv_path,
            index=False
        )

        print()
        print(
            f"[SAVED] {csv_path}"
        )

        # ----------------------------------------------------
        # Print concise summary table
        # ----------------------------------------------------

        print()
        print(
            "======================================"
        )
        print(
            "SUMMARY"
        )
        print(
            "======================================"
        )

        columns_to_show = [
            "gamma",
            "temperature_K",
            "mean_rg_nm",
            "fraction_collapsed",
            "n_collapse_events",
            "n_escape_events",
            "k_collapse_per_ns",
            "k_escape_per_ns",
        ]

        print(
            results[
                columns_to_show
            ].to_string(
                index=False
            )
        )

        # ====================================================
        # PLOTS
        # ====================================================

        # ----------------------------------------------------
        # Mean Rg
        # ----------------------------------------------------

        plot_metric(
            results,
            "mean_rg_nm",
            r"Mean $R_g$ (nm)",
            "Mean radius of gyration",
            args.outdir /
            "mean_rg_vs_temperature",
            temperatures,
            threshold_line=args.threshold
        )

        # ----------------------------------------------------
        # Fraction collapsed
        # ----------------------------------------------------

        plot_metric(
            results,
            "fraction_collapsed",
            "Collapsed fraction",
            "Collapsed population",
            args.outdir /
            "fraction_collapsed_vs_temperature",
            temperatures
        )

        # ----------------------------------------------------
        # Mean collapsed-state Rg
        # ----------------------------------------------------

        plot_metric(
            results,
            "mean_rg_collapsed_nm",
            r"Mean collapsed-state $R_g$ (nm)",
            "Radius of gyration of the collapsed state",
            args.outdir /
            "rg_collapsed_vs_temperature",
            temperatures
        )

        # ----------------------------------------------------
        # Collapse rate
        # ----------------------------------------------------

        plot_metric(
            results,
            "k_collapse_per_ns",
            r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)",
            "Collapse rate",
            args.outdir /
            "k_collapse_vs_temperature",
            temperatures
        )

        # ----------------------------------------------------
        # Escape rate
        # ----------------------------------------------------

        plot_metric(
            results,
            "k_escape_per_ns",
            r"$k_{\mathrm{escape}}$ (ns$^{-1}$)",
            "Escape rate",
            args.outdir /
            "k_escape_vs_temperature",
            temperatures
        )

    # ========================================================
    # Lambda(T) analysis
    # ========================================================

    gammas = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]

    lambda_df = load_lambda_data(
        residues_root=args.residues_root,
        gammas=gammas,
        temperatures=temperatures,
        residues=[
            "V",
            "I",
            "P"
        ]
    )

    if not lambda_df.empty:

        lambda_csv = (
            args.outdir /
            "lambda_VIP_vs_temperature.csv"
        )

        lambda_df.to_csv(
            lambda_csv,
            index=False
        )

        print()
        print(
            f"[SAVED] {lambda_csv}"
        )

        plot_lambda_vip(
            lambda_df,
            args.outdir /
            "lambda_VIP_vs_temperature",
            temperatures
        )

    print()
    print("======================================")
    print("ANALYSIS COMPLETE")
    print("======================================")
    print()
    print(
        f"Results written to:\n"
        f"{args.outdir}"
    )


if __name__ == "__main__":
    main()