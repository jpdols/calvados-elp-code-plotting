import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

import MDAnalysis as mda


# ============================================================
# LAB ARCHITECTURES
# ============================================================

SEQUENCES = [
    "E_tk1",
    "E_tk4",
    "E_tk8",
    "E_RGD6",
    "E_RGD1",
    "E_RGD1_PH",
    "E_AI_triblock",
]

TEMPERATURES = list(range(300, 371, 10))


# ============================================================
# MARKERS
# ============================================================

MARKERS = {
    "E_tk1": "v",
    "E_tk4": "o",
    "E_tk8": "s",
    "E_RGD6": "^",
    "E_RGD1": "D",
    "E_RGD1_PH": "P",
    "E_AI_triblock": "X",
}


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# FIXED THESIS COLOURS
# ============================================================

def make_colors():
    """
    Fixed colours for the laboratory ELP architectures.

    E_AI_triblock retains the same prominent viridis colour
    used in the previous thesis figures.
    """

    cmap = colormaps["viridis"]

    colors = {
        "E_tk1": cmap(0.05),
        "E_tk4": cmap(0.20),
        "E_tk8": cmap(0.35),
        "E_RGD6": cmap(0.50),
        "E_RGD1": cmap(0.65),
        "E_RGD1_PH": cmap(0.95),
        "E_AI_triblock": cmap(0.82),
    }

    return colors


# ============================================================
# FASTA HANDLING
# ============================================================

def read_fasta(fasta_path):
    """
    Read the amino-acid sequence from a FASTA file.
    """

    sequence = []

    with open(fasta_path, "r") as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            sequence.append(line)

    sequence = "".join(sequence).replace(" ", "").upper()

    if not sequence:
        raise ValueError(
            f"No amino-acid sequence found in FASTA:\n"
            f"{fasta_path}"
        )

    return sequence


# ============================================================
# IDEAL-CHAIN REFERENCE
# ============================================================

def calculate_theta_rg(
    sequence_length,
    lp_nm=0.40,
    b_nm=0.38,
):
    """
    Calculate the sequence-specific Hofmann ideal-chain reference.

    L = number of amino-acid residues
    N = L - 1 peptide bonds

    Rg_theta =
        sqrt(2 * lp * b / 6) * sqrt(N)

    Default parameters:
        lp* = 0.40 nm
        b   = 0.38 nm
    """

    if sequence_length < 2:

        raise ValueError(
            "Sequence must contain at least two residues."
        )

    n_bonds = sequence_length - 1

    prefactor = np.sqrt(
        (2.0 * lp_nm * b_nm) / 6.0
    )

    rg_theta_nm = (
        prefactor
        * np.sqrt(n_bonds)
    )

    return n_bonds, rg_theta_nm


# Locate laboratory sequence directories without assuming all are in scale1.
_DISCOVERED_SEQUENCE_ROOTS = {}

def discover_sequence_roots(primary_root, search_root):
    """Find <sequence>/6.0 under primary_root or anywhere below search_root."""
    found = {}
    for name in SEQUENCES:
        direct = primary_root / name / "6.0"
        if direct.is_dir():
            found[name] = direct
    missing = set(SEQUENCES) - set(found)
    if missing:
        print(f"\nSearching {search_root} for {len(missing)} missing sequence directories...")
        for base, dirs, _ in os.walk(search_root):
            # Once all sequences are located, avoid an unnecessary full-tree scan.
            if not missing:
                break
            path = Path(base)
            if path.name in missing and (path / "6.0").is_dir():
                found[path.name] = path / "6.0"
                missing.remove(path.name)
                print(f"[LOCATED] {path.name}: {found[path.name]}")
                dirs[:] = []
                continue
            # Never walk into trajectories or output folders.
            dirs[:] = [d for d in dirs if d not in {".git", "analysis", "plots", "plotting_thesis"}]
    for name in sorted(missing):
        print(f"[NOT FOUND] {name} below {search_root}")
    return found

# ============================================================
# DETERMINE SEQUENCE INFORMATION
# ============================================================

def determine_sequence_information(root):
    """
    Read idr.fasta directly from the simulation input folders.

    Expected structure:

        root/
            E_tk1/
                6.0/
                    300K/
                        input/
                            idr.fasta
                    310K/
                        input/
                            idr.fasta
                    ...

            E_tk4/
            E_tk8/
            E_RGD6/
            E_RGD1/
            E_RGD1_PH/
            E_AI_triblock/

    All available FASTAs for each sequence are checked to confirm
    that the sequence is identical across temperature.
    """

    sequence_info = {}

    print()
    print("=" * 80)
    print("SEQUENCE INFORMATION")
    print("=" * 80)

    for sequence_name in SEQUENCES:

        sequence_root = _DISCOVERED_SEQUENCE_ROOTS.get(sequence_name, root / sequence_name / "6.0")

        if not sequence_root.exists():

            print(
                f"\n[MISSING DIRECTORY] "
                f"{sequence_root}"
            )

            continue

        found_sequences = []

        # ----------------------------------------------------
        # Find FASTA at each temperature
        # ----------------------------------------------------

        for temperature in TEMPERATURES:

            fasta_path = (
                sequence_root
                / f"{temperature}K"
                / "input"
                / "idr.fasta"
            )

            if not fasta_path.exists():
                continue

            aa_sequence = read_fasta(
                fasta_path
            )

            found_sequences.append({
                "temperature_K": temperature,
                "fasta": fasta_path,
                "sequence": aa_sequence,
            })

        # ----------------------------------------------------
        # Require at least one FASTA
        # ----------------------------------------------------

        if not found_sequences:

            raise FileNotFoundError(
                f"\nNo idr.fasta found for {sequence_name} under:\n"
                f"{sequence_root}\n\n"
                f"Expected a path such as:\n"
                f"{sequence_root}\\300K\\input\\idr.fasta"
            )

        # ----------------------------------------------------
        # Use first available sequence as reference
        # ----------------------------------------------------

        reference_sequence = (
            found_sequences[0]["sequence"]
        )

        # ----------------------------------------------------
        # Check sequence consistency
        # ----------------------------------------------------

        inconsistent_temperatures = []

        for entry in found_sequences[1:]:

            if (
                entry["sequence"]
                != reference_sequence
            ):

                inconsistent_temperatures.append(
                    entry["temperature_K"]
                )

        if inconsistent_temperatures:

            raise ValueError(
                f"\nSequence mismatch for {sequence_name}.\n"
                f"The idr.fasta sequence is not identical "
                f"across temperatures.\n"
                f"Different sequence detected at temperatures: "
                f"{inconsistent_temperatures}"
            )

        # ----------------------------------------------------
        # Determine sequence length and theta reference
        # ----------------------------------------------------

        sequence_length = len(
            reference_sequence
        )

        n_bonds, rg_theta_nm = (
            calculate_theta_rg(
                sequence_length
            )
        )

        fasta_used = (
            found_sequences[0]["fasta"]
        )

        sequence_info[sequence_name] = {
            "fasta": fasta_used,
            "sequence": reference_sequence,
            "sequence_length_aa": sequence_length,
            "n_peptide_bonds": n_bonds,
            "rg_theta_nm": rg_theta_nm,
        }

        # ----------------------------------------------------
        # Print sequence information
        # ----------------------------------------------------

        print(
            f"\n{sequence_name}"
        )

        print(
            f"  Length:               "
            f"{sequence_length} aa"
        )

        print(
            f"  Peptide bonds:        "
            f"{n_bonds}"
        )

        print(
            f"  Rg_theta:             "
            f"{rg_theta_nm:.3f} nm"
        )

        print(
            f"  FASTA used:           "
            f"{fasta_used}"
        )

        print(
            f"  FASTAs found:         "
            f"{len(found_sequences)}"
        )

        print(
            f"  Sequence consistency: OK"
        )

    return sequence_info


# ============================================================
# SAVE SEQUENCE INFORMATION
# ============================================================

def save_sequence_summary(
    sequence_info,
    outdir,
):

    rows = []

    for sequence_name in SEQUENCES:

        if sequence_name not in sequence_info:
            continue

        info = (
            sequence_info[
                sequence_name
            ]
        )

        rows.append({

            "sequence":
                sequence_name,

            "sequence_length_aa":
                info[
                    "sequence_length_aa"
                ],

            "n_peptide_bonds":
                info[
                    "n_peptide_bonds"
                ],

            "rg_theta_nm":
                info[
                    "rg_theta_nm"
                ],

            "fasta":
                str(
                    info["fasta"]
                ),

            "amino_acid_sequence":
                info[
                    "sequence"
                ],
        })

    sequence_df = pd.DataFrame(
        rows
    )

    output_path = (
        outdir
        / "lab_sequence_information.csv"
    )

    sequence_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved sequence information:\n"
        f"{output_path}"
    )


# ============================================================
# FIND TOPOLOGY
# ============================================================

def find_topology(dcd_path):
    """
    Find topology associated with a trajectory.

    First preference:
        top.pdb in same directory as DCD.
    """

    trajectory_dir = (
        dcd_path.parent
    )

    # --------------------------------------------------------
    # Prefer top.pdb
    # --------------------------------------------------------

    top_pdb = (
        trajectory_dir
        / "top.pdb"
    )

    if top_pdb.exists():
        return top_pdb

    # --------------------------------------------------------
    # Otherwise any PDB in same directory
    # --------------------------------------------------------

    pdb_files = sorted(
        trajectory_dir.glob(
            "*.pdb"
        )
    )

    if pdb_files:
        return pdb_files[0]

    # --------------------------------------------------------
    # Search one level higher
    # --------------------------------------------------------

    pdb_files = sorted(
        trajectory_dir.parent.glob(
            "*.pdb"
        )
    )

    if pdb_files:
        return pdb_files[0]

    return None


# ============================================================
# FIND TRAJECTORIES
# ============================================================

def find_runs(root):
    """
    Search recursively for trajectories for all requested
    sequences and temperatures.

    Expected example:

        E_tk4/
            6.0/
                370K/
                    E_tk4/
                        top.pdb
                        E_tk4.dcd
    """

    runs = []

    print()
    print("=" * 80)
    print("TRAJECTORY SEARCH")
    print("=" * 80)

    for sequence_name in SEQUENCES:

        sequence_root = _DISCOVERED_SEQUENCE_ROOTS.get(sequence_name, root / sequence_name / "6.0")

        if not sequence_root.exists():

            print(
                f"\n[MISSING SEQUENCE DIRECTORY] "
                f"{sequence_root}"
            )

            continue

        # ----------------------------------------------------
        # Search DCDs only within this 6.0 folder
        # ----------------------------------------------------

        dcd_files = sorted(
            sequence_root.rglob(
                "*.dcd"
            )
        )

        if not dcd_files:

            print(
                f"\n[NO DCD FILES] "
                f"{sequence_name}"
            )

            continue

        # ----------------------------------------------------
        # Search each temperature
        # ----------------------------------------------------

        for temperature in TEMPERATURES:

            temperature_folder = (
                f"{temperature}K"
            )

            candidates = [
                path
                for path in dcd_files
                if temperature_folder
                in path.parts
            ]

            if not candidates:

                print(
                    f"[MISSING] "
                    f"{sequence_name:16s} "
                    f"{temperature} K"
                )

                continue

            # ------------------------------------------------
            # Prefer DCD filename matching sequence name
            # ------------------------------------------------

            exact = [
                path
                for path in candidates
                if (
                    path.stem.lower()
                    == sequence_name.lower()
                )
            ]

            if exact:

                dcd_path = exact[0]

            else:

                dcd_path = candidates[0]

            # ------------------------------------------------
            # Find topology
            # ------------------------------------------------

            topology_path = (
                find_topology(
                    dcd_path
                )
            )

            if topology_path is None:

                print(
                    f"[NO TOPOLOGY] "
                    f"{sequence_name:16s} "
                    f"{temperature} K"
                )

                print(
                    f"    DCD: "
                    f"{dcd_path}"
                )

                continue

            # ------------------------------------------------
            # Save run
            # ------------------------------------------------

            runs.append({

                "sequence":
                    sequence_name,

                "temperature_K":
                    temperature,

                "topology":
                    topology_path,

                "trajectory":
                    dcd_path,
            })

            print(
                f"[FOUND] "
                f"{sequence_name:16s} "
                f"{temperature} K"
            )

            print(
                f"    TOP: "
                f"{topology_path}"
            )

            print(
                f"    DCD: "
                f"{dcd_path}"
            )

    return runs


# ============================================================
# CALCULATE RADIUS OF GYRATION
# ============================================================

def calculate_rg(
    topology,
    trajectory,
    stride=1,
):
    """
    Calculate Rg for every analysed frame.

    MDAnalysis reports distances in Angstrom.
    Convert to nm by dividing by 10.
    """

    universe = mda.Universe(
        str(topology),
        str(trajectory),
    )

    atoms = (
        universe.atoms
    )

    rg_values_nm = []

    for ts in universe.trajectory[::stride]:

        rg_angstrom = (
            atoms.radius_of_gyration()
        )

        rg_nm = (
            rg_angstrom / 10.0
        )

        rg_values_nm.append(
            rg_nm
        )

    return np.asarray(
        rg_values_nm,
        dtype=float,
    )


# ============================================================
# ANALYSE RG TRAJECTORY
# ============================================================

def analyse_rg(
    rg,
    threshold,
    frame_time_ns,
    stride,
):
    """
    Calculate Rg statistics and threshold-crossing statistics.

    Compact:
        Rg < sequence-specific Rg_theta

    Expanded:
        Rg >= sequence-specific Rg_theta
    """

    if len(rg) == 0:

        raise ValueError(
            "Rg trajectory contains no frames."
        )

    # --------------------------------------------------------
    # State classification
    # --------------------------------------------------------

    compact = (
        rg < threshold
    )

    n_frames = len(
        rg
    )

    n_compact = int(
        np.sum(
            compact
        )
    )

    n_expanded = (
        n_frames
        - n_compact
    )

    fraction_compact = (
        n_compact
        / n_frames
    )

    # --------------------------------------------------------
    # Rg statistics
    # --------------------------------------------------------

    mean_rg = float(
        np.mean(
            rg
        )
    )

    std_rg = float(
        np.std(
            rg,
            ddof=0,
        )
    )

    if n_compact > 0:

        mean_rg_compact = float(
            np.mean(
                rg[compact]
            )
        )

    else:

        mean_rg_compact = np.nan

    if n_expanded > 0:

        mean_rg_expanded = float(
            np.mean(
                rg[~compact]
            )
        )

    else:

        mean_rg_expanded = np.nan

    # --------------------------------------------------------
    # State transitions
    # --------------------------------------------------------

    state = (
        compact.astype(
            int
        )
    )

    transitions = (
        np.diff(
            state
        )
    )

    # Expanded -> compact
    n_collapse_events = int(
        np.sum(
            transitions == 1
        )
    )

    # Compact -> expanded
    n_escape_events = int(
        np.sum(
            transitions == -1
        )
    )

    # --------------------------------------------------------
    # State residence times
    # --------------------------------------------------------

    effective_frame_time_ns = (
        frame_time_ns
        * stride
    )

    time_compact_ns = (
        n_compact
        * effective_frame_time_ns
    )

    time_expanded_ns = (
        n_expanded
        * effective_frame_time_ns
    )

    # --------------------------------------------------------
    # Simple threshold-crossing rates
    # --------------------------------------------------------

    if time_expanded_ns > 0:

        k_collapse = (
            n_collapse_events
            / time_expanded_ns
        )

    else:

        k_collapse = np.nan

    if time_compact_ns > 0:

        k_escape = (
            n_escape_events
            / time_compact_ns
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

        "fraction_compact":
            fraction_compact,

        # Compatibility with older naming
        "fraction_collapsed":
            fraction_compact,

        "mean_rg_compact_nm":
            mean_rg_compact,

        "mean_rg_collapsed_nm":
            mean_rg_compact,

        "mean_rg_expanded_nm":
            mean_rg_expanded,

        "n_compact_frames":
            n_compact,

        "n_expanded_frames":
            n_expanded,

        "n_collapse_events":
            n_collapse_events,

        "n_escape_events":
            n_escape_events,

        "time_compact_ns":
            time_compact_ns,

        "time_expanded_ns":
            time_expanded_ns,

        "k_collapse_per_ns":
            k_collapse,

        "k_escape_per_ns":
            k_escape,
    }


# ============================================================
# GENERIC PLOT FUNCTION
# ============================================================

def plot_metric(
    df,
    metric,
    ylabel,
    filename,
    outdir,
    colors,
    markers,
    horizontal_line=None,
    ylim=None,
):
    """
    Plot a metric versus temperature for all lab architectures.
    """

    fig, ax = plt.subplots(
        figsize=(7.2, 4.5)
    )

    for sequence_name in SEQUENCES:

        sub = df[
            df["sequence"]
            == sequence_name
        ].copy()

        if sub.empty:
            continue

        sub = sub.sort_values(
            "temperature_K"
        )

        ax.plot(
            sub["temperature_K"],
            sub[metric],
            marker=markers.get(
                sequence_name,
                "o",
            ),
            markersize=5,
            linewidth=1.8,
            color=colors[
                sequence_name
            ],
            label=sequence_name,
        )

    if horizontal_line is not None:

        ax.axhline(
            horizontal_line,
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )

    ax.set_xlabel(
        "Temperature (K)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(
        TEMPERATURES
    )

    if ylim is not None:

        ax.set_ylim(
            *ylim
        )

    ax.grid(
        alpha=0.18,
        linewidth=0.7,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )

    fig.tight_layout()

    pdf_path = (
        outdir
        / f"{filename}.pdf"
    )

    png_path = (
        outdir
        / f"{filename}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {pdf_path}"
    )

    print(
        f"Saved: {png_path}"
    )


# ============================================================
# NORMALIZED RG PLOT
# ============================================================

def plot_relative_rg(
    df,
    outdir,
    colors,
    markers,
):
    """
    Plot:

        <Rg> / Rg_theta

    This allows sequences of different lengths to be compared
    relative to their own ideal-chain reference.

    Value > 1:
        mean chain dimensions above ideal-chain reference

    Value < 1:
        mean chain dimensions below ideal-chain reference
    """

    plot_df = (
        df.copy()
    )

    plot_df[
        "rg_relative_theta"
    ] = (
        plot_df[
            "mean_rg_nm"
        ]
        /
        plot_df[
            "rg_theta_nm"
        ]
    )

    plot_metric(
        df=plot_df,
        metric="rg_relative_theta",
        ylabel=(
            r"Mean $R_g/R_{g,\Theta}$"
        ),
        filename=(
            "lab_relative_rg_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=markers,
        horizontal_line=1.0,
    )


# ============================================================
# THESIS FIGURE: FOUR PANELS, SINGLE SHARED LEGEND
# ============================================================

def plot_thesis_four_panel(df, outdir, colors, markers):
    data = df.copy()
    data["rg_relative_theta"] = data["mean_rg_nm"] / data["rg_theta_nm"]
    panels = [
        ("rg_relative_theta", r"Mean $R_g/R_{g,\Theta}$", "A", 1.0),
        ("k_collapse_per_ns", r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)", "B", None),
        ("k_escape_per_ns", r"$k_{\mathrm{escape}}$ (ns$^{-1}$)", "C", None),
        ("mean_rg_compact_nm", r"Mean compact-state $R_g$ (nm)", "D", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.0), sharex=True)
    handles = []
    for ax, (metric, ylabel, letter, reference) in zip(axes.flat, panels):
        for name in SEQUENCES:
            sub = data[data["sequence"] == name].sort_values("temperature_K")
            if sub.empty:
                continue
            line, = ax.plot(sub["temperature_K"], sub[metric],
                            color=colors[name], marker=markers[name],
                            markersize=8, linewidth=2.8, label=name)
            if len(handles) < len(SEQUENCES) and letter == "A":
                handles.append(line)
        if reference is not None:
            ax.axhline(reference, color="0.5", linestyle="--", linewidth=1.4, zorder=0)
        ax.set_ylabel(ylabel, fontsize=17, fontweight="bold", labelpad=9)
        ax.set_xlabel("Temperature (K)", fontsize=17, fontweight="bold", labelpad=8)
        ax.set_xticks(TEMPERATURES)
        ax.tick_params(axis="both", labelsize=14, width=1.3, length=5)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.grid(alpha=0.15, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Panel letters in the top-left corner, above each axis,
        # without overlapping the long y-axis labels.
        ax.text(0.0, 1.045, letter, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="bottom", ha="left")
    fig.legend(handles=handles,
               labels=[h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False,
               prop={"size": 16, "weight": "bold"},
               handlelength=2.0, columnspacing=1.6)
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.11, right=0.98,
                        wspace=0.36, hspace=0.38)
    for ext in ("pdf", "png"):
        path = outdir / f"lab_temperature_four_panel_thesis.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyse temperature sweeps of "
            "CALVADOS-TEA laboratory ELP architectures."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root directory containing "
            "laboratory architecture simulations."
        ),
    )

    parser.add_argument(
        "--search-root",
        default=None,
        help="Broader directory to search for missing sequence folders (e.g. runs).",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help=(
            "Directory for CSV files and plots."
        ),
    )

    parser.add_argument(
        "--frame-time-ns",
        type=float,
        default=0.01,
        help=(
            "Time between saved trajectory "
            "frames in ns. Default: 0.01"
        ),
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Trajectory analysis stride. "
            "Default: 1"
        ),
    )

    args = (
        parser.parse_args()
    )

    root = Path(
        args.root
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_root = Path(args.search_root) if args.search_root else root
    _DISCOVERED_SEQUENCE_ROOTS.clear()
    _DISCOVERED_SEQUENCE_ROOTS.update(discover_sequence_roots(root, search_root))

    colors = (
        make_colors()
    )

    # ========================================================
    # 1. DETERMINE SEQUENCE LENGTHS
    # ========================================================

    sequence_info = (
        determine_sequence_information(
            root
        )
    )

    save_sequence_summary(
        sequence_info,
        outdir,
    )

    # ========================================================
    # 2. FIND TRAJECTORIES
    # ========================================================

    runs = (
        find_runs(
            root
        )
    )

    if not runs:

        raise RuntimeError(
            "No trajectories were found."
        )

    # ========================================================
    # 3. ANALYSE TRAJECTORIES
    # ========================================================

    print()
    print("=" * 80)
    print("TRAJECTORY ANALYSIS")
    print("=" * 80)

    results = []

    for index, run in enumerate(
        runs,
        start=1,
    ):

        sequence_name = (
            run["sequence"]
        )

        temperature = (
            run["temperature_K"]
        )

        if (
            sequence_name
            not in sequence_info
        ):

            print(
                f"\n[SKIPPING] "
                f"{sequence_name}: "
                f"no sequence information."
            )

            continue

        info = (
            sequence_info[
                sequence_name
            ]
        )

        threshold = (
            info[
                "rg_theta_nm"
            ]
        )

        print(
            f"\n[{index}/{len(runs)}] "
            f"{sequence_name} "
            f"{temperature} K"
        )

        print(
            f"    Length:   "
            f"{info['sequence_length_aa']} aa"
        )

        print(
            f"    Rg_theta: "
            f"{threshold:.3f} nm"
        )

        print(
            f"    DCD:      "
            f"{run['trajectory']}"
        )

        # ----------------------------------------------------
        # Calculate trajectory Rg
        # ----------------------------------------------------

        rg = calculate_rg(
            topology=(
                run["topology"]
            ),
            trajectory=(
                run["trajectory"]
            ),
            stride=(
                args.stride
            ),
        )

        # ----------------------------------------------------
        # Analyse trajectory
        # ----------------------------------------------------

        stats = analyse_rg(
            rg=rg,
            threshold=threshold,
            frame_time_ns=(
                args.frame_time_ns
            ),
            stride=(
                args.stride
            ),
        )

        row = {

            "sequence":
                sequence_name,

            "temperature_K":
                temperature,

            "sequence_length_aa":
                info[
                    "sequence_length_aa"
                ],

            "n_peptide_bonds":
                info[
                    "n_peptide_bonds"
                ],

            "rg_theta_nm":
                threshold,

            "fasta":
                str(
                    info["fasta"]
                ),

            "topology":
                str(
                    run["topology"]
                ),

            "trajectory":
                str(
                    run["trajectory"]
                ),

            **stats,
        }

        results.append(
            row
        )

    # ========================================================
    # 4. CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        results
    )

    if df.empty:

        raise RuntimeError(
            "No trajectories were successfully analysed."
        )

    df = df.sort_values(
        [
            "sequence",
            "temperature_K",
        ]
    )

    # Check that all seven sequences and all eight temperatures were found.
    expected = pd.MultiIndex.from_product(
        [SEQUENCES, TEMPERATURES], names=["sequence", "temperature_K"]
    )
    actual = pd.MultiIndex.from_frame(df[["sequence", "temperature_K"]])
    missing = expected.difference(actual)
    if len(missing):
        print("\nWARNING: Some sequences/temperatures are missing:")
        for name, temp in missing:
            print(f"  MISSING: {name} at {temp} K")
        print("The four-panel plot will show only available data.")
        print("Check --root and the 6.0/temperature folder structure.")
    else:
        print("\nCOMPLETE: all seven sequences at all eight temperatures.")

    # ========================================================
    # 5. SAVE COMPLETE SUMMARY
    # ========================================================

    summary_path = (
        outdir
        / "lab_temperature_sweep_summary.csv"
    )

    df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved summary:\n"
        f"{summary_path}"
    )

    # ========================================================
    # 6. PRINT THRESHOLD TABLE
    # ========================================================

    print(
        "\nSequence-specific "
        "ideal-chain references:\n"
    )

    threshold_table = (
        df[
            [
                "sequence",
                "sequence_length_aa",
                "n_peptide_bonds",
                "rg_theta_nm",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "sequence"
        )
    )

    print(
        threshold_table.to_string(
            index=False
        )
    )

    # ========================================================
    # 7. CREATE PLOTS
    # ========================================================

    print()
    print("=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    # --------------------------------------------------------
    # Plot 1: raw mean Rg
    # --------------------------------------------------------

    plot_metric(
        df=df,
        metric="mean_rg_nm",
        ylabel=r"Mean $R_g$ (nm)",
        filename=(
            "lab_mean_rg_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
    )

    # --------------------------------------------------------
    # Plot 2: Rg normalized by sequence-specific theta Rg
    # --------------------------------------------------------

    plot_relative_rg(
        df=df,
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
    )

    # --------------------------------------------------------
    # Plot 3: compact-state population
    # --------------------------------------------------------

    plot_metric(
        df=df,
        metric="fraction_compact",
        ylabel=(
            r"Compact-state population, "
            r"$f_{\mathrm{compact}}$"
        ),
        filename=(
            "lab_compact_fraction_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
        horizontal_line=0.5,
        ylim=(0, 1),
    )

    # --------------------------------------------------------
    # Plot 4: mean Rg of compact conformations
    # --------------------------------------------------------

    plot_metric(
        df=df,
        metric="mean_rg_compact_nm",
        ylabel=(
            r"Mean compact-state $R_g$ (nm)"
        ),
        filename=(
            "lab_compact_rg_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
    )

    # --------------------------------------------------------
    # Plot 5: collapse rate
    # --------------------------------------------------------

    plot_metric(
        df=df,
        metric="k_collapse_per_ns",
        ylabel=(
            r"$k_{\mathrm{collapse}}$ "
            r"(ns$^{-1}$)"
        ),
        filename=(
            "lab_k_collapse_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
    )

    # --------------------------------------------------------
    # Plot 6: escape rate
    # --------------------------------------------------------

    plot_metric(
        df=df,
        metric="k_escape_per_ns",
        ylabel=(
            r"$k_{\mathrm{escape}}$ "
            r"(ns$^{-1}$)"
        ),
        filename=(
            "lab_k_escape_vs_temperature"
        ),
        outdir=outdir,
        colors=colors,
        markers=MARKERS,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    plot_thesis_four_panel(df, outdir, colors, MARKERS)

    print(
        f"\nAll outputs saved to:\n"
        f"{outdir}"
    )


if __name__ == "__main__":
    main()