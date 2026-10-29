#!/usr/bin/env python3
"""
contact_heatmaps_by_state.py

Create residue-residue contact-probability heatmaps separately for
collapsed and non-collapsed trajectory frames.

Designed for CALVADOS-like coarse-grained trajectories where the topology
contains one bead per residue, but it also works with atomistic trajectories
if you select one representative atom per residue (e.g. CA).

Example
-------
python contact_heatmaps_by_state.py \
    --topology top.pdb \
    --trajectory traj.dcd \
    --fasta sequence.fasta \
    --outdir contact_maps \
    --rg-threshold 4.30 \
    --contact-cutoff 0.8 \
    --selection "all" \
    --motif-length 5 \
    --exclude-neighbors 1 \
    --stride 1

For atomistic protein trajectories, use e.g.
    --selection "name CA"

Outputs
-------
contact_maps/
    collapsed_contact_probability.csv
    noncollapsed_contact_probability.csv
    collapsed_contact_heatmap.png
    noncollapsed_contact_heatmap.png
    state_summary.csv

Notes
-----
- A DCD file does not contain topology information, so --topology is required.
- "Collapsed" is defined as Rg < --rg-threshold.
- Contacts are counted when selected-bead distance < --contact-cutoff.
- The heatmap uses logarithmic color normalization to prevent contacts with
  probability near 1 (especially local neighbors) from dominating visually.
- Sequence-neighbor contacts can additionally be masked with
  --exclude-neighbors. For example, 1 masks i,i and i,i+/-1.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
except ImportError as exc:
    raise SystemExit(
        "MDAnalysis is required.\n"
        "Install with:\n"
        "  conda install -c conda-forge mdanalysis\n"
        "or\n"
        "  pip install MDAnalysis"
    ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create collapsed and non-collapsed residue contact heatmaps "
            "from a trajectory."
        )
    )

    p.add_argument(
        "--topology",
        required=True,
        type=Path,
        help="Topology file matching the DCD atom/bead order, e.g. PDB.",
    )
    p.add_argument(
        "--trajectory",
        required=True,
        type=Path,
        help="Trajectory file, e.g. DCD.",
    )
    p.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="FASTA sequence used to label residues and motif strips.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("contact_heatmaps"),
    )
    p.add_argument(
        "--rg-threshold",
        type=float,
        default=4.30,
        help="Collapsed if Rg < this threshold in nm. Default: 4.30.",
    )
    p.add_argument(
        "--contact-cutoff",
        type=float,
        default=0.8,
        help="Contact distance cutoff in nm. Default: 0.8.",
    )
    p.add_argument(
        "--selection",
        default="all",
        help=(
            'MDAnalysis atom selection used as residue representatives. '
            'For one-bead-per-residue CG trajectories use "all"; '
            'for atomistic trajectories use e.g. "name CA".'
        ),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Trajectory stride. Default: 1.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="First frame index. Default: 0.",
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index (Python slicing semantics).",
    )
    p.add_argument(
        "--exclude-neighbors",
        type=int,
        default=1,
        help=(
            "Mask contacts with |i-j| <= this value in plots and output "
            "matrices. Default: 1."
        ),
    )
    p.add_argument(
        "--motif-length",
        type=int,
        default=5,
        help=(
            "Length of repeating motif blocks used for axis color strips. "
            "For VPGVG/VPGEG/VPGKG-like ELP motifs use 5. Default: 5."
        ),
    )
    p.add_argument(
        "--min-prob",
        type=float,
        default=1e-3,
        help=(
            "Minimum displayed contact probability for LogNorm. "
            "Values below this are shown at the bottom of the scale. "
            "Default: 1e-3."
        ),
    )
    p.add_argument(
        "--tick-every",
        type=int,
        default=20,
        help="Show residue labels every N residues. Default: 20.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Sequence / motif handling
# ---------------------------------------------------------------------------

def read_fasta(path: Path) -> str:
    sequence_parts: list[str] = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_parts.append(line)

    sequence = re.sub(r"[^A-Za-z]", "", "".join(sequence_parts)).upper()

    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")

    return sequence


def motif_blocks(sequence: str, motif_length: int) -> tuple[list[str], list[str]]:
    """
    Assign a motif label to each residue by slicing the FASTA into fixed-width
    motif blocks.

    For a sequence such as:
        VPGVGVPGEGVPGKG
    with motif_length=5:
        residues 1-5   -> VPGVG
        residues 6-10  -> VPGEG
        residues 11-15 -> VPGKG

    If the final block is shorter, it receives its literal partial sequence.
    """
    residue_to_motif: list[str] = []
    unique_motifs: list[str] = []

    for start in range(0, len(sequence), motif_length):
        motif = sequence[start : start + motif_length]

        if motif not in unique_motifs:
            unique_motifs.append(motif)

        residue_to_motif.extend([motif] * len(motif))

    return residue_to_motif, unique_motifs


# ---------------------------------------------------------------------------
# Rg and contact accumulation
# ---------------------------------------------------------------------------

def radius_of_gyration_nm(positions_angstrom: np.ndarray, masses: np.ndarray) -> float:
    """
    Calculate mass-weighted radius of gyration in nm.

    MDAnalysis coordinates are generally stored in Å, hence division by 10.
    For CALVADOS CG trajectories with equal/near-equal residue bead masses,
    this is the standard mass-weighted Rg definition.
    """
    masses = np.asarray(masses, dtype=float)

    if (
        masses.size != positions_angstrom.shape[0]
        or not np.all(np.isfinite(masses))
        or np.sum(masses) <= 0
    ):
        masses = np.ones(positions_angstrom.shape[0], dtype=float)

    center = np.average(positions_angstrom, axis=0, weights=masses)
    dr = positions_angstrom - center

    rg2_angstrom2 = np.average(
        np.sum(dr * dr, axis=1),
        weights=masses,
    )

    return math.sqrt(rg2_angstrom2) / 10.0


def accumulate_contact_probabilities(
    universe: mda.Universe,
    selection: str,
    rg_threshold_nm: float,
    contact_cutoff_nm: float,
    start: int,
    stop: int | None,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, float]]]:

    group = universe.select_atoms(selection)

    if group.n_atoms == 0:
        raise ValueError(f"Selection returned zero atoms: {selection!r}")

    n = group.n_atoms
    collapsed_counts = np.zeros((n, n), dtype=np.uint32)
    noncollapsed_counts = np.zeros((n, n), dtype=np.uint32)

    n_collapsed = 0
    n_noncollapsed = 0
    frame_rows: list[dict[str, float]] = []

    cutoff_angstrom = contact_cutoff_nm * 10.0

    trajectory_slice = universe.trajectory[start:stop:stride]

    for local_index, ts in enumerate(trajectory_slice):
        positions = group.positions.copy()

        # Rg of the same representative beads used for the contact matrix.
        rg_nm = radius_of_gyration_nm(positions, group.masses)

        # PBC-aware distances when dimensions are valid.
        box = ts.dimensions
        if box is not None and len(box) >= 3 and np.all(np.asarray(box[:3]) > 0):
            distances = distance_array(
                positions,
                positions,
                box=box,
            )
        else:
            distances = distance_array(
                positions,
                positions,
            )

        contacts = distances < cutoff_angstrom

        # The diagonal is not an informative "contact".
        np.fill_diagonal(contacts, False)

        collapsed = rg_nm < rg_threshold_nm

        if collapsed:
            collapsed_counts += contacts
            n_collapsed += 1
            state = "collapsed"
        else:
            noncollapsed_counts += contacts
            n_noncollapsed += 1
            state = "noncollapsed"

        frame_rows.append(
            {
                "frame": int(ts.frame),
                "time_ps": float(ts.time),
                "rg_nm": rg_nm,
                "state": state,
            }
        )

        if (local_index + 1) % 1000 == 0:
            print(
                f"Processed {local_index + 1} sampled frames "
                f"(collapsed={n_collapsed}, noncollapsed={n_noncollapsed})"
            )

    collapsed_prob = (
        collapsed_counts.astype(float) / n_collapsed
        if n_collapsed > 0
        else np.full((n, n), np.nan)
    )

    noncollapsed_prob = (
        noncollapsed_counts.astype(float) / n_noncollapsed
        if n_noncollapsed > 0
        else np.full((n, n), np.nan)
    )

    return (
        collapsed_prob,
        noncollapsed_prob,
        {
            "collapsed_frames": n_collapsed,
            "noncollapsed_frames": n_noncollapsed,
            "total_frames": n_collapsed + n_noncollapsed,
        },
        frame_rows,
    )


def mask_local_neighbors(matrix: np.ndarray, exclude_neighbors: int) -> np.ndarray:
    out = matrix.copy()
    n = out.shape[0]

    if exclude_neighbors < 0:
        return out

    ii, jj = np.indices((n, n))
    mask = np.abs(ii - jj) <= exclude_neighbors
    out[mask] = np.nan

    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_motif_color_map(unique_motifs: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    return {
        motif: cmap(i % 20)
        for i, motif in enumerate(unique_motifs)
    }


def draw_motif_strips(
    ax: plt.Axes,
    residue_to_motif: list[str],
    motif_colors: dict[str, tuple],
) -> None:
    """
    Add top and left color strips showing motif identity per residue.
    """
    n = len(residue_to_motif)

    # Top strip.
    top = ax.inset_axes([0.0, 1.005, 1.0, 0.025], transform=ax.transAxes)
    top.set_xlim(0, n)
    top.set_ylim(0, 1)
    top.axis("off")

    # Left strip.
    left = ax.inset_axes([-0.03, 0.0, 0.025, 1.0], transform=ax.transAxes)
    left.set_xlim(0, 1)
    left.set_ylim(n, 0)
    left.axis("off")

    for i, motif in enumerate(residue_to_motif):
        color = motif_colors[motif]

        top.add_patch(
            Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="none")
        )
        left.add_patch(
            Rectangle((0, i), 1, 1, facecolor=color, edgecolor="none")
        )


def plot_contact_heatmap(
    matrix: np.ndarray,
    sequence: str,
    residue_to_motif: list[str],
    unique_motifs: list[str],
    title: str,
    output: Path,
    min_prob: float,
    tick_every: int,
    dpi: int,
) -> None:

    finite_positive = matrix[np.isfinite(matrix) & (matrix > 0)]

    if finite_positive.size == 0:
        print(f"WARNING: no finite positive contacts for {title}")
        return

    # Keep a stable log range. Probabilities cannot exceed 1.
    vmin = max(min_prob, float(np.nanmin(finite_positive)))
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 9.2))

    image = ax.imshow(
        matrix,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )

    n = len(sequence)

    ticks = np.arange(0, n, tick_every)
    tick_labels = [
        f"{i + 1} {sequence[i]}"
        for i in ticks
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        tick_labels,
        rotation=90,
        fontsize=7,
    )
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        tick_labels,
        fontsize=7,
    )

    ax.set_xlabel("Residue")
    ax.set_ylabel("Residue")
    ax.set_title(title)

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Contact probability (log scale)")

    motif_colors = make_motif_color_map(unique_motifs)
    draw_motif_strips(
        ax,
        residue_to_motif=residue_to_motif,
        motif_colors=motif_colors,
    )

    # Motif legend.
    handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=motif_colors[motif],
            edgecolor="none",
            label=motif,
        )
        for motif in unique_motifs
    ]

    ax.legend(
        handles=handles,
        title="Motif",
        bbox_to_anchor=(1.14, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence = read_fasta(args.fasta)

    universe = mda.Universe(
        str(args.topology),
        str(args.trajectory),
    )

    group = universe.select_atoms(args.selection)

    if group.n_atoms != len(sequence):
        raise ValueError(
            "\nSelection/FASTA length mismatch:\n"
            f"  selected atoms/beads = {group.n_atoms}\n"
            f"  FASTA residues       = {len(sequence)}\n\n"
            "For a CALVADOS one-bead-per-residue trajectory, use:\n"
            '  --selection "all"\n'
            "For an atomistic trajectory, use a one-atom-per-residue selection,\n"
            'for example:\n'
            '  --selection "name CA"\n'
        )

    residue_to_motif, unique_motifs = motif_blocks(
        sequence,
        args.motif_length,
    )

    (
        collapsed_prob,
        noncollapsed_prob,
        counts,
        frame_rows,
    ) = accumulate_contact_probabilities(
        universe=universe,
        selection=args.selection,
        rg_threshold_nm=args.rg_threshold,
        contact_cutoff_nm=args.contact_cutoff,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )

    collapsed_prob = mask_local_neighbors(
        collapsed_prob,
        args.exclude_neighbors,
    )
    noncollapsed_prob = mask_local_neighbors(
        noncollapsed_prob,
        args.exclude_neighbors,
    )

    # Save matrices.
    labels = [
        f"{i + 1}_{aa}"
        for i, aa in enumerate(sequence)
    ]

    pd.DataFrame(
        collapsed_prob,
        index=labels,
        columns=labels,
    ).to_csv(
        args.outdir / "collapsed_contact_probability.csv"
    )

    pd.DataFrame(
        noncollapsed_prob,
        index=labels,
        columns=labels,
    ).to_csv(
        args.outdir / "noncollapsed_contact_probability.csv"
    )

    pd.DataFrame(frame_rows).to_csv(
        args.outdir / "frame_states.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "rg_threshold_nm": args.rg_threshold,
                "contact_cutoff_nm": args.contact_cutoff,
                "stride": args.stride,
                "exclude_neighbors": args.exclude_neighbors,
                **counts,
            }
        ]
    ).to_csv(
        args.outdir / "state_summary.csv",
        index=False,
    )

    plot_contact_heatmap(
        collapsed_prob,
        sequence=sequence,
        residue_to_motif=residue_to_motif,
        unique_motifs=unique_motifs,
        title=(
            "Collapsed-state residue contact probability\n"
            f"Rg < {args.rg_threshold:.2f} nm | "
            f"contact < {args.contact_cutoff:.2f} nm | "
            f"n = {counts['collapsed_frames']} frames"
        ),
        output=args.outdir / "collapsed_contact_heatmap.png",
        min_prob=args.min_prob,
        tick_every=args.tick_every,
        dpi=args.dpi,
    )

    plot_contact_heatmap(
        noncollapsed_prob,
        sequence=sequence,
        residue_to_motif=residue_to_motif,
        unique_motifs=unique_motifs,
        title=(
            "Non-collapsed-state residue contact probability\n"
            f"Rg ≥ {args.rg_threshold:.2f} nm | "
            f"contact < {args.contact_cutoff:.2f} nm | "
            f"n = {counts['noncollapsed_frames']} frames"
        ),
        output=args.outdir / "noncollapsed_contact_heatmap.png",
        min_prob=args.min_prob,
        tick_every=args.tick_every,
        dpi=args.dpi,
    )

    print("\nFinished.")
    print(f"Collapsed frames:     {counts['collapsed_frames']}")
    print(f"Non-collapsed frames: {counts['noncollapsed_frames']}")
    print(f"Outputs written to:   {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
