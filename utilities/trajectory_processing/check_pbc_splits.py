import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np


def load_expected_bondlength_nm(residue_csv: str, sequence: str) -> np.ndarray:
    """
    Read CALVADOS residue table and build expected bond lengths (nm)
    for consecutive residues in the sequence.

    Assumes residue CSV has columns:
        one, bondlength
    """
    import csv

    bond_lookup = {}
    with open(residue_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bond_lookup[row["one"]] = float(row["bondlength"])

    seq = sequence.strip().upper()
    if len(seq) < 2:
        raise ValueError("Sequence must have at least 2 residues.")

    # Expected bond length for bond i-(i+1):
    # use average of the two residue bondlength values
    expected = []
    for a, b in zip(seq[:-1], seq[1:]):
        if a not in bond_lookup or b not in bond_lookup:
            raise KeyError(f"Residue {a} or {b} not found in residue CSV.")
        expected.append(0.5 * (bond_lookup[a] + bond_lookup[b]))
    return np.array(expected, dtype=float)


def read_sequence_from_fasta(fasta_file: str, name: str | None = None) -> str:
    """
    Read sequence from FASTA. If name is given, match that header.
    Otherwise use the first sequence.
    """
    seqs = {}
    current_name = None
    current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    seqs[current_name] = "".join(current_seq)
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        seqs[current_name] = "".join(current_seq)

    if not seqs:
        raise ValueError(f"No sequences found in FASTA: {fasta_file}")

    if name is None:
        return next(iter(seqs.values()))

    if name not in seqs:
        available = ", ".join(seqs.keys())
        raise KeyError(f"Header '{name}' not found. Available: {available}")

    return seqs[name]


def count_split_frames(
    topology_file: str,
    trajectory_file: str,
    sequence: str,
    residue_csv: str,
    selection: str = "all",
    factor: float = 3.0,
    absolute_threshold_nm: float | None = None,
) -> dict:
    """
    Flag a frame as split if any consecutive-bead distance exceeds:
        factor * expected_bondlength
    or, if absolute_threshold_nm is provided:
        absolute_threshold_nm

    Returns summary info and per-frame details.
    """
    u = mda.Universe(topology_file, trajectory_file)
    sel = u.select_atoms(selection)

    n = sel.n_atoms
    if n != len(sequence):
        raise ValueError(
            f"Selected atom count ({n}) does not match sequence length ({len(sequence)})."
        )

    expected = load_expected_bondlength_nm(residue_csv, sequence)

    split_frames = []
    max_bond_per_frame = []

    for ts in u.trajectory:
        pos_nm = sel.positions / 10.0  # Å -> nm
        diffs = pos_nm[1:] - pos_nm[:-1]
        dists = np.linalg.norm(diffs, axis=1)

        if absolute_threshold_nm is not None:
            suspicious = dists > absolute_threshold_nm
        else:
            suspicious = dists > (factor * expected)

        max_dist = float(dists.max())
        max_bond_per_frame.append((ts.frame, ts.time / 1000.0, max_dist))

        if np.any(suspicious):
            bad_indices = np.where(suspicious)[0]
            split_frames.append(
                {
                    "frame": int(ts.frame),
                    "time_ns": float(ts.time / 1000.0),
                    "n_suspicious_bonds": int(len(bad_indices)),
                    "bond_indices": bad_indices.tolist(),
                    "max_bond_nm": max_dist,
                }
            )

    return {
        "n_total_frames": len(u.trajectory),
        "n_split_frames": len(split_frames),
        "fraction_split": len(split_frames) / len(u.trajectory) if len(u.trajectory) > 0 else 0.0,
        "split_frames": split_frames,
        "max_bond_per_frame": max_bond_per_frame,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check how often a trajectory appears split across PBC."
    )
    parser.add_argument("--topology", required=True, help="Topology file, e.g. top.pdb")
    parser.add_argument("--trajectory", required=True, help="Trajectory file, e.g. traj.dcd")
    parser.add_argument("--fasta", required=True, help="FASTA file used for the run")
    parser.add_argument("--residues", required=True, help="CALVADOS residue CSV with bondlength column")
    parser.add_argument("--header", default=None, help="FASTA header to select")
    parser.add_argument("--selection", default="all", help="Atom selection, default: all")
    parser.add_argument(
        "--factor",
        type=float,
        default=3.0,
        help="Flag if bonded distance > factor * expected bondlength (default 3.0)",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=None,
        help="Optional absolute distance threshold in nm, overrides factor-based check",
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help="Optional output text file for flagged frames",
    )

    args = parser.parse_args()

    sequence = read_sequence_from_fasta(args.fasta, args.header)

    result = count_split_frames(
        topology_file=args.topology,
        trajectory_file=args.trajectory,
        sequence=sequence,
        residue_csv=args.residues,
        selection=args.selection,
        factor=args.factor,
        absolute_threshold_nm=args.absolute_threshold,
    )

    print(f"Total frames:   {result['n_total_frames']}")
    print(f"Split frames:   {result['n_split_frames']}")
    print(f"Fraction split: {result['fraction_split']:.4f}")

    if result["split_frames"]:
        print("\nFirst flagged frames:")
        for row in result["split_frames"][:10]:
            print(
                f"  frame={row['frame']:5d}  "
                f"time_ns={row['time_ns']:8.3f}  "
                f"n_bad_bonds={row['n_suspicious_bonds']:3d}  "
                f"max_bond_nm={row['max_bond_nm']:.3f}"
            )
    else:
        print("\nNo suspicious split frames detected.")

    if args.save_report:
        out = Path(args.save_report)
        with out.open("w") as f:
            f.write("# frame time_ns n_suspicious_bonds max_bond_nm bond_indices\n")
            for row in result["split_frames"]:
                f.write(
                    f"{row['frame']} "
                    f"{row['time_ns']:.6f} "
                    f"{row['n_suspicious_bonds']} "
                    f"{row['max_bond_nm']:.6f} "
                    f"{','.join(map(str, row['bond_indices']))}\n"
                )
        print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()