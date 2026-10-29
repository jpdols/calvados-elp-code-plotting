import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import self_distance_array


def infer_topology_from_dcd(dcd_path: Path) -> Path:
    top = dcd_path.parent / "top.pdb"
    if not top.exists():
        raise FileNotFoundError(f"Could not find topology file next to trajectory: {top}")
    return top


def parse_run_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("::")

    if len(parts) == 1:
        traj = Path(parts[0]).expanduser().resolve()
        top = infer_topology_from_dcd(traj)
        return traj.stem, top, traj

    if len(parts) == 2:
        label = parts[0].strip()
        traj = Path(parts[1]).expanduser().resolve()
        top = infer_topology_from_dcd(traj)
        return label, top, traj

    if len(parts) == 3:
        label = parts[0].strip()
        top = Path(parts[1]).expanduser().resolve()
        traj = Path(parts[2]).expanduser().resolve()
        return label, top, traj

    raise ValueError(
        f"Could not parse run spec: {spec}\n"
        "Use one of:\n"
        "  traj.dcd\n"
        "  Label::traj.dcd\n"
        "  Label::path/to/top.pdb::path/to/traj.dcd"
    )


def build_condensed_pair_mask(n_atoms: int, exclude_near: int) -> np.ndarray:
    """
    Mask for self_distance_array condensed output.
    Keeps only pairs (i, j) with j - i > exclude_near.
    """
    keep = []
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            keep.append((j - i) > exclude_near)
    return np.array(keep, dtype=bool)


def count_contacts_per_frame(
    topology_file: str,
    trajectory_file: str,
    selection: str,
    cutoff_nm: float,
    stride: int,
    exclude_near: int,
    normalize: str,
) -> tuple[np.ndarray, int]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if cutoff_nm <= 0:
        raise ValueError(f"cutoff_nm must be > 0, got {cutoff_nm}")

    u = mda.Universe(topology_file, trajectory_file)
    sel = u.select_atoms(selection)

    if sel.n_atoms < 2:
        raise ValueError(f"Selection '{selection}' must contain at least 2 atoms, got {sel.n_atoms}")

    n = sel.n_atoms
    cutoff_A = cutoff_nm * 10.0
    pair_mask = build_condensed_pair_mask(n, exclude_near)

    if pair_mask.sum() == 0:
        raise ValueError(
            f"No valid residue pairs left after exclude_near={exclude_near}. "
            f"Selection has only {n} atoms."
        )

    counts = []

    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue

        dists = self_distance_array(sel.positions)
        contact_count = np.count_nonzero((dists < cutoff_A) & pair_mask)

        if normalize == "none":
            value = float(contact_count)
        elif normalize == "N":
            value = float(contact_count) / n
        elif normalize == "pairs":
            value = float(contact_count) / pair_mask.sum()
        else:
            raise ValueError(f"Unknown normalize mode: {normalize}")

        counts.append(value)

    return np.array(counts, dtype=float), n


def plot_contact_histograms(
    results: list[dict],
    bins: int,
    density: bool,
    output_prefix: Path | None,
) -> None:
    plt.figure(figsize=(9, 5))

    for r in results:
        plt.hist(
            r["contacts"],
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            label=r["label"],
        )

    plt.xlabel(r["xlabel"])
    plt.ylabel("Probability density" if density else "Count")
    plt.title("Distribution of contact number per frame")
    plt.legend()
    plt.tight_layout()

    if output_prefix is not None:
        plt.savefig(output_prefix.with_name(output_prefix.stem + "_contact_distribution.png"), dpi=300)


def plot_contact_timeseries(
    results: list[dict],
    output_prefix: Path | None,
) -> None:
    plt.figure(figsize=(10, 5))

    for r in results:
        frames = np.arange(len(r["contacts"]))
        plt.plot(frames, r["contacts"], linewidth=1, label=r["label"])

    plt.xlabel("Frame index after stride")
    plt.ylabel(results[0]["xlabel"])
    plt.title("Contact number per frame")
    plt.legend()
    plt.tight_layout()

    if output_prefix is not None:
        plt.savefig(output_prefix.with_name(output_prefix.stem + "_contact_timeseries.png"), dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot contact-number histograms for multiple trajectories."
    )
    parser.add_argument("--run", action="append", required=True,
                        help='Repeat: "Label::top.pdb::traj.dcd" or "Label::traj.dcd" or "traj.dcd"')
    parser.add_argument("--selection", default="all",
                        help='Atom selection, default "all"')
    parser.add_argument("--cutoff-nm", type=float, default=1.2,
                        help="Contact cutoff in nm (default: 1.2)")
    parser.add_argument("--stride", type=int, default=10,
                        help="Use every nth frame (default: 10)")
    parser.add_argument("--exclude-near", type=int, default=2,
                        help="Exclude pairs with sequence separation <= this value (default: 2)")
    parser.add_argument("--bins", type=int, default=40,
                        help="Number of histogram bins (default: 40)")
    parser.add_argument("--normalize", choices=["none", "N", "pairs"], default="N",
                        help='Normalize contacts as raw count ("none"), contacts/N ("N"), or fraction of valid pairs ("pairs"). Default: N')
    parser.add_argument("--no-density", action="store_true",
                        help="Plot histogram counts instead of probability density")
    parser.add_argument("--save-prefix", default=None,
                        help="Prefix for output PNGs")
    parser.add_argument("--plot-timeseries", action="store_true",
                        help="Also save/show contact-number timeseries")

    args = parser.parse_args()

    results = []

    for spec in args.run:
        label, top, traj = parse_run_spec(spec)

        print(f"Loading: {label}")
        print(f"  Topology:   {top}")
        print(f"  Trajectory: {traj}")
        print(f"  Selection:  {args.selection}")
        print(f"  Cutoff:     {args.cutoff_nm:.3f} nm")
        print(f"  Stride:     {args.stride}")
        print(f"  Exclude |i-j| <= {args.exclude_near}")
        print(f"  Normalize:  {args.normalize}")

        contacts, n_sel = count_contacts_per_frame(
            topology_file=str(top),
            trajectory_file=str(traj),
            selection=args.selection,
            cutoff_nm=args.cutoff_nm,
            stride=args.stride,
            exclude_near=args.exclude_near,
            normalize=args.normalize,
        )

        if args.normalize == "none":
            xlabel = "Contact number per frame"
        elif args.normalize == "N":
            xlabel = "Contacts per residue"
        else:
            xlabel = "Fraction of valid residue pairs in contact"

        print(f"  Selected atoms:      {n_sel}")
        print(f"  Frames after stride: {len(contacts)}")
        print(f"  Mean contacts:       {contacts.mean():.4f}")
        print(f"  Std contacts:        {contacts.std():.4f}")
        print(f"  Min contacts:        {contacts.min():.4f}")
        print(f"  Max contacts:        {contacts.max():.4f}")

        results.append(
            {
                "label": label,
                "contacts": contacts,
                "xlabel": xlabel,
            }
        )

    output_prefix = Path(args.save_prefix) if args.save_prefix else None

    plot_contact_histograms(
        results=results,
        bins=args.bins,
        density=not args.no_density,
        output_prefix=output_prefix,
    )

    if args.plot_timeseries:
        plot_contact_timeseries(results=results, output_prefix=output_prefix)

    plt.show()


if __name__ == "__main__":
    main()