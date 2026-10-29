import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


def compute_rg_nm(positions_angstrom: np.ndarray) -> float:
    """
    Compute radius of gyration from Cartesian coordinates.

    Parameters
    ----------
    positions_angstrom : np.ndarray
        Coordinates in Å, shape (N, 3).

    Returns
    -------
    float
        Radius of gyration in nm.
    """
    positions_nm = positions_angstrom / 10.0  # Å -> nm
    com = positions_nm.mean(axis=0)
    rg2 = np.mean(np.sum((positions_nm - com) ** 2, axis=1))
    return float(np.sqrt(rg2))


def load_rg_timeseries(
    topology_file: str,
    trajectory_file: str,
    selection: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trajectory and compute Rg for all frames.

    Returns
    -------
    times_ns : np.ndarray
    rg_nm : np.ndarray
    """
    u = mda.Universe(topology_file, trajectory_file)
    sel = u.select_atoms(selection)

    if sel.n_atoms == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms for {trajectory_file}")

    times_ns = []
    rg_nm = []

    for ts in u.trajectory:
        times_ns.append(ts.time / 1000.0)  # ps -> ns
        rg_nm.append(compute_rg_nm(sel.positions))

    return np.array(times_ns), np.array(rg_nm)


def save_rg_dat(times_ns: np.ndarray, rg_nm: np.ndarray, output_dat: Path) -> None:
    data = np.column_stack([times_ns, rg_nm])
    np.savetxt(output_dat, data, header="time_ns rg_nm", fmt="%.6f")


def infer_topology_from_dcd(dcd_path: Path) -> Path:
    """
    Assume topology is in the same folder and named top.pdb.
    """
    top = dcd_path.parent / "top.pdb"
    if not top.exists():
        raise FileNotFoundError(
            f"Could not find topology file next to trajectory: {top}"
        )
    return top


def default_label_from_path(dcd_path: Path) -> str:
    """
    Make a readable label from folder structure.
    Example:
    E:/.../E_AI_triblock/310K/E_AI_triblock/E_AI_triblock.dcd
    -> E_AI_triblock | 310K
    """
    parts = dcd_path.parts
    if len(parts) >= 4:
        try:
            seq_name = parts[-4]
            temp = parts[-3]
            return f"{seq_name} | {temp}"
        except Exception:
            pass
    return dcd_path.stem


def parse_run_spec(spec: str) -> tuple[str, Path, Path]:
    """
    Parse a run specification.

    Accepted formats:
    1) "path/to/traj.dcd"
       -> label inferred, topology assumed to be sibling top.pdb

    2) "Label::path/to/traj.dcd"
       -> custom label, topology assumed to be sibling top.pdb

    3) "Label::path/to/top.pdb::path/to/traj.dcd"
       -> custom label, explicit topology and trajectory
    """
    parts = spec.split("::")

    if len(parts) == 1:
        traj = Path(parts[0]).expanduser().resolve()
        top = infer_topology_from_dcd(traj)
        label = default_label_from_path(traj)
        return label, top, traj

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
        "  Label::top.pdb::traj.dcd"
    )


def plot_overlay_timeseries(results: list[dict], output_png: Path | None = None) -> None:
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.plot(r["times_ns"], r["rg_nm"], linewidth=1, label=r["label"])
    plt.xlabel("Time (ns)")
    plt.ylabel("Radius of gyration (nm)")
    plt.title("Radius of gyration vs time")
    plt.legend()
    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_timeseries.png"), dpi=300)


def plot_overlay_distributions(
    results: list[dict],
    bins: int = 40,
    output_png: Path | None = None,
) -> None:
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.hist(r["rg_nm"], bins=bins, density=True, histtype="step", linewidth=2, label=r["label"])
    plt.xlabel("Radius of gyration (nm)")
    plt.ylabel("Probability density")
    plt.title("Distribution of radius of gyration")
    plt.legend()
    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_distribution.png"), dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Rg timeseries and distributions for multiple DCD runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run specification. Repeat this argument for multiple runs.\n"
            "Formats:\n"
            "  path/to/traj.dcd\n"
            "  Label::path/to/traj.dcd\n"
            "  Label::path/to/top.pdb::path/to/traj.dcd"
        ),
    )
    parser.add_argument(
        "--selection",
        default="all",
        help="MDAnalysis selection, default: all",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bins, default: 40",
    )
    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Optional prefix for saving plots, e.g. rg_compare",
    )
    parser.add_argument(
        "--save-dat",
        action="store_true",
        help="Also save an rg_<label>.dat file for each run",
    )

    args = parser.parse_args()

    results = []

    for spec in args.run:
        label, top, traj = parse_run_spec(spec)

        print(f"Loading: {label}")
        print(f"  Topology:   {top}")
        print(f"  Trajectory: {traj}")

        times_ns, rg_nm = load_rg_timeseries(
            topology_file=str(top),
            trajectory_file=str(traj),
            selection=args.selection,
        )

        print(f"  Frames:   {len(rg_nm)}")
        print(f"  Mean Rg:  {rg_nm.mean():.4f} nm")
        print(f"  Std Rg:   {rg_nm.std():.4f} nm")
        print(f"  Min Rg:   {rg_nm.min():.4f} nm")
        print(f"  Max Rg:   {rg_nm.max():.4f} nm")

        results.append(
            {
                "label": label,
                "top": top,
                "traj": traj,
                "times_ns": times_ns,
                "rg_nm": rg_nm,
            }
        )

        if args.save_dat:
            safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
            out_dat = Path(f"rg_{safe_label}.dat")
            save_rg_dat(times_ns, rg_nm, out_dat)
            print(f"  Saved: {out_dat}")

    output_prefix = Path(args.save_prefix) if args.save_prefix else None

    plot_overlay_timeseries(results, output_prefix)
    plot_overlay_distributions(results, bins=args.bins, output_png=output_prefix)
    plt.show()


if __name__ == "__main__":
    main()