import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


FRAME_TIME_NS = 0.1  # your setup: each saved frame = 0.1 ns


def compute_rg_nm(positions_angstrom: np.ndarray) -> float:
    """
    Compute radius of gyration from Cartesian coordinates in Å.
    Returns Rg in nm.
    """
    positions_nm = positions_angstrom / 10.0
    com = positions_nm.mean(axis=0)
    rg2 = np.mean(np.sum((positions_nm - com) ** 2, axis=1))
    return float(np.sqrt(rg2))


def load_rg_timeseries(
    topology_file: str,
    trajectory_file: str,
    selection: str = "all",
    frame_time_ns: float = FRAME_TIME_NS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trajectory and compute Rg for all frames.
    Time axis is built manually from frame index.
    """
    u = mda.Universe(topology_file, trajectory_file)
    sel = u.select_atoms(selection)

    if sel.n_atoms == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms for {trajectory_file}")

    times_ns = []
    rg_nm = []

    for i, ts in enumerate(u.trajectory):
        times_ns.append(i * frame_time_ns)
        rg_nm.append(compute_rg_nm(sel.positions))

    return np.array(times_ns), np.array(rg_nm)


def save_rg_dat(times_ns: np.ndarray, rg_nm: np.ndarray, output_dat: Path) -> None:
    data = np.column_stack([times_ns, rg_nm])
    np.savetxt(output_dat, data, header="time_ns rg_nm", fmt="%.6f")


def infer_topology_from_dcd(dcd_path: Path) -> Path:
    top = dcd_path.parent / "top.pdb"
    if not top.exists():
        raise FileNotFoundError(f"Could not find topology file next to trajectory: {top}")
    return top


def default_label_from_path(dcd_path: Path) -> str:
    """
    Example:
    .../tests/E_AI_triblock/2.0_310K/E_AI_triblock/E_AI_triblock.dcd
    -> E_AI_triblock | 2.0_310K
    """
    parts = dcd_path.parts
    if len(parts) >= 4:
        seq_name = parts[-4]
        temp = parts[-3]
        return f"{seq_name} | {temp}"
    return dcd_path.stem


def parse_run_spec(spec: str) -> tuple[str, Path, Path]:
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
        "  Label::path/to/top.pdb::path/to/traj.dcd"
    )


def collapsed_stats(rg_nm: np.ndarray, frame_time_ns: float, threshold_nm: float) -> dict:
    mask = rg_nm < threshold_nm
    n_frames = len(rg_nm)
    n_collapsed = int(mask.sum())
    frac_collapsed = n_collapsed / n_frames if n_frames > 0 else 0.0
    time_collapsed_ns = n_collapsed * frame_time_ns
    return {
        "collapsed_mask": mask,
        "collapsed_frames": n_collapsed,
        "fraction_collapsed": frac_collapsed,
        "time_collapsed_ns": time_collapsed_ns,
    }


def plot_overlay_timeseries(results: list[dict], threshold_nm: float, output_png: Path | None = None) -> None:
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.plot(r["times_ns"], r["rg_nm"], linewidth=1, label=r["label"])
    plt.axhline(threshold_nm, linestyle="--", linewidth=1, label=f"Collapsed threshold = {threshold_nm:.1f} nm")
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
    threshold_nm: float = 4.0,
    output_png: Path | None = None,
) -> None:
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.hist(r["rg_nm"], bins=bins, density=True, histtype="step", linewidth=2, label=r["label"])
    plt.axvline(threshold_nm, linestyle="--", linewidth=1, label=f"Collapsed threshold = {threshold_nm:.1f} nm")
    plt.xlabel("Radius of gyration (nm)")
    plt.ylabel("Probability density")
    plt.title("Distribution of radius of gyration")
    plt.legend()
    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_distribution.png"), dpi=300)


def plot_collapsed_time(results: list[dict], threshold_nm: float, output_png: Path | None = None) -> None:
    labels = [r["label"] for r in results]
    times_collapsed = [r["time_collapsed_ns"] for r in results]
    fractions = [100.0 * r["fraction_collapsed"] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, times_collapsed)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Time spent collapsed (ns)")
    plt.title(f"Time spent in collapsed state (Rg < {threshold_nm:.1f} nm)")

    for bar, t_ns, frac in zip(bars, times_collapsed, fractions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t_ns:.1f} ns\n({frac:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_collapsed_time.png"), dpi=300)


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
        "--collapsed-threshold",
        type=float,
        default=4.0,
        help="Rg threshold in nm below which a frame is counted as collapsed (default: 4.0)",
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
            frame_time_ns=FRAME_TIME_NS,
        )

        cstats = collapsed_stats(rg_nm, FRAME_TIME_NS, args.collapsed_threshold)

        print(f"  Frames:   {len(rg_nm)}")
        print(f"  Total t:  {times_ns[-1]:.2f} ns")
        print(f"  Mean Rg:  {rg_nm.mean():.4f} nm")
        print(f"  Std Rg:   {rg_nm.std():.4f} nm")
        print(f"  Min Rg:   {rg_nm.min():.4f} nm")
        print(f"  Max Rg:   {rg_nm.max():.4f} nm")
        print(f"  Collapsed threshold: {args.collapsed_threshold:.2f} nm")
        print(f"  Collapsed frames:    {cstats['collapsed_frames']}")
        print(f"  Fraction collapsed:  {100.0 * cstats['fraction_collapsed']:.2f}%")
        print(f"  Time collapsed:      {cstats['time_collapsed_ns']:.2f} ns")

        results.append(
            {
                "label": label,
                "top": top,
                "traj": traj,
                "times_ns": times_ns,
                "rg_nm": rg_nm,
                "collapsed_frames": cstats["collapsed_frames"],
                "fraction_collapsed": cstats["fraction_collapsed"],
                "time_collapsed_ns": cstats["time_collapsed_ns"],
            }
        )

        if args.save_dat:
            safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
            out_dat = Path(f"rg_{safe_label}.dat")
            save_rg_dat(times_ns, rg_nm, out_dat)
            print(f"  Saved: {out_dat}")

    output_prefix = Path(args.save_prefix) if args.save_prefix else None

    plot_overlay_timeseries(results, args.collapsed_threshold, output_prefix)
    plot_overlay_distributions(results, bins=args.bins, threshold_nm=args.collapsed_threshold, output_png=output_prefix)
    plot_collapsed_time(results, args.collapsed_threshold, output_prefix)
    plt.show()


if __name__ == "__main__":
    main()