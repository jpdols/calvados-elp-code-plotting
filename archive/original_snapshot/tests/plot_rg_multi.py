import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np


def compute_rg_nm(positions_angstrom: np.ndarray) -> float:
    positions_nm = positions_angstrom / 10.0
    com = positions_nm.mean(axis=0)
    rg2 = np.mean(np.sum((positions_nm - com) ** 2, axis=1))
    return float(np.sqrt(rg2))


def load_rg_timeseries(
    topology_file: str,
    trajectory_file: str,
    selection: str = "all",
    frame_time_ns: float = 0.1,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    if frame_time_ns <= 0:
        raise ValueError(f"frame_time_ns must be > 0, got {frame_time_ns}")

    u = mda.Universe(topology_file, trajectory_file)
    sel = u.select_atoms(selection)

    if sel.n_atoms == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms for {trajectory_file}")

    effective_frame_time_ns = frame_time_ns * stride

    times_ns = []
    rg_nm = []

    out_index = 0
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue
        times_ns.append(out_index * effective_frame_time_ns)
        rg_nm.append(compute_rg_nm(sel.positions))
        out_index += 1

    return np.array(times_ns), np.array(rg_nm), effective_frame_time_ns


def compute_rg_change_stats(rg_nm: np.ndarray, frame_time_ns: float) -> dict:
    if len(rg_nm) < 2:
        empty = np.array([], dtype=float)
        return {
            "rg_change_nm": empty,
            "mean_rg_change_nm": 0.0,
            "std_rg_change_nm": 0.0,
            "mean_rg_change_rate_nm_per_ns": 0.0,
            "std_rg_change_rate_nm_per_ns": 0.0,
        }

    rg_change_nm = np.abs(np.diff(rg_nm))
    rg_change_rate_nm_per_ns = rg_change_nm / frame_time_ns

    return {
        "rg_change_nm": rg_change_nm,
        "mean_rg_change_nm": float(rg_change_nm.mean()),
        "std_rg_change_nm": float(rg_change_nm.std()),
        "mean_rg_change_rate_nm_per_ns": float(rg_change_rate_nm_per_ns.mean()),
        "std_rg_change_rate_nm_per_ns": float(rg_change_rate_nm_per_ns.std()),
    }


def save_rg_dat(times_ns: np.ndarray, rg_nm: np.ndarray, output_dat: Path) -> None:
    data = np.column_stack([times_ns, rg_nm])
    np.savetxt(output_dat, data, header="time_ns rg_nm", fmt="%.6f")


def infer_topology_from_dcd(dcd_path: Path) -> Path:
    top = dcd_path.parent / "top.pdb"
    if not top.exists():
        raise FileNotFoundError(f"Could not find topology file next to trajectory: {top}")
    return top


def default_label_from_path(dcd_path: Path) -> str:
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


def rolling_median(x: np.ndarray, window_frames: int) -> np.ndarray:
    if window_frames <= 1:
        return x.copy()

    half = window_frames // 2
    out = np.empty_like(x)

    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        out[i] = np.median(x[start:end])

    return out


def dwell_times_ns(mask: np.ndarray, frame_time_ns: float) -> np.ndarray:
    dwells = []
    current = 0
    for val in mask:
        if val:
            current += 1
        else:
            if current > 0:
                dwells.append(current * frame_time_ns)
                current = 0
    if current > 0:
        dwells.append(current * frame_time_ns)

    return np.array(dwells, dtype=float)


def collapsed_stats(
    rg_nm: np.ndarray,
    frame_time_ns: float,
    threshold_nm: float,
    min_dwell_ns: float = 0.0,
) -> dict:
    mask = rg_nm < threshold_nm
    n_frames = len(rg_nm)
    n_collapsed = int(mask.sum())
    frac_collapsed = n_collapsed / n_frames if n_frames > 0 else 0.0
    time_collapsed_ns = n_collapsed * frame_time_ns

    dwell_ns = dwell_times_ns(mask, frame_time_ns)

    if min_dwell_ns > 0:
        dwell_ns_filtered = dwell_ns[dwell_ns >= min_dwell_ns]
    else:
        dwell_ns_filtered = dwell_ns

    longest_dwell_ns = float(dwell_ns.max()) if dwell_ns.size > 0 else 0.0
    mean_dwell_ns = float(dwell_ns.mean()) if dwell_ns.size > 0 else 0.0
    n_dwells = int(dwell_ns.size)

    longest_dwell_filtered_ns = float(dwell_ns_filtered.max()) if dwell_ns_filtered.size > 0 else 0.0
    mean_dwell_filtered_ns = float(dwell_ns_filtered.mean()) if dwell_ns_filtered.size > 0 else 0.0
    n_dwells_filtered = int(dwell_ns_filtered.size)

    return {
        "collapsed_mask": mask,
        "collapsed_frames": n_collapsed,
        "fraction_collapsed": frac_collapsed,
        "time_collapsed_ns": time_collapsed_ns,
        "dwell_times_ns": dwell_ns,
        "n_dwells": n_dwells,
        "longest_dwell_ns": longest_dwell_ns,
        "mean_dwell_ns": mean_dwell_ns,
        "dwell_times_filtered_ns": dwell_ns_filtered,
        "n_dwells_filtered": n_dwells_filtered,
        "longest_dwell_filtered_ns": longest_dwell_filtered_ns,
        "mean_dwell_filtered_ns": mean_dwell_filtered_ns,
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


def plot_smoothed_timeseries(
    results: list[dict],
    threshold_nm: float,
    smooth_window_ns: float,
    output_png: Path | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    for r in results:
        window_frames = max(1, int(round(smooth_window_ns / r["effective_frame_time_ns"])))
        plt.plot(r["times_ns"], r["rg_nm"], linewidth=0.8, alpha=0.18)
        smoothed = rolling_median(r["rg_nm"], window_frames)
        plt.plot(r["times_ns"], smoothed, linewidth=2.2, label=r["label"])

    plt.axhline(threshold_nm, linestyle="--", linewidth=1, label=f"Collapsed threshold = {threshold_nm:.1f} nm")
    plt.xlabel("Time (ns)")
    plt.ylabel("Radius of gyration (nm)")
    plt.title(f"Smoothed radius of gyration vs time (rolling median, {smooth_window_ns:.1f} ns)")
    plt.legend()
    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_timeseries_smoothed.png"), dpi=300)


def extract_temperature_from_label(label: str) -> float:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)K", label)
    if match is None:
        raise ValueError(f"Could not extract temperature from label: {label}")
    return float(match.group(1))


def plot_overlay_distributions(
    results: list[dict],
    bins: int = 40,
    threshold_nm: float = 4.0,
    output_png: Path | None = None,
    show_threshold: bool = False,
) -> None:
    plt.figure(figsize=(9, 5))

    temps = np.array([extract_temperature_from_label(r["label"]) for r in results], dtype=float)
    tmin, tmax = temps.min(), temps.max()
    cmap = plt.cm.viridis

    for r, temp in zip(results, temps):
        if tmax > tmin:
            color = cmap((temp - tmin) / (tmax - tmin))
        else:
            color = cmap(0.5)

        plt.hist(
            r["rg_nm"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=r["label"],
        )

    if show_threshold:
        plt.axvline(
            threshold_nm,
            linestyle="--",
            linewidth=1,
            color="black",
            label=f"Collapsed threshold = {threshold_nm:.1f} nm",
        )

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


def plot_dwell_times(
    results: list[dict],
    threshold_nm: float,
    min_dwell_ns: float = 0.0,
    output_png: Path | None = None,
) -> None:
    labels = [r["label"] for r in results]
    longest = [r["longest_dwell_filtered_ns"] for r in results]
    mean_vals = [r["mean_dwell_filtered_ns"] for r in results]
    counts = [r["n_dwells_filtered"] for r in results]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5))
    bars1 = plt.bar(x - width / 2, longest, width=width, label="Longest dwell")
    bars2 = plt.bar(x + width / 2, mean_vals, width=width, label="Mean dwell")

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Residence time (ns)")
    title = f"Dwell times in collapsed state (Rg < {threshold_nm:.1f} nm)"
    if min_dwell_ns > 0:
        title += f"\n(filtered: dwell ≥ {min_dwell_ns:.1f} ns)"
    plt.title(title)
    plt.legend()

    for bar, val, count in zip(bars1, longest, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f} ns\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, val in zip(bars2, mean_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f} ns",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_dwell_times.png"), dpi=300)


def plot_mean_rg_change(results: list[dict], output_png: Path | None = None) -> None:
    labels = [r["label"] for r in results]
    mean_changes = [r["mean_rg_change_nm"] for r in results]
    std_changes = [r["std_rg_change_nm"] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, mean_changes, yerr=std_changes, capsize=5)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Mean |ΔRg| per frame (nm)")
    plt.title("Mean absolute change in Rg per frame")

    for bar, mean_val in zip(bars, mean_changes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_png is not None:
        plt.savefig(output_png.with_name(output_png.stem + "_mean_rg_change.png"), dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay Rg timeseries and distributions for multiple DCD runs."
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument(
        "--frame-time",
        action="append",
        type=float,
        required=True,
        help=(
            "Original time per saved frame in ns for the corresponding --run. "
            "Repeat exactly once per run. Example: 0.01 for oversampled runs, 0.1 for normal runs."
        ),
    )
    parser.add_argument(
        "--stride",
        action="append",
        type=int,
        help="Stride applied during loading for the corresponding --run. Repeat once per run."
    )
    parser.add_argument("--selection", default="all")
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--collapsed-threshold", type=float, default=4.0)
    parser.add_argument("--smooth-window-ns", type=float, default=5.0)
    parser.add_argument("--min-dwell-ns", type=float, default=0.0)
    parser.add_argument("--save-prefix", default=None)
    parser.add_argument("--save-dat", action="store_true")

    args = parser.parse_args()

    n_runs = len(args.run)

        # Allow either:
    # 1 frame-time for all runs
    # OR exactly 1 frame-time per run
    if len(args.frame_time) == 1:
        frame_times = args.frame_time * n_runs
    elif len(args.frame_time) == n_runs:
        frame_times = args.frame_time
    else:
        raise ValueError(
            f"Give either one --frame-time for all runs, or exactly one per --run. "
            f"Got {len(args.frame_time)} frame times for {n_runs} runs."
        )

    # Allow either:
    # no stride -> all runs use stride 1
    # 1 stride -> same stride for all runs
    # OR exactly 1 stride per run
    if args.stride is None:
        strides = [1] * n_runs
    elif len(args.stride) == 1:
        strides = args.stride * n_runs
    elif len(args.stride) == n_runs:
        strides = args.stride
    else:
        raise ValueError(
            f"Give either one --stride for all runs, or exactly one per --run. "
            f"Got {len(args.stride)} strides for {n_runs} runs."
        )

    results = []

    for spec, frame_time_ns, stride in zip(args.run, frame_times, strides):
        label, top, traj = parse_run_spec(spec)

        print(f"Loading: {label}")
        print(f"  Topology:   {top}")
        print(f"  Trajectory: {traj}")
        print(f"  Original dt/frame:   {frame_time_ns:.4f} ns")
        print(f"  Stride used during loading: {stride}")

        times_ns, rg_nm, effective_frame_time_ns = load_rg_timeseries(
            topology_file=str(top),
            trajectory_file=str(traj),
            selection=args.selection,
            frame_time_ns=frame_time_ns,
            stride=stride,
        )

        cstats = collapsed_stats(
            rg_nm,
            effective_frame_time_ns,
            args.collapsed_threshold,
            min_dwell_ns=args.min_dwell_ns,
        )

        rg_change_stats = compute_rg_change_stats(rg_nm, effective_frame_time_ns)

        total_time_ns = times_ns[-1] if len(times_ns) > 0 else 0.0

        print(f"  Frames after stride: {len(rg_nm)}")
        print(f"  Effective dt/frame:  {effective_frame_time_ns:.4f} ns")
        print(f"  Total t:             {total_time_ns:.2f} ns")
        print(f"  Mean Rg:             {rg_nm.mean():.4f} nm")
        print(f"  Std Rg:              {rg_nm.std():.4f} nm")
        print(f"  Min Rg:              {rg_nm.min():.4f} nm")
        print(f"  Max Rg:              {rg_nm.max():.4f} nm")
        print(f"  Mean |ΔRg|/frame:    {rg_change_stats['mean_rg_change_nm']:.4f} nm")
        print(f"  Std |ΔRg|/frame:     {rg_change_stats['std_rg_change_nm']:.4f} nm")
        print(f"  Mean |ΔRg|/ns:       {rg_change_stats['mean_rg_change_rate_nm_per_ns']:.4f} nm/ns")
        print(f"  Std |ΔRg|/ns:        {rg_change_stats['std_rg_change_rate_nm_per_ns']:.4f} nm/ns")
        print(f"  Collapsed threshold: {args.collapsed_threshold:.2f} nm")
        print(f"  Collapsed frames:    {cstats['collapsed_frames']}")
        print(f"  Fraction collapsed:  {100.0 * cstats['fraction_collapsed']:.2f}%")
        print(f"  Time collapsed:      {cstats['time_collapsed_ns']:.2f} ns")
        print(f"  Number of collapsed dwells: {cstats['n_dwells']}")
        print(f"  Longest dwell:       {cstats['longest_dwell_ns']:.2f} ns")
        print(f"  Mean dwell:          {cstats['mean_dwell_ns']:.2f} ns")

        if args.min_dwell_ns > 0:
            print(f"  Filtered dwells >= {args.min_dwell_ns:.2f} ns: {cstats['n_dwells_filtered']}")
            print(f"  Longest filtered dwell: {cstats['longest_dwell_filtered_ns']:.2f} ns")
            print(f"  Mean filtered dwell:    {cstats['mean_dwell_filtered_ns']:.2f} ns")

        results.append(
            {
                "label": label,
                "times_ns": times_ns,
                "rg_nm": rg_nm,
                "effective_frame_time_ns": effective_frame_time_ns,
                "collapsed_frames": cstats["collapsed_frames"],
                "fraction_collapsed": cstats["fraction_collapsed"],
                "time_collapsed_ns": cstats["time_collapsed_ns"],
                "longest_dwell_filtered_ns": cstats["longest_dwell_filtered_ns"],
                "mean_dwell_filtered_ns": cstats["mean_dwell_filtered_ns"],
                "n_dwells_filtered": cstats["n_dwells_filtered"],
                "mean_rg_change_nm": rg_change_stats["mean_rg_change_nm"],
                "std_rg_change_nm": rg_change_stats["std_rg_change_nm"],
                "mean_rg_change_rate_nm_per_ns": rg_change_stats["mean_rg_change_rate_nm_per_ns"],
                "std_rg_change_rate_nm_per_ns": rg_change_stats["std_rg_change_rate_nm_per_ns"],
            }
        )

        if args.save_dat:
            safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
            out_dat = Path(f"rg_{safe_label}.dat")
            save_rg_dat(times_ns, rg_nm, out_dat)
            print(f"  Saved: {out_dat}")

    output_prefix = Path(args.save_prefix) if args.save_prefix else None

    plot_overlay_timeseries(results, args.collapsed_threshold, output_prefix)
    plot_smoothed_timeseries(results, args.collapsed_threshold, args.smooth_window_ns, output_prefix)
    plot_overlay_distributions(
	results,
    	bins=args.bins,
    	threshold_nm=args.collapsed_threshold,
    	output_png=output_prefix,
    	show_threshold=False,
    )
    plot_collapsed_time(results, args.collapsed_threshold, output_prefix)
    plot_dwell_times(results, args.collapsed_threshold, args.min_dwell_ns, output_png=output_prefix)
    plot_mean_rg_change(results, output_prefix)

    plt.show()


if __name__ == "__main__":
    main()