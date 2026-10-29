#!/usr/bin/env python
"""
analyse_reference_architecture_370K.py

Compare E_AI, E_AI_triblock, E_AE and E_AE_triblock at 370 K.

For every discovered trajectory:
  - calculate Rg for every analysed frame
  - define compact state as Rg < threshold
  - central 90% Rg range = P95(Rg) - P5(Rg)
  - k_collapse = N(expanded -> compact) / total expanded-state time
  - k_escape   = N(compact -> expanded) / total compact-state time
  - mean compact-state Rg

Replicates:
  - searches recursively below each sequence's 370K directory
  - discovers all DCDs, including rep01/rep02/rep03
  - if multiple DCDs occur in the same run directory, keeps the largest one
  - assigns replicate labels from repXX/runXX/seedXX when possible

Outputs:
  reference_370K_per_trajectory.csv
  reference_370K_summary.csv
  reference_370K_architecture_comparison.pdf/png
  reference_370K_replicate_mapping.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import polymer
except ImportError:
    mda = None


SEQUENCES = ["E_AI", "E_AI_triblock", "E_AE", "E_AE_triblock"]


def natural_key(s):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", str(s))]


def infer_replicate(path):
    s = str(path)
    patterns = [
        r"(?:replicate|rep)[_\-\s]?0*(\d+)",
        r"(?:run)[_\-\s]?0*(\d+)",
        r"(?:seed)[_\-\s]?0*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.I)
        if m:
            return int(m.group(1))
    return None


def find_topology(dcd, sequence_root):
    """Prefer topology near DCD, then search sequence root."""
    candidates = []
    for directory in [dcd.parent, dcd.parent.parent, sequence_root]:
        if directory.exists():
            for name in ["top.pdb", "topology.pdb", f"{sequence_root.name}.pdb"]:
                p = directory / name
                if p.exists():
                    candidates.append(p)
            candidates.extend(directory.glob("*.pdb"))
    if not candidates:
        raise FileNotFoundError(f"No PDB topology found for {dcd}")
    # Prefer top.pdb, then shortest path distance/name.
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda p: (p.name.lower() != "top.pdb", len(str(p)), natural_key(str(p))))
    return candidates[0]


def discover_trajectories(root, sequence, temperature="370K"):
    seq_temp = root / sequence / "6.0" / temperature
    if not seq_temp.exists():
        raise FileNotFoundError(f"Missing directory: {seq_temp}")

    dcds = sorted(seq_temp.rglob("*.dcd"), key=lambda p: natural_key(str(p)))

    # Exclude special-condition trajectories that are not part of the reference comparison.
    dcds = [
        p for p in dcds
        if "lowsalt" not in {part.lower() for part in p.parts}
    ]

    # For E_AI_triblock, use ONLY the explicit rep01/rep02/rep03 runs.
    # The separate unlabelled 370K trajectory is intentionally excluded.
    if sequence == "E_AI_triblock":
        dcds = [
            p for p in dcds
            if any(re.fullmatch(r"rep0?[123]", part, flags=re.I) for part in p.parts)
        ]

    if not dcds:
        raise FileNotFoundError(f"No eligible DCDs found below {seq_temp}")

    # One trajectory per parent/run directory. If segmented/duplicate files exist
    # in the same directory, retain the largest DCD.
    by_parent = {}
    for dcd in dcds:
        parent = dcd.parent
        if parent not in by_parent or dcd.stat().st_size > by_parent[parent].stat().st_size:
            by_parent[parent] = dcd
    dcds = sorted(by_parent.values(), key=lambda p: natural_key(str(p)))

    inferred = [infer_replicate(p) for p in dcds]
    if all(x is not None for x in inferred) and len(set(inferred)) == len(inferred):
        reps = inferred
        sources = ["path label"] * len(dcds)
    else:
        reps = list(range(1, len(dcds) + 1))
        sources = ["natural path order"] * len(dcds)

    rows = []
    for dcd, rep, source in zip(dcds, reps, sources):
        top = find_topology(dcd, seq_temp)
        rows.append({
            "sequence": sequence,
            "replicate": int(rep),
            "replicate_source": source,
            "topology_path": str(top),
            "trajectory_path": str(dcd),
        })
    return rows


def calc_rg_manual(atomgroup):
    """Radius of gyration using masses if available; equal weights otherwise."""
    xyz = atomgroup.positions.astype(np.float64)
    try:
        masses = np.asarray(atomgroup.masses, dtype=float)
        if len(masses) != len(xyz) or not np.all(np.isfinite(masses)) or masses.sum() <= 0:
            raise ValueError
    except Exception:
        masses = np.ones(len(xyz), dtype=float)

    com = np.average(xyz, axis=0, weights=masses)
    dr2 = np.sum((xyz - com) ** 2, axis=1)
    return np.sqrt(np.average(dr2, weights=masses)) / 10.0  # Angstrom -> nm


def analyse_trajectory(topology, trajectory, threshold_nm, frame_dt_ns, stride):
    if mda is None:
        raise ImportError(
            "MDAnalysis is required. Install it in the CALVADOS analysis environment."
        )

    u = mda.Universe(str(topology), str(trajectory))
    ag = u.atoms

    rg = []
    original_frames = []
    for ts in u.trajectory[::stride]:
        rg.append(calc_rg_manual(ag))
        original_frames.append(int(ts.frame))

    rg = np.asarray(rg, dtype=float)
    if rg.size < 2:
        raise ValueError(f"Too few analysed frames in {trajectory}")

    compact = rg < threshold_nm

    # Transitions between consecutive ANALYSED frames.
    # With stride=1 this is the native trajectory transition count.
    n_collapse = int(np.sum((~compact[:-1]) & compact[1:]))
    n_escape = int(np.sum(compact[:-1] & (~compact[1:])))

    analysed_dt_ns = frame_dt_ns * stride
    compact_time_ns = float(compact.sum() * analysed_dt_ns)
    expanded_time_ns = float((~compact).sum() * analysed_dt_ns)

    k_collapse = n_collapse / expanded_time_ns if expanded_time_ns > 0 else np.nan
    k_escape = n_escape / compact_time_ns if compact_time_ns > 0 else np.nan

    p5, p95 = np.percentile(rg, [5, 95])
    mean_compact_rg = float(np.mean(rg[compact])) if np.any(compact) else np.nan

    return {
        "n_frames": int(rg.size),
        "first_original_frame": int(original_frames[0]),
        "last_original_frame": int(original_frames[-1]),
        "mean_rg_nm": float(np.mean(rg)),
        "median_rg_nm": float(np.median(rg)),
        "rg_p05_nm": float(p5),
        "rg_p95_nm": float(p95),
        "rg_range90_nm": float(p95 - p5),
        "fraction_compact": float(np.mean(compact)),
        "mean_rg_compact_nm": mean_compact_rg,
        "n_collapse_transitions": n_collapse,
        "n_escape_transitions": n_escape,
        "compact_time_ns": compact_time_ns,
        "expanded_time_ns": expanded_time_ns,
        "k_collapse_per_ns": float(k_collapse),
        "k_escape_per_ns": float(k_escape),
    }


def summarize(df):
    metrics = [
        "rg_range90_nm",
        "k_collapse_per_ns",
        "k_escape_per_ns",
        "mean_rg_compact_nm",
        "fraction_compact",
        "mean_rg_nm",
    ]
    rows = []
    for seq in SEQUENCES:
        d = df[df["sequence"] == seq]
        if d.empty:
            continue
        row = {
            "sequence": seq,
            "architecture": "triblock" if "triblock" in seq else "multiblock",
            "n_trajectories": len(d),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = d[metric].mean()
            row[f"{metric}_sd"] = d[metric].std(ddof=1) if len(d) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(df, outdir):
    order = ["E_AI", "E_AI_triblock", "E_AE", "E_AE_triblock"]
    display = {
        "E_AI": r"$E_{AI}$",
        "E_AI_triblock": r"$E_{AI}$ triblock",
        "E_AE": r"$E_{AE}$",
        "E_AE_triblock": r"$E_{AE}$ triblock",
    }
    panels = [
        ("rg_range90_nm", r"Central 90% $R_g$ range (nm)", "A"),
        ("k_collapse_per_ns", r"$k_{\mathrm{collapse}}$ (ns$^{-1}$)", "B"),
        ("k_escape_per_ns", r"$k_{\mathrm{escape}}$ (ns$^{-1}$)", "C"),
        ("mean_rg_compact_nm", r"Mean compact-state $R_g$ (nm)", "D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    axes = axes.ravel()
    x = np.arange(len(order), dtype=float)

    # Small deterministic offsets make replicate points visible.
    offsets = {
        1: [0.0],
        2: [-0.045, 0.045],
        3: [-0.065, 0.0, 0.065],
        4: [-0.09, -0.03, 0.03, 0.09],
    }

    for ax, (metric, ylabel, letter) in zip(axes, panels):
        for i, seq in enumerate(order):
            d = df[df["sequence"] == seq].sort_values("replicate")
            if d.empty:
                continue

            vals = d[metric].to_numpy(float)
            n = len(vals)
            offs = offsets.get(n, np.linspace(-0.09, 0.09, n))
            ax.scatter(
                x[i] + np.asarray(offs),
                vals,
                s=34,
                alpha=0.75,
                zorder=3,
            )

            mean = np.nanmean(vals)
            ax.scatter(
                [x[i]], [mean],
                s=85,
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
                zorder=4,
            )

            if np.sum(np.isfinite(vals)) > 1:
                sd = np.nanstd(vals, ddof=1)
                ax.errorbar(
                    x[i], mean, yerr=sd,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.1,
                    capsize=3,
                    zorder=2,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([display[s] for s in order], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.text(
            -0.12, 1.03, letter,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "reference_370K_architecture_comparison.pdf", bbox_inches="tight")
    fig.savefig(outdir / "reference_370K_architecture_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root containing E_AI, E_AI_triblock, E_AE, E_AE_triblock")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--temperature", default="370K")
    ap.add_argument("--threshold", type=float, default=4.61,
                    help="Compact-state Rg threshold in nm (default: 4.61)")
    ap.add_argument("--frame-dt-ns", type=float, default=0.01,
                    help="Native trajectory frame interval in ns (default: 0.01)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Analysis stride. Use 1 for transition kinetics.")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mapping = []
    for seq in SEQUENCES:
        try:
            mapping.extend(discover_trajectories(root, seq, args.temperature))
        except Exception as e:
            print(f"WARNING: {seq}: {e}")

    if not mapping:
        raise RuntimeError("No trajectories discovered.")

    mapping_df = pd.DataFrame(mapping).sort_values(["sequence", "replicate"])
    mapping_df.to_csv(outdir / "reference_370K_replicate_mapping.csv", index=False)

    print("\nDiscovered trajectories:")
    for _, r in mapping_df.iterrows():
        print(f"  {r['sequence']:16s} rep{int(r['replicate']):02d}  {r['trajectory_path']}")

    results = []
    for _, r in mapping_df.iterrows():
        print(f"\nAnalysing {r['sequence']} rep{int(r['replicate']):02d} ...")
        try:
            metrics = analyse_trajectory(
                r["topology_path"],
                r["trajectory_path"],
                args.threshold,
                args.frame_dt_ns,
                args.stride,
            )
            results.append({
                "sequence": r["sequence"],
                "architecture": "triblock" if "triblock" in r["sequence"] else "multiblock",
                "replicate": int(r["replicate"]),
                "temperature_K": int(re.sub(r"\D", "", args.temperature)),
                "threshold_nm": args.threshold,
                "frame_dt_ns": args.frame_dt_ns,
                "stride": args.stride,
                "topology_path": r["topology_path"],
                "trajectory_path": r["trajectory_path"],
                **metrics,
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    if not results:
        raise RuntimeError("All trajectory analyses failed.")

    per = pd.DataFrame(results).sort_values(["sequence", "replicate"])
    per.to_csv(outdir / "reference_370K_per_trajectory.csv", index=False)

    summary = summarize(per)
    summary.to_csv(outdir / "reference_370K_summary.csv", index=False)

    make_figure(per, outdir)

    print("\nPer-trajectory results:")
    print(per[
        ["sequence", "replicate", "rg_range90_nm",
         "k_collapse_per_ns", "k_escape_per_ns", "mean_rg_compact_nm",
         "fraction_compact"]
    ].to_string(index=False))

    print("\nSequence summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")
    print("  reference_370K_per_trajectory.csv")
    print("  reference_370K_summary.csv")
    print("  reference_370K_replicate_mapping.csv")
    print("  reference_370K_architecture_comparison.pdf")
    print("  reference_370K_architecture_comparison.png")


if __name__ == "__main__":
    main()
