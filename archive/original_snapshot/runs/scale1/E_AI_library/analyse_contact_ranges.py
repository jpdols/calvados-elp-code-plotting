from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

def find_replicates(arch_folder):
    arch_folder = Path(arch_folder)
    reps = []
    for repdir in sorted(arch_folder.glob("rep*")):
        sim_dirs = [p for p in repdir.iterdir() if p.is_dir() and p.name.startswith("EAI_")]
        if not sim_dirs:
            continue
        sim = sim_dirs[0]
        top = sim / "top.pdb"
        traj = sim / f"{sim.name}.dcd"
        if not traj.exists():
            dcds = list(sim.glob("*.dcd"))
            traj = dcds[0] if dcds else traj
        if top.exists() and traj.exists():
            reps.append((repdir.name, sim.name, top, traj))
    return reps

def analyse_rep(top, traj, cutoff_nm, stride, max_frames, min_sep):
    u = mda.Universe(str(top), str(traj))
    ag = u.select_atoms("all")
    n = len(ag)

    seqsep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    valid = upper & (seqsep >= min_sep)
    short_mask = valid & (seqsep < 10)
    medium_mask = valid & (seqsep >= 10) & (seqsep < 50)
    long_mask = valid & (seqsep >= 50)

    max_sep = n - 1
    contact_by_sep = np.zeros(max_sep + 1)
    count_by_sep = np.zeros(max_sep + 1)

    short_counts, medium_counts, long_counts, total_counts = [], [], [], []

    cutoff_A = cutoff_nm * 10.0

    frames_done = 0
    for ts in u.trajectory[::stride]:
        if max_frames and frames_done >= max_frames:
            break

        dist = distance_array(ag.positions, ag.positions)
        contacts = dist < cutoff_A

        short = np.sum(contacts & short_mask)
        medium = np.sum(contacts & medium_mask)
        long = np.sum(contacts & long_mask)
        total = np.sum(contacts & valid)

        short_counts.append(short)
        medium_counts.append(medium)
        long_counts.append(long)
        total_counts.append(total)

        sep_vals = seqsep[contacts & valid]
        binc = np.bincount(sep_vals, minlength=max_sep + 1)
        contact_by_sep[:len(binc)] += binc[:max_sep + 1]

        possible_sep_vals = seqsep[valid]
        if frames_done == 0:
            possible_by_sep = np.bincount(possible_sep_vals, minlength=max_sep + 1)

        frames_done += 1

    contact_prob = contact_by_sep / np.maximum(possible_by_sep * frames_done, 1)

    return {
        "short": np.mean(short_counts),
        "medium": np.mean(medium_counts),
        "long": np.mean(long_counts),
        "total": np.mean(total_counts),
        "short_frac": np.mean(short_counts) / np.mean(total_counts),
        "medium_frac": np.mean(medium_counts) / np.mean(total_counts),
        "long_frac": np.mean(long_counts) / np.mean(total_counts),
        "sep": np.arange(max_sep + 1),
        "contact_prob": contact_prob,
        "frames": frames_done,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Architecture folders")
    ap.add_argument("--cutoff", type=float, default=0.8, help="Contact cutoff in nm")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--min-sep", type=int, default=3, help="Ignore contacts closer than this in sequence")
    ap.add_argument("--out", default="contact_range_analysis")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    rows = []
    sep_curves = {}

    for folder in args.folders:
        folder = Path(folder)
        arch = folder.name
        reps = find_replicates(folder)

        if not reps:
            print(f"[SKIP] no complete reps: {arch}")
            continue

        rep_results = []
        sep_probs = []

        for rep_name, sim_name, top, traj in reps:
            print(f"[READ] {arch} {rep_name}")
            r = analyse_rep(top, traj, args.cutoff, args.stride, args.max_frames, args.min_sep)
            rep_results.append(r)
            sep_probs.append(r["contact_prob"])

            rows.append({
                "architecture": arch,
                "replicate": rep_name,
                "short_contacts": r["short"],
                "medium_contacts": r["medium"],
                "long_contacts": r["long"],
                "total_contacts": r["total"],
                "short_fraction": r["short_frac"],
                "medium_fraction": r["medium_frac"],
                "long_fraction": r["long_frac"],
                "frames_used": r["frames"],
            })

        sep_curves[arch] = np.mean(np.vstack(sep_probs), axis=0)

    df = pd.DataFrame(rows)
    df.to_csv(out / "contact_range_replicates.csv", index=False)

    summary = df.groupby("architecture").agg(
        short_mean=("short_contacts", "mean"),
        short_sd=("short_contacts", "std"),
        medium_mean=("medium_contacts", "mean"),
        medium_sd=("medium_contacts", "std"),
        long_mean=("long_contacts", "mean"),
        long_sd=("long_contacts", "std"),
        total_mean=("total_contacts", "mean"),
        total_sd=("total_contacts", "std"),
        short_fraction_mean=("short_fraction", "mean"),
        medium_fraction_mean=("medium_fraction", "mean"),
        long_fraction_mean=("long_fraction", "mean"),
    ).reset_index()

    summary.to_csv(out / "contact_range_summary.csv", index=False)

    # Plot 1: stacked contact fractions
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(summary)), 5))

    ax.bar(x, summary["short_fraction_mean"], label="short |i-j| < 10")
    ax.bar(x, summary["medium_fraction_mean"], bottom=summary["short_fraction_mean"], label="medium 10–50")
    ax.bar(
        x,
        summary["long_fraction_mean"],
        bottom=summary["short_fraction_mean"] + summary["medium_fraction_mean"],
        label="long |i-j| ≥ 50",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["architecture"], rotation=45, ha="right")
    ax.set_ylabel("Fraction of intrachain contacts")
    ax.set_title(f"Short-, medium-, and long-range contact fractions\ncutoff = {args.cutoff} nm")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out / "contact_fraction_stacked.png", dpi=300)
    plt.savefig(out / "contact_fraction_stacked.svg")
    plt.close()

    # Plot 2: long/short ratio
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(summary)), 5))
    ratio = summary["long_mean"] / summary["short_mean"]
    ax.bar(x, ratio)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["architecture"], rotation=45, ha="right")
    ax.set_ylabel("Long / short contact ratio")
    ax.set_title("Relative long-range compaction")
    plt.tight_layout()
    plt.savefig(out / "long_short_contact_ratio.png", dpi=300)
    plt.savefig(out / "long_short_contact_ratio.svg")
    plt.close()

    # Plot 3: contact probability vs sequence separation
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch, prob in sep_curves.items():
        sep = np.arange(len(prob))
        mask = sep >= args.min_sep
        ax.plot(sep[mask], prob[mask], linewidth=1.5, label=arch)

    ax.set_yscale("log")
    ax.set_xlabel("Sequence separation |i-j|")
    ax.set_ylabel("P(contact)")
    ax.set_title(f"Contact probability vs sequence separation\ncutoff = {args.cutoff} nm")
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "contact_probability_vs_sequence_separation.png", dpi=300)
    plt.savefig(out / "contact_probability_vs_sequence_separation.svg")
    plt.close()

    print(f"[OK] results written to: {out.resolve()}")
    print(summary)

if __name__ == "__main__":
    main()