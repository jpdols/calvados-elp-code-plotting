#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda

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

def calc_rg(top, traj, stride, frame_time):
    u = mda.Universe(str(top), str(traj))
    ag = u.select_atoms("all")
    rgs = []
    times = []
    for i, ts in enumerate(u.trajectory[::stride]):
        rgs.append(ag.radius_of_gyration() / 10.0)  # Å to nm
        times.append(i * stride * frame_time)
    return np.array(times), np.array(rgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Architecture folder(s), e.g. EAI_arch01_blockshuffle")
    ap.add_argument("--collapsed-threshold", type=float, default=3.5, help="Rg threshold in nm")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--frame-time", type=float, default=0.05, help="ns per raw frame")
    ap.add_argument("--out", default="analysis_replicates")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    summary_rows = []

    for folder in args.folders:
        folder = Path(folder)
        arch = folder.name
        reps = find_replicates(folder)

        if len(reps) == 0:
            print(f"[SKIP] No complete replicates found in {folder}")
            continue

        rep_metrics = []

        plt.figure(figsize=(8, 4.8))

        for rep_name, sim_name, top, traj in reps:
            print(f"[READ] {arch} {rep_name}")
            t, rg = calc_rg(top, traj, args.stride, args.frame_time)

            mean_rg = rg.mean()
            std_rg = rg.std()
            frac_collapsed = 100 * np.mean(rg < args.collapsed_threshold)

            rep_metrics.append({
                "architecture": arch,
                "replicate": rep_name,
                "mean_Rg_nm": mean_rg,
                "std_Rg_nm": std_rg,
                "min_Rg_nm": rg.min(),
                "max_Rg_nm": rg.max(),
                "fraction_collapsed_percent": frac_collapsed,
                "n_frames": len(rg),
                "trajectory_ns": t[-1] if len(t) else 0,
            })

            plt.plot(t, rg, alpha=0.35, linewidth=1.0, label=rep_name)

        plt.axhline(args.collapsed_threshold, linestyle="--", linewidth=1.5, label=f"collapsed threshold = {args.collapsed_threshold} nm")
        plt.xlabel("Time (ns)")
        plt.ylabel("Rg (nm)")
        plt.title(f"{arch}: Rg time series, {len(reps)} replicates")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out / f"{arch}_Rg_timeseries.png", dpi=300)
        plt.close()

        df = pd.DataFrame(rep_metrics)
        df.to_csv(out / f"{arch}_replicate_metrics.csv", index=False)

        mean_frac = df["fraction_collapsed_percent"].mean()
        std_frac = df["fraction_collapsed_percent"].std(ddof=1) if len(df) > 1 else 0.0
        mean_rg_all = df["mean_Rg_nm"].mean()
        std_rg_all = df["mean_Rg_nm"].std(ddof=1) if len(df) > 1 else 0.0

        summary_rows.append({
            "architecture": arch,
            "n_replicates": len(df),
            "mean_fraction_collapsed_percent": mean_frac,
            "std_fraction_collapsed_percent": std_frac,
            "mean_Rg_nm": mean_rg_all,
            "std_Rg_nm": std_rg_all,
        })

        # Single architecture plot: raw points behind mean ± SD
        x = 0
        jitter = np.linspace(-0.08, 0.08, len(df))

        plt.figure(figsize=(4.2, 5.0))
        plt.scatter(
            x + jitter,
            df["fraction_collapsed_percent"],
            alpha=0.35,
            s=70,
            label="replicates"
        )
        plt.errorbar(
            x,
            mean_frac,
            yerr=std_frac,
            fmt="o",
            markersize=10,
            capsize=6,
            linewidth=2,
            label="mean ± SD"
        )
        plt.xticks([0], [arch], rotation=20, ha="right")
        plt.ylabel("Fraction collapsed (%)")
        plt.title(f"{arch}\ncollapsed if Rg < {args.collapsed_threshold} nm")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out / f"{arch}_collapsed_mean_sd.png", dpi=300)
        plt.close()

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(out / "summary_architectures.csv", index=False)

        # Multi-architecture summary plot
        plt.figure(figsize=(max(6, 0.55 * len(sdf)), 5))
        xs = np.arange(len(sdf))

        for i, row in sdf.iterrows():
            arch_df = pd.read_csv(out / f"{row['architecture']}_replicate_metrics.csv")
            jitter = np.linspace(-0.08, 0.08, len(arch_df))
            plt.scatter(
                np.full(len(arch_df), i) + jitter,
                arch_df["fraction_collapsed_percent"],
                alpha=0.30,
                s=55
            )

        plt.errorbar(
            xs,
            sdf["mean_fraction_collapsed_percent"],
            yerr=sdf["std_fraction_collapsed_percent"],
            fmt="o",
            markersize=8,
            capsize=5,
            linewidth=2
        )
        plt.xticks(xs, sdf["architecture"], rotation=45, ha="right")
        plt.ylabel("Fraction collapsed (%)")
        plt.title(f"Collapsed fraction across architectures\nthreshold Rg < {args.collapsed_threshold} nm")
        plt.tight_layout()
        plt.savefig(out / "all_architectures_collapsed_mean_sd.png", dpi=300)
        plt.close()

        print(f"\n[OK] Results written to: {out.resolve()}")
        print(sdf)

if __name__ == "__main__":
    main()