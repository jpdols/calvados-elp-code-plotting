#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_distance_array


def parse_temp_groups(items):
    groups = []
    current_label = None
    current_paths = []

    i = 0
    while i < len(items):
        item = items[i]
        if item == "--temp":
            if current_label is not None:
                groups.append((current_label, current_paths))
            current_label = items[i + 1]
            current_paths = []
            i += 2
        else:
            current_paths.append(item)
            i += 1

    if current_label is not None:
        groups.append((current_label, current_paths))

    return groups


def find_replicates(arch_folder):
    arch_folder = Path(arch_folder)
    reps = []

    for repdir in sorted(arch_folder.glob("rep*")):

        top_files = list(repdir.rglob("top.pdb"))
        dcd_files = list(repdir.rglob("*.dcd"))

        if not top_files:
            print(f"[SKIP] No top.pdb found in {repdir}")
            continue

        if not dcd_files:
            print(f"[SKIP] No dcd found in {repdir}")
            continue

        reps.append(
            (
                repdir.name,
                top_files[0].parent.name,
                top_files[0],
                dcd_files[0],
            )
        )

    return reps


def calc_metrics(top, traj, stride, frame_time, collapsed_threshold, contact_cutoff):
    u = mda.Universe(str(top), str(traj))
    atoms = u.select_atoms("all")

    rg_values_nm = []
    contact_values = []

    contact_cutoff_angstrom = contact_cutoff * 10.0

    for i, ts in enumerate(u.trajectory[::stride]):
        if i % 500 == 0:
            print(f"  frame {i}", flush=True)

        rg_nm = atoms.radius_of_gyration() / 10.0
        rg_values_nm.append(rg_nm)

        dists = self_distance_array(atoms.positions)
        contact_values.append(np.sum(dists < contact_cutoff_angstrom))

    rg_values_nm = np.array(rg_values_nm)
    contact_values = np.array(contact_values)

    frac = np.mean(rg_values_nm < collapsed_threshold)

    return {
        "mean_rg_nm": rg_values_nm.mean(),
        "std_rg_nm": rg_values_nm.std(),
        "min_rg_nm": rg_values_nm.min(),
        "max_rg_nm": rg_values_nm.max(),
        "fraction_collapsed": frac,
        "fraction_collapsed_percent": 100 * frac,
        "mean_contacts": contact_values.mean(),
        "std_contacts": contact_values.std(),
        "n_frames": len(rg_values_nm),
        "trajectory_ns": (len(rg_values_nm) - 1) * stride * frame_time,
    }


def analyse_architecture(folder, temp_label, args):
    folder = Path(folder)
    arch = folder.name
    reps = find_replicates(folder)

    if not reps:
        print(f"[SKIP] No complete replicates found in {folder}")
        return []

    rows = []

    for rep_name, sim_name, top, traj in reps:
        print(f"\n[READ] {temp_label} / {arch} / {rep_name}")
        print(f"  top:  {top}")
        print(f"  traj: {traj}")

        metrics = calc_metrics(
            top=top,
            traj=traj,
            stride=args.stride,
            frame_time=args.frame_time,
            collapsed_threshold=args.collapsed_threshold,
            contact_cutoff=args.contact_cutoff,
        )

        metrics.update(
            {
                "temperature": temp_label,
                "architecture": arch,
                "replicate": rep_name,
                "sim_name": sim_name,
                "topology": str(top),
                "trajectory": str(traj),
            }
        )

        rows.append(metrics)

    return rows


def prepare_plot_data(df, metric, sort_temperature):
    summary = (
        df.groupby(["architecture", "temperature"])
        .agg(
            mean_value=(metric, "mean"),
            std_value=(metric, "std"),
            n_reps=("replicate", "count"),
        )
        .reset_index()
    )

    summary["std_value"] = summary["std_value"].fillna(0)

    sorter = summary[summary["temperature"] == sort_temperature]
    if len(sorter) == 0:
        order = (
            summary.groupby("architecture")["mean_value"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )
    else:
        order = sorter.sort_values("mean_value")["architecture"].tolist()

    return summary, order


def plot_compare_metric(
    df,
    metric,
    ylabel,
    output,
    title,
    sort_temperature,
    add_hofmann=False,
):
    summary, order = prepare_plot_data(df, metric, sort_temperature)
    temps = sorted(df["temperature"].unique())

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(order))

    colors = {
        "300K": "#A7FF3A",
        "370K": "#0359b8",
    }

    offsets = {}
    if len(temps) == 1:
        offsets[temps[0]] = 0.0
    elif len(temps) == 2:
        offsets[temps[0]] = -0.14
        offsets[temps[1]] = 0.14
    else:
        vals = np.linspace(-0.22, 0.22, len(temps))
        offsets = dict(zip(temps, vals))

    if add_hofmann:
        hofmann = [
            ("Compact globule (ν=1/3)", 1.95),
            ("Theta state (ν=0.5)", 4.51),
            ("Expanded coil (ν=0.6)", 7.00),
        ]

        for label, value in hofmann:
            ax.axhline(
                value,
                linestyle=":",
                linewidth=1.8,
                color="black",
                alpha=0.32,
                zorder=0,
            )
            ax.text(
                len(order) - 0.1,
                value + 0.08,
                f"{label}: {value:.2f} nm",
                ha="right",
                va="bottom",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

    # Difference bars behind 300K/370K means
    if len(temps) >= 2:
        t1, t2 = temps[0], temps[1]
        for i, arch in enumerate(order):
            r1 = summary[(summary["architecture"] == arch) & (summary["temperature"] == t1)]
            r2 = summary[(summary["architecture"] == arch) & (summary["temperature"] == t2)]

            if len(r1) == 1 and len(r2) == 1:
                y1 = float(r1["mean_value"].iloc[0])
                y2 = float(r2["mean_value"].iloc[0])
                ymin = min(y1, y2)
                height = abs(y1 - y2)

                ax.bar(
                    i,
                    height,
                    bottom=ymin,
                    width=0.42,
                    color="grey",
                    alpha=0.16,
                    edgecolor="none",
                    zorder=1,
                )

                ax.plot(
                    [i + offsets[t1], i + offsets[t2]],
                    [y1, y2],
                    color="grey",
                    alpha=0.35,
                    linewidth=1.2,
                    zorder=2,
                )

    # Low-opacity replicate points
    for i, arch in enumerate(order):
        for temp in temps:
            values = df[
                (df["architecture"] == arch)
                & (df["temperature"] == temp)
            ][metric].values

            if len(values) == 0:
                continue

            if len(values) == 1:
                jitter = np.array([0.0])
            else:
                jitter = np.linspace(-0.035, 0.035, len(values))

            color = colors.get(temp, None)

            ax.scatter(
                np.full(len(values), i + offsets[temp]) + jitter,
                values,
                s=45,
                alpha=0.25,
                color=color,
                edgecolor="none",
                zorder=3,
            )

    # Mean ± SD points
    for temp in temps:
        sub = summary[summary["temperature"] == temp]

        xs, ys, yerrs = [], [], []

        for i, arch in enumerate(order):
            row = sub[sub["architecture"] == arch]
            if len(row) == 0:
                continue

            xs.append(i + offsets[temp])
            ys.append(float(row["mean_value"].iloc[0]))
            yerrs.append(float(row["std_value"].iloc[0]))

        color = colors.get(temp, None)

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            fmt="o",
            markersize=10,
            linewidth=2,
            capsize=4,
            capthick=1.6,
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.1,
            label=temp,
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("E_AI architecture")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "fraction_collapsed":
        ax.set_ylim(-0.05, 1.08)
    elif metric == "fraction_collapsed_percent":
        ax.set_ylim(-5, 105)

    ax.legend(frameon=False, title="Temperature")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare E_AI architecture replicate folders across temperatures."
    )

    parser.add_argument(
        "--temp",
        action="append",
        nargs="+",
        required=True,
        help="Temperature group followed by architecture folders, e.g. --temp 300K folder1 folder2",
    )

    parser.add_argument("--collapsed-threshold", type=float, default=3.5)
    parser.add_argument("--contact-cutoff", type=float, default=0.8)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--frame-time", type=float, default=0.05)
    parser.add_argument("--out", default="analysis_EAI_temperature_comparison")
    parser.add_argument("--sort-temperature", default="370K")

    args = parser.parse_args()

    temp_groups = [(group[0], group[1:]) for group in args.temp]

    if not temp_groups:
        print("No --temp groups found.")
        return

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    all_rows = []

    for temp_label, folders in temp_groups:
        for folder in folders:
            all_rows.extend(analyse_architecture(folder, temp_label, args))

    if not all_rows:
        print("\nNo usable runs found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "replicate_summary_temperature_comparison.csv", index=False)

    summary = df.groupby(["architecture", "temperature"]).agg(
        n_reps=("replicate", "count"),
        mean_rg_nm=("mean_rg_nm", "mean"),
        std_rg_nm=("mean_rg_nm", "std"),
        mean_fraction_collapsed=("fraction_collapsed", "mean"),
        std_fraction_collapsed=("fraction_collapsed", "std"),
        mean_fraction_collapsed_percent=("fraction_collapsed_percent", "mean"),
        std_fraction_collapsed_percent=("fraction_collapsed_percent", "std"),
        mean_contacts=("mean_contacts", "mean"),
        std_contacts=("mean_contacts", "std"),
        mean_n_frames=("n_frames", "mean"),
    )

    summary = summary.fillna(0)
    summary.to_csv(out / "architecture_summary_temperature_comparison.csv")

    plot_compare_metric(
        df,
        metric="mean_rg_nm",
        ylabel="Mean radius of gyration (nm)",
        output=out / "average_rg_temperature_comparison.png",
        title="Radius of gyration of E_AI architectures\n300K versus 370K with Hofmann references for N=420",
        sort_temperature=args.sort_temperature,
        add_hofmann=True,
    )

    plot_compare_metric(
        df,
        metric="fraction_collapsed_percent",
        ylabel="Fraction collapsed (%)",
        output=out / "fraction_collapsed_temperature_comparison.png",
        title=f"Collapsed fraction of E_AI architectures\ncollapsed if Rg < {args.collapsed_threshold} nm",
        sort_temperature=args.sort_temperature,
        add_hofmann=False,
    )

    plot_compare_metric(
        df,
        metric="mean_contacts",
        ylabel="Mean number of contacts",
        output=out / "average_contacts_temperature_comparison.png",
        title="Mean intramolecular contacts of E_AI architectures\n300K versus 370K",
        sort_temperature=args.sort_temperature,
        add_hofmann=False,
    )

    print("\n[OK] Results written to:")
    print(out.resolve())
    print(summary)


if __name__ == "__main__":
    main()