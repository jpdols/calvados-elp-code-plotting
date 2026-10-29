import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_distance_array


def find_run(rep_path):
    rep_path = Path(rep_path)

    pdbs = sorted(rep_path.rglob("top.pdb"))
    dcds = sorted(rep_path.rglob("*.dcd"))

    if not pdbs or not dcds:
        raise FileNotFoundError(f"No top.pdb / dcd found in {rep_path}")

    return pdbs[0], dcds[0]

def collapsed_dwells(collapsed_bool, dt_ns):
    """
    Return durations of contiguous collapsed periods in ns.
    collapsed_bool: array of True/False values
    dt_ns: effective time per analysed frame in ns
    """
    dwell_lengths = []
    current = 0

    for is_collapsed in collapsed_bool:
        if is_collapsed:
            current += 1
        else:
            if current > 0:
                dwell_lengths.append(current * dt_ns)
                current = 0

    if current > 0:
        dwell_lengths.append(current * dt_ns)

    return dwell_lengths


def analyse_rep(rep_path, collapsed_threshold=3.5, contact_cutoff=0.8, stride=10, frame_time=0.01):
    top, traj = find_run(rep_path)

    u = mda.Universe(str(top), str(traj))
    atoms = u.atoms

    rg_values_nm = []
    contact_values = []

    contact_cutoff_angstrom = contact_cutoff * 10.0

    for i, ts in enumerate(u.trajectory[::stride]):
        if i % 500 == 0:
            print(f"  frame {i}", flush=True)

        rg_nm = atoms.radius_of_gyration() / 10.0
        rg_values_nm.append(rg_nm)

        dists = self_distance_array(atoms.positions)
        contacts = np.sum(dists < contact_cutoff_angstrom)
        contact_values.append(contacts)

    rg_values_nm = np.array(rg_values_nm)
    contact_values = np.array(contact_values)

    collapsed_bool = rg_values_nm < collapsed_threshold
    effective_dt_ns = frame_time * stride
    dwell_durations_ns = collapsed_dwells(collapsed_bool, effective_dt_ns)

    if len(dwell_durations_ns) > 0:
        mean_dwell_ns = np.mean(dwell_durations_ns)
        longest_dwell_ns = np.max(dwell_durations_ns)
        n_dwells = len(dwell_durations_ns)
    else:
        mean_dwell_ns = 0.0
        longest_dwell_ns = 0.0
        n_dwells = 0

    return {
        "rep": Path(rep_path).name,
        "topology": str(top),
        "trajectory": str(traj),
        "mean_rg_nm": rg_values_nm.mean(),
        "fraction_collapsed": np.mean(collapsed_bool),
        "mean_contacts": contact_values.mean(),
        "n_frames": len(rg_values_nm),
        "n_dwells": n_dwells,
        "mean_dwell_ns": mean_dwell_ns,
        "longest_dwell_ns": longest_dwell_ns,
        "dwell_durations_ns": dwell_durations_ns,
    }


def analyse_architecture(path, collapsed_threshold, contact_cutoff, stride, frame_time, temperature_group):
    path = Path(path)
    rows = []

    # If path is ...\VPGEG\300K, architecture should be VPGEG
    # If path is directly ...\VPGEG, architecture should be VPGEG
    if path.name.endswith("K"):
        architecture_name = path.parent.name
    else:
        architecture_name = path.name

    # First check whether this folder contains replicate folders
    rep_dirs = sorted(
        [
            p for p in path.iterdir()
            if p.is_dir() and p.name.lower().startswith("rep")
        ]
    )

    # If no rep folders are found, fall back to old behaviour
    if not rep_dirs:
        rep_dirs = [path]

    print(f"\nAnalysing {architecture_name} / {temperature_group}", flush=True)

    for rep_dir in rep_dirs:
        try:
            print(f"  Rep/source: {rep_dir.name}", flush=True)

            result = analyse_rep(
                rep_dir,
                collapsed_threshold=collapsed_threshold,
                contact_cutoff=contact_cutoff,
                stride=stride,
                frame_time=frame_time,
            )

            result["architecture"] = architecture_name
            result["temperature_group"] = temperature_group
            result["rep"] = rep_dir.name

            rows.append(result)

        except FileNotFoundError as e:
            print(f"Skipping {architecture_name} / {rep_dir.name} / {temperature_group}: {e}", flush=True)

        except Exception as e:
            print(f"Error in {architecture_name} / {rep_dir.name} / {temperature_group}: {e}", flush=True)

    return rows


def prepare_plot_data(df, metric, main_label):
    summary = (
        df.groupby(["architecture", "temperature_group"])
        .agg(
            mean_value=(metric, "mean"),
            std_value=(metric, "std"),
            n_reps=("rep", "count"),
        )
        .reset_index()
    )

    summary["std_value"] = summary["std_value"].fillna(0)

    main = summary[summary["temperature_group"] == main_label]
    if len(main) > 0:
        order = main.sort_values("mean_value")["architecture"].tolist()
    else:
        order = (
            summary.groupby("architecture")["mean_value"]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

    summary["architecture"] = pd.Categorical(
        summary["architecture"],
        categories=order,
        ordered=True,
    )

    return summary.sort_values(["architecture", "temperature_group"]).reset_index(drop=True), order


def hofmann_values(n_residues, rho0):
    return [
        ("Compact globule (ν=1/3)", rho0 * n_residues ** (1 / 3)),
        ("Theta state (ν=0.5)", rho0 * n_residues ** 0.5),
        ("Expanded coil (ν=0.6)", rho0 * n_residues ** 0.6),
    ]


def plot_point_metric(
    df,
    metric,
    ylabel,
    output,
    title,
    main_label,
    extra_temp,
    add_hofmann=False,
    hofmann_n=450,
    hofmann_rho0=0.216,
):
    plot_df, order = prepare_plot_data(df, metric, main_label)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(order))

    color_main = "#0359b8"
    color_extra = "#A7FF3A"

    offset_main = 0.13
    offset_extra = -0.13

    if add_hofmann:
        for label, value in hofmann_values(hofmann_n, hofmann_rho0):
            ax.axhline(
                value,
                linestyle=":",
                linewidth=1.8,
                color="black",
                alpha=0.30,
                zorder=0,
            )

            ax.text(
                len(order) - 0.1,
                value + 0.08,
                f"{label}: {value:.2f} nm",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="black",
                alpha=0.75,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

    # Difference bars behind points
    for i, arch in enumerate(order):
        vals = plot_df[plot_df["architecture"] == arch]

        main_row = vals[vals["temperature_group"] == main_label]
        extra_row = vals[vals["temperature_group"] == extra_temp]

        if len(main_row) == 1 and len(extra_row) == 1:
            y1 = float(main_row["mean_value"].iloc[0])
            y2 = float(extra_row["mean_value"].iloc[0])

            ymin = min(y1, y2)
            height = abs(y1 - y2)

            ax.bar(
                i,
                height,
                bottom=ymin,
                width=0.38,
                alpha=0.16,
                color="grey",
                edgecolor="none",
                zorder=1,
            )

            ax.plot(
                [i + offset_extra, i + offset_main],
                [y2, y1],
                color="grey",
                alpha=0.35,
                linewidth=1.2,
                zorder=1,
            )

    # Low-opacity replicate points
    for i, arch in enumerate(order):
        for temp_group, base_offset, color in [
            (extra_temp, offset_extra, color_extra),
            (main_label, offset_main, color_main),
        ]:
            values = df[
                (df["architecture"] == arch)
                & (df["temperature_group"] == temp_group)
            ][metric].values

            if len(values) == 0:
                continue

            if len(values) == 1:
                jitter = np.array([0.0])
            else:
                jitter = np.linspace(-0.035, 0.035, len(values))

            ax.scatter(
                np.full(len(values), i + base_offset) + jitter,
                values,
                s=45,
                alpha=0.25,
                color=color,
                edgecolor="none",
                zorder=2,
            )

    # Mean ± std coloured points
    for temp_group, base_offset, color, label in [
        (extra_temp, offset_extra, color_extra, extra_temp),
        (main_label, offset_main, color_main, main_label),
    ]:
        sub = plot_df[plot_df["temperature_group"] == temp_group]

        xs = []
        ys = []
        yerrs = []

        for i, arch in enumerate(order):
            row = sub[sub["architecture"] == arch]
            if len(row) == 0:
                continue

            xs.append(i + base_offset)
            ys.append(float(row["mean_value"].iloc[0]))
            yerrs.append(float(row["std_value"].iloc[0]))

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
            label=label,
            zorder=4,
        )

    ax.set_xticks(x)

    labels = [
        x.replace("FG_EG_", "")
        .replace("_blockshuffle_seed42", "")
        .replace("_", "\n")
        for x in order
    ]

    ax.set_xticklabels(labels, fontsize=10)   

    ax.set_ylabel(ylabel)
    ax.set_xlabel("ELP architecture")
    ax.set_title(title)

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric == "fraction_collapsed":
        ax.set_ylim(-0.05, 1.08)

    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

def plot_dwell_distribution(
    dwell_df,
    order,
    output,
    title,
    main_label,
    extra_temp,
):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(order))

    color_main = "#0359b8"
    color_extra = "#A7FF3A"

    offset_main = 0.13
    offset_extra = -0.13

    for i, arch in enumerate(order):
        for temp_group, base_offset, color in [
            (extra_temp, offset_extra, color_extra),
            (main_label, offset_main, color_main),
        ]:
            values = dwell_df[
                (dwell_df["architecture"] == arch)
                & (dwell_df["temperature_group"] == temp_group)
            ]["dwell_duration_ns"].values

            if len(values) == 0:
                continue

            if len(values) == 1:
                jitter = np.array([0.0])
            else:
                jitter = np.linspace(-0.045, 0.045, len(values))

            # Too many dwell events can overplot, so make them small and transparent
            ax.scatter(
                np.full(len(values), i + base_offset) + jitter,
                values,
                s=14,
                alpha=0.18,
                color=color,
                edgecolor="none",
                zorder=2,
            )

            mean_val = np.mean(values)
            std_val = np.std(values)

            ax.errorbar(
                i + base_offset,
                mean_val,
                yerr=std_val,
                fmt="o",
                markersize=9,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                color=color,
                ecolor=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=4,
            )

    labels = [
        a.replace("FG_EG_", "")
        .replace("_blockshuffle_seed42", "")
        .replace("_", "\n")
        for a in order
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylabel("Collapsed dwell duration (ns)")
    ax.set_xlabel("ELP architecture")
    ax.set_title(title)

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.scatter([], [], color=color_extra, label=extra_temp)
    ax.scatter([], [], color=color_main, label=main_label)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyse CALVADOS replicate folders and plot Rg, collapsed fraction, and contacts."
    )

    
    parser.add_argument(
        "--collapsed-threshold",
        type=float,
        default=3.5,
        help="Rg threshold for collapsed frames, in nm.",
    )

    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=0.8,
        help="Contact cutoff in nm.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Analyse every nth trajectory frame.",
    )

    parser.add_argument(
        "--frame-time",
        type=float,
        default=0.01,
        help="Original trajectory time per frame in ns before stride.",
    )

    parser.add_argument(
        "--outdir",
        default="analysis_replicates",
        help="Output directory.",
    )

    parser.add_argument("--temp", action="append", nargs="+", required=True)

    parser.add_argument(
        "--main-label",
        default="370K",
        help="Main temperature label used for sorting and right-side points.",
    )

    parser.add_argument(
        "--extra-temp",
        default="300K",
        help="Comparison temperature label used for left-side points.",
    )

    parser.add_argument(
        "--hofmann-n",
        type=int,
        default=450,
        help="Sequence length N for Hofmann reference lines.",
    )

    parser.add_argument(
        "--hofmann-rho0",
        type=float,
        default=0.21,
        help="Hofmann rho0 prefactor in nm.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    all_rows = []

    for temp_group in args.temp:
        temperature_label = temp_group[0]
        paths = temp_group[1:]

        for path in paths:
            all_rows.extend(
                analyse_architecture(
                    path,
                    args.collapsed_threshold,
                    args.contact_cutoff,
                    args.stride,
                    args.frame_time,
                    temperature_label,
                )
            )

    if not all_rows:
        print("\nNo usable runs found. Nothing to analyse.")
        return

    df = pd.DataFrame(all_rows)

    dwell_rows = []

    for _, row in df.iterrows():
        for dwell_duration in row["dwell_durations_ns"]:
            dwell_rows.append(
                {
                    "architecture": row["architecture"],
                    "temperature_group": row["temperature_group"],
                    "rep": row["rep"],
                    "dwell_duration_ns": dwell_duration,
                }
            )

    dwell_df = pd.DataFrame(dwell_rows)

    if len(dwell_df) > 0:
        dwell_df.to_csv(outdir / "collapsed_dwell_durations.csv", index=False)

    df_no_lists = df.drop(columns=["dwell_durations_ns"])
    df_no_lists.to_csv(outdir / "replicate_summary_with_300K.csv", index=False)

    summary = df.groupby(["architecture", "temperature_group"]).agg(
        n_reps=("rep", "count"),
        mean_rg_nm=("mean_rg_nm", "mean"),
        std_rg_nm=("mean_rg_nm", "std"),
        mean_fraction_collapsed=("fraction_collapsed", "mean"),
        std_fraction_collapsed=("fraction_collapsed", "std"),
        mean_contacts=("mean_contacts", "mean"),
        std_contacts=("mean_contacts", "std"),
        mean_n_frames=("n_frames", "mean"),
        mean_n_dwells=("n_dwells", "mean"),
        std_n_dwells=("n_dwells", "std"),
        mean_dwell_ns=("mean_dwell_ns", "mean"),
        std_dwell_ns=("mean_dwell_ns", "std"),
        mean_longest_dwell_ns=("longest_dwell_ns", "mean"),
        std_longest_dwell_ns=("longest_dwell_ns", "std"),
    )

    summary = summary.fillna(0)
    summary.to_csv(outdir / "architecture_summary_with_300K.csv")

    plot_point_metric(
        df,
        metric="mean_rg_nm",
        ylabel="Mean radius of gyration (nm)",
        output=outdir / "average_rg_300K_comparison.png",
        title="Radius of gyration of VPGXG ELPs: main runs versus 300K",
        main_label=args.main_label,
        extra_temp=args.extra_temp,
        add_hofmann=True,
        hofmann_n=args.hofmann_n,
        hofmann_rho0=args.hofmann_rho0,
    )

    plot_point_metric(
        df,
        metric="fraction_collapsed",
        ylabel="Fraction collapsed",
        output=outdir / "fraction_collapsed_300K_comparison.png",
        title="Collapsed fraction of VPGXG ELPs: main runs versus 300K",
        main_label=args.main_label,
        extra_temp=args.extra_temp,
        add_hofmann=False,
    )

    
    plot_point_metric(
        df,
        metric="mean_contacts",
        ylabel="Mean number of contacts",
        output=outdir / "average_contacts_300K_comparison.png",
        title="Mean contacts of VPGXG ELPs: main runs versus 300K",
        main_label=args.main_label,
        extra_temp=args.extra_temp,
        add_hofmann=False,
    )

    if len(dwell_df) > 0:
        _, order = prepare_plot_data(df, "mean_rg_nm", args.main_label)

        plot_dwell_distribution(
            dwell_df=dwell_df,
            order=order,
            output=outdir / "collapsed_dwell_duration_distribution.png",
            title="Collapsed dwell duration distributions of VPGXG ELPs",
            main_label=args.main_label,
            extra_temp=args.extra_temp,
        )

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()