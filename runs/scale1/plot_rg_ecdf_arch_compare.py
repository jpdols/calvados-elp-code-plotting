import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda


def find_run(path):
    path = Path(path)
    pdbs = sorted(path.rglob("top.pdb"))
    dcds = sorted(path.rglob("*.dcd"))

    if not pdbs or not dcds:
        raise FileNotFoundError(f"No top.pdb / dcd found in {path}")

    return pdbs[0], dcds[0]


def get_rep_dirs(path):
    path = Path(path)
    reps = sorted(
        [p for p in path.iterdir() if p.is_dir() and p.name.lower().startswith("rep")]
    )
    return reps if reps else [path]


def clean_label(label):
    return (
        label.replace("FG_EG_", "")
        .replace("_mediumblocks_15", "medium blocks")
        .replace("_center_charged", "center charged")
        .replace("_center_hydrophobic", "center hydrophobic")
        .replace("_random_blockshuffle_seed42", "random")
        .replace("_smallblocks_5", "small blocks")
        .replace("_", " ")
    )


def analyse_rg(path, architecture, temperature, stride):
    rows = []

    for rep_dir in get_rep_dirs(path):
        print(f"\nAnalysing {architecture} / {temperature} / {rep_dir.name}", flush=True)

        top, traj = find_run(rep_dir)
        u = mda.Universe(str(top), str(traj))
        atoms = u.atoms

        for i, ts in enumerate(u.trajectory[::stride]):
            if i % 500 == 0:
                print(f"  frame {i}", flush=True)

            rows.append(
                {
                    "architecture": architecture,
                    "temperature": temperature,
                    "rep": rep_dir.name,
                    "frame": i,
                    "rg_nm": atoms.radius_of_gyration() / 10.0,
                }
            )

    return rows


def ecdf(values):
    values = np.sort(np.asarray(values))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def make_summary(df, threshold):
    rows = []

    for (arch, temp), sub in df.groupby(["architecture", "temperature"]):
        values = sub["rg_nm"].values
        below = values < threshold

        rows.append(
            {
                "architecture": arch,
                "temperature": temp,
                "n_frames": len(values),
                "mean_rg_nm": np.mean(values),
                "std_rg_nm": np.std(values),
                "median_rg_nm": np.median(values),
                "p05_rg_nm": np.percentile(values, 5),
                "p10_rg_nm": np.percentile(values, 10),
                "p25_rg_nm": np.percentile(values, 25),
                "p75_rg_nm": np.percentile(values, 75),
                "p90_rg_nm": np.percentile(values, 90),
                "p95_rg_nm": np.percentile(values, 95),
                "iqr_nm": np.percentile(values, 75) - np.percentile(values, 25),
                "fraction_below_threshold": np.mean(below),
                "mean_collapse_depth_nm": np.mean(np.maximum(0, threshold - values)),
                "mean_collapse_depth_collapsed_only_nm": (
                    np.mean(threshold - values[below]) if np.any(below) else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_style_maps(architectures, temperatures):
    # Temperature = colour
    temp_colors = {
        "300K": "#A7FF3A",
        "370K": "#0359b8",
    }

    fallback_colors = [
        "#0359b8",
        "#A7FF3A",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
        "#999999",
    ]

    for i, temp in enumerate(temperatures):
        if temp not in temp_colors:
            temp_colors[temp] = fallback_colors[i % len(fallback_colors)]

    # Architecture = line style
    line_styles = ["-", "--", "-.", ":"]
    arch_linestyles = {
        arch: line_styles[i % len(line_styles)] for i, arch in enumerate(architectures)
    }

    return temp_colors, arch_linestyles


def plot_ecdf(df, summary, threshold, outpath, title):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    architectures = list(dict.fromkeys(df["architecture"].tolist()))
    temperatures = list(dict.fromkeys(df["temperature"].tolist()))

    temp_colors, arch_linestyles = build_style_maps(architectures, temperatures)

    # Light P10-P90 bands
    for _, row in summary.iterrows():
        arch = row["architecture"]
        temp = row["temperature"]

        ax.axvspan(
            row["p10_rg_nm"],
            row["p90_rg_nm"],
            color=temp_colors[temp],
            alpha=0.045,
            zorder=0,
        )

    # ECDF curves + median markers
    for (arch, temp), sub in df.groupby(["architecture", "temperature"]):
        values = sub["rg_nm"].values
        x, y = ecdf(values)

        median = np.median(values)

        ax.plot(
            x,
            y,
            linewidth=2.6,
            color=temp_colors[temp],
            linestyle=arch_linestyles[arch],
            label=f"{clean_label(arch)} {temp}",
            alpha=0.96,
            zorder=3,
        )

        # Median point at y=0.5
        ax.scatter(
            median,
            0.5,
            s=55,
            color=temp_colors[temp],
            edgecolor="white",
            linewidth=1.1,
            marker="o",
            zorder=5,
        )

    ax.axvline(
        threshold,
        linestyle=":",
        linewidth=2.4,
        color="black",
        alpha=0.55,
        label=f"Theta threshold {threshold:.2f} nm",
        zorder=2,
    )

    ax.set_xlabel("Radius of gyration Rg (nm)")
    ax.set_ylabel("Cumulative probability P(Rg < x)")
    ax.set_title(title)

    ax.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(-0.02, 1.02)

    ax.text(
        0.02,
        0.04,
        "Point at y=0.5 = median Rg\nLight bands = P10–P90 range",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_percentile_summary(summary, threshold, outpath, title):
    summary = summary.copy()
    summary["label"] = summary["architecture"].apply(clean_label) + "\n" + summary["temperature"]

    order = summary.sort_values("median_rg_nm")["label"].tolist()
    summary["label"] = pd.Categorical(summary["label"], categories=order, ordered=True)
    summary = summary.sort_values("label")

    temperatures = list(dict.fromkeys(summary["temperature"].tolist()))
    temp_colors, _ = build_style_maps(summary["architecture"].unique(), temperatures)

    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    x = np.arange(len(summary))

    lower = summary["median_rg_nm"].values - summary["p10_rg_nm"].values
    upper = summary["p90_rg_nm"].values - summary["median_rg_nm"].values

    colors = [temp_colors[t] for t in summary["temperature"]]

    ax.errorbar(
        x,
        summary["median_rg_nm"].values,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=1.7,
        capsize=4,
        alpha=0.6,
        zorder=2,
    )

    ax.scatter(
        x,
        summary["median_rg_nm"].values,
        s=95,
        c=colors,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    ax.axhline(
        threshold,
        linestyle=":",
        linewidth=2.4,
        color="black",
        alpha=0.55,
        label=f"Theta threshold {threshold:.2f} nm",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], fontsize=9)
    ax.set_ylabel("Radius of gyration Rg (nm)")
    ax.set_title(title)

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.02,
        0.96,
        "Dots = median Rg\nBars = P10–P90",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85),
    )

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative Rg distributions and percentile summaries."
    )

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: architecture::temperature::path",
    )

    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=4.46)
    parser.add_argument("--outdir", default="rg_distribution_analysis")
    parser.add_argument("--title", default="Rg cumulative distributions")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for run in args.run:
        parts = run.split("::")
        if len(parts) != 3:
            raise ValueError(f"Invalid --run format: {run}")

        architecture, temperature, path = parts
        all_rows.extend(
            analyse_rg(
                path=path,
                architecture=architecture,
                temperature=temperature,
                stride=args.stride,
            )
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "all_rg_values.csv", index=False)

    summary = make_summary(df, args.threshold)
    summary.to_csv(outdir / "rg_distribution_summary.csv", index=False)

    plot_ecdf(
        df=df,
        summary=summary,
        threshold=args.threshold,
        outpath=outdir / "rg_cumulative_distributions.png",
        title=args.title,
    )

    plot_percentile_summary(
        summary=summary,
        threshold=args.threshold,
        outpath=outdir / "rg_percentile_summary.png",
        title="Median Rg with P10–P90 range",
    )

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()