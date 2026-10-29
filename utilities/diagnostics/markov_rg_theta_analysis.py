import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import MDAnalysis as mda


def compute_rg_timeseries(topology, trajectory, stride=10, frame_time_ns=0.05):
    u = mda.Universe(topology, trajectory)
    protein = u.select_atoms("protein")

    times = []
    rgs = []

    for i, ts in enumerate(u.trajectory[::stride]):
        rg = protein.radius_of_gyration()
        times.append(i * stride * frame_time_ns)
        rgs.append(rg / 10.0)  # Angstrom to nm

    return np.array(times), np.array(rgs)


def run_lengths(binary_states, target_state):
    lengths = []
    current = 0

    for s in binary_states:
        if s == target_state:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0

    if current > 0:
        lengths.append(current)

    return np.array(lengths, dtype=float)


def markov_analysis(states, dt_ns):
    # states: 0 = expanded, 1 = collapsed
    counts = np.zeros((2, 2), dtype=int)

    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1

    probs = counts / counts.sum(axis=1, keepdims=True)

    P00, P01 = probs[0, 0], probs[0, 1]
    P10, P11 = probs[1, 0], probs[1, 1]

    k_collapse = P01 / dt_ns
    k_escape = P10 / dt_ns

    return {
        "count_00": counts[0, 0],
        "count_01": counts[0, 1],
        "count_10": counts[1, 0],
        "count_11": counts[1, 1],
        "P00_stay_expanded": P00,
        "P01_collapse": P01,
        "P10_escape": P10,
        "P11_stay_collapsed": P11,
        "k_collapse_per_ns": k_collapse,
        "k_escape_per_ns": k_escape,
    }


def analyse_one_run(label, topology, trajectory, n_residues, threshold_mode,
                    custom_threshold_nm, stride, frame_time_ns):

    times, rgs = compute_rg_timeseries(
        topology=topology,
        trajectory=trajectory,
        stride=stride,
        frame_time_ns=frame_time_ns,
    )

    rho0 = 0.21
    r_theta = rho0 * np.sqrt(n_residues)
    r_globule = rho0 * n_residues ** (1 / 3)
    r_expanded = rho0 * n_residues ** (3 / 5)

    if threshold_mode == "theta":
        threshold = r_theta
    elif threshold_mode == "custom":
        threshold = custom_threshold_nm
    else:
        raise ValueError("threshold_mode must be 'theta' or 'custom'")

    states = (rgs < threshold).astype(int)
    dt_ns = stride * frame_time_ns

    collapsed_lengths = run_lengths(states, 1)
    expanded_lengths = run_lengths(states, 0)

    collapsed_dwells_ns = collapsed_lengths * dt_ns
    expanded_dwells_ns = expanded_lengths * dt_ns

    markov = markov_analysis(states, dt_ns)

    summary = {
        "label": label,
        "N": n_residues,
        "n_frames": len(rgs),
        "dt_ns": dt_ns,
        "threshold_mode": threshold_mode,
        "threshold_nm": threshold,
        "R_theta_nm": r_theta,
        "R_globule_nm": r_globule,
        "R_expanded_nm": r_expanded,
        "mean_Rg_nm": np.mean(rgs),
        "std_Rg_nm": np.std(rgs),
        "Rg_over_Rtheta": np.mean(rgs) / r_theta,
        "fraction_collapsed": np.mean(states),
        "n_collapse_events": int(np.sum((states[:-1] == 0) & (states[1:] == 1))),
        "n_escape_events": int(np.sum((states[:-1] == 1) & (states[1:] == 0))),
        "mean_collapsed_dwell_ns": np.mean(collapsed_dwells_ns) if len(collapsed_dwells_ns) else 0,
        "median_collapsed_dwell_ns": np.median(collapsed_dwells_ns) if len(collapsed_dwells_ns) else 0,
        "max_collapsed_dwell_ns": np.max(collapsed_dwells_ns) if len(collapsed_dwells_ns) else 0,
        "mean_expanded_dwell_ns": np.mean(expanded_dwells_ns) if len(expanded_dwells_ns) else 0,
        "median_expanded_dwell_ns": np.median(expanded_dwells_ns) if len(expanded_dwells_ns) else 0,
        "max_expanded_dwell_ns": np.max(expanded_dwells_ns) if len(expanded_dwells_ns) else 0,
    }

    summary.update(markov)

    # Markov-predicted lifetimes from rates
    summary["markov_pred_expanded_lifetime_ns"] = (
        1 / summary["k_collapse_per_ns"]
        if summary["k_collapse_per_ns"] > 0 else np.nan
    )

    summary["markov_pred_collapsed_lifetime_ns"] = (
        1 / summary["k_escape_per_ns"]
        if summary["k_escape_per_ns"] > 0 else np.nan
    )

    # Ratio: measured dwell / Markov-predicted dwell
    summary["expanded_lifetime_measured_over_markov"] = (
        summary["mean_expanded_dwell_ns"] / summary["markov_pred_expanded_lifetime_ns"]
        if summary["markov_pred_expanded_lifetime_ns"] > 0 else np.nan
    )

    summary["collapsed_lifetime_measured_over_markov"] = (
        summary["mean_collapsed_dwell_ns"] / summary["markov_pred_collapsed_lifetime_ns"]
        if summary["markov_pred_collapsed_lifetime_ns"] > 0 else np.nan
    )

    # Markov-predicted equilibrium populations
    k_collapse = summary["k_collapse_per_ns"]
    k_escape = summary["k_escape_per_ns"]

    summary["markov_pred_fraction_collapsed"] = (
        k_collapse / (k_collapse + k_escape)
        if (k_collapse + k_escape) > 0 else np.nan
    )

    summary["markov_pred_fraction_expanded"] = (
        k_escape / (k_collapse + k_escape)
        if (k_collapse + k_escape) > 0 else np.nan
    )

    # Compare measured and predicted equilibrium populations
    summary["fraction_collapsed_measured_minus_markov"] = (
        summary["fraction_collapsed"]
        - summary["markov_pred_fraction_collapsed"]
    )

    summary["fraction_collapsed_measured_over_markov"] = (
        summary["fraction_collapsed"]
        / summary["markov_pred_fraction_collapsed"]
        if summary["markov_pred_fraction_collapsed"] > 0 else np.nan
    )

    timeseries = pd.DataFrame({
        "time_ns": times,
        "Rg_nm": rgs,
        "state": states,
    })

    return summary, timeseries

def plot_markov_diagram_average(summary_df, outpath, title="Two-state Markov model"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    mean = summary_df.mean(numeric_only=True)
    std = summary_df.std(numeric_only=True)

    P00 = mean["P00_stay_expanded"]
    P01 = mean["P01_collapse"]
    P10 = mean["P10_escape"]
    P11 = mean["P11_stay_collapsed"]

    k01 = mean["k_collapse_per_ns"]
    k10 = mean["k_escape_per_ns"]

    tau_expanded = 1 / k01 if k01 > 0 else np.inf
    tau_collapsed = 1 / k10 if k10 > 0 else np.inf

    frac = mean["fraction_collapsed"]
    n_reps = len(summary_df)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    pos_exp = (3, 3)
    pos_col = (7, 3)

    expanded_color = "#cfe8ff"
    collapsed_color = "#ffd6d6"

    ax.add_patch(Circle(pos_exp, 1.05, facecolor=expanded_color, edgecolor="black", linewidth=2.5))
    ax.add_patch(Circle(pos_col, 1.05, facecolor=collapsed_color, edgecolor="black", linewidth=2.5))

    ax.text(*pos_exp, "Expanded\nRg ≥ Rθ", ha="center", va="center", fontsize=15)
    ax.text(*pos_col, "Collapsed\nRg < Rθ", ha="center", va="center", fontsize=15)

    # Expanded -> collapsed
    ax.add_patch(FancyArrowPatch(
        (4.05, 2.45), (5.95, 2.45),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="->",
        mutation_scale=22,
        linewidth=2.5,
        color="black"
    ))

    # Collapsed -> expanded
    ax.add_patch(FancyArrowPatch(
        (5.95, 3.55), (4.05, 3.55),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="->",
        mutation_scale=22,
        linewidth=2.5,
        color="black"
    ))

    ax.text(
        5, 1.75,
        f"P01 = {P01:.3f}\n"
        f"kcollapse = {k01:.2f} ns⁻¹\n"
        f"τexpanded = {tau_expanded:.2f} ns",
        ha="center", va="center", fontsize=12
    )

    ax.text(
        5, 4.25,
        f"P10 = {P10:.3f}\n"
        f"kescape = {k10:.2f} ns⁻¹\n"
        f"τcollapsed = {tau_collapsed:.2f} ns",
        ha="center", va="center", fontsize=12
    )

    # self loops
    ax.add_patch(FancyArrowPatch(
        (2.2, 3.85), (2.2, 2.15),
        connectionstyle="arc3,rad=1.7",
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color="black"
    ))

    ax.add_patch(FancyArrowPatch(
        (7.8, 2.15), (7.8, 3.85),
        connectionstyle="arc3,rad=1.7",
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color="black"
    ))

    ax.text(1.25, 3.0, f"P00 = {P00:.3f}", ha="center", va="center", fontsize=13)
    ax.text(8.75, 3.0, f"P11 = {P11:.3f}", ha="center", va="center", fontsize=13)

    ax.set_title(
        f"{title}\n"
        f"mean over {n_reps} reps | fraction collapsed = {frac:.3f} ± {std['fraction_collapsed']:.3f}",
        fontsize=17,
        pad=18
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label::topology.pdb::trajectory.dcd",
    )

    parser.add_argument("--n-residues", type=int, required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--frame-time", type=float, default=0.05)

    parser.add_argument(
        "--threshold-mode",
        choices=["theta", "custom"],
        default="theta",
    )

    parser.add_argument(
        "--custom-threshold",
        type=float,
        default=None,
        help="Custom Rg threshold in nm, only used when --threshold-mode custom",
    )

    parser.add_argument("--outdir", default="markov_rg_theta_analysis")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for run in args.run:
        label, topology, trajectory = run.split("::")

        print(f"\nAnalysing {label}")
        summary, timeseries = analyse_one_run(
            label=label,
            topology=topology,
            trajectory=trajectory,
            n_residues=args.n_residues,
            threshold_mode=args.threshold_mode,
            custom_threshold_nm=args.custom_threshold,
            stride=args.stride,
            frame_time_ns=args.frame_time,
        )

        summaries.append(summary)

        timeseries.to_csv(outdir / f"{label}_timeseries_states.csv", index=False)

        
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "markov_summary_per_run.csv", index=False)

    plot_markov_diagram_average(
        summary_df,
        outdir / "markov_diagram_average.png",
        title=args.outdir.replace("_", " ")
    )

    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    mean_df = summary_df[numeric_cols].agg(["mean", "std"]).T
    mean_df.to_csv(outdir / "markov_summary_mean_std.csv")
    mean_df.loc["expanded_lifetime_ns"] = [
        1 / summary_df["k_collapse_per_ns"].mean(),
        summary_df["k_collapse_per_ns"].std() / summary_df["k_collapse_per_ns"].mean()**2
    ]

    mean_df.loc["collapsed_lifetime_ns"] = [
        1 / summary_df["k_escape_per_ns"].mean(),
        summary_df["k_escape_per_ns"].std() / summary_df["k_escape_per_ns"].mean()**2
    ]

    mean_df.to_csv(outdir / "markov_summary_mean_std.csv")

    print("\nDone.")
    print(summary_df[[
    "label",
    "mean_Rg_nm",
    "Rg_over_Rtheta",
    "fraction_collapsed",
    "markov_pred_fraction_collapsed",
    "fraction_collapsed_measured_minus_markov",
    "fraction_collapsed_measured_over_markov",
    "mean_collapsed_dwell_ns",
    "markov_pred_collapsed_lifetime_ns",
    "collapsed_lifetime_measured_over_markov",
    "mean_expanded_dwell_ns",
    "markov_pred_expanded_lifetime_ns",
    "expanded_lifetime_measured_over_markov",
    "P01_collapse",
    "P10_escape",
    "k_collapse_per_ns",
    "k_escape_per_ns",
]])


if __name__ == "__main__":
    main()