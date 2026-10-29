import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def transition_matrix(states, lag=1):
    counts = np.zeros((2, 2), dtype=float)

    for a, b in zip(states[:-lag], states[lag:]):
        counts[int(a), int(b)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(
        counts,
        row_sums,
        out=np.full_like(counts, np.nan),
        where=row_sums != 0,
    )

    return counts, probs


def run_lengths(states, target_state):
    lengths = []
    current = 0

    for s in states:
        if s == target_state:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
            current = 0

    if current > 0:
        lengths.append(current)

    return np.array(lengths, dtype=float)


def chapman_kolmogorov_df(states, dt_ns, lags):
    _, P1 = transition_matrix(states, lag=1)

    rows = []

    for lag in lags:
        _, Plag_obs = transition_matrix(states, lag=lag)
        Plag_pred = np.linalg.matrix_power(P1, lag)

        rows.append({
            "lag_frames": lag,
            "lag_ns": lag * dt_ns,
            "P01_observed": Plag_obs[0, 1],
            "P01_predicted": Plag_pred[0, 1],
            "P10_observed": Plag_obs[1, 0],
            "P10_predicted": Plag_pred[1, 0],
            "P00_observed": Plag_obs[0, 0],
            "P00_predicted": Plag_pred[0, 0],
            "P11_observed": Plag_obs[1, 1],
            "P11_predicted": Plag_pred[1, 1],
            "P01_abs_error": abs(Plag_obs[0, 1] - Plag_pred[0, 1]),
            "P10_abs_error": abs(Plag_obs[1, 0] - Plag_pred[1, 0]),
        })

    return pd.DataFrame(rows)


def memory_test_df(states, history_lengths):
    rows = []

    for history_len in history_lengths:
        n0 = 0
        n0_to_0 = 0
        n0_to_1 = 0

        n1 = 0
        n1_to_0 = 0
        n1_to_1 = 0

        for i in range(history_len, len(states)):
            history = states[i - history_len:i]
            next_state = states[i]

            if np.all(history == 0):
                n0 += 1
                if next_state == 0:
                    n0_to_0 += 1
                else:
                    n0_to_1 += 1

            if np.all(history == 1):
                n1 += 1
                if next_state == 0:
                    n1_to_0 += 1
                else:
                    n1_to_1 += 1

        rows.append({
            "history_len_frames": history_len,
            "P_collapse_after_expanded_history": n0_to_1 / n0 if n0 > 0 else np.nan,
            "P_stay_expanded_after_expanded_history": n0_to_0 / n0 if n0 > 0 else np.nan,
            "P_escape_after_collapsed_history": n1_to_0 / n1 if n1 > 0 else np.nan,
            "P_stay_collapsed_after_collapsed_history": n1_to_1 / n1 if n1 > 0 else np.nan,
            "n_expanded_histories": n0,
            "n_collapsed_histories": n1,
        })

    return pd.DataFrame(rows)


def dwell_summary_df(states, dt_ns):
    rows = []

    for state_value, state_name in [(0, "Expanded"), (1, "Collapsed")]:
        dwell_frames = run_lengths(states, state_value)
        dwell_ns = dwell_frames * dt_ns

        if len(dwell_ns) == 0:
            rows.append({
                "state": state_name,
                "n_dwells": 0,
                "mean_dwell_ns": np.nan,
                "median_dwell_ns": np.nan,
                "max_dwell_ns": np.nan,
                "k_from_dwell_per_ns": np.nan,
            })
            continue

        mean_tau = np.mean(dwell_ns)

        rows.append({
            "state": state_name,
            "n_dwells": len(dwell_ns),
            "mean_dwell_ns": mean_tau,
            "median_dwell_ns": np.median(dwell_ns),
            "max_dwell_ns": np.max(dwell_ns),
            "k_from_dwell_per_ns": 1 / mean_tau if mean_tau > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def analyse_replicate(design_name, rep_label, csv_path, dt_ns, lags, history_lengths, outdir):
    df = pd.read_csv(csv_path)
    states = df["state"].to_numpy().astype(int)

    rep_out = outdir / design_name / rep_label
    rep_out.mkdir(parents=True, exist_ok=True)

    ck = chapman_kolmogorov_df(states, dt_ns, lags)
    ck["design"] = design_name
    ck["replicate"] = rep_label
    ck.to_csv(rep_out / f"{rep_label}_chapman_kolmogorov.csv", index=False)

    mem = memory_test_df(states, history_lengths)
    mem["design"] = design_name
    mem["replicate"] = rep_label
    mem.to_csv(rep_out / f"{rep_label}_memory_test.csv", index=False)

    dwell = dwell_summary_df(states, dt_ns)
    dwell["design"] = design_name
    dwell["replicate"] = rep_label
    dwell.to_csv(rep_out / f"{rep_label}_dwell_summary.csv", index=False)

    _, P1 = transition_matrix(states, lag=1)
    P01 = P1[0, 1]
    P10 = P1[1, 0]

    summary = {
        "design": design_name,
        "replicate": rep_label,
        "n_frames": len(states),
        "fraction_collapsed": np.mean(states),
        "P01_collapse": P01,
        "P10_escape": P10,
        "k_collapse_per_ns": P01 / dt_ns,
        "k_escape_per_ns": P10 / dt_ns,
        "mean_expanded_dwell_ns": dwell.loc[dwell["state"] == "Expanded", "mean_dwell_ns"].iloc[0],
        "mean_collapsed_dwell_ns": dwell.loc[dwell["state"] == "Collapsed", "mean_dwell_ns"].iloc[0],
        "max_expanded_dwell_ns": dwell.loc[dwell["state"] == "Expanded", "max_dwell_ns"].iloc[0],
        "max_collapsed_dwell_ns": dwell.loc[dwell["state"] == "Collapsed", "max_dwell_ns"].iloc[0],
    }

    return {
        "states": states,
        "ck": ck,
        "memory": mem,
        "dwell": dwell,
        "summary": summary,
    }


def mean_std_by_x(df, group_col, value_cols):
    grouped = df.groupby(group_col)
    mean = grouped[value_cols].mean().reset_index()
    std = grouped[value_cols].std().reset_index()

    for col in value_cols:
        mean[f"{col}_std"] = std[col]

    return mean


def parse_design_arg(design_arg):
    parts = design_arg.split("::")
    if len(parts) < 2:
        raise ValueError(
            "Design format must be: DESIGN_NAME::csv1::csv2::csv3"
        )

    design_name = parts[0]
    paths = [Path(p) for p in parts[1:]]

    return design_name, paths


def plot_ck_comparison(design_results, outpath):
    plt.figure(figsize=(8, 5.5))

    for design_name, result in design_results.items():
        ck_all = result["ck_all"]

        mean_df = mean_std_by_x(
            ck_all,
            "lag_ns",
            ["P01_observed", "P01_predicted", "P10_observed", "P10_predicted"],
        )

        plt.plot(
            mean_df["lag_ns"],
            mean_df["P01_observed"],
            marker="o",
            label=f"{design_name} P01 observed",
        )

        plt.plot(
            mean_df["lag_ns"],
            mean_df["P01_predicted"],
            linestyle="--",
            marker="x",
            label=f"{design_name} P01 predicted",
        )

        plt.plot(
            mean_df["lag_ns"],
            mean_df["P10_observed"],
            marker="s",
            label=f"{design_name} P10 observed",
        )

        plt.plot(
            mean_df["lag_ns"],
            mean_df["P10_predicted"],
            linestyle="--",
            marker="+",
            label=f"{design_name} P10 predicted",
        )

    plt.xlabel("Lag time (ns)")
    plt.ylabel("Transition probability")
    plt.title("Chapman-Kolmogorov comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_memory_comparison(design_results, outpath):
    plt.figure(figsize=(8, 5.5))

    for design_name, result in design_results.items():
        memory_all = result["memory_all"]

        mean_df = mean_std_by_x(
            memory_all,
            "history_len_frames",
            [
                "P_collapse_after_expanded_history",
                "P_escape_after_collapsed_history",
            ],
        )

        plt.plot(
            mean_df["history_len_frames"],
            mean_df["P_collapse_after_expanded_history"],
            marker="o",
            label=f"{design_name}: P(collapse | expanded history)",
        )

        plt.plot(
            mean_df["history_len_frames"],
            mean_df["P_escape_after_collapsed_history"],
            marker="s",
            label=f"{design_name}: P(escape | collapsed history)",
        )

    plt.xlabel("Required previous same-state history (frames)")
    plt.ylabel("Next-frame transition probability")
    plt.title("Memory comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def collect_dwell_times(states_list, dt_ns, state_value):
    all_dwells = []

    for states in states_list:
        dwell_frames = run_lengths(states, state_value)
        all_dwells.extend((dwell_frames * dt_ns).tolist())

    return np.array(all_dwells, dtype=float)


def plot_dwell_comparison(design_results, outpath, state_value, state_name):
    plt.figure(figsize=(8, 5.5))

    for design_name, result in design_results.items():
        dwell_ns = collect_dwell_times(result["states_list"], result["dt_ns"], state_value)

        if len(dwell_ns) == 0:
            continue

        mean_tau = np.mean(dwell_ns)
        tmax = np.percentile(dwell_ns, 99)
        bins = np.linspace(0, tmax, 35)

        plt.hist(
            dwell_ns,
            bins=bins,
            density=True,
            alpha=0.35,
            label=f"{design_name} measured",
        )

        tgrid = np.linspace(0, tmax, 300)
        pdf = (1 / mean_tau) * np.exp(-tgrid / mean_tau)

        plt.plot(
            tgrid,
            pdf,
            linewidth=2,
            label=f"{design_name} exp fit, τ={mean_tau:.3f} ns",
        )

    plt.xlabel("Dwell time (ns)")
    plt.ylabel("Probability density")
    plt.title(f"{state_name} dwell-time comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def add_memory_metrics(summary_df, memory_all):
    rows = []

    for _, row in summary_df.iterrows():
        design = row["design"]
        replicate = row["replicate"]

        m = memory_all[
            (memory_all["design"] == design)
            & (memory_all["replicate"] == replicate)
        ]

        m1 = m[m["history_len_frames"] == 1]
        m20 = m[m["history_len_frames"] == 20]

        if len(m1) == 0 or len(m20) == 0:
            continue

        Pcoll_1 = m1["P_collapse_after_expanded_history"].iloc[0]
        Pcoll_20 = m20["P_collapse_after_expanded_history"].iloc[0]

        Pesc_1 = m1["P_escape_after_collapsed_history"].iloc[0]
        Pesc_20 = m20["P_escape_after_collapsed_history"].iloc[0]

        row = row.to_dict()

        row["memory_drop_collapse"] = Pcoll_1 - Pcoll_20
        row["memory_ratio_collapse"] = Pcoll_20 / Pcoll_1 if Pcoll_1 > 0 else np.nan

        row["memory_drop_escape"] = Pesc_1 - Pesc_20
        row["memory_ratio_escape"] = Pesc_20 / Pesc_1 if Pesc_1 > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--design",
        action="append",
        required=True,
        help=(
            "Format: DESIGN_NAME::csv1::csv2::csv3. "
            "Use one csv if only one run is available."
        ),
    )

    parser.add_argument("--dt-ns", type=float, default=0.01)
    parser.add_argument("--outdir", default="markov_design_comparison")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lags = [1, 2, 5, 10, 20, 50, 100, 200]
    history_lengths = [1, 2, 3, 5, 10, 20]

    design_results = {}
    all_summaries = []
    all_ck = []
    all_memory = []
    all_dwell = []

    for design_arg in args.design:
        design_name, paths = parse_design_arg(design_arg)

        print(f"\nAnalysing design: {design_name}")
        print(f"Number of replicates/files: {len(paths)}")

        states_list = []
        ck_list = []
        memory_list = []
        dwell_list = []
        summary_list = []

        for i, path in enumerate(paths, start=1):
            rep_label = f"{design_name}_rep{i:02d}"
            print(f"  {rep_label}: {path}")

            result = analyse_replicate(
                design_name=design_name,
                rep_label=rep_label,
                csv_path=path,
                dt_ns=args.dt_ns,
                lags=lags,
                history_lengths=history_lengths,
                outdir=outdir,
            )

            states_list.append(result["states"])
            ck_list.append(result["ck"])
            memory_list.append(result["memory"])
            dwell_list.append(result["dwell"])
            summary_list.append(result["summary"])

        ck_all = pd.concat(ck_list, ignore_index=True)
        memory_all = pd.concat(memory_list, ignore_index=True)
        dwell_all = pd.concat(dwell_list, ignore_index=True)
        summary_df = pd.DataFrame(summary_list)

        summary_with_memory = add_memory_metrics(summary_df, memory_all)

        design_dir = outdir / design_name
        design_dir.mkdir(parents=True, exist_ok=True)

        ck_all.to_csv(design_dir / f"{design_name}_all_ck.csv", index=False)
        memory_all.to_csv(design_dir / f"{design_name}_all_memory.csv", index=False)
        dwell_all.to_csv(design_dir / f"{design_name}_all_dwell.csv", index=False)
        summary_with_memory.to_csv(design_dir / f"{design_name}_summary_per_rep.csv", index=False)

        mean_summary = summary_with_memory.select_dtypes(include=[np.number]).agg(["mean", "std"]).T
        mean_summary.to_csv(design_dir / f"{design_name}_summary_mean_std.csv")

        design_results[design_name] = {
            "states_list": states_list,
            "ck_all": ck_all,
            "memory_all": memory_all,
            "dwell_all": dwell_all,
            "summary": summary_with_memory,
            "dt_ns": args.dt_ns,
        }

        all_summaries.append(summary_with_memory)
        all_ck.append(ck_all)
        all_memory.append(memory_all)
        all_dwell.append(dwell_all)

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(outdir / "combined_summary_per_rep.csv", index=False)

    numeric_cols = combined_summary.select_dtypes(include=[np.number]).columns

    combined_mean_std = (
        combined_summary
        .groupby("design")[numeric_cols]
        .agg(["mean", "std"])
    )

    combined_mean_std.to_csv(outdir / "combined_summary_mean_std.csv")

    plot_ck_comparison(
        design_results,
        outdir / "comparison_chapman_kolmogorov.png",
    )

    plot_memory_comparison(
        design_results,
        outdir / "comparison_memory_test.png",
    )

    plot_dwell_comparison(
        design_results,
        outdir / "comparison_expanded_dwell_distribution.png",
        state_value=0,
        state_name="Expanded",
    )

    plot_dwell_comparison(
        design_results,
        outdir / "comparison_collapsed_dwell_distribution.png",
        state_value=1,
        state_name="Collapsed",
    )

    print("\nDone.")
    print("Output written to:", outdir)
    print("\nCombined summary:")
    print(combined_summary)


if __name__ == "__main__":
    main()