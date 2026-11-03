import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_column(df, candidates):
    """Find first matching column, case-insensitive."""
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def extract_arch_number(value):
    """Extract leading architecture number from ID."""
    match = re.match(r"(\d+)", str(value))
    return int(match.group(1)) if match else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Plot Rg through time for a selected architecture."
    )

    parser.add_argument("--rg-file", required=True)
    parser.add_argument("--arch", required=True, type=int)
    parser.add_argument("--temperature", default="370K")
    parser.add_argument("--threshold", type=float, default=4.30)
    parser.add_argument(
        "--frame-time",
        type=float,
        default=0.01,
        help="Time per stored frame in ns (default: 0.01 ns)"
    )
    parser.add_argument(
        "--outdir",
        default="rg_time_plots"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading: {args.rg_file}")
    df = pd.read_csv(args.rg_file)

    print(f"Rows loaded: {len(df):,}")
    print("Columns:", list(df.columns))

    # --------------------------------------------------
    # Detect columns
    # --------------------------------------------------

    id_col = find_column(
        df,
        ["id", "architecture_id", "architecture", "name"]
    )

    temp_col = find_column(
        df,
        ["temp", "temperature"]
    )

    rg_col = find_column(
        df,
        ["rg_nm", "rg", "radius_of_gyration", "radius_gyration_nm"]
    )

    frame_col = find_column(
        df,
        ["frame", "frame_number", "frame_idx", "frame_index"]
    )

    if id_col is None:
        raise ValueError("Could not find architecture ID column.")

    if rg_col is None:
        raise ValueError("Could not find Rg column.")

    print(f"ID column:          {id_col}")
    print(f"Temperature column: {temp_col}")
    print(f"Rg column:          {rg_col}")
    print(f"Frame column:       {frame_col}")

    # --------------------------------------------------
    # Select architecture
    # --------------------------------------------------

    df["_arch_number"] = df[id_col].apply(extract_arch_number)

    selected = df[df["_arch_number"] == args.arch].copy()

    if temp_col is not None:
        temp_string = str(args.temperature).replace("K", "").strip()

        selected = selected[
            selected[temp_col]
            .astype(str)
            .str.replace("K", "", regex=False)
            .str.strip()
            == temp_string
        ].copy()

    if selected.empty:
        raise ValueError(
            f"No data found for architecture {args.arch} "
            f"at {args.temperature}"
        )

    # --------------------------------------------------
    # Sort by frame
    # --------------------------------------------------

    if frame_col is not None:
        selected = selected.sort_values(frame_col)
        frames = selected[frame_col].to_numpy()
    else:
        frames = np.arange(len(selected))

    rg = selected[rg_col].astype(float).to_numpy()

    # Time starts at zero
    time_ns = (frames - frames.min()) * args.frame_time

    collapsed = rg < args.threshold

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------

    fraction_collapsed = collapsed.mean()

    n = len(rg)
    quarter_edges = np.linspace(0, n, 5, dtype=int)

    quarter_fractions = []

    for i in range(4):
        start = quarter_edges[i]
        end = quarter_edges[i + 1]

        frac = collapsed[start:end].mean()
        quarter_fractions.append(frac)

    print()
    print("=" * 60)
    print(f"Architecture:       {args.arch}")
    print(f"Temperature:        {args.temperature}")
    print(f"Frames:             {n:,}")
    print(f"Simulation time:    {time_ns[-1]:.2f} ns")
    print(f"Collapse threshold: {args.threshold:.2f} nm")
    print(f"Fraction collapsed: {fraction_collapsed:.4f}")
    print()
    print("Collapsed fraction by simulation quarter:")

    for i, frac in enumerate(quarter_fractions, start=1):
        print(f"  Q{i}: {frac:.4f}")

    print("=" * 60)

    # --------------------------------------------------
    # Cumulative collapsed fraction
    # --------------------------------------------------

    cumulative_fraction = (
        np.cumsum(collapsed) /
        np.arange(1, len(collapsed) + 1)
    )

    # --------------------------------------------------
    # Plot 1: Rg through time
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        time_ns,
        rg,
        linewidth=0.7,
        alpha=0.8
    )

    ax.axhline(
        args.threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Collapse threshold ({args.threshold:.2f} nm)"
    )

    # Highlight collapsed frames
    ax.scatter(
        time_ns[collapsed],
        rg[collapsed],
        s=4,
        alpha=0.5,
        label="Collapsed"
    )

    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("Rg (nm)")
    ax.set_title(
        f"Architecture {args.arch} – {args.temperature}"
    )

    ax.legend()
    fig.tight_layout()

    outfile = os.path.join(
        args.outdir,
        f"{args.arch}_{args.temperature}_rg_vs_time.png"
    )

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"Saved: {outfile}")

    # --------------------------------------------------
    # Plot 2: cumulative collapsed fraction
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        time_ns,
        cumulative_fraction,
        linewidth=1.5
    )

    ax.axhline(
        fraction_collapsed,
        linestyle="--",
        linewidth=1,
        label=f"Final = {fraction_collapsed:.3f}"
    )

    ax.set_xlabel("Simulation time (ns)")
    ax.set_ylabel("Cumulative fraction collapsed")
    ax.set_ylim(0, 1)

    ax.set_title(
        f"Convergence of collapsed population – "
        f"architecture {args.arch}"
    )

    ax.legend()
    fig.tight_layout()

    outfile = os.path.join(
        args.outdir,
        f"{args.arch}_{args.temperature}_cumulative_collapse.png"
    )

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"Saved: {outfile}")

    # --------------------------------------------------
    # Plot 3: collapsed fraction per quarter
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        ["Q1", "Q2", "Q3", "Q4"],
        quarter_fractions
    )

    ax.axhline(
        fraction_collapsed,
        linestyle="--",
        linewidth=1,
        label="Full trajectory"
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction collapsed")
    ax.set_xlabel("Simulation quarter")

    ax.set_title(
        f"Collapse distribution through time – "
        f"architecture {args.arch}"
    )

    ax.legend()
    fig.tight_layout()

    outfile = os.path.join(
        args.outdir,
        f"{args.arch}_{args.temperature}_collapse_quarters.png"
    )

    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()