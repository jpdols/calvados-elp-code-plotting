import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda


def parse_run(run_string):
    parts = run_string.split("::")
    if len(parts) != 3:
        raise ValueError("Run must be: label::topology.pdb::trajectory.dcd")
    return parts[0], Path(parts[1]), Path(parts[2])


def kabsch_align(P, Q):
    """
    Align P onto Q.
    P and Q shape: (N, 3), already centered.
    Returns rotated P.
    """
    C = P.T @ Q
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    return P @ U


def gyration_tensor(pos):
    centered = pos - pos.mean(axis=0)
    return centered.T @ centered / len(pos)


def relative_shape_anisotropy(evals):
    l1, l2, l3 = np.sort(evals)[::-1]
    denom = (l1 + l2 + l3) ** 2
    if denom == 0:
        return np.nan
    return 1.5 * ((l1**2 + l2**2 + l3**2) / denom) - 0.5


def local_angles_deg(pos):
    v1 = pos[:-2] - pos[1:-1]
    v2 = pos[2:] - pos[1:-1]

    dot = np.sum(v1 * v2, axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

    valid = norm > 0
    cosang = np.full(len(dot), np.nan)
    cosang[valid] = np.clip(dot[valid] / norm[valid], -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def distance_matrix(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def analyse_run(label, top, traj, stride, frame_time, contact_cutoff, min_seq_sep, outdir):
    print(f"\nLoading {label}")
    print(f"  Topology:   {top}")
    print(f"  Trajectory: {traj}")

    u = mda.Universe(str(top), str(traj))
    atoms = u.atoms
    n = len(atoms)

    rows = []
    raw_positions = []
    contact_counts = np.zeros((n, n), dtype=np.float64)

    valid_contact_mask = np.ones((n, n), dtype=bool)
    for i in range(n):
        valid_contact_mask[i, max(0, i - min_seq_sep):min(n, i + min_seq_sep + 1)] = False
    np.fill_diagonal(valid_contact_mask, False)

    for frame_i, ts in enumerate(u.trajectory[::stride]):
        if frame_i % 100 == 0:
            print(f"  frame {frame_i}", flush=True)

        pos = atoms.positions.astype(float) / 10.0  # Å to nm
        raw_positions.append(pos.copy())

        gt = gyration_tensor(pos)
        evals = np.linalg.eigvalsh(gt)
        evals = np.sort(evals)[::-1]

        rg = np.sqrt(np.sum(evals))
        kappa2 = relative_shape_anisotropy(evals)

        l1, l2, l3 = evals
        asphericity = l1 - 0.5 * (l2 + l3)

        angles = local_angles_deg(pos)

        dmat = distance_matrix(pos)
        contacts = (dmat < contact_cutoff) & valid_contact_mask
        contact_counts += contacts.astype(float)

        contact_pairs = np.argwhere(np.triu(contacts, k=1))
        if len(contact_pairs) > 0:
            seq_sep = np.abs(contact_pairs[:, 0] - contact_pairs[:, 1])
            mean_contact_order = np.mean(seq_sep)
            norm_contact_order = mean_contact_order / n
            n_contacts = len(contact_pairs)
        else:
            mean_contact_order = np.nan
            norm_contact_order = np.nan
            n_contacts = 0

        rows.append({
            "label": label,
            "frame": frame_i,
            "time_ns": frame_i * stride * frame_time,
            "rg_nm": rg,
            "kappa2": kappa2,
            "asphericity_nm2": asphericity,
            "mean_local_angle_deg": np.nanmean(angles),
            "std_local_angle_deg": np.nanstd(angles),
            "n_contacts": n_contacts,
            "mean_contact_order_residues": mean_contact_order,
            "normalized_contact_order": norm_contact_order,
        })

    frame_df = pd.DataFrame(rows)
    raw_positions = np.array(raw_positions)

    contact_prob = contact_counts / len(frame_df)

    # Correct RMSF: remove translation and rotation by aligning all frames to first frame
    reference = raw_positions[0]
    reference_centered = reference - reference.mean(axis=0)

    aligned_positions = []
    for pos in raw_positions:
        centered = pos - pos.mean(axis=0)
        aligned = kabsch_align(centered, reference_centered)
        aligned_positions.append(aligned)

    aligned_positions = np.array(aligned_positions)
    mean_aligned = aligned_positions.mean(axis=0)
    disp2 = np.sum((aligned_positions - mean_aligned) ** 2, axis=2)
    rmsf = np.sqrt(disp2.mean(axis=0))

    rmsf_df = pd.DataFrame({
        "label": label,
        "bead_index": np.arange(1, n + 1),
        "rmsf_nm": rmsf,
    })

    contact_df = pd.DataFrame(contact_prob)
    contact_df.index = np.arange(1, n + 1)
    contact_df.columns = np.arange(1, n + 1)

    frame_df.to_csv(outdir / f"{label}_frame_metrics.csv", index=False)
    rmsf_df.to_csv(outdir / f"{label}_aligned_rmsf.csv", index=False)
    contact_df.to_csv(outdir / f"{label}_contact_probability_matrix.csv")

    summary = {
        "label": label,
        "n_beads": n,
        "n_frames": len(frame_df),
        "mean_rg_nm": frame_df["rg_nm"].mean(),
        "std_rg_nm": frame_df["rg_nm"].std(),
        "cv_rg": frame_df["rg_nm"].std() / frame_df["rg_nm"].mean(),
        "mean_kappa2": frame_df["kappa2"].mean(),
        "std_kappa2": frame_df["kappa2"].std(),
        "median_kappa2": frame_df["kappa2"].median(),
        "mean_asphericity_nm2": frame_df["asphericity_nm2"].mean(),
        "mean_local_angle_deg": frame_df["mean_local_angle_deg"].mean(),
        "mean_angle_width_deg": frame_df["std_local_angle_deg"].mean(),
        "mean_contacts": frame_df["n_contacts"].mean(),
        "mean_contact_order_residues": frame_df["mean_contact_order_residues"].mean(),
        "normalized_contact_order": frame_df["normalized_contact_order"].mean(),
        "mean_aligned_rmsf_nm": rmsf_df["rmsf_nm"].mean(),
        "max_aligned_rmsf_nm": rmsf_df["rmsf_nm"].max(),
    }

    return frame_df, rmsf_df, contact_prob, summary


def plot_distributions(all_frame_df, all_rmsf_df, outdir):
    metrics = [
        ("rg_nm", "Rg (nm)", "distribution_rg.png"),
        ("kappa2", "Relative shape anisotropy κ²", "distribution_kappa2.png"),
        ("asphericity_nm2", "Asphericity (nm²)", "distribution_asphericity.png"),
        ("mean_local_angle_deg", "Mean local angle (degrees)", "distribution_mean_angle.png"),
        ("std_local_angle_deg", "Local angle width (degrees)", "distribution_angle_width.png"),
        ("n_contacts", "Number of contacts", "distribution_contacts.png"),
        ("normalized_contact_order", "Normalized contact order", "distribution_contact_order.png"),
    ]

    for metric, xlabel, filename in metrics:
        plt.figure(figsize=(8, 5))
        for label, sub in all_frame_df.groupby("label"):
            values = sub[metric].dropna().values
            plt.hist(values, bins=40, density=True, alpha=0.35, label=label)

        plt.xlabel(xlabel)
        plt.ylabel("Probability density")
        plt.title(f"{xlabel} distribution")
        plt.grid(alpha=0.3, linestyle=":")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    for label, sub in all_rmsf_df.groupby("label"):
        plt.hist(sub["rmsf_nm"].dropna().values, bins=40, density=True, alpha=0.35, label=label)

    plt.xlabel("Aligned RMSF (nm)")
    plt.ylabel("Probability density")
    plt.title("Aligned RMSF distribution")
    plt.grid(alpha=0.3, linestyle=":")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "distribution_aligned_rmsf.png", dpi=300)
    plt.close()


def plot_rmsf(all_rmsf_df, outdir):
    plt.figure(figsize=(9, 5))
    for label, sub in all_rmsf_df.groupby("label"):
        plt.plot(sub["bead_index"], sub["rmsf_nm"], label=label, linewidth=1.5)

    plt.xlabel("Bead / residue index")
    plt.ylabel("Aligned RMSF (nm)")
    plt.title("Per-bead RMSF after translation/rotation alignment")
    plt.grid(alpha=0.3, linestyle=":")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "aligned_rmsf_per_bead.png", dpi=300)
    plt.close()


def plot_contact_map(label, contact_prob, outdir, vmax=1.0):
    plt.figure(figsize=(7, 6))
    plt.imshow(contact_prob, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    plt.colorbar(label="Contact probability")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.title(f"Contact probability map: {label}")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_contact_map.png", dpi=300)
    plt.close()


def plot_contact_difference(label_a, map_a, label_b, map_b, outdir):
    n = min(map_a.shape[0], map_b.shape[0])
    diff = map_a[:n, :n] - map_b[:n, :n]

    plt.figure(figsize=(7, 6))
    vmax = np.nanmax(np.abs(diff))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    plt.imshow(diff, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(label=f"Contact probability difference")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.title(f"Contact map difference: {label_a} - {label_b}")
    plt.tight_layout()
    plt.savefig(outdir / f"contact_map_difference_{label_a}_minus_{label_b}.png", dpi=300)
    plt.close()

    pd.DataFrame(diff).to_csv(outdir / f"contact_map_difference_{label_a}_minus_{label_b}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse shape distributions, aligned RMSF, and contact maps for CALVADOS ELP trajectories."
    )

    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run format: label::topology.pdb::trajectory.dcd",
    )

    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--contact-cutoff", type=float, default=0.8)
    parser.add_argument("--min-seq-sep", type=int, default=2)
    parser.add_argument("--outdir", default="shape_contact_analysis")
    parser.add_argument("--timestamp", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.timestamp:
        outdir = outdir / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    all_rmsf = []
    summaries = []
    contact_maps = {}

    for run_string in args.run:
        label, top, traj = parse_run(run_string)

        frame_df, rmsf_df, contact_prob, summary = analyse_run(
            label=label,
            top=top,
            traj=traj,
            stride=args.stride,
            frame_time=args.frame_time,
            contact_cutoff=args.contact_cutoff,
            min_seq_sep=args.min_seq_sep,
            outdir=outdir,
        )

        all_frames.append(frame_df)
        all_rmsf.append(rmsf_df)
        summaries.append(summary)
        contact_maps[label] = contact_prob

        plot_contact_map(label, contact_prob, outdir)

    all_frame_df = pd.concat(all_frames, ignore_index=True)
    all_rmsf_df = pd.concat(all_rmsf, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    all_frame_df.to_csv(outdir / "all_frame_metrics.csv", index=False)
    all_rmsf_df.to_csv(outdir / "all_aligned_rmsf.csv", index=False)
    summary_df.to_csv(outdir / "summary_shape_contacts.csv", index=False)

    plot_distributions(all_frame_df, all_rmsf_df, outdir)
    plot_rmsf(all_rmsf_df, outdir)

    labels = list(contact_maps.keys())
    if len(labels) >= 2:
        plot_contact_difference(labels[0], contact_maps[labels[0]], labels[1], contact_maps[labels[1]], outdir)

    print("\nDone.")
    print(f"Saved outputs to: {outdir.resolve()}")
    print(summary_df)


if __name__ == "__main__":
    main()