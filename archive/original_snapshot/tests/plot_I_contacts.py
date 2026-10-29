import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import self_distance_array


# ============================================================
# SETTINGS
# ============================================================

dt_ps = 0.01
save_every = 5000
frame_dt_ns = dt_ps * save_every / 1000.0   # 0.05 ns/frame written

contact_cutoff_nm = 1.2
skip_frames = 0          # skip only the first N frames
stride = 10             # analyze every 10th frame; set to 1 for all frames

systems = [
    {
        "label": "E_AI_350K_scale6",
        "top":  r"E:\Master thesis\calvados_tea\runs\scale1\E_AI_triblock\6.0\350K\E_AI_triblock\top.pdb",
        "traj": r"E:\Master thesis\calvados_tea\runs\scale1\E_AI_triblock\6.0\350K\E_AI_triblock\E_AI_triblock.dcd"
    },
    {
        "label": "E_AE_350K_scale6",
        "top":  r"E:\Master thesis\calvados_tea\runs\scale1\E_AE_triblock\6.0\350K\E_AE_triblock\top.pdb",
        "traj": r"E:\Master thesis\calvados_tea\runs\scale1\E_AE_triblock\6.0\350K\E_AE_triblock\E_AE_triblock.dcd"
    }
]

LOG_EPS = 1e-4


# ============================================================
# HELPERS
# ============================================================

def select_chain_beads(u):
    """
    Assumes one bead per residue (CALVADOS).
    """
    ag = u.atoms
    if len(ag) == 0:
        raise ValueError("No atoms found in topology.")

    n_res = len(ag.residues)
    if len(ag) != n_res:
        print(
            f"Warning: atoms ({len(ag)}) != residues ({n_res}). "
            "This script assumes one bead per residue."
        )

    print(f"Using all beads: {len(ag)} atoms / {n_res} residues")
    return ag


def analyze_all_contacts_fast(u, chain_atoms, cutoff_nm=1.2, skip_frames=0, stride=1):
    """
    Fast full-chain contact analysis for one-bead-per-residue chains.

    Outputs:
    - mean nearest-residue distance vs time
    - mean all-pair contact fraction vs time
    - average contact probability map
    - contact probability vs sequence separation

    Important:
    - seq_sep_contact_prob is normalized correctly and must stay in [0, 1]
    """
    n = len(chain_atoms)
    if n < 2:
        raise ValueError("Need at least 2 residues/beads.")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    # Upper triangle pair indices
    iu = np.triu_indices(n, k=1)
    pair_i = iu[0]
    pair_j = iu[1]
    pair_sep = pair_j - pair_i  # sequence separation for each unique pair

    # Number of unique pairs for each separation
    seq_sep = np.arange(1, n)
    seq_sep_pair_count = np.bincount(pair_sep, minlength=n)[1:].astype(np.int64)

    # Accumulators
    contact_counts = np.zeros((n, n), dtype=np.float32)
    mean_contact_fraction_series = []
    mean_min_dist_series = []
    seq_sep_contact_sum = np.zeros(n - 1, dtype=np.float64)

    # Reusable full symmetric distance matrix
    full_d = np.empty((n, n), dtype=np.float32)

    n_frames = 0

    for ts in u.trajectory[skip_frames::stride]:
        # Unique pair distances in Å; convert to nm
        d_upper = self_distance_array(
            chain_atoms.positions,
            box=ts.dimensions
        ).astype(np.float32) / 10.0

        # Fill full symmetric distance matrix
        full_d.fill(np.inf)
        full_d[iu] = d_upper
        full_d[(iu[1], iu[0])] = d_upper

        # Mean nearest-residue distance
        min_per_res = np.min(full_d, axis=1)
        mean_min_dist_series.append(float(np.mean(min_per_res)))

        # Unique pair contacts
        contact_upper = d_upper < cutoff_nm
        mean_contact_fraction_series.append(float(np.mean(contact_upper)))

        # Accumulate full symmetric contact map
        contact_upper_f = contact_upper.astype(np.float32)
        contact_counts[iu] += contact_upper_f
        contact_counts[(iu[1], iu[0])] += contact_upper_f

        # Accumulate number of contacting unique pairs by sequence separation
        # bincount returns counts for separation 0..n-1; we drop separation 0
        seq_sep_contact_sum += np.bincount(
            pair_sep,
            weights=contact_upper_f,
            minlength=n
        )[1:]

        n_frames += 1

    if n_frames == 0:
        raise ValueError("No frames analyzed. Check skip_frames / stride.")

    avg_contact_map = contact_counts / float(n_frames)

    # Proper normalization:
    # average over frames and over the number of unique pairs at each sequence separation
    seq_sep_contact_prob = seq_sep_contact_sum / (seq_sep_pair_count * float(n_frames))

    # Safety check
    if np.any(seq_sep_contact_prob < -1e-8) or np.any(seq_sep_contact_prob > 1.0 + 1e-8):
        raise ValueError(
            "Sequence-separation contact probability is outside [0, 1]. "
            "Normalization is wrong."
        )

    return {
        "mean_min_dist_series": np.asarray(mean_min_dist_series, dtype=np.float32),
        "mean_contact_fraction_series": np.asarray(mean_contact_fraction_series, dtype=np.float32),
        "avg_contact_map": avg_contact_map,
        "seq_sep": seq_sep,
        "seq_sep_contact_prob": seq_sep_contact_prob,
        "n_frames": n_frames,
    }


def safe_log10(arr, eps=LOG_EPS):
    return np.log10(np.clip(arr, eps, None))


def plot_contact_map_log(contact_map, label, outname, eps=LOG_EPS):
    log_map = safe_log10(contact_map, eps=eps)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        log_map,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=np.log10(eps),
        vmax=0.0
    )
    cbar = plt.colorbar(im)
    cbar.set_label("log10(contact probability)")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.title(f"Full-chain contact map (log scale)\n{label}")
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()


def plot_difference_map(map_a, map_b, label_a, label_b, outname):
    diff = map_a - map_b
    vmax = np.max(np.abs(diff))

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        diff,
        origin="lower",
        aspect="auto",
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Δ contact probability")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.title(f"Difference contact map\n{label_a} - {label_b}")
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()


def print_basic_stats(label, res, cutoff_nm):
    seqprob = res["seq_sep_contact_prob"]
    print(f"  Frames analyzed: {res['n_frames']}")
    print(f"  Mean nearest-residue distance: {res['mean_min_dist_series'].mean():.3f} nm")
    print(f"  Mean all-pair contact fraction (< {cutoff_nm} nm): {res['mean_contact_fraction_series'].mean():.6f}")
    print(f"  Seq-sep contact probability min/max: {seqprob.min():.6f} / {seqprob.max():.6f}")


# ============================================================
# MAIN
# ============================================================

results = {}

for sys in systems:
    print(f"\nLoading: {sys['label']}")
    u = mda.Universe(sys["top"], sys["traj"])
    chain_atoms = select_chain_beads(u)

    res = analyze_all_contacts_fast(
        u,
        chain_atoms,
        cutoff_nm=contact_cutoff_nm,
        skip_frames=skip_frames,
        stride=stride,
    )

    # Time spacing of analyzed frames increases with stride
    res["time_ns"] = np.arange(len(res["mean_contact_fraction_series"])) * frame_dt_ns * stride
    res["n_residues"] = len(chain_atoms)

    results[sys["label"]] = res

    print_basic_stats(sys["label"], res, contact_cutoff_nm)
    print(f"  Number of residues/beads: {res['n_residues']}")


# ============================================================
# PLOT 1: separate full contact maps on log scale
# ============================================================

for label, r in results.items():
    plot_contact_map_log(
        contact_map=r["avg_contact_map"],
        label=label,
        outname=f"{label}_full_contact_map_log.png",
        eps=LOG_EPS
    )

print("Saved separate log-scale contact maps.")


# ============================================================
# PLOT 2: separate sequence-separation curves on log scale
# ============================================================

for label, r in results.items():
    plt.figure(figsize=(8, 5))
    plt.plot(
        r["seq_sep"],
        np.clip(r["seq_sep_contact_prob"], LOG_EPS, None),
        linewidth=2
    )
    plt.yscale("log")
    plt.xlabel("Sequence separation |i-j|")
    plt.ylabel(f"Contact probability (< {contact_cutoff_nm} nm)")
    plt.title(f"Contact probability vs sequence separation\n{label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{label}_contact_vs_seqsep_log.png", dpi=300)
    plt.close()

print("Saved separate sequence-separation plots.")


# ============================================================
# PLOT 3: difference map
# ============================================================

if len(systems) == 2:
    label_a = systems[0]["label"]
    label_b = systems[1]["label"]

    map_a = results[label_a]["avg_contact_map"]
    map_b = results[label_b]["avg_contact_map"]

    plot_difference_map(
        map_a=map_a,
        map_b=map_b,
        label_a=label_a,
        label_b=label_b,
        outname="full_contact_map_difference.png"
    )

    print("Saved difference contact map.")


# ============================================================
# PLOT 4: difference in sequence-separation contact probability
# ============================================================

if len(systems) == 2:
    label_a = systems[0]["label"]
    label_b = systems[1]["label"]

    sep_a = results[label_a]["seq_sep"]
    sep_b = results[label_b]["seq_sep"]

    if not np.array_equal(sep_a, sep_b):
        raise ValueError("Sequence-separation arrays do not match.")

    diff_sep = (
        results[label_a]["seq_sep_contact_prob"]
        - results[label_b]["seq_sep_contact_prob"]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(sep_a, diff_sep, linewidth=2)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Sequence separation |i-j|")
    plt.ylabel("Δ contact probability")
    plt.title(f"Difference in contact probability vs sequence separation\n{label_a} - {label_b}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("contact_vs_seqsep_difference.png", dpi=300)
    plt.close()

    print("Saved difference sequence-separation plot.")


# ============================================================
# PLOT 5: mean contact fraction vs time
# ============================================================

plt.figure(figsize=(8, 5))
for label, r in results.items():
    plt.plot(r["time_ns"], r["mean_contact_fraction_series"], label=label)

plt.xlabel("Time (ns)")
plt.ylabel(f"Mean all-pair contact fraction (< {contact_cutoff_nm} nm)")
plt.title("Full-chain mean contact fraction vs time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("full_chain_contact_fraction_timeseries.png", dpi=300)
plt.close()

print("Saved contact fraction timeseries.")
print("Done.")