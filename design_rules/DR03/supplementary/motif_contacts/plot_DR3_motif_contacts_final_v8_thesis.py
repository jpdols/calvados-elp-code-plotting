#!/usr/bin/env python
"""
plot_DR3_motif_contacts_final.py

Final thesis-ready analysis for R4 nonlocal higher-stickiness-motif contacts
versus compact-state escape kinetics.

Expected in --root:
    nonlocal_BB_trajectory_summary.csv
    trajectory_replicate_mapping.csv        (optional validation)
    per_run_summary.csv

Outputs:
    figure_DR3_BB_burden_vs_kescape.pdf/png/svg
    figure_DR3_BB_burden_by_architecture.pdf/png/svg
    figure_DR3_BB_burden_vs_kescape_design_means.pdf/png/svg
    DR3_BB_contacts_merged_triplicates.csv
    DR3_BB_contacts_design_means.csv
    DR3_BB_contacts_statistics.csv

Core definitions
----------------
W_BB = sequence-weighted nonlocal B-B contact burden
     = contacts/frame * mean contact sequence separation
     = sum(|i-j| over contact hits) / selected frames

Escape-rate response:
    Δlog2(k_escape) =
    log2(k_escape trajectory / chemistry-specific R4 mean k_escape)

The trajectory-level plot pairs structural replicate r with kinetic replicate r.
Design-level statistics average the three paired trajectories per sequence first.
"""

from pathlib import Path
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, linregress


# ---------------------------------------------------------------------
# THESIS STYLE
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

ARCH_ORDER = [
    "Random shuffle",
    "Alternating",
    "Small blocks",
    "Medium blocks",
    "Large blocks",
    "Higher λ center",
    "Higher λ ends",
    "Diblock",
]

# Fixed semantic architecture colors. Matplotlib tab10 colors, stored explicitly
# so all final figures use the same architecture identity.
TAB10 = plt.get_cmap("tab10").colors
ARCH_COLORS = {
    "Random shuffle": "#7f7f7f",
    "Alternating": "#1f77b4",
    "Small blocks": "#2ca02c",
    "Medium blocks": "#9467bd",
    "Large blocks": "#ff7f0e",
    "Higher λ center": "#d62728",
    "Higher λ ends": "#17becf",
    "Diblock": "#8c564b",
}

FAMILY_ORDER = ["VGIP", "VGIG", "VAIP", "VAIG"]
FAMILY_MARKERS = {
    "VGIP": "o",
    "VGIG": "s",
    "VAIP": "^",
    "VAIG": "D",
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates, required=True, label=None):
    by_norm = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in by_norm:
            return by_norm[key]
    if required:
        raise ValueError(
            f"Could not find {label or candidates[0]!r}.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def architecture_name(x):
    s = str(x).strip()
    q = norm(s)

    if "random" in q or "shuffle" in q:
        return "Random shuffle"
    if "alternat" in q or q == "even" or "evenly" in q:
        return "Alternating"
    if "small" in q:
        return "Small blocks"
    if "medium" in q:
        return "Medium blocks"
    if "large" in q:
        return "Large blocks"
    if "diblock" in q:
        return "Diblock"

    # center before generic higher/lambda logic
    if "center" in q or "centre" in q:
        return "Higher λ center"
    if "endcap" in q or "ends" in q or "end" in q:
        return "Higher λ ends"

    return s


def family_from_text(x):
    s = str(x).upper()
    for fam in FAMILY_ORDER:
        if fam in s:
            return fam
    return np.nan


def ratio_from_text(x):
    m = re.search(r"(?:^|[_\-\s])(R[1-6])(?:[_\-\s]|$)", str(x), flags=re.I)
    return m.group(1).upper() if m else np.nan


def replicate_from_text(x):
    s = str(x)
    patterns = [
        r"(?:replicate|rep|run|seed)[_\-\s]*(\d+)",
        r"[_\-\s]r(\d+)(?:[_\-\s]|$)",
    ]
    for p in patterns:
        m = re.search(p, s, flags=re.I)
        if m:
            return int(m.group(1))
    return np.nan


def clean_id(x):
    s = str(x).strip()
    # Keep numerical sequence IDs stable if pandas read them as 103.0.
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def save_figure(fig, stem):
    for ext in ("pdf", "png", "svg"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(stem.with_suffix("." + ext), **kwargs)
    plt.close(fig)


def correlation_stats(x, y, level):
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return {
            "level": level, "n": len(d),
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "slope": np.nan, "intercept": np.nan,
            "r_squared": np.nan,
        }

    pr, pp = pearsonr(d["x"], d["y"])
    sr, sp = spearmanr(d["x"], d["y"])
    lr = linregress(d["x"], d["y"])
    return {
        "level": level,
        "n": len(d),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_rho": sr,
        "spearman_p": sp,
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r_squared": lr.rvalue ** 2,
    }


def load_structure(path):
    raw = pd.read_csv(path)

    id_col = find_col(
        raw,
        ["id", "sequence_id", "sequence", "design_id", "name"],
        label="sequence/design ID",
    )
    rep_col = find_col(
        raw,
        ["replicate", "replicate_id", "rep", "run", "trajectory_replicate"],
        required=False,
    )
    fam_col = find_col(raw, ["family", "chemistry", "group"], required=False)
    ratio_col = find_col(raw, ["ratio", "composition_ratio"], required=False)
    arch_col = find_col(
        raw,
        ["architecture_plot", "architecture", "architecture_type", "arch"],
        label="architecture",
    )

    burden_col = find_col(
        raw,
        [
            "sequence_weighted_nonlocal_BB_contact_burden",
            "sequence_weighted_BB_contact_burden",
            "sequence_weighted_contact_burden",
            "weighted_contact_burden",
        ],
        required=False,
    )
    contacts_col = find_col(
        raw,
        [
            "mean_contact_hits_per_selected_frame",
            "mean_nonlocal_BB_contacts_per_selected_frame",
            "contacts_per_selected_frame",
            "mean_contacts_per_selected_frame",
        ],
        required=False,
    )
    sep_col = find_col(
        raw,
        [
            "mean_contact_sequence_separation_aa",
            "mean_sequence_separation_aa",
            "mean_contact_sequence_separation",
        ],
        required=False,
    )

    if burden_col is None and (contacts_col is None or sep_col is None):
        raise ValueError(
            "Need either the sequence-weighted burden column, or both "
            "contacts/frame and mean contact sequence separation columns.\n"
            f"Available columns:\n{list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["sequence_id"] = raw[id_col].map(clean_id)
    out["architecture"] = raw[arch_col].map(architecture_name)

    if rep_col is not None:
        out["replicate"] = pd.to_numeric(raw[rep_col], errors="coerce")
    else:
        # Last-resort inference from trajectory/path columns.
        path_col = find_col(
            raw,
            ["trajectory_path", "dcd_path", "path", "replicate_source"],
            required=False,
        )
        if path_col is None:
            raise ValueError(
                "Structural summary has no replicate column and no trajectory "
                "path from which replicate can be inferred."
            )
        out["replicate"] = raw[path_col].map(replicate_from_text)

    if fam_col is not None:
        out["family"] = raw[fam_col].astype(str).str.upper().map(family_from_text)
    else:
        out["family"] = raw[id_col].map(family_from_text)

    if ratio_col is not None:
        out["ratio"] = raw[ratio_col].astype(str).str.upper()
    else:
        out["ratio"] = raw[id_col].map(ratio_from_text)

    if burden_col is not None:
        out["W_BB"] = pd.to_numeric(raw[burden_col], errors="coerce")
    else:
        out["W_BB"] = (
            pd.to_numeric(raw[contacts_col], errors="coerce")
            * pd.to_numeric(raw[sep_col], errors="coerce")
        )

    if contacts_col is not None:
        out["contacts_per_frame"] = pd.to_numeric(raw[contacts_col], errors="coerce")
    if sep_col is not None:
        out["mean_sequence_separation_aa"] = pd.to_numeric(raw[sep_col], errors="coerce")

    out["replicate"] = pd.to_numeric(out["replicate"], errors="coerce").astype("Int64")
    return out


def load_kinetics(path):
    """Load the actual per_run_summary.csv schema used in this project."""
    raw = pd.read_csv(path)

    id_col = find_col(
        raw,
        ["id", "sequence_id", "sequence", "design_id", "name"],
        label="kinetics sequence/design ID",
    )
    temp_col = find_col(raw, ["temp", "temperature", "temperature_K"], required=False)
    rep_col = find_col(raw, ["replicate", "replicate_id", "rep", "run"], label="replicate")
    fam_col = find_col(raw, ["group", "family", "chemistry"], label="chemistry/group")
    arch_col = find_col(
        raw,
        ["architecture_type", "architecture_plot", "architecture", "arch"],
        required=False,
    )
    k_col = find_col(
        raw,
        ["k_escape_per_ns", "kescape_per_ns", "k_escape",
         "escape_rate_per_ns", "escape_rate"],
        label="k_escape",
    )

    # 370 K only.
    if temp_col is not None:
        temp_num = pd.to_numeric(
            raw[temp_col].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0],
            errors="coerce",
        )
        raw = raw.loc[np.isclose(temp_num, 370, equal_nan=False)].copy()

    # Current structural R4 analysis contains these four chemistries only.
    raw["_family"] = raw[fam_col].astype(str).str.upper().str.strip()
    wanted = {"VGIP", "VGIG", "VAIP", "VAIG"}
    raw = raw.loc[raw["_family"].isin(wanted)].copy()

    # Actual values are rep01 / rep02 / rep03. Extract the integer explicitly.
    raw["_replicate"] = pd.to_numeric(
        raw[rep_col].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )

    if raw["_replicate"].isna().any():
        bad = raw.loc[raw["_replicate"].isna(), rep_col].unique()
        raise ValueError(
            "Could not parse replicate labels from per_run_summary.csv: "
            + ", ".join(map(str, bad))
        )

    out = pd.DataFrame({
        "sequence_id": raw[id_col].map(clean_id),
        "design_id": pd.to_numeric(raw[id_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"),
        "replicate": raw["_replicate"].astype(int),
        "family_k": raw["_family"],
        "k_escape_per_ns": pd.to_numeric(raw[k_col], errors="coerce"),
    })

    if arch_col is not None:
        out["architecture_k"] = raw[arch_col].map(architecture_name)

    # The contact dataset is R4; mark these matched kinetics rows accordingly.
    out["ratio_k"] = "R4"

    # Strict validation: 32 designs x 3 replicates = 96 rows.
    counts = out.groupby("sequence_id")["replicate"].nunique()
    if len(out) != 96 or out["sequence_id"].nunique() != 32 or not counts.eq(3).all():
        raise ValueError(
            "Unexpected filtered kinetics dimensions. Expected "
            "96 rows = 32 designs x 3 replicates, but found "
            f"{len(out)} rows, {out['sequence_id'].nunique()} designs. "
            f"Replicate-count distribution: {counts.value_counts().to_dict()}"
        )

    if out.duplicated(["design_id", "replicate"]).any():
        raise ValueError(
            "Duplicate design_id + replicate keys remain in kinetics data."
        )

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--outdir", default="DR3_motif_contact_final")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    structure_path = root / "nonlocal_BB_trajectory_summary.csv"
    kinetics_path = root / "per_run_summary.csv"
    mapping_path = root / "trajectory_replicate_mapping.csv"

    if not structure_path.exists():
        raise FileNotFoundError(structure_path)
    if not kinetics_path.exists():
        raise FileNotFoundError(kinetics_path)

    s = load_structure(structure_path)
    k = load_kinetics(kinetics_path)

    # Canonical numeric design IDs for exact structural ↔ kinetics pairing.
    # load_structure() returns `s`; load_kinetics() returns `k`.
    # Structural IDs are numeric (e.g. 103), while kinetics sequence IDs are
    # full names (e.g. 103_VGIG_R4_even). Match on the leading integer + replicate.
    s["design_id"] = pd.to_numeric(
        s["sequence_id"].astype(str).str.extract(r"^(\d+)")[0],
        errors="coerce",
    )
    k["design_id"] = pd.to_numeric(
        k["sequence_id"].astype(str).str.extract(r"^(\d+)")[0],
        errors="coerce",
    )

    if s["design_id"].isna().any():
        bad = s.loc[s["design_id"].isna(), "sequence_id"].unique()
        raise ValueError(f"Could not extract structural numeric design IDs: {bad}")
    if k["design_id"].isna().any():
        bad = k.loc[k["design_id"].isna(), "sequence_id"].unique()
        raise ValueError(f"Could not extract kinetics numeric design IDs: {bad}")

    s["design_id"] = s["design_id"].astype(int)
    k["design_id"] = k["design_id"].astype(int)

    print("\nSTRUCTURAL DATA")
    print("----------------")
    print(f"Rows: {len(s)}")
    print("Replicates/design:")
    print(s.groupby("sequence_id")["replicate"].nunique().value_counts().sort_index())

    print("\nKINETICS DATA (370 K)")
    print("---------------------")
    print(f"Rows: {len(k)}")
    print("Replicates/design:")
    print(k.groupby("sequence_id")["replicate"].nunique().value_counts().sort_index())

    # Optional mapping validation information.
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        print(f"\nFound replicate mapping: {mapping_path.name} ({len(mapping)} rows)")
    else:
        print("\nNo trajectory_replicate_mapping.csv found; continuing.")

    # Pair structural and kinetic trajectory replicates.
    # At this point each sequence/replicate key must be unique.
    s_dups = s.duplicated(["design_id", "replicate"], keep=False)
    k_dups = k.duplicated(["design_id", "replicate"], keep=False)

    if s_dups.any():
        raise ValueError(
            "Structural design_id + replicate keys are not unique.\n"
            + s.loc[s_dups, ["design_id", "replicate"]]
              .sort_values(["design_id", "replicate"])
              .to_string(index=False)
        )

    if k_dups.any():
        raise ValueError(
            "Kinetics design_id + replicate keys are not unique.\n"
            + k.loc[k_dups, ["design_id", "replicate"]]
              .sort_values(["design_id", "replicate"])
              .to_string(index=False)
        )

    merged = s.merge(
        k,
        on=["design_id", "replicate"],
        how="inner",
        validate="one_to_one",
    )

    # Both inputs contain sequence_id, so pandas suffixes them after the merge.
    # Use the full kinetics sequence name as the canonical design label.
    if "sequence_id_y" in merged.columns:
        merged["sequence_id"] = merged["sequence_id_y"].astype(str)
    elif "sequence_id_x" in merged.columns:
        merged["sequence_id"] = merged["sequence_id_x"].astype(str)

    # Hard validation before any statistics/plotting.
    if len(merged) != 96:
        s_keys = set(map(tuple, s[["design_id", "replicate"]].astype(int).to_numpy()))
        k_keys = set(map(tuple, k[["design_id", "replicate"]].astype(int).to_numpy()))
        raise ValueError(
            f"Expected exactly 96 structural-kinetic pairs, found {len(merged)}.\n"
            f"Missing in kinetics: {sorted(s_keys - k_keys)}\n"
            f"Missing in structure: {sorted(k_keys - s_keys)}"
        )

    # Fill metadata preferentially from structural summary.
    if "family_k" in merged:
        merged["family"] = merged["family"].fillna(merged["family_k"])
    if "ratio_k" in merged:
        merged["ratio"] = merged["ratio"].fillna(merged["ratio_k"])
    if "architecture_k" in merged:
        merged["architecture"] = merged["architecture"].fillna(merged["architecture_k"])

    # This analysis is R4 only by construction:
    # the structural summary is the R4 contact run and kinetics were explicitly
    # restricted to its four R4 chemistry groups above.
    merged["ratio"] = "R4"

    merged = merged.loc[
        merged["W_BB"].notna()
        & merged["k_escape_per_ns"].notna()
        & (merged["k_escape_per_ns"] > 0)
    ].copy()

    # Chemistry-specific R4 baseline across 8 architectures x 3 trajectories = 24.
    baselines = (
        merged.groupby("family", dropna=False)["k_escape_per_ns"]
        .mean()
        .rename("chemistry_mean_k_escape_per_ns")
    )
    merged = merged.join(baselines, on="family")
    merged["delta_log2_kescape"] = np.log2(
        merged["k_escape_per_ns"] / merged["chemistry_mean_k_escape_per_ns"]
    )

    print("\nPAIRED DATA")
    print("-----------")
    print(f"Paired observations: {len(merged)}")
    print(f"Unique designs:      {merged['sequence_id'].nunique()}")
    print(f"Families:            {sorted(merged['family'].dropna().unique())}")
    print(f"Architectures:       {sorted(merged['architecture'].dropna().unique())}")

    if len(merged) != 96:
        warnings.warn(
            f"Expected 96 paired R4 observations (32 designs x 3 replicates), "
            f"but obtained {len(merged)}. Inspect the saved merged CSV before "
            "using inferential statistics."
        )

    merged.to_csv(outdir / "DR3_BB_contacts_merged_triplicates.csv", index=False)

    # Design means: n = 32 independent sequence designs for conservative summary.
    agg = {
        "W_BB": "mean",
        "delta_log2_kescape": "mean",
        "k_escape_per_ns": "mean",
    }
    if "contacts_per_frame" in merged:
        agg["contacts_per_frame"] = "mean"
    if "mean_sequence_separation_aa" in merged:
        agg["mean_sequence_separation_aa"] = "mean"

    design = (
        merged.groupby(
            ["sequence_id", "family", "architecture"],
            as_index=False,
            dropna=False,
        )
        .agg(agg)
    )
    design.to_csv(outdir / "DR3_BB_contacts_design_means.csv", index=False)

    stats = pd.DataFrame([
        correlation_stats(
            merged["W_BB"],
            merged["delta_log2_kescape"],
            "trajectory-level paired replicates",
        ),
        correlation_stats(
            design["W_BB"],
            design["delta_log2_kescape"],
            "design means",
        ),
    ])
    stats.to_csv(outdir / "DR3_BB_contacts_statistics.csv", index=False)

    print("\nCORRELATIONS")
    print("------------")
    print(stats.to_string(index=False))

    # ================================================================
    # FIGURE 1 — FINAL PAIRED TRAJECTORY-LEVEL RELATIONSHIP
    # ================================================================
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    # Thin lines connect the three paired trajectories of one design.
    for _, g in merged.groupby("sequence_id"):
        g = g.sort_values("replicate")
        if len(g) > 1:
            ax.plot(
                g["W_BB"],
                g["delta_log2_kescape"],
                color="0.82",
                lw=0.65,
                zorder=1,
            )

    for arch in ARCH_ORDER:
        for fam in FAMILY_ORDER:
            g = merged[
                (merged["architecture"] == arch)
                & (merged["family"] == fam)
            ]
            if g.empty:
                continue
            ax.scatter(
                g["W_BB"],
                g["delta_log2_kescape"],
                s=28,
                marker=FAMILY_MARKERS[fam],
                facecolor=ARCH_COLORS[arch],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )

    # Global trajectory-level linear trend, descriptive only.
    d = merged[["W_BB", "delta_log2_kescape"]].dropna()
    lr = linregress(d["W_BB"], d["delta_log2_kescape"])
    xx = np.linspace(d["W_BB"].min(), d["W_BB"].max(), 200)
    ax.plot(xx, lr.intercept + lr.slope * xx, color="black", lw=1.15, zorder=2)
    ax.axhline(0, color="0.55", lw=0.75, ls="--", zorder=0)

    st = stats.iloc[0]
    ax.text(
        0.03, 0.035,
        f"Pearson $r$ = {st.pearson_r:.2f}\n"
        f"Spearman $\\rho$ = {st.spearman_rho:.2f}\n"
        f"$n$ = {int(st.n)} trajectories",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.5),
    )

    ax.set_xlabel(
        r"Sequence-weighted motif-contact burden, $W_{\mathrm{BB}}$" "\n"
        r"(residue separation $\times$ contacts frame$^{-1}$)"
    )
    ax.set_ylabel(
        r"$\Delta\log_2(k_{\mathrm{escape}})$"
    )

    # Two compact legends: architecture = color, chemistry = marker.
    arch_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=ARCH_COLORS[a], markeredgecolor="white",
            label=a
        )
        for a in ARCH_ORDER
    ]
    leg1 = ax.legend(
        handles=arch_handles,
        title="Architecture",
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0,
    )
    ax.add_artist(leg1)

    fam_handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[f], linestyle="none",
            markersize=5.5, markerfacecolor="0.45",
            markeredgecolor="white", label=f
        )
        for f in FAMILY_ORDER
    ]
    ax.legend(
        handles=fam_handles,
        title="Chemistry",
        frameon=False,
        bbox_to_anchor=(1.01, 0.36),
        loc="upper left",
        borderaxespad=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.73, bottom=0.18, top=0.96)
    save_figure(fig, outdir / "figure_DR3_BB_burden_vs_kescape")

    # ================================================================
    # FIGURE 2 — ARCHITECTURE DEPENDENCE OF W_BB
    # ================================================================
    fig, ax = plt.subplots(figsize=(6.8, 3.5))

    xmap = {a: i for i, a in enumerate(ARCH_ORDER)}
    offsets = {"VGIP": -0.18, "VGIG": -0.06, "VAIP": 0.06, "VAIG": 0.18}

    # Chemistry-level design means; each point averages the 3 structural reps.
    for fam in FAMILY_ORDER:
        g = design[design["family"] == fam]
        for _, row in g.iterrows():
            arch = row["architecture"]
            if arch not in xmap:
                continue
            ax.scatter(
                xmap[arch] + offsets[fam],
                row["W_BB"],
                s=34,
                marker=FAMILY_MARKERS[fam],
                facecolor=ARCH_COLORS[arch],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    # Black architecture mean ± SD across four chemistries.
    for arch in ARCH_ORDER:
        vals = design.loc[design["architecture"] == arch, "W_BB"].dropna()
        if len(vals) == 0:
            continue
        x = xmap[arch]
        ax.errorbar(
            x, vals.mean(),
            yerr=vals.std(ddof=1) if len(vals) > 1 else 0,
            fmt="o",
            ms=4.5,
            color="black",
            ecolor="black",
            elinewidth=0.9,
            capsize=2.2,
            zorder=5,
        )

    ax.set_xticks(range(len(ARCH_ORDER)))
    ax.set_xticklabels(
        [
            "Random\nshuffle",
            "Alternating",
            "Small\nblocks",
            "Medium\nblocks",
            "Large\nblocks",
            "Higher λ\ncenter",
            "Higher λ\nends",
            "Diblock",
        ],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel(
        r"$W_{\mathrm{BB}}$ (aa$\cdot$contacts/frame)"
    )
    ax.set_xlabel("Sequence architecture")

    fam_handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[f], linestyle="none",
            markersize=5.5, markerfacecolor="0.45",
            markeredgecolor="white", label=f
        )
        for f in FAMILY_ORDER
    ]
    fam_handles.append(
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markersize=4.5, color="black", label="Mean ± SD"
        )
    )
    ax.legend(handles=fam_handles, frameon=False, loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, outdir / "figure_DR3_BB_burden_by_architecture")

    # ================================================================
    # FIGURE 3 — DESIGN-MEAN RELATIONSHIP (n = 32)
    # ================================================================
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for arch in ARCH_ORDER:
        for fam in FAMILY_ORDER:
            g = design[
                (design["architecture"] == arch)
                & (design["family"] == fam)
            ]
            if g.empty:
                continue
            ax.scatter(
                g["W_BB"],
                g["delta_log2_kescape"],
                s=42,
                marker=FAMILY_MARKERS[fam],
                facecolor=ARCH_COLORS[arch],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

    d = design[["W_BB", "delta_log2_kescape"]].dropna()
    lr = linregress(d["W_BB"], d["delta_log2_kescape"])
    xx = np.linspace(d["W_BB"].min(), d["W_BB"].max(), 200)
    ax.plot(xx, lr.intercept + lr.slope * xx, color="black", lw=1.15)
    ax.axhline(0, color="0.55", lw=0.75, ls="--")

    st = stats.iloc[1]
    ax.text(
        0.03, 0.035,
        f"Pearson $r$ = {st.pearson_r:.2f}\n"
        f"Spearman $\\rho$ = {st.spearman_rho:.2f}\n"
        f"$n$ = {int(st.n)} designs",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.5),
    )

    ax.set_xlabel(
        r"Mean $W_{\mathrm{BB}}$" "\n"
        r"(residue separation $\times$ contacts frame$^{-1}$)"
    )
    ax.set_ylabel(
        r"Mean $\Delta\log_2(k_{\mathrm{escape}})$"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    arch_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=ARCH_COLORS[a], markeredgecolor="white",
            label=a
        )
        for a in ARCH_ORDER
    ]
    leg1 = ax.legend(
        handles=arch_handles, title="Architecture", frameon=False,
        bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0
    )
    ax.add_artist(leg1)

    fam_handles = [
        Line2D(
            [0], [0], marker=FAMILY_MARKERS[f], linestyle="none",
            markersize=5.5, markerfacecolor="0.45",
            markeredgecolor="white", label=f
        )
        for f in FAMILY_ORDER
    ]
    ax.legend(
        handles=fam_handles, title="Chemistry", frameon=False,
        bbox_to_anchor=(1.01, 0.36), loc="upper left", borderaxespad=0
    )

    fig.subplots_adjust(left=0.13, right=0.73, bottom=0.18, top=0.96)
    save_figure(fig, outdir / "figure_DR3_BB_burden_vs_kescape_design_means")

    print(f"\nSaved final thesis outputs to:\n{outdir}")


if __name__ == "__main__":
    main()
