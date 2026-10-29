#!/usr/bin/env python
"""DR5 ternary architecture-deviation plots. Requires numpy/pandas/matplotlib.

Full-library reference: equal-weight architecture means within each family,
ratio and temperature. Large dots are equal-weight IG/VG mean effects.
Faint lines span the two family effects; they are not confidence intervals.
Creates the full overview and four focused comparison sets, each with four-panel
metrics and two-panel widths. All 22 architectures are represented.
Default: R1-R5 at 300 and 370 K. References always use the full library.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GROUPS = [
    ("Distributed patterns", [
        ("uniform_opposite_spacing", "Uniform opposite spacing"),
        ("uniform_paired_charges", "Uniform paired charges"),
        ("KG_half_EG_half_distribution", "K-half / E-half distribution"),
    ]),
    ("Charge domains", [
        ("paired_charge_terminal_block", "Paired terminal domain"),
        ("segregated_charge_terminal_block", "Segregated terminal domain"),
        ("central_paired_charges", "Paired central domain"),
        ("central_segregated_charges", "Segregated central domain"),
        ("terminal_charge_triblock", "E–H–K terminal triblock"),
    ]),
    ("Paired clusters", [
        ("paired_charge_endcaps", "Paired-charge endcaps"),
        ("two_paired_charge_clusters", "Two paired-charge clusters"),
        ("three_paired_charge_clusters", "Three paired-charge clusters"),
    ]),
    ("Polarity and placement", [
        ("EG_endcaps_KG_center", "E endcaps / K centre"),
        ("KG_endcaps_EG_center", "K endcaps / E centre"),
        ("EG_outer_KG_inner_clusters", "Outer E / inner K clusters"),
        ("KG_outer_EG_inner_clusters", "Outer K / inner E clusters"),
        ("alternating_polarity_clusters", "Alternating-polarity clusters"),
        ("EG_endcaps_distributed_KG", "E endcaps / distributed K"),
        ("KG_endcaps_distributed_EG", "K endcaps / distributed E"),
    ]),
    ("Block spacing", [
        ("internal_blocks_close_spacing", "Internal blocks: close"),
        ("internal_blocks_medium_spacing", "Internal blocks: medium"),
        ("internal_blocks_wide_spacing", "Internal blocks: wide"),
    ]),
    ("Control", [("random_shuffle", "Random shuffle")]),
]
ORDER = [key for _, members in GROUPS for key, _ in members]
LABELS = {key: label for _, members in GROUPS for key, label in members}
FAMILIES = ["IGKGEG", "VGKGEG"]
RATIOS = [1, 2, 3, 4, 5]
COLOURS = {1: "#0064C8", 2: "#25A7B8", 3: "#55A868",
           4: "#E3A135", 5: "#9073BD"}
OFFSETS = dict(zip(RATIOS, np.linspace(-0.30, 0.30, 5)))
MAIN = [
    ("fraction_compact", "Compact-state population", r"$\Delta f_{\mathrm{compact}}$ (pp)", "difference_pp"),
    ("kcollapse_ns_inv", "Compact-state entry", r"$\log_2(k_{\mathrm{collapse}}/\bar{k}_{\mathrm{collapse}})$", "log2_fold"),
    ("kescape_ns_inv", "Compact-state escape", r"$\log_2(k_{\mathrm{escape}}/\bar{k}_{\mathrm{escape}})$", "log2_fold"),
    ("mean_compact_rg_nm", "Compact-state dimensions", r"$\Delta\langle R_g\rangle_{\mathrm{compact}}$ (nm)", "difference_nm"),
]
WIDTHS = [
    ("iqr_rg_nm", r"$R_g$ central 50% width", r"$\Delta(P_{75}-P_{25})$ (nm)", "difference_nm"),
    ("central90_rg_nm", r"$R_g$ central 90% width", r"$\Delta(P_{95}-P_{5})$ (nm)", "difference_nm"),
]
SPECS = MAIN + WIDTHS


# Each member is a canonical architecture key from the complete overview.
COMPARISONS = [
    ("01_charge_mixing_and_placement", "Charge mixing and domain placement", [
        ("Terminal domain", ["paired_charge_terminal_block", "segregated_charge_terminal_block"]),
        ("Central domain", ["central_paired_charges", "central_segregated_charges"]),
        ("Terminal triblock", ["terminal_charge_triblock"]),
    ]),
    ("02_distributions_and_paired_clusters", "Distributed charges and paired clusters", [
        ("Distributed patterns", ["uniform_opposite_spacing", "uniform_paired_charges", "KG_half_EG_half_distribution"]),
        ("Domain placement", ["paired_charge_terminal_block", "central_paired_charges"]),
        ("Paired clusters", ["paired_charge_endcaps", "two_paired_charge_clusters", "three_paired_charge_clusters"]),
        ("Control", ["random_shuffle"]),
    ]),
    ("03_opposite_charge_block_spacing", "Opposite-charge block spacing", [
        ("Internal block spacing", ["internal_blocks_close_spacing", "internal_blocks_medium_spacing", "internal_blocks_wide_spacing"]),
        ("Terminal triblock", ["terminal_charge_triblock"]),
    ]),
    ("04_charge_polarity_and_placement", "Charge polarity and placement", [
        ("Centre / endcaps", ["EG_endcaps_KG_center", "KG_endcaps_EG_center"]),
        ("Outer / inner clusters", ["EG_outer_KG_inner_clusters", "KG_outer_EG_inner_clusters"]),
        ("Alternating polarity", ["alternating_polarity_clusters"]),
        ("Endcaps / distributed", ["EG_endcaps_distributed_KG", "KG_endcaps_distributed_EG"]),
    ]),
]


def figure_groups(definition):
    return [(name, [(key, LABELS[key]) for key in members]) for name, members in definition]


def verify_coverage():
    represented = {key for _, _, groups in COMPARISONS for _, members in groups for key in members}
    if represented != set(ORDER):
        raise ValueError(f"Comparison coverage mismatch: missing {set(ORDER)-represented}, extra {represented-set(ORDER)}")
    for _, _, groups in COMPARISONS:
        keys = [key for _, members in groups for key in members]
        if len(keys) != len(set(keys)):
            raise ValueError("An architecture occurs twice within a comparison figure.")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.linewidth": .7, "xtick.labelsize": 8,
        "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def prepare(path, temperatures):
    d = pd.read_csv(path)
    required = {"family", "architecture", "ratio", "temperature", "threshold_nm"}
    required.update(spec[0] for spec in SPECS)
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    d["family"] = d.family.astype(str).str.strip().str.upper()
    d["architecture"] = d.architecture.astype(str).str.strip()
    d["temperature"] = pd.to_numeric(d.temperature.astype(str).str.strip().str.replace(
        r"\s*[Kk]$", "", regex=True), errors="raise")
    d["ratio_number"] = pd.to_numeric(d.ratio.astype(str).str.strip().str.extract(
        r"^[Rr]?(\d+)$", expand=False), errors="raise").astype(int)
    d = d[d.ratio_number.isin(RATIOS) & d.temperature.isin(temperatures)].copy()
    if d.empty:
        raise ValueError("No requested R1-R5 temperature conditions found.")
    if set(d.family) != set(FAMILIES):
        raise ValueError("Expected exactly IGKGEG and VGKGEG families.")
    unknown = set(d.architecture) - set(ORDER)
    if unknown:
        raise ValueError("Unknown architectures: " + ", ".join(sorted(unknown)))
    if not np.allclose(pd.to_numeric(d.threshold_nm), 4.61):
        raise ValueError("Summary threshold must be 4.61 nm; recalculate old summaries.")
    if "replicate" not in d:
        d["replicate"] = "single"
    if d.replicate.isna().any():
        raise ValueError("Missing replicate identifiers.")
    keys = ["family", "temperature", "ratio_number", "architecture"]
    if d.duplicated(keys + ["replicate"]).any():
        raise ValueError("Duplicate architecture/condition/replicate rows.")
    if "sequence" in d and d.groupby(keys).sequence.nunique().gt(1).any():
        raise ValueError("Multiple different sequence IDs in an architecture/condition.")
    for metric, _, _, _ in SPECS:
        d[metric] = pd.to_numeric(d[metric], errors="raise")
        if not np.isfinite(d[metric]).all():
            raise ValueError(f"Nonfinite {metric}: inspect summary and event exposure.")
        if metric.startswith("k") and d[metric].le(0).any():
            raise ValueError(f"{metric} has zero/negative values: log ratios are undefined.")
    if not d.fraction_compact.between(0, 1).all():
        raise ValueError("Compact fraction outside [0, 1].")
    if d[["iqr_rg_nm", "central90_rg_nm", "mean_compact_rg_nm"]].lt(0).any().any():
        raise ValueError("Negative width or radius.")
    # Aggregate replicates first. Every architecture then has equal reference weight.
    metrics = [spec[0] for spec in SPECS]
    cells = d.groupby(keys, as_index=False)[metrics].mean()
    counts = d.groupby(keys).size().rename("n_replicates").reset_index()
    cells = cells.merge(counts, on=keys, validate="one_to_one")
    expected = pd.MultiIndex.from_product(
        [FAMILIES, temperatures, RATIOS, ORDER], names=keys)
    actual = pd.MultiIndex.from_frame(cells[keys])
    absent = expected.difference(actual)
    if len(absent):
        raise ValueError("Incomplete library; missing conditions: " + str(list(absent)[:8]))
    baselines = cells.groupby(["family", "temperature", "ratio_number"],
                              as_index=False)[metrics].mean()
    effects = []
    for metric, _, _, transformation in SPECS:
        ref = cells.groupby(["family", "temperature", "ratio_number"])[metric].transform("mean")
        value = cells[metric]
        effect = np.log2(value / ref) if transformation == "log2_fold" else value - ref
        if transformation == "difference_pp":
            effect *= 100
        table = cells[keys + ["n_replicates"]].copy()
        table["metric"] = metric
        table["transformation"] = transformation
        table["raw_value"] = value
        table["architecture_reference_mean"] = ref
        table["effect"] = effect
        effects.append(table)
    effects = pd.concat(effects, ignore_index=True)
    means = effects.groupby(["temperature", "ratio_number", "architecture", "metric"])["effect"].agg(
        mean_family_effect="mean", family_min="min", family_max="max").reset_index()
    return effects, baselines, means


def positions(selected, groups):
    result, sections = {}, []
    x = 0.0
    for group, members in groups:
        members = [(key, label) for key, label in members if key in selected]
        if not members:
            continue
        start = x
        for key, _ in members:
            result[key] = x
            x += 1.0
        sections.append((group, start - .5, x - .5))
        x += .65
    return result, sections


def draw_panel(ax, effects, temperature, spec, selected, xmap, sections, show_groups, local_y_limits):
    metric, title, ylabel, _ = spec
    data = effects[(effects.temperature == temperature) & (effects.metric == metric)]
    lookup = data.set_index(["architecture", "ratio_number", "family"]).effect
    values = []
    for i, (_, left, right) in enumerate(sections):
        if i % 2 == 0:
            ax.axvspan(left, right, color="#F4F5F6", zorder=0)
        if i:
            ax.axvline(left - .325, color="#D9DDE0", linewidth=.6, zorder=0)
        if show_groups:
            ax.text((left + right) / 2, 1.22, sections[i][0],
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8.5, color="#59636D")
    for key in selected:
        for ratio in RATIOS:
            a = float(lookup.loc[(key, ratio, "IGKGEG")])
            b = float(lookup.loc[(key, ratio, "VGKGEG")])
            x = xmap[key] + OFFSETS[ratio]
            colour = COLOURS[ratio]
            values.extend([a, b])
            # Spread is chemistry dependence, not uncertainty.
            ax.plot([x, x], [a, b], color=colour, alpha=.30,
                    linewidth=1.8, solid_capstyle="round", zorder=2)
            ax.scatter(x, a, s=12, marker="o", color=colour, alpha=.30,
                       edgecolors="none", zorder=3)
            ax.scatter(x, b, s=15, marker="^", color=colour, alpha=.30,
                       edgecolors="none", zorder=3)
            ax.scatter(x, (a + b) / 2, s=27, marker="o", color=colour,
                       edgecolors="white", linewidth=.45, zorder=4)
    ax.axhline(0, color="#858C93", linestyle="--", linewidth=.8, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DCE0E3", linestyle=":", linewidth=.6)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, loc="left", pad=9)
    ax.set_xlim(min(xmap.values()) - .65, max(xmap.values()) + .65)
    scale_values = np.asarray(values) if local_y_limits else data.effect.to_numpy()
    lo, hi = float(scale_values.min()), float(scale_values.max())
    span = max(hi - lo, .05)
    ax.set_ylim(min(lo - .16 * span, -.06 * span), max(hi + .16 * span, .06 * span))
    ax.tick_params(length=3, width=.6)


def make_figure(effects, temperature, specs, selected, args, stem, groups, title, overview=False):
    xmap, sections = positions(selected, groups)
    n = len(specs)
    width = args.overview_width if overview else args.width
    # All architectures use stacked panels to retain readable category labels.
    height = 3.9 + n * 2.05
    fig, axes = plt.subplots(n, 1, figsize=(width, height), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, spec in zip(axes, specs):
        draw_panel(ax, effects, temperature, spec, selected, xmap, sections,
                   show_groups=ax is axes[0], local_y_limits=args.local_y_limits)
    axes[-1].set_xticks([xmap[key] for key in selected], [LABELS[key] for key in selected],
                       rotation=58, ha="right", rotation_mode="anchor")
    axes[-1].set_xlabel("")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.subplots_adjust(left=.105, right=.985, top=(.845 if n == 4 else .79),
                        bottom=2.25 / height, hspace=.52)
    for letter, ax in zip("ABCD", axes):
        fig.text(.018, ax.get_position().y1 + .008, letter,
                 fontsize=13, fontweight="bold", va="bottom")
    fig.text(.105, .970, f"{title} | {temperature} K",
             fontsize=13, fontweight="bold", va="top")
    fig.text(.105, .942, "Reference: the 22-architecture mean within each family and composition",
             fontsize=9, color="#59636D", va="top")
    ratio_handles = [Line2D([], [], linestyle="", marker="o", markersize=5.8,
                     markerfacecolor=COLOURS[r], markeredgecolor="white",
                     label=f"R{r} ({100 * r / 6:.1f}% charged motifs)") for r in RATIOS]
    fig.legend(handles=ratio_handles, loc="upper left", bbox_to_anchor=(.098, .927),
               ncol=(5 if width >= 11 else 3), frameon=False, columnspacing=1.4, handletextpad=.4, fontsize=8)
    family_handles = [
        Line2D([], [], linestyle="", marker="o", color="#333333", markersize=5,
               label="IG/VG mean"),
        Line2D([], [], linestyle="", marker="o", color="#AEB3B8", markersize=3.5,
               label="IG effect (faint)"),
        Line2D([], [], linestyle="", marker="^", color="#AEB3B8", markersize=3.8,
               label="VG effect (faint)"),
        Line2D([], [], color="#BBC0C5", linewidth=1.8,
               label="IG–VG spread"),
    ]
    fig.legend(handles=family_handles, loc="lower left", bbox_to_anchor=(.10, .024),
               ncol=4, frameon=False, columnspacing=1.8, handletextpad=.5, fontsize=8)
    fig.text(.105, .012, "Family spread is not a confidence interval. H denotes VPGIG or VPGVG.",
             fontsize=8, color="#65717B")
    for extension in ("pdf", "svg", "png"):
        path = args.outdir / f"{stem}_{temperature}K.{extension}"
        fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("DR5_ternary_comparison_plots"))
    parser.add_argument("--temperatures", type=int, nargs="+", default=[300, 370], choices=[300, 370])
    parser.add_argument("--width", type=float, default=9.5, help="Focused figure width in inches.")
    parser.add_argument("--overview-width", type=float, default=13.0)
    parser.add_argument("--dpi", type=int, default=350, help="PNG resolution; PDF/SVG remain vector.")
    parser.add_argument("--local-y-limits", action="store_true", help="Autoscale each subset; default uses the same metric limits as the overview at that temperature.")
    args = parser.parse_args()
    if min(args.width, args.overview_width) < 7 or args.dpi < 72:
        parser.error("Use figure widths >= 7 inches and dpi >= 72.")
    args.temperatures = sorted(set(args.temperatures))
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    verify_coverage()
    style()
    effects, baselines, means = prepare(args.summary, args.temperatures)
    effects.to_csv(args.outdir / "family_architecture_effects.csv", index=False)
    means.to_csv(args.outdir / "mean_family_architecture_effects.csv", index=False)
    baselines.to_csv(args.outdir / "absolute_family_baselines.csv", index=False)
    sets = [("00_overview", "Ternary ELP architecture overview", GROUPS)] + [
        (name, title, figure_groups(groups)) for name, title, groups in COMPARISONS]
    mapping = []
    for name, _, groups in sets:
        for group_index, (group, members) in enumerate(groups):
            for key, label in members:
                mapping.append(dict(figure_set=name, group=group,
                                    background="grey" if group_index % 2 == 0 else "white",
                                    architecture=key, label=label))
    pd.DataFrame(mapping).to_csv(args.outdir / "figure_architecture_map.csv", index=False)
    inventory = []
    for temperature in args.temperatures:
        for name, title, groups in sets:
            selected = [key for _, members in groups for key, _ in members]
            for suffix, specs in [("main", MAIN), ("widths", WIDTHS)]:
                stem = f"DR5_ternary_{name}_{suffix}"
                make_figure(effects, temperature, specs, selected, args, stem,
                            groups, title, overview=name == "00_overview")
                for extension in ["pdf", "svg", "png"]:
                    inventory.append(dict(figure_set=name, temperature_k=temperature,
                                          panels=suffix, file=f"{stem}_{temperature}K.{extension}"))
                print(f"Saved {stem}_{temperature}K (PDF, SVG, PNG)", flush=True)
    pd.DataFrame(inventory).to_csv(args.outdir / "figure_inventory.csv", index=False)
    notes = (
        "All figures retain architecture on the x-axis and five ratio colors.\n"
        "References ALWAYS use all 22 architectures separately by family, ratio and temperature.\n"
        "Independent replicates are averaged within each architecture before equal-weight reference calculation.\n"
        "Compact fraction: 100*(f-reference), percentage points.\n"
        "Rates: log2(rate/arithmetic architecture mean rate).\n"
        "Compact-state Rg, IQR and central90 width: value-reference, nm.\n"
        "Large point: equal-weight mean of the IG and VG effects.\n"
        "Faint circle: IG; faint triangle: VG; line spans those two effects.\n"
        "Spread is family dependence, NOT a confidence interval or replicate SD.\n"
        "Widths describe the full Rg ensemble; central50=P75-P25, central90=P95-P5.\n"
        "R1-R5 legend percentages are charged MOTIF fractions, not residue FCR.\n"
        "H denotes the hydrophobic guest motif VPGIG or VPGVG.\n"
        "Grey and white groups identify comparisons; every one of the 22 architectures occurs in a focused set.\n"
        "Paired terminal, paired central and terminal triblock recur across sets as comparison anchors.\n"
        "Figures show deviations from the full-library reference, not direct pair-difference curves.\n"
        "Shared y limits across sets at a given temperature unless --local-y-limits is used.\n"
        "No significance tests, smoothing or statistical uncertainty bars.\n"
    )
    (args.outdir / "PLOT_NOTES.txt").write_text(notes, encoding="utf-8")
    print(f"\nAll 22 architectures covered. Saved {len(inventory)//3} figures in: {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}")
