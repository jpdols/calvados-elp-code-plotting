#!/usr/bin/env python
"""Design Rule 6 VG/IG/KG architecture plots. Requires numpy, pandas, matplotlib.
Creates one overview, four focused sets (five-panel main/central50), and a composition overview.
Default 370 K; use --temperatures 300 370 for both. Threshold must be 4.61 nm.
R3 Tk8 excluded by default: supplied sequence has R2 composition. No sequence file required.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GROUPS = [('Central IG / charge allocation', [('central_IG_KG_endcaps', 'Central IG / K endcaps'), ('central_IG_KG_adjacent', 'Central IG / adjacent K'), ('KG_clustered_in_VG', 'K clustered in VG'), ('KG_dispersed_in_VG', 'K dispersed in VG'), ('KG_embedded_in_IG', 'K embedded in IG'), ('single_IG_block_KG_center', 'IG split by central K'), ('KG_dispersed_in_IG', 'K dispersed in IG'), ('central_mixed_sticker_domain', 'Central mixed IG/K domain')]), ('IG fragmentation', [('two_IG_blocks_KG_endcaps', 'Two IG blocks / K endcaps'), ('three_medium_IG_blocks_KG_endcaps', 'Three IG blocks / K endcaps'), ('four_IG_blocks_KG_endcaps', 'Four IG blocks / K endcaps'), ('two_IG_blocks_interfacial_KG', 'Two IG blocks / interfacial K'), ('three_IG_blocks_interfacial_KG', 'Three IG blocks / interfacial K'), ('four_IG_blocks_interfacial_KG', 'Four IG blocks / interfacial K')]), ('Position and terminal order', [('two_IG_blocks_KG_center', 'Two IG blocks / central K'), ('three_IG_blocks_KG_center', 'Three IG blocks / central K'), ('IG_endcaps_KG_center', 'IG endcaps / central K'), ('KG_capped_IG_endblocks', 'K-capped IG endblocks'), ('IG_outermost_KG_inside', 'Outer IG / inner K'), ('IG_one_end_KG_opposite', 'IG and K at opposite ends'), ('KG_then_IG_same_end', 'VG-K-IG terminal order'), ('IG_then_KG_same_end', 'VG-IG-K terminal order')]), ('Asymmetry and gradients', [('asymmetric_IG_blocks_KG_endcaps', 'Asymmetric IG / K endcaps'), ('charged_large_IG_domain', 'K adjacent to larger IG domain'), ('KG_one_end_symmetric_IG', 'One K end / two IG blocks'), ('hydrophobic_gradient', 'Hydrophobic gradient'), ('symmetric_IG_gradient', 'Symmetric IG gradient')]), ('Reference designs', [('original_Tk8_adapted', 'Tk8-adapted reference'), ('even_distribution', 'Even distribution'), ('random_shuffle', 'Random shuffle')])]
ORDER = [key for _, members in GROUPS for key, _ in members]
LABELS = {key: label for _, members in GROUPS for key, label in members}
FAMILIES = ["VGIGKG"]
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
MAIN = MAIN + [WIDTHS[1]]
WIDTHS = [WIDTHS[0]]
SPECS = MAIN + WIDTHS


# Each member is a canonical architecture key from the complete overview.
COMPARISONS = [('01_charge_allocation', 'Charge allocation around central IG', [('Outside IG', ['central_IG_KG_endcaps', 'central_IG_KG_adjacent', 'KG_clustered_in_VG', 'KG_dispersed_in_VG']), ('Clustered within IG', ['KG_embedded_in_IG', 'single_IG_block_KG_center']), ('Mixed within IG', ['KG_dispersed_in_IG', 'central_mixed_sticker_domain'])]), ('02_IG_fragmentation', 'IG fragmentation and charge interfaces', [('K endcaps', ['central_IG_KG_endcaps', 'two_IG_blocks_KG_endcaps', 'three_medium_IG_blocks_KG_endcaps', 'four_IG_blocks_KG_endcaps']), ('Interfacial K', ['two_IG_blocks_interfacial_KG', 'three_IG_blocks_interfacial_KG', 'four_IG_blocks_interfacial_KG'])]), ('03_domain_position', 'Domain position and terminal order', [('Central / separated IG', ['central_IG_KG_endcaps', 'two_IG_blocks_KG_endcaps', 'two_IG_blocks_KG_center', 'three_IG_blocks_KG_center']), ('IG endblocks', ['IG_endcaps_KG_center', 'KG_capped_IG_endblocks', 'IG_outermost_KG_inside']), ('Terminal order', ['IG_one_end_KG_opposite', 'KG_then_IG_same_end', 'IG_then_KG_same_end'])]), ('04_asymmetry_gradients', 'Asymmetry, gradients and reference designs', [('Symmetry / K allocation', ['two_IG_blocks_KG_endcaps', 'asymmetric_IG_blocks_KG_endcaps', 'charged_large_IG_domain', 'KG_one_end_symmetric_IG']), ('Gradients', ['hydrophobic_gradient', 'symmetric_IG_gradient']), ('Reference designs', ['original_Tk8_adapted', 'even_distribution', 'random_shuffle'])])]

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


def prepare(path, temperatures, include_unverified=False):
    d = pd.read_csv(path)
    required = {"family", "architecture", "ratio", "temperature", "threshold_nm"}
    required.update(spec[0] for spec in SPECS)
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    d["family"] = d.family.astype(str).str.strip().str.upper()
    d["architecture"] = d.get("architecture_key", d.architecture).astype(str).str.strip()
    d["architecture"] = d.architecture.str.replace(r"^asymmetric_IG\d+_IG\d+_KG_endcaps$", "asymmetric_IG_blocks_KG_endcaps", regex=True)
    d["temperature"] = pd.to_numeric(d.temperature.astype(str).str.strip().str.replace(
        r"\s*[Kk]$", "", regex=True), errors="raise")
    d["ratio_number"] = pd.to_numeric(d.ratio.astype(str).str.strip().str.extract(
        r"^[Rr]?(\d+)$", expand=False), errors="raise").astype(int)
    d = d[d.ratio_number.isin(RATIOS) & d.temperature.isin(temperatures)].copy()
    if d.empty:
        raise ValueError("No requested R1-R5 temperature conditions found.")
    if set(d.family) != set(FAMILIES):
        raise ValueError("Expected VGIGKG family.")
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
    # The supplied R3 Tk8 expression has R2 counts; retain only after verification.
    flagged = d.ratio_number.eq(3) & d.architecture.eq("original_Tk8_adapted")
    if not include_unverified:
        d = d.loc[~flagged].copy()
        print("R3 Tk8 excluded pending sequence verification; R3 reference uses 29 designs.")
    # Aggregate replicates first. Every architecture then has equal reference weight.
    metrics = [spec[0] for spec in SPECS]
    cells = d.groupby(keys, as_index=False)[metrics].mean()
    counts = d.groupby(keys).size().rename("n_replicates").reset_index()
    cells = cells.merge(counts, on=keys, validate="one_to_one")
    expected = pd.MultiIndex.from_product(
        [FAMILIES, temperatures, RATIOS, ORDER], names=keys)
    actual = pd.MultiIndex.from_frame(cells[keys])
    if not include_unverified:
        expected = expected.drop(("VGIGKG", temperatures[0], 3, "original_Tk8_adapted")) if len(temperatures)==1 else expected.difference(pd.MultiIndex.from_tuples([("VGIGKG", t, 3, "original_Tk8_adapted") for t in temperatures], names=keys))
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
            loc = (key, ratio, "VGIGKG")
            if loc not in lookup.index:
                continue
            value = float(lookup.loc[loc])
            x = xmap[key] + OFFSETS[ratio]
            values.append(value)
            ax.scatter(x, value, s=27, marker="o", color=COLOURS[ratio],
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
    height = max(7.5, 3.9 + n * 2.05)
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
    fig.subplots_adjust(left=.105, right=.985, top=(.845 if n >= 4 else .72),
                        bottom=2.25 / height, hspace=.52)
    for letter, ax in zip("ABCDE", axes):
        fig.text(.018, ax.get_position().y1 + .008, letter,
                 fontsize=13, fontweight="bold", va="bottom")
    fig.text(.105, .970, f"{title} | {temperature} K",
             fontsize=13, fontweight="bold", va="top")
    fig.text(.105, .942, ("Reference: equal-weight architecture mean within each composition (R3: 29 included designs)" if not args.include_unverified_tk8 else "Reference: 30-architecture mean within each composition; R3 Tk8 unverified"),
             fontsize=9, color="#59636D", va="top")
    ratio_handles = [Line2D([], [], linestyle="", marker="o", markersize=5.8,
                     markerfacecolor=COLOURS[r], markeredgecolor="white",
                     label={1:"R1: VG46 / IG30 / K8",2:"R2: VG38 / IG38 / K8",3:"R3: VG30 / IG46 / K8",4:"R4: VG34 / IG34 / K16",5:"R5: VG30 / IG30 / K24"}[r]) for r in RATIOS]
    fig.legend(handles=ratio_handles, loc="upper left", bbox_to_anchor=(.098, .927),
               ncol=(5 if width >= 11 else 3), frameon=False, columnspacing=1.4, handletextpad=.4, fontsize=8)
    fig.text(.105, .040, "Each point is one architecture mean; no replicate uncertainty is available for a single trajectory.",
             fontsize=8, color="#65717B")
    fig.text(.105, .012, "R1-R3: IG replaces VG at fixed K. R2/R4/R5: increasing K at equal IG/VG. R3 Tk8 omitted: supplied sequence has R2 composition.",
             fontsize=7.5, color="#65717B")
    for extension in ("pdf", "svg", "png"):
        path = args.outdir / f"{stem}_{temperature}K.{extension}"
        fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def composition_overview(effects, temperature, args):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for col, (ratios, title) in enumerate([([1,2,3], "IG enrichment at fixed K8"), ([2,4,5], "Increasing K at equal IG/VG")]):
        for row, (metric, ylabel) in enumerate([("fraction_compact", "Compact fraction"), ("mean_compact_rg_nm", r"Mean compact-state $R_g$ (nm)")]):
            ax = axes[row,col]
            t = effects[(effects.temperature==temperature)&(effects.metric==metric)]
            pivot = t.pivot(index="architecture", columns="ratio_number", values="raw_value").reindex(columns=ratios)
            for _, vals in pivot.iterrows():
                ax.plot(range(3), vals, color="#B8BDC2", alpha=.4, lw=.8, zorder=1)
            ax.plot(range(3), pivot.mean(), color="#303D49", marker="o", lw=2, zorder=3)
            for i,r in enumerate(ratios):
                ax.scatter(np.full(len(pivot),i), pivot[r], color=COLOURS[r], s=9, alpha=.35,zorder=2)
            ax.set_xticks(range(3), [f"R{r}" for r in ratios]);ax.set_ylabel(ylabel)
            ax.grid(axis="y",color="#DCE0E3",ls=":",lw=.6)
            if row==0: ax.set_ylim(0,1);ax.set_title(title)
            ax.text(-.16,1.05,"ABCD"[row*2+col],transform=ax.transAxes,fontweight="bold",fontsize=13)
    fig.suptitle(f"VG/IG/KG composition response | {temperature} K",fontweight="bold")
    fig.text(.1,.025,"Faint lines: individual architectures. Dark line: equal-weight mean. R3 Tk8 omitted: supplied sequence has R2 composition.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95))
    for ext in ["pdf","svg","png"]:
        fig.savefig(args.outdir/f"DR6_VGIGKG_composition_overview_{temperature}K.{ext}",dpi=args.dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("DR6_VGIGKG_comparison_plots"))
    parser.add_argument("--temperatures", type=int, nargs="+", default=[370], choices=[300, 370])
    parser.add_argument("--width", type=float, default=9.5, help="Focused figure width in inches.")
    parser.add_argument("--overview-width", type=float, default=17.0)
    parser.add_argument("--dpi", type=int, default=350, help="PNG resolution; PDF/SVG remain vector.")
    parser.add_argument("--local-y-limits", action="store_true", help="Autoscale each subset; default uses the same metric limits as the overview at that temperature.")
    parser.add_argument("--include-unverified-tk8", action="store_true", help="Include R3 Tk8 only after checking its actual sequence composition.")
    args = parser.parse_args()
    if min(args.width, args.overview_width) < 7 or args.dpi < 72:
        parser.error("Use figure widths >= 7 inches and dpi >= 72.")
    args.temperatures = sorted(set(args.temperatures))
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    verify_coverage()
    style()
    effects, baselines, means = prepare(args.summary, args.temperatures, args.include_unverified_tk8)
    effects.to_csv(args.outdir / "architecture_effects.csv", index=False)
    means.to_csv(args.outdir / "mean_architecture_effects.csv", index=False)
    baselines.to_csv(args.outdir / "absolute_composition_baselines.csv", index=False)
    sets = [("00_overview", "VG/IG/KG architecture overview", GROUPS)] + [
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
            for suffix, specs in [("main", MAIN), ("central50", WIDTHS)]:
                stem = f"DR6_VGIGKG_{name}_{suffix}"
                make_figure(effects, temperature, specs, selected, args, stem,
                            groups, title, overview=name == "00_overview")
                for extension in ["pdf", "svg", "png"]:
                    inventory.append(dict(figure_set=name, temperature_k=temperature,
                                          panels=suffix, file=f"{stem}_{temperature}K.{extension}"))
                print(f"Saved {stem}_{temperature}K (PDF, SVG, PNG)", flush=True)
    for temperature in args.temperatures:
        composition_overview(effects, temperature, args)
    pd.DataFrame(inventory).to_csv(args.outdir / "figure_inventory.csv", index=False)
    notes = (
        "References use all validated architectures within each ratio and temperature, never the plotted subset.\n"
        "R3 Tk8 is excluded by default pending actual-sequence verification: R3 has 29 designs, others 30.\n"
        "Replicates are averaged first; architectures have equal reference weights.\n"
        "Compact fraction: 100*(fraction-reference), percentage points.\n"
        "Rates: log2(rate/arithmetic architecture mean rate).\n"
        "Compact Rg and widths: value-reference, nm.\n"
        "Widths describe the complete Rg ensemble, not uncertainty.\n"
        "No family averaging or family spread: this is one VGIGKG library.\n"
        "All 30 architecture types are covered across focused comparisons.\n"
        "R1-R3 change IG/VG at fixed K; R2/R4/R5 increase K at equal IG/VG.\n"
        "Shared metric limits across overview and subsets unless --local-y-limits.\n"
        "Composition overview: faint lines are architecture responses, dark line is equal-weight mean.\n"
    )
    (args.outdir / "PLOT_NOTES.txt").write_text(notes, encoding="utf-8")
    print(f"\nAll 30 architecture types covered. Saved {len(inventory)//3 + len(args.temperatures)} figures in: {args.outdir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}")
