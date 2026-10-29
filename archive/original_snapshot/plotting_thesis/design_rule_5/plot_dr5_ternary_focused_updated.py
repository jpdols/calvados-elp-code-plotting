#!/usr/bin/env python
"""Focused DR5 ternary architecture comparisons (370 K).

Uses full 22-architecture reference for each family and ratio.
Points: equal-weight average of IG and VG effects.
Pale vertical bars: range of IG/VG effects (NOT confidence intervals).
Sequence cartoons use the supplied R3 motif arrangements only.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

FAMILIES = ("IGKGEG", "VGKGEG")
RATIOS = (1, 2, 3, 4, 5)
NAMES = {
"uniform_opposite_spacing": ("Uniform opposite spacing", "(EG3-IG3-KG3-IG3)×7"),
"uniform_paired_charges": ("Uniform paired charges", "(EG3-KG3-IG6)×7"),
"KG_half_EG_half_distribution": ("K-half / E-half distribution", "(IG3-KG3)×7-(IG3-EG3)×7"),
"paired_charge_endcaps": ("Paired-charge endcaps", "[(EG3-KG3)×3-(EG1-KG1)]-(IG)42-[(EG3-KG3)×3-(EG2-KG2)]"),
"two_paired_charge_clusters": ("Two paired-charge clusters", "(IG)10-[(EG3-KG3)×3-(EG1-KG1)]-(IG)21-[(EG3-KG3)×3-(EG2-KG2)]-(IG)11"),
"three_paired_charge_clusters": ("Three paired-charge clusters", "(EG3-KG3)×2-(IG)21-(EG3-KG3)×3-(IG)21-(EG3-KG3)×2"),
"paired_charge_terminal_block": ("Paired terminal domain", "(EG3-KG3)×7-(IG)42"),
"segregated_charge_terminal_block": ("Segregated terminal domain", "(EG)21-(KG)21-(IG)42"),
"central_paired_charges": ("Paired central domain", "(IG)21-(EG3-KG3)×7-(IG)21"),
"central_segregated_charges": ("Segregated central domain", "(IG)21-(EG)21-(KG)21-(IG)21"),
"terminal_charge_triblock": ("E–H–K terminal triblock", "(EG)21-(IG)42-(KG)21"),
"internal_blocks_close_spacing": ("Internal blocks: close", "(IG)18-(EG)21-(IG)6-(KG)21-(IG)18"),
"internal_blocks_medium_spacing": ("Internal blocks: medium", "(IG)14-(EG)21-(IG)14-(KG)21-(IG)14"),
"internal_blocks_wide_spacing": ("Internal blocks: wide", "(IG)8-(EG)21-(IG)26-(KG)21-(IG)8"),
"EG_endcaps_KG_center": ("E endcaps / K centre", "(EG)10-(IG)21-(KG)21-(IG)21-(EG)11"),
"KG_endcaps_EG_center": ("K endcaps / E centre", "(KG)10-(IG)21-(EG)21-(IG)21-(KG)11"),
"EG_outer_KG_inner_clusters": ("Outer E / inner K clusters", "(EG)10-(IG)14-(KG)10-(IG)14-(KG)11-(IG)14-(EG)11"),
"KG_outer_EG_inner_clusters": ("Outer K / inner E clusters", "(KG)10-(IG)14-(EG)10-(IG)14-(EG)11-(IG)14-(KG)11"),
"alternating_polarity_clusters": ("Alternating-polarity clusters", "(KG)10-(IG)14-(EG)10-(IG)14-(KG)11-(IG)14-(EG)11"),
"EG_endcaps_distributed_KG": ("E endcaps / distributed K", "(EG)10-(IG6-KG3)×7-(EG)11"),
"KG_endcaps_distributed_EG": ("K endcaps / distributed E", "(KG)10-(IG6-EG3)×7-(KG)11"),
"random_shuffle": ("Random shuffle", "Shuffle(IG42-EG21-KG21)")
}
# Ordered from locally mixed / dispersed to increasingly separated charge domains.
# Within each figure the green line identifies the most clearly charge-segregated
# architecture (last row). These are qualitative organization orders, not SCD rankings.
COMPARISONS = [
("01_distributed_vs_paired", "Distributed charges and paired clusters", [
"uniform_paired_charges", "uniform_opposite_spacing",
"paired_charge_endcaps", "three_paired_charge_clusters",
"two_paired_charge_clusters", "KG_half_EG_half_distribution"]),
("02_charge_domains_and_spacing", "Charge-domain organization and separation", [
"paired_charge_terminal_block", "central_paired_charges",
"segregated_charge_terminal_block", "central_segregated_charges",
"internal_blocks_close_spacing", "internal_blocks_medium_spacing",
"internal_blocks_wide_spacing", "terminal_charge_triblock"]),
("03_charge_polarity", "Charge polarity and position", [
"EG_endcaps_distributed_KG", "KG_endcaps_distributed_EG",
"EG_outer_KG_inner_clusters", "KG_outer_EG_inner_clusters",
"alternating_polarity_clusters", "EG_endcaps_KG_center",
"KG_endcaps_EG_center"])
]
METRICS = [
("kcollapse_ns_inv", "A", "Compact-state entry", r"$\log_2(k_{\mathrm{collapse}}/\bar{k}_{\mathrm{collapse}})$", "log"),
("kescape_ns_inv", "B", "Compact-state escape", r"$\log_2(k_{\mathrm{escape}}/\bar{k}_{\mathrm{escape}})$", "log"),
("mean_compact_rg_nm", "C", "Compact-state dimensions", r"$\Delta\langle R_g\rangle_{\mathrm{compact}}$ (nm)", "delta")
]
# Hydrophobic IG = established green (as in DR7); EG = orange; KG = magenta.
MOTIF_COLORS = {"IG": "#5C853F", "EG": "#EF8736", "KG": "#C45B90"}
# Distinct architecture line colours. Green is reserved for the last
# (most segregated) design in every comparison.
LINE_COLORS = ["#E78B32","#8B6BB3","#C45B90","#3F94A0",
               "#B28A36","#737A9B","#A35D56","#5A9B58"]
SEGREGATED_GREEN = "#399653"
def parse_track(expr):
    """Parse EG/KG/IG groups, repetitions, brackets and parentheses."""
    if expr.startswith("Shuffle"):
        return None
    tokens = re.findall(r"EG|KG|IG|\d+|×|[()\[\]-]", expr.replace(" ", ""))
    pos = 0
    def read(stop=None):
        nonlocal pos
        out = []
        while pos < len(tokens):
            t = tokens[pos]
            if t == stop:
                pos += 1
                break
            pos += 1
            if t in ("-",):
                continue
            if t in ("(", "["):
                sub = read(")" if t == "(" else "]")
            elif t in MOTIF_COLORS:
                sub = [t]
            else:
                raise ValueError(f"Unexpected token {t} in {expr}")
            if pos < len(tokens) and tokens[pos] == "×":
                pos += 1
                count = int(tokens[pos]); pos += 1
            elif pos < len(tokens) and tokens[pos].isdigit():
                count = int(tokens[pos]); pos += 1
            else:
                count = 1
            out.extend(sub * count)
        return out
    result = read()
    if len(result) != 84:
        raise ValueError(f"R3 motif track has {len(result)} motifs, expected 84: {expr}")
    return result

def load_effects(summary, temperature):
    d = pd.read_csv(summary)
    # Accept the aggregated analysis output (*_mean columns) without
    # confusing between-replicate SD with the IG/VG family range.
    for metric in ("threshold_nm", "kcollapse_ns_inv", "kescape_ns_inv", "mean_compact_rg_nm"):
        if metric not in d.columns and metric + "_mean" in d.columns:
            d[metric] = d[metric + "_mean"]
    required = {"family","architecture","ratio","temperature","threshold_nm",
                "kcollapse_ns_inv","kescape_ns_inv","mean_compact_rg_nm"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    d["family"] = d.family.astype(str).str.strip().str.upper()
    d["architecture"] = d.architecture.astype(str).str.strip()
    d["ratio_num"] = pd.to_numeric(d.ratio.astype(str).str.extract(r"^[Rr]?(\d+)$",expand=False),errors="raise").astype(int)
    d["temperature_num"] = pd.to_numeric(d.temperature.astype(str).str.replace(r"\s*[Kk]$","",regex=True),errors="raise")
    d = d[(d.temperature_num == temperature) & d.ratio_num.isin(RATIOS)].copy()
    if d.empty:
        raise ValueError(f"No data for {temperature} K")
    if set(d.family) != set(FAMILIES):
        raise ValueError(f"Expected families {FAMILIES}, got {sorted(d.family.unique())}")
    if not np.allclose(pd.to_numeric(d.threshold_nm),4.61):
        raise ValueError("Expected threshold_nm = 4.61")
    if set(d.architecture) != set(NAMES):
        raise ValueError(f"Architecture mismatch: missing {set(NAMES)-set(d.architecture)}, extra {set(d.architecture)-set(NAMES)}")
    if "replicate" not in d:
        d["replicate"] = "single"
    key = ["family","ratio_num","architecture"]
    if d.duplicated(key+["replicate"]).any():
        raise ValueError("Duplicate replicate rows")
    metrics = [x[0] for x in METRICS]
    for m in metrics:
        d[m] = pd.to_numeric(d[m],errors="raise")
        if not np.isfinite(d[m]).all():
            raise ValueError(f"Nonfinite {m}")
    if (d[metrics[:2]] <= 0).any().any():
        raise ValueError("Rates must be positive for log2 transformation")
    cells = d.groupby(key,as_index=False)[metrics].mean()
    expected = pd.MultiIndex.from_product([FAMILIES,RATIOS,list(NAMES)],names=key)
    missing = expected.difference(pd.MultiIndex.from_frame(cells[key]))
    if len(missing):
        raise ValueError(f"Missing family/ratio/architecture combinations: {list(missing)[:8]}")
    parts = []
    for m,_,_,_,kind in METRICS:
        reference = cells.groupby(["family","ratio_num"])[m].transform("mean")
        tmp = cells[key].copy()
        tmp["metric"] = m
        tmp["raw_value"] = cells[m]
        tmp["reference"] = reference
        tmp["effect"] = np.log2(cells[m]/reference) if kind=="log" else cells[m]-reference
        parts.append(tmp)
    return pd.concat(parts,ignore_index=True)

def draw_tracks(ax, keys, colors):
    ax.set_xlim(0,1)
    ax.set_ylim(-.7,len(keys)+.65)
    ax.axis("off")
    ax.text(.01, len(keys)+.20, "Sequence architecture: VPGIG / VPGEG / VPGKG (R3)",
            fontsize=10, weight="bold", va="bottom")
    x0, x1 = .49, .985
    for i,key in enumerate(keys):
        y = len(keys)-i-1
        label, expr = NAMES[key]
        ax.plot([.015,.055],[y,y],color=colors[i],lw=3,solid_capstyle="round")
        ax.text(.065,y,f"{label}",va="center",fontsize=9.4)
        track = parse_track(expr)
        if track is None:
            ax.text(x0,y,"Randomized sequence",fontsize=8,va="center",color="#666")
        else:
            w=(x1-x0)/84
            for j,m in enumerate(track):
                ax.add_patch(Rectangle((x0+j*w,y-.17),w+.0001,.34,
                                       color=MOTIF_COLORS[m],linewidth=0))
            ax.add_patch(Rectangle((x0,y-.17),x1-x0,.34,fill=False,
                                   edgecolor="#777",linewidth=.35))
    for j,(motif,col) in enumerate(MOTIF_COLORS.items()):
        ax.add_patch(Rectangle((.51+j*.12,-.62),.013,.12,color=col))
        ax.text(.526+j*.12,-.56,motif,fontsize=7.5,va="center")

def figure(effects, keys, title, stem, outdir, temp, width, dpi):
    colors = LINE_COLORS[:len(keys)-1] + [SEGREGATED_GREEN]
    fig = plt.figure(figsize=(width, 12.8 + .23*len(keys)))
    gs = fig.add_gridspec(4,1,height_ratios=[.35+.075*len(keys),1,1,1],
                          hspace=.35,top=.975,bottom=.065,left=.12,right=.96)
    track_ax = fig.add_subplot(gs[0])
    draw_tracks(track_ax,keys,colors)
    axes = [fig.add_subplot(gs[i]) for i in range(1,4)]
    grouped = effects.groupby(["architecture","ratio_num","metric"]).effect.agg(
        mean="mean",low="min",high="max").reset_index()
    lookup = grouped.set_index(["architecture","ratio_num","metric"])
    for ax,(metric,letter,panel,ylabel,_) in zip(axes,METRICS):
        for i,key in enumerate(keys):
            points = lookup.loc[[(key,r,metric) for r in RATIOS]]
            y = points["mean"].to_numpy()
            lo = points["low"].to_numpy()
            hi = points["high"].to_numpy()
            x = np.array(RATIOS,dtype=float)
            ax.vlines(x,lo,hi,color=colors[i],alpha=.25,lw=2.4,zorder=2)
            ax.plot(x,y,"-o",color=colors[i],lw=2.5,ms=5.2,zorder=3)
        ax.axhline(0,ls="--",color="#666666",lw=1.45,zorder=1)
        ax.grid(axis="y",color="#E6E6E6",lw=.7,zorder=0)
        ax.set_xlim(.8,5.2)
        ax.set_xticks(RATIOS,[f"R{r}\n({100*r/6:.1f}%)" for r in RATIOS])
        ax.set_ylabel(ylabel,fontsize=10)
        ax.set_title(f"{letter}   {panel}", loc="left", weight="bold",
                     fontsize=12, pad=11)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel("Charged-motif fraction (E + K)",fontsize=10)
    
    
    for ext in ("pdf","svg","png"):
        fig.savefig(outdir/f"{stem}_{temp}K.{ext}",dpi=dpi,bbox_inches="tight",pad_inches=.16)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",required=True,type=Path)
    parser.add_argument("--outdir",default=Path("DR5_ternary_focused"),type=Path)
    parser.add_argument("--temperature",default=370,type=int)
    parser.add_argument("--width",default=11.2,type=float)
    parser.add_argument("--dpi",default=350,type=int)
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    for key,(_,pattern) in NAMES.items():
        parse_track(pattern)
    effects=load_effects(args.summary,args.temperature)
    effects.to_csv(args.outdir/"DR5_family_effects.csv",index=False)
    avg=effects.groupby(["architecture","ratio_num","metric"]).effect.agg(
        mean_IG_VG="mean",IG_VG_min="min",IG_VG_max="max").reset_index()
    avg.to_csv(args.outdir/"DR5_IG_VG_mean_and_spread.csv",index=False)
    for stem,title,keys in COMPARISONS:
        figure(effects,keys,title,f"DR5_{stem}",args.outdir,args.temperature,args.width,args.dpi)
        print(f"Saved DR5_{stem}_{args.temperature}K.pdf/.svg/.png")
    print(f"Output: {args.outdir.resolve()}")
if __name__=="__main__":
    main()
