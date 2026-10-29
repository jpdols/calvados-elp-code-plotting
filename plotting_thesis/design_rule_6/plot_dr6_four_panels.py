#!/usr/bin/env python3
"""DR6 candidate-subrule screening, split into two interpretable ratio series.

R1/R2/R3: VG/IG varies at fixed KG=8.
R2/R4/R5: VG=IG while KG rises 8 -> 16 -> 24.

Input may be DR6_architecture_effects.csv, summary_per_replicate.csv,
or summary_across_replicates.csv. Uses the exact R4 motif expressions supplied
by the user; all explicitly supplied expressions are validated as 84 motifs,
with the correct R4 composition (34 VG, 34 IG, 16 KG).

The sequence cartoons show R4 *only*, not the precise architecture for other
compositions. Random shuffle and even distribution do not have an exact motif
order supplied and are clearly marked as schematic/unavailable (no fabricated
sequence). Plots show architecture effects relative to ALL 30 designs at each
ratio. Data represent exploratory MD trajectories; no error bars or statistical
significance are inferred from different architectures or ratios.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

COMPOSITION = {1:(46,30,8), 2:(38,38,8), 3:(30,46,8), 4:(34,34,16), 5:(30,30,24)}
RATIO_GROUPS = {'fixed_KG_R1_R2_R3': (1,2,3), 'balanced_VG_IG_R2_R4_R5': (2,4,5)}
MOTIF_COLORS = {'VG':'#4F86CF', 'IG':'#5E8742', 'KG':'#C15A91'}
LINE_COLORS = ['#E78B32','#8B6BB3','#CA4E8A','#3693A2','#B28A36','#737A9B','#A35D56','#399653','#6F7D3E']
METRICS = [
    ('kcollapse_ns_inv','A','Compact-state entry',r'$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$','log'),
    ('kescape_ns_inv','B','Compact-state escape',r'$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$','log'),
    ('mean_compact_rg_nm','C','Compact-state dimensions',r'$\Delta\langle R_g\rangle_{\mathrm{compact}}$ (nm)','delta'),
]

# The complete, user-supplied R4 architecture manifest. Expressions are
# pentapeptide units; KG=VPGKG, IG=VPGIG, VG=VPGVG. None are inferred from MD.
# 773 and 774 were supplied as descriptions only, so exact sequence is unknown.
R4_MANIFEST = {
  773: ('random_shuffle', None),
  774: ('even_distribution', None),
  775: ('original_Tk8_adapted', '(KG)1-(IG)14-(VG2-KG)3-(VG2-IG)-(VG3-KG)-(VG2-IG)-(VG-KG-IG)-(VG-KG)-(VG2-KG)-(KG-VG2)-(KG-VG)-(IG-KG-VG)-(IG-VG2)-(KG-VG3)-(IG-VG2)-(KG-VG2)3-(IG)14-(KG)1'),
  776: ('central_IG_KG_endcaps', '(KG)8-(VG)17-(IG)34-(VG)17-(KG)8'),
  777: ('central_IG_KG_adjacent', '(VG)17-(KG)8-(IG)34-(KG)8-(VG)17'),
  778: ('KG_embedded_in_IG', '(VG)17-(IG)9-(KG)8-(IG)16-(KG)8-(IG)9-(VG)17'),
  779: ('KG_dispersed_in_IG', '(VG)17-(IG2-KG)16-(IG)2-(VG)17'),
  780: ('central_mixed_sticker_domain', '(VG)17-(IG2-KG)8-(IG)2-(KG-IG2)8-(VG)17'),
  781: ('two_IG_blocks_KG_endcaps', '(KG)8-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9-(KG)8'),
  782: ('two_IG_blocks_interfacial_KG', '(VG)9-(IG)17-(KG)8-(VG)16-(KG)8-(IG)17-(VG)9'),
  783: ('two_IG_blocks_KG_center', '(VG)9-(IG)17-(VG)8-(KG)16-(VG)8-(IG)17-(VG)9'),
  784: ('IG_endcaps_KG_center', '(IG)17-(VG)17-(KG)16-(VG)17-(IG)17'),
  785: ('KG_capped_IG_endblocks', '(KG)8-(IG)17-(VG)34-(IG)17-(KG)8'),
  786: ('IG_outermost_KG_inside', '(IG)17-(KG)8-(VG)34-(KG)8-(IG)17'),
  787: ('three_medium_IG_blocks_KG_endcaps', '(KG)8-(VG)6-(IG)11-(VG)11-(IG)12-(VG)11-(IG)11-(VG)6-(KG)8'),
  788: ('three_IG_blocks_interfacial_KG', '(VG)6-(IG)11-(KG)4-(VG)11-(KG)4-(IG)12-(KG)4-(VG)11-(KG)4-(IG)11-(VG)6'),
  789: ('three_IG_blocks_KG_center', '(VG)6-(IG)11-(VG)11-(IG)12-(KG)16-(IG)11-(VG)17'),
  790: ('four_IG_blocks_KG_endcaps', '(KG)8-(VG)4-(IG)9-(VG)8-(IG)8-(VG)10-(IG)8-(VG)8-(IG)9-(VG)4-(KG)8'),
  791: ('IG_one_end_KG_opposite', '(IG)34-(VG)34-(KG)16'),
  792: ('KG_then_IG_same_end', '(VG)34-(KG)16-(IG)34'),
  793: ('IG_then_KG_same_end', '(VG)34-(IG)34-(KG)16'),
  794: ('asymmetric_IG22_IG12_KG_endcaps', '(KG)8-(IG)22-(VG)17-(IG)12-(VG)17-(KG)8'),
  795: ('charged_large_IG_domain', '(IG)22-(KG)16-(VG)17-(IG)12-(VG)17'),
  796: ('KG_one_end_symmetric_IG', '(KG)16-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9'),
  797: ('hydrophobic_gradient', '(KG)16-(VG)11-(IG)3-(VG)9-(IG)5-(VG)7-(IG)7-(VG)5-(IG)9-(VG)2-(IG)10'),
  798: ('symmetric_IG_gradient', '(KG)8-(VG)7-(IG)4-(VG)5-(IG)6-(VG)5-(IG)14-(VG)5-(IG)6-(VG)5-(IG)4-(VG)7-(KG)8'),
  799: ('single_IG_block_KG_center', '(VG)17-(IG)17-(KG)16-(IG)17-(VG)17'),
  800: ('KG_clustered_in_VG', '(VG)8-(KG)8-(VG)9-(IG)34-(VG)9-(KG)8-(VG)8'),
  801: ('KG_dispersed_in_VG', '(VG2-KG)8-(VG)1-(IG)34-(VG)1-(KG-VG2)8'),
  802: ('four_IG_blocks_interfacial_KG', '(VG)4-(IG)9-(KG)4-(VG)8-(IG)8-(KG)4-(VG)10-(KG)4-(IG)8-(VG)8-(KG)4-(IG)9-(VG)4'),
}

# Each is a hypothesis-generating comparison; broad enough to screen multiple
# candidate subrules. Paired comparisons are additionally quantified below.
COMPARISONS = [
    ('6a_charge_placement_and_dispersion', 'Charge placement and dispersion', [
        'KG_embedded_in_IG', 'single_IG_block_KG_center',
        'KG_dispersed_in_VG', 'KG_dispersed_in_IG',
        'central_mixed_sticker_domain']),
    ('6b_domain_layout_overview', 'KG placement and IG domain layout', [
        'KG_capped_IG_endblocks', 'charged_large_IG_domain',
        'KG_embedded_in_IG', 'four_IG_blocks_KG_endcaps',
        'single_IG_block_KG_center', 'two_IG_blocks_KG_endcaps',
        'IG_endcaps_KG_center', 'two_IG_blocks_KG_center',
        'central_IG_KG_endcaps']),
    ('6c_terminal_vs_interfacial_blocks', 'IG fragmentation: terminal versus interfacial KG', [
        'four_IG_blocks_KG_endcaps', 'single_IG_block_KG_center',
        'two_IG_blocks_KG_endcaps', 'three_medium_IG_blocks_KG_endcaps',
        'two_IG_blocks_interfacial_KG', 'three_IG_blocks_interfacial_KG',
        'four_IG_blocks_interfacial_KG']),
    ('6d_charge_domain_vs_dispersion', 'Coherent IG domains versus charge dispersion', [
        'charged_large_IG_domain', 'KG_embedded_in_IG',
        'single_IG_block_KG_center', 'IG_endcaps_KG_center',
        'KG_dispersed_in_VG', 'KG_clustered_in_VG',
        'even_distribution']),
]

# Explicit within-ratio matched differences. A - B; positive collapse and
# negative escape/Rg indicate improvement in A, but no causality is assumed.
PAIRS = [
 ('KG dispersed in VG - dispersed in IG', 'KG_dispersed_in_VG','KG_dispersed_in_IG'),
 ('KG embedded in IG - dispersed in IG', 'KG_embedded_in_IG','KG_dispersed_in_IG'),
 ('KG clustered in VG - dispersed in VG', 'KG_clustered_in_VG','KG_dispersed_in_VG'),
 ('Central IG: terminal KG - adjacent KG', 'central_IG_KG_endcaps','central_IG_KG_adjacent'),
 ('IG endblocks: terminal KG - inward KG', 'KG_capped_IG_endblocks','IG_outermost_KG_inside'),
 ('Two IG blocks: terminal KG - interfacial KG', 'two_IG_blocks_KG_endcaps','two_IG_blocks_interfacial_KG'),
 ('Three IG blocks: terminal KG - interfacial KG', 'three_medium_IG_blocks_KG_endcaps','three_IG_blocks_interfacial_KG'),
 ('Four IG blocks: terminal KG - interfacial KG', 'four_IG_blocks_KG_endcaps','four_IG_blocks_interfacial_KG'),
 ('Hydrophobic symmetric gradient - one-sided gradient', 'symmetric_IG_gradient','hydrophobic_gradient'),
 ('KG endcaps: symmetric IG - asymmetric IG', 'two_IG_blocks_KG_endcaps','asymmetric_IG22_IG12_KG_endcaps'),
 ('KG endcaps: two IG blocks - one-end KG', 'two_IG_blocks_KG_endcaps','KG_one_end_symmetric_IG'),
 ('KG capping IG - internal KG center', 'KG_capped_IG_endblocks','IG_endcaps_KG_center'),
]


def canonical(raw):
    """Normalize R1–R5 architecture names; unify composition-specific asymmetry labels."""
    s=str(raw).strip().lower()
    s=re.sub(r'^\d{3}[_-]','',s)
    s=re.sub(r'^vgigkg[_-]r[1-5][_-]','',s)
    s=re.sub(r'[^a-z0-9]','',s)
    if re.fullmatch(r'asymmetricig\d+ig\d+kgendcaps',s):
        return 'asymmetric_IG22_IG12_KG_endcaps'
    return s


def parse_motif_pattern(expr):
    """Parse full nested groups, repeated motifs and '-'; e.g. (VG2-KG)3."""
    if expr is None:
        return None
    normalized=re.sub(r'\s+','',expr.upper()).replace('×','*')
    tokens=re.findall(r'VG|IG|KG|\d+|[()\[\]\-]',normalized)
    if ''.join(tokens)!=normalized:
        raise ValueError(f'Unrecognized symbol in motif expression: {expr}')
    pos=0
    def group(stop=None):
        nonlocal pos
        out=[]
        while pos<len(tokens):
            tok=tokens[pos]
            if tok in (')',']'):
                if tok != stop: raise ValueError(f'Mismatched bracket in {expr}')
                pos+=1
                return out
            pos+=1
            if tok=='-':continue
            if tok in ('(', '['):
                block=group(')' if tok=='(' else ']')
            elif tok in MOTIF_COLORS:
                block=[tok]
            else:
                raise ValueError(f'Unexpected token {tok!r} in {expr}')
            repeat=1
            if pos<len(tokens) and tokens[pos].isdigit():
                repeat=int(tokens[pos]);pos+=1
            out+=block*repeat
        if stop is not None: raise ValueError(f'Unclosed bracket in {expr}')
        return out
    parsed=group()
    if pos!=len(tokens):raise ValueError(f'Unparsed tokens in {expr}')
    return parsed


def validate_manifest():
    if sorted(R4_MANIFEST)!=list(range(773,803)):
        raise ValueError('R4 manifest must contain all IDs 773–802 once')
    result={}
    for ident,(name,expr) in R4_MANIFEST.items():
        parsed=parse_motif_pattern(expr)
        if parsed is not None:
            counts=Counter(parsed)
            if len(parsed)!=84 or (counts['VG'],counts['IG'],counts['KG'])!=COMPOSITION[4]:
                raise ValueError(f'Invalid R4 track {ident} {name}: length={len(parsed)}, counts={dict(counts)}')
        result[canonical(name)]=dict(id=ident,name=name,track=parsed)
    if len(result)!=30:raise ValueError('Duplicate normalized architecture name in R4 manifest')
    return result


def load_effects(path,temperature=370):
    d=pd.read_csv(path)
    if {'ratio_num','key','raw_value','metric'}.issubset(d.columns):
        d=d[['ratio_num','key','raw_value','metric']].copy()
        d['ratio_num']=pd.to_numeric(d['ratio_num'],errors='raise').astype(int)
        d['key']=d.key.map(canonical)
        d['raw_value']=pd.to_numeric(d.raw_value,errors='raise')
    else:
        if not {'architecture','ratio','temperature'}.issubset(d.columns):
            raise ValueError(f'Unrecognized input columns: {list(d.columns)}')
        if 'family' in d:
            families=d.family.dropna().astype(str).str.strip().str.upper().unique()
            if set(families)!={'VGIGKG'}:
                d=d.loc[d.family.astype(str).str.strip().str.upper()=='VGIGKG'].copy()
                if d.empty:raise ValueError(f'No VGIGKG family. Available families: {families}')
        temp=pd.to_numeric(d.temperature.astype(str).str.replace(r'\s*[kK]$','',regex=True),errors='coerce')
        ratio=pd.to_numeric(d.ratio.astype(str).str.extract(r'^[Rr]?(\d+)$',expand=False),errors='coerce')
        d=d.loc[temp==temperature].copy()
        d['ratio_num']=ratio[temp==temperature].astype(int)
        if d.empty:raise ValueError(f'No records at {temperature} K')
        d['key']=d.architecture.map(canonical)
        raw_cols={}
        for metric,*_ in METRICS:
            col=metric if metric in d else metric+'_mean'
            if col not in d: raise ValueError(f'Missing input metric {metric} or {metric+"_mean"}')
            raw_cols[metric]=col
        means=d.groupby(['ratio_num','key'],as_index=False)[list(raw_cols.values())].mean()
        means=means.rename(columns={v:k for k,v in raw_cols.items()})
        d=means.melt(id_vars=['ratio_num','key'],value_vars=[x[0] for x in METRICS],var_name='metric',value_name='raw_value')
    d=d[d.ratio_num.isin(COMPOSITION)].copy()
    wanted={m[0] for m in METRICS}
    if set(d.metric)!=wanted:raise ValueError(f'Expected metrics {wanted}; got {set(d.metric)}')
    if d.duplicated(['ratio_num','key','metric']).any():raise ValueError('Duplicate ratio/key/metric; do not merge separate designs')
    if not np.isfinite(d.raw_value).all():raise ValueError('NaN/inf in metrics')
    rates=d.metric.isin(['kcollapse_ns_inv','kescape_ns_inv'])
    if (d.loc[rates,'raw_value']<=0).any():raise ValueError('Nonpositive kinetic rates')
    ref=d.groupby(['ratio_num','metric']).raw_value.transform('mean')
    d['reference']=ref
    d['effect']=np.where(rates,np.log2(d.raw_value/ref),d.raw_value-ref)
    ratio_counts=d[d.metric==METRICS[0][0]].groupby('ratio_num').key.nunique()
    print('Validated architecture counts by ratio:',ratio_counts.to_dict())
    if not all(ratio_counts.get(r,0)==30 for r in COMPOSITION):
        raise ValueError('Require complete 30-architecture libraries for correct normalization')
    expected_keys={canonical(name) for name,_ in R4_MANIFEST.values()}
    for ratio in COMPOSITION:
        actual=set(d.loc[(d.ratio_num==ratio)&(d.metric==METRICS[0][0]),'key'])
        if actual!=expected_keys:
            raise ValueError(f'R{ratio} key mismatch: missing {sorted(expected_keys-actual)}; unexpected {sorted(actual-expected_keys)}')
    return d.sort_values(['ratio_num','metric','key']).reset_index(drop=True)


def args_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input','--summary','--effects',dest='input',type=Path,required=True,
                   help='Existing DR6_architecture_effects.csv or DR6 per/across replicate summary CSV')
    p.add_argument('--outdir',type=Path,default=Path('DR6_ratio_split'))
    p.add_argument('--temperature',type=int,default=370)
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--width',type=float,default=13.1)
    p.add_argument('--sets',choices=['both','fixed_KG','balanced'],default='both')
    p.add_argument('--only',nargs='*',default=None,help='Figure stems/numbers e.g. --only 03 04')
    p.add_argument('--validate-only',action='store_true',help='Check all 30 exact R4 architectures and data then exit')
    return p.parse_args()


def choose_comparisons(only):
    if not only:return COMPARISONS
    return [rec for rec in COMPARISONS if any(rec[0]==x or rec[0].startswith(x) for x in only)]


def draw_tracks(ax,selected,manifest):
    n=len(selected)
    ax.set_xlim(0,1);ax.set_ylim(-0.95,n+.85);ax.axis('off')
    ax.text(.012,n+.46,'R4 sequence architectures (34 VG + 34 IG + 16 KG)',fontsize=14,weight='bold',va='center')
    x0,x1=.53,.99
    for idx,name in enumerate(selected):
        rec=manifest[canonical(name)]
        y=n-idx-1
        col=LINE_COLORS[idx % len(LINE_COLORS)]
        ax.plot([.015,.052],[y,y],color=col,lw=4.1,solid_capstyle='round')
        ax.text(.065,y,f"{rec['id']}  {rec['name'].replace('_',' ')}",fontsize=11,va='center')
        track=rec['track']
        if track is None:
            ax.text(x0+.025,y,'Exact sequence order not supplied',color='#777777',fontsize=10,va='center')
            continue
        w=(x1-x0)/84
        for j,m in enumerate(track):
            ax.add_patch(Rectangle((x0+j*w,y-.20),w+.00004,.40,linewidth=0,facecolor=MOTIF_COLORS[m]))
        ax.add_patch(Rectangle((x0,y-.20),x1-x0,.40,facecolor='none',edgecolor='#888888',lw=.35))
    legend=[Patch(facecolor=c,edgecolor='none',label={'VG':'VPGVG','IG':'VPGIG','KG':'VPGKG'}[k]) for k,c in MOTIF_COLORS.items()]
    # Use artist legend so doesn't collide with architecture labels.
    ax.legend(handles=legend,loc='lower right',ncol=3,bbox_to_anchor=(.995,-.04),frameon=False,fontsize=11,
              borderaxespad=0,handlelength=1.8)


def plot_themes(effects,selected,manifest,group_name,ratios,outdir,dpi,width):
    stem,title,names=selected
    kmap={canonical(name):name for name in names}
    figure=plt.figure(figsize=(width,13.5+0.43*max(0,len(names)-6)))
    topheight=1.02+0.135*len(names)
    gs=figure.add_gridspec(4,1,height_ratios=[topheight,1,1,1],hspace=.41,
                           left=.13,right=.96,top=.965,bottom=.105)
    track_ax=figure.add_subplot(gs[0]);draw_tracks(track_ax,names,manifest)
    subplots=[figure.add_subplot(gs[i]) for i in range(1,4)]
    index=effects.set_index(['ratio_num','key','metric'])
    x=np.arange(len(ratios))
    for ax,(metric,letter,panel,ylabel,kind) in zip(subplots,METRICS):
        for i,name in enumerate(names):
            key=canonical(name)
            y=[float(index.loc[(r,key,metric),'effect']) for r in ratios]
            ax.plot(x,y,'-o',color=LINE_COLORS[i],lw=2.5,ms=7,zorder=3)
        ax.axhline(0,color='#777777',ls='--',lw=1.35,zorder=1)
        ax.grid(axis='y',color='#E8E8E8',lw=.8,zorder=0)
        ax.set_xlim(-.18,len(ratios)-.82)
        ax.set_xticks(x,[f"R{r}\nVG {COMPOSITION[r][0]} / IG {COMPOSITION[r][1]} / KG {COMPOSITION[r][2]}" for r in ratios],fontsize=11)
        ax.set_ylabel(ylabel,fontsize=12)
        ax.set_title(f'{letter}  {panel}',loc='left',weight='bold',fontsize=15,pad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y',labelsize=12)
    subplots[-1].set_xlabel('Composition',fontsize=12,labelpad=9)
    if group_name.startswith('fixed'):
        figure.suptitle('Fixed KG = 8: increasing IG at the expense of VG',fontsize=17,weight='bold',y=.997)
        note='R1–R3 isolate the VG/IG balance at fixed KG content.'
    else:
        note='R2/R4/R5 maintain VG = IG while increasing KG from 8 to 24.'
    figure.text(.13,.043,'Zero = all-30-architecture mean at each ratio  |  Lines = same architecture motif-layout strategy',fontsize=10,color='#444')
    figure.text(.13,.026,f'{note}  |  Tracks are R4 examples only; one simulation per sequence.',fontsize=9.5,color='#666')
    dest=outdir/group_name/f'DR6_{stem}_{group_name}_370K'
    dest.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        figure.savefig(f'{dest}.{ext}',dpi=dpi,bbox_inches='tight',pad_inches=.14,facecolor='white')
    plt.close(figure)
    print('Saved',dest.with_suffix('.pdf'))


def paired_table(effects):
    d=effects.pivot_table(index=['ratio_num','key'],columns='metric',values='raw_value')
    rows=[]
    for label,a,b in PAIRS:
        for r in COMPOSITION:
            ar=d.loc[(r,canonical(a))];br=d.loc[(r,canonical(b))]
            rows.append(dict(pair=label,architecture_A=a,architecture_B=b,ratio_num=r,
                A_collapse=float(ar['kcollapse_ns_inv']),B_collapse=float(br['kcollapse_ns_inv']),
                A_escape=float(ar['kescape_ns_inv']),B_escape=float(br['kescape_ns_inv']),
                A_compact_rg=float(ar['mean_compact_rg_nm']),B_compact_rg=float(br['mean_compact_rg_nm']),
                delta_collapse=float(ar['kcollapse_ns_inv']-br['kcollapse_ns_inv']),
                delta_escape=float(ar['kescape_ns_inv']-br['kescape_ns_inv']),
                delta_compact_rg=float(ar['mean_compact_rg_nm']-br['mean_compact_rg_nm']),
                log2_collapse_ratio=float(np.log2(ar['kcollapse_ns_inv']/br['kcollapse_ns_inv'])),
                log2_escape_ratio=float(np.log2(ar['kescape_ns_inv']/br['kescape_ns_inv']))))
    return pd.DataFrame(rows)


def consistency_table(paired):
    rows=[]
    for pair,g in paired.groupby('pair',sort=False):
        for group,ratios in {'R1_R2_R3':(1,2,3),'R2_R4_R5':(2,4,5),'R1_to_R5':(1,2,3,4,5)}.items():
            x=g[g.ratio_num.isin(ratios)]
            rows.append(dict(pair=pair,ratios=group,
                 faster_entry_count=int((x.delta_collapse>0).sum()),
                 lower_escape_count=int((x.delta_escape<0).sum()),
                 smaller_compact_rg_count=int((x.delta_compact_rg<0).sum()),
                 n_ratios=len(x),mean_delta_collapse=float(x.delta_collapse.mean()),
                 mean_delta_escape=float(x.delta_escape.mean()),
                 mean_delta_compact_rg=float(x.delta_compact_rg.mean())))
    return pd.DataFrame(rows)


PAIRED_GROUPS = [
    ('01_charge_environment', PAIRS[0:3]),
    ('02_charge_position', PAIRS[3:5] + PAIRS[11:12]),
    ('03_fragmentation', PAIRS[5:8]),
    ('04_gradients_and_symmetry', PAIRS[8:11]),
]


def plot_paired(paired, pair_group, group_name, ratios, outdir, dpi):
    pair_stem, pair_specs=pair_group
    included=[p[0] for p in pair_specs]
    subset=paired[paired.pair.isin(included)]
    fig, axes=plt.subplots(3,1,figsize=(12.2,11.4),sharex=True)
    x=np.arange(len(ratios))
    attrs=[('delta_collapse','A   Paired difference: compact-state entry',r'$\Delta k_{\mathrm{collapse}}$ (ns$^{-1}$)'),
           ('delta_escape','B   Paired difference: compact-state escape',r'$\Delta k_{\mathrm{escape}}$ (ns$^{-1}$)'),
           ('delta_compact_rg','C   Paired difference: compact-state size',r'$\Delta\langle R_g\rangle_{\mathrm{compact}}$ (nm)')]
    for ax,(attr,title,ylabel) in zip(axes,attrs):
        for i,pair in enumerate(included):
            y=[float(subset.loc[(subset.pair==pair)&(subset.ratio_num==r),attr].iloc[0]) for r in ratios]
            ax.plot(x,y,'-o',lw=2.65,ms=6.5,color=LINE_COLORS[i],label=pair)
        ax.axhline(0,lw=1.15,ls='--',color='#888')
        ax.grid(axis='y',alpha=.2)
        ax.set_ylabel(ylabel,fontsize=12)
        ax.set_title(title,fontsize=14,weight='bold',loc='left')
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='y',labelsize=11)
    axes[-1].set_xticks(x,[f'R{r} (VG{COMPOSITION[r][0]}/IG{COMPOSITION[r][1]}/KG{COMPOSITION[r][2]})' for r in ratios])
    # Ratio ticks already identify composition; omit xlabel to prevent footnote overlap.
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.50,.034),ncol=1,
               fontsize=10.5,frameon=False)
    fig.subplots_adjust(left=.14,right=.97,top=.95,bottom=.255,hspace=.35)
    fig.text(.14,.205,'Difference = A minus B. Favourable A: entry > 0; escape < 0; compact Rg < 0.',fontsize=10)
    fig.text(.14,.185,'Paired comparisons within each ratio, not statistical uncertainty intervals.',fontsize=9.5,color='#666')
    dest=outdir/group_name/f'DR6_paired_{pair_stem}_{group_name}_370K'
    dest.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(f'{dest}.{ext}',dpi=dpi,bbox_inches='tight',pad_inches=.14,facecolor='white')
    plt.close(fig)
    print('Saved',dest.with_suffix('.pdf'))


def plot_library_baselines(effects,outdir,dpi):
    """Absolute 30-design means; useful because ratio-normalized effect plots hide them."""
    base=(effects.groupby(['ratio_num','metric'],as_index=False)['reference'].first())
    base.to_csv(outdir/'DR6_absolute_library_means.csv',index=False)
    fig,axs=plt.subplots(3,2,figsize=(13.0,11.8))
    for j,(name,ratios) in enumerate(RATIO_GROUPS.items()):
        for i,(metric,letter,title,ylabel,_) in enumerate(METRICS):
            ax=axs[i,j]
            vals=[float(base.loc[(base.ratio_num==r)&(base.metric==metric),'reference'].iloc[0]) for r in ratios]
            ax.plot(range(3),vals,'-o',color='#31688E',lw=3,ms=9)
            ax.set_xticks(range(3),[f'R{r}\nVG{COMPOSITION[r][0]} IG{COMPOSITION[r][1]} KG{COMPOSITION[r][2]}' for r in ratios])
            ax.set_ylabel({'kcollapse_ns_inv':r'Mean $k_{\mathrm{collapse}}$ (ns$^{-1}$)',
                           'kescape_ns_inv':r'Mean $k_{\mathrm{escape}}$ (ns$^{-1}$)',
                           'mean_compact_rg_nm':r'Mean $\langle R_g\rangle_{\mathrm{compact}}$ (nm)'}[metric],fontsize=11)
            ax.grid(axis='y',alpha=.17)
            ax.set_title(f'{letter}  {title}',fontsize=12,fontweight='bold',loc='left')
            ax.spines[['right','top']].set_visible(False)
            ax.tick_params(labelsize=10)
    axs[0,0].text(.5,1.27,'KG fixed: VG → IG',ha='center',va='center',transform=axs[0,0].transAxes,fontsize=17,fontweight='bold')
    axs[0,1].text(.5,1.27,'VG = IG: increasing KG',ha='center',va='center',transform=axs[0,1].transAxes,fontsize=17,fontweight='bold')
    fig.subplots_adjust(top=.91,bottom=.1,left=.09,right=.98,wspace=.3,hspace=.47)
    fig.text(.09,.038,'Absolute means of all 30 architectures per ratio at 370 K; no uncertainty available from single trajectories.',fontsize=10,color='#555')
    dest=outdir/'DR6_00_absolute_library_means_370K'
    for ext in ('pdf','svg','png'):
        fig.savefig(f'{dest}.{ext}',dpi=dpi,bbox_inches='tight',pad_inches=.14,facecolor='white')
    plt.close(fig)
    print('Saved',dest.with_suffix('.pdf'))


def main():
    a=args_parser()
    manifest=validate_manifest()
    print('R4 manifest validated:',len(manifest),'architectures; 28 exact 84-motif tracks + 2 descriptions')
    effects=load_effects(a.input,a.temperature)
    a.outdir.mkdir(parents=True,exist_ok=True)
    if a.validate_only: return
    paired=paired_table(effects)
    score=consistency_table(paired)
    group_specs=[(name,ratios) for name,ratios in RATIO_GROUPS.items() if name.startswith('balanced')]
    comps=choose_comparisons(a.only)
    if a.only is not None and not comps: raise ValueError(f'No thematic figures selected by --only {a.only}')
    for group_name,ratios in group_specs:
        for rec in comps:
            plot_themes(effects,rec,manifest,group_name,ratios,a.outdir,a.dpi,a.width)
    print('\nCandidate subrule consistency:')
    for row in score[score.ratios=='R2_R4_R5'].itertuples(index=False):
        print(f'  {row.pair}: entry {row.faster_entry_count}/3, escape {row.lower_escape_count}/3, compact Rg {row.smaller_compact_rg_count}/3')
    print('Output:',a.outdir.resolve())

if __name__=='__main__':
    main()
