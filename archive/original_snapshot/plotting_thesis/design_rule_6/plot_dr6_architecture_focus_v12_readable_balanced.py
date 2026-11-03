#!/usr/bin/env python
"""Final DR6 thesis plots with verified 84-motif VG/IG/KG sequences.

Inputs: per-replicate summary CSV.
Outputs: four title-free main figures (three kinetic comparisons and compactness), supplementary catalogue and comparisons, optional cross-ratio contrasts.
Dependencies: numpy, pandas, matplotlib.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Literal motif strings transcribed from the supplied design table.
# All expanded patterns are checked against the composition and length.
SEQUENCES = {(4, 'central_IG_KG_endcaps'): '(KG)8-(VG)17-(IG)34-(VG)17-(KG)8', (4, 'central_IG_KG_adjacent'): '(VG)17-(KG)8-(IG)34-(KG)8-(VG)17', (4, 'KG_embedded_in_IG'): '(VG)17-(IG)9-(KG)8-(IG)16-(KG)8-(IG)9-(VG)17', (4, 'KG_dispersed_in_IG'): '(VG)17-(IG2-KG)16-(IG)2-(VG)17', (4, 'central_mixed_sticker_domain'): '(VG)17-(IG2-KG)8-(IG)2-(KG-IG2)8-(VG)17', (4, 'two_IG_blocks_KG_endcaps'): '(KG)8-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9-(KG)8', (4, 'two_IG_blocks_interfacial_KG'): '(VG)9-(IG)17-(KG)8-(VG)16-(KG)8-(IG)17-(VG)9', (4, 'two_IG_blocks_KG_center'): '(VG)9-(IG)17-(VG)8-(KG)16-(VG)8-(IG)17-(VG)9', (4, 'IG_endcaps_KG_center'): '(IG)17-(VG)17-(KG)16-(VG)17-(IG)17', (4, 'KG_capped_IG_endblocks'): '(KG)8-(IG)17-(VG)34-(IG)17-(KG)8', (4, 'IG_outermost_KG_inside'): '(IG)17-(KG)8-(VG)34-(KG)8-(IG)17', (4, 'three_medium_IG_blocks_KG_endcaps'): '(KG)8-(VG)6-(IG)11-(VG)11-(IG)12-(VG)11-(IG)11-(VG)6-(KG)8', (4, 'three_IG_blocks_interfacial_KG'): '(VG)6-(IG)11-(KG)4-(VG)11-(KG)4-(IG)12-(KG)4-(VG)11-(KG)4-(IG)11-(VG)6', (4, 'four_IG_blocks_KG_endcaps'): '(KG)8-(VG)4-(IG)9-(VG)8-(IG)8-(VG)10-(IG)8-(VG)8-(IG)9-(VG)4-(KG)8', (4, 'KG_one_end_symmetric_IG'): '(KG)16-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9', (4, 'single_IG_block_KG_center'): '(VG)17-(IG)17-(KG)16-(IG)17-(VG)17', (4, 'KG_clustered_in_VG'): '(VG)8-(KG)8-(VG)9-(IG)34-(VG)9-(KG)8-(VG)8', (4, 'KG_dispersed_in_VG'): '(VG2-KG)8-(VG)1-(IG)34-(VG)1-(KG-VG2)8', (4, 'four_IG_blocks_interfacial_KG'): '(VG)4-(IG)9-(KG)4-(VG)8-(IG)8-(KG)4-(VG)10-(KG)4-(IG)8-(VG)8-(KG)4-(IG)9-(VG)4', (5, 'central_IG_KG_endcaps'): '(KG)12-(VG)15-(IG)30-(VG)15-(KG)12', (5, 'central_IG_KG_adjacent'): '(VG)15-(KG)12-(IG)30-(KG)12-(VG)15', (5, 'KG_embedded_in_IG'): '(VG)15-(IG)9-(KG)12-(IG)12-(KG)12-(IG)9-(VG)15', (5, 'KG_dispersed_in_IG'): '(VG)15-(IG-KG)24-(IG)6-(VG)15', (5, 'central_mixed_sticker_domain'): '(VG)15-(IG-KG)12-(IG)6-(KG-IG)12-(VG)15', (5, 'two_IG_blocks_KG_endcaps'): '(KG)12-(VG)8-(IG)15-(VG)14-(IG)15-(VG)8-(KG)12', (5, 'two_IG_blocks_interfacial_KG'): '(VG)8-(IG)15-(KG)12-(VG)14-(KG)12-(IG)15-(VG)8', (5, 'two_IG_blocks_KG_center'): '(VG)8-(IG)15-(VG)7-(KG)24-(VG)7-(IG)15-(VG)8', (5, 'IG_endcaps_KG_center'): '(IG)15-(VG)15-(KG)24-(VG)15-(IG)15', (5, 'KG_capped_IG_endblocks'): '(KG)12-(IG)15-(VG)30-(IG)15-(KG)12', (5, 'IG_outermost_KG_inside'): '(IG)15-(KG)12-(VG)30-(KG)12-(IG)15', (5, 'three_medium_IG_blocks_KG_endcaps'): '(KG)12-(VG)5-(IG)10-(VG)10-(IG)10-(VG)10-(IG)10-(VG)5-(KG)12', (5, 'three_IG_blocks_interfacial_KG'): '(VG)5-(IG)10-(KG)6-(VG)10-(KG)6-(IG)10-(KG)6-(VG)10-(KG)6-(IG)10-(VG)5', (5, 'four_IG_blocks_KG_endcaps'): '(KG)12-(VG)4-(IG)8-(VG)7-(IG)7-(VG)8-(IG)7-(VG)7-(IG)8-(VG)4-(KG)12', (5, 'KG_one_end_symmetric_IG'): '(KG)24-(VG)8-(IG)15-(VG)14-(IG)15-(VG)8', (5, 'single_IG_block_KG_center'): '(VG)15-(IG)15-(KG)24-(IG)15-(VG)15', (5, 'KG_clustered_in_VG'): '(VG)7-(KG)12-(VG)8-(IG)30-(VG)8-(KG)12-(VG)7', (5, 'KG_dispersed_in_VG'): '(VG1-KG)12-(VG)3-(IG)30-(VG)3-(KG-VG1)12', (5, 'four_IG_blocks_interfacial_KG'): '(VG)4-(IG)8-(KG)6-(VG)7-(IG)7-(KG)6-(VG)8-(KG)6-(IG)7-(VG)7-(KG)6-(IG)8-(VG)4'}

# Publication narrative ordered by strength of the matched evidence.
# C is strongest at R5: 2/3/4 IG blocks all show faster entry and slower
# escape with terminal KG than with interfacial KG. B broadens to block position;
# D is a controlled central-IG boundary test; A combines former A and E.
FIGURES = {
 'DR6A_charge_clustering': ('Charge clustering and local environment', [
  'KG_embedded_in_IG', 'single_IG_block_KG_center',
  'KG_dispersed_in_VG', 'KG_dispersed_in_IG', 'central_mixed_sticker_domain']),
 'DR6B_charge_block_position': ('Terminal versus internal charge blocks', [
  'KG_capped_IG_endblocks', 'charged_large_IG_domain',
  'four_IG_blocks_KG_endcaps', 'two_IG_blocks_KG_endcaps',
  'central_IG_KG_endcaps', 'KG_embedded_in_IG',
  'single_IG_block_KG_center', 'two_IG_blocks_KG_center',
  'IG_endcaps_KG_center']),
 'DR6C_domain_fragmentation': ('Hydrophobic domain fragmentation', [
  'two_IG_blocks_KG_endcaps', 'two_IG_blocks_interfacial_KG',
  'three_medium_IG_blocks_KG_endcaps', 'three_IG_blocks_interfacial_KG',
  'four_IG_blocks_KG_endcaps', 'four_IG_blocks_interfacial_KG',
  'single_IG_block_KG_center']),
 'DR6D_compactness': ('Charge architecture and compact-state size', [
  'charged_large_IG_domain', 'KG_embedded_in_IG',
  'single_IG_block_KG_center', 'IG_endcaps_KG_center',
  'KG_dispersed_in_VG', 'KG_clustered_in_VG', 'even_distribution']),
}
SUPPLEMENT_FIGURES = {
 'DR6S_central_IG_boundaries': ('Central IG boundary positioning', [
  'central_IG_KG_endcaps', 'central_IG_KG_adjacent',
  'KG_clustered_in_VG', 'KG_dispersed_in_VG']),
}

# Direction is explicit: each contrast is log2(A/B) at identical composition.
CONTRASTS = {
 'DR6A_charge_clustering': [
  ('IG: dispersed / embedded', 'KG_dispersed_in_IG', 'KG_embedded_in_IG'),
  ('IG: mixed / embedded', 'central_mixed_sticker_domain', 'KG_embedded_in_IG'),
  ('VG: dispersed / clustered', 'KG_dispersed_in_VG', 'KG_clustered_in_VG')],
 'DR6B_charge_block_position': [
  ('IG endblocks capped / central IG capped', 'KG_capped_IG_endblocks', 'central_IG_KG_endcaps'),
  ('2 IG terminal / 2 IG central KG', 'two_IG_blocks_KG_endcaps', 'two_IG_blocks_KG_center')],
 'DR6C_domain_fragmentation': [
  ('2 IG: interfacial / terminal KG', 'two_IG_blocks_interfacial_KG', 'two_IG_blocks_KG_endcaps'),
  ('3 IG: interfacial / terminal KG', 'three_IG_blocks_interfacial_KG', 'three_medium_IG_blocks_KG_endcaps'),
  ('4 IG: interfacial / terminal KG', 'four_IG_blocks_interfacial_KG', 'four_IG_blocks_KG_endcaps')],
}
MAIN_RATIO = {stem: 5 for stem in FIGURES}

METRICS = [
 ('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)',1.,'high'),
 ('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)',1.,'low'),
 ('mean_compact_rg_nm',r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)',1.,'low'),
]
COLORS={'VG':'#4B80CF','IG':'#5C853F','KG':'#E000D5'}
RATIO_COLORS={1:'#1767B1',2:'#20A2B7',3:'#4C9B62',4:'#C18732',5:'#8066AC'}


# Additional R5 definitions supplied directly from the design manifest.
# The random_shuffle ordering is not known and is deliberately not fabricated.
EXTRA_R5 = {
 'even_distribution':'(VG-IG-KG)24-(VG-IG)6',
 'original_Tk8_adapted':'(KG)2-(IG)12-(VG2-KG)5-IG-(VG2-KG)5-IG-((VG-KG)2-IG)4-(VG-KG)2-(IG)12-(KG)2',
 'three_IG_blocks_KG_center':'(VG)5-(IG)10-(VG)10-(IG)10-(KG)24-(IG)10-(VG)15',
 'IG_one_end_KG_opposite':'(IG)30-(VG)30-(KG)24',
 'KG_then_IG_same_end':'(VG)30-(KG)24-(IG)30',
 'IG_then_KG_same_end':'(VG)30-(IG)30-(KG)24',
 'asymmetric_IG_blocks_KG_endcaps':'(KG)12-(IG)20-(VG)15-(IG)10-(VG)15-(KG)12',
 'charged_large_IG_domain':'(IG)20-(KG)24-(VG)15-(IG)10-(VG)15',
 'hydrophobic_gradient':'(KG)24-(VG)10-(IG)2-(VG)8-(IG)4-(VG)6-(IG)6-(VG)4-(IG)8-(VG)2-(IG)10',
 'symmetric_IG_gradient':'(KG)12-(VG)6-(IG)3-(VG)5-(IG)5-(VG)4-(IG)14-(VG)4-(IG)5-(VG)5-(IG)3-(VG)6-(KG)12',
}
SEQUENCES.update({(5,k):v for k,v in EXTRA_R5.items()})

def expand(expression):
    tokens=re.findall(r'VG|IG|KG|\d+|[()]',expression.replace('-','').replace(' ',''))
    pos=0
    def parse():
        nonlocal pos
        result=[]
        while pos<len(tokens) and tokens[pos]!=')':
            if tokens[pos]=='(':
                pos+=1;part=parse()
                if pos>=len(tokens) or tokens[pos]!=')':raise ValueError('Unmatched parentheses: '+expression)
                pos+=1
            else:
                if tokens[pos] not in ('VG','IG','KG'):raise ValueError('Bad token: '+str(tokens[pos]))
                part=[tokens[pos]];pos+=1
            n=1
            if pos<len(tokens) and tokens[pos].isdigit():n=int(tokens[pos]);pos+=1
            result.extend(part*n)
        return result
    result=parse()
    if pos!=len(tokens):raise ValueError('Unexpected closing parenthesis: '+expression)
    return result

def validate_sequences():
    for (ratio,key),pattern in SEQUENCES.items():
        motifs=expand(pattern)
        expected={4:{'VG':34,'IG':34,'KG':16},5:{'VG':30,'IG':30,'KG':24}}[ratio]
        actual={motif:motifs.count(motif) for motif in expected}
        if len(motifs)!=84 or actual!=expected:
            raise ValueError(f'Invalid R{ratio} {key}: {len(motifs)} motifs; {actual} != {expected}')

def load_summary(path, temperatures, ratios):
    d=pd.read_csv(path)
    required={'family','architecture','ratio','temperature','threshold_nm'}|{m[0] for m in METRICS}
    missing=required-set(d.columns)
    if missing:raise ValueError('Missing CSV columns: '+', '.join(sorted(missing)))
    d['family']=d['family'].astype(str).str.strip().str.upper()
    d['architecture']=d.get('architecture_key',d['architecture']).astype(str).str.strip()
    d['temperature']=pd.to_numeric(d.temperature.astype(str).str.replace(r'\s*[Kk]$','',regex=True))
    d['ratio_number']=pd.to_numeric(d.ratio.astype(str).str.extract(r'^[Rr]?(\d+)$',expand=False),errors='coerce')
    d=d[(d.family=='VGIGKG')&d.temperature.isin(temperatures)&d.ratio_number.isin(ratios)].copy()
    if d.empty:raise ValueError('No matching VGIGKG data at requested ratios and temperatures')
    if not np.allclose(pd.to_numeric(d.threshold_nm),4.61):raise ValueError('Expected threshold_nm = 4.61')
    d['architecture']=d.architecture.str.replace(r'^asymmetric_IG\d+_IG\d+_KG_endcaps$','asymmetric_IG_blocks_KG_endcaps',regex=True)
    for metric,_,_,_ in METRICS:d[metric]=pd.to_numeric(d[metric],errors='raise')
    if d[[m[0] for m in METRICS]].isna().any().any():raise ValueError('Missing metric values')
    if 'replicate' not in d.columns:d['replicate']='single'
    keys=['temperature','ratio_number','architecture']
    if d.duplicated(keys+['replicate']).any():raise ValueError('Duplicate replicate entries')
    agg=d.groupby(keys,as_index=False).agg(**{m[0]:(m[0],'mean') for m in METRICS},n_replicates=('replicate','size'))
    return agg

def segments(motifs):
    out=[]
    for m in motifs:
        if out and out[-1][0]==m:out[-1]=(m,out[-1][1]+1)
        else:out.append((m,1))
    return out

def draw_sequence(ax,motifs,y,height=.66):
    x=0
    for name,n in segments(motifs):
        ax.add_patch(Rectangle((x,y-height/2),n,height,facecolor=COLORS[name],edgecolor='white',linewidth=.3))
        x+=n
    ax.set_xlim(0,84)

def plot_one(data,temperature,ratio,stem,title,keys,outdir,dpi):
    table=data[(data.temperature==temperature)&(data.ratio_number==ratio)].set_index('architecture')
    missing=[k for k in keys if k not in table.index]
    if missing:raise ValueError(f'R{ratio} {temperature}K missing architectures: {missing}')
    # Sort separately for each ratio/temperature by increasing escape rate.
    # Stable ties preserve the original comparison order.
    keys=sorted(keys,key=lambda k: float(table.loc[k,'kescape_ns_inv']))
    n=len(keys)
    # Equal-weight mean across ALL available architectures in this ratio and temperature,
    # after averaging replicates within each architecture in load_summary().
    # This mean is fixed across the four design-rule subsets.
    library_means={metric: float(table[metric].mean()) for metric,_,_,_ in METRICS}
    fig=plt.figure(figsize=(18.4,max(6.2,1.18*n+2.25)))
    gs=fig.add_gridspec(1,5,width_ratios=[2.85,3.5,1.95,1.95,2.05],left=.075,right=.985,top=.83,bottom=.14,wspace=.17)
    axes=[fig.add_subplot(gs[0,i]) for i in range(5)]
    label_ax,seq_ax,*metric_axes=axes
    ys=np.arange(n)[::-1]
    label_ax.set_xlim(0,1);label_ax.set_ylim(-.65,n-.35);label_ax.axis('off')
    seq_ax.set_ylim(-.65,n-.35)
    seq_ax.set_yticks([]);seq_ax.set_xticks([0,21,42,63,84]);seq_ax.tick_params(axis='x',labelsize=11,length=2)
    seq_ax.set_xlabel('Pentapeptide motif position',fontsize=15.0)
    for s in seq_ax.spines.values():s.set_visible(False)
    for i,(key,y) in enumerate(zip(keys,ys)):
        motifs=expand(SEQUENCES[(ratio,key)])
        label=key.replace('_',' ')
        label_ax.text(.98,y,label,ha='right',va='center',fontsize=16.0,fontweight='semibold')
        draw_sequence(seq_ax,motifs,y)
        if i%2==0:
            for ax in axes[1:]:ax.axhspan(y-.49,y+.49,color='#F5F6F7',zorder=-5)
    label_ax.set_title('Architecture',fontsize=17.0,loc='right',pad=13,fontweight='bold')
    seq_ax.set_title('Exact 84-motif design',fontsize=17.0,pad=13,fontweight='bold')
    for ax,(metric,metric_title,factor,direction) in zip(metric_axes,METRICS):
        vals=np.array([float(table.loc[k,metric])*factor for k in keys])
        # Dot-and-stem rather than bars, allowing a tight column and common zero-free axes.
        mean_value=library_means[metric]*factor
        # Always include the library mean, even when it falls outside the plotted subset.
        low,high=min(vals.min(),mean_value),max(vals.max(),mean_value)
        span=max(high-low,abs(high)*.03,1e-9)
        x0=max(0,low-.18*span);x1=high+.34*span
        ax.set_xlim(x0,x1)
        ax.axvline(mean_value,color="#7A858F",lw=1.1,ls=(0,(3,3)),alpha=.75,zorder=2)
        for i,(value,y) in enumerate(zip(vals,ys)):
            ax.plot(value,y,'o',ms=9.0,color=RATIO_COLORS[ratio],markeredgecolor='white',markeredgewidth=.6,zorder=4)
            ax.text(value+.025*(x1-x0),y,f'{value:.1f}' if metric=='fraction_compact' else f'{value:.3g}',va='center',fontsize=14.3,fontweight='medium')
        ax.set_ylim(-.65,n-.35)
        ax.set_yticks([])
        ax.set_title(metric_title,fontsize=16.5,pad=13,fontweight='bold')
        ax.grid(axis='x',alpha=.2,linewidth=.6)
        ax.tick_params(axis='x',labelsize=11)
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(color=COLORS[m],label=m) for m in ('VG','IG','KG')],loc='upper left',bbox_to_anchor=(.075,.985),ncol=3,frameon=False,fontsize=15.0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color='#7A858F',lw=1.1,ls=(0,(3,3)),label=f'R{ratio} library mean (all architectures)')],
               loc='upper right',bbox_to_anchor=(.985,.985),frameon=False,fontsize=14.0)
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'DR6_{stem}_R{ratio}_{temperature}K.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)



def plot_supplement_catalogue(data, temperature, ratio, outdir, dpi):
    """Catalogue of sequence-verified architectures for the chosen ratio, plus three metrics."""
    keys = [key for (r, key) in SEQUENCES if r == ratio]
    table = data[(data.temperature == temperature) & (data.ratio_number == ratio)].set_index('architecture')
    missing = [key for key in keys if key not in table.index]
    if missing:
        raise ValueError(f'Supplementary R{ratio}: missing architecture metrics: {missing}')
    keys.sort(key=lambda key: (float(table.loc[key, 'kescape_ns_inv']), key))
    n = len(keys)
    fig = plt.figure(figsize=(20.0, max(12.0, .68*n + 2.5)))
    gs = fig.add_gridspec(1, 5, width_ratios=[3.0, 3.8, 1.95, 1.95, 2.05],
                          left=.075, right=.985, top=.86, bottom=.09, wspace=.19)
    label_ax, seq_ax, *metric_axes = [fig.add_subplot(gs[0, j]) for j in range(5)]
    ys = np.arange(n)[::-1]
    label_ax.set_xlim(0, 1); label_ax.set_ylim(-.65, n-.35); label_ax.axis('off')
    seq_ax.set_ylim(-.65, n-.35); seq_ax.set_yticks([])
    seq_ax.set_xticks([0,21,42,63,84]); seq_ax.set_xlabel('Pentapeptide motif position', fontsize=13.7)
    seq_ax.tick_params(axis='x', labelsize=8, length=2)
    for spine in seq_ax.spines.values(): spine.set_visible(False)
    for i, (key, y) in enumerate(zip(keys, ys)):
        if i % 2 == 0:
            for ax in [seq_ax, *metric_axes]:
                ax.axhspan(y-.48, y+.48, color='#F5F6F7', zorder=-5)
        label_ax.text(.99, y, key.replace('_', ' '), ha='right', va='center', fontsize=14.0, fontweight='semibold')
        draw_sequence(seq_ax, expand(SEQUENCES[(ratio, key)]), y, height=.75)
    label_ax.set_title('Architecture', fontsize=17.0, loc='right', pad=12, fontweight='bold')
    seq_ax.set_title('Exact 84-motif design', fontsize=17.0, pad=12, fontweight='bold')
    for ax, (metric, metric_title, factor, _) in zip(metric_axes, METRICS):
        values = np.array([float(table.loc[key, metric])*factor for key in keys])
        lo, hi = values.min(), values.max()
        span = max(hi-lo, abs(hi)*.03, 1e-9)
        x0 = max(0, lo-.22*span); x1 = hi+.38*span
        ax.set_xlim(x0, x1); ax.set_ylim(-.65, n-.35); ax.set_yticks([])
        ax.set_title(metric_title, fontsize=16.5, pad=12, fontweight='bold')
        ax.grid(axis='x', alpha=.2, linewidth=.6)
        ax.tick_params(axis='x', labelsize=8)
        for side in ('top','right','left'): ax.spines[side].set_visible(False)
        for value, y in zip(values, ys):
            ax.plot(value, y, 'o', ms=7.0, color=RATIO_COLORS[ratio],
                    markeredgecolor='white', markeredgewidth=.5, zorder=4)
            ax.text(value+.025*(x1-x0), y, f'{value:.2f}' if metric=='mean_compact_rg_nm' else f'{value:.3g}', va='center', fontsize=13.5)
    fig.legend(handles=[Patch(color=COLORS[m], label=m) for m in ('VG','IG','KG')],
               loc='upper left', bbox_to_anchor=(.075,.985), ncol=3, frameon=False, fontsize=15.0)
    fig.text(.035, .025, f'{n} sequence-verified R{ratio} architectures, sorted by increasing k_escape. '
             'Compact state: Rg < 4.61 nm. Values are means over available replicate rows. ',
             fontsize=12.2, color='#65717B')
    if ratio == 5:
        fig.text(.035, .043, 'Random shuffle omitted: original simulated motif ordering not supplied.', fontsize=12.2, color='#9A4C62')
    stem = f'DR6_SUPPLEMENT_all_architectures_R{ratio}_{temperature}K'
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'{stem}.{ext}', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    export = []
    for key in keys:
        motifs = expand(SEQUENCES[(ratio,key)])
        row = {'ratio':ratio,'temperature':temperature,'architecture':key,
               'pattern':SEQUENCES[(ratio,key)],
               'KG_blocks':sum(m=='KG' and (i==0 or motifs[i-1]!='KG') for i,m in enumerate(motifs))}
        row.update(table.loc[key,[m[0] for m in METRICS]+['n_replicates']].to_dict())
        export.append(row)
    pd.DataFrame(export).to_csv(outdir/f'{stem}.csv',index=False)
    print(f'Saved {stem} (PDF/SVG/PNG + CSV); {n} architectures')

def plot_cross_ratio(data, temperature, stem, comparisons, ratios, outdir, dpi):
    """Two-panel matched log2 fold change of collapse and escape rates across R1-R5.

    Every point is a ratio of two architecture means at the SAME composition.
    R3 can have missing architecture rows; these are skipped and documented.
    """
    subset=data[data.temperature==temperature]
    lookup={(int(r.ratio_number),r.architecture):r for r in subset.itertuples(index=False)}
    metrics=[('kcollapse_ns_inv',r'$\log_2(k_{collapse,A}/k_{collapse,B})$'),
             ('kescape_ns_inv',r'$\log_2(k_{escape,A}/k_{escape,B})$')]
    fig,axes=plt.subplots(1,2,figsize=(13.7,max(4.6,0.9*len(comparisons)+2.0)),sharey=True)
    rows=[]; unavailable=[]
    for j,(metric,xlabel) in enumerate(metrics):
        ax=axes[j]
        for i,(label,a,b) in enumerate(comparisons):
            for ratio in ratios:
                ra=lookup.get((ratio,a));rb=lookup.get((ratio,b))
                if ra is None or rb is None:
                    if j==0: unavailable.append((ratio,label,a,b))
                    continue
                va=float(getattr(ra,metric));vb=float(getattr(rb,metric))
                if va<=0 or vb<=0:
                    if j==0: unavailable.append((ratio,label,a,b))
                    continue
                value=np.log2(va/vb)
                offset=(ratio-3)*.075
                ax.scatter(value,i+offset,color=RATIO_COLORS[ratio],s=49,edgecolors='white',linewidths=.5,zorder=4)
                rows.append({'figure':stem,'temperature':temperature,'ratio':ratio,
                             'contrast':label,'numerator_A':a,'denominator_B':b,
                             'metric':metric,'A':va,'B':vb,'fold_A_over_B':va/vb,'log2_fold':value})
        ax.axvline(0,color='#596775',lw=.9,ls='--')
        ax.grid(axis='x',alpha=.18)
        ax.set_xlabel(xlabel,fontsize=16.7)
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels([x[0] for x in comparisons],fontsize=13.7)
        ax.invert_yaxis()
        ax.set_title('Compact-state entry' if j==0 else 'Compact-state escape',fontsize=18.2)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.suptitle(f'{stem.replace("_"," ")} | cross-ratio matched contrasts | {temperature} K',
                 x=.055,y=.98,ha='left',fontsize=19.8,fontweight='bold')
    fig.legend(handles=[Patch(color=RATIO_COLORS[r],label=f'R{r}') for r in ratios],
               loc='upper center',bbox_to_anchor=(.53,.96),ncol=len(ratios),frameon=False)
    fig.text(.055,.035,'A / B = numerator / denominator in each row. Positive entry = A collapses faster; '
             'negative escape = A escapes slower. Means of replicates; no uncertainty inferred.',
             fontsize=12.2,color='#65717B')
    fig.subplots_adjust(left=.30,right=.97,top=.84,bottom=.16,wspace=.19)
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'{stem}_cross_ratio_{temperature}K.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows,columns=['figure','temperature','ratio','contrast','numerator_A','denominator_B',
             'metric','A','B','fold_A_over_B','log2_fold']).to_csv(outdir/f'{stem}_cross_ratio_{temperature}K.csv',index=False)
    if unavailable:
        pd.DataFrame(unavailable,columns=['ratio','contrast','numerator_A','denominator_B']).drop_duplicates().to_csv(
            outdir/f'{stem}_cross_ratio_missing_{temperature}K.csv',index=False)
    print(f'Saved {stem} cross-ratio ({len(rows)//2} matched comparisons, {len(unavailable)} unavailable)')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--outdir',type=Path,default=Path('DR6_final_thesis_large_fonts'))
    p.add_argument('--temperatures',type=int,nargs='+',default=[370])
    p.add_argument('--ratios',type=int,nargs='+',default=[4,5],help='Accepted for compatibility; main figures use R5')
    p.add_argument('--supplement-ratio',type=int,default=5,choices=[4,5])
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--cross-ratios',type=int,nargs='+',default=[1,2,3,4,5],choices=[1,2,3,4,5])
    p.add_argument('--no-cross-ratio',action='store_true')
    p.add_argument('--no-supporting',action='store_true',help='Omit matched R4 supplementary comparisons')
    p.add_argument('--include-r4-catalogue',action='store_true')
    args=p.parse_args()
    validate_sequences()
    root=args.outdir
    main_dir=root/'01_main_thesis'
    supp_dir=root/'02_supplementary'
    cross_dir=root/'03_cross_ratio_support'
    for directory in (main_dir,supp_dir,cross_dir):directory.mkdir(parents=True,exist_ok=True)
    ratios=sorted(set([4,5]+([] if args.no_cross_ratio else args.cross_ratios)))
    data=load_summary(args.summary,args.temperatures,ratios)
    manifest=[]
    for temp in args.temperatures:
        for index,(stem,(title,keys)) in enumerate(FIGURES.items(),1):
            ratio=MAIN_RATIO[stem]
            plot_one(data,temp,ratio,stem,title,keys,main_dir,args.dpi)
            manifest.append({'order':index,'type':'main','rule':stem,'ratio':ratio,'temperature':temp,
                             'filename':f'01_main_thesis/DR6_{stem}_R{ratio}_{temp}K.pdf','architecture_count':len(keys)})
            print(f'MAIN {index}/4: {stem}, R{ratio}, {len(keys)} architectures')
            if not args.no_supporting and stem != 'DR6D_compactness':
                other_keys=[k for k in keys if (4,k) in SEQUENCES]
                if len(other_keys)>=2:
                    plot_one(data,temp,4,stem,title,other_keys,supp_dir,args.dpi)
                    manifest.append({'order':index,'type':'matched_R4','rule':stem,'ratio':4,'temperature':temp,
                                     'filename':f'02_supplementary/DR6_{stem}_R4_{temp}K.pdf','architecture_count':len(other_keys)})
        for stem,(title,keys) in SUPPLEMENT_FIGURES.items():
            plot_one(data,temp,4,stem,title,keys,supp_dir,args.dpi)
            manifest.append({'order':5,'type':'supplement_boundary','rule':stem,'ratio':4,'temperature':temp,
                             'filename':f'02_supplementary/DR6_{stem}_R4_{temp}K.pdf','architecture_count':len(keys)})
        plot_supplement_catalogue(data,temp,args.supplement_ratio,supp_dir,args.dpi)
        manifest.append({'order':6,'type':'catalogue','rule':'all_architectures','ratio':args.supplement_ratio,
                         'temperature':temp,'filename':f'02_supplementary/DR6_SUPPLEMENT_all_architectures_R{args.supplement_ratio}_{temp}K.pdf',
                         'architecture_count':sum(r==args.supplement_ratio for r,k in SEQUENCES)})
        if args.include_r4_catalogue and args.supplement_ratio!=4:
            plot_supplement_catalogue(data,temp,4,supp_dir,args.dpi)
        if not args.no_cross_ratio:
            for stem,comparisons in CONTRASTS.items():
                plot_cross_ratio(data,temp,stem,comparisons,args.cross_ratios,cross_dir,args.dpi)
                manifest.append({'order':7,'type':'cross_ratio','rule':stem,'ratio':'R1-R5',
                                 'temperature':temp,'filename':f'03_cross_ratio_support/{stem}_cross_ratio_{temp}K.pdf',
                                 'architecture_count':len(comparisons)})
    pd.DataFrame(manifest).to_csv(root/'figure_manifest.csv',index=False)
    means=(data.groupby(['temperature','ratio_number'],as_index=False)
           .agg(n_architectures=('architecture','nunique'),
                **{metric:(metric,'mean') for metric,_,_,_ in METRICS}))
    means.to_csv(root/'library_means_by_ratio.csv',index=False)
    print('Saved manifest:',root/'figure_manifest.csv')
    print('Output:',root.resolve())
if __name__=='__main__':main()
