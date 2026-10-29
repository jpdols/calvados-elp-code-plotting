#!/usr/bin/env python
r"""Design Rule 6A: five matched architecture panels for R1--R5 at 370 K.

No simulated results are hardcoded. The 25 exact 84-motif patterns are embedded
below and checked against the R1--R5 motif compositions before plotting.

Usage (Windows Anaconda Prompt, single line):
  python plot_dr6a_all_ratios.py --summary "E:\Master thesis\calvados_tea\plotting_thesis\design_rule_6\DR6_VGIGKG_summary_R1_R5\summary_per_replicate.csv" --outdir "DR6A_all_ratios"

Requires: numpy, pandas, matplotlib. Outputs PDF/SVG/PNG and CSV comparisons.
"""
import argparse
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

COMPOSITIONS = {1:(46,30,8), 2:(38,38,8), 3:(30,46,8), 4:(34,34,16), 5:(30,30,24)}
ARCHITECTURES = [
    'KG_embedded_in_IG',
    'single_IG_block_KG_center',
    'KG_dispersed_in_VG',
    'KG_dispersed_in_IG',
    'central_mixed_sticker_domain',
]
# Exact motif patterns transcribed from the supplied R1--R4 design table
# and verified R5 manifest. VG=VPGVG, IG=VPGIG, KG=VPGKG.
PATTERNS = {(1, 'KG_dispersed_in_IG'): '(VG)23-(IG3-KG)8-(IG)6-(VG)23',
 (1, 'KG_dispersed_in_VG'): '(VG5-KG)4-(VG)3-(IG)30-(VG)3-(KG-VG5)4',
 (1, 'KG_embedded_in_IG'): '(VG)23-(IG)11-(KG)4-(IG)8-(KG)4-(IG)11-(VG)23',
 (1, 'central_mixed_sticker_domain'): '(VG)23-(IG3-KG)4-(IG)6-(KG-IG3)4-(VG)23',
 (1, 'single_IG_block_KG_center'): '(VG)23-(IG)15-(KG)8-(IG)15-(VG)23',
 (2, 'KG_dispersed_in_IG'): '(VG)19-(IG4-KG)8-(IG)6-(VG)19',
 (2, 'KG_dispersed_in_VG'): '(VG4-KG)4-(VG)3-(IG)38-(VG)3-(KG-VG4)4',
 (2, 'KG_embedded_in_IG'): '(VG)19-(IG)15-(KG)4-(IG)8-(KG)4-(IG)15-(VG)19',
 (2, 'central_mixed_sticker_domain'): '(VG)19-(IG4-KG)4-(IG)6-(KG-IG4)4-(VG)19',
 (2, 'single_IG_block_KG_center'): '(VG)19-(IG)19-(KG)8-(IG)19-(VG)19',
 (3, 'KG_dispersed_in_IG'): '(VG)15-(IG5-KG)8-(IG)6-(VG)15',
 (3, 'KG_dispersed_in_VG'): '(VG3-KG)4-(VG)3-(IG)46-(VG)3-(KG-VG3)4',
 (3, 'KG_embedded_in_IG'): '(VG)15-(IG)19-(KG)4-(IG)8-(KG)4-(IG)19-(VG)15',
 (3, 'central_mixed_sticker_domain'): '(VG)15-(IG5-KG)4-(IG)6-(KG-IG5)4-(VG)15',
 (3, 'single_IG_block_KG_center'): '(VG)15-(IG)23-(KG)8-(IG)23-(VG)15',
 (4, 'KG_dispersed_in_IG'): '(VG)17-(IG2-KG)16-(IG)2-(VG)17',
 (4, 'KG_dispersed_in_VG'): '(VG2-KG)8-(VG)1-(IG)34-(VG)1-(KG-VG2)8',
 (4, 'KG_embedded_in_IG'): '(VG)17-(IG)9-(KG)8-(IG)16-(KG)8-(IG)9-(VG)17',
 (4, 'central_mixed_sticker_domain'): '(VG)17-(IG2-KG)8-(IG)2-(KG-IG2)8-(VG)17',
 (4, 'single_IG_block_KG_center'): '(VG)17-(IG)17-(KG)16-(IG)17-(VG)17',
 (5, 'KG_dispersed_in_IG'): '(VG)15-(IG-KG)24-(IG)6-(VG)15',
 (5, 'KG_dispersed_in_VG'): '(VG1-KG)12-(VG)3-(IG)30-(VG)3-(KG-VG1)12',
 (5, 'KG_embedded_in_IG'): '(VG)15-(IG)9-(KG)12-(IG)12-(KG)12-(IG)9-(VG)15',
 (5, 'central_mixed_sticker_domain'): '(VG)15-(IG-KG)12-(IG)6-(KG-IG)12-(VG)15',
 (5, 'single_IG_block_KG_center'): '(VG)15-(IG)15-(KG)24-(IG)15-(VG)15'}
COLORS = {'VG':'#4B80CF','IG':'#5C853F','KG':'#E000D5'}
RATIO_COLORS = {1:'#5169B1',2:'#7356A7',3:'#8967AA',4:'#A15FB4',5:'#7549A9'}
METRICS = [
    ('kcollapse_ns_inv', r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)'),
    ('kescape_ns_inv', r'$k_{\mathrm{escape}}$ (ns$^{-1}$)'),
    ('mean_compact_rg_nm', r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)'),
]

def expand(expression):
    tokens=re.findall(r'VG|IG|KG|\d+|[()]',expression.replace('-','').replace(' ',''))
    pos=0
    def parse():
        nonlocal pos
        result=[]
        while pos<len(tokens) and tokens[pos]!=')':
            if tokens[pos]=='(':
                pos+=1
                part=parse()
                if pos>=len(tokens) or tokens[pos]!=')':
                    raise ValueError('Unmatched parentheses: '+expression)
                pos+=1
            else:
                if tokens[pos] not in ('VG','IG','KG'):
                    raise ValueError('Bad motif token: '+str(tokens[pos]))
                part=[tokens[pos]]
                pos+=1
            n=1
            if pos<len(tokens) and tokens[pos].isdigit():
                n=int(tokens[pos]);pos+=1
            result.extend(part*n)
        return result
    result=parse()
    if pos!=len(tokens):raise ValueError('Unexpected closing parenthesis: '+expression)
    return result

def validate_patterns():
    for (ratio,key),pattern in PATTERNS.items():
        motifs=expand(pattern)
        actual=tuple(motifs.count(m) for m in ('VG','IG','KG'))
        if len(motifs)!=84 or actual!=COMPOSITIONS[ratio]:
            raise ValueError(f'Invalid R{ratio} {key}: {len(motifs)} motifs; {actual}')
    assert len(PATTERNS)==25

def load_data(path,temperature):
    df=pd.read_csv(path)
    required={'family','architecture','ratio','temperature','threshold_nm'}|{k for k,_ in METRICS}
    missing=required-set(df.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    df['family']=df.family.astype(str).str.strip().str.upper()
    df['architecture']=df.get('architecture_key',df['architecture']).astype(str).str.strip()
    df['temperature']=pd.to_numeric(df.temperature.astype(str).str.replace(r'\s*[Kk]$','',regex=True))
    df['ratio_number']=pd.to_numeric(df.ratio.astype(str).str.extract(r'^[Rr]?(\d+)$',expand=False),errors='coerce')
    df=df[(df.family=='VGIGKG')&(df.temperature==temperature)&(df.ratio_number.isin(COMPOSITIONS))].copy()
    if df.empty:raise ValueError(f'No VGIGKG data at {temperature} K')
    if not np.allclose(pd.to_numeric(df.threshold_nm),4.61):
        raise ValueError('Expected threshold_nm = 4.61 for all selected rows')
    for metric,_ in METRICS:df[metric]=pd.to_numeric(df[metric],errors='raise')
    if df[[k for k,_ in METRICS]].isna().any().any():raise ValueError('Missing metric values')
    if 'replicate' not in df:df['replicate']='single'
    if df.duplicated(['temperature','ratio_number','architecture','replicate']).any():
        raise ValueError('Duplicate replicate rows found')
    agg=df.groupby(['ratio_number','architecture'],as_index=False).agg(
        **{metric:(metric,'mean') for metric,_ in METRICS},n_replicates=('replicate','size'))
    for ratio in COMPOSITIONS:
        have=set(agg.loc[agg.ratio_number==ratio,'architecture'])
        missing=[key for key in ARCHITECTURES if key not in have]
        if missing:raise ValueError(f'R{ratio}: missing architectures in summary CSV: {missing}')
    return agg

def draw_sequence(ax,motifs,y):
    start=0
    for motif in motifs:
        ax.add_patch(Rectangle((start,y-.31),1,.62,facecolor=COLORS[motif],edgecolor='none'))
        start+=1
    ax.set_xlim(0,84)

def plot_ratio(data,ratio,temperature,outdir,dpi,shared_ranges=None):
    table=data[data.ratio_number==ratio].set_index('architecture')
    # Keep identical row order for direct comparison across the five supplements.
    keys=ARCHITECTURES
    n=len(keys)
    fig=plt.figure(figsize=(18.8,8.2))
    gs=fig.add_gridspec(1,5,width_ratios=[2.9,3.45,1.85,1.85,1.9],
                        left=.047,right=.988,top=.78,bottom=.17,wspace=.19)
    label_ax,seq_ax,*metric_axes=[fig.add_subplot(gs[0,i]) for i in range(5)]
    ys=np.arange(n)[::-1]
    label_ax.set_xlim(0,1);label_ax.set_ylim(-.6,n-.4);label_ax.axis('off')
    seq_ax.set_ylim(-.6,n-.4)
    seq_ax.set_yticks([]);seq_ax.set_xticks([0,21,42,63,84])
    seq_ax.tick_params(axis='x',labelsize=12,length=2)
    seq_ax.set_xlabel('Pentapeptide motif position',fontsize=14)
    for s in seq_ax.spines.values():s.set_visible(False)
    for i,(key,y) in enumerate(zip(keys,ys)):
        label_ax.text(.985,y,key.replace('_',' '),ha='right',va='center',fontsize=15.8,fontweight='semibold')
        if i%2==0:
            for ax in [seq_ax,*metric_axes]:ax.axhspan(y-.49,y+.49,color='#F4F5F7',zorder=-5)
        draw_sequence(seq_ax,expand(PATTERNS[(ratio,key)]),y)
    label_ax.set_title('Architecture',fontsize=16.5,fontweight='bold',loc='right',pad=12)
    seq_ax.set_title('Exact 84-motif design',fontsize=16.5,fontweight='bold',pad=12)
    means={metric:float(table[metric].mean()) for metric,_ in METRICS}
    for ax,(metric,heading) in zip(metric_axes,METRICS):
        vals=np.array([float(table.loc[k,metric]) for k in keys])
        mean=means[metric]
        if shared_ranges is None:
            low=min(vals.min(),mean);high=max(vals.max(),mean)
            span=max(high-low,.04*abs(high),.03)
            x0=max(0,low-.2*span);x1=high+.4*span
        else:x0,x1=shared_ranges[metric]
        ax.set_xlim(x0,x1)
        ax.axvline(mean,color='#78828C',lw=1.2,ls=(0,(2,3)),alpha=.8,zorder=2)
        for value,y in zip(vals,ys):
            ax.plot(value,y,'o',ms=8.5,color=RATIO_COLORS[ratio],mec='white',mew=.5,zorder=4)
            # Labels stay inside the metric axis, including at the right boundary.
            frac=(value-x0)/(x1-x0)
            if frac>.78:
                ax.annotate(f'{value:.2f}',(value,y),xytext=(-7,0),textcoords='offset points',
                            ha='right',va='center',fontsize=13,fontweight='medium')
            else:
                ax.annotate(f'{value:.2f}',(value,y),xytext=(7,0),textcoords='offset points',
                            ha='left',va='center',fontsize=13,fontweight='medium')
        ax.set_ylim(-.6,n-.4);ax.set_yticks([])
        ax.set_title(heading,fontsize=15.5,fontweight='bold',pad=12)
        ax.grid(axis='x',alpha=.18,lw=.7)
        ax.tick_params(axis='x',labelsize=11)
        for s in ('top','right','left'):ax.spines[s].set_visible(False)
    fig.legend(handles=[Patch(color=COLORS[m],label=m) for m in ('VG','IG','KG')],
               loc='upper left',bbox_to_anchor=(.047,.96),ncol=3,frameon=False,fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color='#78828C',lw=1.2,ls=(0,(2,3)),
                               label=f'R{ratio} library mean (all architectures)')],
               loc='upper right',bbox_to_anchor=(.988,.96),frameon=False,fontsize=13)
    vg,ig,kg=COMPOSITIONS[ratio]
    fig.text(.5,.88,f'R{ratio}: {vg} VG / {ig} IG / {kg} KG motifs  |  {temperature} K',
             ha='center',va='center',fontsize=18,fontweight='bold')
    stem=f'DR6A_charge_clustering_R{ratio}_{temperature}K'
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'{stem}.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return stem

def comparisons(data,outdir):
    rows=[]
    for ratio in COMPOSITIONS:
        tab=data[data.ratio_number==ratio].set_index('architecture')
        for metric,_ in METRICS:
            def val(key):return float(tab.loc[key,metric])
            rows.extend([
                dict(ratio=f'R{ratio}',metric=metric,contrast='dispersed_IG_minus_dispersed_VG',
                     difference=val('KG_dispersed_in_IG')-val('KG_dispersed_in_VG')),
                dict(ratio=f'R{ratio}',metric=metric,contrast='dispersed_IG_minus_embedded_IG',
                     difference=val('KG_dispersed_in_IG')-val('KG_embedded_in_IG')),
                dict(ratio=f'R{ratio}',metric=metric,contrast='mixed_domain_minus_embedded_IG',
                     difference=val('central_mixed_sticker_domain')-val('KG_embedded_in_IG')),
            ])
    pd.DataFrame(rows).to_csv(outdir/'DR6A_cross_ratio_differences.csv',index=False)
    data[data.architecture.isin(ARCHITECTURES)].sort_values(['ratio_number','architecture']).to_csv(
        outdir/'DR6A_architecture_means_R1_R5.csv',index=False)
    # Interpretation checks, not statistical significance tests.
    print('\nConservation check (descriptive; + and - indicate sign):')
    d=pd.DataFrame(rows)
    for metric,contrast in [('kcollapse_ns_inv','dispersed_IG_minus_dispersed_VG'),
                             ('kescape_ns_inv','dispersed_IG_minus_embedded_IG')]:
        z=d[(d.metric==metric)&(d.contrast==contrast)]
        print(metric,contrast,[(r,round(v,3)) for r,v in zip(z.ratio,z.difference)])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,default=Path('DR6A_all_ratios'))
    parser.add_argument('--temperature',type=int,default=370)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--shared-x',action='store_true',help='Use common x limits across all five figures')
    args=parser.parse_args()
    validate_patterns()
    args.outdir.mkdir(parents=True,exist_ok=True)
    data=load_data(args.summary,args.temperature)
    ranges=None
    if args.shared_x:
        ranges={}
        for metric,_ in METRICS:
            a=data[metric].min();b=data[metric].max();span=max(b-a,.03)
            ranges[metric]=(max(0,a-.1*span),b+.25*span)
    for ratio in COMPOSITIONS:
        stem=plot_ratio(data,ratio,args.temperature,args.outdir,args.dpi,ranges)
        print('Saved',stem)
    comparisons(data,args.outdir)
    print('Output directory:',args.outdir.resolve())

if __name__=='__main__':main()
