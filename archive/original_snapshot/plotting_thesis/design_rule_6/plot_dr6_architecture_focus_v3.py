#!/usr/bin/env python
"""Focused DR6 publication plots with exact, hardcoded VG/IG/KG sequences.

Inputs: same per-replicate CSV as the original plotting script.
Outputs: 3 figures per ratio and temperature (PDF/SVG/PNG), sorted by k_escape.
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

FIGURES = {
 '01_charge_position': ('Where should KG be placed?', [
  'central_IG_KG_endcaps','central_IG_KG_adjacent',
  'KG_capped_IG_endblocks','IG_outermost_KG_inside',
  'two_IG_blocks_KG_center','IG_endcaps_KG_center',
  'KG_one_end_symmetric_IG','single_IG_block_KG_center']),
 '02_charge_environment': ('Charge within IG, VG or at IG boundaries', [
  'central_IG_KG_endcaps','central_IG_KG_adjacent',
  'KG_clustered_in_VG','KG_dispersed_in_VG',
  'KG_embedded_in_IG','single_IG_block_KG_center',
  'KG_dispersed_in_IG','central_mixed_sticker_domain']),
 '03_multidomain_crosslink_sites': ('Multiple IG domains: terminal vs interfacial KG', [
  'two_IG_blocks_KG_endcaps','two_IG_blocks_interfacial_KG',
  'three_medium_IG_blocks_KG_endcaps','three_IG_blocks_interfacial_KG',
  'four_IG_blocks_KG_endcaps','four_IG_blocks_interfacial_KG'])
}

METRICS = [
 ('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)',1.,'high'),
 ('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)',1.,'low'),
 ('mean_compact_rg_nm',r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)',1.,'low'),
]
COLORS={'VG':'#4B80CF','IG':'#5C853F','KG':'#E000D5'}
RATIO_COLORS={4:'#C18732',5:'#8066AC'}

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

def draw_sequence(ax,motifs,y,height=.55):
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
    fig=plt.figure(figsize=(12.8,max(4.1,0.88*n+1.5)))
    gs=fig.add_gridspec(1,5,width_ratios=[2.65,2.85,1.5,1.5,1.6],left=.035,right=.985,top=.865,bottom=.105,wspace=.25)
    axes=[fig.add_subplot(gs[0,i]) for i in range(5)]
    label_ax,seq_ax,*metric_axes=axes
    ys=np.arange(n)[::-1]
    label_ax.set_xlim(0,1);label_ax.set_ylim(-.65,n-.35);label_ax.axis('off')
    seq_ax.set_ylim(-.65,n-.35)
    seq_ax.set_yticks([]);seq_ax.set_xticks([0,21,42,63,84]);seq_ax.tick_params(axis='x',labelsize=8,length=2)
    seq_ax.set_xlabel('Pentapeptide motif position',fontsize=9)
    for s in seq_ax.spines.values():s.set_visible(False)
    for i,(key,y) in enumerate(zip(keys,ys)):
        motifs=expand(SEQUENCES[(ratio,key)])
        label=key.replace('_',' ')
        label_ax.text(.99,y,label,ha='right',va='center',fontsize=8.3)
        draw_sequence(seq_ax,motifs,y)
        if i%2==0:
            for ax in axes[1:]:ax.axhspan(y-.49,y+.49,color='#F5F6F7',zorder=-5)
    label_ax.set_title('Architecture',fontsize=10,loc='right',pad=13)
    seq_ax.set_title('Exact 84-motif design',fontsize=10,pad=13)
    for ax,(metric,metric_title,factor,direction) in zip(metric_axes,METRICS):
        vals=np.array([float(table.loc[k,metric])*factor for k in keys])
        # Dot-and-stem rather than bars, allowing a tight column and common zero-free axes.
        low,high=vals.min(),vals.max();span=max(high-low,abs(high)*.03,1e-9)
        x0=max(0,low-.18*span);x1=high+.34*span
        ax.set_xlim(x0,x1)
        for i,(value,y) in enumerate(zip(vals,ys)):
            ax.plot(value,y,'o',ms=6.5,color=RATIO_COLORS[ratio],markeredgecolor='white',markeredgewidth=.6,zorder=4)
            ax.text(value+.035*(x1-x0),y,f'{value:.1f}' if metric=='fraction_compact' else f'{value:.3g}',va='center',fontsize=8)
        ax.set_ylim(-.65,n-.35)
        ax.set_yticks([])
        ax.set_title(metric_title,fontsize=9.5,pad=13)
        ax.grid(axis='x',alpha=.2,linewidth=.6)
        ax.tick_params(axis='x',labelsize=8)
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
    fig.suptitle(f'DR6 | {title} | R{ratio} | {temperature} K',x=.035,y=.975,ha='left',fontsize=13,fontweight='bold')
    fig.legend(handles=[Patch(color=COLORS[m],label=m) for m in ('VG','IG','KG')],loc='upper left',bbox_to_anchor=(.035,.945),ncol=3,frameon=False,fontsize=9)
    fig.text(.035,.035,'Rows sorted by increasing escape rate (lower = longer compact-state persistence). Compact: Rg < 4.61 nm. Values are replicate means.',fontsize=8,color='#65717B')
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'DR6_{stem}_R{ratio}_{temperature}K.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,required=True,help='summary_per_replicate.csv (same schema as previous script)')
    p.add_argument('--outdir',type=Path,default=Path('DR6_focused_plots'))
    p.add_argument('--temperatures',type=int,nargs='+',default=[370])
    p.add_argument('--ratios',type=int,nargs='+',default=[4,5],choices=[4,5])
    p.add_argument('--dpi',type=int,default=300)
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    validate_sequences()
    data=load_summary(args.summary,args.temperatures,args.ratios)
    rows=[]
    for temp in args.temperatures:
        for ratio in args.ratios:
            for stem,(title,keys) in FIGURES.items():
                plot_one(data,temp,ratio,stem,title,keys,args.outdir,args.dpi)
                subset=data[(data.temperature==temp)&(data.ratio_number==ratio)].set_index('architecture')
                for key in keys:
                    motifs=expand(SEQUENCES[(ratio,key)])
                    row={'figure':stem,'ratio':ratio,'temperature':temp,'architecture':key,'pattern':SEQUENCES[(ratio,key)],'KG_blocks':sum(m=='KG' and (i==0 or motifs[i-1]!='KG') for i,m in enumerate(motifs))}
                    row.update(subset.loc[key,[m[0] for m in METRICS]+['n_replicates']].to_dict())
                    rows.append(row)
                print(f'Saved {stem} R{ratio} {temp}K (PDF/SVG/PNG)')
    pd.DataFrame(rows).sort_values(['figure','ratio','temperature','kescape_ns_inv']).to_csv(args.outdir/'selected_architecture_metrics.csv',index=False)
    print('Output:',args.outdir.resolve())
if __name__=='__main__':main()
