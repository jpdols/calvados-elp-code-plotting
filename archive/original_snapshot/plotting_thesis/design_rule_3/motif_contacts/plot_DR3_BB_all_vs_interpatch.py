#!/usr/bin/env python3
"""DR3 R4: original nonlocal B-B contacts vs strictly interpatch B-B contacts.

Uses contact-level output of analyse_DR3_nonlocal_BB_contacts_R4_triplicates.py.
The aggregate histogram CSV alone cannot distinguish intra- vs interpatch.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

ARCH = ['Random shuffle','Even','Small blocks','Medium blocks','Large blocks','Higher λ center','Higher λ ends','Diblock']
COLORS = {'Random shuffle':'#BDBDBD','Even':'#5B9BD5','Small blocks':'#70AD47',
          'Medium blocks':'#8064A2','Large blocks':'#ED7D31',
          'Higher λ center':'#E74C3C','Higher λ ends':'#2CB1C9','Diblock':'#8C564B'}
ALIASES = {'Higher λ endcaps':'Higher λ ends','High-λ endcaps':'Higher λ ends',
           'High-λ center':'Higher λ center','High-λ centre':'Higher λ center',
           'Higher-lambda endcaps':'Higher λ ends','Higher-lambda center':'Higher λ center'}
DISPLAY = {'Higher λ center':'High-λ center','Higher λ ends':'High-λ endcaps'}
PANEL_B = ['Even','Small blocks','Medium blocks','Large blocks','Higher λ ends']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir',type=Path,required=True,help='Directory with original trajectory analysis CSV outputs')
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--bin-width',type=int,default=5)
    p.add_argument('--dpi',type=int,default=400)
    p.add_argument('--height',type=float,default=13.5)
    args=p.parse_args()
    hits=args.indir/'nonlocal_BB_residue_contact_hits.csv.gz'
    summary=args.indir/'nonlocal_BB_trajectory_summary.csv'
    patches=args.indir/'nonlocal_BB_patch_definitions.csv'
    if not hits.exists(): raise FileNotFoundError(f'Missing contact-level file: {hits}\nThe pooled histogram CSV cannot classify interpatch contacts.')
    if not summary.exists(): raise FileNotFoundError(f'Missing selected-frame denominators: {summary}')
    s=pd.read_csv(summary)
    s['architecture']=s['architecture'].replace(ALIASES)
    s=s[s['ratio'].astype(str).eq('R4')]
    frames=s.groupby('architecture')['selected_frames'].sum().to_dict()
    if patches.exists():
        patch_df=pd.read_csv(patches)
        patch_df['architecture']=patch_df['architecture'].replace(ALIASES)
        patch_counts=patch_df.groupby(['architecture','id','replicate'])['patch_id'].nunique()
        print('B-patch counts by architecture (unique counts across trajectories):')
        for arch in ARCH:
            vals=patch_counts.reset_index().query('architecture == @arch')['patch_id'].unique()
            if len(vals): print(f'  {arch}: {sorted(vals.tolist())}')
    else:
        print('NOTE: patch definitions CSV absent; classifying using patch_i != patch_j in contact hits.')
    edges=np.arange(0,421+args.bin_width,args.bin_width)
    centers=(edges[:-1]+edges[1:])/2
    totals={a:np.zeros(len(centers),dtype=np.int64) for a in ARCH}
    inter={a:np.zeros(len(centers),dtype=np.int64) for a in PANEL_B}
    nrows=0
    for chunk in pd.read_csv(hits,chunksize=300000,usecols=['ratio','architecture','sequence_separation_aa','patch_i','patch_j']):
        chunk=chunk[chunk['ratio'].astype(str).eq('R4')].copy()
        chunk['architecture']=chunk['architecture'].replace(ALIASES)
        nrows+=len(chunk)
        for a,g in chunk.groupby('architecture'):
            if a in totals:
                totals[a]+=np.histogram(g['sequence_separation_aa'],bins=edges)[0]
                if a in inter:
                    diff=g.loc[g['patch_i']!=g['patch_j'],'sequence_separation_aa']
                    inter[a]+=np.histogram(diff,bins=edges)[0]
    if nrows==0: raise ValueError('No R4 contact hits found.')
    print(f'Loaded {nrows:,} contact observations.')
    rows=[]
    for a in ARCH:
        denom=int(frames.get(a,0))
        if not denom: raise ValueError(f'Missing selected-frame denominator for {a}')
        for k,x in enumerate(centers):
            rows.append({'architecture':a,'sequence_separation_bin_center_aa':x,
                         'all_contact_count':int(totals[a][k]),
                         'all_contacts_per_selected_frame':totals[a][k]/denom,
                         'interpatch_contact_count':int(inter[a][k]) if a in inter else np.nan,
                         'interpatch_contacts_per_selected_frame':inter[a][k]/denom if a in inter else np.nan,
                         'total_selected_frames':denom})
    args.outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.outdir/'DR3_R4_BB_all_and_interpatch_by_separation.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.labelsize':22,
                         'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':17,
                         'axes.linewidth':1.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,1,figsize=(13.5,args.height),sharex=True)
    for ax,keys,source,letter in [(axs[0],ARCH,totals,'A'),(axs[1],PANEL_B,inter,'B')]:
        for a in keys:
            ax.plot(centers,source[a]/frames[a],lw=3.2,color=COLORS[a],label=DISPLAY.get(a,a))
        ax.set_xlim(0,420)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.grid(axis='y',alpha=.15,lw=.8)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('B–B contacts per selected compact frame',labelpad=14)
        ax.text(-.065,1.01,letter,transform=ax.transAxes,fontsize=28,fontweight='bold',va='bottom')
        ax.legend(loc='upper right',frameon=False,ncol=2,columnspacing=1.0,handlelength=2.4)
    axs[1].set_xlabel(r'Sequence separation, $|i-j|$ (residues)',labelpad=12)
    axs[1].xaxis.set_major_locator(MultipleLocator(50))
    fig.subplots_adjust(left=.17,right=.98,top=.96,bottom=.09,hspace=.25)
    stem=args.outdir/'DR3_R4_BB_all_vs_interpatch_two_panel'
    for ext in ('pdf','png','svg'):
        fig.savefig(f'{stem}.{ext}',dpi=args.dpi,bbox_inches='tight',pad_inches=.20)
    plt.close(fig)
    print('Saved:',stem)
    for a in PANEL_B:
        print(f'{a}: {inter[a].sum():,} interpatch hits / {totals[a].sum():,} total ({100*inter[a].sum()/max(totals[a].sum(),1):.1f}%)')

if __name__=='__main__': main()
