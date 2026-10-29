#!/usr/bin/env python3
"""Two-panel DR3 B-B contact plot using architecture-specific interpatch distance cutoffs.

For VGIG R4 (42 VG and 42 IG pentapeptides), the nearest residues in
DIFFERENT IG patches have minimum sequence separations:
Even=6; small IG3/VG3=16; medium IG7/VG7=36;
large IG14/VG14=71; IG21-VG42-IG21 endcaps=211 residues.

These thresholds also exceed the maximum possible within-IG-patch
separation (4, 14, 34, 69, 104, respectively), so for the stated
idealized architectures a separation threshold classifies interpatch
contacts exactly at residue level. However, the source is a binned,
pooled histogram, so the boundary is approximated using bin centers.
Only apply these thresholds to the stated VGIG R4 sequences. Do not
interpret this as a chemistry-independent or residue-resolved analysis.
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
MIN_INTERPATCH = {'Even':6,'Small blocks':16,'Medium blocks':36,
                  'Large blocks':71,'Higher λ ends':211}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--height',type=float,default=15.0)
    args=p.parse_args()
    df=pd.read_csv(args.input)
    required={'ratio','architecture','sequence_separation_bin_center_aa','contacts_per_selected_frame'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(df.columns))}')
    df=df[df.ratio.astype(str).eq('R4')].copy()
    df['architecture']=df['architecture'].replace(ALIASES)
    df['s']=pd.to_numeric(df['sequence_separation_bin_center_aa'])
    df['y']=pd.to_numeric(df['contacts_per_selected_frame'])
    for arch in ARCH:
        if arch not in set(df.architecture): raise ValueError(f'Missing architecture {arch}')
    # A bin is included if its center meets the minimum interpatch distance.
    # The binning approximation is explicit and is not a new trajectory calculation.
    df['interpatch_proxy']=np.nan
    for arch,threshold in MIN_INTERPATCH.items():
        m=df.architecture.eq(arch)
        df.loc[m,'interpatch_proxy']=np.where(df.loc[m,'s']>=threshold,df.loc[m,'y'],0.0)
    # Patch sizes retained for reporting possible pairs only.
    # All architectures have 42 B motifs; the block sizes determine the number of pairs.
    patch_motifs = {'Even':[1]*42, 'Small blocks':[3]*14,
                    'Medium blocks':[7]*6, 'Large blocks':[14]*3,
                    'Higher λ ends':[21,21]}
    eligible_pairs = {arch: 25 * sum(a*b for i,a in enumerate(sizes) for b in sizes[i+1:])
                      for arch,sizes in patch_motifs.items()}
    args.outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.outdir/'DR3_R4_BB_interpatch_thresholded_bins.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,'axes.labelsize':24,
                         'xtick.labelsize':20,'ytick.labelsize':20,'legend.fontsize':18,
                         'axes.linewidth':1.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(14,args.height),sharex=True)
    for ax,keys,column,letter in [(axes[0],ARCH,'y','A'),(axes[1],list(MIN_INTERPATCH),'interpatch_proxy','B')]:
        for arch in keys:
            d=df[df.architecture.eq(arch)].sort_values('s')
            ax.plot(d.s,d[column],lw=3.5,color=COLORS[arch],label=DISPLAY.get(arch,arch))
        ax.set_xlim(0,420)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(axis='y',alpha=.16)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylabel('Interpatch B–B contacts / compact frame' if letter == 'B' else 'B–B contacts / compact frame',labelpad=12)
        ax.text(0.0,1.025,letter,transform=ax.transAxes,fontsize=30,fontweight='bold',ha='left',va='bottom',clip_on=False)
        ax.legend(loc='upper right',frameon=False,ncol=2,handlelength=2.4,fontsize=18)
    axes[1].xaxis.set_major_locator(MultipleLocator(50))
    axes[1].set_xlabel(r'Sequence separation, $|i-j|$ (residues)',labelpad=10)
    fig.subplots_adjust(left=.24,right=.985,top=.975,bottom=.075,hspace=.27)
    stem=args.outdir/'DR3_R4_BB_all_vs_interpatch_thresholds'
    for ext in ('pdf','png','svg'):
        fig.savefig(f'{stem}.{ext}',dpi=args.dpi,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)
    print('Architecture-specific minimum interpatch separations (residues):')
    for arch,t in MIN_INTERPATCH.items():
        d=df[df.architecture.eq(arch)]
        original=d.y.sum()
        remaining=d.interpatch_proxy.sum()
        print(f'  {arch:20s}: |i-j| >= {t:3d}; pairs={eligible_pairs[arch]:,} (scaled divisor={eligible_pairs[arch]/10000:.4f}); retained {remaining:.2f}/{original:.2f} ({100*remaining/original:.1f}%)')
    print('Saved',stem)

if __name__=='__main__':main()
