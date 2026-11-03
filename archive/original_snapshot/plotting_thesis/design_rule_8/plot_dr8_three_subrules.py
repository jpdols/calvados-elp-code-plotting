#!/usr/bin/env python3
"""Three DR8 thesis subrule figures: sequence + escape/collapse/compact Rg only.

Run after create_dr8_complete_summary.py. Input files:
  combined_summary_mean_std.csv and design_metadata.csv in --indir.
Figures have two panels, each a deliberately selected architecture comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from plot_dr8_complete_story import expand, COLORS

COMPARISONS = {
 'DR8_R1_terminal_KG': [
  ('A  Two IP domains: KG at termini vs interfaces vs centre', [841,842,843]),
  ('B  IP-rich termini: central, outer or inner KG', [844,845,846])],
 'DR8_R2_charge_dispersion': [
  ('A  One central IP domain: localized vs distributed KG', [836,838,839,840]),
  ('B  KG in VG-rich regions: terminal vs clustered vs dispersed', [836,860,861])],
 'DR8_R3_terminal_symmetry': [
  ('A  Symmetric vs asymmetric terminal organization', [841,854,856,858]),
  ('B  Terminal and asymmetric KG arrangements', [850,851,852,853])],
}
METRICS=[
 ('kescape_ns_inv_mean','kescape_ns_inv_std',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)','lower'),
 ('kcollapse_ns_inv_mean','kcollapse_ns_inv_std',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)','higher'),
 ('mean_compact_rg_nm_mean','mean_compact_rg_nm_std',r'Compact-state $R_g$ (nm)','lower'),
]

def figure(d,meta,name,panels,out,formats,dpi,ref):
    groups=[]
    for heading,ids in panels:
        sub=d.set_index('design_id').reindex(ids).reset_index()
        if sub['kescape_ns_inv_mean'].isna().any():
            raise ValueError(f'Missing values for {heading}')
        groups.append((heading,sub))
    fig=plt.figure(figsize=(17,9.0))
    gs=fig.add_gridspec(2,4,width_ratios=[3.7,1.4,1.4,1.4],height_ratios=[len(x[1])+.8 for x in groups],hspace=.52,wspace=.22)
    limits=[]
    for val,sd,_,_ in METRICS:
        values=np.concatenate([g[val].to_numpy(float) for _,g in groups])
        errors=np.concatenate([g[sd].fillna(0).to_numpy(float) for _,g in groups])
        lo=min(np.min(values-errors),float(ref[val]));hi=max(np.max(values+errors),float(ref[val]))
        margin=max((hi-lo)*.14,.015)
        limits.append((lo-margin,hi+margin))
    for row,(heading,sub) in enumerate(groups):
        n=len(sub);y=np.arange(n)
        axes=[fig.add_subplot(gs[row,k]) for k in range(4)]
        for ax in axes:
            ax.set_ylim(n-.55,-.55)
            ax.spines[['top','right']].set_visible(False)
            for i in range(n):
                if i%2==0:ax.axhspan(i-.5,i+.5,color='#f5f6f7',zorder=0)
        ax=axes[0]
        for i,r in enumerate(sub.itertuples()):
            seq=meta.get(int(r.design_id))
            if seq is None:
                ax.add_patch(Rectangle((0,i-.2),84,.4,facecolor='#dddddd',edgecolor='#999999',hatch='///'))
                ax.text(42,i,'Sequence not verified',ha='center',va='center',fontsize=8)
            else:
                start=0
                for j in range(1,len(seq)+1):
                    if j==len(seq) or seq[j]!=seq[start]:
                        ax.add_patch(Rectangle((start,i-.20),j-start,.40,facecolor=COLORS[seq[start]],edgecolor='none'))
                        start=j
        ax.set_xlim(-1,85);ax.set_xticks([0,21,42,63,84]);ax.set_xlabel('Motif position (pentapeptides)',fontsize=9)
        ax.set_yticks(y,[f'{int(r.design_id)}  {str(r.design_name).replace("_"," ")}' for r in sub.itertuples()],fontsize=9)
        ax.tick_params(axis='y',length=0)
        ax.set_title(heading,loc='left',fontsize=10.5,fontweight='bold',pad=12)
        for k,(val,sd,lab,direction) in enumerate(METRICS,1):
            ax=axes[k]
            for i,r in enumerate(sub.itertuples()):
                mean=float(getattr(r,val));err=float(getattr(r,sd)) if pd.notna(getattr(r,sd)) else 0.
                ax.hlines(i,mean-err,mean+err,lw=2.2,color='#aabdc8',zorder=2)
                ax.scatter(mean,i,s=56,color='#c08c18' if int(r.design_id)==835 else '#236f93',edgecolor='white',linewidth=.6,zorder=3)
            ax.axvline(float(ref[val]),ls='--',lw=1.2,color='#c08c18',zorder=1)
            ax.set_xlim(*limits[k-1]);ax.set_yticks(y);ax.tick_params(axis='y',left=False,labelleft=False)
            ax.set_title(lab,fontsize=10.5,pad=12)
            ax.set_xlabel(f'{direction.capitalize()} preferred',fontsize=9)
            ax.grid(axis='x',alpha=.15)
        sub.to_csv(out/f'{name}_panel_{row+1}_values.csv',index=False)
    fig.subplots_adjust(left=.28,right=.985,top=.91,bottom=.15)
    fig.suptitle(name.replace('DR8_','').replace('_',' '),fontsize=15,y=.985)
    fig.legend(handles=[Patch(facecolor=COLORS[m],label={'VG':'VPGVG','IP':'IPAVG','KG':'VPGKG'}[m]) for m in COLORS],loc='lower left',bbox_to_anchor=(.28,.04),ncol=3,frameon=False)
    fig.text(.54,.065,'Points: mean of 3 replicates; whiskers: ±1 replicate SD; dashed line: adapted E_Tk8 (835).',fontsize=9)
    for fmt in formats:fig.savefig(out/f'{name}.{fmt}',bbox_inches='tight',dpi=dpi,facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--indir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--outdir',type=Path,default=None)
    p.add_argument('--temperature',default='370K')
    p.add_argument('--formats',nargs='+',choices=['pdf','png','svg'],default=['pdf','png'])
    p.add_argument('--dpi',type=int,default=220)
    args=p.parse_args();out=args.outdir or args.indir/'DR8_three_subrules';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.indir/'combined_summary_mean_std.csv');d=d[d.temp.astype(str).eq(args.temperature)].copy()
    if len(d)!=30:raise ValueError(f'Expected 30 designs at {args.temperature}; found {len(d)}')
    for m,s,_,_ in METRICS:
        if m not in d or s not in d:raise ValueError(f'Missing {m} or {s}')
    if 835 not in set(d.design_id):raise ValueError('Reference 835 not found')
    meta={};audit=[]
    for r in pd.read_csv(args.indir/'design_metadata.csv').itertuples():
        try:seq=expand(r.motif_pattern)
        except ValueError:seq=None
        verified=bool(seq and len(seq)==84 and all(seq.count(m)==c for m,c in [('VG',38),('IP',38),('KG',8)]))
        meta[int(r.design_id)]=seq if verified else None
        audit.append({'design_id':r.design_id,'schematic_verified':verified})
    pd.DataFrame(audit).to_csv(out/'schematic_audit.csv',index=False)
    ref=d.set_index('design_id').loc[835]
    for name,panels in COMPARISONS.items():figure(d,meta,name,panels,out,args.formats,args.dpi,ref)
    (out/'figure_comparisons.json').write_text(json.dumps(COMPARISONS,indent=2))
    print('Saved three DR8 subrule figures to',out)

if __name__=='__main__':main()
