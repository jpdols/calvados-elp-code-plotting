#!/usr/bin/env python3
"""Thesis-style R4 nonlocal B-B contact separation curves.

Requires separation-resolved contact counts, not trajectory-level summary statistics.
Accepts long CSV (separation + contacts + architecture/design ID), or wide CSV
(separation + one column per architecture). Optional --input or --root discovery.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ORDER = ['Random shuffle','Even','Small blocks','Medium blocks','Large blocks','High-λ center','High-λ endcaps','Diblock']
COLORS = {'Random shuffle':'#bdbdbd','Even':'#5b9bd5','Small blocks':'#70ad47',
          'Medium blocks':'#8064a2','Large blocks':'#ed7d31','High-λ center':'#f04b3e',
          'High-λ endcaps':'#27b0c6','Diblock':'#8c564b'}
# ID mapping follows R4 eight-architecture ordering; explicit labels take precedence.
ID_MAP = {103+i: a for i,a in enumerate(ORDER)}

def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def match_arch(s):
    t=norm(s)
    if 'random' in t or 'shuffle' in t: return 'Random shuffle'
    if 'even' in t or 'alternat' in t: return 'Even'
    if 'small' in t: return 'Small blocks'
    if 'medium' in t: return 'Medium blocks'
    if 'large' in t: return 'Large blocks'
    if 'diblock' in t: return 'Diblock'
    if 'center' in t or 'centre' in t: return 'High-λ center'
    if 'endcap' in t or 'ends' in t: return 'High-λ endcaps'
    m=re.search(r'(?<!\d)(10[3-9]|110)(?!\d)',str(s))
    return ID_MAP.get(int(m.group(1))) if m else None

def find(df,opts):
    d={norm(c):c for c in df.columns}
    for c in opts:
        if norm(c) in d:return d[norm(c)]
    return None

def read_data(path, min_sep, max_sep):
    raw=pd.read_csv(path)
    sep=find(raw,['sequence_separation','sequence_separation_aa','separation','sep','delta_seq','delta_s','abs_i_minus_j','abs_ij','distance_aa','sequence_distance','s'])
    if sep is None: raise ValueError('Missing sequence-separation column. Columns: '+str(list(raw.columns)))
    temp=find(raw,['temperature','temp','temperature_K'])
    if temp:
        nums=pd.to_numeric(raw[temp].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0],errors='coerce')
        raw=raw.loc[nums.eq(370)].copy()
    ratio=find(raw,['ratio','composition_ratio'])
    if ratio:raw=raw.loc[raw[ratio].astype(str).str.upper().str.strip().eq('R4')].copy()
    family=find(raw,['family','chemistry','group'])
    if family:
        # Default to VGIG if present; use --family all to pool others.
        pass
    arch=find(raw,['architecture','architecture_type','architecture_plot','arch','name','sequence','sequence_id','id','design_id'])
    val=find(raw,['contacts_per_selected_frame','contacts_per_frame','mean_contacts_per_frame','mean_contact_hits_per_selected_frame',
                  'mean_nonlocal_BB_contacts_per_selected_frame','nonlocal_BB_contacts_per_selected_frame',
                  'contact_count','contacts','count','hits','n_contacts','contact_hits','frequency'])
    if arch and val:
        d=raw[[sep,arch,val]].copy();d.columns=['sep','arch','value']
        d['arch']=d['arch'].map(match_arch)
    else:
        # Wide table: sequence separation, plus architecture-named columns.
        cols={c:match_arch(c) for c in raw.columns if c!=sep}
        cols={c:a for c,a in cols.items() if a is not None}
        if not cols: raise ValueError('No architecture columns or (architecture, contact-count) pair found. Columns: '+str(list(raw.columns)))
        d=raw[[sep]+list(cols)].melt(id_vars=sep,var_name='arch',value_name='value').rename(columns={sep:'sep'})
        d['arch']=d['arch'].map(cols)
    d['sep']=pd.to_numeric(d['sep'],errors='coerce');d['value']=pd.to_numeric(d['value'],errors='coerce')
    d=d.dropna(subset=['sep','arch','value']);d=d[d['arch'].isin(ORDER)&d['sep'].between(min_sep,max_sep)]
    if d.empty:raise ValueError('No valid separation-resolved rows after filtering.')
    # If multiple rows per arch/sep are replicates, average; if they are raw hits,
    # user should aggregate them into contacts/frame before plotting.
    return d.groupby(['arch','sep'],as_index=False)['value'].mean()

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':19,
                         'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':15,
                         'axes.linewidth':1.5,'pdf.fonttype':42,'ps.fonttype':42,
                         'savefig.facecolor':'white'})

def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=350,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)

def make_plot(df,out,kind,zoom_start=50):
    fig,ax=plt.subplots(figsize=(13.8,8.0))
    for a in ORDER:
        g=df[df.arch==a].sort_values('sep')
        if g.empty:continue
        x=g.sep.to_numpy();y=g.value.to_numpy(dtype=float)
        if kind=='normalized':
            total=y.sum();y=100*y/total if total else np.zeros_like(y)
        elif kind=='cumulative':
            total=y.sum();y=100*np.cumsum(y)/total if total else np.zeros_like(y)
        elif kind=='longrange':
            mask=x>=zoom_start;x=x[mask];y=y[mask]
        ax.plot(x,y,color=COLORS[a],lw=3.5,label=a,solid_capstyle='round')
    ax.set_xlabel(r'Sequence separation, $|i-j|$ (amino acids)',labelpad=12)
    if kind=='raw':ax.set_ylabel('Nonlocal B–B contacts per selected frame')
    if kind=='normalized':ax.set_ylabel('Fraction of B–B contacts (%)')
    if kind=='cumulative':ax.set_ylabel('Cumulative B–B contacts (%)');ax.set_ylim(-3,104)
    if kind=='longrange':ax.set_ylabel('Nonlocal B–B contacts per selected frame')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=1.4,length=6,pad=7)
    ax.grid(axis='y',alpha=.12)
    if kind=='longrange':ax.set_xlim(left=zoom_start)
    ax.legend(loc='upper right',frameon=False,ncol=2,fontsize=15,columnspacing=1.7,handlelength=2.6)
    fig.subplots_adjust(left=.12,right=.975,bottom=.14,top=.97)
    save(fig,out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,help='CSV with separation-resolved contacts')
    p.add_argument('--root',type=Path,help='Folder to search for a separation-resolved CSV')
    p.add_argument('--outdir',type=Path,default=Path('DR3_R4_BB_separation_thesis'))
    p.add_argument('--min-separation',type=int,default=4)
    p.add_argument('--max-separation',type=int,default=420)
    p.add_argument('--zoom-start',type=int,default=50)
    args=p.parse_args()
    if args.input:path=args.input
    elif args.root:
        candidates=sorted({f for pattern in ['*separation*.csv','*contact*distance*.csv','*BB*contacts*.csv'] for f in args.root.rglob(pattern)})
        if not candidates:raise FileNotFoundError('No separation-resolved contact CSV found in '+str(args.root))
        if len(candidates)>1:raise ValueError('Multiple candidate CSV files. Specify --input:\n'+'\n'.join(map(str,candidates)))
        path=candidates[0]
    else:p.error('Provide --input or --root')
    print('Reading:',path)
    df=read_data(path,args.min_separation,args.max_separation)
    args.outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.outdir/'DR3_R4_separation_curves_processed.csv',index=False)
    for kind,name in [('raw','01_raw_contacts'),('normalized','02_normalized_100_percent'),
                      ('longrange','03_long_range_zoom'),('cumulative','04_cumulative_100_percent')]:
        make_plot(df,args.outdir/name,kind,args.zoom_start)
        print('Saved:',name)
    print('Done:',args.outdir)
if __name__=='__main__':main()
