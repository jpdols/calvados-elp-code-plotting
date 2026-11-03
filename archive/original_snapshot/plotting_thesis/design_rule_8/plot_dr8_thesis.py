#!/usr/bin/env python3
"""DR8 thesis-ready comparison figures, styled after plot_dr7_thesis.py.
Inputs: combined_summary_mean_std.csv, design_metadata.csv (old DR8 pipeline).
Simulation values are never fabricated. For control 834, one VG and one IP motif
are appended as a user-requested completion of the supplied 82-motif FASTA;
these final two positions are assumed, not independently verified.
"""
import argparse, re, json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

COLORS={'VG':'#4B80CF','IP':'#E7B928','KG':'#D81B8A'}
MOTIFS={'VG':'VPGVG','IP':'IPAVG','KG':'VPGKG'}
SETS={
 'DR8_01_terminal_charge_370K':[841,854,856,847,845,850,857],
 'DR8_02_domain_position_370K':[845,846,844,841,842,843],
 'DR8_03_charge_clustering_370K':[837,838,839,836,860,861],
}
METRICS=[('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)',False),
         ('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)',True),
         ('mean_compact_rg_nm',r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)',False)]

# User-supplied exact FASTA sequences for the random and even-distribution controls.
# These override metadata motif_pattern for designs 833 and 834.
CONTROL_FASTA = {
    833: """IPAVGIPAVGIPAVGIPAVGIPAVGVPGVGVPGVGIPAVGVPGVGVPGVGIPAVGVPGVGIPAVGIPAVGVPGVGVPGVG
IPAVGIPAVGVPGVGVPGKGVPGVGIPAVGIPAVGVPGVGVPGKGIPAVGVPGVGVPGVGIPAVGVPGVGIPAVGVPGVG
VPGVGIPAVGIPAVGVPGVGVPGVGIPAVGVPGVGIPAVGIPAVGVPGVGVPGKGIPAVGIPAVGVPGVGIPAVGVPGVG
IPAVGVPGKGVPGVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGVPGVGVPGVGVPGVGIPAVGIPAVGIPAVG
IPAVGVPGVGIPAVGVPGVGVPGVGVPGVGIPAVGVPGKGIPAVGIPAVGVPGKGVPGVGVPGVGVPGVGVPGKGIPAVG
IPAVGVPGVGVPGKGVPGVG""",
    834: """VPGVGIPAVGVPGVGIPAVGVPGKGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGKGVPGVGIPAVG
VPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGKGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVG
IPAVGVPGVGIPAVGVPGKGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGKGVPGVG
IPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGKGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVG
VPGVGIPAVGVPGKGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGVGIPAVGVPGKGVPGVGIPAVG
VPGVGIPAVGVPGVGIPAVG""",
}
# User-requested completion: append one blue VG and one yellow IP to 834.
# Their positions are assumptions; the original FASTA does not specify them.
CONTROL_FASTA[834] += "VPGVGIPAVG"

def parse_control_fasta(raw):
    seq=''.join(raw.split())
    parts=re.findall(r'VPGVG|IPAVG|VPGKG',seq)
    if ''.join(parts)!=seq:
        raise ValueError('FASTA contains unrecognized residue motif(s)')
    return [{'VPGVG':'VG','IPAVG':'IP','VPGKG':'KG'}[x] for x in parts]

def expand(expr):
    """Expand motif patterns such as (VG)12-(IP2-KG)4 and literal pentapeptides."""
    s=str(expr).strip().replace('VPGVG','VG').replace('IPAVG','IP').replace('VPGKG','KG')
    s=re.sub(r'[\s,;_\-]+','',s)
    tokens=re.findall(r'VG|IP|KG|\d+|[()]',s)
    if ''.join(tokens)!=s:raise ValueError(f'Unrecognized pattern: {expr}')
    pos=0
    def parse():
        nonlocal pos
        out=[]
        while pos<len(tokens) and tokens[pos]!=')':
            t=tokens[pos];pos+=1
            if t=='(':
                part=parse()
                if pos>=len(tokens) or tokens[pos]!=')':raise ValueError('Unbalanced parentheses')
                pos+=1
            elif t in COLORS:part=[t]
            else:raise ValueError(f'Unexpected token {t}')
            n=1
            if pos<len(tokens) and tokens[pos].isdigit():n=int(tokens[pos]);pos+=1
            out+=part*n
        return out
    out=parse()
    if pos!=len(tokens):raise ValueError('Unexpected closing parenthesis')
    return out

def load(indir,temp):
    d=pd.read_csv(indir/'combined_summary_mean_std.csv')
    if 'design_id' not in d.columns:raise ValueError('Expected design_id in summary')
    tcol='temp' if 'temp' in d else 'temperature'
    t=pd.to_numeric(d[tcol].astype(str).str.replace(r'\s*[Kk]$','',regex=True),errors='coerce')
    d=d[t.eq(temp)].copy()
    d.design_id=pd.to_numeric(d.design_id).astype(int)
    if d.design_id.duplicated().any():raise ValueError('Duplicate design IDs')
    if len(d)!=30:raise ValueError(f'Expected 30 DR8 designs at {temp} K, found {len(d)}')
    d=d.set_index('design_id')
    missing=set([835]+sum(SETS.values(),[]))-set(d.index)
    if missing:raise ValueError(f'Missing designs: {sorted(missing)}')
    for key,_,_ in METRICS:
        if key+'_mean' not in d:raise ValueError(f'Missing {key}_mean')
        d[key+'_mean']=pd.to_numeric(d[key+'_mean'],errors='raise')
        if not np.isfinite(d[key+'_mean']).all():raise ValueError(f'Nonfinite {key}_mean')
        if key+'_std' in d:d[key+'_std']=pd.to_numeric(d[key+'_std'],errors='coerce')
    meta=pd.read_csv(indir/'design_metadata.csv')
    if not {'design_id','motif_pattern'}.issubset(meta):raise ValueError('Metadata needs design_id and motif_pattern')
    seqs={};audit=[]
    for r in meta.itertuples(index=False):
        i=int(r.design_id)
        try:
            source='metadata'
            if i in CONTROL_FASTA:
                # The control FASTA is authoritative: metadata contains RANDOM/EVEN
                # labels, which cannot be expanded as repeat expressions.
                seq=parse_control_fasta(CONTROL_FASTA[i])
                source='user FASTA'
            else:
                seq=expand(r.motif_pattern)
            count=Counter(seq)
            expected=Counter({'VG':38,'IP':38,'KG':8})
            if count!=expected:
                raise ValueError(f'Expected VG38/IP38/KG8, got {len(seq)} {dict(count)}')
            seqs[i]=seq
            audit.append({'design_id':i,'valid':True,
                          'partial_sequence':False,'source':source,
                          'length':len(seq),'VG':count['VG'],'IP':count['IP'],'KG':count['KG'],
                          'error':'834: appended VG and IP at positions 83-84 (user-requested assumption)' if i==834 else ''})
        except Exception as e:
            audit.append({'design_id':i,'valid':False,'partial_sequence':False,
                          'source':'metadata / user FASTA','length':None,
                          'VG':None,'IP':None,'KG':None,'error':str(e)})
    # All 30 sequences must be valid for the supplementary overview.
    required=set(d.index)
    bad=required-set(seqs)
    if bad:raise ValueError(f'Cannot draw unverified sequences for {sorted(bad)}. Check motif_pattern in design_metadata.csv.')
    return d,seqs,pd.DataFrame(audit)

def make_plot(d,seqs,ids,stem,out,dpi,temperature,formats):
    n=len(ids)
    overview=n>15
    fig=plt.figure(figsize=(19,max(6.0,(.49 if overview else .91)*n+(2.1 if not overview else 2.9))))
    gs=fig.add_gridspec(1,5,width_ratios=[1.25,4.0,2.1,2.1,2.1],left=.035,right=.99,top=.845,bottom=.12,wspace=.23)
    label,seq,*axs=[fig.add_subplot(gs[0,j]) for j in range(5)]
    ys=np.arange(n)[::-1]
    ref=d.loc[835]
    for ax in (label,seq,*axs):ax.set_ylim(-.65,n-.35)
    label.axis('off')
    seq.set_xlim(0,84);seq.set_yticks([]);seq.set_xticks([0,21,42,63,84])
    seq.set_xlabel('Pentapeptide motif position',fontsize=11)
    for side in seq.spines.values():side.set_visible(False)
    for j,(i,y) in enumerate(zip(ids,ys)):
        if j%2==0:
            for ax in (seq,*axs):ax.axhspan(y-.48,y+.48,color='#F4F5F7',zorder=-4)
        label.text(.97,y,str(i),ha='right',va='center',fontsize=9 if overview else 12,fontweight='bold')
        for x,m in enumerate(seqs[i]):
            seq.add_patch(Rectangle((x,y-(.34 if overview else .29)),1,(.68 if overview else .58),facecolor=COLORS[m],edgecolor='none'))
        if len(seqs[i])<84:
            seq.add_patch(Rectangle((len(seqs[i]),y-(.34 if overview else .29)),84-len(seqs[i]),(.68 if overview else .58),facecolor='#D0D0D0',edgecolor='none',hatch='///'))
    label.set_title('Architecture',fontsize=11 if overview else 13,fontweight='bold',loc='right',pad=13)
    seq.set_title('84-motif design',fontsize=11 if overview else 13,fontweight='bold',pad=13)
    for ax,(metric,heading,higher) in zip(axs,METRICS):
        v=d.loc[ids,metric+'_mean'].astype(float)
        lo=min(v.min(),float(ref[metric+'_mean']))
        hi=max(v.max(),float(ref[metric+'_mean']))
        if metric+'_std' in d:
            sd=d.loc[ids,metric+'_std'].fillna(0).astype(float)
            lo=min(lo,float((v-sd).min()))
            hi=max(hi,float((v+sd).max()))
        spread=max(hi-lo,.025 if metric!='mean_compact_rg_nm' else .006)
        ax.set_xlim(lo-.35*spread,hi+.65*spread)
        ax.axvline(float(ref[metric+'_mean']),ls=(0,(3,3)),lw=1.2,color='#B0783E',zorder=1)
        for i,y in zip(ids,ys):
            r=d.loc[i];mean=float(r[metric+'_mean'])
            if metric+'_std' in d and pd.notna(r[metric+'_std']):
                sd=float(r[metric+'_std'])
                ax.plot([mean-sd,mean+sd],[y,y],color='#A7B0BE',lw=3,alpha=.52,solid_capstyle='round',zorder=2)
            ax.plot(mean,y,'o',ms=6.8,color='#D81B8A',mec='white',mew=.5,zorder=4)
            a,b=ax.get_xlim();offset=-8 if (mean-a)/(b-a)>.76 else 8
            ax.annotate(f'{mean:.3f}',(mean,y),xytext=(offset,0),textcoords='offset points',ha='right' if offset<0 else 'left',va='center',fontsize=8.2 if overview else 10.2)
        ax.set_yticks([])
        ax.grid(axis='x',alpha=.15,zorder=0)
        for side in ('top','right','left'):ax.spines[side].set_visible(False)
        ax.tick_params(axis='x',labelsize=10)
        ax.set_title(heading+'\n'+('Higher is favourable ↑' if higher else 'Lower is favourable ↓'),fontsize=10 if overview else 12,fontweight='bold',pad=13)
    fig.legend(handles=[Patch(color=COLORS[k],label=MOTIFS[k]) for k in ('VG','IP','KG')]+[Line2D([0],[0],color='#B0783E',ls='--',label='Adapted E_Tk8 (835)')],loc='upper center',bbox_to_anchor=(.52,.987),frameon=False,ncol=4,fontsize=11)
    note='Dots: mean of 3 replicates; faint lines: ±1 between-replicate SD.' if all(m+'_std' in d for m,_,_ in METRICS) else 'Dots: replicate means; faint lines: ±1 SD when available.'
    fig.text(.035,.033,note+' Dashed line: adapted E_Tk8 (835).',fontsize=9 if overview else 10,color='#646B73')
    for ext in formats:
        target=out/f'{stem}.{ext}'
        try:
            fig.savefig(target,dpi=dpi,bbox_inches='tight',facecolor='white')
        except OSError as exc:
            raise OSError(f'Could not save {target}. Check whether the file is open/locked, the output folder is writable, or try --formats pdf svg and a new --outdir. Original error: {exc}') from exc
    plt.close(fig)
    d.loc[ids].to_csv(out/f'{stem}_values.csv')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir',type=Path,required=True,help='Folder containing combined_summary_mean_std.csv and design_metadata.csv')
    p.add_argument('--outdir',type=Path,default=Path('DR8_thesis_figures'))
    p.add_argument('--temperature',type=int,default=370)
    p.add_argument('--dpi',type=int,default=240)
    p.add_argument('--formats',nargs='+',choices=('pdf','svg','png'),default=['pdf','svg','png'],help='Output formats; use --formats pdf svg to skip PNG')
    args=p.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    d,seqs,audit=load(args.indir,args.temperature)
    audit.to_csv(args.outdir/'DR8_sequence_audit.csv',index=False)
    all_sets={**SETS, 'DR8_04_all_architectures_supplement_370K': sorted(d.index.tolist())}
    for stem,ids in all_sets.items():
        name=stem.replace('370K',f'{args.temperature}K')
        make_plot(d,seqs,ids,name,args.outdir,args.dpi,args.temperature,args.formats)
        print('Created',name,', '.join(args.formats))
    (args.outdir/'DR8_figure_architectures.json').write_text(json.dumps(all_sets,indent=2))
    print('Output:',args.outdir.resolve())
if __name__=='__main__':main()
