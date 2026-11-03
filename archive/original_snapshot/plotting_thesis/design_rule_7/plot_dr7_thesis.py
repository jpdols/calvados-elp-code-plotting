#!/usr/bin/env python3
r"""DR7 thesis figures styled after DR6. Input: summary_across_replicates.csv.
Generates 5 figures (PDF/SVG/PNG), numerical selection tables and sequence audit.
Never invents simulation metrics. Ranking uses mean rank of collapse (descending)
and escape (ascending), excluding original 863 from the top-six selection.
"""
import argparse, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D

COLORS={'VG':'#4B80CF','IG':'#5C853F','IP':'#E7B928'}
METRICS=[('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)',True),
 ('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)',False),
 ('mean_compact_rg_nm',r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)',False)]

DEFINITIONS={863: '(IP)12-(VG2-IG-VG2)12-(IP)12', 864: 'RANDOM', 865: '(IP-VG2-IG-VG2-IP)12', 866: '(IP)24-(VG)48-(IG)12', 867: '(IP)24-(IG)12-(VG)48', 868: '(VG)48-(IP)24-(IG)12', 869: '(IP)12-(VG)24-(IG)12-(VG)24-(IP)12', 870: '(IP)12-(IG)6-(VG)48-(IG)6-(IP)12', 871: '(IP)12-(VG)18-(IG)6-(VG)12-(IG)6-(VG)18-(IP)12', 872: '(IP)12-(VG)12-(IG)4-(VG)12-(IG)4-(VG)12-(IG)4-(VG)12-(IP)12', 873: '(IP)12-(VG)6-(IG)3-(VG)12-(IG)3-(VG)12-(IG)3-(VG)12-(IG)3-(VG)6-(IP)12', 874: '(IP)12-(VG4-IG)12-(IP)12', 875: '(IP)6-(VG2-IG-VG2)4-(IP)6-(VG2-IG-VG2)4-(IP)6-(VG2-IG-VG2)4-(IP)6', 876: '(IP)8-(VG2-IG-VG2)6-(IP)8-(VG2-IG-VG2)6-(IP)8', 877: '(IP4-(VG2-IG-VG2)2)6', 878: '(VG4-IG)6-(IP)24-(IG-VG4)6', 879: '(VG)24-(IG)6-(IP)24-(IG)6-(VG)24', 880: '(IG)6-(VG)24-(IP)24-(VG)24-(IG)6', 881: '(VG)12-(IP)12-(VG)12-(IG)12-(VG)12-(IP)12-(VG)12', 882: '(VG)24-(IP)12-(IG)12-(IP)12-(VG)24', 883: '(VG-IP)12-(VG)12-(IG)12-(VG)12-(IP-VG)12', 884: '(VG)24-(IP2-IG)12-(VG)24', 885: '(VG4-IG-IP2)12', 886: '(IP)24-(VG4-IG)12', 887: '(IP)12-(VG)16-(IG)4-(VG)8-(IG)4-(VG)8-(IG)4-(VG)16-(IP)12', 888: '(IP)6-(VG)12-(IG)3-(VG)12-(IP)6-(VG)12-(IG)6-(VG)12-(IP)6-(IG)3-(IP)6', 889: '(VG)10-(IP)8-(VG)9-(IG)4-(VG)10-(IP)8-(VG)9-(IG)4-(VG)10-(IP)8-(IG)4', 890: '(IP2-VG4-IG)12', 891: '(IP)12-(VG3-IG)6-(VG)12-(IG-VG3)6-(IP)12', 892: '(IP)4-(VG)8-(IG)2-(VG)8-(IP)8-(VG)8-(IG)8-(VG)8-(IP)8-(VG)8-(IG)2-(VG)8-(IP)4', 893: '(VG)24-(IP2-IG)6-(IG-IP2)6-(VG)24', 894: '(VG4-IG)6-(IP)12-(IG-VG4)6-(IP)12', 895: '((VG)12-(IP)6-(IG)3)4', 896: '((VG)16-(IP)8-(IG)4)3', 897: '(IP2-VG2-IG-VG2)12', 898: '(IP)12-(IG)6-(VG)24-(IG)6-(IP)12-(VG)24', 899: '(IP)12-(VG)12-(IG)6-(VG)24-(IG)6-(VG)12-(IP)12', 900: '(VG)12-(IP)12-(IG)6-(VG)24-(IG)6-(IP)12-(VG)12', 901: '(VG)12-(IG)6-(IP)12-(VG)24-(IP)12-(IG)6-(VG)12', 902: '(IP4-IG2-VG8)6', 903: '(IP)12-(VG)18-(IG)2-(VG)12-(IG)4-(VG)8-(IG)6-(VG)10-(IP)12', 904: '(VG)8-(IP)4-(VG)8-(IP)8-(VG)8-(IG)12-(VG)8-(IP)8-(VG)8-(IP)4-(VG)8', 905: '(IP-VG-IP-VG-IG-VG2)12', 906: '(IP6-IG3-VG12)4'}

# Exact 864 sequence provided by the user, parsed in pentapeptide units.
RANDOM_864=(
'IPAVGIPAVGVPGVGIPAVGVPGVGVPGIGVPGVGVPGVGVPGVGVPGIGVPGVGVPGVGIPAVGIPAVGVPGVGVPGVG'
'VPGVGVPGVGVPGVGVPGVGVPGVGVPGVGVPGIGIPAVGVPGVGIPAVGIPAVGVPGVGVPGVGVPGVGVPGVGVPGVG'
'VPGVGVPGVGVPGVGVPGIGVPGVGVPGVGVPGIGVPGVGIPAVGVPGVGIPAVGIPAVGVPGVGVPGIGIPAVGIPAVG'
'VPGIGVPGVGVPGVGVPGVGIPAVGIPAVGVPGIGVPGVGVPGIGVPGVGIPAVGIPAVGVPGVGVPGVGVPGVGVPGVG'
'IPAVGVPGIGIPAVGVPGVGIPAVGIPAVGVPGVGVPGVGVPGVGVPGVGVPGIGVPGVGIPAVGVPGVGVPGIGVPGVG'
'VPGVGIPAVGIPAVGVPGVG')

def expand(expr):
 tokens=re.findall(r'VG|IG|IP|\d+|[()]',expr)
 if ''.join(tokens)!=re.sub(r'[-\s]','',expr):raise ValueError('Unparsed motif expression '+expr)
 p=0
 def parse():
  nonlocal p
  out=[]
  while p<len(tokens) and tokens[p]!=')':
   token=tokens[p];p+=1
   if token=='(':
    part=parse()
    if p>=len(tokens) or tokens[p]!=')':raise ValueError('Unbalanced expression')
    p+=1
   elif token in COLORS:part=[token]
   else:raise ValueError('Bad token '+token)
   n=1
   if p<len(tokens) and tokens[p].isdigit():n=int(tokens[p]);p+=1
   out+=part*n
  return out
 result=parse()
 if p!=len(tokens):raise ValueError('Extra closing parenthesis')
 return result

def sequences():
 seqs={}
 for i,expr in DEFINITIONS.items():
  if i==864:
   pieces=re.findall(r'IPAVG|VPGIG|VPGVG',RANDOM_864)
   if ''.join(pieces)!=RANDOM_864:raise ValueError('Random 864 sequence cannot be fully parsed')
   seq=[{'IPAVG':'IP','VPGIG':'IG','VPGVG':'VG'}[p] for p in pieces]
  else:seq=expand(expr)
  counts=Counter(seq)
  if len(seq)!=84 or any(counts[k]!=v for k,v in [('VG',48),('IG',12),('IP',24)]):
   raise ValueError(f'Architecture {i} invalid: {len(seq)} {counts}. Fix definition before plotting.')
  seqs[i]=seq
 return seqs

def read_summary(path,temp):
 d=pd.read_csv(path)
 required=['architecture_id','temperature']+[m+'_mean' for m,_,_ in METRICS]
 missing=set(required)-set(d.columns)
 if missing:raise ValueError(f'Missing columns: {missing}')
 d['architecture_id']=pd.to_numeric(d.architecture_id,errors='raise').astype(int)
 d['temperature']=pd.to_numeric(d.temperature.astype(str).str.replace(r'\s*[Kk]$','',regex=True),errors='coerce')
 d=d[d.temperature.eq(temp)].copy()
 if d.architecture_id.duplicated().any():raise ValueError('Duplicate architecture IDs')
 if set(d.architecture_id)!=set(DEFINITIONS):raise ValueError(f'Expected 863-906, missing {set(DEFINITIONS)-set(d.architecture_id)}, unexpected {set(d.architecture_id)-set(DEFINITIONS)}')
 for m,_,_ in METRICS:
  for suffix in ('mean','std','min','max'):
   col=m+'_'+suffix
   if col in d:d[col]=pd.to_numeric(d[col],errors='coerce')
  if not np.isfinite(d[m+'_mean'].to_numpy(float)).all():raise ValueError('Missing metric '+m)
 if 'threshold_nm' in d and not np.allclose(pd.to_numeric(d.threshold_nm),4.61):raise ValueError('Expected 4.61 nm cutoff')
 return d.set_index('architecture_id').sort_index()

def read_replicates(path,temp):
 """Optional individual replicate measurements; returns MultiIndex (architecture_id, replicate)."""
 if path is None:return None
 d=pd.read_csv(path)
 required={'architecture_id','temperature','replicate'}|{m for m,_,_ in METRICS}
 missing=required-set(d.columns)
 if missing:raise ValueError(f'Replicate CSV missing columns: {sorted(missing)}')
 d['architecture_id']=pd.to_numeric(d.architecture_id,errors='raise').astype(int)
 d['temperature']=pd.to_numeric(d.temperature.astype(str).str.replace(r'\s*[Kk]$','',regex=True),errors='coerce')
 d=d[d.temperature.eq(temp)].copy()
 for m,_,_ in METRICS:d[m]=pd.to_numeric(d[m],errors='coerce')
 if d.duplicated(['architecture_id','replicate']).any():
  raise ValueError('Duplicate replicate/architecture pairs')
 return d.set_index(['architecture_id','replicate']).sort_index()

def save(fig,name,out,dpi):
 for ext in ('pdf','svg','png'):fig.savefig(out/f'{name}.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
 plt.close(fig)

def decorate(ax):
 ax.grid(axis='x',alpha=.15,zorder=0)
 for side in ('top','right','left'):ax.spines[side].set_visible(False)
 ax.tick_params(axis='x',labelsize=10)

def comparison(d,seqs,ids,title,stem,out,dpi,replicates=None):
 """DR6-inspired, publication-ready motif/metric panel, deliberately without title."""
 n=len(ids)
 fig=plt.figure(figsize=(19,max(5.8,0.92*n+2.15)))
 gs=fig.add_gridspec(1,5,width_ratios=[2.0,4.0,2.1,2.1,2.1],
                     left=.035,right=.99,top=.845,bottom=.115,wspace=.22)
 label,seq,*axes=[fig.add_subplot(gs[0,j]) for j in range(5)]
 ys=np.arange(n)[::-1];base=d.loc[863]
 for ax in (label,seq,*axes):ax.set_ylim(-.65,n-.35)
 label.axis('off');seq.set_xlim(0,84);seq.set_yticks([])
 seq.set_xticks([0,21,42,63,84]);seq.set_xlabel('Pentapeptide motif position',fontsize=11)
 for side in seq.spines.values():side.set_visible(False)
 for j,(i,y) in enumerate(zip(ids,ys)):
  if j%2==0:
   for ax in (seq,*axes):ax.axhspan(y-.48,y+.48,color='#F4F5F7',zorder=-4)
  label.text(.99,y,f'{i}'+('  E_AI triblock' if i==863 else ''),ha='right',va='center',fontweight='bold',fontsize=12)
  for x,m in enumerate(seqs[i]):seq.add_patch(Rectangle((x,y-.29),1,.58,facecolor=COLORS[m],edgecolor='none'))
 label.set_title('Architecture',fontsize=13.5,fontweight='bold',loc='right',pad=13)
 seq.set_title('Exact 84-motif design',fontsize=13.5,fontweight='bold',pad=13)
 for ax,(metric,heading,higher) in zip(axes,METRICS):
  v=d.loc[ids,metric+'_mean'].astype(float)
  lo=min(v.min(),float(base[metric+'_mean']));hi=max(v.max(),float(base[metric+'_mean']))
  spread=max(hi-lo,.035 if metric!='mean_compact_rg_nm' else .008)
  # Include replicate extrema in axis limits, not only the architecture means.
  if replicates is not None:
   raw=replicates.loc[replicates.index.get_level_values('architecture_id').isin(ids),metric].dropna()
   if len(raw):lo=min(lo,float(raw.min()));hi=max(hi,float(raw.max()))
  elif metric+'_std' in d.columns:
   sd=d.loc[ids,metric+'_std'].fillna(0).astype(float)
   lo=min(lo,float((v-sd).min()));hi=max(hi,float((v+sd).max()))
  spread=max(hi-lo,.035 if metric!='mean_compact_rg_nm' else .008)
  ax.set_xlim(lo-.35*spread,hi+.65*spread)
  ax.axvline(float(base[metric+'_mean']),ls=(0,(3,3)),lw=1.2,color='#B0783E',zorder=1)
  for i,y in zip(ids,ys):
   r=d.loc[i];mean=float(r[metric+'_mean'])
   # Faint horizontal replicate range behind the mean; no individual ticks.
   if replicates is not None and i in replicates.index.get_level_values('architecture_id'):
    values=replicates.xs(i,level='architecture_id')[metric].dropna().to_numpy(float)
    if len(values):
     ax.plot([values.min(),values.max()],[y,y],color='#A7B0BE',lw=3.0,
             alpha=.52,solid_capstyle='round',zorder=2)
   elif metric+'_std' in d and pd.notna(r.get(metric+'_std')):
    sd=float(r[metric+'_std'])
    ax.plot([mean-sd,mean+sd],[y,y],color='#A7B0BE',lw=3.0,
            alpha=.52,solid_capstyle='round',zorder=2)
   ax.plot(mean,y,'o',ms=6.8,color='#8051AB' if i!=863 else '#B0783E',mec='white',mew=.5,zorder=4)
   a,b=ax.get_xlim();offset=-8 if (mean-a)/(b-a)>.76 else 8
   ax.annotate(f'{mean:.3f}',(mean,y),xytext=(offset,0),textcoords='offset points',ha='right' if offset<0 else 'left',va='center',fontsize=10.5,fontweight='bold' if i==863 else 'normal')
  ax.set_yticks([]);decorate(ax)
  ax.set_title(heading+'\n'+('Higher is favourable ↑' if higher else 'Lower is favourable ↓'),fontsize=12.5,fontweight='bold',pad=13)
 # Legend sits above headings, with adequate space; no title in figure.
 fig.legend(handles=[Patch(color=COLORS[k],label=name) for k,name in [('VG','VPGVG'),('IG','VPGIG'),('IP','IPAVG')]]+
            [Line2D([0],[0],color='#B0783E',ls='--',label='E_AI reference')],
            loc='upper center',bbox_to_anchor=(.52,.987),frameon=False,ncol=4,fontsize=11)
 fig.text(.035,.033,('Dots: replicate mean; faint lines: range across replicates.' if replicates is not None else 'Dots: replicate mean; faint lines: ±1 replicate SD when available.')+' Dashed line: E_AI triblock.',fontsize=10,color='#646B73')
 save(fig,stem,out,dpi)


def _place_scatter_labels(fig,ax,positions,fontsize=8.8):
 """Collision-avoidance layout in display pixels, with leader lines for moved IDs.

 Uses fixed seed and bounded pixel shifts. The point positions are unchanged.
 """
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 texts={}
 for i,(x,y) in positions.items():
  t=ax.annotate(str(i),(x,y),xytext=(6,7),textcoords='offset points',
     ha='left',va='bottom',fontsize=fontsize,color='#2D3540',
     fontweight='bold' if i in (863,900) else 'medium',zorder=8)
  texts[i]=t
 fig.canvas.draw()
 # iteratively resolve overlap among text boxes and with neighbouring points
 from matplotlib.transforms import Bbox
 ids=list(texts)
 bbox=ax.get_window_extent(renderer).padded(-5)
 pts={i:ax.transData.transform(positions[i]) for i in ids}
 offsets={i:np.array([6.,7.]) for i in ids}
 for iteration in range(360):
  boxes={i:texts[i].get_window_extent(renderer).expanded(1.08,1.18) for i in ids}
  moves={i:np.zeros(2) for i in ids}
  overlaps=0
  for j,i in enumerate(ids):
   bi=boxes[i]
   for k in ids[j+1:]:
    bk=boxes[k]
    if not bi.overlaps(bk):continue
    overlaps+=1
    dx=min(bi.x1-bk.x0,bk.x1-bi.x0)
    dy=min(bi.y1-bk.y0,bk.y1-bi.y0)
    if dx<dy*1.1:
     sign=1 if bi.x0<bk.x0 else -1
     moves[i][0]-=sign*(dx/2+.65);moves[k][0]+=sign*(dx/2+.65)
    else:
     sign=1 if bi.y0<bk.y0 else -1
     moves[i][1]-=sign*(dy/2+.65);moves[k][1]+=sign*(dy/2+.65)
   # keep labels away from other data markers
   for k,pixel in pts.items():
    if k==i:continue
    if bi.x0-3<pixel[0]<bi.x1+3 and bi.y0-3<pixel[1]<bi.y1+3:
     moves[i][1]+=2.2 if (bi.y0+bi.y1)/2>=pixel[1] else -2.2
   if bi.x0<bbox.x0:moves[i][0]+=bbox.x0-bi.x0+1
   if bi.x1>bbox.x1:moves[i][0]-=bi.x1-bbox.x1+1
   if bi.y0<bbox.y0:moves[i][1]+=bbox.y0-bi.y0+1
   if bi.y1>bbox.y1:moves[i][1]-=bi.y1-bbox.y1+1
  if overlaps==0 and max(np.linalg.norm(v) for v in moves.values())<.5:break
  for i in ids:
   offsets[i]+=np.clip(moves[i],-14,14)*.75
   texts[i].set_position(tuple(offsets[i]))
  fig.canvas.draw()
 # Light leader lines connect displaced architecture numbers to exact points.
 for i in ids:
  if np.linalg.norm(offsets[i]-np.array([6.,7.]))>12:
   texts[i].arrow_patch=None
   ax.annotate('',xy=positions[i],xytext=tuple(offsets[i]),textcoords='offset points',
      xycoords='data',arrowprops=dict(arrowstyle='-',lw=.48,color='#A2A8AE'),zorder=5)
 return texts


def scatter(d,out,dpi,temp):
 fig,ax=plt.subplots(figsize=(14.8,10.8))
 fig.subplots_adjust(left=.105,right=.89,bottom=.12,top=.96)
 ref=d.loc[863]
 xref=float(ref.kescape_ns_inv_mean);yref=float(ref.kcollapse_ns_inv_mean)
 x=d.kescape_ns_inv_mean.astype(float);y=d.kcollapse_ns_inv_mean.astype(float)
 rg=d.mean_compact_rg_nm_mean.astype(float)
 ax.set_xlim(x.min()-.052,x.max()+.075)
 ax.set_ylim(y.min()-.18,y.max()+.18)
 # Four distinct, very light quadrants. No labels or interpretation inside plot.
 xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim()
 ax.add_patch(Rectangle((xmin,yref),xref-xmin,ymax-yref,facecolor='#E9F2EB',alpha=.72,zorder=-4))
 ax.add_patch(Rectangle((xref,yref),xmax-xref,ymax-yref,facecolor='#F4F5F6',alpha=.7,zorder=-4))
 ax.add_patch(Rectangle((xmin,ymin),xref-xmin,yref-ymin,facecolor='#F3F5F7',alpha=.65,zorder=-4))
 ax.add_patch(Rectangle((xref,ymin),xmax-xref,yref-ymin,facecolor='#F8F2EF',alpha=.58,zorder=-4))
 ax.axvline(xref,color='#A58460',lw=1.0,ls=(0,(4,3)),zorder=1)
 ax.axhline(yref,color='#A58460',lw=1.0,ls=(0,(4,3)),zorder=1)
 both=(x<xref)&(y>yref)
 # Compact Rg is represented by point AREA: smaller Rg -> larger symbol.
 rmin,rmax=float(rg.min()),float(rg.max())
 def size(v):return 46+165*(rmax-float(v))/(rmax-rmin) if rmax>rmin else 115
 sizes=np.array([size(rg.loc[i]) for i in d.index])
 for selected,color in [(~both,'#A7AFBA'),(both,'#426E9D')]:
  ids=d.index[selected]
  ax.scatter(x.loc[ids],y.loc[ids],s=[size(rg.loc[i]) for i in ids],
             c=color,edgecolors='white',linewidths=.9,alpha=.92,zorder=3)
 ax.scatter([x[863]],[y[863]],s=size(rg[863])*1.3,marker='*',
            c='#B0783E',edgecolor='white',linewidth=.8,zorder=6)
 ax.scatter([x[900]],[y[900]],s=size(rg[900])*1.1,marker='D',
            c='#BA523B',edgecolor='white',linewidth=.8,zorder=6)
 _place_scatter_labels(fig,ax,{int(i):(float(x[i]),float(y[i])) for i in d.index})
 ax.set_xlabel(METRICS[1][1],fontsize=14,fontweight='bold',labelpad=11)
 ax.set_ylabel(METRICS[0][1],fontsize=14,fontweight='bold',labelpad=12)
 ax.grid(alpha=.13,lw=.7)
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(labelsize=11)
 # Explicit size legend. Label is a size measurement, not a second performance score.
 legend_rg=np.linspace(rmin,rmax,3)
 handles=[ax.scatter([],[],s=size(v),c='#667F99',edgecolor='white',linewidth=.8)
          for v in legend_rg]
 leg=ax.legend(handles,[f'{v:.3f} nm' for v in legend_rg],
        title='Mean compact $R_g$\n(larger marker = smaller $R_g$)',
        loc='upper left',bbox_to_anchor=(1.012,.99),frameon=False,
        labelspacing=2.1,handletextpad=1.2,borderaxespad=0.,fontsize=10,title_fontsize=10)
 leg._legend_box.align='left'
 save(fig,f'DR7_01_kinetic_scatter_{temp}K',out,dpi)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--summary',type=Path,required=True)
 p.add_argument('--outdir',type=Path,default=Path('DR7_thesis_figures'))
 p.add_argument('--replicates',type=Path,default=None,help='Optional summary_per_replicate.csv for actual replicate ranges')
 p.add_argument('--temperature',type=int,default=370)
 p.add_argument('--dpi',type=int,default=240)
 args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
 seqs=sequences();d=read_summary(args.summary,args.temperature)
 replicates=read_replicates(args.replicates,args.temperature)
 ref=d.loc[863]
 both=(d.kescape_ns_inv_mean<ref.kescape_ns_inv_mean)&(d.kcollapse_ns_inv_mean>ref.kcollapse_ns_inv_mean)
 allthree=both&(d.mean_compact_rg_nm_mean<ref.mean_compact_rg_nm_mean)
 both.loc[863]=False;allthree.loc[863]=False
 print(f'Both kinetic metrics: {int(both.sum())}/43; all three: {int(allthree.sum())}/43')
 # Balanced top six: equal-weight ordinal ranks among 43 redesigns, no compact Rg in score.
 ranked=d.drop(index=863).copy()
 ranked['collapse_rank']=ranked.kcollapse_ns_inv_mean.rank(ascending=False,method='average')
 ranked['escape_rank']=ranked.kescape_ns_inv_mean.rank(ascending=True,method='average')
 ranked['balanced_score']=ranked.collapse_rank+ranked.escape_rank
 ranked=ranked.sort_values(['balanced_score','kescape_ns_inv_mean','architecture_id'])
 ranked.to_csv(args.outdir/'DR7_all_redesigns_kinetic_ranking.csv')
 ids_top=[863]+ranked.head(6).index.tolist()
 ids_three=[863]+[i for i in d[allthree].sort_values('kescape_ns_inv_mean').index if i != 891]
 print('Top six balanced:',ids_top[1:]);print('All-three improvements:',ids_three[1:])
 pd.DataFrame({'architecture_id':d.index,'improves_both_kinetics':both,'improves_all_three':allthree}).to_csv(args.outdir/'DR7_improvement_flags.csv',index=False)
 pd.DataFrame([dict(architecture_id=i,expression=DEFINITIONS[i],motifs=len(seqs[i]),**Counter(seqs[i])) for i in sorted(seqs)]).to_csv(args.outdir/'DR7_sequence_audit.csv',index=False)
 scatter(d,args.outdir,args.dpi,args.temperature)
 comparison(d,seqs,ids_top,f'DR7 | Six best balanced kinetic architectures | {args.temperature} K',f'DR7_02_top6_kinetics_{args.temperature}K',args.outdir,args.dpi,replicates)
 comparison(d,seqs,ids_three,f'DR7 | Redesigns improving all three metrics | {args.temperature} K',f'DR7_03_all_three_improved_{args.temperature}K',args.outdir,args.dpi,replicates)
 comparison(d,seqs,[863,869,870,898,895,899,900,901],f'DR7 | Domain positioning and kinetic trade-offs | {args.temperature} K',f'DR7_04_domain_position_{args.temperature}K',args.outdir,args.dpi,replicates)
 comparison(d,seqs,[863,865,870,878,879,882,884,899,900],f'DR7 | Block segregation and compact-state size | {args.temperature} K',f'DR7_05_segregation_compact_rg_{args.temperature}K',args.outdir,args.dpi,replicates)
 print('Saved figures and CSVs to',args.outdir.resolve())

if __name__=='__main__':main()
