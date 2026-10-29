#!/usr/bin/env python3
"""DR7 three-replicate thesis overview: architecture strips, three metrics and heatmap.
Run with --summary and --per-replicate. No fraction compact panel or combined score.
"""
import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize

DEFINITIONS = {863: '(IP)12-(VG2-IG-VG2)12-(IP)12', 864: 'RANDOM', 865: '(IP-VG2-IG-VG2-IP)12', 866: '(IP)24-(VG)48-(IG)12', 867: '(IP)24-(IG)12-(VG)48', 868: '(VG)48-(IP)24-(IG)12', 869: '(IP)12-(VG)24-(IG)12-(VG)24-(IP)12', 870: '(IP)12-(IG)6-(VG)48-(IG)6-(IP)12', 871: '(IP)12-(VG)18-(IG)6-(VG)12-(IG)6-(VG)18-(IP)12', 872: '(IP)12-(VG)12-(IG)4-(VG)12-(IG)4-(VG)12-(IG)4-(VG)12-(IP)12', 873: '(IP)12-(VG)6-(IG)3-(VG)12-(IG)3-(VG)12-(IG)3-(VG)12-(IG)3-(VG)6-(IP)12', 874: '(IP)12-(VG4-IG)12-(IP)12', 875: '(IP)6-(VG2-IG-VG2)4-(IP)6-(VG2-IG-VG2)4-(IP)6-(VG2-IG-VG2)4-(IP)6', 876: '(IP)8-(VG2-IG-VG2)6-(IP)8-(VG2-IG-VG2)6-(IP)8', 877: '(IP4-(VG2-IG-VG2)2)6', 878: '(VG4-IG)6-(IP)24-(IG-VG4)6', 879: '(VG)24-(IG)6-(IP)24-(IG)6-(VG)24', 880: '(IG)6-(VG)24-(IP)24-(VG)24-(IG)6', 881: '(VG)12-(IP)12-(VG)12-(IG)12-(VG)12-(IP)12-(VG)12', 882: '(VG)24-(IP)12-(IG)12-(IP)12-(VG)24', 883: '(VG-IP)12-(VG)12-(IG)12-(VG)12-(IP-VG)12', 884: '(VG)24-(IP2-IG)12-(VG)24', 885: '(VG4-IG-IP2)12', 886: '(IP)24-(VG4-IG)12', 887: '(IP)12-(VG)16-(IG)4-(VG)8-(IG)4-(VG)8-(IG)4-(VG)16-(IP)12', 888: '(IP)6-(VG)12-(IG)3-(VG)12-(IP)6-(VG)12-(IG)6-(VG)12-(IP)6-(IG)3-(IP)6', 889: '(VG)10-(IP)8-(VG)9-(IG)4-(VG)10-(IP)8-(VG)9-(IG)4-(VG)10-(IP)8-(IG)4', 890: '(IP2-VG4-IG)12', 891: '(IP)12-(VG3-IG)6-(VG)12-(IG-VG3)6-(IP)12', 892: '(IP)4-(VG)8-(IG)2-(VG)8-(IP)8-(VG)8-(IG)8-(VG)8-(IP)8-(VG)8-(IG)2-(VG)8-(IP)4', 893: '(VG)24-(IP2-IG)6-(IG-IP2)6-(VG)24', 894: '(VG4-IG)6-(IP)12-(IG-VG4)6-(IP)12', 895: '((VG)12-(IP)6-(IG)3)4', 896: '((VG)16-(IP)8-(IG)4)3', 897: '(IP2-VG2-IG-VG2)12', 898: '(IP)12-(IG)6-(VG)24-(IG)6-(IP)12-(VG)24', 899: '(IP)12-(VG)12-(IG)6-(VG)24-(IG)6-(VG)12-(IP)12', 900: '(VG)12-(IP)12-(IG)6-(VG)24-(IG)6-(IP)12-(VG)12', 901: '(VG)12-(IG)6-(IP)12-(VG)24-(IP)12-(IG)6-(VG)12', 902: '(IP4-IG2-VG8)6', 903: '(IP)12-(VG)18-(IG)2-(VG)12-(IG)4-(VG)8-(IG)6-(VG)10-(IP)12', 904: '(VG)8-(IP)4-(VG)8-(IP)8-(VG)8-(IG)12-(VG)8-(IP)8-(VG)8-(IP)4-(VG)8', 905: '(IP-VG-IP-VG-IG-VG2)12', 906: '(IP6-IG3-VG12)4'}
GROUPS = {'References': [863, 864, 865, 905], 'Triblock order': [866, 867, 868], 'IP endcaps / IG organization': [869, 870, 871, 872, 873, 874], 'IP fragmentation / central IP': [875, 876, 877, 878, 879, 880, 881, 882, 883], 'Paired / periodic / repeating': [884, 885, 886, 889, 892, 893, 894, 895, 896, 897, 902, 906, 890], 'Gradients / layers / asymmetry': [887, 888, 891, 898, 899, 900, 901, 903, 904]}
MOTIF_COLORS = {'VG': '#2684c7', 'IG': '#39a65a', 'IP': '#f1c40f'}

METRICS=[('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)',False),('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)',True),('mean_compact_rg_nm',r'Mean compact $R_g$ (nm)',False)]
def expand(expr):
 tokens=re.findall(r'VG|IG|IP|\d+|[()]',expr); pos=0
 def parse():
  nonlocal pos
  out=[]
  while pos<len(tokens) and tokens[pos]!=')':
   tok=tokens[pos];pos+=1
   if tok=='(':
    part=parse()
    if pos>=len(tokens) or tokens[pos]!=')':raise ValueError('Unbalanced '+expr)
    pos+=1
   elif tok in MOTIF_COLORS:part=[tok]
   else:raise ValueError('Unexpected '+tok)
   count=1
   if pos<len(tokens) and tokens[pos].isdigit():count=int(tokens[pos]);pos+=1
   out.extend(part*count)
  return out
 result=parse()
 if pos!=len(tokens):raise ValueError('Unbalanced '+expr)
 return result

def save(fig,out,name,formats,dpi):
 for ext in formats:fig.savefig(out/f'{name}.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--summary',required=True);p.add_argument('--per-replicate',required=True)
 p.add_argument('--outdir',default='DR7_overview_plots');p.add_argument('--temperature',type=int,default=370)
 p.add_argument('--formats',nargs='+',default=['pdf','svg','png'],choices=['pdf','svg','png']);p.add_argument('--dpi',type=int,default=220)
 p.add_argument('--sort',choices=['id','escape','collapse','compact_rg'],default='id')
 a=p.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.summary);rep=pd.read_csv(a.per_replicate)
 d=d.loc[(d.temperature==a.temperature)&d.architecture_id.between(863,906)].copy()
 rep=rep.loc[(rep.temperature==a.temperature)&rep.architecture_id.between(863,906)].copy()
 needed=['architecture_id','architecture','n_replicates']+[m+'_'+s for m,_,_ in METRICS for s in ('mean','min','max')]
 missing=[c for c in needed if c not in d.columns]
 if missing: p.error('Missing summary columns: '+', '.join(missing))
 if d.architecture_id.duplicated().any():p.error('Duplicate architecture IDs')
 if d.empty:p.error('No rows for selected temperature')
 if not set(d.architecture_id).issubset(DEFINITIONS):p.error('Unknown architecture definition')
 if not set(d.architecture_id).issubset(set(rep.architecture_id)):p.error('Some architectures have no replicate rows')
 if (d.n_replicates!=3).any():print('WARNING: some architectures have fewer/more than 3 replicates')
 for ident,expr in DEFINITIONS.items():
  if expr!='RANDOM':
   seq=expand(expr)
   if {m:seq.count(m) for m in MOTIF_COLORS}!={'VG':48,'IG':12,'IP':24}:
    print(f'WARNING architecture {ident} motif counts differ from expected 48/12/24')
 sortcol={'id':'architecture_id','escape':'kescape_ns_inv_mean','collapse':'kcollapse_ns_inv_mean','compact_rg':'mean_compact_rg_nm_mean'}[a.sort]
 d=d.sort_values(sortcol,ascending=a.sort!='collapse').reset_index(drop=True)
 n=len(d);y=np.arange(n);height=max(9,n*.34+2.8)
 fig,axs=plt.subplots(1,4,figsize=(19,height),sharey=True,gridspec_kw={'width_ratios':[3.3,1.25,1.25,1.25]})
 for ax in axs:
  ax.set_ylim(n-.5,-.5);ax.spines[['top','right']].set_visible(False)
  for j in range(n):
   if j%2==0:ax.axhspan(j-.5,j+.5,color='#f5f6f7',zorder=0)
  ax.tick_params(axis='y',length=0)
 for i,r in d.iterrows():
  expr=DEFINITIONS[int(r.architecture_id)]
  if expr=='RANDOM':
   axs[0].add_patch(Rectangle((0,i-.22),84,.44,facecolor='#dddddd',edgecolor='#777777',hatch='///'))
  else:
   seq=expand(expr);start=0
   for k in range(1,len(seq)+1):
    if k==len(seq) or seq[k]!=seq[start]:
     axs[0].add_patch(Rectangle((start,i-.22),k-start,.44,facecolor=MOTIF_COLORS[seq[start]],lw=0));start=k
 axs[0].set_xlim(0,84);axs[0].set_xticks([0,21,42,63,84]);axs[0].set_xlabel('Motif position (84 pentapeptides)')
 axs[0].set_title('Sequence architecture',loc='left',fontweight='bold')
 axs[0].set_yticks(y,[f'{int(r.architecture_id)}  {str(r.architecture).replace("_"," ")}' for _,r in d.iterrows()],fontsize=8)
 base=d.loc[d.architecture_id==863]
 for ax,(metric,title,higher) in zip(axs[1:],METRICS):
  means=d[metric+'_mean'].to_numpy(float);lo=d[metric+'_min'].to_numpy(float);hi=d[metric+'_max'].to_numpy(float)
  ax.hlines(y,lo,hi,color='#b4c2cd',lw=2,zorder=2)
  ax.scatter(lo,y,s=11,color='#a5b5c2',zorder=3);ax.scatter(hi,y,s=11,color='#a5b5c2',zorder=3)
  ax.scatter(means,y,s=34,c=['#c4881b' if int(i)==863 else '#286e9a' for i in d.architecture_id],edgecolors='white',linewidths=.4,zorder=4)
  if not base.empty:ax.axvline(float(base.iloc[0][metric+'_mean']),color='#c4881b',ls='--',lw=1)
  ax.set_xlabel(title+'\n'+('Higher favourable' if higher else 'Lower favourable'))
  ax.set_title(chr(65+METRICS.index((metric,title,higher))),loc='left',fontweight='bold')
  ax.grid(axis='x',alpha=.18)
 fig.suptitle(f'DR7 | E_AI architecture overview | {a.temperature} K | three independent replicates',fontsize=15,y=.995)
 fig.legend(handles=[Patch(color=MOTIF_COLORS[m],label={'VG':'VPGVG','IG':'VPGIG','IP':'IPAVG'}[m]) for m in ('VG','IG','IP')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.48,.01))
 fig.subplots_adjust(left=.25,right=.99,top=.965,bottom=.065,wspace=.20)
 save(fig,out,f'DR7_all_architectures_3metrics_{a.temperature}K',a.formats,a.dpi)
 # Independent raw-value column scales, oriented so yellow is favourable.
 fig,axs=plt.subplots(1,4,figsize=(15,height),sharey=True,gridspec_kw={'width_ratios':[3.2,1,1,1]})
 for i,r in d.iterrows():
  expr=DEFINITIONS[int(r.architecture_id)]
  if expr=='RANDOM':axs[0].add_patch(Rectangle((0,i-.23),84,.46,facecolor='#dddddd',hatch='///',edgecolor='#777777'))
  else:
   seq=expand(expr);start=0
   for k in range(1,len(seq)+1):
    if k==len(seq) or seq[k]!=seq[start]:
     axs[0].add_patch(Rectangle((start,i-.23),k-start,.46,facecolor=MOTIF_COLORS[seq[start]],lw=0));start=k
 axs[0].set_xlim(0,84);axs[0].set_xticks([0,21,42,63,84]);axs[0].set_xlabel('Motif position');axs[0].set_title('Architecture',loc='left')
 axs[0].set_yticks(y,[f'{int(r.architecture_id)}  {str(r.architecture).replace("_"," ")}' for _,r in d.iterrows()],fontsize=8)
 for ax,(metric,title,higher) in zip(axs[1:],METRICS):
  v=d[metric+'_mean'].to_numpy(float)
  if not np.isfinite(v).all():raise ValueError(f'Nonfinite {metric}; inspect replicate data')
  vmin,vmax=v.min(),v.max();norm=Normalize(vmin,vmax if vmax>vmin else vmin+1)
  cmap=plt.get_cmap('viridis' if higher else 'viridis_r')
  ax.imshow(v[:,None],aspect='auto',cmap=cmap,norm=norm,extent=(-.5,.5,n-.5,-.5),interpolation='nearest')
  for j,value in enumerate(v):
   rgba=cmap(norm(value));bright=.299*rgba[0]+.587*rgba[1]+.114*rgba[2]
   ax.text(0,j,f'{value:.3f}',ha='center',va='center',fontsize=8,color='black' if bright>.53 else 'white')
  ax.set_xlim(-.5,.5);ax.set_xticks([]);ax.set_title(title+'\n'+('Higher favourable' if higher else 'Lower favourable'),fontsize=10)
  ax.tick_params(axis='y',length=0)
 for ax in axs:ax.set_ylim(n-.5,-.5);ax.spines[['top','right','bottom']].set_visible(False)
 fig.suptitle(f'DR7 | Three-metric heatmap | {a.temperature} K',fontsize=15,y=.995)
 fig.subplots_adjust(left=.30,right=.99,top=.96,bottom=.06,wspace=.12)
 save(fig,out,f'DR7_all_architectures_heatmap_{a.temperature}K',a.formats,a.dpi)
 # Kinetic map: every point and min-max replicate spread; colour is compact Rg.
 fig,ax=plt.subplots(figsize=(11,8))
 for _,r in d.iterrows():
  x=r.kescape_ns_inv_mean;y0=r.kcollapse_ns_inv_mean
  ax.errorbar(x,y0,xerr=[[max(0,x-r.kescape_ns_inv_min)],[max(0,r.kescape_ns_inv_max-x)]],yerr=[[max(0,y0-r.kcollapse_ns_inv_min)],[max(0,r.kcollapse_ns_inv_max-y0)]],fmt='none',ecolor='#c8d1d9',lw=.9,zorder=1)
 sc=ax.scatter(d.kescape_ns_inv_mean,d.kcollapse_ns_inv_mean,c=d.mean_compact_rg_nm_mean,cmap='viridis_r',s=55,edgecolor='white',lw=.6,zorder=3)
 for _,r in d.iterrows():ax.annotate(str(int(r.architecture_id)),(r.kescape_ns_inv_mean,r.kcollapse_ns_inv_mean),xytext=(5,5),textcoords='offset points',fontsize=7)
 fig.colorbar(sc,ax=ax,label=r'Mean compact $R_g$ (nm)')
 ax.set_xlabel(r'$k_{\mathrm{escape}}$ (ns$^{-1}$)');ax.set_ylabel(r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)')
 ax.set_title(f'DR7 | Collapse versus escape | {a.temperature} K');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();save(fig,out,f'DR7_kinetic_map_{a.temperature}K',a.formats,a.dpi)
 d.to_csv(out/f'DR7_plotted_values_{a.temperature}K.csv',index=False)
 (out/'plot_settings.json').write_text(json.dumps(dict(summary=a.summary,per_replicate=a.per_replicate,temperature=a.temperature,threshold_nm=4.61,sort=a.sort,metrics=[x[0] for x in METRICS],note='Error bars are replicate min/max, not confidence intervals. No fraction compact or composite ranking.'),indent=2))
 print(f'Done: {len(d)} architectures, plots saved to {out.resolve()}')
if __name__=='__main__':main()
