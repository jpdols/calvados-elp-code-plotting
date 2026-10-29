#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DR4 v20: three print-ready figures: 4-panel overview, IGKG all-architecture kinetics by ratio, IGKG block averages.

Input: combined rg_summary_all_libraries.csv.
All effects normalized WITHIN each library x ratio:
 Overview: reference = mean of all eight architectures.
 IGKG all-architecture kinetics: reference = mean of all eight architectures.
 IGKG block summary: reference = mean of four block architectures.
 Rates: log2(rate / reference rate), undefined if either rate <= 0.
 Collapsed fractions: percentage-point differences.
 Figure 3: arithmetic average of four normalized ratio values (R1-R4), min-max
 across ratios as pale vertical bars (not confidence intervals).
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':18,'axes.labelsize':17,
 'axes.titlesize':19,'xtick.labelsize':17,'ytick.labelsize':17,
 'legend.fontsize':17,'axes.linewidth':1.5,'lines.linewidth':3.3,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':600})
LIBRARIES=['VGEG','VGKG','IGKG']
ARCHS=['randomshuffle','even','smallblocks','mediumblocks','largeblocks',
       'charge_center','charge_ends','diblock']
BLOCKS=['even','smallblocks','mediumblocks','largeblocks']
LABELS={'randomshuffle':'Random shuffle','even':'Even','smallblocks':'Small blocks',
 'mediumblocks':'Medium blocks','largeblocks':'Large blocks',
 'charge_center':'Charge centre','charge_ends':'Charge ends','diblock':'Diblock'}
COLORS={'randomshuffle':'#BDBDBD','even':'#5B9BD5','smallblocks':'#70AD47',
 'mediumblocks':'#8064A2','largeblocks':'#ED7D31','charge_center':'#E74C3C',
 'charge_ends':'#2CB1C9','diblock':'#8C564B'}
RATIOS=['R1','R2','R3','R4','R5','R6']
RATIO_LABELS={'R1':'6:1\n(14%)','R2':'3:1\n(25%)','R3':'2:1\n(33%)',
 'R4':'1:1\n(50%)','R5':'1:2\n(67%)','R6':'1:5\n(83%)'}


def canonical_arch(s):
 s=re.sub(r'_+','_',str(s).lower().replace(' ','_').replace('-','_'))
 if 'random' in s or 'shuffle' in s:return 'randomshuffle'
 if 'diblock' in s:return 'diblock'
 for a in ('small','medium','large'):
  if a in s:return a+'blocks'
 if 'even' in s or 'alternat' in s:return 'even'
 if 'center' in s or 'centre' in s:return 'charge_center'
 if any(x in s for x in ('endcap','ends','egends','kgends')):return 'charge_ends'
 return s

def canonical_ratio(s):
 s=str(s).strip().upper()
 m=re.search(r'R\s*([1-6])',s)
 if m:return 'R'+m.group(1)
 return {'6:1':'R1','3:1':'R2','2:1':'R3','1:1':'R4',
         '1:2':'R5','1:5':'R6'}.get(s,s)

def load(path):
 raw=pd.read_csv(path)
 needed=['library','ratio','architecture','mean_Rg','k_collapse_per_ns','k_escape_per_ns']
 missing=[x for x in needed if x not in raw.columns]
 if not ('collapsed_fraction' in raw or 'collapsed_percent' in raw):missing.append('collapsed_fraction or collapsed_percent')
 if missing:raise ValueError(f'Missing columns: {missing}\nFound: {list(raw.columns)}')
 fraction_col='collapsed_fraction' if 'collapsed_fraction' in raw else 'collapsed_percent'
 d=pd.DataFrame({'library':raw.library.astype(str).str.upper().str.strip(),
  'ratio':raw.ratio.map(canonical_ratio),'architecture':raw.architecture.map(canonical_arch),
  'rg':pd.to_numeric(raw.mean_Rg,errors='coerce'),
  'collapsed_pp':pd.to_numeric(raw[fraction_col],errors='coerce'),
  'entry':pd.to_numeric(raw.k_collapse_per_ns,errors='coerce'),
  'escape':pd.to_numeric(raw.k_escape_per_ns,errors='coerce')})
 if fraction_col=='collapsed_fraction':d['collapsed_pp']*=100
 d=d[d.library.isin(LIBRARIES)&d.ratio.isin(RATIOS)&d.architecture.isin(ARCHS)].copy()
 d=d.groupby(['library','ratio','architecture'],as_index=False)[['rg','collapsed_pp','entry','escape']].mean()
 counts=d.groupby(['library','ratio']).architecture.nunique().unstack(fill_value=0).reindex(index=LIBRARIES,columns=RATIOS)
 print('Architectures per library and ratio:\n',counts.to_string())
 if (counts!=8).any().any():print('WARNING: Not all library-ratio groups have eight architectures.')
 return d

def normalize(d, architectures):
 sub=d[d.architecture.isin(architectures)].copy()
 for metric in ('rg','collapsed_pp'):
  sub['delta_'+metric]=sub[metric]-sub.groupby(['library','ratio'])[metric].transform('mean')
 for metric in ('entry','escape'):
  ref=sub.groupby(['library','ratio'])[metric].transform('mean')
  with np.errstate(divide='ignore',invalid='ignore'):
   val=np.log2(sub[metric]/ref)
  sub['log2_'+metric]=val.where((sub[metric]>0)&(ref>0)&np.isfinite(val),np.nan)
 return sub

def clean(ax):
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 ax.tick_params(direction='out',length=6,width=1.2)
 ax.axhline(0,color='0.65',ls='--',lw=1.2,zorder=0)

def letter(ax,s):
 ax.text(-.12,1.045,s,transform=ax.transAxes,fontsize=24,fontweight='bold',ha='right',va='bottom')

def handles(archs):
 return [Line2D([0],[0],color=COLORS[a],marker='o',markeredgecolor='white',
  markersize=11,lw=3.5,label=LABELS[a]) for a in archs]

def save(fig,outdir,name,dpi):
 for ext in ('pdf','png'):
  p=outdir/f'{name}.{ext}';fig.savefig(p,bbox_inches='tight',dpi=dpi);print('Saved:',p)
 plt.close(fig)

def ratio_lines(ax,table,archs,metric,ratios,spread=False):
 clean(ax);x=np.arange(len(ratios))
 for a in archs:
  t=table[table.architecture==a].set_index('ratio').reindex(ratios)
  y=t[metric].to_numpy(float)
  if spread:
   lo=t[metric+'_min'].to_numpy(float);hi=t[metric+'_max'].to_numpy(float)
   for xi,l,h in zip(x,lo,hi):
    if np.isfinite(l) and np.isfinite(h):ax.vlines(xi,l,h,color=COLORS[a],alpha=.27,lw=2.2,zorder=1)
  ax.plot(x,y,color=COLORS[a],marker='o',markersize=9.5,markeredgecolor='white',
          markeredgewidth=.7,lw=3.3,zorder=3)
 ax.set_xticks(x,[RATIO_LABELS[r] for r in ratios]);ax.set_xlabel('Motif composition (A:C)',labelpad=6)


def figure1(allnorm,outdir,dpi):
 # Top row: E + K averaged after independent library-ratio normalization.
 # Bottom row: IGKG alone, same all-eight reference.
 ek=allnorm[allnorm.library.isin(['VGEG','VGKG'])]
 ek=ek.groupby(['ratio','architecture'],as_index=False).agg(
  delta_rg=('delta_rg','mean'),delta_rg_min=('delta_rg','min'),delta_rg_max=('delta_rg','max'),
  delta_collapsed_pp=('delta_collapsed_pp','mean'),
  delta_collapsed_pp_min=('delta_collapsed_pp','min'),
  delta_collapsed_pp_max=('delta_collapsed_pp','max'),n_libraries=('library','nunique'))
 ig=allnorm[allnorm.library=='IGKG'].copy()
 ek.to_csv(outdir/'01_overview_EK_combined.csv',index=False)
 ig.to_csv(outdir/'01_overview_IGKG.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(11.5,10.9))
 panels=[(ek,'delta_rg',True,r'$\Delta$ mean $R_g$ (nm)','Mean dimensions'),
  (ek,'delta_collapsed_pp',True,r'$\Delta$ collapsed fraction (pp)','Collapsed population'),
  (ig,'delta_rg',False,r'$\Delta$ mean $R_g$ (nm)','Mean dimensions'),
  (ig,'delta_collapsed_pp',False,r'$\Delta$ collapsed fraction (pp)','Collapsed population')]
 for i,(tab,metric,spread,ylabel,title) in enumerate(panels):
  ax=axes.ravel()[i];ratio_lines(ax,tab,ARCHS,metric,RATIOS,spread)
  ax.set_ylabel(ylabel,fontsize=15,labelpad=3);ax.set_title(title,fontweight='bold',pad=9);letter(ax,'ABCD'[i])
 fig.legend(handles=handles(ARCHS),loc='lower center',bbox_to_anchor=(.5,.015),
  ncol=4,frameon=False,title='Architecture',title_fontsize=18,labelspacing=0.55,columnspacing=0.75,handlelength=1.45)
 fig.subplots_adjust(left=.14,right=.98,top=.93,bottom=.27,wspace=.43,hspace=.55)
 save(fig,outdir,'01_DR4_architecture_overview_EK_and_IGKG',dpi)

def figure2(allnorm,outdir,dpi):
 # All eight architectures; normalize against all-eight composition mean.
 ig=allnorm[(allnorm.library=='IGKG')&allnorm.ratio.isin(RATIOS[:4])].copy()
 ig.to_csv(outdir/'02_IGKG_all_arch_kinetics_R1_R4.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11.5,6.7))
 panels=[('log2_entry',r'$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$','Compact-state entry'),
         ('log2_escape',r'$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$','Compact-state escape')]
 for i,(metric,ylabel,title) in enumerate(panels):
  ax=axes[i];ratio_lines(ax,ig,ARCHS,metric,RATIOS[:4]);ax.set_ylabel(ylabel,fontsize=14,labelpad=2)
  ax.set_title(title,fontweight='bold',pad=9);letter(ax,'AB'[i])
 fig.legend(handles=handles(ARCHS),loc='lower center',bbox_to_anchor=(.5,.01),
  ncol=4,frameon=False,title='Architecture',title_fontsize=18,labelspacing=0.55,columnspacing=0.75,handlelength=1.45)
 fig.subplots_adjust(left=.15,right=.98,top=.86,bottom=.42,wspace=.48)
 save(fig,outdir,'02_DR4_IGKG_all_arch_kinetics_R1_R4',dpi)

def figure3(blocknorm,outdir,dpi):
 ig=blocknorm[(blocknorm.library=='IGKG')&blocknorm.ratio.isin(RATIOS[:4])].copy()
 panels=[('delta_collapsed_pp',r'$\Delta$ collapsed fraction (pp)','Collapsed population'),
  ('delta_rg',r'$\Delta$ mean $R_g$ (nm)','Mean dimensions'),
  ('log2_entry',r'$\log_2(k_{\mathrm{collapse}}/\overline{k}_{\mathrm{collapse}})$','Compact-state entry'),
  ('log2_escape',r'$\log_2(k_{\mathrm{escape}}/\overline{k}_{\mathrm{escape}})$','Compact-state escape')]
 rows=[]
 for a in BLOCKS:
  part=ig[ig.architecture==a]
  for metric,_,_ in panels:
   vals=part[metric].replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
   rows.append({'architecture':a,'metric':metric,'n_ratios':len(vals),
    'mean':np.mean(vals) if len(vals) else np.nan,
    'min':np.min(vals) if len(vals) else np.nan,
    'max':np.max(vals) if len(vals) else np.nan})
 summary=pd.DataFrame(rows)
 summary.to_csv(outdir/'03_IGKG_block_R1_R4_averages.csv',index=False)
 fig,axes=plt.subplots(2,2,figsize=(11.5,10.0))
 x=np.arange(len(BLOCKS))
 for i,(metric,ylabel,title) in enumerate(panels):
  ax=axes.ravel()[i];clean(ax)
  for j,a in enumerate(BLOCKS):
   r=summary[(summary.architecture==a)&(summary.metric==metric)].iloc[0]
   if np.isfinite(r['min']) and np.isfinite(r['max']):
    ax.vlines(j,r['min'],r['max'],color=COLORS[a],alpha=.33,lw=3.0,zorder=1)
   if np.isfinite(r['mean']):
    ax.scatter(j,r['mean'],s=125,color=COLORS[a],edgecolor='white',linewidth=.7,zorder=3)
  ax.set_xticks(x,[LABELS[a] for a in BLOCKS],rotation=18,ha='right');ax.tick_params(axis='x',labelsize=14)
  ax.set_xlim(-.45,len(BLOCKS)-.55)
  ax.set_ylabel(ylabel,fontsize=15,labelpad=3);ax.set_title(title,fontweight='bold',pad=9);letter(ax,'ABCD'[i])
 fig.subplots_adjust(left=.15,right=.98,top=.92,bottom=.19,wspace=.47,hspace=.70)
 save(fig,outdir,'03_DR4_IGKG_block_R1_R4_average_four_metrics',dpi)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True,type=Path)
 p.add_argument('--outdir',required=True,type=Path)
 p.add_argument('--dpi',type=int,default=600)
 args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
 d=load(args.input)
 allnorm=normalize(d,ARCHS)
 blocknorm=normalize(d,BLOCKS)
 allnorm.to_csv(args.outdir/'00_all_architecture_normalized.csv',index=False)
 blocknorm.to_csv(args.outdir/'00_four_block_normalized.csv',index=False)
 figure1(allnorm,args.outdir,args.dpi)
 figure2(allnorm,args.outdir,args.dpi)
 figure3(blocknorm,args.outdir,args.dpi)
 print('Done: exactly three DR4 figures.')
if __name__=='__main__':main()
