#!/usr/bin/env python3
"""Thesis plots for pooled DR3 R4 nonlocal B-B contacts.
Usage: python plot_DR3_nonlocal_BB_contact_options.py --input nonlocal_BB_contacts_by_sequence_separation.csv --outdir DR3_BB_plots
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ORDER=['Random shuffle','Even','Small blocks','Medium blocks','Large blocks','Higher λ center','Higher λ ends','Diblock']
COLORS={'Random shuffle':'#BDBDBD','Even':'#5B9BD5','Small blocks':'#70AD47','Medium blocks':'#8064A2','Large blocks':'#ED7D31','Higher λ center':'#E53935','Higher λ ends':'#20B2C9','Diblock':'#8C564B'}
LABELS={'Higher λ center':'High-λ center','Higher λ ends':'High-λ endcaps'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'axes.labelsize':19,'axes.titlesize':21,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':15,'axes.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42})

def decorate(ax,xlabel='Sequence separation, |i − j| (residues)'):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=5,width=1.2)
    ax.set_xlim(0,420)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlabel(xlabel,labelpad=10)
    ax.grid(axis='y',color='#dddddd',alpha=.45,lw=.8)
    ax.set_axisbelow(True)

def save(fig,outdir,name):
    fig.savefig(outdir/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(outdir/(name+'.png'),bbox_inches='tight',dpi=350)
    plt.close(fig)

def legend(fig, handles, labels):
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,title='Architecture',title_fontsize=17,handlelength=2.5,columnspacing=1.5)

def lines(data,outdir,column,ylabel,name,title,percentage=False):
    fig,ax=plt.subplots(figsize=(12.5,7.5))
    handles=[]; labels=[]
    for arch in ORDER:
        d=data[data.architecture==arch].sort_values('sequence_separation_bin_center_aa')
        y=d[column].to_numpy()* (100 if percentage else 1)
        line,=ax.plot(d.sequence_separation_bin_center_aa,y,color=COLORS[arch],lw=3.1,label=LABELS.get(arch,arch))
        handles.append(line);labels.append(LABELS.get(arch,arch))
    decorate(ax)
    ax.set_ylabel(ylabel,labelpad=12)
    ax.set_title(title,fontweight='bold',pad=17)
    fig.subplots_adjust(left=.13,right=.98,top=.87,bottom=.30)
    legend(fig,handles,labels)
    save(fig,outdir,name)

def survival(data,outdir):
    fig,ax=plt.subplots(figsize=(12.5,7.5))
    handles=[]; labels=[]
    for arch in ORDER:
        d=data[data.architecture==arch].sort_values('sequence_separation_bin_center_aa')
        x=d.sequence_separation_bin_center_aa.to_numpy()
        p=d.fraction_of_contacts.to_numpy()
        survival=np.cumsum(p[::-1])[::-1]*100
        line,=ax.plot(x,survival,color=COLORS[arch],lw=3.1)
        handles.append(line);labels.append(LABELS.get(arch,arch))
    decorate(ax,'Minimum sequence separation, |i − j| (residues)')
    ax.set_ylim(0,101)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.set_ylabel('Contacts with sequence separation ≥ threshold (%)',labelpad=12)
    ax.set_title('Long-range B–B contact fraction',fontweight='bold',pad=17)
    fig.subplots_adjust(left=.15,right=.98,top=.87,bottom=.30)
    legend(fig,handles,labels)
    save(fig,outdir,'03_DR3_BB_long_range_survival')

def thresholds(data,outdir):
    thresholds=[50,100,150,200]
    rows=[]
    for arch in ORDER:
        d=data[data.architecture==arch]
        for t in thresholds:
            # Bin midpoint approximates threshold; bins have width 5 aa.
            pct=100*d.loc[d.sequence_separation_bin_center_aa>=t,'fraction_of_contacts'].sum()
            rows.append({'architecture':arch,'threshold_aa':t,'fraction_percent':pct})
    tab=pd.DataFrame(rows)
    tab.to_csv(outdir/'DR3_BB_long_range_threshold_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(12.5,7.3))
    xx=np.arange(len(ORDER));w=.19
    for i,t in enumerate(thresholds):
        ys=[tab.loc[(tab.architecture==a)&(tab.threshold_aa==t),'fraction_percent'].iloc[0] for a in ORDER]
        ax.bar(xx+(i-1.5)*w,ys,width=w,label=f'≥ {t} aa',color=['#c7d6e6','#94b7d5','#598dbb','#245784'][i])
    ax.set_xticks(xx,[LABELS.get(a,a) for a in ORDER],rotation=28,ha='right')
    ax.set_ylabel('Fraction of B–B contacts (%)')
    ax.set_title('Long-range contacts by architecture',fontweight='bold',pad=16)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.23),fontsize=15)
    fig.subplots_adjust(left=.12,right=.98,top=.87,bottom=.35)
    save(fig,outdir,'04_DR3_BB_long_range_threshold_bars')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    required={'architecture','sequence_separation_bin_center_aa','contacts_per_selected_frame','fraction_of_contacts'}
    missing=required-set(df.columns)
    if missing:raise ValueError(f'Missing columns: {missing}')
    unknown=set(df.architecture)-set(ORDER)
    if unknown:raise ValueError(f'Unknown architecture names: {unknown}')
    for a,g in df.groupby('architecture'):
        if not np.isclose(g.fraction_of_contacts.sum(),1,atol=1e-4):raise ValueError(f'Fractions do not sum to 1: {a}')
    lines(df,args.outdir,'contacts_per_selected_frame','B–B contacts per selected compact frame','01_DR3_BB_absolute_contacts','Nonlocal B–B contacts')
    lines(df,args.outdir,'fraction_of_contacts','Fraction of B–B contacts per bin (%)','02_DR3_BB_normalized_contact_distribution','Normalized B–B contact distribution',True)
    survival(df,args.outdir)
    thresholds(df,args.outdir)
    print('Saved 4 figures as PDF and PNG to:',args.outdir)
    print('NOTE: input pools chemistries and replicates; no uncertainty is inferred.')
if __name__=='__main__':main()
