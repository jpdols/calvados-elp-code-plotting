#!/usr/bin/env python3
"""DR8 VGIPKG thesis story figures, styled after DR7 E_AI story.
Reads combined_summary_mean_std.csv, replicate_summary.csv, design_metadata.csv
from the same directory. No kinetic claims from static summary statistics.
"""
import argparse, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

COLORS={'VG':'#2684c7','IP':'#f1c40f','KG':'#ed2ca8'}
# Four combined figures, each with two explicitly labelled matched comparisons.
COMPARISON_FIGURES = {
 'C01_KG_placement': (
   ('A | KG placement around one central IP domain', [836,837,838,839,840,859]),
   ('B | KG clustering versus dispersal in VG-rich domains', [836,860,861])),
 'C02_domain_organization': (
   ('A | Two IP blocks: terminal, interfacial or central KG', [841,842,843]),
   ('B | Number of IP domains with KG endcaps', [836,841,847,850])),
 'C03_terminal_architecture': (
   ('A | Symmetry and terminal charge distribution', [841,854,856,858]),
   ('B | Reversing terminal IP and KG arrangement', [844,845,846])),
 'C04_reference_and_tradeoffs': (
   ('A | Reference versus representative high-performing designs', [835,841,845,850,856]),
   ('B | Charge aggregation, kinetics and compactness', [835,839,860,861])),
}

def expand(expr):
    """Expand motif expressions with parentheses, nested groups, and repetition."""
    if not isinstance(expr,str) or not re.search(r'VG|IP|KG',expr):return None
    tokens=re.findall(r'VG|IP|KG|\d+|[()]',expr.replace(' ','').replace('-',''))
    pos=0
    def parse():
        nonlocal pos
        out=[]
        while pos<len(tokens) and tokens[pos]!=')':
            tok=tokens[pos];pos+=1
            if tok=='(':
                item=parse()
                if pos>=len(tokens) or tokens[pos]!=')':raise ValueError('Unbalanced pattern')
                pos+=1
            elif tok in COLORS:item=[tok]
            else:raise ValueError('Unexpected motif token: '+tok)
            n=1
            if pos<len(tokens) and tokens[pos].isdigit():n=int(tokens[pos]);pos+=1
            out.extend(item*n)
        return out
    seq=parse()
    if pos!=len(tokens):raise ValueError('Unexpected trailing pattern token')
    return seq

def save(fig,folder,name,formats,dpi):
    for fmt in formats:fig.savefig(folder/f'{name}.{fmt}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def plot_story(sub,title,name,out,formats,dpi,reference,meta):
    if sub.empty:return
    n=len(sub);height=max(4.5,0.47*n+1.8)
    fig,axs=plt.subplots(1,5,figsize=(22,height),sharey=True,gridspec_kw={'width_ratios':[3.5,1.2,1.2,1.2,1.2]})
    y=np.arange(n)
    for ax in axs:
        for i in range(n):
            if i%2==0:ax.axhspan(i-.5,i+.5,color='#f4f4f4',zorder=0)
        ax.spines[['top','right']].set_visible(False)
    ax=axs[0]
    for i,r in enumerate(sub.itertuples()):
        seq=meta.get(int(r.design_id))
        if seq is None:
            ax.add_patch(Rectangle((0,i-.19),84,.38,facecolor='#dedede',edgecolor='#888',hatch='///',lw=.4))
            ax.text(42,i,'Sequence not verified',ha='center',va='center',fontsize=7)
            continue
        st=0
        for j in range(1,len(seq)+1):
            if j==len(seq) or seq[j]!=seq[st]:
                ax.add_patch(Rectangle((st,i-.19),j-st,.38,facecolor=COLORS[seq[st]],edgecolor='none'))
                st=j
    ax.set_xlim(-1,85);ax.set_xticks([0,21,42,63,84]);ax.set_xlabel('Motif position (84 pentapeptides)')
    ax.set_title('Sequence architecture',fontsize=11,pad=12)
    labels=[f'{int(r.design_id)}  {r.design_name.replace("_"," ")}' for r in sub.itertuples()]
    ax.set_yticks(y,labels,fontsize=9);ax.set_ylim(n-.5,-.5);ax.tick_params(axis='y',length=0)
    metrics=[('kescape_ns_inv_mean','kescape_ns_inv_std',r'$k_{escape}$ (ns$^{-1}$)',False),('kcollapse_ns_inv_mean','kcollapse_ns_inv_std',r'$k_{collapse}$ (ns$^{-1}$)',True),('collapsed_percent','std_collapsed_percent_between_reps','Collapsed frames (%)',True),('mean_compact_rg_nm_mean','mean_compact_rg_nm_std',r'Compact $R_g$ (nm)',False)]
    for k,(m,sd,lab,high) in enumerate(metrics,1):
        ax=axs[k]
        for i,r in enumerate(sub.itertuples()):
            val=float(getattr(r,m));err=float(getattr(r,sd)) if pd.notna(getattr(r,sd)) else 0
            ax.hlines(i,val-err,val+err,color='#a8bac6',lw=2,zorder=2)
            ax.scatter([val-err,val+err],[i,i],s=12,color='#a8bac6',zorder=3)
            ax.scatter(val,i,s=48,color='#bd8300' if r.design_id==835 else '#236f93',edgecolor='white',lw=.6,zorder=4)
        ax.axvline(float(reference[m]),color='#bd8300',ls='--',lw=1.1)
        ax.set_xlabel(lab+'\n'+('Higher is favourable' if high else 'Lower is favourable'))
        ax.set_title(chr(64+k)+'  '+lab,fontsize=11,pad=12)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        values=sub[m].to_numpy(float);errs=sub[sd].fillna(0).to_numpy(float)
        spread=max(np.ptp(values+errs),np.ptp(values-errs),.2)
        ax.set_xlim(np.min(values-errs)-.14*spread,np.max(values+errs)+.14*spread)
    fig.suptitle(title,fontsize=14,y=.995)
    fig.subplots_adjust(left=.24,right=.99,top=.89,bottom=.17,wspace=.17)
    fig.legend(handles=[Patch(color=COLORS[m],label={'VG':'VPGVG','IP':'IPAVG','KG':'VPGKG'}[m]) for m in COLORS],loc='lower left',bbox_to_anchor=(.24,.01),ncol=3,frameon=False)
    fig.text(.48,.055,'Point = replicate mean; whiskers = ±1 between-replicate SD; dashed line = adapted Tk8 (835).',fontsize=8)
    save(fig,out,name,formats,dpi)

def plot_combined(d, panels, name, out, formats, dpi, reference, meta):
    """Two controlled comparisons in one publication figure, sharing metric columns."""
    groups=[]
    for heading,ids in panels:
        sub=d.set_index('design_id').reindex(ids).dropna(subset=['mean_rg_nm']).reset_index()
        if len(sub)!=len(ids):
            raise ValueError(f'Missing designs in {name}: {set(ids)-set(sub.design_id)}')
        groups.append((heading,sub))
    counts=[len(sub) for _,sub in groups]
    fig=plt.figure(figsize=(21.5,max(8,0.50*sum(counts)+3.2)))
    gs=fig.add_gridspec(2,5,width_ratios=[3.5,1.2,1.2,1.2,1.2],height_ratios=[c+1.4 for c in counts],hspace=.49,wspace=.23)
    metrics=[('kescape_ns_inv_mean','kescape_ns_inv_std',r'$k_{escape}$ (ns$^{-1}$)',False),
             ('kcollapse_ns_inv_mean','kcollapse_ns_inv_std',r'$k_{collapse}$ (ns$^{-1}$)',True),
             ('collapsed_percent','std_collapsed_percent_between_reps','Collapsed frames (%)',True),
             ('mean_compact_rg_nm_mean','mean_compact_rg_nm_std',r'Compact $R_g$ (nm)',False)]
    # Use common limits for the same metric across the two panels.
    limits=[]
    for m,sd,_,_ in metrics:
        v=pd.concat([sub[m] for _,sub in groups]).to_numpy(float)
        e=pd.concat([sub[sd].fillna(0) for _,sub in groups]).to_numpy(float)
        spread=max(np.ptp(v+e),np.ptp(v-e),.1)
        limits.append((np.min(v-e)-.13*spread,np.max(v+e)+.13*spread))
    for pi,(heading,sub) in enumerate(groups):
        n=len(sub); y=np.arange(n)
        axes=[fig.add_subplot(gs[pi,j]) for j in range(5)]
        for ax in axes:
            for i in range(n):
                if i%2==0:ax.axhspan(i-.5,i+.5,color='#f4f4f4',zorder=0)
            ax.spines[['top','right']].set_visible(False)
            ax.set_ylim(n-.5,-.5)
        ax=axes[0]
        for i,r in enumerate(sub.itertuples()):
            seq=meta.get(int(r.design_id))
            if seq is None:
                ax.add_patch(Rectangle((0,i-.19),84,.38,facecolor='#dedede',edgecolor='#888',hatch='///',lw=.4))
                ax.text(42,i,'Sequence not verified',ha='center',va='center',fontsize=7)
                continue
            st=0
            for j in range(1,len(seq)+1):
                if j==len(seq) or seq[j]!=seq[st]:
                    ax.add_patch(Rectangle((st,i-.19),j-st,.38,facecolor=COLORS[seq[st]],edgecolor='none'))
                    st=j
        ax.set_xlim(-1,85);ax.set_xticks([0,21,42,63,84]);ax.set_xlabel('Motif position (84 pentapeptides)',fontsize=9)
        ax.set_yticks(y,[f'{int(r.design_id)}  {r.design_name.replace("_"," ")}' for r in sub.itertuples()],fontsize=9)
        ax.tick_params(axis='y',length=0)
        ax.set_title(heading,loc='left',fontsize=11,fontweight='bold',pad=15)
        for k,(m,sd,lab,high) in enumerate(metrics):
            ax=axes[k+1]
            for i,r in enumerate(sub.itertuples()):
                val=float(getattr(r,m));err=float(getattr(r,sd)) if pd.notna(getattr(r,sd)) else 0
                ax.hlines(i,val-err,val+err,color='#a8bac6',lw=2,zorder=2)
                ax.scatter([val-err,val+err],[i,i],s=11,color='#a8bac6',zorder=3)
                ax.scatter(val,i,s=46,color='#bd8300' if int(r.design_id)==835 else '#236f93',edgecolor='white',lw=.6,zorder=4)
            ax.axvline(float(reference[m]),color='#bd8300',ls='--',lw=1.1)
            ax.set_xlim(*limits[k]);ax.set_yticks(y);ax.tick_params(axis='y',left=False,labelleft=False)
            ax.set_title(chr(65+k)+'  '+lab,fontsize=10,pad=12)
            ax.set_xlabel(('Higher' if high else 'Lower')+' is favourable',fontsize=9)
            ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(name.replace('_',' '),fontsize=14,y=.995)
    fig.subplots_adjust(left=.24,right=.99,top=.93,bottom=.12)
    fig.legend(handles=[Patch(color=COLORS[m],label={'VG':'VPGVG','IP':'IPAVG','KG':'VPGKG'}[m]) for m in COLORS],loc='lower left',bbox_to_anchor=(.24,.015),ncol=3,frameon=False)
    fig.text(.51,.045,'Points: replicate means; whiskers: ±1 between-replicate SD; dashed line: adapted Tk8 (835).',fontsize=8)
    save(fig,out,'DR8_'+name,out_formats(formats),dpi)

def out_formats(formats):
    return formats

def plot_heatmap(d,out,formats,dpi):
    n=len(d)
    fig,axs=plt.subplots(1,4,figsize=(14,max(5,.36*n+2)),sharey=True)
    metrics=[
      ('kescape_ns_inv_mean',r'$k_{escape}$ (ns$^{-1}$)',False),
      ('kcollapse_ns_inv_mean',r'$k_{collapse}$ (ns$^{-1}$)',True),
      ('collapsed_percent','Collapsed frames (%)',True),
      ('mean_compact_rg_nm_mean',r'Mean compact $R_g$ (nm)',False)]
    for ax,(m,label,high) in zip(axs,metrics):
        v=d[m].to_numpy(float)
        finite=v[np.isfinite(v)]
        if not len(finite):continue
        lo,hi=float(finite.min()),float(finite.max())
        norm=matplotlib.colors.Normalize(vmin=lo,vmax=hi if hi>lo else lo+1)
        cmap='viridis' if high else 'viridis_r'
        ax.imshow(v[:,None],aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
        for i,value in enumerate(v):
            if not np.isfinite(value):continue
            rgba=plt.get_cmap(cmap)(norm(value))
            brightness=np.dot(rgba[:3],[.299,.587,.114])
            ax.text(0,i,f'{value:.2f}' if m=='collapsed_percent' else f'{value:.4f}',ha='center',va='center',fontsize=8,color='black' if brightness>.55 else 'white')
            if int(d.iloc[i].design_id)==835:ax.add_patch(Rectangle((-.49,i-.49),.98,.98,fill=False,edgecolor='#e25f30',lw=2))
        ax.set_xticks([]);ax.set_title(label+'\n'+('Higher is favourable' if high else 'Lower is favourable'),fontsize=10)
        ax.set_yticks(np.arange(n),[f'{int(r.design_id)}  {r.design_name.replace("_"," ")}' for r in d.itertuples()],fontsize=9)
        ax.tick_params(axis='y',length=0)
    fig.suptitle('DR8 VGIPKG: four-objective optimization (replicate means)',fontsize=14)
    fig.subplots_adjust(left=.36,right=.98,top=.93,bottom=.07,wspace=.10)
    fig.text(.36,.025,'Independent raw-value colour scale per column; yellow = favourable. Orange outline: adapted Tk8.',fontsize=9)
    save(fig,out,'DR8_S02_architecture_heatmap',formats,dpi)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--outdir',type=Path,default=None)
    p.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf','svg'])
    p.add_argument('--dpi',type=int,default=200)
    p.add_argument('--temperature',default='370K',help='Temperature to plot, e.g. 370K')
    args=p.parse_args();indir=args.indir;out=args.outdir or indir/f'DR8_thesis_plots_{args.temperature}';out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(indir/'combined_summary_mean_std.csv')
    if 'temp' not in d.columns: p.error('Temperature missing: rerun temperature-fixed summary script')
    d=d[d.temp.astype(str).eq(args.temperature)].copy()
    if d.empty: p.error(f'No architectures at {args.temperature}')
    md=pd.read_csv(indir/'design_metadata.csv')
    required={'design_id','design_name','collapsed_percent','mean_rg_nm','std_collapsed_percent_between_reps','std_rg_between_reps_nm'}
    if missing:=required-set(d.columns):p.error(f'Summary missing columns: {sorted(missing)}')
    if len(d)!=30:p.error(f'Expected 30 architectures at {args.temperature}; found {len(d)}')
    if 835 not in set(d.design_id):p.error('Reference 835 missing from summary')
    if d.design_id.duplicated().any():p.error('Duplicate design IDs')
    audit=[];seqmap={}
    for r in md.itertuples():
        try:seq=expand(r.motif_pattern)
        except ValueError:seq=None
        counts={m:seq.count(m) if seq else np.nan for m in COLORS}
        valid=seq is not None and counts=={'VG':38,'IP':38,'KG':8}
        seqmap[int(r.design_id)]=seq if valid else None
        audit.append(dict(design_id=r.design_id,design_name=r.design_name,motif_pattern=r.motif_pattern,**counts,composition_matches=valid,schematic_verified=valid))
    pd.DataFrame(audit).to_csv(out/'architecture_definition_audit.csv',index=False)
    invalid=[int(a['design_id']) for a in audit if not a['composition_matches']]
    if invalid:print('WARNING: no verified schematic for designs:',invalid,'(see architecture_definition_audit.csv)')
    # Equal-weight average rank of four objectives, not a fitted physical score.
    objectives=[('kescape_ns_inv_mean',True),('kcollapse_ns_inv_mean',False),
                ('collapsed_percent',False),('mean_compact_rg_nm_mean',True)]
    for m,ascending in objectives:
        if m not in d.columns: p.error(f'Missing kinetic metric {m}; run create_vgipkg_summary_kinetics.py first')
        d['rank_'+m]=d[m].rank(ascending=ascending,method='average')
    d['balanced_rank']=d[['rank_'+m for m,_ in objectives]].mean(axis=1)
    d.sort_values('balanced_rank').to_csv(out/'ranking_four_objectives.csv',index=False)
    ref=d[d.design_id.eq(835)].iloc[0]
    d=d.sort_values(['balanced_rank','design_id']).reset_index(drop=True)
    d.to_csv(out/'ranking_collapse_fraction.csv',index=False)
    plot_story(d,'All VGIPKG architectures: ranked by four-objective score','DR8_S01_all_architectures_schematics',out,args.formats,args.dpi,ref,seqmap)
    plot_heatmap(d,out,args.formats,args.dpi)
    for key,panels in COMPARISON_FIGURES.items():
        for pi,(heading,ids) in enumerate(panels,1):
            sub=d.set_index('design_id').reindex(ids).dropna(subset=['mean_rg_nm']).reset_index()
            sub.to_csv(out/f'DR8_{key}_panel_{pi}_values.csv',index=False)
        plot_combined(d,panels,key,out,args.formats,args.dpi,ref,seqmap)
    (out/'plot_settings.json').write_text(json.dumps(dict(threshold_nm=float(d.threshold_nm.iloc[0]) if 'threshold_nm' in d else None,reference=835,comparisons=COMPARISON_FIGURES,metric_definitions={'kescape_ns_inv_mean':'observed compact-to-expanded transitions / compact exposure time','kcollapse_ns_inv_mean':'observed expanded-to-compact transitions / expanded exposure time','collapsed_percent':'mean of replicate collapsed percentages','mean_compact_rg_nm_mean':'mean of replicate compact-state mean Rg','balanced_rank':'equal-weight mean of four objective ranks'},whiskers='between-replicate standard deviation; not confidence intervals',schematics='only shown if supplied motif pattern expands to 38 VG, 38 IP, 8 KG'),indent=2),encoding='utf-8')
    print(f'Created {2+len(COMPARISON_FIGURES)} DR8 thesis figures in {out}')

if __name__=='__main__':main()
