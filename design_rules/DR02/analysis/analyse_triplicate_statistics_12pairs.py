"""Replicate-aware DR2 architecture statistics for all twelve 1:1 motif pairs.

Requires: numpy pandas scipy statsmodels matplotlib
Input: three per_run_summary.csv files under --root (or --input FILE repeated).
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

ROOT = Path(r'E:\Master thesis\calvados_tea\plotting_thesis\design_rule_2')
FOLDERS = ('lib1R4_triplicate_stride1', '907_930_3reps_stride1', '931_954_triplicate_stride1')
LAM = {'VPAVG':.6294581781012,'VPGNG':.6527600406134,'VPGAG':.6568240708464,
       'VPGVG':.6850471057402,'VPGGG':.7124129984854,'IPAVG':.7430614622074,
       'VPGIG':.7986503898464,'VPGLG':.8147436293176,'VPGFG':.8506676135856}
PAIRS = {'IGLG':('VPGIG','VPGLG'),'AVGAG':('VPAVG','VPGAG'),
         'IGIP':('VPGIG','IPAVG'),'VGAVG':('VPGVG','VPAVG'),
         'VGIP':('VPGVG','IPAVG'),'GGNG':('VPGGG','VPGNG'),
         'LGIP':('VPGLG','IPAVG'),'GGAVG':('VPGGG','VPAVG'),
         'VAIP':('VPAVG','IPAVG'),'VGIG':('VPGVG','VPGIG'),
         'VAIG':('VPAVG','VPGIG'),'AVGFG':('VPAVG','VPGFG')}
METRICS = ('rg_range_90_nm','iqr_rg_nm','fraction_collapsed',
           'mean_rg_nm','mean_rg_compact_nm','mean_collapse_depth_nm',
           'k_collapse_per_ns','k_escape_per_ns')
GREEN, BLUE = '#90F018', '#245D83'


def get_pair(row):
    # New triplicate sets have fixed, known eight-ID blocks.
    try:
        i = int(float(row.get('id')))
        if 907 <= i <= 930:
            return ('AVGFG','AVGAG','IGLG')[(i-907)//8]
        if 931 <= i <= 954:
            return ('LGIP','GGAVG','GGNG')[(i-931)//8]
    except (TypeError, ValueError):
        pass
    for col in ('group','pair','motif_pair','name','architecture','architecture_type','system'):
        text = str(row.get(col, '')).upper()
        for pair in sorted(PAIRS, key=len, reverse=True):
            if pair in text:
                return pair
    return None


def get_arch(row):
    # ID identifies each unique sequence, including directional architectures.
    return str(row['id']).strip()


def analyse(groups):
    vals = np.concatenate(groups)
    k, n = len(groups), len(vals)
    grand = vals.mean()
    ssb = sum(len(g)*(g.mean()-grand)**2 for g in groups)
    ssw = sum(((g-g.mean())**2).sum() for g in groups)
    msw = ssw/(n-k)
    sst = ssb+ssw
    omega = max(0., (ssb-(k-1)*msw)/(sst+msw)) if sst+msw>0 else np.nan
    means = np.array([g.mean() for g in groups])
    spread = float(means.max()-means.min())
    sd = float(np.sqrt(msw))
    try:
        w = anova_oneway(groups,use_var='unequal',welch_correction=True)
        p = float(w.pvalue)
        F = float(w.statistic)
        df_denom = float(w.df_denom)
    except Exception as exc:
        warnings.warn(f'Welch test failed: {exc}')
        p,F,df_denom = np.nan,np.nan,np.nan
    try:
        t = pairwise_tukeyhsd(vals,np.concatenate([np.repeat(str(i),len(g)) for i,g in enumerate(groups)]))
        tukey_n = int(np.sum(t.reject))
    except Exception:
        tukey_n = np.nan
    return dict(omega_squared=omega, welch_p=p, welch_F=F, welch_df2=df_denom,
                architecture_spread=spread,pooled_within_arch_sd=sd,
                spread_to_noise=spread/sd if sd>0 else np.nan,
                n_significant_tukey=tukey_n,n_possible_tukey=k*(k-1)//2,
                n_architectures=k,n_runs=n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--input',type=Path,action='append',help='Override default sources; repeat for multiple CSVs')
    parser.add_argument('--outdir',type=Path,default=None)
    parser.add_argument('--temp',default='370K')
    parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    out=args.outdir or args.root/'DR2_statistics_12pairs'
    out.mkdir(parents=True,exist_ok=True)
    sources=args.input or [args.root/f/'per_run_summary.csv' for f in FOLDERS]
    frames=[]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f'Missing input: {source}')
        frame=pd.read_csv(source)
        frame['source_file']=str(source)
        frames.append(frame)
        print(f'Loaded {len(frame)} rows: {source}')
    df=pd.concat(frames,ignore_index=True)
    for c in ('id','replicate'):
        if c not in df:
            raise ValueError(f'Missing required column: {c}')
    if 'temp' in df:
        df=df[df.temp.astype(str).str.contains(args.temp.replace('K',''),regex=False)].copy()
    df['pair']=df.apply(get_pair,axis=1)
    missing_pair=df.pair.isna().sum()
    if missing_pair:
        warnings.warn(f'{missing_pair} rows have no recognised pair and will be excluded')
    df=df[df.pair.notna()].copy()
    df['arch_id']=df.apply(get_arch,axis=1)
    df['replicate']=df.replicate.astype(str)
    # Never count duplicate rows from overlapping input sources as independent replicates.
    dup=df.duplicated(['pair','arch_id','replicate'],keep=False)
    if dup.any():
        examples=df.loc[dup,['pair','arch_id','replicate','source_file']].head(12)
        raise ValueError(f'Duplicate run identities across input files:\n{examples.to_string(index=False)}')
    counts=df.groupby(['pair','arch_id']).replicate.nunique().rename('n_replicates').reset_index()
    counts.to_csv(out/'DR2_replicate_counts.csv',index=False)
    summary=counts.groupby('pair').agg(n_architectures=('arch_id','nunique'),
                                       min_reps=('n_replicates','min'),max_reps=('n_replicates','max')).reset_index()
    summary.to_csv(out/'DR2_completeness.csv',index=False)
    print('\nCompleteness:\n'+summary.to_string(index=False))
    missing=set(PAIRS)-set(summary.pair)
    if missing:
        raise ValueError(f'Missing motif pairs: {sorted(missing)}. Check group labels and sources.')
    invalid=summary[(summary.n_architectures!=8)|(summary.min_reps!=3)|(summary.max_reps!=3)]
    if not invalid.empty and not args.allow_incomplete:
        raise ValueError('Expected eight architectures and three replicates for every pair. '
                         'See DR2_completeness.csv; use --allow-incomplete only for exploration.')
    available=[m for m in METRICS if m in df.columns]
    print('\nAvailable metrics:',', '.join(available))
    if not available:
        raise ValueError('No supported metric columns found. Available: '+str(df.columns.tolist()))
    rows=[]
    for metric in available:
        for pair,(a,b) in PAIRS.items():
            sub=df[df.pair==pair]
            groups=[g[metric].dropna().to_numpy(dtype=float) for _,g in sub.groupby('arch_id')]
            groups=[g[np.isfinite(g)] for g in groups]
            if len(groups)<2 or any(len(g)<2 for g in groups):
                warnings.warn(f'Skipping {pair} / {metric}: insufficient valid replicate values')
                continue
            d=analyse(groups)
            d.update(pair=pair,motif_pair=f'{a}–{b}',metric=metric,
                     delta_lambda=abs(LAM[a]-LAM[b]),mean_lambda=(LAM[a]+LAM[b])/2)
            rows.append(d)
    result=pd.DataFrame(rows)
    result['welch_p_fdr']=np.nan
    for metric in result.metric.unique():
        idx=result.index[(result.metric==metric)&result.welch_p.notna()]
        result.loc[idx,'welch_p_fdr']=multipletests(result.loc[idx,'welch_p'],method='fdr_bh')[1]
    result['significant']=result.welch_p_fdr<.05
    result['point_color']=np.where(result.significant,GREEN,BLUE)
    result=result.sort_values(['metric','delta_lambda','pair'])
    result.to_csv(out/'DR2_global_architecture_statistics_12pairs.csv',index=False)
    for metric in result.metric.unique():
        r=result[result.metric==metric]
        r.to_csv(out/f'DR2_{metric}_statistics.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,6.5))
        for _,v in r.iterrows():
            ax.scatter(v.delta_lambda,v.omega_squared,s=220,color=v.point_color,edgecolor='white',linewidth=1.4,zorder=3)
            ax.annotate(v.pair,(v.delta_lambda,v.omega_squared),xytext=(6,6),textcoords='offset points',fontsize=11)
        ax.set(xlabel=r'Motif stickiness contrast, $\Delta\bar{\lambda}$',ylabel=r'Architecture effect size, $\omega^2$',ylim=(-.035,1.04))
        ax.xaxis.label.set_fontweight('bold');ax.yaxis.label.set_fontweight('bold')
        ax.grid(alpha=.16)
        fig.tight_layout()
        fig.savefig(out/f'DR2_{metric}_omega2_12pairs.pdf',bbox_inches='tight')
        fig.savefig(out/f'DR2_{metric}_omega2_12pairs.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    print('\nRESULTS (central 90% Rg):')
    if 'rg_range_90_nm' in available:
        r=result[result.metric=='rg_range_90_nm']
        print(r[['pair','delta_lambda','omega_squared','welch_p_fdr','significant','n_significant_tukey']].to_string(index=False,float_format=lambda x:f'{x:.4g}'))
    print('\nSaved to:',out)

if __name__=='__main__':
    main()
