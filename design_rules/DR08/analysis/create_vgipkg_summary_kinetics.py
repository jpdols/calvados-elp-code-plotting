#!/usr/bin/env python3
"""Summarize three CALVADOS all_rg_values.csv replicates. Python >=3.9; pandas, numpy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
"""Design Rule 8 VGIPKG metadata (motif notation transcribed from supplied table)."""
DESIGNS = [
(833,'random_shuffle','Random permutation of 38 VG, 38 IP and 8 KG','random'),
(834,'even_distribution','Evenly dispersed VG, IP and KG','distributed'),
(835,'original_Tk8_adapted','(KG)1-(IP)16-(VG3-KG-VG3-KG-VG3-KG-VG2-IP-VG3-IP-VG3-IP-VG4-IP-VG3-IP-VG3-IP-VG2-KG-VG3-KG-VG3-KG-VG3)-(IP)16-(KG)1','reference'),
(836,'central_IP_KG_endcaps','(KG)4-(VG)19-(IP)38-(VG)19-(KG)4','endcaps'),
(837,'central_IP_KG_adjacent','(VG)19-(KG)4-(IP)38-(KG)4-(VG)19','interface'),
(838,'KG_embedded_in_IP','(VG)19-(IP)15-(KG)4-(IP)8-(KG)4-(IP)15-(VG)19','embedded'),
(839,'KG_dispersed_in_IP','(VG)19-(IP4-KG)8-(IP)6-(VG)19','distributed'),
(840,'central_mixed_sticker_domain','(VG)19-(IP4-KG)4-(IP)6-(KG-IP4)4-(VG)19','embedded'),
(841,'two_IP_blocks_KG_endcaps','(KG)4-(VG)10-(IP)19-(VG)18-(IP)19-(VG)10-(KG)4','endcaps'),
(842,'two_IP_blocks_interfacial_KG','(VG)10-(IP)19-(KG)4-(VG)18-(KG)4-(IP)19-(VG)10','interface'),
(843,'two_IP_blocks_KG_center','(VG)10-(IP)19-(VG)9-(KG)8-(VG)9-(IP)19-(VG)10','center'),
(844,'IP_endcaps_KG_center','(IP)19-(VG)19-(KG)8-(VG)19-(IP)19','center'),
(845,'KG_capped_IP_endblocks','(KG)4-(IP)19-(VG)38-(IP)19-(KG)4','endcaps'),
(846,'IP_outermost_KG_inside','(IP)19-(KG)4-(VG)38-(KG)4-(IP)19','interface'),
(847,'three_medium_IP_blocks_KG_endcaps','(KG)4-(VG)6-(IP)13-(VG)13-(IP)12-(VG)13-(IP)13-(VG)6-(KG)4','endcaps'),
(848,'three_IP_blocks_interfacial_KG','(VG)6-(IP)13-(KG)2-(VG)13-(KG)2-(IP)12-(KG)2-(VG)13-(KG)2-(IP)13-(VG)6','interface'),
(849,'three_IP_blocks_KG_center','(VG)6-(IP)13-(VG)13-(IP)12-(KG)8-(IP)13-(VG)19','center'),
(850,'four_IP_blocks_KG_endcaps','(KG)4-(VG)5-(IP)10-(VG)9-(IP)9-(VG)10-(IP)9-(VG)9-(IP)10-(VG)5-(KG)4','endcaps'),
(851,'IP_one_end_KG_opposite','(IP)38-(VG)38-(KG)8','asymmetric'),
(852,'KG_then_IP_same_end','(VG)38-(KG)8-(IP)38','asymmetric'),
(853,'IP_then_KG_same_end','(VG)38-(IP)38-(KG)8','asymmetric'),
(854,'asymmetric_IP25_IP13_KG_endcaps','(KG)4-(IP)25-(VG)19-(IP)13-(VG)19-(KG)4','endcaps'),
(855,'charged_large_IP_domain','(IP)25-(KG)8-(VG)19-(IP)13-(VG)19','embedded'),
(856,'KG_one_end_symmetric_IP','(KG)8-(VG)10-(IP)19-(VG)18-(IP)19-(VG)10','asymmetric'),
(857,'hydrophobic_gradient','(KG)8-(VG)13-(IP)3-(VG)10-(IP)5-(VG)7-(IP)7-(VG)5-(IP)10-(VG)3-(IP)13','gradient'),
(858,'symmetric_IP_gradient','(KG)4-(VG)8-(IP)4-(VG)6-(IP)6-(VG)5-(IP)18-(VG)5-(IP)6-(VG)6-(IP)4-(VG)8-(KG)4','gradient'),
(859,'single_IP_block_KG_center','(VG)19-(IP)19-(KG)8-(IP)19-(VG)19','center'),
(860,'KG_clustered_in_VG','(VG)9-(KG)4-(VG)10-(IP)38-(VG)10-(KG)4-(VG)9','embedded'),
(861,'KG_dispersed_in_VG','(VG4-KG)4-(VG)3-(IP)38-(VG)3-(KG-VG4)4','distributed'),
(862,'four_IP_blocks_interfacial_KG','(VG)5-(IP)10-(KG)2-(VG)9-(IP)9-(KG)2-(VG)10-(KG)2-(IP)9-(VG)9-(KG)2-(IP)10-(VG)5','interface'),
]

def metadata():
    import pandas as pd
    return pd.DataFrame([dict(design_id=i,architecture=f'{i}_VGIPKG_R2_{name}',design_name=name,motif_pattern=pattern,design_group=group,VG_nominal=38,IP_nominal=38,KG_nominal=8) for i,name,pattern,group in DESIGNS])



DEFAULT = [r'E:\Master thesis\calvados_tea\analysis_library_2\VGIPKG\all_rg_values.csv',r'E:\Master thesis\calvados_tea\analysis_library_2\VGIPKG_rep02\all_rg_values.csv',r'E:\Master thesis\calvados_tea\analysis_library_2\VGIPKG_rep03\all_rg_values.csv']

def pick(cols, candidates):
    norm = {str(c).lower().strip().replace(' ','_'): c for c in cols}
    for x in candidates:
        if x in norm: return norm[x]
    return None

def load(path, rep, arch_col=None, rg_col=None):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    a = arch_col or pick(df.columns, ['architecture','architecture_id','arch','design','sequence_name','sequence','name','protein','run','label','id'])
    r = rg_col or pick(df.columns, ['rg','rg_nm','rg_value','radius_of_gyration','radius_of_gyration_nm','rg_(nm)','rg_values'])
    if a and r:
        out = df[[c for c in [a,r,'time_ns','original_frame'] if c in df.columns]].rename(columns={a:'architecture',r:'rg_nm'})
    elif a and not r:
        # Rows are time frames; architecture columns are numeric.
        numeric = [c for c in df if c != a and pd.to_numeric(df[c],errors='coerce').notna().mean() > .8]
        if len(numeric) != 1: raise ValueError(f'Cannot determine Rg column in {path}; columns={list(df.columns)}. Use --rg-col.')
        out = df[[a,numeric[0]]].rename(columns={a:'architecture',numeric[0]:'rg_nm'})
    elif not a and not r:
        # Wide format: one architecture per column; drop common time/frame columns.
        skip = {'frame','time','time_ns','step','index','unnamed:_0'}
        cols = [c for c in df if c.lower().replace(' ','_') not in skip and pd.to_numeric(df[c],errors='coerce').notna().mean() > .8]
        if not cols: raise ValueError(f'Cannot infer format of {path}. Columns: {list(df.columns)}')
        out = df[cols].melt(var_name='architecture',value_name='rg_nm')
    else:
        raise ValueError(f'Ambiguous columns in {path}: {list(df.columns)}. Supply --arch-col and --rg-col if long format.')
    out['rg_nm'] = pd.to_numeric(out['rg_nm'],errors='coerce')
    out = out.dropna(subset=['architecture','rg_nm'])
    out['architecture'] = out['architecture'].astype(str).str.strip()
    out = out[np.isfinite(out.rg_nm) & (out.rg_nm > 0)].copy()
    out['replicate'] = rep
    if out.empty: raise ValueError(f'No valid Rg values in {path}')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',nargs=3,default=DEFAULT)
    p.add_argument('--outdir',default=str(Path(__file__).resolve().parent))
    p.add_argument('--threshold',type=float,default=4.61)
    p.add_argument('--arch-col',default=None);p.add_argument('--rg-col',default=None)
    p.add_argument('--save-raw',action='store_true',help='Also save concatenated per-frame Rg (potentially large)')
    args=p.parse_args(); out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    parts=[]
    for i,src in enumerate(args.inputs,1):
        path=Path(src)
        if not path.is_file(): raise FileNotFoundError(f'Missing replicate {i}: {path}')
        x=load(path,f'rep{i:02d}',args.arch_col,args.rg_col)
        print(f'{path.name} replicate {i}: {len(x):,} valid Rg values; {x.architecture.nunique()} architectures')
        parts.append(x)
    raw=pd.concat(parts,ignore_index=True)
    meta=metadata()
    # Map either full architecture names or numeric IDs onto the canonical design labels.
    import re
    lookup={str(row.design_id):row.architecture for row in meta.itertuples()}
    def canonical(value):
        name=str(value).strip()
        match=re.match(r'^(\d{3})(?:_|$)',name)
        if match and match.group(1) in lookup: return lookup[match.group(1)]
        return name
    raw['architecture']=raw.architecture.map(canonical)
    unknown=sorted(set(raw.architecture)-set(meta.architecture))
    if unknown: print('WARNING: Unrecognized architectures:',unknown)
    absent=sorted(set(meta.architecture)-set(raw.architecture))
    if absent: print('WARNING: Designs missing from all replicates:',absent)
    meta.to_csv(out/'design_metadata.csv',index=False)

    raw['collapsed']=raw.rg_nm < args.threshold
    rep=raw.groupby(['architecture','replicate'],sort=True).agg(n_frames=('rg_nm','size'),mean_rg_nm=('rg_nm','mean'),std_rg_nm=('rg_nm','std'),median_rg_nm=('rg_nm','median'),collapsed_fraction=('collapsed','mean')).reset_index()

    # Frame-based kinetic estimates. Rates = observed transitions / state exposure time.
    # A transition is only counted for adjacent frames in the original trajectory.
    kinetics=[]
    for (arch,repname),g in raw.groupby(['architecture','replicate'],sort=False):
        g=g.sort_values('original_frame' if 'original_frame' in g else 'time_ns' if 'time_ns' in g else 'rg_nm').copy()
        if 'time_ns' not in g:
            raise ValueError('Kinetic analysis requires time_ns in all_rg_values.csv')
        t=g.time_ns.to_numpy(dtype=float)
        z=g.collapsed.to_numpy(dtype=bool)
        dt=np.diff(t)
        positive=dt[np.isfinite(dt)&(dt>0)]
        if not len(positive):raise ValueError(f'No positive time increments for {arch}, {repname}')
        typical=float(np.median(positive))
        adjacent=np.isfinite(dt)&(dt>0)&(dt<=1.5*typical)
        if 'original_frame' in g:
            frame_step=np.diff(g.original_frame.to_numpy(dtype=float))
            frame_pos=frame_step[frame_step>0]
            if len(frame_pos):
                adjacent &= frame_step==np.median(frame_pos)
        exits=int(np.sum(adjacent & z[:-1] & ~z[1:]))
        entries=int(np.sum(adjacent & ~z[:-1] & z[1:]))
        compact_exposure=float(np.sum(dt[adjacent & z[:-1]]))
        expanded_exposure=float(np.sum(dt[adjacent & ~z[:-1]]))
        compact_rg=g.loc[g.collapsed,'rg_nm']
        kinetics.append(dict(architecture=arch,replicate=repname,
          kescape_ns_inv=exits/compact_exposure if compact_exposure>0 else np.nan,
          kcollapse_ns_inv=entries/expanded_exposure if expanded_exposure>0 else np.nan,
          n_escapes=exits,n_collapse_events=entries,
          compact_exposure_ns=compact_exposure,expanded_exposure_ns=expanded_exposure,
          mean_compact_rg_nm=compact_rg.mean() if len(compact_rg) else np.nan,
          n_compact_frames=len(compact_rg),median_step_ns=typical))
    kinetic_rep=pd.DataFrame(kinetics)
    kinetic_rep.to_csv(out/'kinetic_replicate_summary.csv',index=False)
    rep=rep.merge(kinetic_rep,on=['architecture','replicate'],how='left')


    # Complete descriptive and dwell-time characterization (per replicate).
    extra=[]
    dwell_records=[]
    for (arch,repname),g in raw.groupby(['architecture','replicate'],sort=False):
        g=g.sort_values('time_ns',kind='stable')
        t=g.time_ns.to_numpy(float); r=g.rg_nm.to_numpy(float); z=g.collapsed.to_numpy(bool)
        dt=np.diff(t); good=dt[np.isfinite(dt)&(dt>0)]
        step=float(np.median(good))
        # Split at missing/non-consecutive time steps as well as state changes.
        connected=np.isfinite(dt)&(dt>0)&(dt<=1.5*step)
        if 'original_frame' in g:
            frames=np.diff(g.original_frame.to_numpy(float))
            pos=frames[frames>0]
            if len(pos):connected &= np.isclose(frames,np.median(pos))
        boundaries=np.r_[0,np.where((~connected)|(z[:-1]!=z[1:]))[0]+1,len(z)]
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            # Run duration includes one sample interval; boundaries are censored.
            duration=(t[end-1]-t[start]+step) if end>start else 0
            dwell_records.append(dict(architecture=arch,replicate=repname,
                state='compact' if z[start] else 'expanded',start_time_ns=t[start],
                end_time_ns=t[end-1],duration_ns=duration,n_frames=end-start,
                left_censored=start==0 or not connected[start-1],
                right_censored=end==len(z) or not connected[end-1]))
        compact=r[z];expanded=r[~z]
        def state_stats(prefix,x):
            return {f'{prefix}_{k}':v for k,v in dict(n_frames=len(x),
                mean_rg_nm=float(np.mean(x)) if len(x) else np.nan,
                std_rg_nm=float(np.std(x,ddof=1)) if len(x)>1 else np.nan,
                median_rg_nm=float(np.median(x)) if len(x) else np.nan).items()}
        extra.append(dict(architecture=arch,replicate=repname,
            rg_q05_nm=float(np.quantile(r,.05)),rg_q25_nm=float(np.quantile(r,.25)),
            rg_q75_nm=float(np.quantile(r,.75)),rg_q95_nm=float(np.quantile(r,.95)),
            iqr_rg_nm=float(np.quantile(r,.75)-np.quantile(r,.25)),
            central90_rg_nm=float(np.quantile(r,.95)-np.quantile(r,.05)),
            expanded_fraction=float(np.mean(~z)),**state_stats('compact',compact),
            **state_stats('expanded',expanded)))
    dwell=pd.DataFrame(dwell_records)
    dwell.to_csv(out/'dwell_events.csv',index=False)
    extra=pd.DataFrame(extra)
    dwell_stats=[]
    for (arch,repname),g in dwell.groupby(['architecture','replicate']):
        record=dict(architecture=arch,replicate=repname)
        for state in ['compact','expanded']:
            x=g[g.state==state]
            complete=x[~(x.left_censored|x.right_censored)].duration_ns.to_numpy(float)
            record.update({f'{state}_n_dwells':len(x),
                f'{state}_n_complete_dwells':len(complete),
                f'{state}_mean_dwell_ns':float(np.mean(complete)) if len(complete) else np.nan,
                f'{state}_median_dwell_ns':float(np.median(complete)) if len(complete) else np.nan,
                f'{state}_max_dwell_ns':float(np.max(complete)) if len(complete) else np.nan})
        dwell_stats.append(record)
    rep=rep.merge(extra,on=['architecture','replicate']).merge(
        pd.DataFrame(dwell_stats),on=['architecture','replicate'])

    rep['collapsed_percent']=rep.collapsed_fraction*100
    rep=rep.merge(metadata(),on='architecture',how='left')
    rep.to_csv(out/'replicate_summary.csv',index=False)
    rep.to_csv(out/'complete_replicate_summary.csv',index=False)
    # Equal weight per replicate, even if trajectory lengths differ.
    s=rep.groupby('architecture').agg(n_replicates=('replicate','nunique'),n_frames_total=('n_frames','sum'),mean_rg_nm=('mean_rg_nm','mean'),std_rg_between_reps_nm=('mean_rg_nm','std'),mean_within_rep_std_rg_nm=('std_rg_nm','mean'),median_rg_nm=('median_rg_nm','mean'),collapsed_fraction=('collapsed_fraction','mean'),std_collapsed_fraction_between_reps=('collapsed_fraction','std')).reset_index()
    s=s.merge(meta,on='architecture',how='left')

    # Every numeric replicate-level metric gets mean, SD, min and max across replicates.
    # Do not average identifiers, threshold, nominal composition or replicate counts.
    numeric=[c for c in rep.select_dtypes(include=[np.number]).columns
             if c not in {'design_id','VG_nominal','IP_nominal','KG_nominal'}]
    for metric in numeric:
        if metric in {'n_frames','mean_rg_nm','std_rg_nm','median_rg_nm',
                      'collapsed_fraction'}:
            # Retain original summary fields; still provide full suffix statistics.
            pass
        stats=rep.groupby('architecture')[metric].agg(['mean','std','min','max']).reset_index()
        stats=stats.rename(columns={k:f'{metric}_{k}' for k in ['mean','std','min','max']})
        s=s.merge(stats,on='architecture',how='left')
    s['fraction_compact_mean']=s['collapsed_fraction']
    s['fraction_compact_min']=s['collapsed_fraction_min']
    s['fraction_compact_max']=s['collapsed_fraction_max']
    s['fraction_compact_std']=s['std_collapsed_fraction_between_reps']
    s['collapsed_percent']=100*s.collapsed_fraction
    s['std_collapsed_percent_between_reps']=100*s.std_collapsed_fraction_between_reps
    s['sem_rg_between_reps_nm']=s.std_rg_between_reps_nm/np.sqrt(s.n_replicates)
    s['threshold_nm']=args.threshold
    s=s.sort_values(['collapsed_fraction','mean_rg_nm'],ascending=[False,True])
    s.insert(0,'collapse_rank',range(1,len(s)+1))
    s.to_csv(out/'combined_summary_mean_std.csv',index=False)
    # Pooled frame stats are provided separately; not equivalent to replicate-weighted means.
    pooled=raw.groupby('architecture').agg(n_frames=('rg_nm','size'),pooled_mean_rg_nm=('rg_nm','mean'),pooled_std_rg_nm=('rg_nm','std'),pooled_collapsed_fraction=('collapsed','mean')).reset_index()
    pooled.to_csv(out/'pooled_frame_summary.csv',index=False)
    s.to_csv(out/'summary_across_replicates.csv',index=False)
    s.to_csv(out/'complete_summary_across_replicates.csv',index=False)
    if args.save_raw: raw.to_csv(out/'all_rg_values_combined.csv',index=False)
    incomplete=s[s.n_replicates!=3]
    if not incomplete.empty: print('WARNING: missing replicates for:', ', '.join(incomplete.architecture))
    print(f'Saved {len(s)} architectures to {out}; threshold {args.threshold:.2f} nm')
if __name__=='__main__': main()
