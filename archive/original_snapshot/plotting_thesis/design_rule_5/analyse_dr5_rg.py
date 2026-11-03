#!/usr/bin/env python
"""Summarize frame-resolved Rg, with separate trajectories and censored dwells.
Requires numpy and pandas. Rg input must be in nm. Rates are in ns^-1.
Kinetics are operational threshold-crossing statistics, not fitted Markov rates.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
 'sequence': ['sequence_id','sequence_name','design_id','design','sequence','seq_id','seq','name','id','protein'],
 'replicate': ['replicate','replicate_id','rep','rep_id','replica','run','run_id','trajectory','trajectory_id'],
 'temperature': ['temperature_k','temperature','temp_k','temp','t_k'],
 'rg': ['rg_nm','rg','radius_of_gyration_nm','radius_of_gyration','r_g'],
 'frame': ['frame','frame_index','frame_idx','frame_number','frame_id'],
 'time': ['time_ns','time','simulation_time_ns','t_ns'],
}
def norm(s): return re.sub(r'[^a-z0-9]+','_',str(s).strip().lower()).strip('_')
def resolve(columns, kind, override):
 if override:
  if override not in columns: raise ValueError(f'Column {override!r} not found. Available: {list(columns)}')
  return override
 lookup = {norm(c):c for c in columns}
 return next((lookup[a] for a in ALIASES[kind] if a in lookup),None)
def avg(a): return float(np.mean(a)) if len(a) else np.nan

def analyze_group(g, args, key):
 # Explicit coordinates required; no assumption about input row order.
 coordinate = pd.to_numeric(g['_time'],errors='coerce').to_numpy(float)
 rg = pd.to_numeric(g['_rg'],errors='coerce').to_numpy(float)
 if not np.isfinite(coordinate).all(): raise ValueError(f'{key}: missing or invalid frame/time coordinates.')
 order = np.argsort(coordinate,kind='stable'); coordinate=coordinate[order]; rg=rg[order]
 d = np.diff(coordinate)
 if np.any(d<=0): raise ValueError(f'{key}: duplicate timestamps/frames. Check trajectory grouping columns.')
 if len(d)==0: raise ValueError(f'{key}: fewer than two rows; kinetics cannot be calculated.')
 dt = args.sample_dt_ns if args.sample_dt_ns else float(np.min(d))
 if dt<=0: raise ValueError('Sampling interval must be positive.')
 # Missing samples split trajectories, preventing artificial transitions.
 linked=np.isclose(d,dt,rtol=1e-5,atol=1e-8)
 valid=np.isfinite(rg)
 if np.any(rg[valid]<0): raise ValueError(f'{key}: negative Rg values.')
 vals=rg[valid]
 if not len(vals): raise ValueError(f'{key}: no finite Rg values.')
 compact=rg < args.threshold
 breaks=np.r_[True,~linked] | ~valid | np.r_[False,~valid[:-1]]
 starts=np.flatnonzero(breaks & valid)
 # Obtain contiguous blocks of finite, regularly spaced samples.
 blocks=[]
 for start in starts:
  end=start+1
  while end<len(rg) and valid[end] and linked[end-1]: end+=1
  blocks.append((start,end))
 events=[]; entries=exits=0; tc=te=0.; exposure_c=exposure_e=0.
 for block_id,(start,end) in enumerate(blocks,1):
  s=compact[start:end]; n=len(s)
  tc+=np.count_nonzero(s)*dt; te+=np.count_nonzero(~s)*dt
  # Assign each observable transition interval to its source state.
  exposure_c+=np.count_nonzero(s[:-1])*dt
  exposure_e+=np.count_nonzero(~s[:-1])*dt
  changes=np.flatnonzero(s[1:]!=s[:-1])+1
  entries+=int(np.count_nonzero(~s[:-1]&s[1:]))
  exits+=int(np.count_nonzero(s[:-1]&~s[1:]))
  boundaries=np.r_[0,changes,n]
  for a,b in zip(boundaries[:-1],boundaries[1:]):
   events.append(dict(block=block_id,state='compact' if s[a] else 'expanded',
    start_time_ns=coordinate[start+a],last_sample_time_ns=coordinate[start+b-1],
    n_frames=int(b-a),duration_ns=(b-a)*dt,
    left_censored=bool(a==0),right_censored=bool(b==n),
    complete=bool(a>0 and b<n)))
 q=np.percentile(vals,[5,25,50,75,95])
 c=vals[vals<args.threshold]; e=vals[vals>=args.threshold]
 row=dict(n_rows=len(g),n_frames=len(vals),n_missing_rg=int((~valid).sum()),
  n_blocks=len(blocks),n_frame_gaps=int((~linked).sum()),sample_dt_ns=dt,
  time_span_ns=coordinate[-1]-coordinate[0],observed_frame_time_ns=tc+te,
  linked_interval_time_ns=exposure_c+exposure_e,threshold_nm=args.threshold,
  mean_rg_nm=avg(vals),median_rg_nm=q[2],sd_rg_nm=float(np.std(vals,ddof=1)) if len(vals)>1 else np.nan,
  p5_rg_nm=q[0],p25_rg_nm=q[1],p75_rg_nm=q[3],p95_rg_nm=q[4],
  iqr_rg_nm=q[3]-q[1],central90_rg_nm=q[4]-q[0],
  fraction_compact=len(c)/len(vals),mean_compact_rg_nm=avg(c),mean_expanded_rg_nm=avg(e),
  mean_collapse_depth_nm=avg(args.threshold-c),
  n_entries=entries,n_exits=exits,compact_frame_time_ns=tc,expanded_frame_time_ns=te,
  compact_transition_exposure_ns=exposure_c,expanded_transition_exposure_ns=exposure_e,
  kcollapse_ns_inv=entries/exposure_e if exposure_e else np.nan,
  kescape_ns_inv=exits/exposure_c if exposure_c else np.nan)
 for state in ['compact','expanded']:
  ev=[v for v in events if v['state']==state]
  complete=[v['duration_ns'] for v in ev if v['complete']]
  observed=[v['duration_ns'] for v in ev]
  row.update({f'n_{state}_events':len(ev),f'n_{state}_complete_events':len(complete),
   f'n_{state}_censored_events':len(ev)-len(complete),
   f'mean_{state}_dwell_complete_ns':avg(complete),
   f'median_{state}_dwell_complete_ns':float(np.median(complete)) if complete else np.nan,
   f'mean_{state}_observed_event_ns':avg(observed)})
 return row,events

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--input',required=True)
 ap.add_argument('--outdir',default='EGKG_summary')
 ap.add_argument('--threshold',type=float,default=4.61)
 ap.add_argument('--frame-time-ns',type=float,default=0.01,help='Time per ORIGINAL frame index, not per CSV row.')
 ap.add_argument('--sample-dt-ns',type=float,help='Expected spacing of CSV samples; optional, otherwise minimum per trajectory.')
 ap.add_argument('--time-unit',choices=['ns','ps'],default='ns')
 for kind in ALIASES: ap.add_argument(f'--{kind}-col')
 ap.add_argument('--group-cols',nargs='+',help='Additional columns that identify separate trajectories (e.g. cycle).')
 ap.add_argument('--assume-single-replicate',action='store_true',help='Only use when CSV genuinely contains one replicate.')
 ap.add_argument('--temperature-k',type=float,help='Constant temperature if no temperature column exists.')
 args=ap.parse_args()
 if args.threshold<=0 or args.frame_time_ns<=0: ap.error('Threshold and frame-time-ns must be positive.')
 df=pd.read_csv(args.input,sep=None,engine='python')
 columns={kind:resolve(df.columns,kind,getattr(args,kind+'_col')) for kind in ALIASES}
 print('Detected columns:',json.dumps(columns,indent=2))
 for kind in ['sequence','rg']:
  if columns[kind] is None: raise ValueError(f'Cannot detect {kind} column. Use --{kind}-col. Available: {list(df.columns)}')
 if not columns['frame'] and not columns['time']: raise ValueError('A frame or time column is required for correct kinetics.')
 work=pd.DataFrame({'sequence':df[columns['sequence']], '_rg':df[columns['rg']]})
 if columns['replicate']:work['replicate']=df[columns['replicate']]
 elif args.assume_single_replicate:work['replicate']='single'
 else:raise ValueError('No replicate column detected. Supply --replicate-col, or --assume-single-replicate only for a single trajectory per sequence.')
 if columns['temperature']:work['temperature']=df[columns['temperature']]
 elif args.temperature_k is not None:work['temperature']=args.temperature_k
 else:raise ValueError('No temperature column detected. Use --temperature-col or --temperature-k.')
 if columns['time']:
  work['_time']=pd.to_numeric(df[columns['time']],errors='coerce')*(.001 if args.time_unit=='ps' else 1.)
 else:work['_time']=pd.to_numeric(df[columns['frame']],errors='coerce')*args.frame_time_ns
 groups=['sequence','temperature','replicate']
 for col in args.group_cols or []:
  if col not in df:raise ValueError(f'Additional grouping column {col!r} not found.')
  if col in groups:raise ValueError(f'Additional grouping column {col!r} duplicates a standard group.')
  work[col]=df[col];groups.append(col)
 if work[groups].isna().any().any():raise ValueError('Missing trajectory identifiers. Correct grouping before analysis.')
 rows=[];events=[]
 for key,g in work.groupby(groups,sort=False,dropna=False):
  ids=dict(zip(groups,key));row,ev=analyze_group(g,args,key)
  rows.append({**ids,**row});events.extend([{**ids,**v} for v in ev])
 result=pd.DataFrame(rows)
 # Each row is independent; do not pool raw frames across replicates.
 summary_groups=[v for v in groups if v!='replicate']
 metrics=[v for v in result.select_dtypes(include=np.number).columns if v not in groups]
 long=[];wide=[]
 for key,g in result.groupby(summary_groups,sort=False,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  ids=dict(zip(summary_groups,key));wr={**ids,'n_replicates':len(g)}
  for metric in metrics:
   values=g[metric].dropna().to_numpy(float); n=len(values)
   mean=avg(values);sd=float(np.std(values,ddof=1)) if n>1 else np.nan
   sem=sd/np.sqrt(n) if n>1 else np.nan
   long.append({**ids,'metric':metric,'n_valid_replicates':n,'mean':mean,'sd':sd,'sem':sem})
   wr.update({metric+'_mean':mean,metric+'_sd':sd,metric+'_sem':sem,metric+'_n':n})
  wide.append(wr)
 out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
 result.to_csv(out/'summary_per_replicate.csv',index=False)
 pd.DataFrame(wide).to_csv(out/'summary_across_replicates.csv',index=False)
 pd.DataFrame(long).to_csv(out/'summary_across_replicates_long.csv',index=False)
 pd.DataFrame(events).to_csv(out/'dwell_events.csv',index=False)
 check=['n_rows','n_frames','n_missing_rg','n_blocks','n_frame_gaps','sample_dt_ns','time_span_ns','linked_interval_time_ns']
 result[groups+check].to_csv(out/'data_quality.csv',index=False)
 with open(out/'analysis_settings.json','w') as f:json.dump({'arguments':vars(args),'detected_columns':columns,'input_columns':list(df.columns)},f,indent=2)
 (out/'METRIC_NOTES.txt').write_text(
  'Rg input is assumed to be nm. Compact means Rg < threshold; equality is expanded.\n'
  'Replicate summaries use equal weighting; SD and SEM are between trajectories.\n'
  'Rates = crossing counts / linked source-state intervals in ns. Gaps are excluded.\n'
  'Complete dwells exclude the first and last event of every contiguous block.\n'
  'Complete-dwell averages can be biased towards shorter events: inspect censoring counts.\n'
  'Observed event lengths include boundary-censored events and are not complete dwell estimates.\n'
  'Durations use number of assigned samples times sampling interval.\n'
  'Missing Rg and coordinate gaps split trajectories; no crossings are counted across gaps.\n'
  'No smoothing or minimum dwell filter is applied. Rates depend on sampling interval.\n'
  'These metrics cannot establish an electrostatic mechanism without additional analysis.\n',encoding='utf-8')
 print(f'Analyzed {len(result)} trajectories; outputs: {out.resolve()}')
 print(result[groups+['mean_rg_nm','iqr_rg_nm','fraction_compact','kcollapse_ns_inv','kescape_ns_inv']].to_string(index=False))
if __name__=='__main__':
 try:main()
 except (ValueError,KeyError) as error:raise SystemExit(f'ERROR: {error}')
