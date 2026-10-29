#!/usr/bin/env python
"""Summarize Design Rule 6 VGIGKG R1-R5 trajectories at 300 and 370 K.
Requires numpy and pandas. Rg must be in nm; input time must be in ns.
Old collapsed/threshold columns are ignored. No sequence file is required.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    "sequence": ["id", "sequence_id", "sequence", "sequence_name", "design_id"],
    "temperature": ["temp", "temperature_k", "temperature", "temp_k"],
    "rg": ["rg_nm", "rg", "radius_of_gyration_nm", "r_g"],
    "time": ["time_ns", "simulation_time_ns", "t_ns"],
    "frame": ["original_frame", "frame", "frame_index", "analysed_index"],
    "replicate": ["replicate", "replicate_id", "rep", "rep_id", "replica",
                  "run_id", "trajectory_id"],
}

def resolve(columns, kind, override):
    if override:
        if override not in columns:
            raise ValueError(f"Column {override!r} not found: {list(columns)}")
        return override
    lookup = {str(c).strip().lower(): c for c in columns}
    return next((lookup[a] for a in ALIASES[kind] if a in lookup), None)

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

def load_family(path, family, args):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    header = pd.read_csv(path, nrows=0)
    cols = {k: resolve(header.columns, k, getattr(args, k + "_col"))
            for k in ALIASES}
    print(f"\n{family}: {path}")
    print("Detected columns:", json.dumps(cols))
    for kind in ("sequence", "temperature", "rg"):
        if cols[kind] is None:
            raise ValueError(f"Cannot detect {kind}; use --{kind}-col.")
    if not cols["time"] and not cols["frame"]:
        raise ValueError("A time_ns or frame coordinate is required.")
    if not cols["replicate"]:
        print("No replicate column: treating each design/temperature as one trajectory; "
              "duplicate times will raise an error.")
    usecols = list(dict.fromkeys(c for c in cols.values() if c))
    stringcols = [cols[k] for k in ("sequence", "temperature", "replicate") if cols[k]]
    data = defaultdict(list)
    n_read = n_kept = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize,
                             dtype={c: "string" for c in stringcols}):
        n_read += len(chunk)
        ids = chunk[cols["sequence"]].str.strip()
        ratios = pd.to_numeric(ids.str.extract(r"_R(\d+)(?:_|$)", expand=False),
                               errors="coerce")
        temperatures = pd.to_numeric(
            chunk[cols["temperature"]].str.strip().str.replace(
                r"\s*[Kk]$", "", regex=True), errors="coerce")
        keep = (ids.str.contains(r"(?:^|_)VGIGKG_R[1-5]_", regex=True, na=False)
                & ratios.isin([1, 2, 3, 4, 5]) & temperatures.isin([300, 370]))
        c = chunk.loc[keep]
        if c.empty:
            continue
        n_kept += len(c)
        work = pd.DataFrame(index=c.index)
        work["sequence"] = ids.loc[keep]
        work["temperature"] = temperatures.loc[keep].astype(int)
        if cols["replicate"]:
            work["replicate"] = c[cols["replicate"]].str.strip()
            if work["replicate"].isna().any() or work["replicate"].eq("").any():
                raise ValueError(f"{family}: missing replicate identifier.")
        else:
            work["replicate"] = "single"
        coordinate_col = cols["time"] or cols["frame"]
        work["_time"] = pd.to_numeric(c[coordinate_col], errors="coerce")
        if not cols["time"]:
            work["_time"] *= args.frame_time_ns
        work["_rg"] = pd.to_numeric(c[cols["rg"]], errors="coerce")
        for key, group in work.groupby(["sequence", "temperature", "replicate"],
                                       sort=False):
            data[key].append(group[["_time", "_rg"]].to_numpy(dtype=float))
    if not data:
        raise ValueError(f"{family}: no R1-R5 rows at 300/370 K. Check IDs and temp column.")
    print(f"Rows read: {n_read:,}; retained R1-R5, 300/370 K: {n_kept:,}")
    rows = []
    event_path = Path(args.outdir) / "dwell_events.csv"
    for index, key in enumerate(sorted(data), 1):
        sequence, temperature, replicate = key
        values = np.concatenate(data.pop(key), axis=0)
        group = pd.DataFrame(values, columns=["_time", "_rg"])
        row, events = analyze_group(group, args, (family,) + key)
        match = re.search(r"_R(\d+)_(.+)$", sequence)
        if match is None:
            raise ValueError(f"Cannot extract architecture from {sequence!r}")
        ratio = int(match.group(1))
        architecture = match.group(2)
        architecture_key = re.sub(r"^asymmetric_IG\d+_IG\d+_KG_endcaps$",
                                  "asymmetric_IG_blocks_KG_endcaps", architecture)
        meta = dict(family=family, sequence=sequence, ratio=f"R{ratio}",
                    ratio_number=ratio, architecture=architecture, architecture_key=architecture_key,
                    temperature=temperature, replicate=replicate)
        rows.append({**meta, **row})
        if args.save_dwell_events and events:
            pd.DataFrame([{**meta, **event} for event in events]).to_csv(
                event_path, mode="a", header=not event_path.exists(), index=False)
        if index % 20 == 0 or not data:
            print(f"  Summarized {index} trajectories")
    return rows, dict(path=str(path.resolve()), detected_columns=cols,
                      n_rows_read=n_read, n_rows_retained=n_kept)


def aggregate_replicates(result):
    groupcols = ["family", "sequence", "ratio", "ratio_number", "architecture", "architecture_key", "temperature"]
    metrics = [c for c in result.select_dtypes(include=np.number).columns
               if c not in groupcols]
    wide, long = [], []
    for key, group in result.groupby(groupcols, sort=False):
        meta = dict(zip(groupcols, key))
        record = {**meta, "n_replicates": len(group)}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = avg(values)
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            sem = sd / np.sqrt(n) if n > 1 else np.nan
            record.update({metric + "_mean": mean, metric + "_sd": sd,
                           metric + "_sem": sem, metric + "_n": n})
            long.append({**meta, "metric": metric, "n_valid_replicates": n,
                         "mean": mean, "sd": sd, "sem": sem})
        wide.append(record)
    return pd.DataFrame(wide), pd.DataFrame(long)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="VGIGKG all_rg_values.csv")
    parser.add_argument("--outdir", default="DR6_VGIGKG_summary_R1_R5")
    parser.add_argument("--threshold", type=float, default=4.61)
    parser.add_argument("--frame-time-ns", type=float, default=0.01,
                        help="Time per original frame; only used if time_ns is absent.")
    parser.add_argument("--sample-dt-ns", type=float, default=None,
                        help="Optional expected CSV sample spacing; otherwise inferred.")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--save-dwell-events", action="store_true",
                        help="Also save individual dwell events; this CSV may be large.")
    for kind in ALIASES:
        parser.add_argument(f"--{kind}-col", default=None)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("Threshold must be finite and positive.")
    if not np.isfinite(args.frame_time_ns) or args.frame_time_ns <= 0:
        parser.error("frame-time-ns must be finite and positive.")
    if args.sample_dt_ns is not None and (
            not np.isfinite(args.sample_dt_ns) or args.sample_dt_ns <= 0):
        parser.error("sample-dt-ns must be finite and positive.")
    if args.chunksize < 1:
        parser.error("chunksize must be positive.")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / "dwell_events.csv"
    if events_path.exists():
        events_path.unlink()  # Remove events from an earlier run of this script.
    rows, sources = [], {}
    for family, path in [("VGIGKG", args.input)]:
        family_rows, source = load_family(path, family, args)
        rows.extend(family_rows)
        sources[family] = source
    result = pd.DataFrame(rows).sort_values(
        ["family", "ratio_number", "architecture", "temperature", "replicate"])
    # Nominal composition from the design table, not verified FASTA counts.
    counts = {1: (46, 30, 8), 2: (38, 38, 8), 3: (30, 46, 8),
              4: (34, 34, 16), 5: (30, 30, 24)}
    for i, motif in enumerate(["VG", "IG", "KG"]):
        result["nominal_" + motif.lower() + "_motifs"] = result.ratio_number.map(
            {r: c[i] for r, c in counts.items()})
    result["sequence_composition_check_required"] = (
        result.ratio_number.eq(3) & result.architecture.eq("original_Tk8_adapted"))
    result.to_csv(out / "summary_per_replicate.csv", index=False)
    result.loc[result.sequence_composition_check_required,
               ["sequence", "ratio", "architecture", "temperature", "replicate"]].to_csv(
        out / "sequence_checks_required.csv", index=False)
    wide, long = aggregate_replicates(result)
    wide.to_csv(out / "summary_across_replicates.csv", index=False)
    long.to_csv(out / "summary_across_replicates_long.csv", index=False)
    idcols = ["family", "sequence", "ratio", "architecture", "temperature", "replicate"]
    qc = ["n_rows", "n_frames", "n_missing_rg", "n_blocks", "n_frame_gaps",
          "sample_dt_ns", "time_span_ns", "observed_frame_time_ns",
          "linked_interval_time_ns", "n_entries", "n_exits"]
    result[idcols + qc].to_csv(out / "data_quality.csv", index=False)
    coverage = result.groupby(["family", "ratio", "temperature"]).agg(
        n_designs=("sequence", "nunique"), n_trajectories=("sequence", "size"),
        min_frames=("n_frames", "min"), max_frames=("n_frames", "max"))
    coverage.to_csv(out / "coverage.csv")
    with open(out / "analysis_settings.json", "w", encoding="utf-8") as handle:
        json.dump(dict(arguments=vars(args), ratios=[1, 2, 3, 4, 5],
                       temperatures_k=[300, 370], sources=sources), handle, indent=2)
    notes = (
        "Compact: Rg < threshold_nm; equality is expanded. Rg in nm, time in ns.\n"
        "Existing collapsed and threshold columns are ignored. All supplied frames are used.\n"
        "No replicate column: one recorded trajectory per design/temperature; duplicates rejected.\n"
        "Rates = crossings / linked source-state exposure (ns); units ns^-1.\n"
        "Missing Rg/time gaps split trajectories; no transitions across gaps.\n"
        "State frame times = assigned sample count * sample_dt_ns.\n"
        "Transition exposure excludes the final sample in each contiguous block.\n"
        "No smoothing, burn-in removal, or minimum-dwell filter is applied.\n"
        "Complete dwells exclude boundary-censored events and can favor shorter events.\n"
        "Observed event means include truncated boundary events.\n"
        "Across-replicate means use equal trajectory weights; SD/SEM need >=2 trajectories.\n"
        "Quantile widths describe conformational variation, not confidence intervals.\n"
        "SCD and structural contact metrics require sequences/coordinates and are not inferred here.\n"
        "Composition columns are nominal design counts, not verified sequence counts.\n"
        "R3 original_Tk8_adapted is retained and flagged: supplied expression matches R2 counts.\n"
        "Verify actual simulated sequence before composition-matched plotting/baselines.\n"
        "architecture_key harmonizes asymmetric IG block names across ratios.\n"
    )
    (out / "METRIC_NOTES.txt").write_text(notes, encoding="utf-8")
    print("\nCoverage:")
    print(coverage.to_string())
    print(f"\nFinished: {len(result)} trajectories. Outputs: {out.resolve()}")
    print("Use summary_per_replicate.csv as the master input for plotting.")
    if result.sequence_composition_check_required.any():
        print("Check sequence_checks_required.csv: R3 Tk8 composition needs verification.")
    if result["n_frame_gaps"].gt(0).any() or result["n_missing_rg"].gt(0).any():
        print("Some trajectories have gaps/missing Rg: inspect data_quality.csv.")
    if (result["n_blocks"] > 1).any():
        print("Rates were calculated only within contiguous sampling blocks.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}")
