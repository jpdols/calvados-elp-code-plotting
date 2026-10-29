#!/usr/bin/env python3
"""Design Rule 6 architecture effects across R1--R5, in the DR5 focused figure style.

Input: the library summary CSV with columns family (optional), architecture,
ratio, temperature, kcollapse_ns_inv, kescape_ns_inv, mean_compact_rg_nm.

Each effect is relative to the mean across ALL 30 architectures AT THE SAME RATIO.
Rate effects: log2(rate / reference). Compact Rg effect: Rg - reference (nm).
No family averaging; pale error bars are shown only if replicate data exist.
Sequence tracks: optional, read from R5 CALVADOS top.pdb or FASTA under --library-root.
Tracks are never fabricated from R3 compositions or hard-coded arrangements.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RATIOS = [1, 2, 3, 4, 5]
COMPOSITION = {1:(46,30,8), 2:(38,38,8), 3:(30,46,8), 4:(34,34,16), 5:(30,30,24)}
MOTIF_COLORS = {'VPGVG':'#7CA8D1','VPGIG':'#65A97B','VPGKG':'#F3B6E6'}
LINE_COLORS = ['#E78B32','#8B6BB3','#C45B90','#3F94A0','#B28A36','#737A9B','#A35D56','#5A9B58','#5B7293']
METRICS = [
  ('kcollapse_ns_inv','A','Compact-state entry',r'$\log_2(k_{\mathrm{collapse}}/\bar{k}_{\mathrm{collapse}})$','log'),
  ('kescape_ns_inv','B','Compact-state escape',r'$\log_2(k_{\mathrm{escape}}/\bar{k}_{\mathrm{escape}})$','log'),
  ('mean_compact_rg_nm','C','Compact-state dimensions',r'$\Delta\langle R_g\rangle_{\mathrm{compact}}$ (nm)','delta'),
]
# These are name-based thematic selections, not fabricated numerical rankings.
# Matched architecture keys are resolved against the actual summary at runtime.
COMPARISONS = [
 ('01_charge_clustering','Charge clustering and motif environment',[
   'KG_dispersed_in_IG','KG_clustered_in_IG','KG_embedded_in_IG',
   'single_IG_block_KG_center','KG_dispersed_in_VG','KG_clustered_in_VG',
   'even_distribution']),
 ('02_charge_block_position','Position of KG blocks relative to IG-rich domains',[
   'KG_capped_IG_endblocks','four_IG_blocks_KG_endcaps','charged_large_IG_domain',
   'central_IG_KG_endcaps','IG_endcaps_KG_center','two_IG_blocks_KG_center',
   'single_IG_block_KG_center']),
 ('03_domain_fragmentation','IG-domain fragmentation and interfacial charge',[
   'two_IG_blocks_KG_endcaps','two_IG_blocks_interfacial_KG',
   'three_IG_blocks_KG_endcaps','three_IG_blocks_interfacial_KG',
   'four_IG_blocks_KG_endcaps','four_IG_blocks_interfacial_KG',
   'single_IG_block_KG_center']),
 ('04_compact_state_size','Charge organisation and compact-state dimensions',[
   'charged_large_IG_domain','KG_embedded_in_IG','single_IG_block_KG_center',
   'IG_endcaps_KG_center','KG_dispersed_in_VG','KG_clustered_in_VG',
   'even_distribution']),
]
AA3 = dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),
               'ARNDCQEGHILKMFPSTWYV'))


def slug(s):
    s = str(s).strip()
    s = re.sub(r'^\d{3,4}[_-]', '', s)
    s = re.sub(r'^(VGIGKG|IGVGKG|VGIGKG|IGKG|VGIGKG)[_-]R[1-5][_-]', '', s, flags=re.I)
    return re.sub(r'[^a-z0-9]+','',s.lower())


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--summary', type=Path, required=True, help='Library summary CSV')
    ap.add_argument('--outdir', type=Path, default=Path('DR6_ternary_focused'))
    ap.add_argument('--temperature', type=int, default=370)
    ap.add_argument('--library-root', type=Path, default=None, help='Optional CALVADOS library; reads real R5 sequence tracks')
    ap.add_argument('--track-ratio', type=int, default=5, choices=RATIOS)
    ap.add_argument('--family', default=None, help='Explicit family filter if CSV includes multiple motif families')
    ap.add_argument('--width', type=float, default=11.2)
    ap.add_argument('--dpi', type=int, default=350)
    ap.add_argument('--list-architectures', action='store_true')
    ap.add_argument('--strict', action='store_true', help='Fail if a selected comparison architecture is missing')
    return ap.parse_args()


def load_effects(a):
    d = pd.read_csv(a.summary)
    need = ['architecture','ratio','temperature'] + [m[0] for m in METRICS]
    missing = set(need) - set(d.columns)
    if missing: raise ValueError(f'Missing summary columns: {sorted(missing)}; available: {list(d.columns)}')
    d['ratio_num'] = pd.to_numeric(d['ratio'].astype(str).str.extract(r'^[Rr]?(\d+)$',expand=False), errors='coerce')
    d['temperature_num'] = pd.to_numeric(d['temperature'].astype(str).str.replace(r'\s*[Kk]$','',regex=True),errors='coerce')
    d = d[d['ratio_num'].isin(RATIOS) & (d['temperature_num'] == a.temperature)].copy()
    if 'family' in d.columns:
        if a.family:
            d = d[d.family.astype(str).str.casefold() == a.family.casefold()].copy()
        else:
            families = d.family.dropna().astype(str).str.upper().unique()
            if len(families) > 1:
                # Follow DR6 family identifiers without pooling other libraries.
                candidates = [v for v in families if ('IG' in v and 'VG' in v and 'KG' in v and 'EG' not in v)]
                if len(candidates) != 1:
                    raise ValueError(f'Multiple families {list(families)}; use --family to select DR6')
                d = d[d.family.astype(str).str.upper() == candidates[0]].copy()
                print(f'Filtered family: {candidates[0]}')
    if d.empty: raise ValueError('No DR6 rows at selected temperature/family')
    d['key'] = d.architecture.map(slug)
    if (d['key'] == '').any(): raise ValueError('Empty architecture name after normalization')
    for m,*_ in METRICS:
        d[m] = pd.to_numeric(d[m],errors='raise')
        if not np.isfinite(d[m].to_numpy()).all(): raise ValueError(f'Nonfinite {m}')
    for m in ('kcollapse_ns_inv','kescape_ns_inv'):
        if (d[m] <= 0).any(): raise ValueError(f'Nonpositive values in {m}; cannot take logarithms')
    if 'replicate' not in d.columns: d['replicate'] = 'single'
    if d.duplicated(['ratio_num','key','replicate']).any():
        raise ValueError('Duplicate ratio/architecture/replicate rows; check family filtering')
    means = d.groupby(['ratio_num','key'],as_index=False)[[m[0] for m in METRICS]].mean()
    per_ratio = means.groupby('ratio_num').key.nunique().to_dict()
    print('Unique architecture counts by ratio:',per_ratio)
    if any(per_ratio.get(r,0) != 30 for r in RATIOS):
        print('WARNING: expected 30 architectures at each R1–R5. Ratios with incomplete data will use available-design reference means.')
    # Require same architecture set to draw interpretable curves across all ratios.
    present = {r: set(means.loc[means.ratio_num==r,'key']) for r in RATIOS}
    common = set.intersection(*present.values()) if all(present.values()) else set()
    print(f'Common architecture names across all ratios: {len(common)}')
    if a.list_architectures:
        print('Architecture names in the summary:')
        ref = d[d.ratio_num == max(present, key=lambda r: len(present[r]))][['architecture','key']].drop_duplicates()
        for row in ref.itertuples(index=False): print(f'  {row.architecture} | key={row.key}')
    raw = []
    for metric,_,_,_,kind in METRICS:
        tmp = means[['ratio_num','key',metric]].rename(columns={metric:'raw_value'}).copy()
        tmp['metric'] = metric
        tmp['reference'] = tmp.groupby('ratio_num')['raw_value'].transform('mean')
        tmp['effect'] = np.log2(tmp.raw_value/tmp.reference) if kind == 'log' else tmp.raw_value-tmp.reference
        raw.append(tmp)
    effects = pd.concat(raw,ignore_index=True)
    # For multi-replicate data, show SEM of transformed replicate effect (relative to ratio mean).
    rep_spread = None
    if d.groupby(['ratio_num','key']).size().max() > 1:
        blocks=[]
        for metric,_,_,_,kind in METRICS:
            cell = d[['ratio_num','key','replicate',metric]].copy()
            refs = effects[effects.metric==metric][['ratio_num','key','reference']]
            cell=cell.merge(refs,on=['ratio_num','key'],validate='many_to_one')
            cell['effect'] = np.log2(cell[metric]/cell.reference) if kind=='log' else cell[metric]-cell.reference
            se = cell.groupby(['ratio_num','key']).effect.agg(['count','std']).reset_index()
            se['sem']=se['std']/np.sqrt(se['count'])
            se['metric']=metric
            blocks.append(se[['ratio_num','key','metric','sem']])
        rep_spread=pd.concat(blocks,ignore_index=True)
    return d, effects, rep_spread, common


def extract_sequence(path):
    try:
        if path.suffix.lower() in ('.fasta','.fa','.faa','.fas'):
            seq=''.join(t.strip() for t in path.read_text().splitlines() if t.strip() and not t.startswith('>'))
            seq=re.sub('[^A-Z]','',seq.upper())
        elif path.suffix.lower()=='.pdb':
            residues={}
            for line in path.open():
                if line.startswith(('ATOM  ','HETATM')):
                    rid=(line[21:22],line[22:26].strip(),line[26:27])
                    if rid not in residues: residues[rid]=AA3.get(line[17:20].strip().upper(),'?')
            seq=''.join(residues.values())
        else: return None
        if len(seq)!=420 or any(seq[i:i+5] not in MOTIF_COLORS for i in range(0,len(seq),5)):
            return None
        return [seq[i:i+5] for i in range(0,len(seq),5)]
    except (OSError,UnicodeError):
        return None


def get_tracks(root, ratio, keys):
    if root is None: return {}
    if not root.is_dir():
        print(f'WARNING: library root not found: {root}');return {}
    # Discover top-level architecture folders only and avoid scanning every trajectory.
    tracks={}
    for folder in root.iterdir():
        if not folder.is_dir() or not re.search(rf'(?:^|_)R{ratio}(?:_|$)',folder.name,re.I):continue
        folder_key=slug(folder.name)
        if folder_key not in keys: continue
        preferred=list(folder.glob('**/top.pdb'))
        if not preferred: preferred=list(folder.glob('**/*.fa*'))
        for p in preferred:
            motifs=extract_sequence(p)
            if motifs:
                counts=(motifs.count('VPGVG'),motifs.count('VPGIG'),motifs.count('VPGKG'))
                if counts != COMPOSITION[ratio]:
                    print(f'WARNING: {folder.name} track counts {counts}, expected {COMPOSITION[ratio]}')
                    continue
                tracks[folder_key]=motifs; break
    print(f'Loaded exact R{ratio} sequence tracks: {len(tracks)}')
    return tracks


def choose_selections(available, strict):
    aliases = {
      'kgdispersedinig':['KG_distributed_in_IG','KG_dispersed_through_IG','KG_dispersed_in_IG_blocks'],
      'kgclusteredinig':['KG_cluster_in_IG','KG_clustered_within_IG'],
      'kgembeddedinig':['KG_embedded_inside_IG'],
      'singleigblockkgcenter':['single_IG_KG_center'],
      'twoigblockskgendcaps':['two_IG_blocks_KG_ends'],
      'threeigblockskgendcaps':['three_IG_blocks_KG_ends'],
      'fourigblockskgendcaps':['four_IG_blocks_KG_ends'],
    }
    result=[]
    for stem,title,requested in COMPARISONS:
        resolved=[];missing=[]
        for name in requested:
            key=slug(name)
            if key in available:
                resolved.append((key,name))
                continue
            variants=[slug(v) for v in aliases.get(key,[])]
            match=next((v for v in variants if v in available),None)
            if match:resolved.append((match,name))
            else:missing.append(name)
        if missing:
            msg=f'{stem}: missing {missing}'
            if strict: raise ValueError(msg)
            print('WARNING:',msg)
        if len(resolved)<2:
            print(f'SKIP {stem}: fewer than 2 matched designs')
            continue
        result.append((stem,title,resolved))
    return result


def draw_tracks(ax, selected, tracks):
    n=len(selected)
    ax.set_xlim(0,1);ax.set_ylim(-.65,n+.75);ax.axis('off')
    if tracks:
        ax.text(.01,n+.3,'Sequence architecture: VPGVG / VPGIG / VPGKG (R5)',fontsize=10,weight='bold')
    else:
        ax.text(.01,n+.3,'Architecture comparisons (sequence tracks unavailable; supply --library-root)',fontsize=10,weight='bold')
    x0,x1=.48,.985
    for i,(key,label) in enumerate(selected):
        y=n-1-i
        color=LINE_COLORS[i%len(LINE_COLORS)]
        ax.plot([.015,.055],[y,y],lw=3,color=color,solid_capstyle='round')
        ax.text(.065,y,label.replace('_',' '),va='center',fontsize=9.2)
        motif=tracks.get(key)
        if motif is None:
            ax.text(x0,y,'Track not available',color='#777',fontsize=8,va='center')
            continue
        w=(x1-x0)/len(motif)
        for j,m in enumerate(motif):
            ax.add_patch(Rectangle((x0+j*w,y-.17),w+.0001,.34,color=MOTIF_COLORS[m],linewidth=0))
        ax.add_patch(Rectangle((x0,y-.17),x1-x0,.34,fill=False,edgecolor='#777',linewidth=.35))
    if tracks:
        for j,(motif,col) in enumerate(MOTIF_COLORS.items()):
            ax.add_patch(Rectangle((.49+j*.16,-.56),.014,.13,color=col))
            ax.text(.51+j*.16,-.5,motif,fontsize=8,va='center')


def plot_one(effects, spread, selected, title, stem, tracks, a):
    keys=[x[0] for x in selected]
    fig=plt.figure(figsize=(a.width,12.8+.23*len(keys)))
    gs=fig.add_gridspec(4,1,height_ratios=[.35+.075*len(keys),1,1,1],hspace=.35,
                         top=.975,bottom=.091,left=.13,right=.96)
    track_ax=fig.add_subplot(gs[0]);draw_tracks(track_ax,selected,tracks)
    axes=[fig.add_subplot(gs[i]) for i in range(1,4)]
    lookup=effects.set_index(['key','ratio_num','metric'])
    sem_lookup=spread.set_index(['key','ratio_num','metric']) if spread is not None else None
    for ax,(metric,letter,panel,ylabel,_) in zip(axes,METRICS):
        for i,(key,_) in enumerate(selected):
            sub=lookup.loc[[(key,r,metric) for r in RATIOS]]
            y=sub.effect.to_numpy(); x=np.array(RATIOS,dtype=float)
            color=LINE_COLORS[i%len(LINE_COLORS)]
            if sem_lookup is not None:
                se=sem_lookup.reindex([(key,r,metric) for r in RATIOS]).sem.to_numpy()
                if np.isfinite(se).any():ax.vlines(x,y-np.nan_to_num(se),y+np.nan_to_num(se),color=color,lw=2,alpha=.26)
            ax.plot(x,y,'-o',color=color,lw=1.9,ms=5,zorder=3)
        ax.axhline(0,ls='--',color='#666',lw=1.4,zorder=1)
        ax.grid(axis='y',color='#E6E6E6',lw=.75,zorder=0)
        ax.set_xlim(.8,5.2)
        ax.set_xticks(RATIOS,[f'R{r}\n(KG {COMPOSITION[r][2]}/84)' for r in RATIOS])
        ax.set_ylabel(ylabel,fontsize=10)
        ax.set_title(f'{letter}   {panel}',loc='left',weight='bold',fontsize=12,pad=11)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[-1].set_xlabel('VG / IG / KG composition ratio (R1–R5)',fontsize=10)
    fig.text(.13,.04,'Each line: same architecture across R1–R5   |   Zero: mean of all 30 designs at the same ratio',fontsize=9,color='#555')
    fig.text(.13,.023, f'Sequence tracks show actual R{a.track_ratio} motifs only. R1–R3 each contain eight KG motifs; R4 has 16 and R5 has 24.',fontsize=8.5,color='#666')
    for ext in ('pdf','svg','png'):
        dest=a.outdir/f'DR6_{stem}_{a.temperature}K.{ext}'
        fig.savefig(dest,dpi=a.dpi,bbox_inches='tight',pad_inches=.16,facecolor='white')
    plt.close(fig)
    print('Saved',f'DR6_{stem}_{a.temperature}K.pdf/.svg/.png')


def main():
    a=parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    data,effects,spread,common=load_effects(a)
    effects.to_csv(a.outdir/'DR6_architecture_effects.csv',index=False)
    if spread is not None:spread.to_csv(a.outdir/'DR6_effect_sem.csv',index=False)
    selection=choose_selections(common,a.strict)
    if not selection:
        raise ValueError('No architecture comparisons matched. Rerun with --list-architectures and share the printed names.')
    all_keys=set(k for _,_,items in selection for k,_ in items)
    tracks=get_tracks(a.library_root,a.track_ratio,all_keys)
    for stem,title,selected in selection:
        plot_one(effects,spread,selected,title,stem,tracks,a)
    print('Output directory:',a.outdir.resolve())

if __name__=='__main__':
    main()
