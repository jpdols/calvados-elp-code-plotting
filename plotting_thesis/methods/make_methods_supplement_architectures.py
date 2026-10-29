"""Produce Methods subsets and complete Supplement figures for DR5 and DR6.
Usage: python make_methods_supplement_architectures.py --outdir architecture_figures_final
Uses hardcoded R4 definitions from plot_motif_architectures.py.
"""
import argparse
from pathlib import Path
from plot_motif_architectures import (DR5,DR6,COLORS,GROUPS_DR5,GROUPS_DR6,build,render)

# Selected to illustrate controlled contrasts, not chosen by observed performance.
METHODS_DR5 = [
 ('Local charge mixing', [592,593,594]),
 ('Charge-domain topology', [595,596,599,600]),
 ('Hydrophobic separator', [605,607]),
 ('Charge polarity', [610,611]),
]
METHODS_DR6 = [
 ('Reference and dispersed controls', [775,774,779,801]),
 ('Charge clustering and local environment', [778,800]),
 ('Charge-domain placement', [785,799]),
 ('Hydrophobic fragmentation and interfaces', [781,782,790,802]),
]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--outdir',default='architecture_figures_final')
 p.add_argument('--show-ids',action='store_true')
 args=p.parse_args()
 out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
 libraries=[
  ('DR5',DR5,{'IG':28,'EG':28,'KG':28},GROUPS_DR5,METHODS_DR5,
   'DR5 | Opposite-charge sequence architectures — R4 (28 / 28 / 28)'),
  ('DR6',DR6,{'VG':34,'IG':34,'KG':16},GROUPS_DR6,METHODS_DR6,
   'DR6 | Hydrophobic and charged architectures — R4 (34 / 34 / 16)'),
 ]
 for name,data,counts,full,subset,title in libraries:
  seqs=build(data,counts)
  assert len({n for _,ids in subset for n in ids})==sum(len(ids) for _,ids in subset)
  assert set(n for _,ids in subset for n in ids).issubset(seqs)
  # Restrict legends to actual motifs in this library.
  colors={m:COLORS[m] for m in counts}
  for suffix,groups in [('methods_subset',subset),('supplement_complete',full)]:
   path=out/f'{name}_R4_{suffix}'
   render(data,groups,seqs,colors,title,path,show_ids=args.show_ids)
   print(path, 'rows:',sum(len(ids) for _,ids in groups))
 print('Note: random shuffle and DR6 even distribution are reproducible illustrative constructions, not verified simulation manifest sequences.')

if __name__=='__main__': main()
