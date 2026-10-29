#!/usr/bin/env python
"""Methods-only DR6 architecture overview (no simulation data required).

Place beside plot_dr6_architecture_focus_v12_readable_balanced.py.
The random-shuffle row is deliberately uncoloured: exact ordering unavailable.
"""
import argparse
import importlib.util
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

GROUPS = [
 ('Reference and distributed arrangements', [
  'original_Tk8_adapted','even_distribution','KG_dispersed_in_IG',
  'KG_dispersed_in_VG','central_mixed_sticker_domain']),
 ('KG clustering and local environment', [
  'KG_embedded_in_IG','KG_clustered_in_VG','single_IG_block_KG_center',
  'central_IG_KG_adjacent','charged_large_IG_domain']),
 ('Charge block positioning and symmetry', [
  'KG_capped_IG_endblocks','central_IG_KG_endcaps','IG_endcaps_KG_center',
  'IG_outermost_KG_inside','KG_one_end_symmetric_IG','IG_one_end_KG_opposite',
  'KG_then_IG_same_end','IG_then_KG_same_end','asymmetric_IG_blocks_KG_endcaps']),
 ('Hydrophobic-domain number and interfaces', [
  'two_IG_blocks_KG_endcaps','two_IG_blocks_KG_center','two_IG_blocks_interfacial_KG',
  'three_medium_IG_blocks_KG_endcaps','three_IG_blocks_KG_center',
  'three_IG_blocks_interfacial_KG','four_IG_blocks_KG_endcaps',
  'four_IG_blocks_interfacial_KG']),
 ('Hydrophobic gradients', ['hydrophobic_gradient','symmetric_IG_gradient']),
]


def load_design_module(path):
 spec=importlib.util.spec_from_file_location('dr6_design_definitions',path)
 if spec is None or spec.loader is None:raise RuntimeError('Could not load design definitions')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.validate_sequences()
 return module


def make_plot(module,output,ratio=5,dpi=240,random_label=True):
 if ratio!=5:raise ValueError('The complete 29-design catalogue is sequence-verified for R5 only.')
 keys=[key for _,group in GROUPS for key in group]
 if len(keys)!=29 or len(set(keys))!=29:raise ValueError('Expected 29 distinct designs')
 known={k for r,k in module.SEQUENCES if r==ratio}
 if set(keys)!=known:raise ValueError(f'Overview and manifest differ: missing={known-set(keys)}, unknown={set(keys)-known}')
 rows=[]
 for title,group in GROUPS:
  rows.append(('header',title,None))
  rows.extend(('design',key,module.expand(module.SEQUENCES[(ratio,key)])) for key in group)
 if random_label:rows.extend([('header','Randomized control',None),('random','random shuffle',None)])
 n=len(rows)
 fig,ax=plt.subplots(figsize=(15.4,max(13.8,n*.43+1.6)))
 fig.subplots_adjust(left=.055,right=.985,top=.945,bottom=.065)
 ax.set_xlim(-78,86)
 ax.set_ylim(-1,n)
 ax.axis('off')
 colors=module.COLORS
 for idx,(kind,name,motifs) in enumerate(rows):
  y=n-idx-1
  if kind=='header':
   ax.axhspan(y-.43,y+.45,color='#EDF0F3',zorder=-2)
   ax.text(-75,y,name,fontsize=12.8,fontweight='bold',va='center',ha='left',color='#293544')
   continue
  label=name.replace('_',' ')
  ax.text(-3.2,y,label,fontsize=11.3,va='center',ha='right',fontweight='medium')
  if kind=='random':
   ax.add_patch(Rectangle((0,y-.27),84,.54,facecolor='#F0F0F0',edgecolor='#8D8D8D',lw=.8,hatch='///'))
   ax.text(42,y,'Exact motif order unavailable',ha='center',va='center',fontsize=10.6,color='#4B5563')
  else:
   assert len(motifs)==84 and Counter(motifs)==Counter({'VG':30,'IG':30,'KG':24})
   for x,m in enumerate(motifs):
    ax.add_patch(Rectangle((x,y-.28),1,.56,facecolor=colors[m],edgecolor='none'))
   for x in (0,21,42,63,84):
    ax.plot([x,x],[y-.31,y-.38],color='#555',lw=.4)
 ax.text(-3.2,n-.05,'Architecture',ha='right',va='bottom',fontsize=13,fontweight='bold')
 ax.text(42,n-.05,'Exact 84-pentapeptide motif arrangement (R5)',ha='center',va='bottom',fontsize=13,fontweight='bold')
 fig.legend(handles=[Patch(facecolor=colors[m],label=m) for m in ('VG','IG','KG')],loc='upper right',bbox_to_anchor=(.965,.99),frameon=False,ncol=3,fontsize=11.5)
 fig.text(.66,.027,'Motif position:  1                         21                         42                         63                         84',ha='center',fontsize=10)
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 for ext in ('pdf','png','svg'):
  fig.savefig(output.with_suffix('.'+ext),dpi=dpi,bbox_inches='tight',pad_inches=.16)
 plt.close(fig)
 print(f'Wrote {len(keys)} verified R5 designs + {int(random_label)} uncoloured random control to {output.parent}')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--definitions',default=str(Path(__file__).with_name('plot_dr6_architecture_focus_v12_readable_balanced.py')))
 parser.add_argument('--outdir',default='DR6_methods_library_design')
 parser.add_argument('--dpi',type=int,default=240)
 args=parser.parse_args()
 mod=load_design_module(Path(args.definitions))
 make_plot(mod,Path(args.outdir)/'DR6_methods_R5_architecture_overview',dpi=args.dpi)
