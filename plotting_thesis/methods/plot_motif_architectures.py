"""Reproduce R4 ELP motif architecture figures from explicit motif definitions.

Usage: python plot_motif_architectures.py --outdir figures
Dependencies: matplotlib (standard Python library otherwise).
Outputs: PNG, PDF and SVG for both R4 libraries.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Deliberately preserve the orange/pink of the source figures.
COLORS = {'VG':'#4D84D4', 'IG':'#5A8B42', 'EG':'#F18732', 'KG':'#BE568F'}

# id, descriptive label, literal expression. Parenthesized numbers are motif counts;
# ×N or a trailing count after a group repeats the entire group.
DR6 = [
(773,'Random shuffle', 'SHUFFLE'),
(774,'Even distribution', 'EVEN'),
(775,'Original Tk8 adapted', '(KG)1-(IG)14-(VG2-KG)3-(VG2-IG)-(VG3-KG)-(VG2-IG)-(VG-KG-IG)-(VG-KG)-(VG2-KG)-(KG-VG2)-(KG-VG)-(IG-KG-VG)-(IG-VG2)-(KG-VG3)-(IG-VG2)-(KG-VG2)3-(IG)14-(KG)1'),
(776,'Central IG, KG endcaps','(KG)8-(VG)17-(IG)34-(VG)17-(KG)8'),
(777,'Central IG, adjacent KG','(VG)17-(KG)8-(IG)34-(KG)8-(VG)17'),
(778,'KG embedded in IG','(VG)17-(IG)9-(KG)8-(IG)16-(KG)8-(IG)9-(VG)17'),
(779,'KG dispersed in IG','(VG)17-(IG2-KG)16-(IG)2-(VG)17'),
(780,'Central mixed sticker domain','(VG)17-(IG2-KG)8-(IG)2-(KG-IG2)8-(VG)17'),
(781,'Two IG blocks, KG endcaps','(KG)8-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9-(KG)8'),
(782,'Two IG blocks, interfacial KG','(VG)9-(IG)17-(KG)8-(VG)16-(KG)8-(IG)17-(VG)9'),
(783,'Two IG blocks, KG center','(VG)9-(IG)17-(VG)8-(KG)16-(VG)8-(IG)17-(VG)9'),
(784,'IG endcaps, KG center','(IG)17-(VG)17-(KG)16-(VG)17-(IG)17'),
(785,'KG-capped IG endblocks','(KG)8-(IG)17-(VG)34-(IG)17-(KG)8'),
(786,'IG outermost, KG inside','(IG)17-(KG)8-(VG)34-(KG)8-(IG)17'),
(787,'Three medium IG blocks, KG endcaps','(KG)8-(VG)6-(IG)11-(VG)11-(IG)12-(VG)11-(IG)11-(VG)6-(KG)8'),
(788,'Three IG blocks, interfacial KG','(VG)6-(IG)11-(KG)4-(VG)11-(KG)4-(IG)12-(KG)4-(VG)11-(KG)4-(IG)11-(VG)6'),
(789,'Three IG blocks, KG center','(VG)6-(IG)11-(VG)11-(IG)12-(KG)16-(IG)11-(VG)17'),
(790,'Four IG blocks, KG endcaps','(KG)8-(VG)4-(IG)9-(VG)8-(IG)8-(VG)10-(IG)8-(VG)8-(IG)9-(VG)4-(KG)8'),
(791,'IG one end, KG opposite','(IG)34-(VG)34-(KG)16'),
(792,'KG then IG, same end','(VG)34-(KG)16-(IG)34'),
(793,'IG then KG, same end','(VG)34-(IG)34-(KG)16'),
(794,'Asymmetric IG blocks, KG endcaps','(KG)8-(IG)22-(VG)17-(IG)12-(VG)17-(KG)8'),
(795,'Charged large IG domain','(IG)22-(KG)16-(VG)17-(IG)12-(VG)17'),
(796,'KG one end, symmetric IG','(KG)16-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9'),
(797,'Hydrophobic gradient','(KG)16-(VG)11-(IG)3-(VG)9-(IG)5-(VG)7-(IG)7-(VG)5-(IG)9-(VG)2-(IG)10'),
(798,'Symmetric IG gradient','(KG)8-(VG)7-(IG)4-(VG)5-(IG)6-(VG)5-(IG)14-(VG)5-(IG)6-(VG)5-(IG)4-(VG)7-(KG)8'),
(799,'Single IG block, KG center','(VG)17-(IG)17-(KG)16-(IG)17-(VG)17'),
(800,'KG clustered in VG','(VG)8-(KG)8-(VG)9-(IG)34-(VG)9-(KG)8-(VG)8'),
(801,'KG dispersed in VG','(VG2-KG)8-(VG)1-(IG)34-(VG)1-(KG-VG2)8'),
(802,'Four IG blocks, interfacial KG','(VG)4-(IG)9-(KG)4-(VG)8-(IG)8-(KG)4-(VG)10-(KG)4-(IG)8-(VG)8-(KG)4-(IG)9-(VG)4'),
]
DR5 = [
(591,'Random shuffle','SHUFFLE'),
(592,'Uniform opposite spacing','(EG4-IG2-KG4-IG2)×7'),
(593,'Uniform paired charges','(EG4-KG4-IG4)×7'),
(594,'KG-half / EG-half distribution','(IG2-KG4)×7-(IG2-EG4)×7'),
(595,'Terminal charge triblock','(EG)28-(IG)28-(KG)28'),
(596,'Paired-charge endcaps','[EG2-KG2-(EG4-KG4)×3]-(IG)28-[(KG4-EG4)×3-KG2-EG2]'),
(597,'Paired-charge terminal block','(EG4-KG4)×7-(IG)28'),
(598,'Segregated charge terminal block','(EG)28-(KG)28-(IG)28'),
(599,'Central paired charges','(IG)14-(EG4-KG4)×7-(IG)14'),
(600,'Central segregated charges','(IG)14-(EG)28-(KG)28-(IG)14'),
(601,'EG endcaps, KG center','(EG)14-(IG)14-(KG)28-(IG)14-(EG)14'),
(602,'KG endcaps, EG center','(KG)14-(IG)14-(EG)28-(IG)14-(KG)14'),
(603,'Two paired-charge clusters','(IG)7-[EG2-KG2-(EG4-KG4)×3]-(IG)14-[(KG4-EG4)×3-KG2-EG2]-(IG)7'),
(604,'Three paired-charge clusters','(EG4-KG4)×2-(IG)14-(EG4-KG4)×3-(IG)14-(EG4-KG4)×2'),
(605,'Internal blocks: close spacing','(IG)12-(EG)28-(IG)4-(KG)28-(IG)12'),
(606,'Internal blocks: medium spacing','(IG)10-(EG)28-(IG)8-(KG)28-(IG)10'),
(607,'Internal blocks: wide spacing','(IG)6-(EG)28-(IG)16-(KG)28-(IG)6'),
(608,'KG endcaps, distributed EG','(KG)14-(IG4-EG4)×7-(KG)14'),
(609,'EG endcaps, distributed KG','(EG)14-(IG4-KG4)×7-(EG)14'),
(610,'Outer KG, inner EG clusters','(KG)14-(IG)9-(EG)14-(IG)10-(EG)14-(IG)9-(KG)14'),
(611,'Outer EG, inner KG clusters','(EG)14-(IG)9-(KG)14-(IG)10-(KG)14-(IG)9-(EG)14'),
(612,'Alternating-polarity clusters','(KG)14-(IG)9-(EG)14-(IG)10-(KG)14-(IG)9-(EG)14'),
]

# Recursive grammar: sequence := (motif[count] | (sequence)[count] |
# [sequence][count])*, with optional '-' separators and × multipliers.
def expand(expr):
    expr = re.sub(r'\s+', '', expr)
    pos = 0
    def parse(end=None):
        nonlocal pos
        result = []
        while pos < len(expr):
            ch = expr[pos]
            if ch in '-':
                pos += 1
                continue
            if ch in ')]':
                if ch != end:
                    raise ValueError(f'Unexpected closing {ch} at {pos}: {expr}')
                pos += 1
                return result
            if ch in '([':
                pos += 1
                item = parse(')' if ch == '(' else ']')
            else:
                match = re.match(r'(VG|IG|EG|KG)', expr[pos:])
                if not match:
                    raise ValueError(f'Unexpected token at {pos}: {expr[pos:]}')
                item = [match.group(1)]
                pos += len(match.group(1))
            if pos < len(expr) and expr[pos] == '×':
                pos += 1
            match = re.match(r'\d+', expr[pos:])
            n = int(match.group()) if match else 1
            if match:
                pos += len(match.group())
            result.extend(item * n)
        if end is not None:
            raise ValueError(f'Unclosed {end}: {expr}')
        return result
    seq = parse()
    if pos != len(expr):
        raise ValueError(f'Not fully parsed: {expr}')
    return seq

GROUPS_DR6 = [
('Reference and distributed arrangements', [775,774,779,801,780]),
('KG clustering and local environment', [778,800,799,777,795]),
('Charge block positioning and symmetry', [785,776,784,786,796,791,792,793,794]),
('Hydrophobic-domain number and interfaces', [781,783,782,787,789,788,790,802]),
('Hydrophobic gradients', [797,798]),
('Randomized control', [773]),
]
GROUPS_DR5 = [
('Uniform distributions', [592,593,594]),
('Terminal and central charge domains', [595,596,597,598,599,600,601,602]),
('Multiple charge clusters and block spacing', [603,604,605,606,607]),
('Distributed charges and polarity arrangements', [608,609,610,611,612]),
('Randomized control', [591]),
]

def build(data, counts, seed=42):
    seqs = {}
    for number,label,expression in data:
        if expression == 'SHUFFLE':
            seq = sum(([motif]*n for motif,n in counts.items()), [])
            random.Random(seed + number).shuffle(seq)
        elif expression == 'EVEN':
            # Deterministic balanced distribution with exact counts.
            # Greedy proportional interleaving (not claimed to reproduce an unspecified manifest).
            used = {m: 0 for m in counts}
            seq = []
            total = sum(counts.values())
            for i in range(total):
                motif = max(
                    (m for m in counts if used[m] < counts[m]),
                    key=lambda m: ((i + 1) * counts[m] / total - used[m],
                                   -list(counts).index(m))
                )
                seq.append(motif)
                used[motif] += 1
        else:
            seq = expand(expression)
        got = Counter(seq)
        if got != Counter(counts):
            raise ValueError(f'{number} {label}: expected {counts}, got {dict(got)}; length {len(seq)}')
        seqs[number] = seq
    return seqs

def render(data, groups, seqs, colors, title, output, show_ids=False):
    lookup = {n:label for n,label,_ in data}
    total_rows = sum(len(ids)+1 for _,ids in groups)
    fig_h = 1.45 + total_rows * 0.31
    fig,ax = plt.subplots(figsize=(12.5,fig_h))
    fig.patch.set_facecolor('white')
    ax.set_xlim(-77, 86)
    ax.set_ylim(total_rows+0.7,-1.3)
    ax.axis('off')
    y=0
    for heading,ids in groups:
        ax.add_patch(Rectangle((-76,y-0.37),161,0.72,facecolor='#F0F2F4',edgecolor='none',zorder=0))
        ax.text(-73,y,heading,va='center',ha='left',fontsize=10.2,fontweight='bold',color='#263343')
        y+=1
        for n in ids:
            seq=seqs[n]
            label=lookup[n]
            ax.text(-2.5,y,label,va='center',ha='right',fontsize=8.2,color='#1D2730')
            if show_ids:ax.text(-74,y,str(n),va='center',ha='left',fontsize=7,color='#7B8490')
            for i,m in enumerate(seq):
                ax.add_patch(Rectangle((i,y-0.20),1,0.40,facecolor=colors[m],edgecolor='white',linewidth=0.07))
            y+=1
    ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=16)
    legend=[Patch(facecolor=colors[m],label=m) for m in colors]
    ax.legend(handles=legend,loc='lower right',bbox_to_anchor=(1.0,1.005),frameon=False,ncol=len(colors),fontsize=9)
    fig.text(0.60,0.015,'Each bar represents 84 pentapeptide motifs; each segment represents one motif.',ha='center',fontsize=8,color='#5E6670')
    fig.subplots_adjust(left=0.045,right=0.98,top=0.945,bottom=0.045)
    for ext in ['png','pdf','svg']:
        fig.savefig(str(output)+'.'+ext,dpi=300,facecolor='white',bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outdir',default='figures')
    parser.add_argument('--show-ids',action='store_true')
    args=parser.parse_args()
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    dr6=build(DR6,{'VG':34,'IG':34,'KG':16})
    dr5=build(DR5,{'IG':28,'EG':28,'KG':28})
    render(DR6,GROUPS_DR6,dr6,COLORS,'DR6 | VPGVG / VPGIG / VPGKG — R4 (34 / 34 / 16)',out/'DR6_R4_architectures',args.show_ids)
    render(DR5,GROUPS_DR5,dr5,COLORS,'DR5 | VPGIG / VPGEG / VPGKG — R4 (28 / 28 / 28)',out/'DR5_R4_architectures',args.show_ids)
    print(f'Validated {len(dr6)} DR6 and {len(dr5)} DR5 architectures; figures saved in {out.resolve()}')
if __name__=='__main__':main()
