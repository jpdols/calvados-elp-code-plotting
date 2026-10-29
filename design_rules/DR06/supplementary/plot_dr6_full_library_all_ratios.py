#!/usr/bin/env python
r"""DR6 full R1-R5 library: 30 architecture rows per ratio, 370 K.

Generates 5 supplementary figures with motif diagrams, collapse and escape
rates and compact-state Rg, using summary_per_replicate.csv. No results
are hardcoded. All explicit sequences are embedded below and validated.

Some exact motif orders were not provided in the original design table:
- random_shuffle (R1-R5): shown as neutral gray; NEVER a fabricated shuffle.
- even_distribution (R1-R4): neutral gray, exact motif order unavailable.
- original_Tk8_adapted (R3): supplied expression has 38 VG/38 IG/8 KG,
  inconsistent with R3 target 30 VG/46 IG/8 KG. Shown neutral gray.
These rows still show real metrics when available in the CSV.
"""
import argparse
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

COMPOSITIONS = {1:(46,30,8),2:(38,38,8),3:(30,46,8),4:(34,34,16),5:(30,30,24)}
COLORS = {'VG':'#4B80CF','IG':'#5C853F','KG':'#E000D5'}
METRICS = [('kcollapse_ns_inv',r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)'),
           ('kescape_ns_inv',r'$k_{\mathrm{escape}}$ (ns$^{-1}$)'),
           ('mean_compact_rg_nm',r'$\langle R_g\rangle_{\mathrm{compact}}$ (nm)')]

# Identical 30-architecture ordering for all five ratios.
ARCHITECTURES = [
 'random_shuffle','even_distribution','original_Tk8_adapted',
 'central_IG_KG_endcaps','central_IG_KG_adjacent','KG_embedded_in_IG',
 'KG_dispersed_in_IG','central_mixed_sticker_domain',
 'two_IG_blocks_KG_endcaps','two_IG_blocks_interfacial_KG',
 'two_IG_blocks_KG_center','IG_endcaps_KG_center','KG_capped_IG_endblocks',
 'IG_outermost_KG_inside','three_medium_IG_blocks_KG_endcaps',
 'three_IG_blocks_interfacial_KG','three_IG_blocks_KG_center',
 'four_IG_blocks_KG_endcaps','IG_one_end_KG_opposite','KG_then_IG_same_end',
 'IG_then_KG_same_end','asymmetric_IG_blocks_KG_endcaps',
 'charged_large_IG_domain','KG_one_end_symmetric_IG','hydrophobic_gradient',
 'symmetric_IG_gradient','single_IG_block_KG_center','KG_clustered_in_VG',
 'KG_dispersed_in_VG','four_IG_blocks_interfacial_KG'
]
# Source: uploaded R1-R4 architecture design table and verified R5 v12 manifest.
# Missing/invalid patterns are deliberately not fabricated.

PATTERNS = {(1, 'IG_endcaps_KG_center'): '(IG)15-(VG)23-(KG)8-(VG)23-(IG)15',
 (1, 'IG_one_end_KG_opposite'): '(IG)30-(VG)46-(KG)8',
 (1, 'IG_outermost_KG_inside'): '(IG)15-(KG)4-(VG)46-(KG)4-(IG)15',
 (1, 'IG_then_KG_same_end'): '(VG)46-(IG)30-(KG)8',
 (1, 'KG_capped_IG_endblocks'): '(KG)4-(IG)15-(VG)46-(IG)15-(KG)4',
 (1, 'KG_clustered_in_VG'): '(VG)11-(KG)4-(VG)12-(IG)30-(VG)12-(KG)4-(VG)11',
 (1, 'KG_dispersed_in_IG'): '(VG)23-(IG3-KG)8-(IG)6-(VG)23',
 (1, 'KG_dispersed_in_VG'): '(VG5-KG)4-(VG)3-(IG)30-(VG)3-(KG-VG5)4',
 (1, 'KG_embedded_in_IG'): '(VG)23-(IG)11-(KG)4-(IG)8-(KG)4-(IG)11-(VG)23',
 (1, 'KG_one_end_symmetric_IG'): '(KG)8-(VG)12-(IG)15-(VG)22-(IG)15-(VG)12',
 (1, 'KG_then_IG_same_end'): '(VG)46-(KG)8-(IG)30',
 (1, 'asymmetric_IG_blocks_KG_endcaps'): '(KG)4-(IG)20-(VG)23-(IG)10-(VG)23-(KG)4',
 (1, 'central_IG_KG_adjacent'): '(VG)23-(KG)4-(IG)30-(KG)4-(VG)23',
 (1, 'central_IG_KG_endcaps'): '(KG)4-(VG)23-(IG)30-(VG)23-(KG)4',
 (1, 'central_mixed_sticker_domain'): '(VG)23-(IG3-KG)4-(IG)6-(KG-IG3)4-(VG)23',
 (1, 'charged_large_IG_domain'): '(IG)20-(KG)8-(VG)23-(IG)10-(VG)23',
 (1, 'four_IG_blocks_KG_endcaps'): '(KG)4-(VG)6-(IG)8-(VG)11-(IG)7-(VG)12-(IG)7-(VG)11-(IG)8-(VG)6-(KG)4',
 (1, 'four_IG_blocks_interfacial_KG'): '(VG)6-(IG)8-(KG)2-(VG)11-(IG)7-(KG)2-(VG)12-(KG)2-(IG)7-(VG)11-(KG)2-(IG)8-(VG)6',
 (1, 'hydrophobic_gradient'): '(KG)8-(VG)16-(IG)2-(VG)12-(IG)4-(VG)8-(IG)6-(VG)6-(IG)8-(VG)4-(IG)10',
 (1, 'original_Tk8_adapted'): '(KG)1-(IG)12-(VG2-KG-VG4-KG-VG4-KG-VG3-IG-VG4-IG-VG4-IG-VG4-IG-VG4-IG-VG4-IG-VG3-KG-VG4-KG-VG4-KG-VG2)-(IG)12-(KG)1',
 (1, 'single_IG_block_KG_center'): '(VG)23-(IG)15-(KG)8-(IG)15-(VG)23',
 (1, 'symmetric_IG_gradient'): '(KG)4-(VG)12-(IG)3-(VG)7-(IG)5-(VG)4-(IG)14-(VG)4-(IG)5-(VG)7-(IG)3-(VG)12-(KG)4',
 (1, 'three_IG_blocks_KG_center'): '(VG)8-(IG)10-(VG)15-(IG)10-(KG)8-(IG)10-(VG)23',
 (1, 'three_IG_blocks_interfacial_KG'): '(VG)8-(IG)10-(KG)2-(VG)15-(KG)2-(IG)10-(KG)2-(VG)15-(KG)2-(IG)10-(VG)8',
 (1, 'three_medium_IG_blocks_KG_endcaps'): '(KG)4-(VG)8-(IG)10-(VG)15-(IG)10-(VG)15-(IG)10-(VG)8-(KG)4',
 (1, 'two_IG_blocks_KG_center'): '(VG)12-(IG)15-(VG)11-(KG)8-(VG)11-(IG)15-(VG)12',
 (1, 'two_IG_blocks_KG_endcaps'): '(KG)4-(VG)12-(IG)15-(VG)22-(IG)15-(VG)12-(KG)4',
 (1, 'two_IG_blocks_interfacial_KG'): '(VG)12-(IG)15-(KG)4-(VG)22-(KG)4-(IG)15-(VG)12',
 (2, 'IG_endcaps_KG_center'): '(IG)19-(VG)19-(KG)8-(VG)19-(IG)19',
 (2, 'IG_one_end_KG_opposite'): '(IG)38-(VG)38-(KG)8',
 (2, 'IG_outermost_KG_inside'): '(IG)19-(KG)4-(VG)38-(KG)4-(IG)19',
 (2, 'IG_then_KG_same_end'): '(VG)38-(IG)38-(KG)8',
 (2, 'KG_capped_IG_endblocks'): '(KG)4-(IG)19-(VG)38-(IG)19-(KG)4',
 (2, 'KG_clustered_in_VG'): '(VG)9-(KG)4-(VG)10-(IG)38-(VG)10-(KG)4-(VG)9',
 (2, 'KG_dispersed_in_IG'): '(VG)19-(IG4-KG)8-(IG)6-(VG)19',
 (2, 'KG_dispersed_in_VG'): '(VG4-KG)4-(VG)3-(IG)38-(VG)3-(KG-VG4)4',
 (2, 'KG_embedded_in_IG'): '(VG)19-(IG)15-(KG)4-(IG)8-(KG)4-(IG)15-(VG)19',
 (2, 'KG_one_end_symmetric_IG'): '(KG)8-(VG)10-(IG)19-(VG)18-(IG)19-(VG)10',
 (2, 'KG_then_IG_same_end'): '(VG)38-(KG)8-(IG)38',
 (2, 'asymmetric_IG_blocks_KG_endcaps'): '(KG)4-(IG)25-(VG)19-(IG)13-(VG)19-(KG)4',
 (2, 'central_IG_KG_adjacent'): '(VG)19-(KG)4-(IG)38-(KG)4-(VG)19',
 (2, 'central_IG_KG_endcaps'): '(KG)4-(VG)19-(IG)38-(VG)19-(KG)4',
 (2, 'central_mixed_sticker_domain'): '(VG)19-(IG4-KG)4-(IG)6-(KG-IG4)4-(VG)19',
 (2, 'charged_large_IG_domain'): '(IG)25-(KG)8-(VG)19-(IG)13-(VG)19',
 (2, 'four_IG_blocks_KG_endcaps'): '(KG)4-(VG)5-(IG)10-(VG)9-(IG)9-(VG)10-(IG)9-(VG)9-(IG)10-(VG)5-(KG)4',
 (2, 'four_IG_blocks_interfacial_KG'): '(VG)5-(IG)10-(KG)2-(VG)9-(IG)9-(KG)2-(VG)10-(KG)2-(IG)9-(VG)9-(KG)2-(IG)10-(VG)5',
 (2, 'hydrophobic_gradient'): '(KG)8-(VG)13-(IG)3-(VG)10-(IG)5-(VG)7-(IG)7-(VG)5-(IG)10-(VG)3-(IG)13',
 (2, 'original_Tk8_adapted'): '(KG)1-(IG)16-(VG3-KG-VG3-KG-VG3-KG-VG2-IG-VG3-IG-VG3-IG-VG4-IG-VG3-IG-VG3-IG-VG2-KG-VG3-KG-VG3-KG-VG3)-(IG)16-(KG)1',
 (2, 'single_IG_block_KG_center'): '(VG)19-(IG)19-(KG)8-(IG)19-(VG)19',
 (2, 'symmetric_IG_gradient'): '(KG)4-(VG)8-(IG)4-(VG)6-(IG)6-(VG)5-(IG)18-(VG)5-(IG)6-(VG)6-(IG)4-(VG)8-(KG)4',
 (2, 'three_IG_blocks_KG_center'): '(VG)6-(IG)13-(VG)13-(IG)12-(KG)8-(IG)13-(VG)19',
 (2, 'three_IG_blocks_interfacial_KG'): '(VG)6-(IG)13-(KG)2-(VG)13-(KG)2-(IG)12-(KG)2-(VG)13-(KG)2-(IG)13-(VG)6',
 (2, 'three_medium_IG_blocks_KG_endcaps'): '(KG)4-(VG)6-(IG)13-(VG)13-(IG)12-(VG)13-(IG)13-(VG)6-(KG)4',
 (2, 'two_IG_blocks_KG_center'): '(VG)10-(IG)19-(VG)9-(KG)8-(VG)9-(IG)19-(VG)10',
 (2, 'two_IG_blocks_KG_endcaps'): '(KG)4-(VG)10-(IG)19-(VG)18-(IG)19-(VG)10-(KG)4',
 (2, 'two_IG_blocks_interfacial_KG'): '(VG)10-(IG)19-(KG)4-(VG)18-(KG)4-(IG)19-(VG)10',
 (3, 'IG_endcaps_KG_center'): '(IG)23-(VG)15-(KG)8-(VG)15-(IG)23',
 (3, 'IG_one_end_KG_opposite'): '(IG)46-(VG)30-(KG)8',
 (3, 'IG_outermost_KG_inside'): '(IG)23-(KG)4-(VG)30-(KG)4-(IG)23',
 (3, 'IG_then_KG_same_end'): '(VG)30-(IG)46-(KG)8',
 (3, 'KG_capped_IG_endblocks'): '(KG)4-(IG)23-(VG)30-(IG)23-(KG)4',
 (3, 'KG_clustered_in_VG'): '(VG)7-(KG)4-(VG)8-(IG)46-(VG)8-(KG)4-(VG)7',
 (3, 'KG_dispersed_in_IG'): '(VG)15-(IG5-KG)8-(IG)6-(VG)15',
 (3, 'KG_dispersed_in_VG'): '(VG3-KG)4-(VG)3-(IG)46-(VG)3-(KG-VG3)4',
 (3, 'KG_embedded_in_IG'): '(VG)15-(IG)19-(KG)4-(IG)8-(KG)4-(IG)19-(VG)15',
 (3, 'KG_one_end_symmetric_IG'): '(KG)8-(VG)8-(IG)23-(VG)14-(IG)23-(VG)8',
 (3, 'KG_then_IG_same_end'): '(VG)30-(KG)8-(IG)46',
 (3, 'asymmetric_IG_blocks_KG_endcaps'): '(KG)4-(IG)30-(VG)15-(IG)16-(VG)15-(KG)4',
 (3, 'central_IG_KG_adjacent'): '(VG)15-(KG)4-(IG)46-(KG)4-(VG)15',
 (3, 'central_IG_KG_endcaps'): '(KG)4-(VG)15-(IG)46-(VG)15-(KG)4',
 (3, 'central_mixed_sticker_domain'): '(VG)15-(IG5-KG)4-(IG)6-(KG-IG5)4-(VG)15',
 (3, 'charged_large_IG_domain'): '(IG)30-(KG)8-(VG)15-(IG)16-(VG)15',
 (3, 'four_IG_blocks_KG_endcaps'): '(KG)4-(VG)4-(IG)12-(VG)7-(IG)11-(VG)8-(IG)11-(VG)7-(IG)12-(VG)4-(KG)4',
 (3, 'four_IG_blocks_interfacial_KG'): '(VG)4-(IG)12-(KG)2-(VG)7-(IG)11-(KG)2-(VG)8-(KG)2-(IG)11-(VG)7-(KG)2-(IG)12-(VG)4',
 (3, 'hydrophobic_gradient'): '(KG)8-(VG)10-(IG)3-(VG)8-(IG)6-(VG)6-(IG)9-(VG)4-(IG)12-(VG)2-(IG)16',
 (3, 'single_IG_block_KG_center'): '(VG)15-(IG)23-(KG)8-(IG)23-(VG)15',
 (3, 'symmetric_IG_gradient'): '(KG)4-(VG)6-(IG)5-(VG)5-(IG)7-(VG)4-(IG)22-(VG)4-(IG)7-(VG)5-(IG)5-(VG)6-(KG)4',
 (3, 'three_IG_blocks_KG_center'): '(VG)5-(IG)15-(VG)10-(IG)16-(KG)8-(IG)15-(VG)15',
 (3, 'three_IG_blocks_interfacial_KG'): '(VG)5-(IG)15-(KG)2-(VG)10-(KG)2-(IG)16-(KG)2-(VG)10-(KG)2-(IG)15-(VG)5',
 (3, 'three_medium_IG_blocks_KG_endcaps'): '(KG)4-(VG)5-(IG)15-(VG)10-(IG)16-(VG)10-(IG)15-(VG)5-(KG)4',
 (3, 'two_IG_blocks_KG_center'): '(VG)8-(IG)23-(VG)7-(KG)8-(VG)7-(IG)23-(VG)8',
 (3, 'two_IG_blocks_KG_endcaps'): '(KG)4-(VG)8-(IG)23-(VG)14-(IG)23-(VG)8-(KG)4',
 (3, 'two_IG_blocks_interfacial_KG'): '(VG)8-(IG)23-(KG)4-(VG)14-(KG)4-(IG)23-(VG)8',
 (4, 'IG_endcaps_KG_center'): '(IG)17-(VG)17-(KG)16-(VG)17-(IG)17',
 (4, 'IG_one_end_KG_opposite'): '(IG)34-(VG)34-(KG)16',
 (4, 'IG_outermost_KG_inside'): '(IG)17-(KG)8-(VG)34-(KG)8-(IG)17',
 (4, 'IG_then_KG_same_end'): '(VG)34-(IG)34-(KG)16',
 (4, 'KG_capped_IG_endblocks'): '(KG)8-(IG)17-(VG)34-(IG)17-(KG)8',
 (4, 'KG_clustered_in_VG'): '(VG)8-(KG)8-(VG)9-(IG)34-(VG)9-(KG)8-(VG)8',
 (4, 'KG_dispersed_in_IG'): '(VG)17-(IG2-KG)16-(IG)2-(VG)17',
 (4, 'KG_dispersed_in_VG'): '(VG2-KG)8-(VG)1-(IG)34-(VG)1-(KG-VG2)8',
 (4, 'KG_embedded_in_IG'): '(VG)17-(IG)9-(KG)8-(IG)16-(KG)8-(IG)9-(VG)17',
 (4, 'KG_one_end_symmetric_IG'): '(KG)16-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9',
 (4, 'KG_then_IG_same_end'): '(VG)34-(KG)16-(IG)34',
 (4, 'asymmetric_IG_blocks_KG_endcaps'): '(KG)8-(IG)22-(VG)17-(IG)12-(VG)17-(KG)8',
 (4, 'central_IG_KG_adjacent'): '(VG)17-(KG)8-(IG)34-(KG)8-(VG)17',
 (4, 'central_IG_KG_endcaps'): '(KG)8-(VG)17-(IG)34-(VG)17-(KG)8',
 (4, 'central_mixed_sticker_domain'): '(VG)17-(IG2-KG)8-(IG)2-(KG-IG2)8-(VG)17',
 (4, 'charged_large_IG_domain'): '(IG)22-(KG)16-(VG)17-(IG)12-(VG)17',
 (4, 'four_IG_blocks_KG_endcaps'): '(KG)8-(VG)4-(IG)9-(VG)8-(IG)8-(VG)10-(IG)8-(VG)8-(IG)9-(VG)4-(KG)8',
 (4, 'four_IG_blocks_interfacial_KG'): '(VG)4-(IG)9-(KG)4-(VG)8-(IG)8-(KG)4-(VG)10-(KG)4-(IG)8-(VG)8-(KG)4-(IG)9-(VG)4',
 (4, 'hydrophobic_gradient'): '(KG)16-(VG)11-(IG)3-(VG)9-(IG)5-(VG)7-(IG)7-(VG)5-(IG)9-(VG)2-(IG)10',
 (4, 'original_Tk8_adapted'): '(KG)1-(IG)14-(VG2-KG)3-(VG2-IG)-(VG3-KG)-(VG2-IG)-(VG-KG-IG)-(VG-KG)-(VG2-KG)-(KG-VG2)-(KG-VG)-(IG-KG-VG)-(IG-VG2)-(KG-VG3)-(IG-VG2)-(KG-VG2)3-(IG)14-(KG)1',
 (4, 'single_IG_block_KG_center'): '(VG)17-(IG)17-(KG)16-(IG)17-(VG)17',
 (4, 'symmetric_IG_gradient'): '(KG)8-(VG)7-(IG)4-(VG)5-(IG)6-(VG)5-(IG)14-(VG)5-(IG)6-(VG)5-(IG)4-(VG)7-(KG)8',
 (4, 'three_IG_blocks_KG_center'): '(VG)6-(IG)11-(VG)11-(IG)12-(KG)16-(IG)11-(VG)17',
 (4, 'three_IG_blocks_interfacial_KG'): '(VG)6-(IG)11-(KG)4-(VG)11-(KG)4-(IG)12-(KG)4-(VG)11-(KG)4-(IG)11-(VG)6',
 (4, 'three_medium_IG_blocks_KG_endcaps'): '(KG)8-(VG)6-(IG)11-(VG)11-(IG)12-(VG)11-(IG)11-(VG)6-(KG)8',
 (4, 'two_IG_blocks_KG_center'): '(VG)9-(IG)17-(VG)8-(KG)16-(VG)8-(IG)17-(VG)9',
 (4, 'two_IG_blocks_KG_endcaps'): '(KG)8-(VG)9-(IG)17-(VG)16-(IG)17-(VG)9-(KG)8',
 (4, 'two_IG_blocks_interfacial_KG'): '(VG)9-(IG)17-(KG)8-(VG)16-(KG)8-(IG)17-(VG)9',
 (5, 'IG_endcaps_KG_center'): '(IG)15-(VG)15-(KG)24-(VG)15-(IG)15',
 (5, 'IG_one_end_KG_opposite'): '(IG)30-(VG)30-(KG)24',
 (5, 'IG_outermost_KG_inside'): '(IG)15-(KG)12-(VG)30-(KG)12-(IG)15',
 (5, 'IG_then_KG_same_end'): '(VG)30-(IG)30-(KG)24',
 (5, 'KG_capped_IG_endblocks'): '(KG)12-(IG)15-(VG)30-(IG)15-(KG)12',
 (5, 'KG_clustered_in_VG'): '(VG)7-(KG)12-(VG)8-(IG)30-(VG)8-(KG)12-(VG)7',
 (5, 'KG_dispersed_in_IG'): '(VG)15-(IG-KG)24-(IG)6-(VG)15',
 (5, 'KG_dispersed_in_VG'): '(VG1-KG)12-(VG)3-(IG)30-(VG)3-(KG-VG1)12',
 (5, 'KG_embedded_in_IG'): '(VG)15-(IG)9-(KG)12-(IG)12-(KG)12-(IG)9-(VG)15',
 (5, 'KG_one_end_symmetric_IG'): '(KG)24-(VG)8-(IG)15-(VG)14-(IG)15-(VG)8',
 (5, 'KG_then_IG_same_end'): '(VG)30-(KG)24-(IG)30',
 (5, 'asymmetric_IG_blocks_KG_endcaps'): '(KG)12-(IG)20-(VG)15-(IG)10-(VG)15-(KG)12',
 (5, 'central_IG_KG_adjacent'): '(VG)15-(KG)12-(IG)30-(KG)12-(VG)15',
 (5, 'central_IG_KG_endcaps'): '(KG)12-(VG)15-(IG)30-(VG)15-(KG)12',
 (5, 'central_mixed_sticker_domain'): '(VG)15-(IG-KG)12-(IG)6-(KG-IG)12-(VG)15',
 (5, 'charged_large_IG_domain'): '(IG)20-(KG)24-(VG)15-(IG)10-(VG)15',
 (5, 'even_distribution'): '(VG-IG-KG)24-(VG-IG)6',
 (5, 'four_IG_blocks_KG_endcaps'): '(KG)12-(VG)4-(IG)8-(VG)7-(IG)7-(VG)8-(IG)7-(VG)7-(IG)8-(VG)4-(KG)12',
 (5, 'four_IG_blocks_interfacial_KG'): '(VG)4-(IG)8-(KG)6-(VG)7-(IG)7-(KG)6-(VG)8-(KG)6-(IG)7-(VG)7-(KG)6-(IG)8-(VG)4',
 (5, 'hydrophobic_gradient'): '(KG)24-(VG)10-(IG)2-(VG)8-(IG)4-(VG)6-(IG)6-(VG)4-(IG)8-(VG)2-(IG)10',
 (5, 'original_Tk8_adapted'): '(KG)2-(IG)12-(VG2-KG)5-IG-(VG2-KG)5-IG-((VG-KG)2-IG)4-(VG-KG)2-(IG)12-(KG)2',
 (5, 'single_IG_block_KG_center'): '(VG)15-(IG)15-(KG)24-(IG)15-(VG)15',
 (5, 'symmetric_IG_gradient'): '(KG)12-(VG)6-(IG)3-(VG)5-(IG)5-(VG)4-(IG)14-(VG)4-(IG)5-(VG)5-(IG)3-(VG)6-(KG)12',
 (5, 'three_IG_blocks_KG_center'): '(VG)5-(IG)10-(VG)10-(IG)10-(KG)24-(IG)10-(VG)15',
 (5, 'three_IG_blocks_interfacial_KG'): '(VG)5-(IG)10-(KG)6-(VG)10-(KG)6-(IG)10-(KG)6-(VG)10-(KG)6-(IG)10-(VG)5',
 (5, 'three_medium_IG_blocks_KG_endcaps'): '(KG)12-(VG)5-(IG)10-(VG)10-(IG)10-(VG)10-(IG)10-(VG)5-(KG)12',
 (5, 'two_IG_blocks_KG_center'): '(VG)8-(IG)15-(VG)7-(KG)24-(VG)7-(IG)15-(VG)8',
 (5, 'two_IG_blocks_KG_endcaps'): '(KG)12-(VG)8-(IG)15-(VG)14-(IG)15-(VG)8-(KG)12',
 (5, 'two_IG_blocks_interfacial_KG'): '(VG)8-(IG)15-(KG)12-(VG)14-(KG)12-(IG)15-(VG)8'}


def expand(expression):
    tokens=re.findall(r'VG|IG|KG|\d+|[()]',expression.replace('-','').replace(' ',''))
    if ''.join(tokens)!=expression.replace('-','').replace(' ',''):
        raise ValueError('Unexpected token in '+expression)
    pos=0
    def parse():
        nonlocal pos
        result=[]
        while pos<len(tokens) and tokens[pos]!=')':
            if tokens[pos]=='(':
                pos+=1; part=parse()
                if pos>=len(tokens) or tokens[pos]!=')':raise ValueError('Unmatched parentheses')
                pos+=1
            else:
                if tokens[pos] not in ('VG','IG','KG'):raise ValueError('Bad motif token')
                part=[tokens[pos]];pos+=1
            count=1
            if pos<len(tokens) and tokens[pos].isdigit():count=int(tokens[pos]);pos+=1
            result.extend(part*count)
        return result
    result=parse()
    if pos!=len(tokens):raise ValueError('Unexpected closing parenthesis')
    return result

def validate():
    for r in COMPOSITIONS:
        for key in ARCHITECTURES:
            if (r,key) not in PATTERNS:continue
            motifs=expand(PATTERNS[(r,key)])
            counts=tuple(motifs.count(k) for k in ('VG','IG','KG'))
            if len(motifs)!=84 or counts!=COMPOSITIONS[r]:
                raise ValueError(f'Invalid sequence R{r} {key}: {counts}, length={len(motifs)}')
    extra=set(PATTERNS)-{(r,k) for r in COMPOSITIONS for k in ARCHITECTURES}
    if extra:raise ValueError(f'Unexpected sequence keys: {extra}')
    print(f'Validated {len(PATTERNS)} exact sequences; {150-len(PATTERNS)} rows have neutral sequence placeholders.')

def load_data(path,temperature):
    df=pd.read_csv(path)
    needed={'family','architecture','ratio','temperature','threshold_nm'}|{k for k,_ in METRICS}
    if not needed.issubset(df.columns):raise ValueError(f'Missing columns: {needed-set(df.columns)}')
    df['family']=df['family'].astype(str).str.strip().str.upper()
    if 'architecture_key' in df.columns:
        key=df['architecture_key'].fillna(df['architecture'])
    else:key=df['architecture']
    df['architecture']=key.astype(str).str.strip()
    df['temperature']=pd.to_numeric(df['temperature'].astype(str).str.replace(r'\s*[Kk]$','',regex=True),errors='coerce')
    df['ratio_number']=pd.to_numeric(df['ratio'].astype(str).str.extract(r'^[Rr]?(\d+)$',expand=False),errors='coerce')
    df=df[(df.family=='VGIGKG')&(df.temperature==temperature)&df.ratio_number.isin(COMPOSITIONS)].copy()
    if df.empty:raise ValueError('No matching VGIGKG simulation data')
    if not np.allclose(pd.to_numeric(df.threshold_nm),4.61):
        raise ValueError('Unexpected threshold_nm: expected 4.61 for all selected rows')
    for key,_ in METRICS:df[key]=pd.to_numeric(df[key],errors='coerce')
    if 'replicate' not in df.columns:df['replicate']='single'
    if df.duplicated(['ratio_number','architecture','replicate']).any():
        raise ValueError('Duplicate replicate rows for an architecture')
    out=df.groupby(['ratio_number','architecture'],as_index=False).agg(
        **{key:(key,'mean') for key,_ in METRICS},n_replicates=('replicate','size'))
    return out

def draw_motifs(ax,motifs,y):
    for x,motif in enumerate(motifs):
        ax.add_patch(Rectangle((x,y-.30),1,.60,facecolor=COLORS[motif],edgecolor='none'))

def plot_ratio(data,ratio,temp,outdir,dpi):
    table=data[data.ratio_number==ratio].set_index('architecture')
    n=len(ARCHITECTURES)
    fig=plt.figure(figsize=(21.0,25.5))
    gs=fig.add_gridspec(1,5,width_ratios=[3.8,3.3,1.9,1.9,1.9],
                        left=.04,right=.985,top=.925,bottom=.045,wspace=.19)
    axes=[fig.add_subplot(gs[0,i]) for i in range(5)]
    lab,seq,*metrics=axes
    ys=np.arange(n)[::-1]
    for ax in axes:
        ax.set_ylim(-.7,n-.3)
    lab.set_xlim(0,1);lab.axis('off')
    seq.set_xlim(0,84);seq.set_yticks([])
    seq.set_xticks([0,21,42,63,84]);seq.tick_params(axis='x',labelsize=11)
    seq.set_xlabel('Pentapeptide motif position',fontsize=12)
    for side in seq.spines.values():side.set_visible(False)
    for i,(key,y) in enumerate(zip(ARCHITECTURES,ys)):
        if i%2==0:
            for ax in axes[1:]:ax.axhspan(y-.49,y+.49,color='#F5F6F8',zorder=-4)
        lab.text(.985,y,key.replace('_',' '),ha='right',va='center',fontsize=12.5,fontweight='semibold')
        if (ratio,key) in PATTERNS:
            draw_motifs(seq,expand(PATTERNS[(ratio,key)]),y)
        else:
            seq.add_patch(Rectangle((0,y-.30),84,.60,facecolor='#D7D9DD',edgecolor='none'))
            seq.text(42,y,'sequence not available',ha='center',va='center',fontsize=9.5,color='#454A52')
    lab.set_title('Architecture',fontsize=15,fontweight='bold',loc='right',pad=14)
    seq.set_title('Exact 84-motif design',fontsize=15,fontweight='bold',pad=14)
    for ax,(metric,title) in zip(metrics,METRICS):
        values=pd.to_numeric(table[metric],errors='coerce') if metric in table else pd.Series(dtype=float)
        valid=values.dropna()
        if len(valid):
            lo=float(valid.min());hi=float(valid.max());spread=max(hi-lo,abs(hi)*.04,.03)
            x0=max(0,lo-.35*spread);x1=hi+.65*spread
            mean=float(valid.mean())
            ax.axvline(mean,color='#8A929C',lw=1.15,ls=(0,(2,3)),zorder=1)
        else:x0,x1,mean=0,1,np.nan
        ax.set_xlim(x0,x1)
        for key,y in zip(ARCHITECTURES,ys):
            if key not in table.index or pd.isna(table.loc[key,metric]):continue
            value=float(table.loc[key,metric]);ax.plot(value,y,'o',ms=5.7,color='#8051AB',mec='white',mew=.4,zorder=4)
            if (value-x0)/(x1-x0)>.73:offset,align=-7,'right'
            else:offset,align=7,'left'
            ax.annotate(f'{value:.2f}',(value,y),xytext=(offset,0),textcoords='offset points',
                        ha=align,va='center',fontsize=10.5)
        ax.set_yticks([]);ax.grid(axis='x',alpha=.17)
        ax.tick_params(axis='x',labelsize=10)
        ax.set_title(title,fontsize=13.5,fontweight='bold',pad=14)
        for side in ('left','top','right'):ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(color=COLORS[k],label=k) for k in ('VG','IG','KG')]+[
        Patch(color='#D7D9DD',label='Sequence unavailable')],
        loc='upper left',bbox_to_anchor=(.04,.991),frameon=False,ncol=4,fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color='#8A929C',lw=1.15,ls=(0,(2,3)),
                               label=f'R{ratio} library mean (all available architectures)')],
               loc='upper right',bbox_to_anchor=(.985,.991),frameon=False,fontsize=11)
    vg,ig,kg=COMPOSITIONS[ratio]
    fig.text(.5,.959,f'R{ratio}: {vg} VG / {ig} IG / {kg} KG motifs  |  {temp} K',
             ha='center',fontsize=17,fontweight='bold')
    fig.text(.04,.014,'Gray tracks: exact simulated motif order unavailable or inconsistent in source table. '
             'Values are means over available replicates.',fontsize=10,color='#626A73')
    stem=f'DR6_full_library_R{ratio}_{temp}K'
    for ext in ('pdf','svg','png'):
        fig.savefig(outdir/f'{stem}.{ext}',dpi=dpi,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    missing=[key for key in ARCHITECTURES if key not in table.index]
    if missing:print(f'WARNING R{ratio}: missing data for {len(missing)} architectures: {missing}')
    print(f'R{ratio}: {n-len(missing)}/{n} architecture metrics found, saved {stem}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--outdir',type=Path,default=Path('DR6_full_library_R1_R5'))
    parser.add_argument('--temperature',type=int,default=370)
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args()
    validate();args.outdir.mkdir(parents=True,exist_ok=True)
    data=load_data(args.summary,args.temperature)
    data.sort_values(['ratio_number','architecture']).to_csv(args.outdir/'DR6_all_architecture_means.csv',index=False)
    for ratio in COMPOSITIONS:plot_ratio(data,ratio,args.temperature,args.outdir,args.dpi)
    print('Finished:',args.outdir.resolve())

if __name__=='__main__':main()
