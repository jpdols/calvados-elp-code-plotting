#!/usr/bin/env python
"""Plot DR5 summaries against full-residue SCD. Requires numpy/pandas/matplotlib.
Use --fasta to calculate all SCD values from actual sequences. Without it,
regular architectures are reconstructed from their names; shuffle is supplied.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ARCH = [
    ('alternating', 'Alternating', 'o'),
    ('smallblocks_3', 'Small blocks (3)', 's'),
    ('mediumblocks_7', 'Medium blocks (7)', '^'),
    ('randomshuffle', 'Random shuffle', 'D'),
    ('largeblocks_14', 'Large blocks (14)', 'v'),
    ('Ecenter_Kends', 'E centre / K ends', '<'),
    ('Kcenter_Eends', 'K centre / E ends', '>'),
    ('diblock', 'Diblock', 'P'),
]
COLOURS = {300: '#0064C8', 370: '#98F522'}
SHUFFLE = '''VPGEGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGEG
VPGKGVPGKGVPGKGVPGEGVPGEGVPGEGVPGEGVPGEGVPGKGVPGEGVPGEGVPGEGVPGEGVPGEGVPGEGVPGEG
VPGKGVPGKGVPGKGVPGEGVPGEGVPGKGVPGEGVPGKGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGKGVPGKG
VPGKGVPGEGVPGEGVPGEGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGEGVPGEG
VPGEGVPGEGVPGEGVPGKGVPGEGVPGEGVPGEGVPGEGVPGKGVPGEGVPGKGVPGEGVPGKGVPGEGVPGEGVPGEG
VPGKGVPGKGVPGEGVPGKG'''.replace('\n', '')


def scd(sequence):
    # Full sequence, including neutral residues. Terminal charges excluded.
    q = np.array([{'E': -1, 'K': 1}.get(c, 0) for c in sequence])
    i, j = np.tril_indices(len(q), -1)
    return float(np.sum(q[i] * q[j] * np.sqrt(i-j)) / len(q))


def architecture(name):
    return next((key for key, _, _ in ARCH if name.endswith('_'+key) or name == key), None)


def sequences(fasta=None):
    if fasta:
        records, name, parts = {}, None, []
        for line in fasta.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    records[name] = ''.join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError('FASTA sequence before first header.')
                parts.append(line)
        if name is not None:
            records[name] = ''.join(parts).upper()
        result = {}
        for name, seq in records.items():
            key = architecture(name)
            if key:
                if key in result:
                    raise ValueError('Duplicate FASTA architecture: '+key)
                result[key] = seq
        missing = {key for key, _, _ in ARCH} - set(result)
        if missing:
            raise ValueError('Missing FASTA architectures: '+', '.join(sorted(missing)))
    else:
        motifs = {
            'alternating': 'EK'*42,
            'smallblocks_3': ('E'*3+'K'*3)*14,
            'mediumblocks_7': ('E'*7+'K'*7)*6,
            'largeblocks_14': ('E'*14+'K'*14)*3,
            'Ecenter_Kends': 'K'*21+'E'*42+'K'*21,
            'Kcenter_Eends': 'E'*21+'K'*42+'E'*21,
            'diblock': 'E'*42+'K'*42,
        }
        result = {key: ''.join('VPG'+c+'G' for c in pattern) for key, pattern in motifs.items()}
        result['randomshuffle'] = SHUFFLE
        print('Using ideal regular motif arrangements; use --fasta to verify actual sequences.')
    for key, seq in result.items():
        if len(seq) != 420 or seq.count('E') != 42 or seq.count('K') != 42:
            raise ValueError(f'{key}: expected 420 residues, 42 E and 42 K.')
        if set(seq) - set('VPGEK'):
            raise ValueError(f'{key}: unexpected residues.')
    return result


def read_summary(path):
    d = pd.read_csv(path)
    required = {'sequence', 'temperature', 'threshold_nm', 'mean_rg_nm',
        'fraction_compact', 'kcollapse_ns_inv', 'kescape_ns_inv',
        'iqr_rg_nm', 'central90_rg_nm', 'mean_compact_rg_nm'}
    if required - set(d.columns):
        raise ValueError('Missing columns: '+', '.join(sorted(required-set(d.columns))))
    d['architecture'] = d.sequence.astype(str).map(architecture)
    d['T'] = pd.to_numeric(d.temperature.astype(str).str.replace('K', '', regex=False).str.strip())
    if d.architecture.isna().any() or len(d) != 16 or set(d['T']) != {300, 370}:
        raise ValueError('Expected eight known architectures at 300 and 370 K.')
    if d.duplicated(['architecture', 'T']).any():
        raise ValueError('Expected one trajectory per architecture/temperature.')
    if not np.allclose(d.threshold_nm, 4.61):
        raise ValueError('Recalculate the summary with threshold 4.61 nm.')
    return d.set_index(['architecture', 'T'])


def panel(ax, d, scores, metric, label, threshold=False, fraction=False):
    for i, (key, _, _) in enumerate(ARCH):
        a, b = [float(d.loc[(key, t), metric]) for t in (300, 370)]
        if np.isfinite(a) and np.isfinite(b):
            ax.add_patch(Rectangle((i-.17, min(a,b)), .34, abs(b-a),
                facecolor='#E8E8E8', edgecolor='none', zorder=1))
            ax.plot([i-.08, i+.08], [a,b], color='#B5B5B5', linewidth=1, zorder=2)
        for t, value, dx in [(300,a,-.08),(370,b,.08)]:
            ax.scatter(i+dx, value, s=25, color=COLOURS[t],
                       edgecolor='white', linewidth=.55, zorder=3)
    labels = {
        'alternating':'Alternating', 'smallblocks_3':'Small blocks\n(3)',
        'mediumblocks_7':'Medium blocks\n(7)', 'randomshuffle':'Random\nshuffle',
        'largeblocks_14':'Large blocks\n(14)', 'Ecenter_Kends':'E centre\nK ends',
        'Kcenter_Eends':'K centre\nE ends', 'diblock':'Diblock'}
    ax.set_xticks(range(8), [labels[key]+f'\nSCD {scores[key]:.2f}' for key,_,_ in ARCH])
    ax.tick_params(axis='x', labelsize=7, length=3)
    ax.set_xlim(-.5,7.5)
    ax.set_ylabel(label, labelpad=7)
    ax.grid(axis='y', linestyle=':', color='#E5E5E5', linewidth=.65)
    ax.set_axisbelow(True)
    ax.margins(y=.15)
    if threshold:
        ax.axhline(4.61, color='#888888', linestyle='--', linewidth=.8)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo,4.45),hi)
    if fraction:
        ax.set_ylim(0,1)
        ax.axhline(.5,color='#AAAAAA',linestyle='--',linewidth=.7)


def finish(fig, axes, outdir, name):
    handles = [Line2D([],[],marker='o',linestyle='',color=COLOURS[t],
        markeredgecolor='white',markersize=5,label=f'{t} K') for t in (300,370)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.995),
               ncol=2,frameon=False)
    fig.subplots_adjust(left=.15,right=.98,top=.94,bottom=.10,hspace=.68)
    for ax,letter in zip(axes,'ABCDE'):
        box=ax.get_position()
        fig.text(.025,box.y1-.005,letter,fontsize=11,fontweight='bold',va='top')
    for extension in ('pdf','svg','png'):
        fig.savefig(outdir/f'{name}.{extension}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, default=Path('EGKG_summary/summary_per_replicate.csv'))
    p.add_argument('--outdir', type=Path, default=Path('EGKG_plots_SCD_labels'))
    p.add_argument('--fasta', type=Path, help='Optional actual sequences for all eight architectures.')
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':9, 'legend.fontsize':7, 'axes.linewidth':.7,
        'axes.spines.top':False, 'axes.spines.right':False,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.dpi':450})
    seqs = sequences(args.fasta)
    scores = {key: scd(seq) for key, seq in seqs.items()}
    d = read_summary(args.summary)
    pd.DataFrame([{'architecture':key, 'SCD':scores[key], 'sequence':seqs[key],
        'sequence_source':'FASTA' if args.fasta else 'supplied shuffle / ideal regular patterns'}
        for key in sorted(scores, key=scores.get, reverse=True)]).to_csv(args.outdir/'SCD_values.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0))
    panel(axes[0], d, scores, 'mean_rg_nm', r'Mean $R_g$ (nm)', threshold=True)
    panel(axes[1], d, scores, 'fraction_compact', 'Collapsed fraction', fraction=True)
    finish(fig, axes, args.outdir, 'DR5_01_population_and_mean_Rg')
    fig, axes = plt.subplots(5, 1, figsize=(7.4, 12.0))
    specs = [
        ('kcollapse_ns_inv', r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)'),
        ('kescape_ns_inv', r'$k_{\mathrm{escape}}$ (ns$^{-1}$)'),
        ('iqr_rg_nm', r'$R_g$ central 50% width (nm)'),
        ('central90_rg_nm', r'$R_g$ central 90% width (nm)'),
        ('mean_compact_rg_nm', r'Mean collapsed-state $R_g$ (nm)'),
    ]
    for ax, (metric, label) in zip(axes, specs):
        panel(ax, d, scores, metric, label)
    finish(fig, axes, args.outdir, 'DR5_02_rates_widths_and_collapsed_Rg')
    (args.outdir/'PLOT_NOTES.txt').write_text(
        'SCD calculated over all 420 residues: E=-1, K=+1, other residues=0. No terminal charges.\n'
        'Equally spaced architecture categories sorted by raw SCD; labels give actual SCD. No numerical x scale.\n'
        'Architecture names and SCD values appear below each panel. Lines connect temperatures, not uncertainty. One trajectory per condition.\n'
        'Central 50% width=P75-P25; central 90% width=P95-P5. These are not confidence intervals.\n'
        'Collapsed states: Rg<4.61 nm. Rates are unsmoothed threshold-crossing rates.\n'
        'Regular sequences are ideal reconstructions unless --fasta is supplied.\n', encoding='utf-8')
    print(f'Saved PDF, SVG, PNG and SCD_values.csv in {args.outdir.resolve()}')


if __name__ == '__main__':
    main()
