#!/usr/bin/env python
"""DR5: four 370 K observables against continuous, full-residue SCD.

Outputs one thesis-ready 2x2 PDF/SVG/PNG figure and CSV with fit statistics.
Dependencies: numpy, pandas, matplotlib.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
SHUFFLE = '''VPGEGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGEG
VPGKGVPGKGVPGKGVPGEGVPGEGVPGEGVPGEGVPGEGVPGKGVPGEGVPGEGVPGEGVPGEGVPGEGVPGEGVPGEG
VPGKGVPGKGVPGKGVPGEGVPGEGVPGKGVPGEGVPGKGVPGKGVPGKGVPGEGVPGKGVPGKGVPGKGVPGKGVPGKG
VPGKGVPGEGVPGEGVPGEGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGEGVPGKGVPGKGVPGEGVPGEG
VPGEGVPGEGVPGEGVPGKGVPGEGVPGEGVPGEGVPGEGVPGKGVPGEGVPGKGVPGEGVPGKGVPGEGVPGEGVPGEG
VPGKGVPGKGVPGEGVPGKG'''.replace('\n', '')

METRICS = [
    ('mean_rg_nm', r'Mean $R_g$ (nm)'),
    ('mean_compact_rg_nm', r'Mean compact-state $R_g$ (nm)'),
    ('kcollapse_ns_inv', r'$k_{\mathrm{collapse}}$ (ns$^{-1}$)'),
    ('kescape_ns_inv', r'$k_{\mathrm{escape}}$ (ns$^{-1}$)'),
]


def architecture(name):
    return next((key for key, _, _ in ARCH if str(name).endswith('_' + key) or str(name) == key), None)


def scd(sequence):
    q = np.array([{'E': -1, 'K': 1}.get(c, 0) for c in sequence], dtype=float)
    i, j = np.tril_indices(len(q), -1)
    return float(np.sum(q[i] * q[j] * np.sqrt(i - j)) / len(q))


def sequences(fasta=None):
    if fasta is not None:
        records, name, parts = {}, None, []
        for line in fasta.read_text(encoding='utf-8').splitlines():
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
                    raise ValueError('Duplicate FASTA architecture: ' + key)
                result[key] = seq
        missing = {key for key, _, _ in ARCH} - set(result)
        if missing:
            raise ValueError('Missing FASTA architectures: ' + ', '.join(sorted(missing)))
    else:
        motifs = {
            'alternating': 'EK' * 42,
            'smallblocks_3': ('E' * 3 + 'K' * 3) * 14,
            'mediumblocks_7': ('E' * 7 + 'K' * 7) * 6,
            'largeblocks_14': ('E' * 14 + 'K' * 14) * 3,
            'Ecenter_Kends': 'K' * 21 + 'E' * 42 + 'K' * 21,
            'Kcenter_Eends': 'E' * 21 + 'K' * 42 + 'E' * 21,
            'diblock': 'E' * 42 + 'K' * 42,
        }
        result = {key: ''.join('VPG' + c + 'G' for c in pattern)
                  for key, pattern in motifs.items()}
        result['randomshuffle'] = SHUFFLE
        print('Note: ideal regular sequences reconstructed; pass --fasta to use actual sequences.')
    for key, seq in result.items():
        if len(seq) != 420 or seq.count('E') != 42 or seq.count('K') != 42:
            raise ValueError(f'{key}: expected 420 residues with 42 E and 42 K.')
        if set(seq) - set('VPGEK'):
            raise ValueError(f'{key}: unexpected residue(s).')
    return result


def read_summary(path):
    d = pd.read_csv(path)
    needed = {'sequence', 'temperature', 'threshold_nm'} | {key for key, _ in METRICS}
    missing = needed - set(d.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    d['architecture'] = d['sequence'].astype(str).map(architecture)
    d['T'] = pd.to_numeric(d['temperature'].astype(str).str.replace('K', '', regex=False).str.strip())
    d = d[d['T'] == 370].copy()
    if d['architecture'].isna().any() or len(d) != 8 or d['architecture'].nunique() != 8:
        raise ValueError('Expected exactly one 370 K row for each of eight known architectures.')
    if not np.allclose(pd.to_numeric(d['threshold_nm']), 4.61):
        raise ValueError('Expected compact-state threshold 4.61 nm.')
    for metric, _ in METRICS:
        d[metric] = pd.to_numeric(d[metric], errors='coerce')
    return d.set_index('architecture')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('DR5_SCD_370K_plots'))
    p.add_argument('--fasta', type=Path, help='Actual FASTA for all eight architectures (recommended).')
    p.add_argument('--dpi', type=int, default=400)
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seqs = sequences(args.fasta)
    scores = {key: scd(seq) for key, seq in seqs.items()}
    d = read_summary(args.summary)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 19,
        'axes.labelsize': 21, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
        'legend.fontsize': 17, 'axes.linewidth': 1.4,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 10.1), sharex=True)
    fig.subplots_adjust(left=.14, right=.985, bottom=.245, top=.97, wspace=.42, hspace=.29)
    x = np.array([scores[key] for key, _, _ in ARCH], dtype=float)
    fit_rows, point_rows = [], []
    for ax, (metric, ylabel), letter in zip(axs.flat, METRICS, 'ABCD'):
        y = np.array([float(d.loc[key, metric]) for key, _, _ in ARCH])
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3 or np.unique(x[valid]).size < 2:
            raise ValueError(f'Insufficient finite data for fitting {metric}.')
        # Ordinary least-squares straight-line fit to all architecture points.
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        predicted = slope*x[valid] + intercept
        ss_res = np.sum((y[valid]-predicted)**2)
        ss_tot = np.sum((y[valid]-np.mean(y[valid]))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        r = np.corrcoef(x[valid], y[valid])[0,1] if ss_tot > 0 else np.nan
        xx = np.linspace(x.min(), x.max(), 250)
        ax.plot(xx, slope*xx + intercept, color='#555555', lw=2.6,
                zorder=2, label='Linear fit')
        for key, label, marker in ARCH:
            val = float(d.loc[key, metric])
            if np.isfinite(val):
                ax.scatter(scores[key], val, s=150, marker=marker,
                           color='#98F522', edgecolor='#262626',
                           linewidth=.9, zorder=4)
            point_rows.append({'architecture': key, 'SCD': scores[key],
                               'metric': metric, 'value': val, 'temperature_K': 370})
        ax.text(.035, .965, letter, transform=ax.transAxes, ha='left', va='top',
                fontsize=25, weight='bold')
        ax.text(.965, .055, rf'$R^2 = {r2:.2f}$', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=18, color='#444444')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#DEDEDE', lw=.85, alpha=.9)
        ax.tick_params(direction='out', length=6, width=1.3)
        ax.margins(x=.085, y=.17)
        # Keep the signed SCD values, but display weak to strong segregation left-to-right.
        fit_rows.append({'metric': metric, 'n_points': int(valid.sum()),
                         'slope_per_SCD': slope, 'intercept': intercept,
                         'Pearson_r': r, 'R_squared': r2})
    # Explicitly set reversed limits on the shared SCD axis:
    # 0 (weak segregation) at left, increasingly negative SCD to the right.
    padding = 0.055 * (float(np.max(x)) - float(np.min(x)))
    axs[0, 0].set_xlim(float(np.max(x)) + padding,
                       float(np.min(x)) - padding)
    assert axs[0, 0].get_xlim()[0] > axs[0, 0].get_xlim()[1]
    for ax in axs[1, :]:
        ax.set_xlabel('Sequence charge decoration (SCD)')
    # Actual signed SCD tick values, decreasing left-to-right (more segregation).
    legend = [Line2D([], [], marker=marker, linestyle='None', color='#98F522',
                     markeredgecolor='#262626', markersize=11, label=label)
              for _, label, marker in ARCH]
    legend.append(Line2D([], [], color='#555555', lw=2.6, label='Linear fit'))
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.56, .012), columnspacing=1.15,
               handletextpad=.40, labelspacing=.7)
    stem = 'DR5_SCD_370K_four_metrics_linear_fit'
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(args.outdir / f'{stem}.{ext}', dpi=args.dpi,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.outdir / 'DR5_SCD_linear_fit_statistics.csv', index=False)
    pd.DataFrame(point_rows).to_csv(args.outdir / 'DR5_SCD_370K_points.csv', index=False)
    pd.DataFrame([{'architecture': k, 'SCD': scores[k], 'sequence': seqs[k]}
                  for k, _, _ in ARCH]).to_csv(args.outdir / 'DR5_SCD_sequences.csv', index=False)
    print('Saved figure and CSV tables to:', args.outdir.resolve())
    print(pd.DataFrame(fit_rows).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

if __name__ == '__main__':
    main()
