# plot_block_hydrophobicity_barcodes.py
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("architectures", nargs="+", help="Architecture folders")
ap.add_argument("--residues", required=True, help="Residue CSV with lambda values")
ap.add_argument("--rep", default="rep01", help="Replicate folder")
ap.add_argument("--out", default="block_hydrophobicity_barcodes")
args = ap.parse_args()

motif_colors = {
    "VPGVG": "#440154",  # low λ  purple
    "VPGIG": "#21918c",  # medium λ teal
    "IPAVG": "#fde725",  # high λ yellow
}

res_df = pd.read_csv(args.residues)
lambda_map = dict(zip(res_df["one"], res_df["lambdas"]))

def motif_lambda(motif):
    return np.mean([lambda_map[aa] for aa in motif])

motif_lambdas = {m: motif_lambda(m) for m in motif_colors}

rows = []
labels = []

for arch in args.architectures:
    arch = Path(arch)
    fasta = arch / args.rep / "input" / "idr.fasta"

    if not fasta.exists():
        print(f"[SKIP] missing fasta: {fasta}")
        continue

    lines = fasta.read_text().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))

    blocks = [seq[i:i+5] for i in range(0, len(seq), 5)]

    row = []
    for block in blocks:
        if block not in motif_colors:
            raise ValueError(f"Unknown motif {block} in {arch.name}")
        row.append(list(motif_colors).index(block))

    rows.append(row)
    labels.append(arch.name)

barcode = np.array(rows)

out = Path(args.out)
out.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, max(3, 0.38 * len(labels))))

cmap = plt.matplotlib.colors.ListedColormap([motif_colors[m] for m in motif_colors])

ax.imshow(
    barcode,
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-0.5,
    vmax=len(motif_colors)-0.5
)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

ax.set_xlabel("5-aa block position")
ax.set_ylabel("Architecture")
ax.set_title("Motif barcode of block-shuffled architectures")

legend_handles = [
    mpatches.Patch(
        color=motif_colors[m],
        label=f"{m}   mean λ={motif_lambdas[m]:.3f}"
    )
    for m in motif_colors
]

ax.legend(
    handles=legend_handles,
    title="5-aa motif",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    frameon=False
)

plt.tight_layout()

outfile = out / "motif_barcode.png"
plt.savefig(outfile, dpi=300)
plt.savefig(out / "motif_barcode.svg")

print(f"[OK] saved: {outfile}")
plt.show()