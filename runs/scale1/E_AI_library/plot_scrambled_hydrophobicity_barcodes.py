# plot_scrambled_hydrophobicity_barcodes.py
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("architectures", nargs="+", help="Architecture folders")
ap.add_argument("--residues", required=True, help="Residue CSV with lambda values")
ap.add_argument("--rep", default="rep01", help="Replicate folder")
ap.add_argument("--out", default="scrambled_hydrophobicity_barcodes")
ap.add_argument("--cmap", default="viridis")
args = ap.parse_args()

res_df = pd.read_csv(args.residues)
lambda_map = dict(zip(res_df["one"], res_df["lambdas"]))

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
        block_lambda = np.mean([lambda_map[aa] for aa in block])
        row.append(block_lambda)

    rows.append(row)
    labels.append(arch.name)

if not rows:
    raise RuntimeError("No valid architectures found.")

barcode = np.array(rows)

out = Path(args.out)
out.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, max(3, 0.45 * len(labels))))

im = ax.imshow(
    barcode,
    aspect="auto",
    cmap=args.cmap,
    interpolation="nearest"
)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)

ax.set_xlabel("5-aa block position")
ax.set_ylabel("Architecture")
ax.set_title("Block hydrophobicity barcode of full AA scrambles")

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("Mean block λ at 370 K")

plt.tight_layout()

outfile = out / "scrambled_block_lambda_barcode.png"
plt.savefig(outfile, dpi=300)
plt.savefig(out / "scrambled_block_lambda_barcode.svg")

print(f"[OK] saved: {outfile}")
plt.show()