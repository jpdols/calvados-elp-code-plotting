# plot_lambda_chain.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap

RESIDUES = Path(r"E:\Master thesis\calvados_tea\runs\scale1\E_AI_library\EAI_arch26_full_AA_scramble\rep01\input\residues_gamma_6p0_370K_argdefault_minus.csv")

seq = """
VVPGGPPVGGVGPVPAGPPIVVVGVGPVPPIIIIVGIIVGVGIPVPGAGVGGVVPPGVVVGPGVVVVVAIPGPAVVGGGV
VIAGVGVVVIGVGGVGVVGIGVPAAVPPGVGPAPVGPPGVVPPIPGGVPAVPVVGVVVPGVGGVGGVGVGVGPGVVGIVA
VGGGGPVPGGPGPGPGAPGVVGGPGPVVVIIPGVGGVPVAPGPVVGGGPGVVGVVGIGIVPGGVVVPGVGVGGPVPIIPV
PGVGGAAGPVIGPGPVIGGIIVVGAGAPGGGIGPPVPVPVIAGPPPPPVGPIVGGGGGVPPGAGVVGVPIVGVGVAVVGA
VVPPVGGGGGVGVGPVGVGVPGIGGVGGPPPVAGGGGIVVVVGPGVGIPGPPGGVVIIGGGIGGVVPGVAGVVGVVVGGP
GVAGPVAGVVGGGIPVPVPI
"""

seq = "".join(seq.split())

df = pd.read_csv(RESIDUES)
lambda_map = dict(zip(df["one"], df["lambdas"]))

values = [lambda_map[aa] for aa in seq]

n_per_row = 70
rows = textwrap.wrap(seq, n_per_row)

vmin = min(lambda_map.values())
vmax = max(lambda_map.values())

cmap = "viridis"
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(16, 5.5))

for y, row in enumerate(rows):
    for x, aa in enumerate(row):
        lam = lambda_map[aa]
        ax.scatter(x, -y, s=95, c=[lam], cmap=cmap, norm=norm)
        ax.text(x, -y, aa, ha="center", va="center", fontsize=6, color="white")

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("EAI_arch26_full_AA_scramble — residue λ at 370 K", fontsize=14)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("CALVADOS λ / hydrophobic stickiness")

plt.tight_layout()
plt.savefig("EAI_arch26_lambda_chain.png", dpi=300)
plt.savefig("EAI_arch26_lambda_chain.svg")
plt.show()