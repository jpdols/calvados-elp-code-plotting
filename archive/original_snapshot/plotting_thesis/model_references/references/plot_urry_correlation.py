from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# ============================================================
# SETTINGS
# ============================================================

URRY_FILE = Path(
    r"E:\Master thesis\calvados_tea\runs\scale1\references\Urry\Urry.xlsx"
)

RESIDUE_DIR = Path(
    r"E:\mnt\data\tea_fixed\output\residues"
)

OUTDIR = URRY_FILE.parent / "correlation_plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES = [300, 360, 370]

# Column names in Urry.xlsx
URRY_RESIDUE_COL = "Residue"
URRY_TT_COL = "Urry_Tt_C"


# ============================================================
# HELPERS
# ============================================================

def find_column(df, possibilities):
    """
    Find a column without depending too strongly on exact capitalization.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # Exact matches first
    for candidate in possibilities:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    # Partial matches second
    for col in df.columns:
        col_lower = str(col).strip().lower()

        for candidate in possibilities:
            if candidate.lower() in col_lower:
                return col

    raise ValueError(
        f"Could not find one of {possibilities}.\n"
        f"Available columns: {list(df.columns)}"
    )


def load_urry():
    """
    Load Urry Tt-based hydrophobicity values.
    """

    df = pd.read_excel(URRY_FILE)

    print("\nUrry Excel columns:")
    print(list(df.columns))

    residue_col = find_column(
        df,
        [
            URRY_RESIDUE_COL,
            "One-letter",
            "AA",
            "amino_acid",
        ]
    )

    tt_col = find_column(
        df,
        [
            URRY_TT_COL,
            "Tt",
            "T_t",
            "transition temperature",
        ]
    )

    out = df[[residue_col, tt_col]].copy()

    out.columns = [
        "Residue",
        "Urry_Tt_C",
    ]

    out["Residue"] = (
        out["Residue"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["Urry_Tt_C"] = pd.to_numeric(
        out["Urry_Tt_C"],
        errors="coerce"
    )

    out = out.dropna()

    # Important check:
    # one value per amino acid is required for the correlation.
    duplicates = out[
        out["Residue"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        print("\nWARNING: duplicate Urry residue entries:")
        print(duplicates)

        raise ValueError(
            "\nUrry.xlsx contains multiple values for one or more residues.\n"
            "Create/select one value per residue before calculating the "
            "correlation (important for ionization-state-dependent residues)."
        )

    return out


def load_calvados(T):
    """
    Load CALVADOS residue parameters for one temperature.
    """

    file = RESIDUE_DIR / (
        f"residues_gamma_6p0_{T}K_argdefault_minus.csv"
    )

    if not file.exists():
        raise FileNotFoundError(
            f"Could not find:\n{file}"
        )

    df = pd.read_csv(file)

    print(f"\n{T} K CALVADOS columns:")
    print(list(df.columns))

    residue_col = find_column(
        df,
        [
            "one",
            "residue",
            "AA",
            "amino_acid",
        ]
    )

    lambda_col = find_column(
        df,
        [
            "lambda",
        ]
    )

    out = df[[residue_col, lambda_col]].copy()

    out.columns = [
        "Residue",
        "lambda",
    ]

    out["Residue"] = (
        out["Residue"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out["lambda"] = pd.to_numeric(
        out["lambda"],
        errors="coerce"
    )

    out = out.dropna()

    return out


def get_data(T, urry):
    """
    Merge Urry and CALVADOS values by amino-acid code.
    """

    calvados = load_calvados(T)

    merged = pd.merge(
        urry,
        calvados,
        on="Residue",
        how="inner",
    )

    merged = merged.sort_values("Urry_Tt_C")

    if len(merged) < 3:
        raise ValueError(
            f"Only {len(merged)} residues matched at {T} K."
        )

    print(f"\nMatched residues at {T} K:")
    print(
        merged[
            ["Residue", "Urry_Tt_C", "lambda"]
        ].to_string(index=False)
    )

    return merged


def calculate_stats(df):

    x = df["Urry_Tt_C"].to_numpy()
    y = df["lambda"].to_numpy()

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


def add_scatter(ax, df, T, panel=None):

    x = df["Urry_Tt_C"].to_numpy()
    y = df["lambda"].to_numpy()

    stats = calculate_stats(df)

    # --------------------------------------------------------
    # Scatter
    # --------------------------------------------------------

    ax.scatter(
        x,
        y,
        s=55,
        zorder=3,
    )

    # --------------------------------------------------------
    # Linear regression
    # --------------------------------------------------------

    slope, intercept = np.polyfit(x, y, 1)

    xx = np.linspace(
        x.min(),
        x.max(),
        300,
    )

    yy = slope * xx + intercept

    ax.plot(
        xx,
        yy,
        linewidth=1.5,
        zorder=2,
    )

    # --------------------------------------------------------
    # Residue labels
    # --------------------------------------------------------

    for _, row in df.iterrows():

        ax.annotate(
            row["Residue"],
            (
                row["Urry_Tt_C"],
                row["lambda"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats_text = (
        rf"Pearson $r$ = {stats['pearson_r']:.3f}"
        "\n"
        rf"Spearman $\rho$ = {stats['spearman_rho']:.3f}"
        "\n"
        rf"$n$ = {len(df)}"
    )

    ax.text(
        0.97,
        0.96,
        stats_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
        ),
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Urry extrapolated $T_t$-based value ($^\circ$C)"
    )

    ax.set_ylabel(
        rf"CALVADOS $\lambda_i$ ({T} K)"
    )

    if panel is None:
        ax.set_title(
            f"{T} K",
            fontsize=12,
        )
    else:
        ax.set_title(
            f"{panel}  {T} K",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(
        alpha=0.20,
    )

    return stats


# ============================================================
# LOAD DATA
# ============================================================

urry = load_urry()

datasets = {}

for T in TEMPERATURES:
    datasets[T] = get_data(
        T,
        urry,
    )


# ============================================================
# SAVE MERGED DATA
# ============================================================

all_data = []

for T, df in datasets.items():

    temp = df.copy()

    temp.insert(
        0,
        "Temperature_K",
        T,
    )

    all_data.append(temp)

combined_data = pd.concat(
    all_data,
    ignore_index=True,
)

combined_data.to_csv(
    OUTDIR / "urry_calvados_merged_data.csv",
    index=False,
)


# ============================================================
# 1. SEPARATE PLOTS
# ============================================================

statistics = []

for T in TEMPERATURES:

    fig, ax = plt.subplots(
        figsize=(6.2, 5.0)
    )

    stats = add_scatter(
        ax,
        datasets[T],
        T,
    )

    fig.suptitle(
        "CALVADOS residue stickiness versus "
        "Urry ELP hydrophobicity scale",
        fontsize=13,
    )

    fig.tight_layout()

    fig.savefig(
        OUTDIR / f"urry_lambda_correlation_{T}K.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUTDIR / f"urry_lambda_correlation_{T}K.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    statistics.append(
        {
            "Temperature_K": T,
            **stats,
            "n": len(datasets[T]),
        }
    )


# ============================================================
# 2. DOUBLE FIGURE: 300 K + 370 K
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.0, 4.8),
    sharey=True,
)

add_scatter(
    axes[0],
    datasets[300],
    300,
    panel="A",
)

add_scatter(
    axes[1],
    datasets[370],
    370,
    panel="B",
)

# Only one y-label needed
axes[1].set_ylabel("")

fig.suptitle(
    "CALVADOS residue stickiness versus "
    "Urry ELP hydrophobicity scale",
    fontsize=14,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

fig.savefig(
    OUTDIR / "urry_lambda_correlation_300K_370K.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUTDIR / "urry_lambda_correlation_300K_370K.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats_df = pd.DataFrame(statistics)

stats_df.to_csv(
    OUTDIR / "urry_lambda_correlation_statistics.csv",
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n======================================")
print("CORRELATION SUMMARY")
print("======================================")

print(
    stats_df[
        [
            "Temperature_K",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
            "n",
        ]
    ].to_string(index=False)
)

print("\nPlots saved to:")
print(OUTDIR)