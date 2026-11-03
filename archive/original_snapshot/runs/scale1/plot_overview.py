import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# LOAD DATA
# ============================================================

csv_path = r"E:\Master thesis\calvados_tea\runs\scale1\combined_temperature_summary_v4.csv"
df = pd.read_csv(csv_path)

# Only scale = 6.0
df = df[df["scale_factor"] == 6.0].copy()

# Sort for nice lines
df = df.sort_values(["construct", "temperature_K"])

# ============================================================
# VISUAL SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# Fixed colors per construct
# Change any colors here if you want
color_map = {
    "E_AI": "#d62728",            # red
    "E_AI_triblock": "#ff7f0e",   # orange
    "E_AE": "#1f77b4",            # blue
    "E_AE_triblock": "#17becf",   # cyan
    "E_KE": "#2ca02c",            # green
    "E_KI": "#9467bd",            # purple
    "PTMA": "#7f7f7f",            # grey
    "SE_E": "#8c564b",            # brown
    "SE_I": "#e377c2",            # pink
}

marker_map = {
    "E_AI": "o",
    "E_AI_triblock": "s",
    "E_AE": "o",
    "E_AE_triblock": "s",
    "E_KE": "^",
    "E_KI": "D",
    "PTMA": "v",
    "SE_E": "P",
    "SE_I": "X",
}

# Fallback color/marker if a construct is not in the maps
default_color = "black"
default_marker = "o"

# Nice plotting order
preferred_order = [
    "E_AI",
    "E_AI_triblock",
    "E_AE",
    "E_AE_triblock",
    "E_KI",
    "E_KE",
    "SE_I",
    "SE_E",
    "PTMA",
]

present = [x for x in preferred_order if x in df["construct"].unique()]
remaining = sorted([x for x in df["construct"].unique() if x not in preferred_order])
plot_order = present + remaining


def style_axes(ax):
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(df["temperature_K"].min() - 2, df["temperature_K"].max() + 2)


def plot_metric(
    ycol: str,
    ylabel: str,
    title: str,
    filename: str,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for construct in plot_order:
        g = df[df["construct"] == construct].sort_values("temperature_K")
        ax.plot(
            g["temperature_K"],
            g[ycol],
            label=construct,
            color=color_map.get(construct, default_color),
            marker=marker_map.get(construct, default_marker),
            linewidth=2.4,
            markersize=6.5,
        )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axes(ax)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.show()


# ============================================================
# PLOTS
# ============================================================

plot_metric(
    ycol="mean_rg_nm",
    ylabel="Mean Rg (nm)",
    title="Radius of gyration vs Temperature (scale = 6)",
    filename="rg_vs_temperature_scale6_improved.png",
)

plot_metric(
    ycol="fraction_collapsed_percent",
    ylabel="Fraction collapsed (%)",
    title="Fraction collapsed vs Temperature (scale = 6)",
    filename="fraction_collapsed_scale6_improved.png",
)

plot_metric(
    ycol="mean_dwell_ns",
    ylabel="Mean dwell time (ns)",
    title="Mean dwell time vs Temperature (scale = 6)",
    filename="mean_dwell_scale6_improved.png",
)

plot_metric(
    ycol="mean_abs_drg_per_ns_nm",
    ylabel="Mean |ΔRg| (nm/ns)",
    title="Chain dynamics vs Temperature (scale = 6)",
    filename="drg_scale6_improved.png",
)