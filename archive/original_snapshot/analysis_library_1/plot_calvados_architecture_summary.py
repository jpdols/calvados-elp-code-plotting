#!/usr/bin/env python3
"""
Publication-style plots for one CALVADOS motif-composition analysis folder.

Designed for output downloaded from analyse_library_overview.py.

Main outputs
------------
1. paired_mean_rg_300_370.png
2. delta_rg_heatmap.png
3. design_landscape_delta_rg_vs_rg370.png
4. overlay_rg_300K.png
5. overlay_rg_370K.png
6. overlay_collapsed_dwell_300K.png
7. overlay_collapsed_dwell_370K.png
8. overlay_distributions_2x2.png
9. column_detection_report.txt

The script:
- reads per_architecture_300_vs_370.csv;
- recursively searches for per-frame Rg values and collapsed dwell-time values;
- automatically recognizes common column/file naming variants;
- derives ratio and architecture type from IDs such as:
      031_VGIP_R1_even
      007_EGKG_alternating

If raw distributions cannot be found, the summary plots are still generated and
the detection report explains which files/columns were seen.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Flexible column aliases
# ---------------------------------------------------------------------------

ID_ALIASES = [
    "id", "architecture", "architecture_id", "design", "name",
    "sequence_id", "label", "traj_id", "trajectory",
]

TEMP_ALIASES = [
    "temperature", "temp", "temperature_k", "temp_k", "t",
]

RG_VALUE_ALIASES = [
    "rg", "radius_of_gyration", "radius_gyration", "rg_nm",
    "r_g", "gyration_radius",
]

DWELL_VALUE_ALIASES = [
    "collapsed_dwell_time", "collapsed_dwell_ns", "dwell_collapsed",
    "collapsed_dwell", "dwell_time", "duration_ns", "duration",
]

SUMMARY_RG300_ALIASES = [
    "mean_rg_300", "mean_rg_300k", "rg_300", "rg300",
    "mean_rg_300_k", "mean_rg_temp300",
]

SUMMARY_RG370_ALIASES = [
    "mean_rg_370", "mean_rg_370k", "rg_370", "rg370",
    "mean_rg_370_k", "mean_rg_temp370",
]

SUMMARY_DELTA_ALIASES = [
    "delta_rg", "deltarg", "rg_delta", "delta_rg_370_300",
    "rg370_minus_rg300", "mean_rg_delta",
]

SUMMARY_COLLAPSED_DWELL_300_ALIASES = [
    "mean_collapsed_dwell_300", "mean_collapsed_dwell_300k",
    "collapsed_dwell_300", "collapsed_dwell_300k",
]

SUMMARY_COLLAPSED_DWELL_370_ALIASES = [
    "mean_collapsed_dwell_370", "mean_collapsed_dwell_370k",
    "collapsed_dwell_370", "collapsed_dwell_370k",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make publication-style CALVADOS architecture plots."
    )
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="One downloaded analysis folder, e.g. EGKG_detailed_stride1.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output folder. Default: <root>/publication_plots",
    )
    p.add_argument(
        "--summary",
        default="per_architecture_300_vs_370.csv",
        help="Summary filename.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=4.30,
        help="Theta/collapse threshold in nm. Default: 4.30",
    )
    p.add_argument(
        "--expanded-rg",
        type=float,
        default=7.87,
        help="Expanded-coil reference Rg in nm. Default: 7.87",
    )
    p.add_argument(
        "--compact-rg",
        type=float,
        default=1.57,
        help="Compact-globule reference Rg in nm. Default: 1.57",
    )
    p.add_argument(
        "--max-architectures",
        type=int,
        default=8,
        help="Maximum number of architectures in overlay plots. Default: 8",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution. Default: 300",
    )
    return p.parse_args()


def normalize(text: object) -> str:
    s = str(text).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    mapping = {normalize(c): c for c in df.columns}

    # Exact normalized match first.
    for alias in aliases:
        n = normalize(alias)
        if n in mapping:
            return mapping[n]

    # Then conservative substring matching.
    for alias in aliases:
        n = normalize(alias)
        for norm_col, original in mapping.items():
            if n and (norm_col.startswith(n) or norm_col.endswith(n)):
                return original
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def shorten_id(raw: object) -> str:
    """Remove numeric prefix and shorten common architecture labels."""
    s = str(raw)
    s = re.sub(r"^\d+_", "", s)

    replacements = {
        "smallblocks": "small blocks",
        "mediumblocks": "medium blocks",
        "largeblocks": "large blocks",
        "randomshuffle": "random shuffle",
        "IPcenter_VGends": "IP center / VG ends",
        "VGcenter_IPends": "VG center / IP ends",
        "IPendcaps": "IP endcaps",
        "VAendcaps": "VA endcaps",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    parts = s.split("_")
    # Usually remove the composition prefix and ratio from the display label.
    if len(parts) >= 3 and re.fullmatch(r"R\d+", parts[1], flags=re.I):
        s = "_".join(parts[2:])
    elif len(parts) >= 2:
        s = "_".join(parts[1:])

    return s.replace("_", " ")


def infer_ratio_and_architecture(raw: object) -> tuple[str, str]:
    """
    Examples:
      031_VGIP_R1_even -> (VGIP_R1, even)
      007_EGKG_alternating -> (EGKG, alternating)
    """
    s = re.sub(r"^\d+_", "", str(raw))
    parts = s.split("_")

    ratio_idx = next(
        (i for i, p in enumerate(parts) if re.fullmatch(r"R\d+", p, flags=re.I)),
        None,
    )

    if ratio_idx is not None:
        row = "_".join(parts[: ratio_idx + 1])
        arch = "_".join(parts[ratio_idx + 1:]) or "unknown"
        return row, arch

    composition = parts[0] if parts else "unknown"
    arch = "_".join(parts[1:]) if len(parts) > 1 else s
    return composition, arch


def read_csv_safely(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return None


def temperature_from_path(path: Path) -> int | None:
    text = str(path).lower()
    match = re.search(r"(?<!\d)(300|370)\s*k?(?!\d)", text)
    return int(match.group(1)) if match else None


def architecture_from_path(path: Path, known_ids: list[str]) -> str | None:
    full = str(path).replace("\\", "/").lower()

    # Prefer exact known architecture IDs in the path.
    matches = [arch for arch in known_ids if str(arch).lower() in full]
    if matches:
        return max(matches, key=len)

    # Fallback to a directory/file stem that resembles an architecture ID.
    candidates = list(path.parts[::-1]) + [path.stem]
    for item in candidates:
        if re.match(r"^\d+_[A-Za-z0-9]+_", item):
            return item
    return None


def kde_curve(values: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight Gaussian KDE without SciPy.
    Returns x and normalized density.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.array([]), np.array([])

    if values.size == 1 or np.allclose(values, values[0]):
        center = values[0]
        width = max(abs(center) * 0.04, 0.02)
        x = np.linspace(center - 3 * width, center + 3 * width, n)
        y = np.exp(-0.5 * ((x - center) / width) ** 2)
        return x, y / y.max()

    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    robust_sigma = min(std, iqr / 1.349) if iqr > 0 else std
    sigma = robust_sigma if robust_sigma > 0 else std
    bandwidth = 0.9 * sigma * values.size ** (-1 / 5)
    bandwidth = max(bandwidth, np.ptp(values) / 200, 1e-4)

    margin = 3 * bandwidth
    x = np.linspace(values.min() - margin, values.max() + margin, n)
    z = (x[:, None] - values[None, :]) / bandwidth
    y = np.exp(-0.5 * z**2).sum(axis=1)
    y /= values.size * bandwidth * math.sqrt(2 * math.pi)

    if y.max() > 0:
        y = y / y.max()
    return x, y


# ---------------------------------------------------------------------------
# Summary loading
# ---------------------------------------------------------------------------

def load_summary(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {path}")

    columns: dict[str, str] = {}

    id_col = find_column(df, ID_ALIASES)
    if id_col is None:
        # Fall back to first text column.
        text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if not text_cols:
            raise ValueError("Could not identify architecture ID column.")
        id_col = text_cols[0]
    columns["id"] = id_col

    rg300 = find_column(df, SUMMARY_RG300_ALIASES)
    rg370 = find_column(df, SUMMARY_RG370_ALIASES)

    if rg300 is None or rg370 is None:
        # More flexible fallback: identify numeric columns containing rg and temp.
        for c in df.columns:
            n = normalize(c)
            if "rg" in n and "300" in n and rg300 is None:
                rg300 = c
            if "rg" in n and "370" in n and rg370 is None:
                rg370 = c

    if rg300 is None or rg370 is None:
        raise ValueError(
            "Could not find the 300 K and 370 K mean-Rg columns.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    columns["rg300"] = rg300
    columns["rg370"] = rg370

    delta = find_column(df, SUMMARY_DELTA_ALIASES)
    if delta is not None:
        columns["delta"] = delta

    dwell300 = find_column(df, SUMMARY_COLLAPSED_DWELL_300_ALIASES)
    dwell370 = find_column(df, SUMMARY_COLLAPSED_DWELL_370_ALIASES)
    if dwell300:
        columns["dwell300"] = dwell300
    if dwell370:
        columns["dwell370"] = dwell370

    out = df.copy()
    out["_architecture_id"] = out[id_col].astype(str)
    out["_label"] = out["_architecture_id"].map(shorten_id)
    parsed = out["_architecture_id"].map(infer_ratio_and_architecture)
    out["_ratio"] = [x[0] for x in parsed]
    out["_archtype"] = [x[1] for x in parsed]

    out["_rg300"] = safe_numeric(out[rg300])
    out["_rg370"] = safe_numeric(out[rg370])

    # Always define ΔRg consistently as Rg370 - Rg300.
    out["_delta_rg"] = out["_rg370"] - out["_rg300"]

    return out, columns


# ---------------------------------------------------------------------------
# Raw-distribution discovery
# ---------------------------------------------------------------------------

def collect_raw_distributions(
    root: Path,
    known_ids: list[str],
    report_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      rg_long: architecture, temperature, value, source_file
      dwell_long: architecture, temperature, value, source_file
    """
    rg_rows: list[pd.DataFrame] = []
    dwell_rows: list[pd.DataFrame] = []

    csv_files = sorted(root.rglob("*.csv"))
    report_lines.append(f"CSV files scanned: {len(csv_files)}")
    report_lines.append("")

    for path in csv_files:
        if path.name.lower() == "per_architecture_300_vs_370.csv":
            continue

        df = read_csv_safely(path)
        if df is None or df.empty:
            report_lines.append(f"UNREADABLE/EMPTY: {path}")
            continue

        temp_col = find_column(df, TEMP_ALIASES)
        id_col = find_column(df, ID_ALIASES)
        rg_col = find_column(df, RG_VALUE_ALIASES)
        dwell_col = find_column(df, DWELL_VALUE_ALIASES)

        inferred_temp = temperature_from_path(path)
        inferred_arch = architecture_from_path(path, known_ids)

        report_lines.append(
            f"{path}\n"
            f"  columns={list(df.columns)}\n"
            f"  detected id={id_col}, temp={temp_col}, rg={rg_col}, "
            f"dwell={dwell_col}, path_temp={inferred_temp}, "
            f"path_arch={inferred_arch}"
        )

        if rg_col is not None:
            tmp = pd.DataFrame()
            tmp["value"] = safe_numeric(df[rg_col])

            if id_col:
                tmp["architecture"] = df[id_col].astype(str)
            elif inferred_arch:
                tmp["architecture"] = inferred_arch
            else:
                tmp["architecture"] = None

            if temp_col:
                tmp["temperature"] = safe_numeric(df[temp_col])
            else:
                tmp["temperature"] = inferred_temp

            tmp["source_file"] = str(path)
            tmp = tmp.dropna(subset=["value", "architecture", "temperature"])
            tmp["temperature"] = tmp["temperature"].astype(int)
            tmp = tmp[tmp["temperature"].isin([300, 370])]
            if not tmp.empty:
                rg_rows.append(tmp)

        if dwell_col is not None:
            tmp = pd.DataFrame()
            tmp["value"] = safe_numeric(df[dwell_col])

            if id_col:
                tmp["architecture"] = df[id_col].astype(str)
            elif inferred_arch:
                tmp["architecture"] = inferred_arch
            else:
                tmp["architecture"] = None

            if temp_col:
                tmp["temperature"] = safe_numeric(df[temp_col])
            else:
                tmp["temperature"] = inferred_temp

            tmp["source_file"] = str(path)
            tmp = tmp.dropna(subset=["value", "architecture", "temperature"])
            tmp["temperature"] = tmp["temperature"].astype(int)
            tmp = tmp[tmp["temperature"].isin([300, 370])]
            # Dwell times must be positive for logarithmic plotting.
            tmp = tmp[tmp["value"] > 0]
            if not tmp.empty:
                dwell_rows.append(tmp)

    rg_long = (
        pd.concat(rg_rows, ignore_index=True)
        if rg_rows else
        pd.DataFrame(columns=["architecture", "temperature", "value", "source_file"])
    )
    dwell_long = (
        pd.concat(dwell_rows, ignore_index=True)
        if dwell_rows else
        pd.DataFrame(columns=["architecture", "temperature", "value", "source_file"])
    )

    return rg_long, dwell_long


# ---------------------------------------------------------------------------
# Plot 1: paired mean Rg
# ---------------------------------------------------------------------------

def plot_paired_rg(
    summary: pd.DataFrame,
    outpath: Path,
    threshold: float,
    expanded_rg: float,
    compact_rg: float,
    dpi: int,
) -> None:
    df = summary.dropna(subset=["_rg300", "_rg370"]).copy()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.15), 6.5))

    # Soft response bars and connecting lines.
    for i, row in df.reset_index(drop=True).iterrows():
        lo = min(row["_rg300"], row["_rg370"])
        hi = max(row["_rg300"], row["_rg370"])
        ax.bar(
            i,
            hi - lo,
            bottom=lo,
            width=0.55,
            color="0.90",
            edgecolor="none",
            zorder=1,
        )
        ax.plot(
            [i, i],
            [row["_rg300"], row["_rg370"]],
            color="0.72",
            lw=1.8,
            zorder=2,
        )

    ax.scatter(x, df["_rg300"], s=115, label="300 K", zorder=4)
    ax.scatter(x, df["_rg370"], s=115, label="370 K", zorder=4)

    ax.axhline(expanded_rg, color="0.65", ls=":", lw=2)
    ax.axhline(threshold, color="0.45", ls=":", lw=2)
    ax.axhline(compact_rg, color="0.65", ls=":", lw=2)

    xmax = len(df) - 0.5
    ax.text(
        xmax, expanded_rg + 0.07,
        f"Expanded coil: {expanded_rg:.2f} nm",
        ha="right", va="bottom", color="0.35",
    )
    ax.text(
        xmax, threshold + 0.07,
        f"Theta state: {threshold:.2f} nm",
        ha="right", va="bottom", color="0.35",
    )
    ax.text(
        xmax, compact_rg + 0.07,
        f"Compact globule: {compact_rg:.2f} nm",
        ha="right", va="bottom", color="0.35",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["_label"], rotation=45, ha="right")
    ax.set_ylabel("Mean radius of gyration, Rg (nm)")
    ax.set_xlabel("Architecture")
    ax.set_title("Mean radius of gyration: 300 K versus 370 K")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.6, len(df) - 0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: ΔRg heatmap
# ---------------------------------------------------------------------------

def plot_delta_heatmap(
    summary: pd.DataFrame,
    outpath: Path,
    dpi: int,
) -> None:
    df = summary.dropna(subset=["_delta_rg"]).copy()

    ratio_order = list(dict.fromkeys(df["_ratio"].tolist()))
    arch_order = list(dict.fromkeys(df["_archtype"].tolist()))

    pivot = df.pivot_table(
        index="_ratio",
        columns="_archtype",
        values="_delta_rg",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=ratio_order, columns=arch_order)

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return

    # Sequential scale is appropriate because ΔRg is generally negative.
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    fig_width = max(9, 1.35 * len(pivot.columns) + 2.5)
    fig_height = max(4.5, 0.75 * len(pivot.index) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [str(x).replace("_", " ") for x in pivot.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Architecture type")
    ax.set_ylabel("Motif ratio")
    ax.set_title("Temperature response heatmap: ΔRg = Rg370 − Rg300")

    # Cell annotations.
    span = max(abs(vmax - vmin), 1e-12)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                normalized = (value - vmin) / span
                text_color = "white" if normalized < 0.42 else "black"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("ΔRg (nm)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: design landscape
# ---------------------------------------------------------------------------

def plot_design_landscape(
    summary: pd.DataFrame,
    outpath: Path,
    threshold: float,
    dpi: int,
) -> None:
    df = summary.dropna(subset=["_rg370", "_delta_rg"]).copy()
    groups = list(dict.fromkeys(df["_ratio"].tolist()))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8.7, 7))

    for idx, group in enumerate(groups):
        part = df[df["_ratio"] == group]
        ax.scatter(
            part["_rg370"],
            part["_delta_rg"],
            s=68,
            alpha=0.82,
            color=cmap(idx % 20),
            edgecolor="white",
            linewidth=0.5,
            label=group,
        )

    ax.axhline(0, color="0.35", ls=":", lw=1.7)
    ax.axvline(threshold, color="0.35", ls=":", lw=1.7)

    ax.set_xlabel("Mean Rg at 370 K (nm)")
    ax.set_ylabel("ΔRg = Rg370 − Rg300 (nm)")
    ax.set_title("Design landscape: final compactness versus temperature response")
    ax.grid(ls=":", alpha=0.35)

    if len(groups) <= 12:
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.legend(
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=2,
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4-7: overlay distributions
# ---------------------------------------------------------------------------

def select_architectures(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    max_architectures: int,
) -> list[str]:
    summary_order = summary["_architecture_id"].astype(str).tolist()
    available = set(raw["architecture"].astype(str).tolist())
    selected = [x for x in summary_order if x in available]

    if not selected:
        selected = list(dict.fromkeys(raw["architecture"].astype(str).tolist()))

    return selected[:max_architectures]


def plot_overlay_distributions(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    temperature: int,
    metric: str,
    outpath: Path,
    max_architectures: int,
    dpi: int,
    ax: plt.Axes | None = None,
) -> bool:
    data = raw[raw["temperature"] == temperature].copy()
    if data.empty:
        return False

    architectures = select_architectures(summary, data, max_architectures)
    data = data[data["architecture"].astype(str).isin(architectures)]
    if data.empty:
        return False

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("tab10")

    for idx, architecture in enumerate(architectures):
        values = safe_numeric(
            data.loc[
                data["architecture"].astype(str) == architecture,
                "value",
            ]
        ).dropna().to_numpy()

        if metric == "dwell":
            values = values[values > 0]
            if values.size:
                transformed = np.log10(values)
                x_log, y = kde_curve(transformed)
                x = 10**x_log
            else:
                x, y = np.array([]), np.array([])
        else:
            x, y = kde_curve(values)

        if x.size == 0:
            continue

        label = shorten_id(architecture)
        color = cmap(idx % 10)
        ax.plot(x, y, lw=2, color=color, label=label)
        ax.fill_between(x, 0, y, color=color, alpha=0.08)

        median = np.median(values)
        ax.axvline(median, color=color, lw=0.9, alpha=0.55)

    if metric == "dwell":
        ax.set_xscale("log")
        ax.set_xlabel("Collapsed dwell time (ns)")
        title_metric = "Collapsed dwell-time distributions"
    else:
        ax.set_xlabel("Radius of gyration, Rg (nm)")
        title_metric = "Rg distributions"

    ax.set_ylabel("Normalized density")
    ax.set_title(f"{title_metric} at {temperature} K")
    ax.grid(ls=":", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    if own_figure:
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return True


def plot_overlay_panel(
    rg_raw: pd.DataFrame,
    dwell_raw: pd.DataFrame,
    summary: pd.DataFrame,
    outpath: Path,
    max_architectures: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    combinations = [
        (axes[0, 0], rg_raw, 300, "rg"),
        (axes[0, 1], rg_raw, 370, "rg"),
        (axes[1, 0], dwell_raw, 300, "dwell"),
        (axes[1, 1], dwell_raw, 370, "dwell"),
    ]

    any_plot = False
    for ax, raw, temp, metric in combinations:
        success = plot_overlay_distributions(
            raw=raw,
            summary=summary,
            temperature=temp,
            metric=metric,
            outpath=outpath,
            max_architectures=max_architectures,
            dpi=dpi,
            ax=ax,
        )
        any_plot = any_plot or success
        if not success:
            ax.text(
                0.5, 0.5,
                f"No raw {metric} distribution found\nfor {temp} K",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    if not any_plot:
        plt.close(fig)
        return

    fig.suptitle(
        "Architecture-resolved Rg and collapsed dwell-time distributions",
        fontsize=18,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    outdir = (
        args.outdir.expanduser().resolve()
        if args.outdir is not None
        else root / "publication_plots"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = root / args.summary
    if not summary_path.exists():
        matches = list(root.rglob(args.summary))
        if len(matches) == 1:
            summary_path = matches[0]
        elif not matches:
            raise FileNotFoundError(
                f"Could not find {args.summary} below:\n{root}"
            )
        else:
            raise RuntimeError(
                f"Found multiple {args.summary} files. "
                "Point --root to one composition folder."
            )

    summary, detected_columns = load_summary(summary_path)

    report_lines = [
        "CALVADOS plotting column/file detection report",
        "=" * 52,
        f"Root: {root}",
        f"Summary: {summary_path}",
        "",
        "Detected summary columns:",
    ]
    report_lines.extend(
        f"  {key}: {value}" for key, value in detected_columns.items()
    )
    report_lines.extend([
        "",
        f"Architectures in summary: {len(summary)}",
        "",
        "Raw-file scan:",
    ])

    rg_raw, dwell_raw = collect_raw_distributions(
        root=root,
        known_ids=summary["_architecture_id"].astype(str).tolist(),
        report_lines=report_lines,
    )

    report_lines.extend([
        "",
        f"Raw Rg observations found: {len(rg_raw)}",
        f"Raw collapsed-dwell observations found: {len(dwell_raw)}",
    ])
    (outdir / "column_detection_report.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # Save cleaned summary used by the plots.
    summary.to_csv(outdir / "plotting_summary_cleaned.csv", index=False)

    plot_paired_rg(
        summary,
        outdir / "paired_mean_rg_300_370.png",
        threshold=args.threshold,
        expanded_rg=args.expanded_rg,
        compact_rg=args.compact_rg,
        dpi=args.dpi,
    )

    plot_delta_heatmap(
        summary,
        outdir / "delta_rg_heatmap.png",
        dpi=args.dpi,
    )

    plot_design_landscape(
        summary,
        outdir / "design_landscape_delta_rg_vs_rg370.png",
        threshold=args.threshold,
        dpi=args.dpi,
    )

    # Separate overlay plots.
    plot_overlay_distributions(
        rg_raw, summary, 300, "rg",
        outdir / "overlay_rg_300K.png",
        args.max_architectures, args.dpi,
    )
    plot_overlay_distributions(
        rg_raw, summary, 370, "rg",
        outdir / "overlay_rg_370K.png",
        args.max_architectures, args.dpi,
    )
    plot_overlay_distributions(
        dwell_raw, summary, 300, "dwell",
        outdir / "overlay_collapsed_dwell_300K.png",
        args.max_architectures, args.dpi,
    )
    plot_overlay_distributions(
        dwell_raw, summary, 370, "dwell",
        outdir / "overlay_collapsed_dwell_370K.png",
        args.max_architectures, args.dpi,
    )

    # Combined overview.
    plot_overlay_panel(
        rg_raw,
        dwell_raw,
        summary,
        outdir / "overlay_distributions_2x2.png",
        args.max_architectures,
        args.dpi,
    )

    print("\nFinished.")
    print(f"Summary read from: {summary_path}")
    print(f"Plots written to: {outdir}")
    print(f"Raw Rg observations detected: {len(rg_raw)}")
    print(f"Raw dwell observations detected: {len(dwell_raw)}")
    print(
        "Check column_detection_report.txt if one of the raw distribution "
        "plots was not created."
    )


if __name__ == "__main__":
    main()
