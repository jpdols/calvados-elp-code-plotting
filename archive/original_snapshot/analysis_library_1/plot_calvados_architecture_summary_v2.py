#!/usr/bin/env python3
"""
Publication-style plots for one downloaded CALVADOS analysis folder.

Expected files in --root
------------------------
all_rg_values.csv
dwell_events.csv
failed_runs.csv
per_architecture_300_vs_370.csv
per_run_summary.csv

Windows may hide the ".csv" extension; the script also accepts extensionless
files with these names.

Plots
-----
1. paired_mean_rg_300_370.png
2. delta_rg_heatmap.png
3. design_landscape_delta_rg_vs_rg370.png
4. overlay_rg_300K.png
5. overlay_rg_370K.png
6. overlay_collapsed_dwell_300K.png
7. overlay_collapsed_dwell_370K.png
8. overlay_distributions_2x2.png
9. plotting_summary_cleaned.csv
10. column_detection_report.txt
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
from matplotlib.colors import to_rgba


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Flexible aliases
# ---------------------------------------------------------------------------

ID_ALIASES = [
    "id", "architecture", "architecture_id", "arch_id", "design",
    "name", "sequence_id", "label", "traj_id", "trajectory",
]

TEMP_ALIASES = [
    "temperature", "temp", "temperature_k", "temp_k", "t",
]

RG_VALUE_ALIASES = [
    "rg", "radius_of_gyration", "radius_gyration", "rg_nm",
    "r_g", "gyration_radius", "rg_value",
]

DWELL_VALUE_ALIASES = [
    "duration_ns", "dwell_time_ns", "collapsed_dwell_time",
    "collapsed_dwell_ns", "dwell_collapsed", "collapsed_dwell",
    "dwell_time", "duration",
]

STATE_ALIASES = [
    "state", "event_state", "conformation", "phase",
]

SUMMARY_RG300_ALIASES = [
    "mean_rg_300", "mean_rg_300k", "rg_300", "rg300",
    "mean_rg_300_k", "mean_rg_temp300",
]

SUMMARY_RG370_ALIASES = [
    "mean_rg_370", "mean_rg_370k", "rg_370", "rg370",
    "mean_rg_370_k", "mean_rg_temp370",
]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot CALVADOS architecture summaries and raw distributions."
    )
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=4.30)
    p.add_argument("--expanded-rg", type=float, default=7.87)
    p.add_argument("--compact-rg", type=float, default=1.57)
    p.add_argument("--max-architectures", type=int, default=8)
    p.add_argument(
        "--landscape-colour-by",
        choices=["architecture", "ratio"],
        default="architecture",
        help="Colour/legend grouping for the design landscape.",
    )
    p.add_argument(
        "--landscape-folder-shapes",
        action="store_true",
        help=(
            "Use a different marker shape for each motif analysis folder. "
            "Colours represent architecture type and opacity represents ratio."
        ),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--include-folders",
        nargs="+",
        default=None,
        help=(
            "Only load analysis folders whose names contain one of these strings. "
            "Example: --include-folders VGIP VGIG"
        ),
    )
    return p.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def parse_temperature(series: pd.Series) -> pd.Series:
    """
    Convert temperature values such as 300, 300.0, '300K', and '370 K'
    to numeric Kelvin values.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("K", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    mapping = {normalize(c): c for c in df.columns}

    for alias in aliases:
        key = normalize(alias)
        if key in mapping:
            return mapping[key]

    for alias in aliases:
        key = normalize(alias)
        for norm_col, original in mapping.items():
            if key and (
                norm_col.startswith(key)
                or norm_col.endswith(key)
                or key in norm_col
            ):
                return original
    return None


def locate_named_file(root: Path, stem: str) -> Path:
    """
    Find stem.csv, stem, or a recursively located equivalent.
    """
    direct_candidates = [
        root / f"{stem}.csv",
        root / stem,
        root / f"{stem}.CSV",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    matches = [
        p for p in root.rglob("*")
        if p.is_file() and normalize(p.stem) == normalize(stem)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{stem}.csv' or '{stem}' below:\n{root}"
        )
    raise RuntimeError(
        f"Found multiple files matching '{stem}':\n"
        + "\n".join(str(p) for p in matches)
    )


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def shorten_id(raw: object) -> str:
    s = re.sub(r"^\d+_", "", str(raw))

    parts = s.split("_")
    if len(parts) >= 3 and re.fullmatch(r"R\d+", parts[1], flags=re.I):
        s = "_".join(parts[2:])
    elif len(parts) >= 2:
        s = "_".join(parts[1:])

    replacements = {
        "smallblocks": "small blocks",
        "mediumblocks": "medium blocks",
        "largeblocks": "large blocks",
        "randomshuffle": "random shuffle",
        "blockshuffle": "block shuffle",
        "IPcenter_VGends": "IP center / VG ends",
        "VGcenter_IPends": "VG center / IP ends",
        "IPendcaps": "IP endcaps",
        "VAendcaps": "VA endcaps",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    return s.replace("_", " ")

def clean_motif_folder(value: object) -> str:
    return (
        str(value)
        .replace("_detailed_stride1", "")
        .replace("_detailed", "")
    )

ARCHITECTURE_ORDER = [
    "even",
    "small blocks",
    "medium blocks",
    "large blocks",
    "motif 1 ends",
    "motif 2 ends",
    "diblock",
    "random shuffle",
]


def extract_motif_codes(ratio: object) -> tuple[str | None, str | None]:
    """
    Extract motif 1 and motif 2 from ratio names such as:

        VGIG_R1 -> ("vg", "ig")
        VGIP_R4 -> ("vg", "ip")
        VAIG_R2 -> ("va", "ig")
        EGKG_R3 -> ("eg", "kg")

    Assumes the motif-composition prefix consists of two two-letter codes.
    """
    composition = str(ratio).split("_R", 1)[0]
    composition = normalize(composition).replace("_", "")

    if len(composition) < 4:
        return None, None

    return composition[:2], composition[2:4]


def canonical_architecture_type(
    architecture: object,
    ratio: object,
) -> str:
    """
    Convert motif-specific architecture names into shared,
    motif-agnostic architecture categories.

    Motif 1 is the first motif in the composition name.
    Motif 2 is the second motif in the composition name.

    Examples:
        VGIG + IGcenter_VGends -> motif 1 ends
        VGIG + VGcenter_IGends -> motif 2 ends
        VGIG + IGendcaps       -> motif 2 ends
    """
    key = normalize(architecture)
    motif_1, motif_2 = extract_motif_codes(ratio)

    simple_mapping = {
        "even": "even",
        "smallblocks": "small blocks",
        "small_blocks": "small blocks",
        "mediumblocks": "medium blocks",
        "medium_blocks": "medium blocks",
        "largeblocks": "large blocks",
        "large_blocks": "large blocks",
        "diblock": "diblock",
        "randomshuffle": "random shuffle",
        "random_shuffle": "random shuffle",
        "blockshuffle": "random shuffle",
        "block_shuffle": "random shuffle",
    }

    if key in simple_mapping:
        return simple_mapping[key]

    # Pattern such as:
    # IGcenter_VGends
    # VGcenter_IPends
    centre_ends_match = re.fullmatch(
        r"([a-z0-9]+)center_([a-z0-9]+)ends",
        key,
    )

    if centre_ends_match:
        end_motif = centre_ends_match.group(2)

        if motif_1 is not None and end_motif == motif_1:
            return "motif 1 ends"

        if motif_2 is not None and end_motif == motif_2:
            return "motif 2 ends"

    # Pattern such as:
    # IGendcaps
    # IPendcaps
    # VGendcaps
    endcap_match = re.fullmatch(
        r"([a-z0-9]+)endcaps",
        key,
    )

    if endcap_match:
        end_motif = endcap_match.group(1)

        if motif_1 is not None and end_motif == motif_1:
            return "motif 1 ends"

        if motif_2 is not None and end_motif == motif_2:
            return "motif 2 ends"

    # Readable fallback for any unexpected architecture.
    return str(architecture).replace("_", " ")


def infer_ratio_and_architecture(raw: object) -> tuple[str, str]:
    s = re.sub(r"^\d+_", "", str(raw))
    parts = s.split("_")

    ratio_index = next(
        (i for i, part in enumerate(parts)
         if re.fullmatch(r"R\d+", part, flags=re.I)),
        None,
    )

    if ratio_index is not None:
        ratio = "_".join(parts[:ratio_index + 1])
        architecture = "_".join(parts[ratio_index + 1:]) or "unknown"
        return ratio, architecture

    composition = parts[0] if parts else "unknown"
    architecture = "_".join(parts[1:]) if len(parts) > 1 else s
    return composition, architecture


def canonical_architecture_id(
    series: pd.Series,
    known_ids: list[str],
) -> pd.Series:
    """
    Map raw IDs to exact summary IDs when possible.
    """
    exact = {str(x): str(x) for x in known_ids}
    normalized = {normalize(x): str(x) for x in known_ids}

    def convert(value: object) -> str:
        text = str(value)
        if text in exact:
            return exact[text]

        key = normalize(text)
        if key in normalized:
            return normalized[key]

        # Match when a run suffix is appended, e.g. _rep1, _300K.
        candidates = [
            original for norm, original in normalized.items()
            if norm in key or key in norm
        ]
        if candidates:
            return max(candidates, key=len)

        return text

    return series.map(convert)


def kde_curve(values: np.ndarray, n: int = 450) -> tuple[np.ndarray, np.ndarray]:
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
    bandwidth = max(bandwidth, np.ptp(values) / 250, 1e-5)

    x = np.linspace(
        values.min() - 3 * bandwidth,
        values.max() + 3 * bandwidth,
        n,
    )
    z = (x[:, None] - values[None, :]) / bandwidth
    y = np.exp(-0.5 * z**2).sum(axis=1)
    y /= values.size * bandwidth * math.sqrt(2 * math.pi)

    if y.max() > 0:
        y /= y.max()

    return x, y


# ---------------------------------------------------------------------------
# Load summary and raw data
# ---------------------------------------------------------------------------

def load_summary(
    path: Path,
    motif_folder: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    
    df = read_table(path)
    if df.empty:
        raise ValueError(f"Empty summary file: {path}")

    id_col = find_column(df, ID_ALIASES)
    if id_col is None:
        non_numeric = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not non_numeric:
            raise ValueError("Could not identify architecture ID column.")
        id_col = non_numeric[0]

    def find_temperature_rg_column(
        dataframe: pd.DataFrame,
        temperature: int,
    ) -> str | None:
        """
        Find the actual mean-Rg column for one temperature while explicitly
        excluding delta/change columns such as rg370_minus_rg300.
        """
        candidates = []

        for col in dataframe.columns:
            key = normalize(col)

            if "rg" not in key:
                continue

            if str(temperature) not in key:
                continue

            # Do not accidentally select ΔRg or comparison columns.
            forbidden = (
                "delta",
                "difference",
                "minus",
                "change",
                "response",
                "370_300",
                "300_370",
            )
            if any(term in key for term in forbidden):
                continue

            score = 0

            if "mean" in key:
                score += 5

            if key in {
                f"mean_rg_{temperature}",
                f"mean_rg_{temperature}k",
                f"rg_{temperature}",
                f"rg_{temperature}k",
            }:
                score += 20

            # Penalize columns containing both temperatures.
            if "300" in key and "370" in key:
                score -= 20

            candidates.append((score, col))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]


    rg300_col = find_temperature_rg_column(df, 300)
    rg370_col = find_temperature_rg_column(df, 370)

    if rg300_col is None or rg370_col is None:
        raise ValueError(
            "Could not identify mean Rg columns.\n"
            f"Columns found: {list(df.columns)}"
        )

    out = df.copy()

    if motif_folder is None:
        motif_folder = path.parent.name

    motif_folder = clean_motif_folder(motif_folder)

    out["_motif_folder"] = motif_folder
    out["_architecture_id"] = out[id_col].astype(str)

    out["_label"] = out["_architecture_id"].map(shorten_id)

    parsed = out["_architecture_id"].map(
        infer_ratio_and_architecture
    )

    out["_ratio"] = [
        item[0]
        for item in parsed
    ]

    raw_architecture_types = [
        item[1]
        for item in parsed
    ]

    out["_architecture_type"] = [
        canonical_architecture_type(
            architecture=architecture,
            ratio=ratio,
        )
        for architecture, ratio in zip(
            raw_architecture_types,
            out["_ratio"],
        )
    ]

    out["_rg300"] = safe_numeric(out[rg300_col])
    out["_rg370"] = safe_numeric(out[rg370_col])
    out["_delta_rg"] = out["_rg300"] - out["_rg370"]

    return out, {
        "id": id_col,
        "rg300": rg300_col,
        "rg370": rg370_col,
    }

def load_all_summaries(
    root: Path,
    include_folders: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Path], dict[str, dict[str, str]]]:
    """
    Recursively load every per_architecture_300_vs_370.csv below root.

    Example root:
        analysis_library_1/
            VGIP_detailed_stride1/
            VGIG_detailed_stride1/
    """
    summary_paths = sorted(
        root.rglob("per_architecture_300_vs_370.csv")
    )

    if include_folders:
        include_lower = [x.lower() for x in include_folders]

        summary_paths = [
            path
            for path in summary_paths
            if any(
                token in path.parent.name.lower()
                for token in include_lower
            )
        ]

    if not summary_paths:
        # Also support files where Windows hides or removed the extension.
        summary_paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and normalize(path.stem) == normalize(
                "per_architecture_300_vs_370"
            )
        )

    if not summary_paths:
        raise FileNotFoundError(
            "No matching per_architecture_300_vs_370 files found below:\n"
            f"{root}\n"
            f"Folder filters: {include_folders}"
        )

    frames = []
    detected_columns = {}

    for summary_path in summary_paths:
        motif_folder = summary_path.parent.name

        frame, columns = load_summary(
            summary_path,
            motif_folder=motif_folder,
        )

        frames.append(frame)
        detected_columns[motif_folder] = columns

        print(
            f"Loaded {len(frame)} architectures from "
            f"{motif_folder}"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined, summary_paths, detected_columns

def load_all_rg_files(
    summary_paths: list[Path],
    known_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, str | None]]]:
    """
    Load all_rg_values.csv from every selected motif folder.
    """
    frames = []
    detected = {}

    for summary_path in summary_paths:
        folder = summary_path.parent
        motif_folder = folder.name

        try:
            rg_path = locate_named_file(folder, "all_rg_values")
        except FileNotFoundError:
            print(f"WARNING: no all_rg_values file in {motif_folder}")
            continue

        frame, columns = load_all_rg_values(
            rg_path,
            known_ids,
        )

        frame = frame.copy()
        frame["_motif_folder"] = clean_motif_folder(motif_folder)
        frames.append(frame)
        detected[motif_folder] = columns

        print(
            f"Loaded {len(frame)} raw Rg values from "
            f"{motif_folder}"
        )

    if not frames:
        return (
            pd.DataFrame(
                columns=[
                    "architecture",
                    "temperature",
                    "value",
                    "_motif_folder",
                ]
            ),
            detected,
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined, detected


def load_all_dwell_files(
    summary_paths: list[Path],
    known_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, str | None]]]:
    """
    Load dwell_events.csv from every selected motif folder.
    """
    frames = []
    detected = {}

    for summary_path in summary_paths:
        folder = summary_path.parent
        motif_folder = folder.name

        try:
            dwell_path = locate_named_file(folder, "dwell_events")
        except FileNotFoundError:
            print(f"WARNING: no dwell_events file in {motif_folder}")
            continue

        frame, columns = load_dwell_events(
            dwell_path,
            known_ids,
        )

        frame = frame.copy()
        frame["_motif_folder"] = clean_motif_folder(motif_folder)
        frames.append(frame)
        detected[motif_folder] = columns

        print(
            f"Loaded {len(frame)} collapsed dwell events from "
            f"{motif_folder}"
        )

    if not frames:
        return (
            pd.DataFrame(
                columns=[
                    "architecture",
                    "temperature",
                    "value",
                    "_motif_folder",
                ]
            ),
            detected,
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined, detected



def load_all_rg_values(
    path: Path,
    known_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = read_table(path)

    id_col = find_column(df, ID_ALIASES)
    temp_col = find_column(df, TEMP_ALIASES)
    rg_col = find_column(df, RG_VALUE_ALIASES)

    # Stronger fallbacks based on common output conventions.
    if rg_col is None:
        numeric_candidates = []
        for col in df.columns:
            key = normalize(col)
            if "frame" in key or "time" in key or "temp" in key:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_candidates.append(col)
        if len(numeric_candidates) == 1:
            rg_col = numeric_candidates[0]

    if id_col is None or temp_col is None or rg_col is None:
        raise ValueError(
            "Could not identify required columns in all_rg_values.\n"
            f"Detected: id={id_col}, temp={temp_col}, rg={rg_col}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "architecture": canonical_architecture_id(
            df[id_col].astype(str),
            known_ids,
        ),
        "temperature": parse_temperature(df[temp_col]),
        "value": safe_numeric(df[rg_col]),
    })
    out = out.dropna()
    out["temperature"] = out["temperature"].astype(int)
    out = out[out["temperature"].isin([300, 370])]

    return out, {
        "id": id_col,
        "temperature": temp_col,
        "rg": rg_col,
    }


def load_dwell_events(
    path: Path,
    known_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = read_table(path)

    id_col = find_column(df, ID_ALIASES)
    temp_col = find_column(df, TEMP_ALIASES)
    dwell_col = find_column(df, DWELL_VALUE_ALIASES)
    state_col = find_column(df, STATE_ALIASES)

    if dwell_col is None:
        for col in df.columns:
            key = normalize(col)
            if ("dwell" in key or "duration" in key) and pd.api.types.is_numeric_dtype(df[col]):
                dwell_col = col
                break

    if id_col is None or temp_col is None or dwell_col is None:
        raise ValueError(
            "Could not identify required columns in dwell_events.\n"
            f"Detected: id={id_col}, temp={temp_col}, dwell={dwell_col}, "
            f"state={state_col}\n"
            f"Available columns: {list(df.columns)}"
        )

    work = df.copy()

    # Keep only collapsed-state events when a state column is present.
    if state_col is not None:
        state_text = work[state_col].astype(str).str.lower()
        collapsed_mask = (
            state_text.str.contains("collap")
            | state_text.isin(["1", "true", "c"])
        )
        if collapsed_mask.any():
            work = work[collapsed_mask].copy()

    out = pd.DataFrame({
        "architecture": canonical_architecture_id(
            work[id_col].astype(str),
            known_ids,
        ),
        "temperature": parse_temperature(work[temp_col]),
        "value": safe_numeric(work[dwell_col]),
    })
    out = out.dropna()
    out["temperature"] = out["temperature"].astype(int)
    out = out[
        out["temperature"].isin([300, 370])
        & (out["value"] > 0)
    ]

    return out, {
        "id": id_col,
        "temperature": temp_col,
        "dwell": dwell_col,
        "state": state_col,
    }


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------
def add_within_ratio_response_ranking(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank architectures separately within each motif ratio according to:

        ΔRg = Rg300 - Rg370

    Largest ΔRg:
        rank 1 = darkest green
        rank 2 = medium green
        rank 3 = light green

    Smallest ΔRg:
        worst = red

    All other architectures remain grey.

    The original dataframe row order is preserved.
    """
    out = dataframe.copy()

    out["_response_rank"] = np.nan
    out["_response_status"] = "unranked"
    out["_response_colour"] = "0.88"
    out["_response_annotation"] = ""

    ranking_colours = {
        "top1": "#145A32",   # darkest green
        "top2": "#2E8B57",   # medium green
        "top3": "#8FD694",   # light green
        "worst": "#C0392B",  # red
        "unranked": "0.88",  # neutral grey
    }

    # groupby(sort=False) preserves the order in which ratios occur.
    for ratio, group in out.groupby("_ratio", sort=False):
        valid = group.dropna(subset=["_delta_rg"])

        if valid.empty:
            continue

        # This sorting is used only to identify rankings.
        # It does not change the x-axis order in `out`.
        ranked_indices = (
            valid["_delta_rg"]
            .sort_values(
                ascending=False,
                kind="mergesort",
            )
            .index
            .tolist()
        )

        top_indices = ranked_indices[:min(3, len(ranked_indices))]

        for rank_number, row_index in enumerate(
            top_indices,
            start=1,
        ):
            status = f"top{rank_number}"

            out.loc[row_index, "_response_rank"] = rank_number
            out.loc[row_index, "_response_status"] = status
            out.loc[row_index, "_response_colour"] = (
                ranking_colours[status]
            )
            out.loc[row_index, "_response_annotation"] = (
                f"#{rank_number}"
            )

        # Highlight only one worst architecture per ratio.
        # Do not overwrite a top-three highlight when a ratio has
        # fewer than four valid architectures.
        worst_index = ranked_indices[-1]

        if worst_index not in top_indices:
            out.loc[worst_index, "_response_status"] = "worst"
            out.loc[worst_index, "_response_colour"] = (
                ranking_colours["worst"]
            )
            out.loc[worst_index, "_response_annotation"] = "Worst"

    return out

def plot_paired_rg(
    summary: pd.DataFrame,
    output: Path,
    threshold: float,
    expanded_rg: float,
    compact_rg: float,
    dpi: int,
) -> None:
    df = summary.dropna(
        subset=["_rg300", "_rg370", "_delta_rg"]
    ).copy()

    # Add rankings without changing the original row/x-axis order.
    df = add_within_ratio_response_ranking(df)

    # Resetting the index only gives sequential plotting positions.
    # It does not sort or rearrange the rows.
    df = df.reset_index(drop=True)

    x = np.arange(len(df))

    fig, ax = plt.subplots(
        figsize=(max(10, len(df) * 1.15), 6.8)
    )

    # ---------------------------------------------------------
    # Draw paired bars, connecting lines and individual points
    # ---------------------------------------------------------
    for i, row in df.iterrows():
        rg300 = float(row["_rg300"])
        rg370 = float(row["_rg370"])

        low = min(rg300, rg370)
        high = max(rg300, rg370)

        highlight_colour = row["_response_colour"]
        status = row["_response_status"]

        is_highlighted = status != "unranked"

        bar_colour = (
            highlight_colour
            if is_highlighted
            else "0.90"
        )

        line_colour = (
            highlight_colour
            if is_highlighted
            else "0.72"
        )

        bar_alpha = 0.72 if is_highlighted else 1.0
        line_width = 2.4 if is_highlighted else 1.8

        # Difference bar between Rg300 and Rg370.
        ax.bar(
            i,
            high - low,
            bottom=low,
            width=0.55,
            color=bar_colour,
            alpha=bar_alpha,
            edgecolor="none",
            zorder=1,
        )

        # Connecting line.
        ax.plot(
            [i, i],
            [rg300, rg370],
            color=line_colour,
            lw=line_width,
            alpha=0.95 if is_highlighted else 1.0,
            zorder=2,
        )

        # 300 K point remains consistently coloured so that the
        # highlighted colour specifically communicates ranking.
        ax.scatter(
            i,
            rg300,
            s=115,
            color="#4C78A8",
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )

        # The 370 K point uses the same ranking colour as the bar.
        rg370_colour = (
            highlight_colour
            if is_highlighted
            else "0.55"
        )

        ax.scatter(
            i,
            rg370,
            s=125 if is_highlighted else 115,
            color=rg370_colour,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )

        # Annotate top 1, top 2, top 3 and worst within each ratio.
        annotation = str(row["_response_annotation"])

        if annotation:
            ax.annotate(
                annotation,
                xy=(i, rg370),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=highlight_colour,
                zorder=6,
            )

    # ---------------------------------------------------------
    # Reference-state lines
    # ---------------------------------------------------------
    references = [
        (
            expanded_rg,
            f"Expanded coil: {expanded_rg:.2f} nm",
            "0.65",
        ),
        (
            threshold,
            f"Theta state: {threshold:.2f} nm",
            "0.45",
        ),
        (
            compact_rg,
            f"Compact globule: {compact_rg:.2f} nm",
            "0.65",
        ),
    ]

    for value, label, colour in references:
        ax.axhline(
            value,
            color=colour,
            ls=":",
            lw=2,
            zorder=0,
        )

        ax.text(
            len(df) - 0.5,
            value + 0.06,
            label,
            ha="right",
            va="bottom",
            color="0.35",
        )

    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["_label"],
        rotation=45,
        ha="right",
    )

    ax.set_ylabel("Mean radius of gyration, Rg (nm)")
    ax.set_xlabel("Architecture")
    ax.set_title(
        "Mean radius of gyration: 300 K versus 370 K\n"
        "Temperature-response ranking within each motif ratio"
    )

    ax.grid(
        axis="y",
        ls=":",
        alpha=0.35,
    )

    ax.set_xlim(-0.6, len(df) - 0.4)

    # ---------------------------------------------------------
    # Custom legend
    # ---------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#4C78A8",
            markeredgecolor="white",
            markersize=9,
            label="Mean Rg at 300 K",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.55",
            markeredgecolor="white",
            markersize=9,
            label="Unranked Rg at 370 K",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#145A32",
            markeredgecolor="none",
            markersize=9,
            label="Largest ΔRg within ratio",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#2E8B57",
            markeredgecolor="none",
            markersize=9,
            label="Second-largest ΔRg",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#8FD694",
            markeredgecolor="none",
            markersize=9,
            label="Third-largest ΔRg",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#C0392B",
            markeredgecolor="none",
            markersize=9,
            label="Smallest ΔRg within ratio",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        title="Within-ratio response ranking",
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_architecture_heatmap(
    summary: pd.DataFrame,
    value_column: str,
    output: Path,
    title: str,
    colourbar_label: str,
    dpi: int,
    cmap: str = "viridis",
    value_format: str = ".2f",
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
) -> None:
    """
    Plot a motif-ratio × architecture heatmap.

    All heatmaps use the same canonical architecture order so that
    equivalent architectures align across different motif pairs.
    """
    df = summary.dropna(
        subset=[value_column]
    ).copy()

    if df.empty:
        print(
            f"WARNING: no values found for heatmap "
            f"column '{value_column}'."
        )
        return

    # Preserve the ratio order found in the input files.
    row_order = list(
        dict.fromkeys(
            df["_ratio"].astype(str)
        )
    )

    present_architectures = list(
        dict.fromkeys(
            df["_architecture_type"].astype(str)
        )
    )

    # Known rational architecture order first.
    column_order = [
        architecture
        for architecture in ARCHITECTURE_ORDER
        if architecture in present_architectures
    ]

    # Keep any unexpected architectures after the standard ones.
    column_order.extend(
        architecture
        for architecture in present_architectures
        if architecture not in column_order
    )

    matrix = (
        df.pivot_table(
            index="_ratio",
            columns="_architecture_type",
            values=value_column,
            aggfunc="mean",
        )
        .reindex(
            index=row_order,
            columns=column_order,
        )
    )

    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        print(
            f"WARNING: no finite values found for heatmap "
            f"'{output.name}'."
        )
        return

    vmin = (
        fixed_vmin
        if fixed_vmin is not None
        else float(np.nanmin(values))
    )

    vmax = (
        fixed_vmax
        if fixed_vmax is not None
        else float(np.nanmax(values))
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                9,
                1.35 * len(matrix.columns) + 2.5,
            ),
            max(
                4.5,
                0.75 * len(matrix.index) + 2.2,
            ),
        )
    )

    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(matrix.columns))
    )
    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(matrix.index))
    )
    ax.set_yticklabels(
        matrix.index,
    )

    ax.set_xlabel("Architecture type")
    ax.set_ylabel("Motif ratio")
    ax.set_title(title)

    span = max(vmax - vmin, 1e-12)

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[
                row_index,
                column_index,
            ]

            if not np.isfinite(value):
                continue

            relative = (value - vmin) / span

            ax.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white"
                    if relative < 0.42
                    else "black"
                ),
            )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.04,
        pad=0.03,
    )
    colourbar.set_label(colourbar_label)

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

def extract_ratio_number(value: object) -> float:
    """
    Extract ratio number from values such as:
        VGIP_R1
        VGIG_R4
        R6

    Returns NaN when no ratio is present.
    """
    match = re.search(r"(?:^|_)R(\d+)(?:_|$)", str(value), flags=re.I)

    if match:
        return float(match.group(1))

    return np.nan

def plot_design_landscape(
    summary: pd.DataFrame,
    output: Path,
    threshold: float,
    colour_by: str,
    folder_shapes: bool,
    dpi: int,
) -> None:
    df = summary.dropna(
        subset=["_rg370", "_delta_rg"]
    ).copy()

    # ---------------------------------------------------------
    # Original plotting modes
    # ---------------------------------------------------------
    if not folder_shapes:
        if colour_by == "ratio":
            group_col = "_ratio"
            legend_title = "Motif ratio"
        else:
            group_col = "_architecture_type"
            legend_title = "Architecture"

        groups = list(
            dict.fromkeys(df[group_col].astype(str))
        )

        cmap = plt.get_cmap("tab20")
        colour_map = {
            group: cmap(index % 20)
            for index, group in enumerate(groups)
        }

        fig, ax = plt.subplots(figsize=(9.2, 7))

        for group in groups:
            part = df[
                df[group_col].astype(str) == group
            ]

            ax.scatter(
                part["_rg370"],
                part["_delta_rg"],
                s=82,
                alpha=0.88,
                color=colour_map[group],
                edgecolor="white",
                linewidth=0.7,
                label=group.replace("_", " "),
                zorder=3,
            )

        ax.axhline(
            0,
            color="0.35",
            ls=":",
            lw=1.8,
            zorder=1,
        )
        ax.axvline(
            threshold,
            color="0.35",
            ls=":",
            lw=1.8,
            zorder=1,
        )

        ax.set_xlabel("Mean Rg at 370 K (nm)")
        ax.set_ylabel("ΔRg = Rg300 − Rg370 (nm)")
        ax.set_title(
            "Design landscape: final compactness versus "
            "temperature response"
        )
        ax.grid(ls=":", alpha=0.35)

        ax.legend(
            title=legend_title,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1 if len(groups) <= 10 else 2,
        )

        fig.tight_layout()
        fig.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)
        return

    # ---------------------------------------------------------
    # New combined-folder styling
    #
    # Colour  = architecture
    # Shape   = motif folder
    # Opacity = ratio
    # ---------------------------------------------------------

    if "_motif_folder" not in df.columns:
        df["_motif_folder"] = "unknown"

    df["_ratio_number"] = df["_ratio"].map(
        extract_ratio_number
    )

    architecture_order = list(
        dict.fromkeys(
            df["_architecture_type"].astype(str)
        )
    )
    folder_order = list(
        dict.fromkeys(
            df["_motif_folder"].astype(str)
        )
    )

    colour_cmap = plt.get_cmap("tab10")
    architecture_colours = {
        architecture: colour_cmap(index % 10)
        for index, architecture in enumerate(
            architecture_order
        )
    }

    marker_options = [
        "o",
        "^",
        "s",
        "D",
        "P",
        "X",
        "v",
        "<",
        ">",
        "*",
    ]
    folder_markers = {
        folder: marker_options[index % len(marker_options)]
        for index, folder in enumerate(folder_order)
    }

    finite_ratios = df["_ratio_number"].dropna()

    if finite_ratios.empty:
        minimum_ratio = 1.0
        maximum_ratio = 1.0
    else:
        minimum_ratio = finite_ratios.min()
        maximum_ratio = finite_ratios.max()

    def ratio_to_alpha(ratio: float) -> float:
        """
        Map the lowest ratio to a light point and the highest ratio
        to a dark/opaque point.
        """
        if not np.isfinite(ratio):
            return 0.80

        if maximum_ratio == minimum_ratio:
            return 0.85

        fraction = (
            (ratio - minimum_ratio)
            / (maximum_ratio - minimum_ratio)
        )

        return 0.32 + 0.68 * fraction

    fig, ax = plt.subplots(figsize=(13.8, 7.3))

    # Plot every point separately because each point can have a
    # different architecture colour, folder marker and ratio alpha.
    for _, row in df.iterrows():
        architecture = str(row["_architecture_type"])
        folder = str(row["_motif_folder"])
        ratio_number = row["_ratio_number"]

        base_colour = architecture_colours[architecture]
        alpha = ratio_to_alpha(ratio_number)
        rgba = to_rgba(base_colour, alpha=alpha)

        ax.scatter(
            row["_rg370"],
            row["_delta_rg"],
            s=92,
            marker=folder_markers[folder],
            facecolor=rgba,
            edgecolor=to_rgba(base_colour, alpha=1.0),
            linewidth=0.9,
            zorder=3,
        )

    ax.axhline(
        0,
        color="0.35",
        ls=":",
        lw=1.8,
        zorder=1,
    )
    ax.axvline(
        threshold,
        color="0.35",
        ls=":",
        lw=1.8,
        zorder=1,
    )

    ax.set_xlabel("Mean Rg at 370 K (nm)")
    ax.set_ylabel("ΔRg = Rg300 − Rg370 (nm)")
    ax.set_title(
        "Design landscape: architecture, motif composition "
        "and ratio"
    )
    ax.grid(ls=":", alpha=0.35)

    # ---------------------------------------------------------
    # Legend 1: architecture colours
    # ---------------------------------------------------------
    architecture_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=architecture_colours[architecture],
            markeredgecolor=architecture_colours[architecture],
            markersize=8,
            label=architecture.replace("_", " "),
        )
        for architecture in architecture_order
    ]

    architecture_legend = ax.legend(
        handles=architecture_handles,
        title="Architecture colour",
        frameon=False,
        bbox_to_anchor=(1.02, 1.00),
        loc="upper left",
    )
    ax.add_artist(architecture_legend)

    # ---------------------------------------------------------
    # Legend 2: motif-folder shapes
    # ---------------------------------------------------------
    folder_handles = [
        Line2D(
            [0],
            [0],
            marker=folder_markers[folder],
            linestyle="none",
            markerfacecolor="0.55",
            markeredgecolor="0.25",
            markersize=8,
            label=folder.replace("_detailed_stride1", ""),
        )
        for folder in folder_order
    ]

    folder_legend = ax.legend(
        handles=folder_handles,
        title="Motif folder shape",
        frameon=False,
        bbox_to_anchor=(1.02, 0.51),
        loc="upper left",
    )
    ax.add_artist(folder_legend)

    # ---------------------------------------------------------
    # Legend 3: ratio opacity
    # ---------------------------------------------------------
    unique_ratios = sorted(
        int(value)
        for value in finite_ratios.unique()
    )

    ratio_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=to_rgba(
                "0.20",
                alpha=ratio_to_alpha(float(ratio)),
            ),
            markeredgecolor="0.20",
            markersize=8,
            label=f"R{ratio}",
        )
        for ratio in unique_ratios
    ]

    ratio_legend = None

    if ratio_handles:
        ratio_legend = ax.legend(
            handles=ratio_handles,
            title="Ratio opacity",
            frameon=False,
            bbox_to_anchor=(1.02, 0.18),
            loc="upper left",
        )
        ax.add_artist(ratio_legend)

    # Reserve the right side of the figure for the legends.
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 1.0])

    # Explicitly include externally positioned legends when determining
    # the saved figure bounding box.
    extra_artists = [
        architecture_legend,
        folder_legend,
    ]

    if ratio_legend is not None:
        extra_artists.append(ratio_legend)

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=tuple(extra_artists),
    )

    plt.close(fig)

# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def select_architectures(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    max_architectures: int,
    motif_folder: str | None = None,
    ratio: str | None = None,
) -> list[str]:
    """
    Select architectures in summary-table order, optionally restricted
    to one motif folder and one ratio.
    """
    selected_summary = summary.copy()

    if motif_folder is not None:
        selected_summary = selected_summary[
            selected_summary["_motif_folder"].astype(str)
            == str(motif_folder)
        ]

    if ratio is not None:
        selected_summary = selected_summary[
            selected_summary["_ratio"].astype(str)
            == str(ratio)
        ]

    available = set(raw["architecture"].astype(str))

    ordered = [
        architecture
        for architecture in selected_summary[
            "_architecture_id"
        ].astype(str)
        if architecture in available
    ]

    return ordered[:max_architectures]

def add_raw_metadata(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add motif folder, ratio and architecture type from the summary table
    to every raw Rg or dwell observation.
    """
    metadata = (
        summary[
            [
                "_architecture_id",
                "_motif_folder",
                "_ratio",
                "_architecture_type",
            ]
        ]
        .drop_duplicates(subset="_architecture_id")
        .rename(columns={"_architecture_id": "architecture"})
    )

    out = raw.merge(
        metadata,
        on="architecture",
        how="left",
        suffixes=("", "_summary"),
    )

    # Raw loaders may already have _motif_folder.
    if "_motif_folder_summary" in out.columns:
        if "_motif_folder" in out.columns:
            out["_motif_folder"] = out["_motif_folder"].fillna(
                out["_motif_folder_summary"]
            )
        else:
            out["_motif_folder"] = out["_motif_folder_summary"]

        out = out.drop(columns="_motif_folder_summary")

    return out

def add_fraction_collapsed_370(
    summary: pd.DataFrame,
    rg_raw: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Calculate the fraction of raw 370 K frames with:

        Rg < threshold

    Values are calculated per architecture across all available
    raw trajectory frames and added to the summary dataframe as:

        _fraction_collapsed_370
    """
    out = summary.copy()

    data_370 = rg_raw[
        rg_raw["temperature"] == 370
    ].copy()

    data_370["value"] = safe_numeric(
        data_370["value"]
    )

    data_370 = data_370.dropna(
        subset=["architecture", "value"]
    )

    if data_370.empty:
        out["_fraction_collapsed_370"] = np.nan
        print(
            "WARNING: no raw 370 K Rg values available for "
            "fraction-collapsed heatmap."
        )
        return out

    collapsed_fraction = (
        data_370
        .assign(
            _is_collapsed=(
                data_370["value"] < threshold
            )
        )
        .groupby("architecture", sort=False)[
            "_is_collapsed"
        ]
        .mean()
    )

    out["_fraction_collapsed_370"] = (
        out["_architecture_id"]
        .astype(str)
        .map(collapsed_fraction)
    )

    return out

def draw_overlay(
    ax: plt.Axes,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    temperature: int,
    metric: str,
    max_architectures: int,
    motif_folder: str | None = None,
    ratio: str | None = None,
) -> bool:
    data = raw[
        raw["temperature"] == temperature
    ].copy()

    if motif_folder is not None:
        data = data[
            data["_motif_folder"].astype(str)
            == str(motif_folder)
        ]

    if ratio is not None:
        data = data[
            data["_ratio"].astype(str)
            == str(ratio)
        ]

    if data.empty:
        return False

    architectures = select_architectures(
        summary=summary,
        raw=data,
        max_architectures=max_architectures,
        motif_folder=motif_folder,
        ratio=ratio,
    )

    data = data[
        data["architecture"].astype(str).isin(architectures)
    ]

    if data.empty:
        return False

    cmap = plt.get_cmap("tab10")

    for index, architecture in enumerate(architectures):
        values = safe_numeric(
            data.loc[
                data["architecture"].astype(str) == architecture,
                "value",
            ]
        ).dropna().to_numpy()

        if metric == "dwell":
            values = values[values > 0]

            if values.size == 0:
                continue

            log_values = np.log10(values)
            x_log, density = kde_curve(log_values)
            x = 10 ** x_log

        else:
            x, density = kde_curve(values)

        if x.size == 0:
            continue

        colour = cmap(index % 10)
        label = shorten_id(architecture)

        ax.plot(
            x,
            density,
            lw=2.1,
            color=colour,
            label=label,
        )

        ax.fill_between(
            x,
            0,
            density,
            color=colour,
            alpha=0.08,
        )

        ax.axvline(
            np.median(values),
            color=colour,
            lw=0.9,
            alpha=0.55,
        )

    if metric == "dwell":
        ax.set_xscale("log")
        ax.set_xlabel("Collapsed dwell time (ns)")
        metric_title = "Collapsed dwell-time distributions"
    else:
        ax.set_xlabel("Radius of gyration, Rg (nm)")
        metric_title = "Rg distributions"

    title_parts = [metric_title, f"{temperature} K"]

    if motif_folder is not None:
        clean_folder = (
            str(motif_folder)
            .replace("_detailed_stride1", "")
            .replace("_detailed", "")
        )
        title_parts.append(clean_folder)

    if ratio is not None:
        title_parts.append(str(ratio).split("_")[-1])

    ax.set_title(" | ".join(title_parts))
    ax.set_ylabel("Normalized density")
    ax.grid(ls=":", alpha=0.30)

    ax.legend(
        title="Architecture",
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    return True


def plot_single_overlay(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    temperature: int,
    metric: str,
    output: Path,
    max_architectures: int,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    success = draw_overlay(
        ax,
        raw,
        summary,
        temperature,
        metric,
        max_architectures,
    )

    if not success:
        plt.close(fig)
        return

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_overlays_per_ratio(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    metric: str,
    outdir: Path,
    max_architectures: int,
    dpi: int,
) -> None:
    """
    Create one plot for every:
        motif folder × ratio × temperature

    Each plot overlays up to max_architectures architectures.
    """
    if raw.empty:
        print(f"WARNING: no raw data available for {metric}")
        return

    metric_dir = outdir / f"{metric}_distributions_per_ratio"
    metric_dir.mkdir(parents=True, exist_ok=True)

    folders = list(
        dict.fromkeys(
            summary["_motif_folder"].astype(str)
        )
    )

    for motif_folder in folders:
        folder_summary = summary[
            summary["_motif_folder"].astype(str)
            == motif_folder
        ]

        ratios = list(
            dict.fromkeys(
                folder_summary["_ratio"].astype(str)
            )
        )

        clean_folder = (
            motif_folder
            .replace("_detailed_stride1", "")
            .replace("_detailed", "")
        )

        for ratio in ratios:
            clean_ratio = str(ratio).split("_")[-1]

            for temperature in (300, 370):
                fig, ax = plt.subplots(figsize=(9, 6))

                success = draw_overlay(
                    ax=ax,
                    raw=raw,
                    summary=summary,
                    temperature=temperature,
                    metric=metric,
                    max_architectures=max_architectures,
                    motif_folder=motif_folder,
                    ratio=ratio,
                )

                if not success:
                    plt.close(fig)
                    print(
                        f"WARNING: no {metric} data for "
                        f"{clean_folder} {clean_ratio} {temperature} K"
                    )
                    continue

                output = metric_dir / (
                    f"{clean_folder}_{clean_ratio}_"
                    f"{metric}_{temperature}K.png"
                )

                fig.tight_layout()
                fig.savefig(
                    output,
                    dpi=dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)

def plot_overlay_panel(
    rg_raw: pd.DataFrame,
    dwell_raw: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    max_architectures: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    jobs = [
        (axes[0, 0], rg_raw, 300, "rg"),
        (axes[0, 1], rg_raw, 370, "rg"),
        (axes[1, 0], dwell_raw, 300, "dwell"),
        (axes[1, 1], dwell_raw, 370, "dwell"),
    ]

    for ax, data, temperature, metric in jobs:
        success = draw_overlay(
            ax,
            data,
            summary,
            temperature,
            metric,
            max_architectures,
        )
        if not success:
            ax.text(
                0.5,
                0.5,
                f"No {metric} values found for {temperature} K",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    fig.suptitle(
        "Architecture-resolved Rg and collapsed dwell-time distributions",
        fontsize=18,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
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

    summary, summary_paths, summary_columns = load_all_summaries(
        root,
        include_folders=args.include_folders,
    )

    known_ids = summary["_architecture_id"].astype(str).tolist()

    rg_raw, rg_columns = load_all_rg_files(
        summary_paths,
        known_ids,
    )

    dwell_raw, dwell_columns = load_all_dwell_files(
        summary_paths,
        known_ids,
    )

    rg_raw = add_raw_metadata(
        rg_raw,
        summary,
    )

    dwell_raw = add_raw_metadata(
        dwell_raw,
        summary,
    )

    summary = add_fraction_collapsed_370(
        summary=summary,
        rg_raw=rg_raw,
        threshold=args.threshold,
    )

    print("\nSummary files loaded:")
    for summary_path in summary_paths:
        print(f"  {summary_path}")

    #rg_raw, rg_columns = load_all_rg_values(rg_path, known_ids)
    #dwell_raw, dwell_columns = load_dwell_events(dwell_path, known_ids)
    print("\nDetected raw Rg columns:")
    print(rg_columns)
    print(rg_raw["temperature"].value_counts().sort_index())

    print("\nDetected dwell-event columns:")
    print(dwell_columns)
    print(dwell_raw["temperature"].value_counts().sort_index())

    report = [
        "CALVADOS combined plotting data-detection report",
        "=" * 55,
        f"Root: {root}",
        f"Included folder filters: {args.include_folders}",
        "",
        "Summary files:",
    ]

    report.extend(
        f"  {path}"
        for path in summary_paths
    )

    report.extend([
        "",
        f"Architectures loaded: {len(summary)}",
        f"Raw Rg observations loaded: {len(rg_raw)}",
        f"Collapsed dwell events loaded: {len(dwell_raw)}",
        "",
        f"Summary columns by folder: {summary_columns}",
        "",
        f"Rg columns by folder: {rg_columns}",
        "",
        f"Dwell columns by folder: {dwell_columns}",
        "",
        "Raw Rg counts by temperature:",
        str(
            rg_raw.groupby("temperature").size()
        ),
        "",
        "Dwell counts by temperature:",
        str(
            dwell_raw.groupby("temperature").size()
        ),
    ])

    summary.to_csv(
        outdir / "plotting_summary_cleaned.csv",
        index=False,
    )

    plot_paired_rg(
        summary,
        outdir / "paired_mean_rg_300_370.png",
        args.threshold,
        args.expanded_rg,
        args.compact_rg,
        args.dpi,
    )

    plot_architecture_heatmap(
        summary=summary,
        value_column="_delta_rg",
        output=outdir / "delta_rg_heatmap.png",
        title="Temperature response heatmap: ΔRg = Rg300 − Rg370",
        colourbar_label="ΔRg (nm)",
        dpi=args.dpi,
        cmap="viridis",
        value_format=".2f",
    )

    plot_architecture_heatmap(
        summary=summary,
        value_column="_rg370",
        output=outdir / "mean_rg_370_heatmap.png",
        title="Mean radius of gyration at 370 K",
        colourbar_label="Mean Rg at 370 K (nm)",
        dpi=args.dpi,
        cmap="viridis_r",
        value_format=".2f",
    )

    plot_architecture_heatmap(
        summary=summary,
        value_column="_fraction_collapsed_370",
        output=outdir / "fraction_collapsed_370_heatmap.png",
        title=(
            "Fraction collapsed at 370 K: "
            f"Rg < {args.threshold:.2f} nm"
        ),
        colourbar_label="Fraction collapsed",
        dpi=args.dpi,
        cmap="viridis",
        value_format=".2f",
        fixed_vmin=0.0,
        fixed_vmax=1.0,
    )

    plot_design_landscape(
        summary,
        outdir / "design_landscape_delta_rg_vs_rg370.png",
        args.threshold,
        args.landscape_colour_by,
        args.landscape_folder_shapes,
        args.dpi,
    )

    print("\nSummary motif folders:")
    print(summary["_motif_folder"].unique())

    print("\nRaw Rg motif folders:")
    print(rg_raw["_motif_folder"].unique())

    print("\nRaw dwell motif folders:")
    print(dwell_raw["_motif_folder"].unique())

    plot_overlays_per_ratio(
        raw=rg_raw,
        summary=summary,
        metric="rg",
        outdir=outdir,
        max_architectures=args.max_architectures,
        dpi=args.dpi,
    )

    plot_overlays_per_ratio(
        raw=dwell_raw,
        summary=summary,
        metric="dwell",
        outdir=outdir,
        max_architectures=args.max_architectures,
        dpi=args.dpi,
    )

    

    print("\nFinished.")
    print(f"Raw Rg observations: {len(rg_raw)}")
    print(f"Collapsed dwell observations: {len(dwell_raw)}")
    print(f"Summary files loaded: {len(summary_paths)}")
    print(f"Raw Rg observations: {len(rg_raw)}")
    print(f"Collapsed dwell events: {len(dwell_raw)}")
    print(f"Plots written to: {outdir}")


if __name__ == "__main__":
    main()
