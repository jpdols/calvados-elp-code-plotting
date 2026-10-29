#!/usr/bin/env python3
"""Create a two-temperature CALVADOS manifest from the '3 motifs' tab.

Expected input workbook columns on the selected sheet:
    A: architecture ID/name
    B: motif architecture

Supported notation examples:
    (EG)29-(IG)14-(KG)41
    (EG4-IG1-KG6-IG1)×6-(EG5-IG1-KG5-IG1)
    Shuffle(IG14-EG29-KG41)
    Random permutation of 30 VG, 30 IG and 24 KG
    Evenly dispersed VG, IG and KG
    (KG)2-(IG)12-(VG2-KG)5-IG-(VG2-KG)5-IG-((VG-KG)2-IG)4-(VG-KG)2-(IG)12-(KG)2

Output workbook sheets:
    manifest:
        id | temp | aa_count | average_lambda | sequence

    validation:
        id | architecture | status | message | motif_count | aa_count |
        motif_composition

The script prints:
    - total architectures read
    - number successfully converted
    - number invalid
    - total manifest rows written
    - every invalid ID and its calculated amino-acid length, when available
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_INPUT = Path(
    r"E:\Master thesis\calvados_tea\data\library_ELPsV1.xlsx"
)
DEFAULT_SHEET = "3 motifs"
DEFAULT_LAMBDA_300 = Path(
    r"E:\Master thesis\calvados_tea\data"
    r"\residues_gamma_6p0_300K_argdefault_minus.csv"
)
DEFAULT_LAMBDA_370 = Path(
    r"E:\Master thesis\calvados_tea\data"
    r"\residues_gamma_6p0_370K_argdefault_minus.csv"
)
DEFAULT_OUTPUT = Path(
    r"E:\Master thesis\calvados_tea\data"
    r"\library_manifest_3_motifs_300K_370K.xlsx"
)

MOTIF_TO_SEQUENCE: Dict[str, str] = {
    "VG": "VPGVG",
    "IG": "VPGIG",
    "KG": "VPGKG",
    "EG": "VPGEG",
    "IP": "IPAVG",
    "AVG": "VPAVG",
}

MOTIF_NAMES = sorted(MOTIF_TO_SEQUENCE, key=len, reverse=True)
ID_RE = re.compile(r"^\d+_[A-Za-z0-9]+_")


class PatternError(ValueError):
    """Raised when an architecture expression cannot be parsed."""


def stable_seed(text: str, base_seed: int) -> int:
    """Return a reproducible integer seed derived from an ID."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:8], "big")


def normalize_pattern(text: str) -> str:
    """Normalize architecture notation before parsing."""
    text = str(text).strip()
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("×", "*")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    return text


def parse_repeat_number(text: str, pos: int) -> Tuple[int, int]:
    """Parse an optional repeat number, optionally preceded by '*'."""
    if pos < len(text) and text[pos] == "*":
        pos += 1

    match = re.match(r"\d+", text[pos:])
    if not match:
        return 1, pos

    return int(match.group()), pos + len(match.group())


def parse_expression(text: str) -> List[str]:
    """Expand nested motif notation into a flat motif list."""
    s = normalize_pattern(text)

    def parse_sequence(
        pos: int,
        stop_at_closing: bool,
    ) -> Tuple[List[str], int]:
        output: List[str] = []

        while pos < len(s):
            char = s[pos]

            if char == "-":
                pos += 1
                continue

            if char == ")":
                if stop_at_closing:
                    return output, pos + 1
                raise PatternError(
                    f"Unexpected ')' at position {pos} in {text!r}"
                )

            if char == "(":
                inner, pos = parse_sequence(
                    pos + 1,
                    stop_at_closing=True,
                )
                repeat, pos = parse_repeat_number(s, pos)
                output.extend(inner * repeat)
                continue

            motif = next(
                (m for m in MOTIF_NAMES if s.startswith(m, pos)),
                None,
            )
            if motif is None:
                raise PatternError(
                    f"Unknown token near {s[pos:pos + 20]!r} "
                    f"at position {pos} in {text!r}"
                )

            pos += len(motif)
            repeat, pos = parse_repeat_number(s, pos)
            output.extend([motif] * repeat)

        if stop_at_closing:
            raise PatternError(f"Missing ')' in {text!r}")

        return output, pos

    motifs, end = parse_sequence(0, stop_at_closing=False)

    if end != len(s):
        raise PatternError(f"Could not parse complete pattern {text!r}")

    return motifs


def parse_composition_text(text: str) -> Counter:
    """Parse counts from Shuffle(...) or natural-language random rows."""
    clean = str(text).strip()

    shuffle_match = re.fullmatch(
        r"\s*Shuffle\((.*)\)\s*",
        clean,
        flags=re.IGNORECASE,
    )
    if shuffle_match:
        return Counter(parse_expression(shuffle_match.group(1)))

    if clean.lower().startswith("random permutation of"):
        counts = Counter()

        for number, motif in re.findall(
            r"(\d+)\s*(VG|IG|KG|EG|IP|AVG)\b",
            clean,
            flags=re.IGNORECASE,
        ):
            counts[motif.upper()] += int(number)

        if not counts:
            raise PatternError(
                f"Could not read composition from {text!r}"
            )

        return counts

    raise PatternError(f"Not a composition row: {text!r}")


def group_key(identifier: str) -> str:
    """Return a family/ratio key such as VGIGKG_R3."""
    parts = identifier.split("_")

    if len(parts) < 3:
        return identifier

    return f"{parts[1]}_{parts[2]}"


def evenly_disperse(
    composition: Counter,
    motif_order: Sequence[str],
) -> List[str]:
    """Spread motifs deterministically while preserving exact counts."""
    remaining = Counter(composition)
    total = sum(remaining.values())
    placed = Counter()
    result: List[str] = []

    order_index = {
        motif: index
        for index, motif in enumerate(motif_order)
    }

    for position in range(1, total + 1):
        candidates = [
            motif
            for motif, count in remaining.items()
            if count > 0
        ]

        def score(motif: str) -> Tuple[float, int, int]:
            ideal = position * composition[motif] / total
            deficit = ideal - placed[motif]

            return (
                deficit,
                remaining[motif],
                -order_index.get(motif, 999),
            )

        chosen = max(candidates, key=score)
        result.append(chosen)
        remaining[chosen] -= 1
        placed[chosen] += 1

    return result


def architecture_to_motifs(
    identifier: str,
    architecture: str,
    known_compositions: Dict[str, Counter],
    base_seed: int,
) -> List[str]:
    """Convert one architecture cell into a flat list of motif names."""
    text = str(architecture).strip()
    lower = text.lower()

    if lower.startswith("shuffle("):
        composition = parse_composition_text(text)
        motifs = list(composition.elements())

        random.Random(
            stable_seed(identifier, base_seed)
        ).shuffle(motifs)

        return motifs

    if lower.startswith("random permutation of"):
        composition = parse_composition_text(text)
        motifs = list(composition.elements())

        random.Random(
            stable_seed(identifier, base_seed)
        ).shuffle(motifs)

        return motifs

    if lower.startswith("evenly dispersed"):
        key = group_key(identifier)

        if key not in known_compositions:
            raise PatternError(
                f"No composition baseline found for {identifier}. "
                "Place a Random permutation/Shuffle row in the "
                "same family and ratio."
            )

        names = re.findall(
            r"\b(VG|IG|KG|EG|IP|AVG)\b",
            text,
            flags=re.IGNORECASE,
        )
        motif_order = [name.upper() for name in names]

        for motif in known_compositions[key]:
            if motif not in motif_order:
                motif_order.append(motif)

        return evenly_disperse(
            known_compositions[key],
            motif_order,
        )

    return parse_expression(text)


def load_lambda_table(path: Path) -> Dict[str, float]:
    """Load a residue-lambda CSV with columns 'one' and 'lambdas'."""
    table = pd.read_csv(path)
    required = {"one", "lambdas"}
    missing = required.difference(table.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return {
        str(row["one"]).strip().upper(): float(row["lambdas"])
        for _, row in table.iterrows()
    }


def mean_lambda(
    sequence: str,
    residue_lambdas: Dict[str, float],
) -> float:
    """Calculate mean residue lambda for an amino-acid sequence."""
    missing = sorted(set(sequence).difference(residue_lambdas))

    if missing:
        raise ValueError(
            f"Lambda table lacks residues: {missing}"
        )

    return (
        sum(residue_lambdas[aa] for aa in sequence)
        / len(sequence)
    )


def read_architecture_rows(
    input_path: Path,
    sheet_name: str,
) -> List[Tuple[str, str]]:
    """Read architecture IDs and patterns from columns A:B."""

    raw = pd.read_excel(
        input_path,
        sheet_name=sheet_name,
        usecols="A:B",
        header=None,
        names=["id", "architecture"],
        dtype=str,
    )

    print("\n--- INPUT DIAGNOSTICS ---")
    print(f"Total physical rows read from A:B: {len(raw)}")

    rows: List[Tuple[str, str]] = []
    skipped_empty = []
    skipped_bad_id = []

    for excel_index, row in raw.iterrows():
        excel_row = excel_index + 1

        identifier = (
            ""
            if pd.isna(row["id"])
            else str(row["id"]).strip()
        )

        architecture = (
            ""
            if pd.isna(row["architecture"])
            else str(row["architecture"]).strip()
        )

        if not identifier or not architecture:
            skipped_empty.append(
                {
                    "excel_row": excel_row,
                    "id": identifier,
                    "architecture": architecture,
                }
            )
            continue

        if not ID_RE.match(identifier):
            skipped_bad_id.append(
                {
                    "excel_row": excel_row,
                    "id": identifier,
                    "architecture": architecture,
                }
            )
            continue

        rows.append((identifier, architecture))

    print(f"Accepted architecture rows: {len(rows)}")
    print(f"Skipped empty A/B rows:     {len(skipped_empty)}")
    print(f"Skipped invalid-ID rows:    {len(skipped_bad_id)}")

    if rows:
        print(f"First accepted ID: {rows[0][0]}")
        print(f"Last accepted ID:  {rows[-1][0]}")

    if skipped_bad_id:
        print("\nRows skipped because the ID format did not match:")
        for item in skipped_bad_id[:20]:
            print(
                f"Excel row {item['excel_row']}: "
                f"{item['id']!r}"
            )

        if len(skipped_bad_id) > 20:
            print(
                f"...and {len(skipped_bad_id) - 20} more"
            )

    print("-------------------------\n")

    if not rows:
        raise ValueError(
            f"No architecture rows found in columns A:B "
            f"of sheet {sheet_name!r}."
        )

    return rows


def build_manifest(
    rows: Sequence[Tuple[str, str]],
    lambda_300: Dict[str, float],
    lambda_370: Dict[str, float],
    base_seed: int,
    expected_aa_count: int | None,
    strict: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert rows and return manifest and validation dataframes."""
    known_compositions: Dict[str, Counter] = {}

    # First pass: collect baseline compositions for rows such as
    # "Evenly dispersed VG, IG and KG".
    for identifier, architecture in rows:
        lower = architecture.lower()

        if (
            lower.startswith("shuffle(")
            or lower.startswith("random permutation of")
        ):
            known_compositions[group_key(identifier)] = (
                parse_composition_text(architecture)
            )

    manifest_rows: List[dict] = []
    validation_rows: List[dict] = []

    for identifier, architecture in rows:
        motifs: List[str] | None = None
        sequence: str | None = None
        counts: Counter = Counter()
        aa_count: int | None = None

        try:
            motifs = architecture_to_motifs(
                identifier,
                architecture,
                known_compositions,
                base_seed,
            )

            sequence = "".join(
                MOTIF_TO_SEQUENCE[motif]
                for motif in motifs
            )
            counts = Counter(motifs)
            aa_count = len(sequence)

            if (
                expected_aa_count is not None
                and aa_count != expected_aa_count
            ):
                validation_rows.append(
                    {
                        "id": identifier,
                        "architecture": architecture,
                        "status": "ERROR",
                        "message": (
                            f"Expected {expected_aa_count} aa, "
                            f"obtained {aa_count} aa "
                            f"from {len(motifs)} motifs"
                        ),
                        "motif_count": len(motifs),
                        "aa_count": aa_count,
                        "motif_composition": "; ".join(
                            f"{motif}={counts[motif]}"
                            for motif in sorted(counts)
                        ),
                    }
                )

                if strict:
                    raise ValueError(
                        f"{identifier}: expected "
                        f"{expected_aa_count} aa, "
                        f"obtained {aa_count} aa"
                    )

                continue

            validation_rows.append(
                {
                    "id": identifier,
                    "architecture": architecture,
                    "status": "OK",
                    "message": "",
                    "motif_count": len(motifs),
                    "aa_count": aa_count,
                    "motif_composition": "; ".join(
                        f"{motif}={counts[motif]}"
                        for motif in sorted(counts)
                    ),
                }
            )

            for temperature, lambda_table in (
                (300, lambda_300),
                (370, lambda_370),
            ):
                manifest_rows.append(
                    {
                        "id": identifier,
                        "temp": temperature,
                        "aa_count": aa_count,
                        "average_lambda": mean_lambda(
                            sequence,
                            lambda_table,
                        ),
                        "sequence": sequence,
                    }
                )

        except Exception as exc:
            # Avoid adding a duplicate validation entry when a strict
            # length error was already recorded above.
            already_recorded = any(
                item["id"] == identifier
                and item["status"] == "ERROR"
                for item in validation_rows
            )

            if not already_recorded:
                validation_rows.append(
                    {
                        "id": identifier,
                        "architecture": architecture,
                        "status": "ERROR",
                        "message": str(exc),
                        "motif_count": (
                            len(motifs)
                            if motifs is not None
                            else None
                        ),
                        "aa_count": aa_count,
                        "motif_composition": (
                            "; ".join(
                                f"{motif}={counts[motif]}"
                                for motif in sorted(counts)
                            )
                            if counts
                            else ""
                        ),
                    }
                )

            if strict:
                raise ValueError(
                    f"{identifier}: {exc}. "
                    f"Pattern: {architecture}"
                ) from exc

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "id",
            "temp",
            "aa_count",
            "average_lambda",
            "sequence",
        ],
    )
    validation = pd.DataFrame(
        validation_rows,
        columns=[
            "id",
            "architecture",
            "status",
            "message",
            "motif_count",
            "aa_count",
            "motif_composition",
        ],
    )

    return manifest, validation


def write_output(
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write manifest and validation sheets to Excel."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        manifest.to_excel(
            writer,
            sheet_name="manifest",
            index=False,
        )
        validation.to_excel(
            writer,
            sheet_name="validation",
            index=False,
        )

        manifest_ws = writer.book["manifest"]
        validation_ws = writer.book["validation"]

        manifest_ws.freeze_panes = "A2"
        validation_ws.freeze_panes = "A2"

        manifest_ws.auto_filter.ref = manifest_ws.dimensions
        validation_ws.auto_filter.ref = validation_ws.dimensions

        manifest_widths = {
            "A": 58,
            "B": 10,
            "C": 12,
            "D": 18,
            "E": 95,
        }
        for column, width in manifest_widths.items():
            manifest_ws.column_dimensions[column].width = width

        validation_widths = {
            "A": 58,
            "B": 90,
            "C": 12,
            "D": 65,
            "E": 14,
            "F": 12,
            "G": 35,
        }
        for column, width in validation_widths.items():
            validation_ws.column_dimensions[column].width = width

        for cell in manifest_ws[1]:
            cell.font = cell.font.copy(bold=True)

        for cell in validation_ws[1]:
            cell.font = cell.font.copy(bold=True)

        for cell in manifest_ws["D"][1:]:
            cell.number_format = "0.000000"


def print_summary(
    rows: Sequence[Tuple[str, str]],
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
    input_path: Path,
    sheet_name: str,
    output_path: Path,
    temperatures: Sequence[int],
) -> None:
    """Print a detailed conversion and validation summary."""
    valid = validation[validation["status"] == "OK"]
    invalid = validation[validation["status"] == "ERROR"]

    print()
    print("=" * 72)
    print("CALVADOS LIBRARY MANIFEST GENERATOR")
    print("=" * 72)
    print(f"Input file              : {input_path}")
    print(f"Input sheet             : {sheet_name}")
    print(f"Architectures read      : {len(rows)}")
    print(f"Successfully converted  : {len(valid)}")
    print(f"Invalid architectures   : {len(invalid)}")
    print(f"Manifest rows written   : {len(manifest)}")
    print(
        "Temperatures            : "
        + ", ".join(f"{temp} K" for temp in temperatures)
    )
    print(f"Output file             : {output_path}")

    if not invalid.empty:
        print()
        print("-" * 72)
        print("ARCHITECTURES NOT CONVERTED")
        print("-" * 72)

        for _, row in invalid.iterrows():
            aa_value = row["aa_count"]

            if pd.isna(aa_value):
                aa_text = "unknown length"
            else:
                aa_text = f"{int(aa_value)} aa"

            print(f"{row['id']}  ->  {aa_text}")
            print(f"  Reason: {row['message']}")

        print("-" * 72)
        print(
            "These rows are excluded from the manifest. "
            "See the validation sheet for details."
        )
    else:
        print()
        print("All architectures are exactly the expected length.")

    print("=" * 72)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert columns A:B of the '3 motifs' tab "
            "into a 300/370 K CALVADOS manifest."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_SHEET,
    )
    parser.add_argument(
        "--lambda-300",
        type=Path,
        default=DEFAULT_LAMBDA_300,
    )
    parser.add_argument(
        "--lambda-370",
        type=Path,
        default=DEFAULT_LAMBDA_370,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
    )
    parser.add_argument(
        "--expected-aa-count",
        type=int,
        default=420,
        help=(
            "Required sequence length. "
            "Set to 0 to disable the fixed-length check."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop immediately on the first invalid architecture. "
            "Without this flag, invalid rows are listed and skipped."
        ),
    )
    parser.add_argument(
        "--fail-if-invalid",
        action="store_true",
        help=(
            "Write the workbook and full validation report, "
            "then exit with status code 1 if invalid rows exist."
        ),
    )
    args = parser.parse_args()

    expected = (
        None
        if args.expected_aa_count == 0
        else args.expected_aa_count
    )

    rows = read_architecture_rows(
        args.input,
        args.sheet,
    )
    lambda_300 = load_lambda_table(args.lambda_300)
    lambda_370 = load_lambda_table(args.lambda_370)

    manifest, validation = build_manifest(
        rows=rows,
        lambda_300=lambda_300,
        lambda_370=lambda_370,
        base_seed=args.seed,
        expected_aa_count=expected,
        strict=args.strict,
    )

    write_output(
        manifest=manifest,
        validation=validation,
        output_path=args.output,
    )

    print_summary(
        rows=rows,
        manifest=manifest,
        validation=validation,
        input_path=args.input,
        sheet_name=args.sheet,
        output_path=args.output,
        temperatures=(300, 370),
    )

    error_count = int(
        (validation["status"] == "ERROR").sum()
    )

    if args.fail_if_invalid and error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()