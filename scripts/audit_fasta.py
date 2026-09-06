#!/usr/bin/env python3
"""Audit a peptide FASTA file and write a cleaned CSV dataset.

The cleaning policy mirrors the reference-dataset notebook: keep only unique,
non-empty sequences of 8–50 canonical amino acids. The input FASTA is never
modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Iterable


DEFAULT_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CLEANED_COLUMNS = [
    "record_id",
    "charge",
    "disulfide",
    "source_databases",
    "activity_tags",
    "header",
    "sequence",
    "sequence_length",
]
REJECTED_COLUMNS = [
    "source_record_number",
    "record_id",
    "charge",
    "disulfide",
    "source_databases",
    "activity_tags",
    "header",
    "sequence",
    "sequence_length",
    "invalid_residues",
    "duplicate_of_record_number",
    "rejection_reasons",
]


def _parse_header(header: str) -> dict[str, object]:
    """Extract structured organizer metadata from a FASTA header."""
    tokens = header.split()
    metadata = dict(
        token.split("=", maxsplit=1) for token in tokens[1:] if "=" in token
    )
    return {
        "record_id": tokens[0] if tokens else "",
        "charge": float(metadata["charge"]) if "charge" in metadata else None,
        "disulfide": int(metadata["disulfide"])
        if "disulfide" in metadata
        else None,
        "source_databases": metadata.get("dbs", "").split("|")
        if metadata.get("dbs")
        else [],
        "activity_tags": metadata.get("activity", "").split("|")
        if metadata.get("activity")
        else [],
    }


def read_fasta(path: Path) -> list[dict[str, object]]:
    """Read a FASTA file while preserving each complete header."""
    records: list[dict[str, object]] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def append_record() -> None:
        if header is None:
            return
        sequence = "".join(sequence_parts).upper()
        records.append(
            {
                "source_record_number": len(records) + 1,
                **_parse_header(header),
                "header": header,
                "sequence": sequence,
                "sequence_length": len(sequence),
            }
        )

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            append_record()
            header = line[1:].strip()
            sequence_parts = []
        elif header is None:
            raise ValueError(
                f"Found sequence content before the first FASTA header at line {line_number}."
            )
        else:
            sequence_parts.append(line)

    append_record()
    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    return records


def _sample_unique_pairs(
    n_sequences: int,
    n_pairs: int,
    seed: int,
) -> Iterable[tuple[int, int]]:
    """Return reproducible unordered, unique index pairs without all-vs-all work."""
    max_pairs = n_sequences * (n_sequences - 1) // 2
    target_pairs = min(n_pairs, max_pairs)
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()

    while len(pairs) < target_pairs:
        first_index, second_index = rng.sample(range(n_sequences), k=2)
        pairs.add(tuple(sorted((first_index, second_index))))

    return pairs


def sampled_similarity_summary(
    sequences: list[str],
    n_pairs: int,
    seed: int,
) -> dict[str, int | float] | None:
    """Summarize sampled Levenshtein similarity for a unique sequence set."""
    if n_pairs == 0 or len(sequences) < 2:
        return None

    try:
        from Levenshtein import ratio
    except ImportError as error:
        raise RuntimeError(
            "Levenshtein is required for similarity analysis. "
            "Install project dependencies with 'uv sync'."
        ) from error

    similarities = [
        ratio(sequences[first_index], sequences[second_index])
        for first_index, second_index in _sample_unique_pairs(
            len(sequences), n_pairs, seed
        )
    ]
    return {
        "sampled_pairs": len(similarities),
        "seed": seed,
        "mean": round(statistics.fmean(similarities), 6),
        "median": round(statistics.median(similarities), 6),
        "fraction_above_0_60": round(
            sum(value > 0.60 for value in similarities) / len(similarities), 6
        ),
        "fraction_above_0_80": round(
            sum(value > 0.80 for value in similarities) / len(similarities), 6
        ),
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: json.dumps(row[fieldname])
                    if isinstance(row.get(fieldname), list)
                    else row.get(fieldname, "")
                    for fieldname in fieldnames
                }
            )


def audit_fasta(
    input_path: Path,
    cleaned_output: Path,
    rejected_output: Path,
    report_output: Path,
    min_length: int,
    max_length: int,
    alphabet: str,
    sample_pairs: int,
    seed: int,
) -> dict[str, object]:
    """Audit input records, write outputs, and return the JSON-ready report."""
    if min_length > max_length:
        raise ValueError("min_length must not exceed max_length.")
    if sample_pairs < 0:
        raise ValueError("sample_pairs must be zero or positive.")

    records = read_fasta(input_path)
    allowed_residues = set(alphabet)
    clean_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    first_clean_occurrence: dict[str, int] = {}
    rejection_counts: Counter[str] = Counter()

    for record in records:
        sequence = str(record["sequence"])
        invalid_residues = "".join(sorted(set(sequence) - allowed_residues))
        rejection_reasons: list[str] = []
        duplicate_of_record_number = ""

        if not str(record["header"]):
            rejection_reasons.append("missing_header")
        if not sequence:
            rejection_reasons.append("empty_sequence")
        if invalid_residues:
            rejection_reasons.append(f"invalid_residues:{invalid_residues}")
        if not min_length <= int(record["sequence_length"]) <= max_length:
            rejection_reasons.append(
                f"length_outside_{min_length}_{max_length}"
            )

        if not rejection_reasons and sequence in first_clean_occurrence:
            duplicate_of_record_number = str(first_clean_occurrence[sequence])
            rejection_reasons.append("duplicate_sequence")

        if rejection_reasons:
            for reason in rejection_reasons:
                rejection_counts[reason.split(":", maxsplit=1)[0]] += 1
            rejected_records.append(
                {
                    **record,
                    "invalid_residues": invalid_residues,
                    "duplicate_of_record_number": duplicate_of_record_number,
                    "rejection_reasons": ";".join(rejection_reasons),
                }
            )
        else:
            first_clean_occurrence[sequence] = int(record["source_record_number"])
            clean_records.append(
                {
                    fieldname: record[fieldname] for fieldname in CLEANED_COLUMNS
                }
            )

    source_sequences = [str(record["sequence"]) for record in records]
    nonempty_source_sequences = [sequence for sequence in source_sequences if sequence]
    source_lengths = [int(record["sequence_length"]) for record in records]
    source_sequence_counts = Counter(nonempty_source_sequences)
    amino_acid_counts = Counter("".join(str(record["sequence"]) for record in clean_records))

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        },
        "policy": {
            "canonical_alphabet": alphabet,
            "min_length": min_length,
            "max_length": max_length,
            "duplicates": "Keep the first clean occurrence; reject later occurrences.",
        },
        "counts": {
            "input_records": len(records),
            "unique_nonempty_sequences": len(source_sequence_counts),
            "exact_duplicate_records": len(nonempty_source_sequences)
            - len(source_sequence_counts),
            "clean_records": len(clean_records),
            "rejected_records": len(rejected_records),
        },
        "lengths": {
            "minimum": min(source_lengths),
            "maximum": max(source_lengths),
            "mean": round(statistics.fmean(source_lengths), 6),
            "median": statistics.median(source_lengths),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "canonical_amino_acid_counts": {
            amino_acid: amino_acid_counts[amino_acid] for amino_acid in alphabet
        },
        "sampled_pairwise_levenshtein_similarity": sampled_similarity_summary(
            [str(record["sequence"]) for record in clean_records],
            sample_pairs,
            seed,
        ),
        "outputs": {
            "cleaned_csv": str(cleaned_output),
            "rejected_csv": str(rejected_output),
        },
    }

    _write_csv(cleaned_output, clean_records, CLEANED_COLUMNS)
    _write_csv(rejected_output, rejected_records, REJECTED_COLUMNS)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/antibacterial.fasta"),
        help="Input FASTA file.",
    )
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=Path("data/processed/antibacterial_clean.csv"),
        help="CSV file for records that pass every cleaning rule.",
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=Path("data/processed/antibacterial_rejected.csv"),
        help="CSV file for excluded records and their rejection reasons.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/processed/antibacterial_audit.json"),
        help="JSON audit report.",
    )
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=100_000,
        help="Number of unique pairs for sampled Levenshtein similarity; use 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = audit_fasta(
        input_path=args.input,
        cleaned_output=args.cleaned_output,
        rejected_output=args.rejected_output,
        report_output=args.report_output,
        min_length=args.min_length,
        max_length=args.max_length,
        alphabet=args.alphabet,
        sample_pairs=args.sample_pairs,
        seed=args.seed,
    )
    counts = report["counts"]
    print(f"Input records: {counts['input_records']}")
    print(f"Clean records: {counts['clean_records']}")
    print(f"Rejected records: {counts['rejected_records']}")
    print(f"Audit report: {args.report_output}")


if __name__ == "__main__":
    main()
