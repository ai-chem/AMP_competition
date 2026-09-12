"""Load AMP peptide tables and make a reproducible train/val split."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from amp_competition.constants import MAX_LENGTH, MIN_LENGTH, STANDARD_AMINO_ACIDS


def is_train_peptide(
    sequence: str,
    *,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
) -> bool:
    return min_length <= len(sequence) <= max_length and all(
        residue in STANDARD_AMINO_ACIDS for residue in sequence
    )


def load_peptide_csv(
    path: Path | str,
    *,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
    sequence_column: str = "sequence",
) -> tuple[list[str], dict[str, int]]:
    """Read unique challenge-alphabet peptides from a CSV with a ``sequence`` column."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    stats = {
        "rows": 0,
        "empty": 0,
        "invalid": 0,
        "duplicates": 0,
        "accepted": 0,
    }
    seen: set[str] = set()
    sequences: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or sequence_column not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing column {sequence_column!r}")
        for row in reader:
            stats["rows"] += 1
            sequence = (row.get(sequence_column) or "").strip().upper()
            if not sequence:
                stats["empty"] += 1
                continue
            if not is_train_peptide(sequence, min_length=min_length, max_length=max_length):
                stats["invalid"] += 1
                continue
            if sequence in seen:
                stats["duplicates"] += 1
                continue
            seen.add(sequence)
            sequences.append(sequence)
    stats["accepted"] = len(sequences)
    return sequences, stats


def split_train_val(
    sequences: list[str],
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Length-stratified shuffle split so every length can appear in val."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(sequences) < 2:
        raise ValueError("Need at least two sequences to split")

    by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in sequences:
        by_length[len(sequence)].append(sequence)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    for group in by_length.values():
        rng.shuffle(group)
        n_val = int(round(len(group) * val_fraction))
        if len(group) >= 10:
            n_val = max(1, n_val)
        n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    if not train or not val:
        raise ValueError("Train/val split produced an empty split")
    return train, val


def length_histogram(sequences: list[str]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for sequence in sequences:
        counts[len(sequence)] += 1
    return {str(length): counts[length] for length in sorted(counts)}


def write_split(path: Path, train: list[str], val: list[str], extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_train": len(train),
        "n_val": len(val),
        "train_length_hist": length_histogram(train),
        "val_length_hist": length_histogram(val),
        "train": train,
        "val": val,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
