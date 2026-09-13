"""Build AMP and putative non-AMP cohorts for physicochemical comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from amp_competition.constants import MAX_LENGTH, MIN_LENGTH
from amp_competition.features.physchem_constants import CANONICAL_AA
from amp_competition.io import FastaRecord, iter_fasta, write_fasta_records

if TYPE_CHECKING:
    import numpy as np


def load_amp_records(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    seen: set[str] = set()

    for header, sequence in iter_fasta(path):
        if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
            continue
        if not set(sequence) <= CANONICAL_AA:
            continue
        if sequence in seen:
            continue
        seen.add(sequence)
        records.append((header, sequence))

    if not records:
        raise ValueError("No valid AMP sequences remained after filtering.")

    return records


def clean_nonamp_records(
    nonamp_fasta: Path,
    amp_sequences: set[str],
    output_path: Path,
) -> tuple[list[FastaRecord], Counter]:
    """
    Filter UniProt records while preserving the first accession for each unique
    sequence. Exact AMP-reference matches are excluded.
    """
    seen: set[str] = set()
    kept: list[FastaRecord] = []
    stats: Counter = Counter()

    for header, sequence in iter_fasta(nonamp_fasta):
        stats["input_records"] += 1

        if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
            stats["invalid_length"] += 1
            continue

        if not set(sequence) <= CANONICAL_AA:
            stats["noncanonical"] += 1
            continue

        if sequence in amp_sequences:
            stats["exact_amp_match"] += 1
            continue

        if sequence in seen:
            stats["duplicate_sequence"] += 1
            continue

        seen.add(sequence)
        kept.append((header, sequence))

    stats["kept_unique_nonamp"] = len(kept)
    write_fasta_records(kept, output_path)

    return kept, stats


def sample_unmatched(
    records: list[FastaRecord],
    amp_size: int,
    size_spec: str,
    rng: np.random.Generator,
) -> list[FastaRecord]:
    if size_spec == "all":
        return list(records)

    if size_spec == "amp":
        target = amp_size
    else:
        target = int(size_spec)

    target = min(target, len(records))
    indices = rng.choice(len(records), size=target, replace=False)
    return [records[i] for i in indices]


def sample_length_matched(
    records: list[FastaRecord],
    amp_records: list[FastaRecord],
    rng: np.random.Generator,
) -> list[FastaRecord]:
    """
    Exact frequency matching by sequence length.

    For each peptide length L, sample the same number of non-AMP sequences as
    present in the AMP dataset. Sampling is without replacement.
    """
    negatives_by_length: defaultdict[int, list[FastaRecord]] = defaultdict(list)
    for record in records:
        negatives_by_length[len(record[1])].append(record)

    amp_length_counts = Counter(len(sequence) for _, sequence in amp_records)
    matched: list[FastaRecord] = []
    shortages: dict[int, dict[str, int]] = {}

    for length, required in sorted(amp_length_counts.items()):
        candidates = negatives_by_length.get(length, [])
        available = len(candidates)

        if available < required:
            shortages[length] = {
                "required": required,
                "available": available,
            }
            chosen = candidates
        else:
            indices = rng.choice(available, size=required, replace=False)
            chosen = [candidates[i] for i in indices]

        matched.extend(chosen)

    if shortages:
        raise RuntimeError(
            "Not enough negatives for exact length matching. "
            f"Shortages: {shortages}"
        )

    return matched
