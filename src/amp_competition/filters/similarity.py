"""Levenshtein similarity filtering with the organizer's ratio.

``Levenshtein.ratio`` is ``(len1 + len2 - dist) / (len1 + len2)``, the same
function ``scripts/verify_submission.py`` uses. A peptide fails when that
ratio is strictly greater than ``max_similarity``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import Levenshtein
from Bio.SeqRecord import SeqRecord


def _ratio_length_bounds(length: int, threshold: float) -> tuple[int, int]:
    """Lengths that can still produce ``Levenshtein.ratio`` above ``threshold``.

    The maximum ratio at a fixed pair of lengths is ``2 * min / (len1 + len2)``,
    reached when the edit distance is only the length difference.
    """
    if length <= 0 or not 0.0 < threshold <= 1.0:
        return 1, 0
    minimum = math.floor(length * threshold / (2.0 - threshold)) + 1
    maximum = math.ceil(length * (2.0 - threshold) / threshold) - 1
    return max(minimum, 1), maximum


class LevenshteinNoveltyFilter:
    """Filters sequences based on the Levenshtein distance to the reference for top-100."""

    def __init__(
        self,
        reference_sequences: Iterable[SeqRecord],
        max_similarity: float = 0.80,
    ) -> None:
        """Args:

        reference_sequences: Collection of sequences from the reference dataset.
        max_similarity: The maximum allowable similarity fraction (default is 0.80 =
        80%).
        """
        if not (0.0 < max_similarity <= 1.0):
            raise ValueError(f"max_similarity must be in (0.0, 1.0], got {max_similarity}")

        self.max_similarity = max_similarity
        self._ref_by_length: dict[int, list[str]] = defaultdict(list)

        for record in reference_sequences:
            seq_str = str(record.seq).upper().strip()
            if seq_str:
                self._ref_by_length[len(seq_str)].append(seq_str) 

        self.total_references = sum(len(seqs) for seqs in self._ref_by_length.values())

    def is_novel(self, record: SeqRecord) -> bool:
        """Return True when every reference has ratio <= max_similarity."""
        if self.total_references == 0:
            return True

        seq = str(record.seq).upper().strip()
        threshold = self.max_similarity
        min_len, max_len = _ratio_length_bounds(len(seq), threshold)

        for ref_len in range(min_len, max_len + 1):
            candidates = self._ref_by_length.get(ref_len)
            if not candidates:
                continue

            for ref_seq in candidates:
                if Levenshtein.ratio(seq, ref_seq) > threshold:
                    return False

        return True


def select_top_novel(
    ranked_candidates: Sequence[SeqRecord],
    reference_sequences: Iterable[SeqRecord],
    top_k: int = 100,
    max_similarity: float = 0.80,
) -> tuple[list[SeqRecord], dict[str, Any]]:
    """Selects top_k candidates from the ranked list that satisfy the novelty condition.

    Args:
        ranked_candidates: The SeqRecord list, ALREADY sorted by model score.
        reference_sequences: Reference dataset.
        top_k: The required number of sequences in the top list (default is 100).
        max_similarity: Similarity threshold (default is 0.80).

    Returns:
        (top_sequences, stats_dict)
    """
    novelty_filter = LevenshteinNoveltyFilter(
        reference_sequences=reference_sequences,
        max_similarity=max_similarity,
    )

    selected: list[SeqRecord] = []
    inspected = 0
    rejected_similarity = 0

    for record in ranked_candidates:
        inspected += 1
        if novelty_filter.is_novel(record):
            selected.append(record)
            if len(selected) >= top_k:
                break
        else:
            rejected_similarity += 1

    stats = {
        "top_k_requested": top_k,
        "top_k_collected": len(selected),
        "candidates_inspected": inspected,
        "rejected_by_similarity": rejected_similarity,
    }

    return selected, stats


def filter_reference_novel(
    ranked_candidates: Sequence[SeqRecord],
    reference_sequences: Iterable[SeqRecord],
    max_similarity: float = 0.80,
    log_every: int = 2000,
) -> tuple[list[SeqRecord], dict[str, Any]]:
    """Drop every candidate too similar to the reference.

    The walk covers the whole input and does not stop at a target count.
    Similarity among the candidates themselves is not checked. Input order is
    preserved, so a score-sorted list stays score-sorted.
    """
    import logging
    import time

    log = logging.getLogger("generate")
    novelty = LevenshteinNoveltyFilter(
        reference_sequences=reference_sequences,
        max_similarity=max_similarity,
    )
    kept: list[SeqRecord] = []
    rejected = 0
    started = time.monotonic()
    total = len(ranked_candidates)

    for index, record in enumerate(ranked_candidates, start=1):
        if novelty.is_novel(record):
            kept.append(record)
        else:
            rejected += 1
        if log_every and index % log_every == 0:
            log.info(
                "REFERENCE SIMILARITY progress %s/%s kept=%s rejected=%s seconds=%.1f",
                index,
                total,
                len(kept),
                rejected,
                time.monotonic() - started,
            )

    stats = {
        "inspected": total,
        "kept": len(kept),
        "rejected_by_reference_similarity": rejected,
        "max_similarity": max_similarity,
        "seconds": round(time.monotonic() - started, 1),
    }
    log.info(
        "REFERENCE SIMILARITY done inspected=%s kept=%s rejected=%s seconds=%.1f",
        total,
        len(kept),
        rejected,
        time.monotonic() - started,
    )
    return kept, stats


class InternalDiversityFilter:
    """Internal diversity selection.

    It goes through a sorted list of candidates and adds the peptide to the pool
    if it is no more than 80% similar to those already selected for this pool.
    """

    def __init__(self, max_internal_similarity: float = 0.80) -> None:
        self.max_similarity = max_internal_similarity
        self._pool_by_length: dict[int, list[str]] = defaultdict(list)
        self.selected_pool: list[SeqRecord] = []
        
    def can_add(self, record: SeqRecord) -> bool:
        seq = str(record.seq).upper().strip()
        threshold = self.max_similarity
        min_len, max_len = _ratio_length_bounds(len(seq), threshold)

        for ref_len in range(min_len, max_len + 1):
            pool_group = self._pool_by_length.get(ref_len)
            if not pool_group:
                continue

            for accepted_seq in pool_group:
                if Levenshtein.ratio(seq, accepted_seq) > threshold:
                    return False

        return True

    def add(self, record: SeqRecord) -> None:
        seq_str = str(record.seq).upper().strip()
        self._pool_by_length[len(seq_str)].append(seq_str)
        self.selected_pool.append(record)



def drop_similar(
    candidates: Sequence[SeqRecord],
    reference_sequences: Iterable[SeqRecord],
    max_similarity: float = 0.80,
    log_every: int = 2000,
) -> tuple[list[SeqRecord], dict[str, Any]]:
    """Drop every candidate that fails the Levenshtein checks.

    A candidate is removed when its Levenshtein ratio is above ``max_similarity``
    to the reference or to a candidate already kept. The walk covers the whole
    input and does not stop at a target count.
    """
    import logging
    import time

    log = logging.getLogger("generate")
    novelty = LevenshteinNoveltyFilter(
        reference_sequences=reference_sequences,
        max_similarity=max_similarity,
    )
    internal = InternalDiversityFilter(max_internal_similarity=max_similarity)
    kept: list[SeqRecord] = []
    rejected_reference = 0
    rejected_internal = 0
    started = time.monotonic()
    total = len(candidates)

    for index, record in enumerate(candidates, start=1):
        if not novelty.is_novel(record):
            rejected_reference += 1
        elif not internal.can_add(record):
            rejected_internal += 1
        else:
            internal.add(record)
            kept.append(record)
        if log_every and index % log_every == 0:
            log.info(
                "LEVENSHTEIN progress %s/%s kept=%s rejected_reference=%s rejected_internal=%s seconds=%.1f",
                index,
                total,
                len(kept),
                rejected_reference,
                rejected_internal,
                time.monotonic() - started,
            )

    stats = {
        "inspected": total,
        "kept": len(kept),
        "rejected_reference": rejected_reference,
        "rejected_internal": rejected_internal,
        "max_similarity": max_similarity,
    }
    return kept, stats


def select_diverse_library(
    ranked_candidates: Sequence[SeqRecord],
    target_count: int = 50_000,
    max_internal_similarity: float = 0.80,
) -> tuple[list[SeqRecord], dict[str, Any]]:
    """Selects target_count candidates with internal diversity control.

    Args:
        ranked_candidates: A list of sequences sorted by quality/speed.
        target_count: How many candidates need to be collected (default = 50,000).
        max_internal_similarity: The threshold of similarity between the selected (default = 0.80).
    """
    engine = InternalDiversityFilter(max_internal_similarity=max_internal_similarity)

    inspected = 0
    rejected_internal = 0

    for record in ranked_candidates:
        inspected += 1
        if engine.can_add(record):
            engine.add(record)
            if len(engine.selected_pool) >= target_count:
                break
        else:
            rejected_internal += 1

    stats = {
        "target_count": target_count,
        "collected_count": len(engine.selected_pool),
        "inspected_candidates": inspected,
        "rejected_internal_similarity": rejected_internal,
    }
    return engine.selected_pool, stats