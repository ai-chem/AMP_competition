"""Levenshtein similarity filtering against reference sequences using RapidFuzz."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from Bio.SeqRecord import SeqRecord
from rapidfuzz.distance import Levenshtein


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
        """Returns True if the sequence is new (similarity <= max_similarity with all).
        Returns False if at least one match with similarity > max_similarity is found.
        """
        if self.total_references == 0:
            return True

        seq = str(record.seq).upper().strip()
        target_len = len(seq)
        threshold = self.max_similarity

        # Mathematical boundaries of lengths: outside this range, the similarity cannot be > threshold
        min_len = math.ceil(target_len * threshold)
        max_len = math.floor(target_len / threshold)

        for ref_len in range(min_len, max_len + 1):
            candidates = self._ref_by_length.get(ref_len)
            if not candidates:
                continue

            for ref_seq in candidates:
                sim = Levenshtein.normalized_similarity(seq, ref_seq, score_cutoff=threshold)
                if sim > threshold:
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
        target_len = len(seq)
        threshold = self.max_similarity

        min_len = math.ceil(target_len * threshold)
        max_len = math.floor(target_len / threshold)

        for ref_len in range(min_len, max_len + 1):
            pool_group = self._pool_by_length.get(ref_len)
            if not pool_group:
                continue

            for accepted_seq in pool_group:
                sim = Levenshtein.normalized_similarity(seq, accepted_seq, score_cutoff=threshold)
                if sim > threshold:
                    return False

        return True

    def add(self, record: SeqRecord) -> None:
        seq_str = str(record.seq).upper().strip()
        self._pool_by_length[len(seq_str)].append(seq_str)
        self.selected_pool.append(record)



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