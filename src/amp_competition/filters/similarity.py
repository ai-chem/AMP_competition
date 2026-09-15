"""Levenshtein similarity filtering against reference sequences using RapidFuzz."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from rapidfuzz.distance import Levenshtein


class LevenshteinNoveltyFilter:
    """Filters sequences based on the Levenshtein distance to the reference."""

    def __init__(
        self,
        reference_sequences: Iterable[str],
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

        for seq in reference_sequences:
            s = seq.strip().upper()
            if s:
                self._ref_by_length[len(s)].append(s)

        self.total_references = sum(len(seqs) for seqs in self._ref_by_length.values())

    def is_novel(self, sequence: str) -> bool:
        """Returns True if the sequence is new (similarity <= max_similarity with all).
        Returns False if at least one match with similarity > max_similarity is found.
        """
        if self.total_references == 0:
            return True

        seq = sequence.strip().upper()
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
    ranked_candidates: Sequence[tuple[float, str]],
    reference_sequences: Iterable[str],
    top_k: int = 100,
    max_similarity: float = 0.80,
) -> tuple[list[str], dict[str, Any]]:
    """Selects top_k candidates from the ranked list that satisfy the novelty condition.

    Args:
        ranked_candidates: A list of tuples (score, sequence), sorted in descending order of score.
        reference_sequences: Reference dataset.
        top_k: The required number of sequences in the top list.
        max_similarity: Similarity threshold (default is 0.80).

    Returns:
        (top_sequences, stats_dict)
    """
    novelty_filter = LevenshteinNoveltyFilter(
        reference_sequences=reference_sequences,
        max_similarity=max_similarity,
    )

    selected: list[str] = []
    inspected = 0
    rejected_similarity = 0

    for _, seq in ranked_candidates:
        inspected += 1
        if novelty_filter.is_novel(seq):
            selected.append(seq)
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