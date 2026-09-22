"""Sequential pipeline for peptide filtering and library building.

Pipeline Steps:
1. Syntactic filtering (length 8-50, 20 canonical AAs, pool dedup, exact ref matches)
2. Predictors scoring & sorting (placeholder scoring function)
3. Internal diversity selection (50k library with internal Levenshtein similarity <= 80%)
4. Reference novelty filtering (Levenshtein identity to reference <= 80%)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from amp_competition.filters.similarity import (
    LevenshteinNoveltyFilter,
    select_diverse_library,
)
from amp_competition.filters.syntactic import SyntacticFilter


def placeholder_score(sequences: list[str], seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(len(sequences)).tolist()


def run_pipeline(
    raw_fasta: Path,
    reference_fasta: Path,
    output_fasta: Path,
    output_top: Path,
    target_count: int = 50_000,
    top_k: int = 100,
    max_similarity: float = 0.80,
    seed: int = 42,
) -> None:
    print(f"=== Starting Peptide Filtering Pipeline ===", file=sys.stderr)
    start_time = time.monotonic()

    if not raw_fasta.is_file():
        raise FileNotFoundError(f"The input file was not found: {raw_fasta}")
    if not reference_fasta.is_file():
        raise FileNotFoundError(f"The reference file was not found: {reference_fasta}")

    # 1. Syntactic filtering
    print(f"\n[Step 1/5] Applying Syntactic Filters to {raw_fasta}...", file=sys.stderr)
    syn_filter = SyntacticFilter.from_reference_fasta(
        reference_fasta_path=reference_fasta,
        min_length=8,
        max_length=50,
    )

    raw_records_iter = SeqIO.parse(raw_fasta, "fasta")
    valid_records, syn_stats = syn_filter.filter_records(raw_records_iter)

    print(
        f"  Total input:        {syn_stats['total_input']:,}\n"
        f"  Passed:             {syn_stats['passed']:,}\n"
        f"  Rejected (length):  {syn_stats['rejected_length']:,}\n"
        f"  Rejected (alpha):   {syn_stats['rejected_alphabet']:,}\n"
        f"  Rejected (dups):    {syn_stats['rejected_duplicate']:,}\n"
        f"  Rejected (ref_hit): {syn_stats['rejected_exact_reference']:,}\n"
        f"  Attrition rate:     {syn_stats['attrition_rate']:.2%}",
        file=sys.stderr,
    )

    if not valid_records:
        raise RuntimeError("After syntactic filtering, no candidates remain!")

    # Predictors scoring & sorting (placeholder scoring function)
    print(f"\n[Step 2/5] Scoring and ranking {len(valid_records):,} candidates...", file=sys.stderr)
    sequences_only = [str(r.seq).upper().strip() for r in valid_records]

    scores = placeholder_score(sequences_only, seed=seed)

    ranked_pairs = sorted(
        zip(scores, valid_records, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )

    ranked_records: list[SeqRecord] = []
    for score, record in ranked_pairs:
        record.description = f"{record.id} score={score:.4f}".strip()
        ranked_records.append(record)

    print(f"  Ranked {len(ranked_records):,} candidates from highest to lowest score.", file=sys.stderr)

    # Internal diversity selection (50k library with internal Levenshtein similarity <= 80%)
    print(
        f"\n[Step 3/5] Selecting {target_count:,} diverse candidates (internal identity <= {max_similarity:.0%})...",
        file=sys.stderr,
    )
    div_start = time.monotonic()

    diverse_50k, div_stats = select_diverse_library(
        ranked_candidates=ranked_records,
        target_count=target_count,
        max_internal_similarity=max_similarity,
    )

    print(
        f"  Candidates inspected:   {div_stats['inspected_candidates']:,}\n"
        f"  Collected into pool:    {div_stats['collected_count']:,}\n"
        f"  Rejected (internal sim):{div_stats['rejected_internal_similarity']:,}\n"
        f"  Time taken:             {time.monotonic() - div_start:.2f}s",
        file=sys.stderr,
    )

    # Reference novelty filtering (Levenshtein identity to reference <= 80%)
    print(
        f"\n[Step 4/5] Filtering 50k library against reference dataset (identity <= {max_similarity:.0%})...",
        file=sys.stderr,
    )
    ref_records = list(SeqIO.parse(reference_fasta, "fasta"))
    novelty_filter = LevenshteinNoveltyFilter(
        reference_sequences=ref_records,
        max_similarity=max_similarity,
    )

    final_library: list[SeqRecord] = []
    rejected_by_ref = 0

    for rec in diverse_50k:
        if novelty_filter.is_novel(rec):
            final_library.append(rec)
        else:
            rejected_by_ref += 1

    print(
        f"  Input to novelty filter: {len(diverse_50k):,}\n"
        f"  Passed (identity <= 80%):{len(final_library):,}\n"
        f"  Rejected (> 80% to ref): {rejected_by_ref:,}",
        file=sys.stderr,
    )

    # ШАГ 5: Selecting Top-100 candidates
    # =========================================================================
    print(f"\n[Step 5/5] Selecting Top-{top_k} candidates...", file=sys.stderr)

    top_records = final_library[:top_k]

    if len(top_records) < top_k:
        print(
            f"  WARNING: Only {len(top_records)} candidates available for top-{top_k}!",
            file=sys.stderr,
        )

    # SAVING THE RESULT
    SeqIO.write(top_records, output_top, "fasta")
    print(f"  Top-{len(top_records)} saved: {len(top_records)} records → {output_top}", file=sys.stderr)

    total_time = time.monotonic() - start_time
    print(
        f"\n=== Pipeline Completed in {total_time:.1f}s ===\n"
        f"1. Full library: {len(final_library):,} records → {output_fasta}\n"
        f"2. Top-{len(top_records)} list: {len(top_records)} records → {output_top}\n",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full peptide filtering and selection pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("generate/library.fasta"),
        help="Path to raw sampled FASTA",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/external/antibacterial.fasta"),
        help="Path to reference FASTA dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generate/library2.fasta"),
        help="Output path for filtered 50k library FASTA",
    )
    parser.add_argument(
        "--output_top",
        type=Path,
        default=Path("generate/top2.fasta"),
        help="Output path for filtered 50k library FASTA",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50_000,
        help="Target size of diverse library (default: 50000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates to select (default: 100)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help="Maximum allowable similarity threshold (default: 0.80)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    run_pipeline(
        raw_fasta=args.input,
        reference_fasta=args.reference,
        output_fasta=args.output,
        output_top=args.output_top,
        target_count=args.target_count,
        top_k=args.top_k,
        max_similarity=args.similarity_threshold,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()