"""Challenge entry point: `uv run generate`.

ProtGPT3 generation (tasks 1.3–1.4) is not wired yet. This command currently
writes a deterministic placeholder library so the repository layout, CLI, and
fixed seed already match the submission format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from amp_competition.config import load_config
from amp_competition.constants import (
    DEFAULT_SEED,
    LIBRARY_SIZE,
    MAX_LENGTH,
    MIN_LENGTH,
    STANDARD_AMINO_ACIDS,
    TOP_SIZE,
)
from amp_competition.io import write_fasta

_ALPHABET = np.array(list(STANDARD_AMINO_ACIDS))


def placeholder_generate(
    n_sequences: int,
    *,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Deterministic unique sequences in the legal AMP Challenge alphabet."""
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    sequences: list[str] = []
    while len(sequences) < n_sequences:
        length = int(rng.integers(min_length, max_length + 1))
        sequence = "".join(rng.choice(_ALPHABET, size=length))
        if sequence not in seen:
            seen.add(sequence)
            sequences.append(sequence)
    return sequences


def placeholder_score(sequences: list[str], seed: int = DEFAULT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(len(sequences)).tolist()


def main() -> None:
    config = load_config()
    generation = config.get("generation", {})

    parser = argparse.ArgumentParser(description="Generate AMP Challenge FASTA outputs.")
    parser.add_argument("--n-sequences", type=int, default=generation.get("n_sequences", LIBRARY_SIZE))
    parser.add_argument("--top-k", type=int, default=generation.get("top_k", TOP_SIZE))
    parser.add_argument("--min-length", type=int, default=generation.get("min_length", MIN_LENGTH))
    parser.add_argument("--max-length", type=int, default=generation.get("max_length", MAX_LENGTH))
    parser.add_argument("--seed", type=int, default=config.get("seed", DEFAULT_SEED))
    args = parser.parse_args()

    out_dir = Path(Path(sys.argv[0]).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Using placeholder generator (ProtGPT3 is not connected yet). "
        f"seed={args.seed}",
        file=sys.stderr,
    )
    sequences = placeholder_generate(
        args.n_sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    library_path = out_dir / "library.fasta"
    write_fasta(sequences, library_path)
    print(f"Generated {len(sequences)} sequences → {library_path}")

    scores = placeholder_score(sequences, seed=args.seed)
    ranked = sorted(zip(scores, sequences, strict=True), key=lambda item: item[0], reverse=True)
    top_sequences = [sequence for _, sequence in ranked[: args.top_k]]

    top_path = out_dir / "top.fasta"
    write_fasta(top_sequences, top_path)
    print(f"Top {args.top_k} sequences → {top_path}")
