"""Batched ProtGPT3 sampling filtered to AMP Challenge peptides."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from amp_competition.config import seed_everything
from amp_competition.constants import MAX_LENGTH, MIN_LENGTH
from amp_competition.generator.protgpt3 import (
    DIRECTION_N2C,
    challenge_suppress_token_ids,
    decode_protein,
    generate_sequences,
    is_challenge_alphabet,
)
from amp_competition.io import write_fasta

def load_reference_fasta(path: Path) -> frozenset[str]:
    """Fast loading of sequences from FASTA into a hash set in O(1)."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference FASTA file not found: {path}")

    sequences: set[str] = set()
    current_seq: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.add("".join(current_seq).upper())
                    current_seq.clear()
            else:
                current_seq.append(line)
        if current_seq:
            sequences.add("".join(current_seq).upper())

    return frozenset(sequences)

def is_valid_peptide(
    sequence: str,
    *,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
) -> bool:
    return min_length <= len(sequence) <= max_length and is_challenge_alphabet(sequence)


def _accept(text: str, min_length: int, max_length: int) -> str | None:
    parsed = decode_protein(text)
    if parsed.direction != "N2C":
        return None
    if not is_valid_peptide(parsed.sequence, min_length=min_length, max_length=max_length):
        return None
    return parsed.sequence


def generate_library(
    tokenizer,
    model,
    n_sequences: int,
    *,
    reference_sequences: Set[str] | None = None,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
    batch_size: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 42,
    prompt: str = DIRECTION_N2C,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 10_000,
) -> tuple[list[str], dict[str, Any]]:
    """Sample unique N→C peptides until ``n_sequences`` valid ones are collected."""
    if n_sequences <= 0:
        raise ValueError("n_sequences must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("ProtGPT3 mass generation requires CUDA")

    seed_everything(seed)
    length_rng = np.random.default_rng(seed)

    suppress_tokens = challenge_suppress_token_ids(tokenizer)
    seen: set[str] = set()
    sequences: list[str] = []
    stats = {
        "drawn": 0,
        "accepted": 0,
        "duplicates": 0,
        "reference_matches": 0,
        "rejected": 0,
        "oom_retries": 0,
        "batch_size": batch_size,
        "seed": seed,
        "min_length": min_length,
        "max_length": max_length,
    }
    started = time.monotonic()
    progress = tqdm(total=n_sequences, desc="peptides", unit="seq")

    try:
        while len(sequences) < n_sequences:
            needed = n_sequences - len(sequences)
            current_batch = min(batch_size, max(needed, 8))
            target_length = int(length_rng.integers(min_length, max_length + 1))
            try:
                texts = generate_sequences(
                    tokenizer,
                    model,
                    prompt,
                    num_return_sequences=current_batch,
                    min_new_tokens=target_length,
                    max_new_tokens=target_length,
                    temperature=temperature,
                    top_p=top_p,
                    seed=None,
                    suppress_tokens=suppress_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                stats["oom_retries"] += 1
                stats["batch_size"] = batch_size
                tqdm.write(f"CUDA OOM, reducing batch_size to {batch_size}")
                continue

            stats["drawn"] += len(texts)
            for text in texts:
                peptide = _accept(text, min_length, max_length)
                if peptide is None:
                    stats["rejected"] += 1
                    continue
                if peptide in seen:
                    stats["duplicates"] += 1
                    continue
                if reference_sequences is not None and peptide in reference_sequences:
                    stats["reference_matches"] += 1
                    continue
                seen.add(peptide)
                sequences.append(peptide)
                stats["accepted"] += 1
                progress.update(1)
                if (
                    checkpoint_path is not None
                    and checkpoint_every > 0
                    and stats["accepted"] % checkpoint_every == 0
                ):
                    write_fasta(sequences, checkpoint_path)
                    tqdm.write(f"checkpoint {len(sequences)} → {checkpoint_path}")
                if len(sequences) >= n_sequences:
                    break
    finally:
        progress.close()

    stats["elapsed_sec"] = round(time.monotonic() - started, 2)
    stats["n_sequences"] = len(sequences)
    if sequences:
        lengths = [len(seq) for seq in sequences]
        stats["length_min"] = min(lengths)
        stats["length_max"] = max(lengths)
        stats["length_mean"] = round(sum(lengths) / len(lengths), 2)
        stats["length_hist"] = {str(k): int(v) for k, v in sorted(Counter(lengths).items())}
        stats["accept_rate"] = round(stats["accepted"] / max(stats["drawn"], 1), 4)
    return sequences[:n_sequences], stats


def write_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
