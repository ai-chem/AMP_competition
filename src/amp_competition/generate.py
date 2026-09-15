"""Challenge entry point: `uv run generate`.

Mass-samples ProtGPT3-1.3B peptides of length 8–50 (unique, 20-AA alphabet).
Ranking for top.fasta is still a placeholder until MIC/hemolysis predictors
are wired.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from amp_competition.config import (
    DEFAULT_SEED,
    REPO_ROOT,
    finish_run,
    load_config,
    save_run,
    seed_everything,
)
from amp_competition.constants import LIBRARY_SIZE, MAX_LENGTH, MIN_LENGTH, TOP_SIZE
from amp_competition.generator.protgpt3 import DIRECTION_N2C, load_from_config
from amp_competition.generator.sample import generate_library, load_reference_fasta, write_stats
from amp_competition.io import write_fasta
from amp_competition.filters.similarity import select_top_novel


def placeholder_score(sequences: list[str], seed: int = DEFAULT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(len(sequences)).tolist()


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="generate.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    generation = config.get("generation", {})

    default_ref = generation.get("reference", "data/external/antibacterial.fasta")
    ref_path = Path(default_ref)
    if not ref_path.is_absolute():
        ref_path = REPO_ROOT / ref_path

    parser = argparse.ArgumentParser(description="Generate AMP Challenge FASTA outputs.")
    parser.add_argument("--config", default=pre_args.config, help="YAML in configs/ or a path")
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=int(generation.get("n_sequences", LIBRARY_SIZE)),
    )
    parser.add_argument("--top-k", type=int, default=int(generation.get("top_k", TOP_SIZE)))
    parser.add_argument("--min-length", type=int, default=int(generation.get("min_length", MIN_LENGTH)))
    parser.add_argument("--max-length", type=int, default=int(generation.get("max_length", MAX_LENGTH)))
    parser.add_argument("--batch-size", type=int, default=int(generation.get("batch_size", 256)))
    parser.add_argument("--temperature", type=float, default=float(generation.get("temperature", 0.8)))
    parser.add_argument("--top-p", type=float, default=float(generation.get("top_p", 0.9)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    parser.add_argument(
        "--reference",
        type=Path,
        default=ref_path,
        help="Path to reference FASTA dataset to ensure novelty.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "generate",
        help="Directory for library.fasta / top.fasta (default: repo generate/)",
    )
    args = parser.parse_args()

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("generation", {})
    resolved["generation"].update(
        {
            "n_sequences": args.n_sequences,
            "top_k": args.top_k,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("generate", resolved, out_dir=out_dir, extra={"out_dir": str(out_dir)})

    print(
        f"ProtGPT3 mass generation: n={args.n_sequences} "
        f"len={args.min_length}-{args.max_length} batch={args.batch_size} "
        f"seed={args.seed} run_id={run['run_id']}",
        file=sys.stderr,
    )

    reference_sequences: frozenset[str] = frozenset()
    if args.reference.is_file():
        reference_sequences = load_reference_fasta(args.reference)
        print(
            f"Loaded {len(reference_sequences)} reference sequences from {args.reference}",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: Reference dataset not found at {args.reference}. Exact match filter skipped!",
            file=sys.stderr,
        )

    status = "failed"
    stats: dict = {}
    try:
        tokenizer, model = load_from_config(resolved)
        sequences, stats = generate_library(
            tokenizer,
            model,
            args.n_sequences,
            reference_sequences=reference_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            prompt=str(generation.get("prompt", DIRECTION_N2C)),
            checkpoint_path=out_dir / "library.partial.fasta",
            checkpoint_every=int(generation.get("checkpoint_every", 10_000)),
        )

        library_path = out_dir / "library.fasta"
        write_fasta(sequences, library_path)
        write_stats(stats, out_dir / "library.stats.json")
        print(f"Generated {len(sequences)} sequences → {library_path}")
        print(f"stats={stats}")

        scores = placeholder_score(sequences, seed=args.seed)
        ranked = sorted(zip(scores, sequences, strict=True), key=lambda item: item[0], reverse=True)

        top_sequences, top_filter_stats = select_top_novel(
            ranked_candidates=ranked,
            reference_sequences=reference_sequences,
            top_k=args.top_k,
            max_similarity=0.80
        )
        print(f"Top-K novelty filtering stats: {top_filter_stats}", file=sys.stderr)
    
        top_path = out_dir / "top.fasta"
        write_fasta(top_sequences, top_path)
        print(
            f"Top {len(top_sequences)} sequences → {top_path} "
            "(placeholder scores; predictors not wired)",
            file=sys.stderr,
        )
        status = "completed"
        finish_run(
            run,
            extra={
                "library": str(library_path),
                "top": str(top_path),
                "n_sequences": len(sequences),
                "stats": stats,
            },
            status=status,
        )
    except Exception:
        finish_run(run, extra={"stats": stats}, status=status)
        raise


if __name__ == "__main__":
    main()
