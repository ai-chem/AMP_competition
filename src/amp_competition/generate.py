"""Challenge entry point: `uv run generate`.

Samples the V2 conditional adapter at five (Q, H) points, 50_000 accepted
peptides per point. Acceptance during sampling is length, alphabet, N→C
direction, within-run duplicates, and exact reference copies.

MIC and HC50 are scored on that pool. HC50 is the ESM-2 35M fine-tune
(µM from the censored head), not the HemoPI2 composition script. The combined
score sorts the pool.
``generate/library.fasta`` is the first 50_000. Reference Levenshtein above
80% is applied only while collecting ``generate/top.fasta`` from that library.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import logging
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from tqdm import tqdm

from amp_competition.config import (
    REPO_ROOT,
    finish_run,
    load_config,
    save_run,
    seed_everything,
)
from amp_competition.constants import LIBRARY_SIZE, MAX_LENGTH, MIN_LENGTH, TOP_SIZE
from amp_competition.filters.similarity import select_top_novel
from amp_competition.filters.syntactic import SyntacticFilter
from amp_competition.generator.conditioning import (
    decode_generated,
    default_suppress,
    load_v2_bundle,
)
from amp_competition.generator.sample import write_stats
from amp_competition.io import write_fasta

LOG = logging.getLogger("generate")


@dataclass
class _Seq:
    """Enough for the Levenshtein filters: they only read ``record.seq``."""

    seq: str
    id: str
    score: float


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    LOG.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(stream)
    LOG.addHandler(file_handler)


def _stage(name: str) -> float:
    LOG.info("STAGE %s", name)
    for handler in LOG.handlers:
        handler.flush()
    return time.monotonic()


def _done(name: str, started: float, **fields: object) -> None:
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    LOG.info("DONE %s seconds=%.1f %s", name, time.monotonic() - started, extra)
    for handler in LOG.handlers:
        handler.flush()


def _load_targets(path: Path) -> list[dict[str, float | str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No condition rows in {path}")
    targets: list[dict[str, float | str]] = []
    for row in rows:
        cluster = str(row["cluster"]).strip()
        targets.append(
            {
                "name": f"cluster{cluster}",
                "charge": float(row["charge_pH7_4"]),
                "hydrophobicity": float(row["hydrophobicity_interfaceScale_pH8"]),
            }
        )
    return targets


def _load_safety_predict():
    path = REPO_ROOT / "peptide-safety-inference" / "predict.py"
    spec = importlib.util.spec_from_file_location("peptide_safety_inference_predict", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hc50_micromolar(sequences: list[str]) -> np.ndarray:
    """HC50 in µM from the ESM-2 35M fine-tune. Skips the slow applicability-domain alignments."""
    root = REPO_ROOT / "peptide-safety-inference"
    checkpoint = root / "models" / "esm2_35M_ft.pt"
    bundle_path = root / "models" / "hc50_bundle.pkl"
    if checkpoint.stat().st_size < 1_000_000 or bundle_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"HC50 weights are missing or still Git LFS pointers: {checkpoint}")
    module = _load_safety_predict()
    bundle = module.load_bundle(bundle_path)
    device = module._resolve_device("cuda")
    head, tokenizer = module._load_ft_head(bundle, device)
    batch = 32
    while True:
        try:
            mu, _sigma = module._predict_ft(head, tokenizer, sequences, device, bs=batch)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 4:
                raise
            batch //= 2
            LOG.info("HC50 batch reduced to %s after OOM", batch)
    del head, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.exp(np.asarray(mu, dtype=np.float64))


def _combined_scores(mic: np.ndarray, hc50: np.ndarray) -> np.ndarray:
    if not np.isfinite(mic).all() or not np.isfinite(hc50).all():
        raise RuntimeError("MIC or HC50 scores contain non-finite values")
    if np.any(hc50 <= 0):
        raise RuntimeError("HC50 scores must be positive")
    count = len(mic)
    mic_rank = rankdata(mic, method="average") / count
    hc_rank = rankdata(np.log(hc50), method="average") / count
    return mic_rank + hc_rank


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
    default_checkpoint = Path(generation.get("checkpoint", "checkpoint/lora_cond_v2/best"))
    default_conditions = Path(generation.get("conditions", "configs/selected_generation_conditions.csv"))
    default_per_cluster = int(generation.get("n_per_cluster", LIBRARY_SIZE))

    parser = argparse.ArgumentParser(description="Generate AMP Challenge FASTA outputs from V2.")
    parser.add_argument("--config", default=pre_args.config, help="YAML in configs/ or a path")
    parser.add_argument("--n-per-cluster", type=int, default=default_per_cluster)
    parser.add_argument("--library-size", type=int, default=int(generation.get("library_size", LIBRARY_SIZE)))
    parser.add_argument("--top-k", type=int, default=int(generation.get("top_k", TOP_SIZE)))
    parser.add_argument("--min-length", type=int, default=int(generation.get("min_length", MIN_LENGTH)))
    parser.add_argument("--max-length", type=int, default=int(generation.get("max_length", MAX_LENGTH)))
    parser.add_argument("--batch-size", type=int, default=int(generation.get("batch_size", 32)))
    parser.add_argument("--temperature", type=float, default=float(generation.get("temperature", 0.8)))
    parser.add_argument("--top-p", type=float, default=float(generation.get("top_p", 0.9)))
    parser.add_argument("--max-similarity", type=float, default=float(generation.get("max_similarity", 0.80)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 42)))
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--conditions", type=Path, default=default_conditions)
    parser.add_argument("--reference", type=Path, default=ref_path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "generate",
        help="Directory for library.fasta / top.fasta (default: repo generate/)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=REPO_ROOT / "outputs" / "uv_generate_stages.log",
    )
    args = parser.parse_args()

    _configure_logging(args.log_file if args.log_file.is_absolute() else REPO_ROOT / args.log_file)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    conditions_path = args.conditions if args.conditions.is_absolute() else REPO_ROOT / args.conditions
    reference = args.reference if args.reference.is_absolute() else REPO_ROOT / args.reference
    if not (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "conditioner.pt").is_file():
        raise SystemExit(f"missing V2 checkpoint at {checkpoint}")
    if not conditions_path.is_file():
        raise SystemExit(f"missing condition points at {conditions_path}")
    if not reference.is_file():
        raise SystemExit(f"missing reference FASTA at {reference}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("generation", {})
    resolved["generation"].update(
        {
            "n_per_cluster": args.n_per_cluster,
            "library_size": args.library_size,
            "top_k": args.top_k,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_similarity": args.max_similarity,
            "checkpoint": str(checkpoint),
            "conditions": str(conditions_path),
            "reference": str(reference),
        }
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("generate", resolved, out_dir=out_dir, extra={"out_dir": str(out_dir)})

    targets = _load_targets(conditions_path)
    syntactic = SyntacticFilter.from_reference_fasta(
        reference,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    LOG.info(
        "uv run generate clusters=%s per_cluster=%s library=%s top=%s seed=%s reference=%s log=%s",
        len(targets),
        args.n_per_cluster,
        args.library_size,
        args.top_k,
        args.seed,
        reference,
        args.log_file,
    )

    status = "failed"
    stats: dict = {"per_target": {}, "rejected": {}}
    try:
        started = _stage("generation")
        tokenizer, wrapped = load_v2_bundle(resolved, checkpoint, device="cuda", trainable=False)
        suppress = default_suppress(tokenizer)
        batch_size = args.batch_size
        sequences: list[str] = []
        rejected = {
            "direction": 0,
            "length": 0,
            "alphabet": 0,
            "duplicate": 0,
            "exact_reference": 0,
        }

        for target_index, target in enumerate(targets):
            seed_everything(args.seed + 17 * (target_index + 1), deterministic=True)
            before = len(sequences)
            name = str(target["name"])
            cluster_rejected = {key: 0 for key in rejected}
            progress = tqdm(total=args.n_per_cluster, desc=name, unit="seq", file=sys.stderr)
            while len(sequences) - before < args.n_per_cluster:
                take = min(batch_size, max(8, args.n_per_cluster - (len(sequences) - before)))
                try:
                    texts = wrapped.generate(
                        tokenizer,
                        float(target["charge"]),
                        float(target["hydrophobicity"]),
                        num_return_sequences=take,
                        max_new_tokens=args.max_length,
                        min_new_tokens=args.min_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        suppress_tokens=suppress,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    LOG.info("CUDA OOM, batch_size=%s", batch_size)
                    continue
                for item in decode_generated(texts):
                    if item["direction"] != "N2C":
                        rejected["direction"] += 1
                        cluster_rejected["direction"] += 1
                        continue
                    reason = syntactic.consider(item["sequence"])
                    if reason is not None:
                        rejected[reason] += 1
                        cluster_rejected[reason] += 1
                        continue
                    sequences.append(item["sequence"])
                    progress.update(1)
                    if len(sequences) - before >= args.n_per_cluster:
                        break
            progress.close()
            accepted = len(sequences) - before
            stats["per_target"][name] = {
                "quota": args.n_per_cluster,
                "accepted": accepted,
                "charge": target["charge"],
                "hydrophobicity": target["hydrophobicity"],
                "rejected": cluster_rejected,
            }
            LOG.info(
                "CLUSTER %s accepted=%s rejected=%s",
                name,
                accepted,
                cluster_rejected,
            )
            if accepted != args.n_per_cluster:
                raise RuntimeError(f"{name} accepted {accepted}, expected {args.n_per_cluster}")

        stats["rejected"] = rejected
        _done("generation", started, accepted=len(sequences), rejected=rejected)
        del wrapped
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        started = _stage("mic")
        from amp_competition.predictors.inference import AMPRanker

        ranker = AMPRanker(models_dir=REPO_ROOT / "models", batch_size=32)
        mic = np.asarray(ranker.predict(sequences), dtype=np.float64)
        _done(
            "mic",
            started,
            n=len(mic),
            min=f"{float(np.min(mic)):.4f}",
            max=f"{float(np.max(mic)):.4f}",
        )
        del ranker
        gc.collect()
        torch.cuda.empty_cache()

        started = _stage("hc50")
        hc50 = _hc50_micromolar(sequences)
        _done(
            "hc50",
            started,
            n=len(hc50),
            min=f"{float(np.min(hc50)):.4f}",
            max=f"{float(np.max(hc50)):.4f}",
            model="esm2_35M_ft",
        )

        started = _stage("combined_score")
        combined = _combined_scores(mic, hc50)
        _done(
            "combined_score",
            started,
            n=len(combined),
            min=f"{float(np.min(combined)):.4f}",
            max=f"{float(np.max(combined)):.4f}",
        )

        started = _stage("rank")
        records = [
            _Seq(seq=sequence, id=f"amp{index}", score=float(combined[index - 1]))
            for index, sequence in enumerate(sequences, start=1)
        ]
        records.sort(key=lambda record: (-record.score, record.seq))
        library_records = records[: args.library_size]
        library = [record.seq for record in library_records]
        _done("rank", started, library=len(library))

        started = _stage("top_similarity")
        from Bio import SeqIO

        reference_records = list(SeqIO.parse(reference, "fasta"))
        top_records, top_stats = select_top_novel(
            library_records,
            reference_records,
            top_k=args.top_k,
            max_similarity=args.max_similarity,
        )
        stats["top_similarity"] = top_stats
        _done("top_similarity", started, **top_stats)
        if len(top_records) < args.top_k:
            raise RuntimeError(
                f"Reference similarity left {len(top_records)} peptides in the top list, "
                f"need {args.top_k}. Submission files were not written."
            )

        started = _stage("write")
        top = [record.seq for record in top_records]
        library_path = out_dir / "library.fasta"
        top_path = out_dir / "top.fasta"
        write_fasta(library, library_path, prefix="amp")
        write_fasta(top, top_path, prefix="top")
        write_stats(stats, out_dir / "library.stats.json")
        _done("write", started, library=len(library), top=len(top), library_path=library_path, top_path=top_path)
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
