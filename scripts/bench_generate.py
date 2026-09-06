"""Benchmark base ProtGPT3 sampling: speed, GPU memory, reproducibility.

Usage:
    python scripts/bench_generate.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import (
    DEFAULT_SEED,
    REPO_ROOT,
    finish_run,
    load_config,
    save_run,
    seed_everything,
)
from amp_competition.generator.protgpt3 import load_from_config
from amp_competition.generator.sample import generate_library


def _nvidia_smi() -> dict[str, float] | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    used, total = raw.splitlines()[0].split(",")
    return {"smi_used_mib": float(used), "smi_total_mib": float(total)}


def _cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def _run_library(tokenizer, model, generation: dict, seed: int, n: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    sequences, stats = generate_library(
        tokenizer,
        model,
        n,
        min_length=int(generation.get("min_length", 8)),
        max_length=int(generation.get("max_length", 50)),
        batch_size=int(generation.get("batch_size", 256)),
        temperature=float(generation.get("temperature", 0.8)),
        top_p=float(generation.get("top_p", 0.9)),
        seed=seed,
        prompt=str(generation.get("prompt", "1")),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = sum(len(seq) for seq in sequences)
    return {
        "seed": seed,
        "n": len(sequences),
        "elapsed_sec": round(elapsed, 3),
        "seq_per_sec": round(len(sequences) / elapsed, 3),
        "aa_per_sec": round(tokens / elapsed, 2),
        "mean_length": round(tokens / max(len(sequences), 1), 2),
        "memory": _cuda_mem(),
        "nvidia_smi": _nvidia_smi(),
        "stats": {k: stats[k] for k in ("drawn", "accepted", "duplicates", "rejected") if k in stats},
        "sequences": sequences,
    }


def _match_report(a: list[str], b: list[str]) -> dict:
    n = min(len(a), len(b))
    positional = sum(x == y for x, y in zip(a, b, strict=False))
    set_a, set_b = set(a), set(b)
    return {
        "n": n,
        "positional_equal": positional,
        "positional_frac": round(positional / max(n, 1), 4),
        "set_jaccard": round(len(set_a & set_b) / max(len(set_a | set_b), 1), 4),
        "identical_lists": a == b,
    }


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="bench_generate.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    generation = config.get("generation", {})

    parser = argparse.ArgumentParser(description="Benchmark base ProtGPT3 generation.")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--n-sequences", type=int, default=int(generation.get("n_sequences", 512)))
    parser.add_argument("--batch-size", type=int, default=int(generation.get("batch_size", 256)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    parser.add_argument("--alt-seed", type=int, default=43)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "bench_generate")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("generation", {})
    resolved["generation"]["n_sequences"] = args.n_sequences
    resolved["generation"]["batch_size"] = args.batch_size

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("bench_generate", resolved, out_dir=args.out_dir)

    print(
        f"device={torch.cuda.get_device_name(0)} n={args.n_sequences} "
        f"batch={args.batch_size} seed={args.seed} run_id={run['run_id']}",
        flush=True,
    )

    status = "failed"
    try:
        load_t0 = time.perf_counter()
        tokenizer, model = load_from_config(resolved)
        torch.cuda.synchronize()
        load_sec = round(time.perf_counter() - load_t0, 3)
        mem_after_load = {**_cuda_mem(), **(_nvidia_smi() or {})}
        print(f"load_sec={load_sec} mem_after_load={mem_after_load}", flush=True)

        print("warmup n=64", flush=True)
        _run_library(tokenizer, model, resolved["generation"], args.seed, 64)

        print(f"run A seed={args.seed}", flush=True)
        run_a = _run_library(tokenizer, model, resolved["generation"], args.seed, args.n_sequences)
        print(f"run B seed={args.seed}", flush=True)
        run_b = _run_library(tokenizer, model, resolved["generation"], args.seed, args.n_sequences)
        print(f"run C seed={args.alt_seed}", flush=True)
        run_c = _run_library(tokenizer, model, resolved["generation"], args.alt_seed, args.n_sequences)

        same = _match_report(run_a["sequences"], run_b["sequences"])
        different = _match_report(run_a["sequences"], run_c["sequences"])
        report = {
            "gpu": torch.cuda.get_device_name(0),
            "load_sec": load_sec,
            "mem_after_load": mem_after_load,
            "run_a": {k: v for k, v in run_a.items() if k != "sequences"},
            "run_b": {k: v for k, v in run_b.items() if k != "sequences"},
            "run_c": {k: v for k, v in run_c.items() if k != "sequences"},
            "reproducibility_same_seed": same,
            "reproducibility_alt_seed": different,
            "preview_a": run_a["sequences"][:3],
            "preview_b": run_b["sequences"][:3],
            "preview_c": run_c["sequences"][:3],
        }
        out_json = args.out_dir / "bench.json"
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        status = "completed"
        finish_run(run, extra={"report": report}, status=status)
        return 0
    except Exception:
        finish_run(run, status=status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
