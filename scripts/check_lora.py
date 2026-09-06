"""Verify LoRA adapters train while ProtGPT3 base weights stay frozen.

Usage:
    python scripts/check_lora.py
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, finish_run, load_config, save_run, seed_everything
from amp_competition.generator.lora import attach_lora, train_adapter_steps
from amp_competition.generator.protgpt3 import load_from_config

FALLBACK_PEPTIDES = [
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "GLFDIVKKVVGALGSL",
    "RGGRLCYCRRRFCVCVGR",
    "ILPWKWPWWPWRR",
    "ACDEFGHIKLMNPQRS",
    "MAGAININPEPTIDEK",
    "RRWWRRWRR",
]


def _ok(passed: bool, label: str, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")
    return passed


def _first_peptides(path: Path, n: int) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current).upper())
                current = []
                if len(sequences) >= n:
                    return sequences
            continue
        current.append(line)
    if current and len(sequences) < n:
        sequences.append("".join(current).upper())
    return sequences[:n]


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="train_lora.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    train = config.get("train", {})
    lora_cfg = config.get("lora", {})

    parser = argparse.ArgumentParser(description="Check LoRA training with a frozen ProtGPT3 base.")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--steps", type=int, default=int(train.get("steps", 8)))
    parser.add_argument("--batch-size", type=int, default=int(train.get("batch_size", 8)))
    parser.add_argument("--lr", type=float, default=float(train.get("lr", 1e-4)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "checkpoint" / "lora_check",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for LoRA training check", file=sys.stderr)
        return 1

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("train", {})
    resolved["train"].update({"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr})

    args.save_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("train_lora", resolved, out_dir=args.save_dir)

    library = REPO_ROOT / "generate" / "library.fasta"
    peptides = _first_peptides(library, args.batch_size) if library.is_file() else []
    if len(peptides) < args.batch_size:
        peptides = FALLBACK_PEPTIDES[: args.batch_size]
    print(
        f"peptides={len(peptides)} steps={args.steps} lr={args.lr} "
        f"seed={args.seed} run_id={run['run_id']}"
    )

    status = "failed"
    report: dict = {}
    try:
        tokenizer, model = load_from_config(resolved, for_training=True)
        model.to("cuda")
        dropout = train.get("dropout", lora_cfg.get("lora_dropout", 0.0))
        model = attach_lora(model, lora_cfg, dropout=float(dropout))

        report = train_adapter_steps(
            model,
            tokenizer,
            peptides,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
        )

        ok = True
        ok &= _ok(report["n_lora_tensors"] > 0, "LoRA tensors attached", f"n={report['n_lora_tensors']}")
        ok &= _ok(
            not report["leaked_base"],
            "only LoRA parameters require_grad",
            f"leaked={report['leaked_base'][:4]}",
        )
        ok &= _ok(
            report["trainable_params"] > 0 and report["trainable_pct"] < 5,
            "trainable fraction is adapter-sized",
            f"{report['trainable_params']:,} / {report['total_params']:,} ({report['trainable_pct']}%)",
        )
        ok &= _ok(not report["base_grad_seen"], "frozen base received no gradients")
        ok &= _ok(report["lora_grad_max"] > 0, "LoRA gradients are non-zero", f"max={report['lora_grad_max']:.4g}")
        ok &= _ok(
            report["base_weight_max_delta"] == 0.0,
            "base weights unchanged after AdamW",
            f"max_delta={report['base_weight_max_delta']:.4g} probes={report['base_probe'][:2]}",
        )
        ok &= _ok(
            report["lora_weight_max_delta"] > 0,
            "LoRA weights updated",
            f"max_delta={report['lora_weight_max_delta']:.4g}",
        )
        ok &= _ok(
            report["loss_end"] < report["loss_start"],
            "loss decreased on the tiny overfit batch",
            f"{report['loss_start']:.4f} → {report['loss_end']:.4f}",
        )

        model.save_pretrained(args.save_dir)
        saved = sorted(path.name for path in args.save_dir.iterdir() if path.is_file())
        ok &= _ok(
            any(name.startswith("adapter_model") for name in saved)
            and any("adapter_config" in name for name in saved),
            "adapters saved",
            f"{saved}",
        )
        (args.save_dir / "check_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        status = "completed" if ok else "failed"
        finish_run(run, extra={"peptides": peptides, "report": report, "ok": ok}, status=status)
        print("\nOVERALL:", "OK" if ok else "FAILED")
        return 0 if ok else 1
    except Exception:
        finish_run(run, extra={"report": report}, status=status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
