"""Generate peptides from V2 given target charge and interface hydrophobicity.

Usage:
    PYTHONPATH=src python3 scripts/generate_cond.py --charge 4.0 --hydrophobicity -0.2 --n 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, load_config, seed_everything
from amp_competition.data.conditions import conditions_for_sequence
from amp_competition.generator.conditioning import decode_generated, default_suppress, load_v2_bundle
from amp_competition.io import write_fasta


def main() -> int:
    parser = argparse.ArgumentParser(description="Conditional ProtGPT3 V2 generation.")
    parser.add_argument("--config", default="train_cond.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/lora_cond_v2/best"))
    parser.add_argument("--charge", type=float, required=True)
    parser.add_argument("--hydrophobicity", type=float, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    if not (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "conditioner.pt").is_file():
        print(f"missing V2 best checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    config = load_config(args.config)
    seed_everything(args.seed, deterministic=bool(config.get("deterministic", True)))
    tokenizer, wrapped = load_v2_bundle(config, checkpoint, device="cuda", trainable=False)
    suppress = default_suppress(tokenizer)

    remaining = args.n
    rows: list[dict] = []
    while remaining > 0:
        take = min(args.batch_size, remaining)
        texts = wrapped.generate(
            tokenizer,
            args.charge,
            args.hydrophobicity,
            num_return_sequences=take,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            suppress_tokens=suppress,
        )
        decoded = decode_generated(texts)
        for item in decoded:
            if item["valid_alphabet"] and item["direction"] == "N2C" and item["sequence"]:
                charge, hydrophobicity = conditions_for_sequence(item["sequence"])
                item["charge_pH7_4"] = charge
                item["hydrophobicity_interfaceScale_pH8"] = hydrophobicity
                rows.append(item)
        remaining -= take

    sequences = [row["sequence"] for row in rows]
    summary = {
        "target_charge": args.charge,
        "target_hydrophobicity": args.hydrophobicity,
        "n": len(rows),
        "mean_charge": round(sum(row["charge_pH7_4"] for row in rows) / max(len(rows), 1), 4),
        "mean_hydrophobicity": round(
            sum(row["hydrophobicity_interfaceScale_pH8"] for row in rows) / max(len(rows), 1), 4
        ),
        "length_mean": round(sum(row["length"] for row in rows) / max(len(rows), 1), 2),
        "sequences": sequences,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "sequences"}, indent=2))
    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        if out.suffix.lower() in {".fa", ".fasta"}:
            write_fasta(sequences, out)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
