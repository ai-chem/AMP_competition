"""Evaluate LoRA (vs base) generation: EOS stopping, length hist, train copying.

Usage:
    PYTHONPATH=src python3 scripts/eval_lora_generate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, load_config, seed_everything
from amp_competition.constants import MAX_LENGTH, MIN_LENGTH
from amp_competition.generator.protgpt3 import (
    DIRECTION_N2C,
    challenge_suppress_token_ids,
    decode_protein,
    encode_prompt,
    is_challenge_alphabet,
    load_from_config,
)


def sample_batch(
    tokenizer,
    model,
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    suppress_tokens: list[int],
) -> list[dict]:
    device = next(model.parameters()).device
    input_ids = encode_prompt(tokenizer, DIRECTION_N2C, device=device)
    prompt_len = int(input_ids.shape[-1])
    eos_id = int(tokenizer.eos_token_id)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=n,
        suppress_tokens=suppress_tokens,
    )
    rows: list[dict] = []
    for row in output_ids:
        new_ids = row[prompt_len:].tolist()
        hit_eos = eos_id in new_ids
        if hit_eos:
            new_ids = new_ids[: new_ids.index(eos_id)]
        text = tokenizer.decode(row, skip_special_tokens=True)
        parsed = decode_protein(text)
        sequence = parsed.sequence
        rows.append(
            {
                "sequence": sequence,
                "direction": parsed.direction,
                "length": len(sequence),
                "hit_eos": hit_eos,
                "n_new_tokens": len(new_ids),
                "valid_alphabet": is_challenge_alphabet(sequence) if sequence else False,
            }
        )
    return rows


def summarize(rows: list[dict], train: set[str], val: set[str], max_new_tokens: int) -> dict:
    lengths = [row["length"] for row in rows]
    sequences = [row["sequence"] for row in rows]
    unique = set(sequences)
    in_train = [seq for seq in sequences if seq in train]
    in_val = [seq for seq in sequences if seq in val]
    in_8_50 = [
        row
        for row in rows
        if MIN_LENGTH <= row["length"] <= MAX_LENGTH and row["valid_alphabet"] and row["direction"] == "N2C"
    ]
    return {
        "n": len(rows),
        "hit_eos": sum(row["hit_eos"] for row in rows),
        "hit_eos_pct": round(100.0 * sum(row["hit_eos"] for row in rows) / max(len(rows), 1), 2),
        "capped_no_eos": sum(not row["hit_eos"] and row["n_new_tokens"] >= max_new_tokens for row in rows),
        "valid_8_50": len(in_8_50),
        "valid_8_50_pct": round(100.0 * len(in_8_50) / max(len(rows), 1), 2),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": round(sum(lengths) / max(len(lengths), 1), 2),
        "length_hist": {str(k): int(v) for k, v in sorted(Counter(lengths).items())},
        "unique": len(unique),
        "exact_train_copies": len(in_train),
        "exact_train_copy_pct": round(100.0 * len(in_train) / max(len(rows), 1), 2),
        "exact_val_copies": len(in_val),
        "unique_train_copies": len(set(in_train)),
        "sample_train_copies": sorted(set(in_train))[:10],
    }


def generate_n(tokenizer, model, *, n: int, batch_size: int, **kwargs) -> list[dict]:
    rows: list[dict] = []
    remaining = n
    progress = tqdm(total=n, desc="sample", unit="seq")
    while remaining > 0:
        take = min(batch_size, remaining)
        batch = sample_batch(tokenizer, model, n=take, **kwargs)
        rows.extend(batch)
        remaining -= take
        progress.update(take)
    progress.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LoRA EOS / length / train copying.")
    parser.add_argument("--config", default="train_lora.yaml")
    parser.add_argument("--adapter-dir", type=Path, default=Path("checkpoint/lora_antibacterial/best"))
    parser.add_argument("--split", type=Path, default=Path("checkpoint/lora_antibacterial/split.json"))
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--n-base", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval_lora_generate.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    adapter_dir = args.adapter_dir if args.adapter_dir.is_absolute() else REPO_ROOT / args.adapter_dir
    split_path = args.split if args.split.is_absolute() else REPO_ROOT / args.split
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = set(split["train"])
    val = set(split["val"])

    config = load_config(args.config)
    seed_everything(args.seed, deterministic=bool(config.get("deterministic", True)))
    tokenizer, model = load_from_config(config, for_training=True)
    model.to("cuda")
    model.eval()
    suppress = challenge_suppress_token_ids(tokenizer)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "suppress_tokens": suppress,
    }

    print(f"base n={args.n_base} adapter={adapter_dir} n={args.n}", flush=True)
    base_rows = generate_n(tokenizer, model, n=args.n_base, batch_size=args.batch_size, **gen_kwargs)
    base_stats = summarize(base_rows, train, val, args.max_new_tokens)
    print("BASE", json.dumps({k: v for k, v in base_stats.items() if k != "length_hist"}), flush=True)
    print("BASE hist", base_stats["length_hist"], flush=True)

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    lora_rows = generate_n(tokenizer, model, n=args.n, batch_size=args.batch_size, **gen_kwargs)
    lora_stats = summarize(lora_rows, train, val, args.max_new_tokens)
    print("LORA", json.dumps({k: v for k, v in lora_stats.items() if k != "length_hist"}), flush=True)
    print("LORA hist", lora_stats["length_hist"], flush=True)

    payload = {
        "adapter_dir": str(adapter_dir),
        "n_train": len(train),
        "n_val": len(val),
        "base": base_stats,
        "lora": lora_stats,
        "lora_examples": [row["sequence"] for row in lora_rows[:20]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
