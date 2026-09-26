"""Controllability check: generate under several (Q, H) targets and recompute descriptors.

Usage:
    PYTHONPATH=src python3 scripts/eval_controllability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, load_config, seed_everything
from amp_competition.data.conditions import CHARGE_KEY, HYDROPHOBICITY_KEY, conditions_for_sequence
from amp_competition.generator.conditioning import (
    decode_generated,
    default_suppress,
    load_v2_bundle,
)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def sample_target(wrapped, tokenizer, suppress, charge: float, hydrophobicity: float, n: int, batch_size: int):
    remaining = n
    rows: list[dict] = []
    while remaining > 0:
        take = min(batch_size, remaining)
        texts = wrapped.generate(
            tokenizer,
            charge,
            hydrophobicity,
            num_return_sequences=take,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            suppress_tokens=suppress,
        )
        for item in decode_generated(texts):
            if not (item["valid_alphabet"] and item["direction"] == "N2C" and 8 <= item["length"] <= 50):
                continue
            q, h = conditions_for_sequence(item["sequence"])
            item[CHARGE_KEY] = q
            item[HYDROPHOBICITY_KEY] = h
            rows.append(item)
        remaining -= take
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that V2 targets move realized descriptors.")
    parser.add_argument("--config", default="train_cond.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/lora_cond_v2/best"))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("checkpoint/lora_cond_v2/controllability.json"))
    parser.add_argument("--report", type=Path, default=Path("checkpoint/lora_cond_v2/controllability.md"))
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("checkpoint/lora_cond_v2/controllability_sequences.csv"),
    )
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    if not (checkpoint / "adapter_config.json").is_file() or not (checkpoint / "conditioner.pt").is_file():
        print(f"missing V2 best checkpoint at {checkpoint}", file=sys.stderr)
        return 1
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    config = load_config(args.config)
    seed_everything(args.seed, deterministic=bool(config.get("deterministic", True)))
    tokenizer, wrapped = load_v2_bundle(config, checkpoint, device="cuda", trainable=False)
    suppress = default_suppress(tokenizer)
    normalizer = wrapped.normalizer

    targets = [
        {"name": "lowQ_lowH", "charge": normalizer.p10[0], "hydrophobicity": normalizer.p10[1]},
        {"name": "lowQ_highH", "charge": normalizer.p10[0], "hydrophobicity": normalizer.p90[1]},
        {"name": "meanQ_meanH", "charge": normalizer.p50[0], "hydrophobicity": normalizer.p50[1]},
        {"name": "highQ_lowH", "charge": normalizer.p90[0], "hydrophobicity": normalizer.p10[1]},
        {"name": "highQ_highH", "charge": normalizer.p90[0], "hydrophobicity": normalizer.p90[1]},
    ]

    results = []
    sequence_rows: list[dict[str, object]] = []
    pooled_target_q: list[float] = []
    pooled_real_q: list[float] = []
    pooled_target_h: list[float] = []
    pooled_real_h: list[float] = []
    for target in tqdm(targets, desc="targets"):
        rows = sample_target(
            wrapped,
            tokenizer,
            suppress,
            target["charge"],
            target["hydrophobicity"],
            args.n,
            args.batch_size,
        )
        for index, row in enumerate(rows, start=1):
            sequence_rows.append(
                {
                    "id": f"{target['name']}_{index:03d}",
                    "target": target["name"],
                    "sequence": row["sequence"],
                    "length": row["length"],
                    "target_charge": round(target["charge"], 6),
                    "target_hydrophobicity": round(target["hydrophobicity"], 6),
                    "charge_pH7_4": round(row[CHARGE_KEY], 6),
                    "hydrophobicity_interfaceScale_pH8": round(row[HYDROPHOBICITY_KEY], 6),
                }
            )
        charges = [row[CHARGE_KEY] for row in rows]
        hydros = [row[HYDROPHOBICITY_KEY] for row in rows]
        summary = {
            "name": target["name"],
            "target_charge": target["charge"],
            "target_hydrophobicity": target["hydrophobicity"],
            "n_valid": len(rows),
            "mean_charge": _mean(charges) if charges else None,
            "mean_hydrophobicity": _mean(hydros) if hydros else None,
            "delta_charge": (_mean(charges) - target["charge"]) if charges else None,
            "delta_hydrophobicity": (_mean(hydros) - target["hydrophobicity"]) if hydros else None,
            "length_mean": _mean([row["length"] for row in rows]) if rows else None,
            "examples": [row["sequence"] for row in rows[:8]],
        }
        results.append(summary)
        for charge, hydro in zip(charges, hydros):
            pooled_target_q.append(target["charge"])
            pooled_real_q.append(charge)
            pooled_target_h.append(target["hydrophobicity"])
            pooled_real_h.append(hydro)
        print(
            f"{target['name']}: n={len(rows)} Q {target['charge']:.3f}→{summary['mean_charge']} "
            f"H {target['hydrophobicity']:.3f}→{summary['mean_hydrophobicity']}",
            flush=True,
        )

    low_q = next(item for item in results if item["name"] == "lowQ_lowH")
    high_q = next(item for item in results if item["name"] == "highQ_highH")
    low_h = next(item for item in results if item["name"] == "highQ_lowH")
    high_h = next(item for item in results if item["name"] == "lowQ_highH")
    charge_moves = (
        high_q["mean_charge"] is not None
        and low_q["mean_charge"] is not None
        and high_q["mean_charge"] > low_q["mean_charge"]
    )
    hydro_moves = (
        high_h["mean_hydrophobicity"] is not None
        and low_h["mean_hydrophobicity"] is not None
        and high_h["mean_hydrophobicity"] > low_h["mean_hydrophobicity"]
    )

    payload = {
        "checkpoint": str(checkpoint),
        "normalizer": normalizer.to_dict(),
        "n_per_target": args.n,
        "pearson_charge": _pearson(pooled_target_q, pooled_real_q),
        "pearson_hydrophobicity": _pearson(pooled_target_h, pooled_real_h),
        "charge_high_gt_low": charge_moves,
        "hydrophobicity_high_gt_low": hydro_moves,
        "targets": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq_path = args.sequences if args.sequences.is_absolute() else REPO_ROOT / args.sequences
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    with seq_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "target",
                "sequence",
                "length",
                "target_charge",
                "target_hydrophobicity",
                "charge_pH7_4",
                "hydrophobicity_interfaceScale_pH8",
            ],
        )
        writer.writeheader()
        writer.writerows(sequence_rows)
    print(f"wrote {seq_path} n={len(sequence_rows)}")
    if args.skip_report:
        return 0 if charge_moves else 1
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V2 controllability",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Samples per target: {args.n}",
        "",
        f"- Pearson(target Q, realized Q): **{payload['pearson_charge']}**",
        f"- Pearson(target H, realized H): **{payload['pearson_hydrophobicity']}**",
        f"- High-Q mean > low-Q mean: **{charge_moves}**",
        f"- High-H mean > low-H mean: **{hydro_moves}**",
        "",
        "| target | Q* | Q mean | ΔQ | H* | H mean | ΔH | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['target_charge']:.3f} | {item['mean_charge']:.3f} | "
            f"{item['delta_charge']:.3f} | {item['target_hydrophobicity']:.3f} | "
            f"{item['mean_hydrophobicity']:.3f} | {item['delta_hydrophobicity']:.3f} | {item['n_valid']} |"
        )
    lines.append("")
    lines.append(
        "Targets are train-set 10/50/90 percentiles of `charge_pH7_4` and "
        "`hydrophobicity_interfaceScale_pH8` from `descriptors.py`. "
        "InterfaceScale is Wimley–White ΔG: more negative is more interfacial; "
        "the high-H target is therefore the 90th percentile of that signed scale."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    return 0 if charge_moves else 1


if __name__ == "__main__":
    raise SystemExit(main())
