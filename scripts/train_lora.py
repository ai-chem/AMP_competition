"""LoRA fine-tune ProtGPT3 on a peptide CSV. Saves the lowest-val-loss adapter.

Usage:
    PYTHONPATH=src python3 scripts/train_lora.py --config train_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amp_competition.config import DEFAULT_SEED, REPO_ROOT, finish_run, load_config, save_run, seed_everything
from amp_competition.constants import MAX_LENGTH, MIN_LENGTH
from amp_competition.data.peptides import load_peptide_csv, split_train_val, write_split
from amp_competition.generator.lora import assert_base_frozen, attach_lora, encode_train_batch
from amp_competition.generator.protgpt3 import load_from_config


class PeptideDataset(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        return self.sequences[index]


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, tokenizer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for sequences in loader:
        batch = _move(encode_train_batch(tokenizer, sequences, "cpu"), device)
        loss = model(**batch).loss
        if loss is None:
            raise RuntimeError("Validation forward did not return a loss")
        n = batch["input_ids"].size(0)
        total += float(loss.item()) * n
        count += n
    model.train()
    return total / max(count, 1)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="train_lora.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    lora_cfg = config.get("lora", {})
    generation = config.get("generation", {})

    parser = argparse.ArgumentParser(description="LoRA fine-tune ProtGPT3 on antibacterial peptides.")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(data_cfg.get("csv", "data/raw/antibacterial_clean.csv")),
    )
    parser.add_argument("--epochs", type=int, default=int(train_cfg.get("epochs", 5)))
    parser.add_argument("--batch-size", type=int, default=int(train_cfg.get("batch_size", 64)))
    parser.add_argument("--lr", type=float, default=float(train_cfg.get("lr", 1e-4)))
    parser.add_argument("--val-fraction", type=float, default=float(data_cfg.get("val_fraction", 0.1)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(train_cfg.get("save_dir", "checkpoint/lora_antibacterial")),
    )
    parser.add_argument("--min-length", type=int, default=int(generation.get("min_length", MIN_LENGTH)))
    parser.add_argument("--max-length", type=int, default=int(generation.get("max_length", MAX_LENGTH)))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for LoRA training", file=sys.stderr)
        return 1

    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    save_dir = args.save_dir if args.save_dir.is_absolute() else REPO_ROOT / args.save_dir
    best_dir = save_dir / "best"
    last_dir = save_dir / "last"
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("data", {})
    resolved["data"].update(
        {
            "csv": str(csv_path),
            "val_fraction": args.val_fraction,
            "min_length": args.min_length,
            "max_length": args.max_length,
        }
    )
    resolved.setdefault("train", {})
    resolved["train"].update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "save_dir": str(save_dir),
        }
    )

    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("train_lora", resolved, out_dir=save_dir)

    sequences, data_stats = load_peptide_csv(
        csv_path,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    train_seqs, val_seqs = split_train_val(sequences, val_fraction=args.val_fraction, seed=args.seed)
    write_split(
        save_dir / "split.json",
        train_seqs,
        val_seqs,
        extra={"csv": str(csv_path), "seed": args.seed, "data_stats": data_stats},
    )

    print(
        f"csv={csv_path} rows={data_stats['rows']} unique={data_stats['accepted']} "
        f"train={len(train_seqs)} val={len(val_seqs)} epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr} seed={args.seed}",
        flush=True,
    )

    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    patience = int(train_cfg.get("patience", 2))
    dropout = float(train_cfg.get("dropout", lora_cfg.get("lora_dropout", 0.05)))

    status = "failed"
    history: list[dict] = []
    try:
        tokenizer, model = load_from_config(resolved, for_training=True)
        device = torch.device("cuda")
        model.to(device)
        model = attach_lora(model, lora_cfg, dropout=dropout)
        report = assert_base_frozen(model)
        print(
            f"trainable={report['trainable_params']:,} / {report['total_params']:,} "
            f"({report['trainable_pct']}%) device={torch.cuda.get_device_name(0)}",
            flush=True,
        )

        def collate(batch: list[str]):
            return batch

        train_loader = DataLoader(
            PeptideDataset(train_seqs),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            PeptideDataset(val_seqs),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=collate,
        )

        optimizer = torch.optim.AdamW(
            (param for param in model.parameters() if param.requires_grad),
            lr=args.lr,
            weight_decay=weight_decay,
        )
        total_steps = max(1, args.epochs * len(train_loader))
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val = math.inf
        bad_epochs = 0
        started = time.monotonic()
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
            for sequences in progress:
                batch = _move(encode_train_batch(tokenizer, sequences, "cpu"), device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                if loss is None:
                    raise RuntimeError("Training forward did not return a loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad),
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                n = batch["input_ids"].size(0)
                running += float(loss.item()) * n
                seen += n
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            train_loss = running / max(seen, 1)
            val_loss = evaluate(model, tokenizer, val_loader, device)
            record = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": scheduler.get_last_lr()[0],
                "best": val_loss < best_val - 1e-6,
            }
            history.append(record)
            print(
                f"epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} best={record['best']}",
                flush=True,
            )

            model.save_pretrained(last_dir)
            if record["best"]:
                best_val = val_loss
                bad_epochs = 0
                model.save_pretrained(best_dir)
                (best_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "step": global_step,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"saved best adapter → {best_dir} val_loss={val_loss:.4f}", flush=True)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop after {epoch} epochs (patience={patience})", flush=True)
                    break

        elapsed = round(time.monotonic() - started, 2)
        metrics = {
            "best_val_loss": None if best_val is math.inf else best_val,
            "epochs_ran": len(history),
            "steps": global_step,
            "elapsed_sec": elapsed,
            "n_train": len(train_seqs),
            "n_val": len(val_seqs),
            "data_stats": data_stats,
            "trainable_params": report["trainable_params"],
            "history": history,
            "best_dir": str(best_dir),
            "last_dir": str(last_dir),
        }
        (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        status = "completed" if history else "failed"
        finish_run(run, extra=metrics, status=status)
        print(
            f"done status={status} best_val_loss={metrics['best_val_loss']} "
            f"elapsed_sec={elapsed} best={best_dir}",
            flush=True,
        )
        return 0 if status == "completed" else 1
    except Exception:
        finish_run(
            run,
            extra={"history": history, "data_stats": data_stats},
            status=status,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
