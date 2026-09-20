"""Train V2: AMP-LoRA V1 + soft-prompt MLP on charge and interface hydrophobicity.

Usage:
    PYTHONPATH=src python3 scripts/train_cond_lora.py --config train_cond.yaml
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
from amp_competition.data.conditions import (
    CONDITION_KEYS,
    ConditionedPeptide,
    build_conditioned_split,
)
from amp_competition.generator.conditioning import (
    ConditionMLP,
    ConditionNormalizer,
    ConditionalProtGPT3,
    encode_conditional_batch,
    load_v1_peft,
)
from amp_competition.generator.lora import assert_base_frozen


class ConditionedDataset(Dataset):
    def __init__(self, rows: list[ConditionedPeptide]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ConditionedPeptide:
        return self.rows[index]


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _require_finite(loss: torch.Tensor | None, where: str) -> torch.Tensor:
    if loss is None:
        raise RuntimeError(f"{where} did not return a loss")
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"non-finite loss at {where}: {float(loss.detach().float().mean().cpu())}")
    return loss


def _encode_rows(
    wrapped: ConditionalProtGPT3,
    tokenizer,
    rows: list[ConditionedPeptide],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return _move(
        encode_conditional_batch(
            tokenizer,
            [row.sequence for row in rows],
            [row.vector() for row in rows],
            "cpu",
            n_tokens=wrapped.conditioner.n_tokens,
        ),
        device,
    )


@torch.no_grad()
def evaluate(wrapped: ConditionalProtGPT3, tokenizer, loader: DataLoader, device: torch.device) -> float:
    wrapped.eval()
    total = 0.0
    count = 0
    for rows in loader:
        batch = _encode_rows(wrapped, tokenizer, list(rows), device)
        loss = _require_finite(wrapped(**batch).loss, "validation")
        n = batch["input_ids"].size(0)
        total += float(loss.item()) * n
        count += n
    wrapped.train()
    return total / max(count, 1)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="train_cond.yaml")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    cond_cfg = config.get("condition", {})
    v1_cfg = config.get("v1", {})
    generation = config.get("generation", {})

    parser = argparse.ArgumentParser(description="Train conditional ProtGPT3 V2 from AMP-LoRA V1.")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--csv", type=Path, default=Path(data_cfg.get("csv", "data/raw/antibacterial_clean.csv")))
    parser.add_argument("--split", type=Path, default=Path(data_cfg.get("split", "checkpoint/lora_antibacterial/split.json")))
    parser.add_argument("--v1-adapter", type=Path, default=Path(v1_cfg.get("adapter", "checkpoint/lora_antibacterial/best")))
    parser.add_argument("--epochs", type=int, default=int(train_cfg.get("epochs", 3)))
    parser.add_argument("--batch-size", type=int, default=int(train_cfg.get("batch_size", 64)))
    parser.add_argument("--lr", type=float, default=float(train_cfg.get("lr", 1e-4)))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", DEFAULT_SEED)))
    parser.add_argument("--save-dir", type=Path, default=Path(train_cfg.get("save_dir", "checkpoint/lora_cond_v2")))
    parser.add_argument("--n-tokens", type=int, default=int(cond_cfg.get("n_tokens", 4)))
    parser.add_argument("--mlp-hidden", type=int, default=int(cond_cfg.get("mlp_hidden", 128)))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for V2 training", file=sys.stderr)
        return 1

    csv_path = args.csv if args.csv.is_absolute() else REPO_ROOT / args.csv
    split_path = args.split if args.split.is_absolute() else REPO_ROOT / args.split
    adapter_dir = args.v1_adapter if args.v1_adapter.is_absolute() else REPO_ROOT / args.v1_adapter
    save_dir = args.save_dir if args.save_dir.is_absolute() else REPO_ROOT / args.save_dir
    best_dir = save_dir / "best"
    last_dir = save_dir / "last"
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved = deepcopy(config)
    resolved["seed"] = args.seed
    resolved.setdefault("data", {})
    resolved["data"].update({"csv": str(csv_path), "split": str(split_path)})
    resolved.setdefault("v1", {})
    resolved["v1"]["adapter"] = str(adapter_dir)
    resolved.setdefault("condition", {})
    resolved["condition"].update({"n_tokens": args.n_tokens, "mlp_hidden": args.mlp_hidden})
    resolved.setdefault("train", {})
    resolved["train"].update(
        {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "save_dir": str(save_dir)}
    )

    seed_everything(args.seed, deterministic=bool(resolved.get("deterministic", True)))
    run = save_run("train_cond_lora", resolved, out_dir=save_dir)

    print("annotating sequences with descriptors.py …", flush=True)
    train_rows, val_rows, data_stats = build_conditioned_split(
        csv_path,
        split_path=split_path if split_path.is_file() else None,
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=args.seed,
        min_length=int(generation.get("min_length", MIN_LENGTH)),
        max_length=int(generation.get("max_length", MAX_LENGTH)),
    )
    normalizer = ConditionNormalizer.fit([row.vector() for row in train_rows], CONDITION_KEYS)
    normalizer.save(save_dir / "normalizer.json")
    print(
        f"train={len(train_rows)} val={len(val_rows)} "
        f"Q mean={normalizer.mean[0]:.3f} H mean={normalizer.mean[1]:.3f} "
        f"v1={adapter_dir}",
        flush=True,
    )

    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    patience = int(train_cfg.get("patience", 2))

    status = "failed"
    history: list[dict] = []
    try:
        tokenizer, peft_model = load_v1_peft(resolved, adapter_dir, trainable=True)
        device = torch.device("cuda")
        peft_model.to(device)
        report = assert_base_frozen(peft_model)
        embed_dim = int(getattr(peft_model.config, "hidden_size", 1024))
        conditioner = ConditionMLP(
            hidden=args.mlp_hidden,
            n_tokens=args.n_tokens,
            embed_dim=embed_dim,
        ).to(device)
        wrapped = ConditionalProtGPT3(peft_model, conditioner, normalizer).to(device)
        cond_params = sum(param.numel() for param in conditioner.parameters())
        print(
            f"lora={report['trainable_params']:,} mlp={cond_params:,} "
            f"embed_dim={embed_dim} n_tokens={args.n_tokens} device={torch.cuda.get_device_name(0)}",
            flush=True,
        )

        train_loader = DataLoader(
            ConditionedDataset(train_rows),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        val_loader = DataLoader(
            ConditionedDataset(val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

        optimizer = torch.optim.AdamW(
            (param for param in wrapped.parameters() if param.requires_grad),
            lr=args.lr,
            weight_decay=weight_decay,
        )
        total_steps = max(1, args.epochs * len(train_loader))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * warmup_ratio)),
            num_training_steps=total_steps,
        )

        print("smoke: 1-batch finite-loss check …", flush=True)
        wrapped.train()
        smoke = _encode_rows(wrapped, tokenizer, train_rows[: args.batch_size], device)
        smoke_loss = _require_finite(wrapped(**smoke).loss, "smoke forward")
        smoke_loss.backward()
        for name, param in wrapped.named_parameters():
            if param.requires_grad and param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"non-finite grad at smoke: {name}")
        optimizer.zero_grad(set_to_none=True)
        print(f"smoke_loss={float(smoke_loss.item()):.4f}", flush=True)

        best_val = math.inf
        bad_epochs = 0
        started = time.monotonic()
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            wrapped.train()
            running = 0.0
            seen = 0
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
            for rows in progress:
                batch = _encode_rows(wrapped, tokenizer, list(rows), device)
                optimizer.zero_grad(set_to_none=True)
                loss = _require_finite(wrapped(**batch).loss, f"train step {global_step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (param for param in wrapped.parameters() if param.requires_grad),
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
            if not math.isfinite(train_loss):
                raise RuntimeError(f"non-finite epoch train_loss={train_loss}")
            val_loss = evaluate(wrapped, tokenizer, val_loader, device)
            if not math.isfinite(val_loss):
                raise RuntimeError(f"non-finite epoch val_loss={val_loss}")
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
            wrapped.save_bundle(last_dir)
            if record["best"]:
                best_val = val_loss
                bad_epochs = 0
                wrapped.save_bundle(best_dir)
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
                print(f"saved best V2 → {best_dir} val_loss={val_loss:.4f}", flush=True)
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
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "data_stats": data_stats,
            "lora_params": report["trainable_params"],
            "mlp_params": cond_params,
            "history": history,
            "normalizer": normalizer.to_dict(),
            "best_dir": str(best_dir),
            "last_dir": str(last_dir),
        }
        (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        has_best = (best_dir / "adapter_config.json").is_file() and math.isfinite(best_val)
        status = "completed" if has_best else "failed"
        finish_run(run, extra=metrics, status=status)
        print(f"done status={status} best_val_loss={metrics['best_val_loss']} elapsed_sec={elapsed}", flush=True)
        return 0 if status == "completed" else 1
    except Exception:
        finish_run(run, extra={"history": history, "data_stats": data_stats}, status=status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
