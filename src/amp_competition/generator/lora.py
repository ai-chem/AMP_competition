"""Attach LoRA adapters to ProtGPT3 and run short adapter-only updates."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from amp_competition.generator.protgpt3 import DIRECTION_N2C, encode_prompt

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def build_lora_config(lora_cfg: dict[str, Any] | None = None, *, dropout: float | None = None) -> LoraConfig:
    cfg = lora_cfg or {}
    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05) if dropout is None else dropout),
        target_modules=list(cfg.get("target_modules", DEFAULT_TARGET_MODULES)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )


def attach_lora(model, lora_cfg: dict[str, Any] | None = None, *, dropout: float | None = None) -> PeftModel:
    """Wrap ``model`` with LoRA. Base weights are frozen by PEFT."""
    if isinstance(model, PeftModel):
        return model
    peft_model = get_peft_model(model, build_lora_config(lora_cfg, dropout=dropout))
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    peft_model.config.use_cache = False
    peft_model.train()
    return peft_model


def parameter_report(model) -> dict[str, Any]:
    trainable = 0
    frozen = 0
    lora_tensors = 0
    trainable_names: list[str] = []
    leaked_base: list[str] = []
    for name, param in model.named_parameters():
        n = param.numel()
        is_lora = "lora_" in name
        if param.requires_grad:
            trainable += n
            trainable_names.append(name)
            if not is_lora:
                leaked_base.append(name)
        else:
            frozen += n
        if is_lora:
            lora_tensors += 1
    total = trainable + frozen
    return {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        "n_trainable_tensors": len(trainable_names),
        "n_lora_tensors": lora_tensors,
        "leaked_base": leaked_base,
    }


def assert_base_frozen(model) -> dict[str, Any]:
    report = parameter_report(model)
    if report["n_lora_tensors"] == 0:
        raise RuntimeError("No LoRA tensors found; target_modules probably did not match")
    if report["leaked_base"]:
        raise RuntimeError(f"Base weights are trainable: {report['leaked_base'][:8]}")
    if report["trainable_params"] == 0:
        raise RuntimeError("No trainable LoRA parameters")
    return report


def encode_train_batch(tokenizer, sequences: list[str], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Causal LM batch: <|bos|> 1 SEQUENCE <|eos|>, padded on the right."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or eos_id is None:
        raise RuntimeError("Tokenizer missing pad/eos ids")
    rows: list[list[int]] = []
    for sequence in sequences:
        ids = encode_prompt(tokenizer, DIRECTION_N2C + sequence).tolist()[0]
        rows.append(ids + [int(eos_id)])
    max_len = max(len(row) for row in rows)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    labels: list[list[int]] = []
    for row in rows:
        pad_n = max_len - len(row)
        input_ids.append(row + [int(pad_id)] * pad_n)
        attention.append([1] * len(row) + [0] * pad_n)
        labels.append(row + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, device=device, dtype=torch.long),
        "attention_mask": torch.tensor(attention, device=device, dtype=torch.long),
        "labels": torch.tensor(labels, device=device, dtype=torch.long),
    }


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max().item())


def _snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _snapshot_base_probe(model, n_tensors: int = 4) -> dict[str, torch.Tensor]:
    probes: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad or "lora_" in name:
            continue
        if param.ndim < 2:
            continue
        probes[name] = param.detach().clone()
        if len(probes) >= n_tensors:
            break
    if not probes:
        raise RuntimeError("Could not find a frozen base weight to probe")
    return probes


def train_adapter_steps(
    model,
    tokenizer,
    sequences: list[str],
    *,
    steps: int = 8,
    lr: float = 1e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run a few AdamW steps on LoRA only; return freeze / update diagnostics."""
    device = next(model.parameters()).device
    model.train()
    report = assert_base_frozen(model)
    base_before = _snapshot_base_probe(model)
    lora_before = _snapshot_trainable(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    losses: list[float] = []
    batch = sequences[:batch_size] or sequences
    inputs = encode_train_batch(tokenizer, batch, device)

    lora_grad_max = 0.0
    base_grad_seen = False
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Causal LM forward did not return a loss")
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if param.requires_grad:
                lora_grad_max = max(lora_grad_max, float(param.grad.detach().abs().max().item()))
            else:
                base_grad_seen = True
        optimizer.step()
        losses.append(float(loss.detach().item()))

    named = dict(model.named_parameters())
    base_delta = max(_max_abs_diff(named[name], tensor) for name, tensor in base_before.items())
    lora_delta = max(_max_abs_diff(named[name], tensor) for name, tensor in lora_before.items())
    report.update(
        {
            "steps": steps,
            "lr": lr,
            "batch_size": len(batch),
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "losses": losses,
            "lora_grad_max": lora_grad_max,
            "base_grad_seen": base_grad_seen,
            "base_weight_max_delta": base_delta,
            "lora_weight_max_delta": lora_delta,
            "base_probe": list(base_before),
        }
    )
    return report
