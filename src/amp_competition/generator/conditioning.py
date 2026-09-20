"""Soft-prompt conditioning: MLP(Q, H) injected after BOS+direction via embed hook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from peft import PeftModel

from amp_competition.generator.lora import encode_train_batch
from amp_competition.generator.protgpt3 import (
    DIRECTION_N2C,
    challenge_suppress_token_ids,
    decode_protein,
    encode_prompt,
    is_challenge_alphabet,
    load_from_config,
)

# <|bos|> and direction ``1`` stay first; condition tokens sit before amino acids.
CONDITION_INSERT_AT = 2


def condition_dummy_id(tokenizer) -> int:
    """In-vocab placeholder for condition slots. Embeddings are overwritten anyway.

    ProtGPT3's Mixtral table is ``vocab_size`` (31). Extra tokenizer symbols such as
    ``<unk>=33`` are out of range for ``embed_tokens`` and crash CUDA gather.
    """
    limit = int(tokenizer.vocab_size)
    for cand in (tokenizer.pad_token_id, tokenizer.bos_token_id, 0):
        if cand is not None and 0 <= int(cand) < limit:
            return int(cand)
    raise RuntimeError(f"no in-vocab dummy token (vocab_size={limit})")


@dataclass
class ConditionNormalizer:
    """Z-score two condition channels from the train set."""

    names: tuple[str, str]
    mean: tuple[float, float]
    std: tuple[float, float]
    p10: tuple[float, float]
    p50: tuple[float, float]
    p90: tuple[float, float]

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std).clamp_min(1e-6)
        return (values - mean) / std

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": list(self.mean),
            "std": list(self.std),
            "p10": list(self.p10),
            "p50": list(self.p50),
            "p90": list(self.p90),
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ConditionNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            names=tuple(payload["names"]),  # type: ignore[arg-type]
            mean=tuple(payload["mean"]),  # type: ignore[arg-type]
            std=tuple(payload["std"]),  # type: ignore[arg-type]
            p10=tuple(payload["p10"]),  # type: ignore[arg-type]
            p50=tuple(payload["p50"]),  # type: ignore[arg-type]
            p90=tuple(payload["p90"]),  # type: ignore[arg-type]
        )

    @classmethod
    def fit(cls, vectors: Sequence[tuple[float, float]], names: tuple[str, str]) -> "ConditionNormalizer":
        if not vectors:
            raise ValueError("Cannot fit a normalizer on an empty set")
        tensor = torch.tensor(vectors, dtype=torch.float64)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0, unbiased=True).clamp_min(1e-6)
        q = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64), dim=0)
        return cls(
            names=names,
            mean=(float(mean[0]), float(mean[1])),
            std=(float(std[0]), float(std[1])),
            p10=(float(q[0, 0]), float(q[0, 1])),
            p50=(float(q[1, 0]), float(q[1, 1])),
            p90=(float(q[2, 0]), float(q[2, 1])),
        )


class ConditionMLP(nn.Module):
    """Map a 2-D z-scored condition vector to ``n_tokens`` hidden-state embeddings."""

    def __init__(self, *, cond_dim: int = 2, hidden: int = 128, n_tokens: int = 4, embed_dim: int = 1024):
        super().__init__()
        self.cond_dim = cond_dim
        self.hidden = hidden
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_tokens * embed_dim),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        projected = self.net(conditions.to(dtype=self.net[0].weight.dtype))
        return projected.view(conditions.size(0), self.n_tokens, self.embed_dim)

    def config_dict(self) -> dict[str, int]:
        return {
            "cond_dim": self.cond_dim,
            "hidden": self.hidden,
            "n_tokens": self.n_tokens,
            "embed_dim": self.embed_dim,
            "insert_at": CONDITION_INSERT_AT,
        }

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "ConditionMLP":
        return cls(
            cond_dim=int(payload.get("cond_dim", 2)),
            hidden=int(payload.get("hidden", 128)),
            n_tokens=int(payload.get("n_tokens", 4)),
            embed_dim=int(payload.get("embed_dim", 1024)),
        )


def insert_condition_slots(
    batch: dict[str, torch.Tensor],
    *,
    n_tokens: int,
    dummy_id: int,
    insert_at: int = CONDITION_INSERT_AT,
) -> dict[str, torch.Tensor]:
    """Insert dummy token ids after BOS+direction. Embeddings are replaced in the hook."""
    input_ids = batch["input_ids"]
    attention = batch["attention_mask"]
    labels = batch["labels"]
    batch_size = input_ids.size(0)
    dummy = input_ids.new_full((batch_size, n_tokens), int(dummy_id))
    ones = attention.new_ones((batch_size, n_tokens))
    ignore = labels.new_full((batch_size, n_tokens), -100)
    batch["input_ids"] = torch.cat([input_ids[:, :insert_at], dummy, input_ids[:, insert_at:]], dim=1)
    batch["attention_mask"] = torch.cat([attention[:, :insert_at], ones, attention[:, insert_at:]], dim=1)
    batch["labels"] = torch.cat([labels[:, :insert_at], ignore, labels[:, insert_at:]], dim=1)
    return batch


class ConditionalProtGPT3(nn.Module):
    """V1 PeftModel + trainable soft-prompt MLP. Base ProtGPT3 stays frozen.

    Dummy ids are placed after ``<|bos|> 1``; an embedding hook overwrites those
    positions with MLP outputs so the LM still runs on ``input_ids``.
    """

    def __init__(self, model: PeftModel, conditioner: ConditionMLP, normalizer: ConditionNormalizer):
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.normalizer = normalizer
        self.insert_at = CONDITION_INSERT_AT
        self._pending_cond: torch.Tensor | None = None
        self._hook = self.model.get_input_embeddings().register_forward_hook(self._on_embed)

    def _on_embed(self, _module, _args, output: torch.Tensor) -> torch.Tensor:
        cond = self._pending_cond
        n_tokens = self.conditioner.n_tokens
        if cond is None or output.size(1) < self.insert_at + n_tokens:
            return output
        patched = output.clone()
        patched[:, self.insert_at : self.insert_at + n_tokens, :] = cond.to(dtype=output.dtype)
        return patched

    def _set_condition(self, conditions: torch.Tensor) -> None:
        self._pending_cond = self.conditioner(self.normalizer.encode(conditions))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        conditions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        self._set_condition(conditions)
        try:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._pending_cond = None

    def save_bundle(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        torch.save(self.conditioner.state_dict(), directory / "conditioner.pt")
        (directory / "conditioner_config.json").write_text(
            json.dumps(self.conditioner.config_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        self.normalizer.save(directory / "normalizer.json")

    @torch.inference_mode()
    def generate(
        self,
        tokenizer,
        charge: float,
        hydrophobicity: float,
        *,
        num_return_sequences: int = 8,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        suppress_tokens: list[int] | None = None,
    ) -> list[str]:
        self.model.eval()
        self.conditioner.eval()
        device = next(self.model.parameters()).device
        prefix = encode_prompt(tokenizer, DIRECTION_N2C, device=device)
        prefix = prefix.expand(num_return_sequences, -1)
        dummy_id = condition_dummy_id(tokenizer)
        dummies = prefix.new_full((num_return_sequences, self.conditioner.n_tokens), dummy_id)
        input_ids = torch.cat([prefix, dummies], dim=1)
        attention = torch.ones_like(input_ids)
        conditions = torch.tensor(
            [[charge, hydrophobicity]] * num_return_sequences,
            device=device,
            dtype=torch.float32,
        )
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if suppress_tokens:
            generate_kwargs["suppress_tokens"] = suppress_tokens
        self._set_condition(conditions)
        try:
            output_ids = self.model.generate(**generate_kwargs)
        finally:
            self._pending_cond = None
        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return texts


def load_v1_peft(config: dict[str, Any], adapter_dir: Path | str, *, trainable: bool) -> tuple[Any, PeftModel]:
    tokenizer, base = load_from_config(config, for_training=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=trainable)
    if trainable:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
    else:
        model.eval()
    return tokenizer, model


def load_v2_bundle(
    config: dict[str, Any],
    checkpoint_dir: Path | str,
    *,
    device: torch.device | str = "cuda",
    trainable: bool = False,
) -> tuple[Any, ConditionalProtGPT3]:
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer, peft_model = load_v1_peft(config, checkpoint_dir, trainable=trainable)
    peft_model.to(device)
    cond_cfg = json.loads((checkpoint_dir / "conditioner_config.json").read_text(encoding="utf-8"))
    conditioner = ConditionMLP.from_config(cond_cfg)
    state = torch.load(checkpoint_dir / "conditioner.pt", map_location=device, weights_only=True)
    conditioner.load_state_dict(state)
    conditioner.to(device)
    normalizer = ConditionNormalizer.load(checkpoint_dir / "normalizer.json")
    wrapped = ConditionalProtGPT3(peft_model, conditioner, normalizer)
    if "insert_at" in cond_cfg:
        wrapped.insert_at = int(cond_cfg["insert_at"])
    wrapped.to(device)
    return tokenizer, wrapped


def encode_conditional_batch(
    tokenizer,
    sequences: list[str],
    vectors: Sequence[tuple[float, float]] | torch.Tensor,
    device: torch.device | str,
    *,
    n_tokens: int,
) -> dict[str, torch.Tensor]:
    dummy_id = condition_dummy_id(tokenizer)
    if dummy_id >= int(tokenizer.vocab_size):
        raise RuntimeError(f"dummy id {dummy_id} >= vocab_size {tokenizer.vocab_size}")
    batch = encode_train_batch(tokenizer, sequences, device)
    batch = insert_condition_slots(batch, n_tokens=n_tokens, dummy_id=dummy_id)
    if isinstance(vectors, torch.Tensor):
        conditions = vectors.to(device=device, dtype=torch.float32)
    else:
        conditions = torch.tensor(list(vectors), device=device, dtype=torch.float32)
    batch["conditions"] = conditions
    return batch


def decode_generated(texts: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in texts:
        parsed = decode_protein(text)
        sequence = parsed.sequence
        rows.append(
            {
                "sequence": sequence,
                "direction": parsed.direction,
                "length": len(sequence),
                "valid_alphabet": bool(sequence) and is_challenge_alphabet(sequence),
            }
        )
    return rows


def default_suppress(tokenizer) -> list[int]:
    return challenge_suppress_token_ids(tokenizer)
