"""Load ProtGPT3, encode/decode protein strings, and sample sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amp_competition.config import load_config
from amp_competition.constants import STANDARD_AMINO_ACIDS

Direction = Literal["N2C", "C2N", "unknown"]

DIRECTION_N2C = "1"
DIRECTION_C2N = "2"
MODEL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBOUXZ"


@dataclass(frozen=True)
class ProteinDecode:
    """Decoded ProtGPT3 text with direction stripped and oriented N-to-C."""

    raw: str
    direction: Direction
    sequence: str


def load_tokenizer(model_id: str, trust_remote_code: bool = True, revision: str | None = None):
    """Match the ProtGPT3 model card: BOS on, EOS off, left padding."""
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "add_bos_token": True,
        "add_eos_token": False,
        "padding_side": "left",
    }
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def load_model(
    model_id: str,
    *,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    device_map: str | dict[str, Any] | None = "auto",
    eval_mode: bool = True,
    revision: str | None = None,
):
    kwargs: dict[str, Any] = {
        "dtype": _dtype(dtype),
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if revision:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if eval_mode:
        model.eval()
    return model


def load_from_config(config: dict[str, Any] | None = None, *, for_training: bool = False):
    if config is None:
        config = load_config()
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("id", "AI4PD/ProtGPT3-1.3B")
    revision = model_cfg.get("revision") or None
    tokenizer = load_tokenizer(
        model_id,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        revision=revision,
    )
    model = load_model(
        model_id,
        dtype=str(model_cfg.get("dtype", "bfloat16")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        device_map=None if for_training else "auto",
        eval_mode=not for_training,
        revision=revision,
    )
    return tokenizer, model


def encode_prompt(tokenizer, prompt: str, device: torch.device | str | None = None) -> torch.Tensor:
    """Encode a prompt and guarantee a leading BOS token.

    ProtGPT3 expects ``<|bos|>`` then optional direction ``1`` (N→C) or ``2`` (C→N).
    """
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise RuntimeError("Tokenizer is missing bos_token_id")
    if input_ids.shape[-1] == 0 or int(input_ids[0, 0]) != int(bos_id):
        bos = torch.tensor([[bos_id]], dtype=input_ids.dtype)
        input_ids = bos if input_ids.shape[-1] == 0 else torch.cat([bos, input_ids], dim=1)
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids


def decode_protein(text: str) -> ProteinDecode:
    """Strip direction token and return an N-to-C amino-acid string."""
    raw = "".join(text.split())
    if raw.startswith(DIRECTION_N2C):
        return ProteinDecode(raw=raw, direction="N2C", sequence=raw[1:])
    if raw.startswith(DIRECTION_C2N):
        return ProteinDecode(raw=raw, direction="C2N", sequence=raw[1:][::-1])
    return ProteinDecode(raw=raw, direction="unknown", sequence=raw)


def is_model_alphabet(sequence: str) -> bool:
    return bool(sequence) and all(residue in MODEL_AMINO_ACIDS for residue in sequence)


def is_challenge_alphabet(sequence: str) -> bool:
    return bool(sequence) and all(residue in STANDARD_AMINO_ACIDS for residue in sequence)


def challenge_suppress_token_ids(tokenizer) -> list[int]:
    """Ban everything except the 20 challenge amino acids and EOS."""
    allowed = set(STANDARD_AMINO_ACIDS)
    eos_id = tokenizer.eos_token_id
    banned: set[int] = set()
    for token, idx in tokenizer.get_vocab().items():
        if idx == eos_id or token in allowed:
            continue
        banned.add(int(idx))
    return sorted(banned)


@torch.inference_mode()
def generate_sequences(
    tokenizer,
    model,
    prompt: str = DIRECTION_N2C,
    *,
    num_return_sequences: int = 4,
    min_new_tokens: int | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int | None = 42,
    suppress_tokens: list[int] | None = None,
) -> list[str]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    input_ids = encode_prompt(tokenizer, prompt, device=device)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "num_return_sequences": num_return_sequences,
    }
    if min_new_tokens is not None:
        generate_kwargs["min_new_tokens"] = min_new_tokens
    if suppress_tokens:
        generate_kwargs["suppress_tokens"] = suppress_tokens
    output_ids = model.generate(input_ids, **generate_kwargs)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
