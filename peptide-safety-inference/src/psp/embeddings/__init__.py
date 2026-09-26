"""Frozen protein language model embeddings with on-disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from psp.paths import EMBEDDINGS, ensure_dirs

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
LARGE_MODEL = "facebook/esm2_t33_650M_UR50D"


def _device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cache_path(model_name: str, pooling: str, sequences: Sequence[str]) -> Path:
    ensure_dirs()
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(pooling.encode())
    for s in sequences:
        h.update(s.encode())
        h.update(b"|")
    return EMBEDDINGS / f"{model_name.replace('/', '_')}__{pooling}__{h.hexdigest()[:16]}.npz"


def embed_sequences(
    sequences: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    pooling: str = "mean",
    batch_size: int = 8,
    device: str = "auto",
    cache: bool = True,
) -> np.ndarray:
    """Return (N, D) float32 embeddings. Cached by content hash when cache=True."""
    if not sequences:
        return np.zeros((0, 0), dtype=np.float32)
    cache_file = _cache_path(model_name, pooling, sequences)
    if cache and cache_file.exists():
        data = np.load(cache_file)
        return data["embeddings"]

    from transformers import AutoModel, AutoTokenizer

    dev = _device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # fp16 only on CUDA; Pascal (GTX 1060) supports fp16
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    model.to(dev)
    model.eval()

    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"embed:{model_name}"):
            batch = list(sequences[i : i + batch_size])
            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1022,
            )
            toks = {k: v.to(dev) for k, v in toks.items()}
            out = model(**toks)
            hidden = out.last_hidden_state  # (B, L, D)
            mask = toks["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            # drop special tokens roughly by mask; mean over non-pad
            if pooling == "mean":
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = summed / denom
            elif pooling == "max":
                masked = hidden.masked_fill(mask == 0, -1e4)
                pooled = masked.max(dim=1).values
            elif pooling == "cls":
                pooled = hidden[:, 0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            outs.append(pooled.float().cpu().numpy())

    emb = np.concatenate(outs, axis=0).astype(np.float32)
    if cache:
        np.savez_compressed(cache_file, embeddings=emb, sequences=np.array(sequences, dtype=object))
    # free VRAM
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return emb


def embed_unique_then_expand(
    sequences: Sequence[str],
    **kwargs,
) -> np.ndarray:
    """Embed unique sequences once, then expand back to original order."""
    uniq = list(dict.fromkeys(sequences))
    emb_u = embed_sequences(uniq, **kwargs)
    index = {s: i for i, s in enumerate(uniq)}
    return np.stack([emb_u[index[s]] for s in sequences], axis=0)
