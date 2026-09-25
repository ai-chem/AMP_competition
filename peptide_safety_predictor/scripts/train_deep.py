#!/usr/bin/env python
"""Train deep sequence models (CNN / multiscale / BiLSTM) and optional ESM fine-tune."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.evaluation import regression_metrics  # noqa: E402
from psp.models.deep import (  # noqa: E402
    BiLSTMRegressor,
    CNNRegressor,
    MultiscaleCNN,
    censor_type_to_code,
    censored_nll_torch,
    encode_sequences,
)
from psp.paths import MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402


def load_train():
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = df[df["split"] == "train"].copy()
    train["_pref"] = (train["censor_type"] != "exact").astype(int)
    return train.sort_values(["sequence", "_pref"]).groupby("sequence", as_index=False).first()


def train_one(model, train_loader, val_pack, device, epochs=40, lr=1e-3, patience=8):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    bad = 0
    Xv, yv, cv, lo, hi = val_pack
    for epoch in range(epochs):
        model.train()
        for xb, yb, cb, lob, hib in train_loader:
            xb = xb.to(device)
            yb, cb, lob, hib = yb.to(device), cb.to(device), lob.to(device), hib.to(device)
            mu, log_sigma = model(xb)
            loss = censored_nll_torch(mu, log_sigma, yb, cb, lob, hib)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mu, log_sigma = model(Xv.to(device))
            vloss = censored_nll_torch(
                mu, log_sigma, yv.to(device), cv.to(device), lo.to(device), hi.to(device)
            ).item()
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def oof_deep(name, factory, sequences, y_log, censor_type, lo, hi, groups, strata, device, seed=0):
    max_len = min(60, max(len(s) for s in sequences))
    X = encode_sequences(sequences, max_len=max_len)
    y = torch.tensor(np.nan_to_num(y_log, nan=0.0), dtype=torch.float32)
    c = torch.tensor(censor_type_to_code(censor_type), dtype=torch.int64)
    lo_t = torch.tensor(np.nan_to_num(lo, nan=0.0), dtype=torch.float32)
    hi_t = torch.tensor(np.nan_to_num(hi, nan=0.0), dtype=torch.float32)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    mu_oof = np.full(len(sequences), np.nan)
    for tr, va in cv.split(X, strata, groups):
        model = factory().to(device)
        ds = TensorDataset(X[tr], y[tr], c[tr], lo_t[tr], hi_t[tr])
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        val_pack = (X[va], y[va], c[va], lo_t[va], hi_t[va])
        train_one(model, loader, val_pack, device)
        model.eval()
        with torch.no_grad():
            mu, _ = model(X[va].to(device))
            mu_oof[va] = mu.cpu().numpy()
    exact = np.asarray(censor_type) == "exact"
    met = regression_metrics(y_log[exact], mu_oof[exact])
    # save one full-train model
    model = factory().to(device)
    ds = TensorDataset(X, y, c, lo_t, hi_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    # use last 10% as internal early-stop only (still within train)
    n = len(sequences)
    cut = int(n * 0.9)
    val_pack = (X[cut:], y[cut:], c[cut:], lo_t[cut:], hi_t[cut:])
    train_one(model, DataLoader(TensorDataset(X[:cut], y[:cut], c[:cut], lo_t[:cut], hi_t[:cut]), batch_size=64, shuffle=True), val_pack, device)
    torch.save(
        {"state_dict": model.state_dict(), "max_len": max_len, "name": name},
        MODELS / "final" / f"{name}.pt",
    )
    return met, mu_oof


def try_finetune_esm(sequences, y_log, censor_type, lo, hi, groups, strata, device):
    """One ESM-2 35M fine-tune attempt with censored head."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        return {"model": "esm2_35M_ft", "family": "finetune", "error": str(exc)}

    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModel.from_pretrained(model_name)
    hidden = base.config.hidden_size

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.head = nn.Linear(hidden, 2)

        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            h = self.head(pooled)
            return h[:, 0], h[:, 1]

    # Use a single train/val split (not full 5-fold — GPU time) inside training portion
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, va = next(gss.split(sequences, groups=groups))
    model = Head().to(device)
    # freeze most layers; unfreeze last 2 encoder layers + head
    for p in model.base.parameters():
        p.requires_grad = False
    for layer in model.base.encoder.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    codes = censor_type_to_code(censor_type)
    best = float("inf")
    best_state = None
    bad = 0
    for epoch in range(8):
        model.train()
        order = np.random.default_rng(epoch).permutation(tr)
        for i in range(0, len(order), 4):
            idx = order[i : i + 4]
            batch = [sequences[j] for j in idx]
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            toks = {k: v.to(device) for k, v in toks.items()}
            mu, log_sigma = model(toks["input_ids"], toks["attention_mask"])
            yb = torch.tensor(np.nan_to_num(y_log[idx], nan=0.0), dtype=torch.float32, device=device)
            cb = torch.tensor(codes[idx], dtype=torch.int64, device=device)
            lob = torch.tensor(np.nan_to_num(lo[idx], nan=0.0), dtype=torch.float32, device=device)
            hib = torch.tensor(np.nan_to_num(hi[idx], nan=0.0), dtype=torch.float32, device=device)
            loss = censored_nll_torch(mu, log_sigma, yb, cb, lob, hib)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            toks = tokenizer([sequences[j] for j in va], return_tensors="pt", padding=True, truncation=True, max_length=128)
            toks = {k: v.to(device) for k, v in toks.items()}
            mu, log_sigma = model(toks["input_ids"], toks["attention_mask"])
            yb = torch.tensor(np.nan_to_num(y_log[va], nan=0.0), dtype=torch.float32, device=device)
            cb = torch.tensor(codes[va], dtype=torch.int64, device=device)
            lob = torch.tensor(np.nan_to_num(lo[va], nan=0.0), dtype=torch.float32, device=device)
            hib = torch.tensor(np.nan_to_num(hi[va], nan=0.0), dtype=torch.float32, device=device)
            vloss = censored_nll_torch(mu, log_sigma, yb, cb, lob, hib).item()
            pred = mu.cpu().numpy()
        exact = np.asarray(censor_type)[va] == "exact"
        met = regression_metrics(y_log[va][exact], pred[exact])
        print(f"  ft epoch {epoch}: val_nll={vloss:.4f} pearson={met.get('pearson_r')}")
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_met = met
            bad = 0
        else:
            bad += 1
            if bad >= 3:
                break
    if best_state:
        model.load_state_dict(best_state)
        torch.save({"state_dict": best_state, "model_name": model_name}, MODELS / "final" / "esm2_35M_ft.pt")
    best_met = best_met if "best_met" in dir() else {}
    return {"model": "esm2_35M_ft", "family": "finetune", "features": "esm2_t12_35M", **best_met}


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    train = load_train()
    sequences = train["sequence"].tolist()
    y_log = train["hc50_log_value"].to_numpy(dtype=float)
    censor_type = train["censor_type"].to_numpy()
    lo = (
        train["censor_lower_log"].to_numpy(dtype=float)
        if "censor_lower_log" in train.columns
        else np.where(
            train["censor_lower"].notna() & (train["censor_lower"] > 0),
            np.log(train["censor_lower"].astype(float)),
            np.nan,
        )
    )
    hi = (
        train["censor_upper_log"].to_numpy(dtype=float)
        if "censor_upper_log" in train.columns
        else np.where(
            train["censor_upper"].notna() & (train["censor_upper"] > 0),
            np.log(train["censor_upper"].astype(float)),
            np.nan,
        )
    )
    groups = train["cluster_id70"].to_numpy()
    strata = np.where(censor_type == "exact", 0, 1)

    results = []
    factories = {
        "cnn1d": lambda: CNNRegressor(),
        "multiscale_cnn": lambda: MultiscaleCNN(),
        "bilstm": lambda: BiLSTMRegressor(),
    }
    for name, factory in factories.items():
        print(f"Training {name}...")
        try:
            met, _ = oof_deep(name, factory, sequences, y_log, censor_type, lo, hi, groups, strata, device)
            results.append({"model": name, "family": "deep", "features": "sequence", **met})
            print(f"  {name}: pearson={met.get('pearson_r')}")
        except Exception as exc:  # noqa: BLE001
            results.append({"model": name, "family": "deep", "error": str(exc)})
            print(f"  {name} failed: {exc}")

    print("Fine-tuning ESM-2 35M (one run)...")
    try:
        results.append(try_finetune_esm(sequences, y_log, censor_type, lo, hi, groups, strata, device))
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "esm2_35M_ft", "family": "finetune", "error": str(exc)})
        print(f"Fine-tune failed: {exc}")

    out = pd.DataFrame(results)
    out.to_csv(REPORTS / "deep_results.csv", index=False)
    all_path = REPORTS / "all_experiments.csv"
    if all_path.exists():
        prev = pd.read_csv(all_path)
        flat = out.copy()
        flat["features"] = flat.get("features", "sequence")
        flat["seed"] = 0
        pd.concat([prev, flat], ignore_index=True).to_csv(all_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
