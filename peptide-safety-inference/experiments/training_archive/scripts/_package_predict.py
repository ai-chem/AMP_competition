#!/usr/bin/env python
"""Inference CLI — production peptide safety predictor.

Example:
  python predict.py --input examples/controllability_sequences.csv --output predictions.csv --device auto
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from psp.embeddings import embed_sequences  # noqa: E402
from psp.features import (  # noqa: E402
    PHYSCHEM_SUMMARY_COLS,
    WW_INTERFACE,
    _mean_scale,
    camsol_like_score,
    featurize_frame,
    net_charge,
)
from psp.ood import ApplicabilityDomain  # noqa: E402
from psp.paths import AA20_SET, FINAL_MODELS, HC50_SAFE_THRESHOLD_UM  # noqa: E402
from psp.uncertainty import gaussian_interval  # noqa: E402

LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))
ESM_DEFAULT = "facebook/esm2_t12_35M_UR50D"


def normalize_sequence(raw) -> tuple[str | None, str | None]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, "empty sequence"
    text = str(raw).strip().upper()
    if not text or text == "NAN":
        return None, "empty sequence"
    bad = sorted(set(text) - AA20_SET)
    if bad:
        return None, "nonstandard residues: " + "".join(bad)
    return text, None


def load_bundle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_ft_head(bundle: dict, device: torch.device):
    model_name = bundle.get("embedding_model", ESM_DEFAULT)
    ckpt_path = Path(bundle["esm_ft_ckpt"])
    if not ckpt_path.is_file():
        alt = FINAL_MODELS / "esm2_35M_ft.pt"
        if alt.is_file():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"ESM fine-tune checkpoint not found: {bundle['esm_ft_ckpt']}")
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

    head = Head().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, tokenizer


@torch.no_grad()
def _predict_ft(head, tokenizer, sequences, device, bs: int = 8):
    mus, sigs = [], []
    for i in range(0, len(sequences), bs):
        batch = sequences[i : i + bs]
        toks = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        mu, log_sigma = head(toks["input_ids"], toks["attention_mask"])
        mus.append(mu.cpu().numpy())
        sigs.append(np.exp(np.clip(log_sigma.cpu().numpy(), -5, 5)))
    return np.concatenate(mus), np.concatenate(sigs)


def predict_hc50(sequences: list[str], bundle: dict, device: str = "auto") -> pd.DataFrame:
    feats = featurize_frame(sequences)
    phys = np.nan_to_num(feats[PHYSCHEM_SUMMARY_COLS].to_numpy(dtype=float))
    emb = embed_sequences(
        sequences,
        model_name=bundle.get("embedding_model", ESM_DEFAULT),
        pooling="mean",
        batch_size=8,
        device=device,
    )
    X = np.concatenate([phys, emb], axis=1)

    torch_device = _resolve_device(device)
    head, tokenizer = _load_ft_head(bundle, torch_device)
    mu, sigma_ft = _predict_ft(head, tokenizer, sequences, torch_device)
    q = float(bundle.get("conformal_q", 1.0))
    sigma = np.maximum(sigma_ft, q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    if X.shape[1] == bundle["clf"].n_features_in_:
        p_raw = bundle["clf"].predict_proba(X)[:, 1]
    else:
        n_need = bundle["clf"].n_features_in_
        Xc = phys[:, :n_need] if phys.shape[1] >= n_need else np.pad(
            phys, ((0, 0), (0, n_need - phys.shape[1]))
        )
        p_raw = bundle["clf"].predict_proba(Xc)[:, 1]
    p_cal = bundle["calibrator"].transform(p_raw)
    p_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.clip(sigma, 1e-6, None))
    p_safe = 0.5 * p_cal + 0.5 * p_reg

    ood = ApplicabilityDomain(
        bundle["ood_train_sequences"],
        bundle["ood_train_embeddings"],
        bundle["ood_train_physchem"],
    ).score(sequences, emb, phys, identity_train_cap=2000)

    return pd.DataFrame(
        {
            "hc50_log_uM_pred": mu,
            "hc50_uM_pred": np.exp(mu),
            "hc50_log_lower_95": lo95,
            "hc50_log_upper_95": hi95,
            "hc50_lower_95_uM": np.exp(lo95),
            "hc50_upper_95_uM": np.exp(hi95),
            "p_hc50_gt_128": p_safe,
            "hemolysis_risk_prob": 1.0 - p_safe,
            "hc50_uncertainty": sigma,
            "max_train_identity": ood.max_train_identity,
            "embedding_ood_score": ood.embedding_ood_score,
            "physchem_ood_score": ood.physchem_ood_score,
            "in_domain": ood.in_domain,
            "prediction_confidence": ood.prediction_confidence,
            "charge_pH7_4_recomputed": [net_charge(s, 7.4) for s in sequences],
            "hydrophobicity_WW_recomputed": [_mean_scale(s, WW_INTERFACE) for s in sequences],
            "solubility_proxy_score": [camsol_like_score(s) for s in sequences],
        }
    )


def maybe_solubility_model(sequences: list[str], out: pd.DataFrame) -> pd.DataFrame:
    path = FINAL_MODELS / "solubility_aqueous_pooled.pkl"
    if not path.exists():
        out["solubility_endpoint"] = "aqueous model missing"
        return out
    with path.open("rb") as f:
        bundle = pickle.load(f)
    feats = featurize_frame(sequences)
    cols = list(bundle["feature_cols"])
    base_cols = [c for c in cols if not c.startswith("solv_")]
    for c in base_cols:
        if c not in feats.columns:
            feats[c] = 0.0
    X = np.nan_to_num(feats[base_cols].to_numpy(dtype=float))
    if bundle.get("solvent_dummies"):
        solv_cols = [c for c in cols if c.startswith("solv_")]
        default = bundle.get("default_solvent", "0.1 M PBS")
        target = f"solv_{default}"
        mat = np.zeros((len(sequences), len(solv_cols)), dtype=float)
        if target in solv_cols:
            mat[:, solv_cols.index(target)] = 1.0
        elif any("PBS" in c for c in solv_cols):
            j = next(i for i, c in enumerate(solv_cols) if "PBS" in c and "DPBS" not in c)
            mat[:, j] = 1.0
        X = np.concatenate([X, mat], axis=1)
    proba = bundle["model"].predict_proba(X)[:, 1]
    out["solubility_aqueous_probability"] = proba
    out["solubility_endpoint"] = bundle.get(
        "endpoint",
        "Experimental aqueous/buffer soluble vs insoluble (SolPepBench)",
    )
    out["solubility_condition"] = bundle.get("default_solvent", "0.1 M PBS")
    return out


def maybe_stability_model(sequences: list[str], out: pd.DataFrame) -> pd.DataFrame:
    path = FINAL_MODELS / "stability_peplife2_protease.pkl"
    if not path.exists():
        return out
    with path.open("rb") as f:
        bundle = pickle.load(f)
    feats = featurize_frame(sequences)
    cols = bundle["feature_cols"]
    for c in cols:
        if c not in feats.columns:
            feats[c] = 0.0
    X = np.nan_to_num(feats[cols].to_numpy(dtype=float))
    pred = bundle["model"].predict(X)
    out["stability_log_half_life"] = pred
    out["stability_half_life_hours"] = np.exp(pred)
    out["stability_endpoint"] = bundle.get("endpoint", "PEPlife2 protease half-life (log hours)")
    return out


def run(input_csv: Path, output_csv: Path, sequence_col: str, device: str) -> pd.DataFrame:
    bundle_path = FINAL_MODELS / "hc50_bundle.pkl"
    if not bundle_path.exists():
        raise SystemExit(f"Missing {bundle_path}")
    table = pd.read_csv(input_csv)
    if sequence_col not in table.columns:
        raise SystemExit(f"No '{sequence_col}' column. Columns: {list(table.columns)}")

    cleaned, errors = [], []
    for raw in table[sequence_col]:
        seq, err = normalize_sequence(raw)
        cleaned.append(seq or "")
        errors.append(err)
    if any(errors):
        bad = [f"row {i}: {e}" for i, e in enumerate(errors) if e]
        raise SystemExit("Invalid sequences:\n" + "\n".join(bad[:20]))

    uniq = list(dict.fromkeys(cleaned))
    print(f"Predicting {len(uniq)} unique sequences (from {len(cleaned)} rows)...")
    bundle = load_bundle(bundle_path)
    preds_u = predict_hc50(uniq, bundle, device=device)
    preds_u = maybe_solubility_model(uniq, preds_u)
    preds_u = maybe_stability_model(uniq, preds_u)
    index = {s: i for i, s in enumerate(uniq)}
    preds = preds_u.iloc[[index[s] for s in cleaned]].reset_index(drop=True)

    if "charge_pH7_4" in table.columns:
        preds["charge_pH7_4_deviation"] = (
            preds["charge_pH7_4_recomputed"].to_numpy() - table["charge_pH7_4"].to_numpy()
        )
    if "hydrophobicity_interfaceScale_pH8" in table.columns:
        preds["hydrophobicity_deviation"] = (
            preds["hydrophobicity_WW_recomputed"].to_numpy()
            - table["hydrophobicity_interfaceScale_pH8"].to_numpy()
        )

    out = pd.concat([table.reset_index(drop=True), preds], axis=1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} rows -> {output_csv}")
    return out


def main():
    p = argparse.ArgumentParser(description="Predict peptide HC50 / aqueous solubility / stability")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    run(args.input, args.output, args.sequence_col, args.device)


if __name__ == "__main__":
    main()
