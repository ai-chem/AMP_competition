#!/usr/bin/env python
"""Freeze final HC50 bundle: ESM-2 fine-tune primary (selected on FT val before test).

Selection rationale (no locked-test peeking for choice):
  - CatBoost OOF pearson = 0.352 (5-fold group CV)
  - ESM-2 FT validation pearson = 0.398 (GroupShuffleSplit, censored NLL)
  FT wins on the pre-registered validation metric → primary regressor.
  Classifier / calibrator / AD retained from v5. Locked-test metrics recorded once.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.embeddings import embed_sequences  # noqa: E402
from psp.evaluation import classification_metrics, regression_metrics  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.ood import ApplicabilityDomain  # noqa: E402
from psp.paths import (  # noqa: E402
    CONFIGS,
    FINAL_MODELS,
    HC50_SAFE_THRESHOLD_UM,
    PROCESSED,
    REPORTS,
)
from psp.uncertainty import gaussian_interval  # noqa: E402

ESM = "facebook/esm2_t12_35M_UR50D"
LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))
CKPT = FINAL_MODELS / "esm2_35M_ft.pt"


def collapse(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    part["_pref"] = (part["censor_type"] != "exact").astype(int)
    return (
        part.sort_values(["sequence", "_pref"])
        .drop_duplicates("sequence", keep="first")
        .drop(columns="_pref")
        .reset_index(drop=True)
    )


def build_head(device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ESM)
    base = AutoModel.from_pretrained(ESM)
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

    model = Head().to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_ft(model, tokenizer, sequences, device, bs=8):
    preds = []
    for i in range(0, len(sequences), bs):
        batch = sequences[i : i + bs]
        toks = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        mu, log_sigma = model(toks["input_ids"], toks["attention_mask"])
        preds.append(mu.cpu().numpy())
        if i == 0:
            sigma0 = np.exp(log_sigma.cpu().numpy())
    # sigma from last batch only used as fallback; recompute properly
    return np.concatenate(preds, 0)


@torch.no_grad()
def predict_ft_dist(model, tokenizer, sequences, device, bs=8):
    mus, sigs = [], []
    for i in range(0, len(sequences), bs):
        batch = sequences[i : i + bs]
        toks = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        mu, log_sigma = model(toks["input_ids"], toks["attention_mask"])
        mus.append(mu.cpu().numpy())
        sigs.append(np.exp(np.clip(log_sigma.cpu().numpy(), -5, 5)))
    return np.concatenate(mus, 0), np.concatenate(sigs, 0)


def rc_consistency(pred, censor_type, lo):
    right = censor_type == "right"
    if not right.any():
        return float("nan")
    floors = lo[right]
    ok = np.isfinite(floors) & np.isfinite(pred[right])
    if not ok.any():
        return float("nan")
    return float(np.mean(pred[right][ok] >= floors[ok] - 1e-6))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (FINAL_MODELS / "hc50_bundle.pkl").open("rb") as f:
        bundle = pickle.load(f)

    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])
    tr_seqs = train["sequence"].tolist()
    te_seqs = test["sequence"].tolist()

    model, tokenizer = build_head(device)
    print("Predict locked test with ESM-FT...")
    mu, sigma_ft = predict_ft_dist(model, tokenizer, te_seqs, device)
    q = float(bundle.get("conformal_q", 1.5))
    sigma = np.maximum(sigma_ft, q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    y_te = test["hc50_log_value"].to_numpy(float)
    exact_te = test["censor_type"] == "exact"
    reg = regression_metrics(y_te[exact_te], mu[exact_te])
    rc = rc_consistency(
        mu, test["censor_type"].to_numpy(), test["censor_lower_log"].to_numpy(float)
    )
    cover = float(
        np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te]))
    )

    # classifier from v5 features
    feats_te = featurize_frame(te_seqs)
    phys_te = np.nan_to_num(feats_te[PHYSCHEM_SUMMARY_COLS].to_numpy(float))
    emb_te = embed_sequences(te_seqs, model_name=ESM, pooling="mean", batch_size=16)
    Xte = np.concatenate([phys_te, emb_te], axis=1)
    p_raw = bundle["clf"].predict_proba(Xte)[:, 1]
    p_cal = bundle["calibrator"].transform(p_raw)
    p_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.maximum(sigma, 1e-3))
    p_safe = 0.5 * p_cal + 0.5 * p_reg
    y_safe = test["y_safe_gt_128"].to_numpy(float)
    known = np.isfinite(y_safe)
    clf_met = classification_metrics(y_safe[known], p_safe[known])

    print(
        f"reg pearson={reg.get('pearson_r'):.3f} spearman={reg.get('spearman_rho'):.3f} "
        f"r2={reg.get('r2'):.3f}"
    )
    print(
        f"clf AUC={clf_met.get('roc_auc'):.3f} MCC={clf_met.get('mcc'):.3f} "
        f"rc_cons={rc:.3f} cover90={cover:.3f}"
    )

    feats_tr = featurize_frame(tr_seqs)
    phys_tr = np.nan_to_num(feats_tr[PHYSCHEM_SUMMARY_COLS].to_numpy(float))
    emb_tr = embed_sequences(tr_seqs, model_name=ESM, pooling="mean", batch_size=16)
    ad = ApplicabilityDomain(tr_seqs, emb_tr, phys_tr)
    ood = ad.score(te_seqs, emb_te, phys_te, identity=False)

    metrics = {
        "protocol": {
            "selection": "esm2_35M_ft primary (val pearson 0.398 > catboost OOF 0.352)",
            "ft_val_pearson": 0.397548,
            "catboost_oof_pearson": 0.352,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_exact": int(exact_te.sum()),
            "locked_split_seed": 20260925,
            "note": (
                "Regressor = ESM-2 35M fine-tune with censored Gaussian head. "
                "Classifier/calibrator from v5 physchem+ESM ExtraTrees. "
                "Chosen on validation before locked test."
            ),
        },
        "regression_exact": reg,
        "classification_safe": clf_met,
        "interval_coverage_90": cover,
        "right_censored_consistency": rc,
        "calibrator": bundle.get("calibrator_name"),
        "conformal_q": q,
    }
    (REPORTS / "final_v7_ft_primary_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    pred = test[
        [
            "sequence",
            "censor_type",
            "hc50_value",
            "hc50_log_value",
            "censor_lower",
            "y_safe_gt_128",
            "cluster_id70",
            "source",
        ]
    ].copy()
    pred["pred_log_hc50"] = mu
    pred["pred_hc50_uM"] = np.exp(mu)
    pred["pred_log_lo95"] = lo95
    pred["pred_log_hi95"] = hi95
    pred["p_hc50_gt_128"] = p_safe
    pred["hemolysis_risk_prob"] = 1 - p_safe
    pred["ood_max_train_identity"] = ood.max_train_identity
    pred["in_domain"] = ood.in_domain
    pred["prediction_confidence"] = ood.prediction_confidence
    pred.to_csv(REPORTS / "final_v7_ft_primary_predictions.csv", index=False)

    bundle["primary_regressor"] = "esm2_35M_ft"
    bundle["esm_ft_ckpt"] = str(CKPT)
    bundle["esm_ft_weight"] = 1.0
    bundle["version"] = "v7_ft_primary"
    bundle["selection"] = {
        "model": "esm2_35M_ft",
        "features": "esm2_t12_35M_finetune",
        "cv_pearson": 0.397548,
    }
    bundle["metrics_locked_test"] = metrics
    bundle["ood_train_sequences"] = tr_seqs
    bundle["ood_train_embeddings"] = emb_tr
    bundle["ood_train_physchem"] = phys_tr
    with (FINAL_MODELS / "hc50_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    cfg = {
        "model": "esm2_35M_ft",
        "features": "esm2_t12_35M_finetune",
        "version": "v7_ft_primary",
        "checkpoint": str(CKPT),
        "locked_test": {
            "pearson": reg.get("pearson_r"),
            "spearman": reg.get("spearman_rho"),
            "roc_auc": clf_met.get("roc_auc"),
            "right_censored_consistency": rc,
            "coverage_90": cover,
        },
        "aqueous_solubility": {
            "source": "SolPepBench / PepSol2000",
            "default_model": "solubility_aqueous_pooled.pkl",
            "cv_roc_auc_pooled": 0.785,
            "note": "NOT E. coli expression — aqueous/buffer binary solubility only",
        },
        "stability": {
            "source": "PEPlife2",
            "default_model": "stability_peplife2_protease.pkl",
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved bundle v7_ft_primary")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
