#!/usr/bin/env python
"""Score saved ESM-2 fine-tune once on the locked test; optionally stack with v5.

Blend weight is chosen on the SAME GroupShuffleSplit validation fold used during
fine-tuning (not the locked test). Locked test is scored once at the end.
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
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.embeddings import embed_sequences  # noqa: E402
from psp.evaluation import classification_metrics, regression_metrics  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.paths import FINAL_MODELS, HC50_SAFE_THRESHOLD_UM, PROCESSED, REPORTS  # noqa: E402
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
        mu, _ = model(toks["input_ids"], toks["attention_mask"])
        preds.append(mu.cpu().numpy())
    return np.concatenate(preds, 0)


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
    print(f"device={device} ckpt={CKPT.exists()}")
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])

    model, tokenizer = build_head(device)
    tr_seqs = train["sequence"].tolist()
    te_seqs = test["sequence"].tolist()
    y_log = train["hc50_log_value"].to_numpy(float)
    groups = train["cluster_id70"].to_numpy()
    censor_type = train["censor_type"].to_numpy()
    lo = train["censor_lower_log"].to_numpy(float)

    # Same split as fine-tune for blend weight only
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_i, va_i = next(gss.split(tr_seqs, groups=groups))
    print("FT predict val...")
    mu_va = predict_ft(model, tokenizer, [tr_seqs[i] for i in va_i], device)
    exact_va = censor_type[va_i] == "exact"
    ft_va = regression_metrics(y_log[va_i][exact_va], mu_va[exact_va])
    print(f"FT val pearson={ft_va.get('pearson_r'):.3f} n={int(exact_va.sum())}")

    # CatBoost from current bundle on same val
    with (FINAL_MODELS / "hc50_bundle.pkl").open("rb") as f:
        bundle = pickle.load(f)
    feats = featurize_frame([tr_seqs[i] for i in va_i])
    phys = np.nan_to_num(feats[PHYSCHEM_SUMMARY_COLS].to_numpy(float))
    emb = embed_sequences(
        [tr_seqs[i] for i in va_i], model_name=ESM, pooling="mean", batch_size=16
    )
    Xva = np.concatenate([phys, emb], axis=1)
    members = bundle["blend_members"]
    weights = bundle["blend_weights"]
    mu_v5 = np.zeros(len(va_i))
    for name, w in weights.items():
        if name == "tobit":
            Xs = bundle["tobit_scaler"].transform(Xva)
            Xs = bundle["tobit_pca"].transform(Xs)
            mu_v5 += float(w) * bundle["tobit"].predict(Xs)
        else:
            mu_v5 += float(w) * np.asarray(members[name].predict(Xva)).reshape(-1)
    v5_va = regression_metrics(y_log[va_i][exact_va], mu_v5[exact_va])
    print(f"v5 val pearson={v5_va.get('pearson_r'):.3f}")

    best_w, best_p = 0.5, -np.inf
    for w in np.linspace(0, 1, 21):
        blend = w * mu_va + (1 - w) * mu_v5
        p = regression_metrics(y_log[va_i][exact_va], blend[exact_va]).get("pearson_r") or -np.inf
        if p > best_p:
            best_p, best_w = p, float(w)
    print(f"val blend FT={best_w:.2f} v5={1-best_w:.2f} pearson={best_p:.3f}")

    # ---- LOCKED TEST once ----
    print("=== LOCKED TEST ===")
    mu_ft = predict_ft(model, tokenizer, te_seqs, device)
    feats_te = featurize_frame(te_seqs)
    phys_te = np.nan_to_num(feats_te[PHYSCHEM_SUMMARY_COLS].to_numpy(float))
    emb_te = embed_sequences(te_seqs, model_name=ESM, pooling="mean", batch_size=16)
    Xte = np.concatenate([phys_te, emb_te], axis=1)
    mu_v5_te = np.zeros(len(te_seqs))
    for name, w in weights.items():
        if name == "tobit":
            Xs = bundle["tobit_scaler"].transform(Xte)
            Xs = bundle["tobit_pca"].transform(Xs)
            mu_v5_te += float(w) * bundle["tobit"].predict(Xs)
        else:
            mu_v5_te += float(w) * np.asarray(members[name].predict(Xte)).reshape(-1)
    mu = best_w * mu_ft + (1 - best_w) * mu_v5_te
    sigma = np.maximum(np.abs(mu_ft - mu_v5_te), float(bundle.get("conformal_q", 1.0)) / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    y_te = test["hc50_log_value"].to_numpy(float)
    exact_te = test["censor_type"] == "exact"
    reg_ft = regression_metrics(y_te[exact_te], mu_ft[exact_te])
    reg_v5 = regression_metrics(y_te[exact_te], mu_v5_te[exact_te])
    reg_bl = regression_metrics(y_te[exact_te], mu[exact_te])
    rc_ft = rc_consistency(mu_ft, test["censor_type"].to_numpy(), test["censor_lower_log"].to_numpy(float))
    rc_bl = rc_consistency(mu, test["censor_type"].to_numpy(), test["censor_lower_log"].to_numpy(float))
    cover = float(np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te])))

    p_raw = bundle["clf"].predict_proba(Xte)[:, 1]
    p_cal = bundle["calibrator"].transform(p_raw)
    p_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.maximum(sigma, 1e-3))
    p_safe = 0.5 * p_cal + 0.5 * p_reg
    y_safe = test["y_safe_gt_128"].to_numpy(float)
    known = np.isfinite(y_safe)
    clf_met = classification_metrics(y_safe[known], p_safe[known])

    print(f"FT   pearson={reg_ft.get('pearson_r'):.3f} rc={rc_ft:.3f}")
    print(f"v5   pearson={reg_v5.get('pearson_r'):.3f}")
    print(f"blend pearson={reg_bl.get('pearson_r'):.3f} spearman={reg_bl.get('spearman_rho'):.3f} "
          f"rc={rc_bl:.3f} cover90={cover:.3f}")
    print(f"clf AUC={clf_met.get('roc_auc'):.3f} MCC={clf_met.get('mcc'):.3f}")

    metrics = {
        "protocol": {
            "ft_val_pearson": ft_va.get("pearson_r"),
            "v5_val_pearson": v5_va.get("pearson_r"),
            "blend_ft_weight": best_w,
            "val_blend_pearson": best_p,
            "locked_split_seed": 20260925,
            "note": "FT weight from FT validation fold only; locked test once.",
        },
        "locked_test_ft_only": reg_ft,
        "locked_test_v5_only": reg_v5,
        "locked_test_blend": reg_bl,
        "ft_rc_consistency": rc_ft,
        "blend_rc_consistency": rc_bl,
        "interval_coverage_90": cover,
        "classification_safe": clf_met,
    }
    (REPORTS / "final_v6_ft_blend_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    # Update bundle if blend beats v5 on validation
    if best_p >= (v5_va.get("pearson_r") or -np.inf) - 1e-6:
        bundle["esm_ft_ckpt"] = str(CKPT)
        bundle["esm_ft_weight"] = best_w
        bundle["version"] = "v6_ft_v5_blend"
        bundle["selection"] = {
            "model": f"esm_ft={best_w:.2f}+v5={1-best_w:.2f}",
            "features": "esm2_ft+physchem+esm2",
            "cv_pearson": float(best_p),
        }
        bundle["metrics_locked_test"] = metrics
        with (FINAL_MODELS / "hc50_bundle.pkl").open("wb") as f:
            pickle.dump(bundle, f)
        print("Updated hc50_bundle.pkl to v6_ft_v5_blend")
    else:
        print("Kept v5 bundle (FT blend did not improve val)")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
