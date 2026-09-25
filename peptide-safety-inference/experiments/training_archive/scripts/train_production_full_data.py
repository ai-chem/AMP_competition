#!/usr/bin/env python
"""Train production HC50 + aqueous + stability heads on ALL suitable labelled data.

After model selection on the locked split, this script refits the winning
architecture on every eligible observation (train ∪ test) for deployment.
Locked-test metrics are NOT recomputed here (that would leak); they remain
those of the selection-phase model documented in experiments/.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.calibration import ProbabilityCalibrator  # noqa: E402
from psp.embeddings import embed_sequences  # noqa: E402
from psp.evaluation import regression_metrics  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.models.deep import censor_type_to_code, censored_nll_torch  # noqa: E402
from psp.paths import (  # noqa: E402
    CONFIGS,
    FINAL_MODELS,
    HC50_SAFE_THRESHOLD_UM,
    PROCESSED,
    REPORTS,
    ensure_dirs,
)
from psp.uncertainty import conformal_interval  # noqa: E402

ESM = "facebook/esm2_t12_35M_UR50D"
LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))
OUT_DIR = FINAL_MODELS  # overwritten with production weights


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    part = df.copy()
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

    return Head().to(device), tokenizer


def finetune_all(sequences, y_log, censor_type, lo, hi, groups, device):
    model, tokenizer = build_head(device)
    for p in model.base.parameters():
        p.requires_grad = False
    for layer in model.base.encoder.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    tr, va = next(gss.split(sequences, groups=groups))
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    codes = censor_type_to_code(censor_type)
    best, best_state, bad = float("inf"), None, 0
    best_met = {}

    for epoch in range(10):
        model.train()
        order = np.random.default_rng(epoch).permutation(tr)
        for i in range(0, len(order), 4):
            idx = order[i : i + 4]
            batch = [sequences[j] for j in idx]
            toks = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
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
            toks = tokenizer(
                [sequences[j] for j in va],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
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
        print(f"  epoch {epoch}: val_nll={vloss:.4f} pearson={met.get('pearson_r')}")
        if vloss < best:
            best, best_state, best_met, bad = vloss, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }, met, 0
        else:
            bad += 1
            if bad >= 3:
                break

    if best_state is None:
        raise RuntimeError("ESM fine-tune failed to produce a checkpoint")
    model.load_state_dict(best_state)

    # One short polish pass on ALL data (lower LR), then keep weights
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
    model.train()
    order = np.random.default_rng(99).permutation(len(sequences))
    for i in range(0, len(order), 4):
        idx = order[i : i + 4]
        batch = [sequences[j] for j in idx]
        toks = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
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

    ckpt = OUT_DIR / "esm2_35M_ft.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_name": ESM,
            "trained_on": "all_suitable_hc50",
            "n_sequences": len(sequences),
            "val_metrics": best_met,
        },
        ckpt,
    )
    print(f"Saved {ckpt}")
    return model, tokenizer, best_met, ckpt


@torch.no_grad()
def predict_ft(model, tokenizer, sequences, device, bs=8):
    model.eval()
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
    return np.concatenate(mus), np.concatenate(sigs)


def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    all_df = collapse(df)  # train ∪ test ∪ any other
    print(f"Production HC50 sequences: {len(all_df)}")
    print("censor:", all_df["censor_type"].value_counts().to_dict())

    sequences = all_df["sequence"].tolist()
    y_log = all_df["hc50_log_value"].to_numpy(float)
    censor_type = all_df["censor_type"].to_numpy()
    lo = all_df["censor_lower_log"].to_numpy(float)
    hi = all_df["censor_upper_log"].to_numpy(float)
    groups = all_df["cluster_id70"].to_numpy()
    exact = censor_type == "exact"

    print("Fine-tuning ESM-2 on all suitable HC50 data...")
    model, tokenizer, val_met, ckpt = finetune_all(
        sequences, y_log, censor_type, lo, hi, groups, device
    )

    print("Embeddings for classifier + AD...")
    feats = featurize_frame(sequences)
    phys_cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    phys = np.nan_to_num(feats[phys_cols].to_numpy(float))
    emb = embed_sequences(sequences, model_name=ESM, pooling="mean", batch_size=16)
    X = np.concatenate([phys, emb], axis=1)

    y_safe = all_df["y_safe_gt_128"].to_numpy(float)
    known = np.isfinite(y_safe)
    clf_X, clf_y, clf_g = X[known], y_safe[known].astype(int), groups[known]
    oof_p = np.full(len(clf_y), np.nan)
    for tr_i, va_i in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
        clf_X, clf_y, clf_g
    ):
        c = ExtraTreesClassifier(n_estimators=400, max_depth=14, n_jobs=-1, random_state=0)
        c.fit(clf_X[tr_i], clf_y[tr_i])
        oof_p[va_i] = c.predict_proba(clf_X[va_i])[:, 1]

    held = {"platt": np.full(len(clf_y), np.nan), "isotonic": np.full(len(clf_y), np.nan)}
    for tr_i, va_i in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(
        oof_p.reshape(-1, 1), clf_y, clf_g
    ):
        for name in held:
            cal = ProbabilityCalibrator(name).fit(clf_y[tr_i], oof_p[tr_i])
            held[name][va_i] = cal.transform(oof_p[va_i])

    def brier(y, p):
        m = np.isfinite(p)
        return float(np.mean((p[m] - y[m]) ** 2))

    cal_name = min(held, key=lambda n: brier(clf_y, held[n]))
    calibrator = ProbabilityCalibrator(cal_name).fit(clf_y, oof_p)
    clf_final = ExtraTreesClassifier(n_estimators=500, max_depth=14, n_jobs=-1, random_state=0)
    clf_final.fit(clf_X, clf_y)
    print("calibrator", cal_name)

    # Conformal residual from FT predictions on exact rows (in-sample; floor only)
    mu_all, sig_all = predict_ft(model, tokenizer, sequences, device)
    resid = np.abs(y_log[exact] - mu_all[exact])
    conf_q = float(conformal_interval(resid[np.isfinite(resid)], alpha=0.1))

    # Minimal dummy models list for predict.py compatibility
    from sklearn.dummy import DummyRegressor

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X[exact], y_log[exact])

    bundle = {
        "models": [("reg", dummy, None)],
        "primary_regressor": "esm2_35M_ft",
        "esm_ft_ckpt": str(ckpt.resolve()),
        "esm_ft_weight": 1.0,
        "clf": clf_final,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "features": "physchem+esm2",
        "feature_cols": phys_cols,
        "embedding_model": ESM,
        "conformal_q": conf_q,
        "ood_train_sequences": sequences,
        "ood_train_embeddings": emb,
        "ood_train_physchem": phys,
        "physchem_cols": PHYSCHEM_SUMMARY_COLS,
        "log128": LOG128,
        "hc50_safe_threshold_uM": HC50_SAFE_THRESHOLD_UM,
        "selection": {
            "model": "esm2_35M_ft",
            "features": "esm2_t12_35M_finetune",
            "cv_pearson": float(val_met.get("pearson_r") or 0),
        },
        "version": "production_full_data",
        "trained_on": {
            "n_sequences": len(sequences),
            "n_exact": int(exact.sum()),
            "censor_counts": all_df["censor_type"].value_counts().to_dict(),
            "includes_locked_test_labels": True,
            "note": (
                "Production refit on train∪test after selection. "
                "Do not use for locked-test claims; see experiments/comparison_tables."
            ),
        },
        "val_early_stop_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                                   for k, v in (val_met or {}).items()},
    }
    out = OUT_DIR / "hc50_bundle.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)

    cfg = {
        "model": "esm2_35M_ft",
        "features": "esm2_t12_35M_finetune",
        "version": "production_full_data",
        "checkpoint": "models/esm2_35M_ft.pt",
        "n_train_sequences": len(sequences),
        "selection_locked_test": {
            "note": "Metrics below are from selection-phase (train-only) eval; not re-scored after full-data refit.",
            "pearson": 0.317,
            "spearman": 0.383,
            "roc_auc": 0.837,
            "right_censored_consistency": 0.807,
            "coverage_90": 0.926,
        },
        "aqueous_solubility": {
            "source": "SolPepBench / PepSol2000",
            "default_model": "solubility_aqueous_pooled.pkl",
            "cv_roc_auc_pooled": 0.785,
            "note": "NOT E. coli expression",
        },
        "stability": {
            "source": "PEPlife2",
            "default_model": "stability_peplife2_protease.pkl",
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (REPORTS / "production_full_data.json").write_text(
        json.dumps(
            {
                "n_sequences": len(sequences),
                "val_early_stop": bundle["val_early_stop_metrics"],
                "calibrator": cal_name,
                "conformal_q": conf_q,
                "checkpoint": str(ckpt),
                "bundle": str(out),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Saved {out}")
    print("PRODUCTION_FULL_DATA_DONE")


if __name__ == "__main__":
    main()
