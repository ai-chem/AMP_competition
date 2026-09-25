#!/usr/bin/env python
"""Train and freeze the final HC50 safety predictor on the expanded dataset.

Protocol (single locked-test evaluation):
  1. Cluster-disjoint StratifiedGroupKFold on TRAIN only for model selection.
  2. Fit the selected model (5 seeds) on all train.
  3. Score locked test ONCE.
  4. Package ``models/final/hc50_bundle.pkl`` for ``predict.py``.

Candidate pool is deliberately small (the winners from the first study and from
the QMAP benchmark): SVR / ExtraTrees / Ridge on physchem and physchem+ESM-2.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.calibration import ProbabilityCalibrator  # noqa: E402
from psp.embeddings import embed_sequences  # noqa: E402
from psp.evaluation import (  # noqa: E402
    classification_metrics,
    regression_metrics,
)
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.ood import ApplicabilityDomain  # noqa: E402
from psp.paths import (  # noqa: E402
    CONFIGS,
    FINAL_MODELS,
    HC50_SAFE_THRESHOLD_UM,
    PROCESSED,
    REPORTS,
    ensure_dirs,
)
from psp.uncertainty import conformal_interval, gaussian_interval  # noqa: E402

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
SEEDS = [0, 1, 2, 3, 4]
LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = PROCESSED / "hc50_observations_with_splits.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run make_splits.py first")
    df = pd.read_parquet(path)
    # One primary row per sequence (prefer exact).
    def collapse(part: pd.DataFrame) -> pd.DataFrame:
        part = part.copy()
        part["_pref"] = (part["censor_type"] != "exact").astype(int)
        return (
            part.sort_values(["sequence", "_pref"])
            .drop_duplicates("sequence", keep="first")
            .drop(columns="_pref")
            .reset_index(drop=True)
        )

    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])
    return train, test


def features_for(sequences: list[str], kind: str, cache: dict) -> np.ndarray:
    if kind in cache:
        return cache[kind]
    feats = featurize_frame(sequences)
    phys = np.nan_to_num(
        feats[[c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]].to_numpy(dtype=float)
    )
    if kind == "physchem":
        X = phys
    elif kind == "full":
        cols = [c for c in feats.columns if not c.startswith(("dpc_", "term_"))]
        X = np.nan_to_num(feats[cols].to_numpy(dtype=float))
    elif kind == "esm2":
        X = embed_sequences(sequences, model_name=ESM_MODEL, pooling="mean", batch_size=16)
    elif kind == "physchem+esm2":
        emb = embed_sequences(sequences, model_name=ESM_MODEL, pooling="mean", batch_size=16)
        X = np.concatenate([phys, emb], axis=1)
    else:
        raise ValueError(kind)
    cache[kind] = X
    return X


def candidates(seed: int) -> dict:
    return {
        "svr": Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))]),
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "extratrees": ExtraTreesRegressor(
            n_estimators=400, max_depth=14, n_jobs=-1, random_state=seed
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    train, test = load_split()
    print(f"Train seqs={len(train)} test seqs={len(test)}")
    print(f"  train censor: {train['censor_type'].value_counts().to_dict()}")
    print(f"  test  censor: {test['censor_type'].value_counts().to_dict()}")

    # Exact-only for classical regression selection.
    exact_tr = train["censor_type"] == "exact"
    exact_te = test["censor_type"] == "exact"
    print(f"Exact train={exact_tr.sum()} exact test={exact_te.sum()}")

    feat_kinds = ["physchem", "full", "physchem+esm2"]
    if args.quick:
        feat_kinds = ["physchem", "full"]

    # Feature matrices for train (all sequences — needed for hybrid).
    tr_cache: dict = {}
    te_cache: dict = {}
    tr_seqs = train["sequence"].tolist()
    te_seqs = test["sequence"].tolist()
    for k in feat_kinds:
        print(f"Building train features: {k}")
        features_for(tr_seqs, k, tr_cache)
        print(f"Building test features: {k}")
        features_for(te_seqs, k, te_cache)

    y_tr = train["hc50_log_value"].to_numpy(dtype=float)
    y_te = test["hc50_log_value"].to_numpy(dtype=float)
    g_tr = train["cluster_id70"].to_numpy()
    # Strata for SGKF
    band = train.get("hc50_band", pd.Series(["unk"] * len(train))).astype(str)
    strata = np.where(exact_tr, "exact_" + band, "censored").astype(object)
    _, strata_i = np.unique(strata, return_inverse=True)

    # ---- Model selection on TRAIN exact only (never touch test) ----
    seeds = SEEDS[:2] if args.quick else SEEDS
    rows = []
    for kind in feat_kinds:
        X = tr_cache[kind]
        Xe, ye, ge, se = X[exact_tr], y_tr[exact_tr], g_tr[exact_tr], strata_i[exact_tr]
        for seed in seeds:
            for name, model in candidates(seed).items():
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                oof = np.full(len(ye), np.nan)
                try:
                    for tr_i, va_i in cv.split(Xe, se, ge):
                        m = clone(model)
                        m.fit(Xe[tr_i], ye[tr_i])
                        oof[va_i] = np.asarray(m.predict(Xe[va_i])).reshape(-1)
                    met = regression_metrics(ye, oof)
                    met.update(
                        {
                            "model": name,
                            "features": kind,
                            "seed": seed,
                            "n": int(np.isfinite(oof).sum()),
                        }
                    )
                    rows.append(met)
                    pr = met.get("pearson_r")
                    sp = met.get("spearman_rho")
                    print(
                        f"  CV {name}/{kind}/s{seed}: "
                        f"pearson={pr if pr is None else f'{pr:.3f}'} "
                        f"spearman={sp if sp is None else f'{sp:.3f}'}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {name}/{kind}/s{seed}: {exc}")

    reg = pd.DataFrame(rows)
    reg.to_csv(REPORTS / "final_v2_selection.csv", index=False)
    if reg.empty or "pearson_r" not in reg.columns:
        raise SystemExit(f"No successful CV runs. rows={len(reg)} cols={list(reg.columns)}")
    # Select by mean pearson across seeds, requiring full OOF coverage.
    full_n = reg["n"].max()
    eligible = reg[reg["n"] >= 0.9 * full_n].dropna(subset=["pearson_r"]).copy()
    summary = (
        eligible.groupby(["model", "features"])
        .agg(
            pearson=("pearson_r", "mean"),
            spearman=("spearman_rho", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values("pearson", ascending=False)
    )
    summary.to_csv(REPORTS / "final_v2_selection_summary.csv", index=False)
    best = summary.iloc[0]
    best_model, best_feat = best["model"], best["features"]
    print(f"SELECTED: {best_model}/{best_feat} (CV pearson={best['pearson']:.3f})")

    # ---- Fit 5-seed ensemble on all train exact ----
    Xtr = tr_cache[best_feat]
    Xte = te_cache[best_feat]
    models = []
    seed_oof = []
    for seed in seeds:
        model = candidates(seed)[best_model]
        model.fit(Xtr[exact_tr], y_tr[exact_tr])
        models.append(("reg", model, None))
        seed_oof.append(np.asarray(model.predict(Xtr[exact_tr])).reshape(-1))
    seed_oof = np.stack(seed_oof, axis=0)
    mu_oof = seed_oof.mean(axis=0)
    resid = np.abs(y_tr[exact_tr] - mu_oof)
    conf_q = float(conformal_interval(resid[np.isfinite(resid)], alpha=0.1))
    if not np.isfinite(conf_q):
        conf_q = float(np.std(resid[np.isfinite(resid)])) * 1.64 if np.isfinite(resid).any() else 1.0

    # ---- Classification head P(HC50 > 128) ----
    y_safe = train["y_safe_gt_128"].to_numpy(dtype=float)
    known = np.isfinite(y_safe)
    clf_X = Xtr[known]
    clf_y = y_safe[known].astype(int)
    clf_g = g_tr[known]
    clf_s = clf_y  # binary strata
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof_p = np.full(len(clf_y), np.nan)
    for tr_i, va_i in cv.split(clf_X, clf_s, clf_g):
        clf = ExtraTreesClassifier(
            n_estimators=400, max_depth=14, n_jobs=-1, random_state=0
        )
        clf.fit(clf_X[tr_i], clf_y[tr_i])
        oof_p[va_i] = clf.predict_proba(clf_X[va_i])[:, 1]
    # Calibrator selection on nested held-out
    held = {"platt": np.full(len(clf_y), np.nan), "isotonic": np.full(len(clf_y), np.nan)}
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    for tr_i, va_i in inner.split(oof_p.reshape(-1, 1), clf_y, clf_g):
        for name in held:
            cal = ProbabilityCalibrator(name).fit(clf_y[tr_i], oof_p[tr_i])
            held[name][va_i] = cal.transform(oof_p[va_i])
    # Pick lower Brier
    def brier(y, p):
        m = np.isfinite(p)
        return float(np.mean((p[m] - y[m]) ** 2))

    cal_name = min(held, key=lambda n: brier(clf_y, held[n]))
    calibrator = ProbabilityCalibrator(cal_name).fit(clf_y, oof_p)
    print(f"Calibrator selected: {cal_name}")

    clf_final = ExtraTreesClassifier(
        n_estimators=500, max_depth=14, n_jobs=-1, random_state=0
    )
    clf_final.fit(clf_X, clf_y)

    # ---- Applicability domain ----
    feats_tr = featurize_frame(tr_seqs)
    feats_te = featurize_frame(te_seqs)
    phys_tr = np.nan_to_num(
        feats_tr[[c for c in PHYSCHEM_SUMMARY_COLS if c in feats_tr.columns]].to_numpy(dtype=float)
    )
    phys_te = np.nan_to_num(
        feats_te[[c for c in PHYSCHEM_SUMMARY_COLS if c in feats_te.columns]].to_numpy(dtype=float)
    )
    print("Building ESM-2 embeddings for applicability domain...")
    emb_tr = embed_sequences(tr_seqs, model_name=ESM_MODEL, pooling="mean", batch_size=16)
    emb_te = embed_sequences(te_seqs, model_name=ESM_MODEL, pooling="mean", batch_size=16)
    ad = ApplicabilityDomain(tr_seqs, emb_tr, phys_tr)
    ood = ad.score(te_seqs, emb_te, phys_te)

    # ---- LOCKED TEST — scored once ----
    print("=== LOCKED TEST (single evaluation) ===")
    seed_preds = []
    for _, model, _ in models:
        seed_preds.append(np.asarray(model.predict(Xte)).reshape(-1))
    seed_preds = np.stack(seed_preds, axis=0)
    mu = seed_preds.mean(axis=0)
    sigma_ens = seed_preds.std(axis=0)
    sigma = np.maximum(sigma_ens, conf_q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    # Regression metrics on exact test only
    reg_met = regression_metrics(y_te[exact_te], mu[exact_te])
    print(f"Test regression (exact n={exact_te.sum()}): "
          f"pearson={reg_met.get('pearson_r'):.3f} spearman={reg_met.get('spearman_r'):.3f} "
          f"r2={reg_met.get('r2'):.3f} mae={reg_met.get('mae'):.3f}")

    # Classification
    p_raw = clf_final.predict_proba(Xte)[:, 1]
    p_cal = calibrator.transform(p_raw)
    # Also derive from regression: P(HC50>128) ≈ 1 - Φ((log128 - mu)/sigma)
    from scipy.stats import norm

    p_from_reg = 1.0 - norm.cdf((LOG128 - mu) / np.maximum(sigma, 1e-3))
    p_safe = 0.5 * p_cal + 0.5 * p_from_reg

    y_safe_te = test["y_safe_gt_128"].to_numpy(dtype=float)
    known_te = np.isfinite(y_safe_te)
    clf_met = classification_metrics(y_safe_te[known_te], p_safe[known_te])
    print(f"Test P(HC50>128) (n={known_te.sum()}): "
          f"AUC={clf_met.get('roc_auc'):.3f} PR={clf_met.get('pr_auc'):.3f} "
          f"MCC={clf_met.get('mcc'):.3f}")

    # Interval coverage on exact test
    cover = float(np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te])))
    # Right-censored consistency: prediction should be above the floor often
    right_te = test["censor_type"] == "right"
    if right_te.any():
        floors = test.loc[right_te, "censor_lower_log"].to_numpy(dtype=float)
        okf = np.isfinite(floors)
        rc_cons = float(np.mean(mu[right_te][okf] >= floors[okf] - 1e-6)) if okf.any() else float("nan")
    else:
        rc_cons = float("nan")

    # AD flags
    ad_flags = ad.flag(te_seqs, Xte)

    metrics = {
        "protocol": {
            "locked_split_seed": "see reports/split_summary.json",
            "selection": f"{best_model}/{best_feat}",
            "selection_cv_pearson": float(best["pearson"]),
            "n_train_seq": int(len(train)),
            "n_test_seq": int(len(test)),
            "n_test_exact": int(exact_te.sum()),
            "note": "Locked test scored once. Previous seed-42 test is retired.",
        },
        "regression_exact": reg_met,
        "classification_safe": clf_met,
        "interval_coverage_90": cover,
        "right_censored_consistency": rc_cons,
        "calibrator": cal_name,
        "conformal_q_90": conf_q,
    }
    (REPORTS / "final_v2_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    pred_df = test[["sequence", "censor_type", "hc50_value", "hc50_log_value",
                     "censor_lower", "y_safe_gt_128", "cluster_id70", "source"]].copy()
    pred_df["pred_log_hc50"] = mu
    pred_df["pred_hc50_uM"] = np.exp(mu)
    pred_df["pred_log_lo95"] = lo95
    pred_df["pred_log_hi95"] = hi95
    pred_df["p_hc50_gt_128"] = p_safe
    pred_df["hemolysis_risk_prob"] = 1.0 - p_safe
    pred_df["ood_max_train_identity"] = ood.max_train_identity
    pred_df["ood_embedding_score"] = ood.embedding_ood_score
    pred_df["ood_physchem_score"] = ood.physchem_ood_score
    pred_df["in_domain"] = ood.in_domain
    pred_df["prediction_confidence"] = ood.prediction_confidence
    pred_df.to_csv(REPORTS / "final_v2_test_predictions.csv", index=False)

    # Feature column list for physchem-only path
    feats0 = featurize_frame(tr_seqs[:2])
    if best_feat == "physchem":
        fcols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats0.columns]
    elif best_feat == "full":
        fcols = [c for c in feats0.columns if not c.startswith(("dpc_", "term_"))]
    else:
        fcols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats0.columns]

    bundle = {
        "models": models,
        "clf": clf_final,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "features": best_feat,
        "feature_cols": fcols,
        "embedding_model": ESM_MODEL if "esm2" in best_feat else ESM_MODEL,
        "conformal_q": conf_q,
        "ood_train_sequences": tr_seqs,
        "ood_train_embeddings": emb_tr,
        "ood_train_physchem": phys_tr,
        "physchem_cols": PHYSCHEM_SUMMARY_COLS,
        "log128": LOG128,
        "hc50_safe_threshold_uM": HC50_SAFE_THRESHOLD_UM,
        "selection": {
            "model": best_model,
            "features": best_feat,
            "cv_pearson": float(best["pearson"]),
        },
        "metrics_locked_test": metrics,
        "version": "v2_expanded",
    }
    out = FINAL_MODELS / "hc50_bundle.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)

    cfg = {
        "model": best_model,
        "features": best_feat,
        "seeds": seeds,
        "esm_model": ESM_MODEL if "esm2" in best_feat else None,
        "calibrator": cal_name,
        "version": "v2_expanded",
        "locked_test_metrics": {
            "pearson": reg_met.get("pearson_r"),
            "spearman": reg_met.get("spearman_r"),
            "roc_auc": clf_met.get("roc_auc"),
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved {out}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
