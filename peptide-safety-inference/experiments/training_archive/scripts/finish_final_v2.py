#!/usr/bin/env python
"""Continue final training from existing CV selection CSV."""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
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
from psp.evaluation import classification_metrics, regression_metrics  # noqa: E402
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

ESM = "facebook/esm2_t12_35M_UR50D"
LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))
SEEDS = [0, 1, 2, 3, 4]


def collapse(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    part["_pref"] = (part["censor_type"] != "exact").astype(int)
    return (
        part.sort_values(["sequence", "_pref"])
        .drop_duplicates("sequence", keep="first")
        .drop(columns="_pref")
        .reset_index(drop=True)
    )


def candidates(seed: int) -> dict:
    return {
        "svr": Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))]),
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "extratrees": ExtraTreesRegressor(
            n_estimators=400, max_depth=14, n_jobs=-1, random_state=seed
        ),
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])
    exact_tr = train["censor_type"] == "exact"
    exact_te = test["censor_type"] == "exact"
    print(f"train={len(train)} test={len(test)} exact_tr={exact_tr.sum()} exact_te={exact_te.sum()}")

    sel = pd.read_csv(REPORTS / "final_v2_selection.csv")
    summary = (
        sel.dropna(subset=["pearson_r"])
        .groupby(["model", "features"])
        .agg(pearson=("pearson_r", "mean"), spearman=("spearman_rho", "mean"), r2=("r2", "mean"))
        .reset_index()
        .sort_values("pearson", ascending=False)
    )
    summary.to_csv(REPORTS / "final_v2_selection_summary.csv", index=False)
    best = summary.iloc[0]
    best_model, best_feat = best["model"], best["features"]
    print(f"SELECTED: {best_model}/{best_feat} CV pearson={best['pearson']:.3f}")

    tr_seqs = train["sequence"].tolist()
    te_seqs = test["sequence"].tolist()
    print("physchem features...")
    feats_tr = featurize_frame(tr_seqs)
    feats_te = featurize_frame(te_seqs)
    phys_cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats_tr.columns]
    phys_tr = np.nan_to_num(feats_tr[phys_cols].to_numpy(dtype=float))
    phys_te = np.nan_to_num(feats_te[phys_cols].to_numpy(dtype=float))
    print("ESM-2 embeddings...")
    emb_tr = embed_sequences(tr_seqs, model_name=ESM, pooling="mean", batch_size=16)
    emb_te = embed_sequences(te_seqs, model_name=ESM, pooling="mean", batch_size=16)

    if best_feat == "physchem":
        Xtr, Xte, fcols = phys_tr, phys_te, phys_cols
    elif best_feat == "full":
        cols = [c for c in feats_tr.columns if not c.startswith(("dpc_", "term_"))]
        Xtr = np.nan_to_num(feats_tr[cols].to_numpy(dtype=float))
        Xte = np.nan_to_num(feats_te[cols].to_numpy(dtype=float))
        fcols = cols
    else:
        Xtr = np.concatenate([phys_tr, emb_tr], axis=1)
        Xte = np.concatenate([phys_te, emb_te], axis=1)
        fcols = phys_cols

    y_tr = train["hc50_log_value"].to_numpy(dtype=float)
    y_te = test["hc50_log_value"].to_numpy(dtype=float)
    g_tr = train["cluster_id70"].to_numpy()

    models = []
    seed_oof = []
    for seed in SEEDS:
        m = candidates(seed)[best_model]
        m.fit(Xtr[exact_tr], y_tr[exact_tr])
        models.append(("reg", m, None))
        seed_oof.append(np.asarray(m.predict(Xtr[exact_tr])).reshape(-1))
    seed_oof = np.stack(seed_oof, axis=0)
    mu_oof = seed_oof.mean(axis=0)
    resid = np.abs(y_tr[exact_tr] - mu_oof)
    conf_q = float(conformal_interval(resid[np.isfinite(resid)], alpha=0.1))
    print(f"conformal_q={conf_q:.3f}")

    y_safe = train["y_safe_gt_128"].to_numpy(dtype=float)
    known = np.isfinite(y_safe)
    clf_X, clf_y, clf_g = Xtr[known], y_safe[known].astype(int), g_tr[known]
    oof_p = np.full(len(clf_y), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr_i, va_i in cv.split(clf_X, clf_y, clf_g):
        c = ExtraTreesClassifier(n_estimators=400, max_depth=14, n_jobs=-1, random_state=0)
        c.fit(clf_X[tr_i], clf_y[tr_i])
        oof_p[va_i] = c.predict_proba(clf_X[va_i])[:, 1]

    held = {"platt": np.full(len(clf_y), np.nan), "isotonic": np.full(len(clf_y), np.nan)}
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    for tr_i, va_i in inner.split(oof_p.reshape(-1, 1), clf_y, clf_g):
        for name in held:
            cal = ProbabilityCalibrator(name).fit(clf_y[tr_i], oof_p[tr_i])
            held[name][va_i] = cal.transform(oof_p[va_i])

    def brier(y, p):
        m = np.isfinite(p)
        return float(np.mean((p[m] - y[m]) ** 2))

    cal_name = min(held, key=lambda n: brier(clf_y, held[n]))
    calibrator = ProbabilityCalibrator(cal_name).fit(clf_y, oof_p)
    print("calibrator", cal_name)
    clf_final = ExtraTreesClassifier(n_estimators=500, max_depth=14, n_jobs=-1, random_state=0)
    clf_final.fit(clf_X, clf_y)

    print("AD scoring (this can take a few minutes)...")
    ad = ApplicabilityDomain(tr_seqs, emb_tr, phys_tr)
    ood = ad.score(te_seqs, emb_te, phys_te)

    print("=== LOCKED TEST ===")
    seed_preds = np.stack([np.asarray(m.predict(Xte)).reshape(-1) for _, m, _ in models], 0)
    mu = seed_preds.mean(0)
    sigma = np.maximum(seed_preds.std(0), conf_q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)
    reg_met = regression_metrics(y_te[exact_te], mu[exact_te])
    print(
        f"reg pearson={reg_met.get('pearson_r'):.3f} "
        f"spearman={reg_met.get('spearman_rho'):.3f} "
        f"r2={reg_met.get('r2'):.3f} n={int(exact_te.sum())}"
    )

    p_raw = clf_final.predict_proba(Xte)[:, 1]
    p_cal = calibrator.transform(p_raw)
    p_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.maximum(sigma, 1e-3))
    p_safe = 0.5 * p_cal + 0.5 * p_reg
    y_safe_te = test["y_safe_gt_128"].to_numpy(dtype=float)
    known_te = np.isfinite(y_safe_te)
    clf_met = classification_metrics(y_safe_te[known_te], p_safe[known_te])
    print(
        f"clf AUC={clf_met.get('roc_auc'):.3f} PR={clf_met.get('pr_auc'):.3f} "
        f"MCC={clf_met.get('mcc'):.3f} n={int(known_te.sum())}"
    )

    cover = float(
        np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te]))
    )
    right_te = test["censor_type"] == "right"
    if right_te.any():
        floors = test.loc[right_te, "censor_lower_log"].to_numpy(dtype=float)
        okf = np.isfinite(floors)
        rc = (
            float(np.mean(mu[right_te][okf] >= floors[okf] - 1e-6))
            if okf.any()
            else float("nan")
        )
    else:
        rc = float("nan")

    yt = y_te[exact_te]
    yp = mu[exact_te]
    clus = test.loc[exact_te, "cluster_id70"].to_numpy()
    rng = np.random.default_rng(0)
    uniq = np.unique(clus)
    boots = []
    for _ in range(500):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(clus, sample)
        if mask.sum() < 5 or np.std(yt[mask]) < 1e-9 or np.std(yp[mask]) < 1e-9:
            continue
        boots.append(float(stats.pearsonr(yt[mask], yp[mask])[0]))
    pci = (
        (float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        if boots
        else (reg_met.get("pearson_r"), None, None)
    )

    metrics = {
        "protocol": {
            "selection": f"{best_model}/{best_feat}",
            "cv_pearson": float(best["pearson"]),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_exact": int(exact_te.sum()),
            "locked_split_seed": 20260925,
        },
        "regression_exact": reg_met,
        "pearson_cluster_bootstrap_mean_lo_hi": pci,
        "classification_safe": clf_met,
        "interval_coverage_90": cover,
        "right_censored_consistency": rc,
        "calibrator": cal_name,
        "conformal_q": conf_q,
    }
    (REPORTS / "final_v2_test_metrics.json").write_text(
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
    pred.to_csv(REPORTS / "final_v2_test_predictions.csv", index=False)

    bundle = {
        "models": models,
        "clf": clf_final,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "features": best_feat,
        "feature_cols": fcols,
        "embedding_model": ESM,
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
        "seeds": SEEDS,
        "version": "v2_expanded",
        "locked_test": {
            "pearson": reg_met.get("pearson_r"),
            "spearman": reg_met.get("spearman_rho"),
            "roc_auc": clf_met.get("roc_auc"),
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved {out}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
