#!/usr/bin/env python
"""Improve HC50 generalization: censored Tobit + SVR fusion under group CV.

Uses ALL train observations (exact + right/left/interval) under a censored
likelihood for the Tobit head, and exact-only SVR on physchem+ESM-2. Locked
test is scored once with a stacked blend selected on OOF only.
"""

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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
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
    censored_gaussian_nll,
    classification_metrics,
    regression_metrics,
)
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.models.censored import TobitGaussian  # noqa: E402
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


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])
    print(f"train={len(train)} test={len(test)}")
    print("censor train:", train["censor_type"].value_counts().to_dict())

    tr_seqs = train["sequence"].tolist()
    te_seqs = test["sequence"].tolist()
    feats_tr = featurize_frame(tr_seqs)
    feats_te = featurize_frame(te_seqs)
    phys_cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats_tr.columns]
    phys_tr = np.nan_to_num(feats_tr[phys_cols].to_numpy(float))
    phys_te = np.nan_to_num(feats_te[phys_cols].to_numpy(float))
    print("ESM embeddings...")
    emb_tr = embed_sequences(tr_seqs, model_name=ESM, pooling="mean", batch_size=16)
    emb_te = embed_sequences(te_seqs, model_name=ESM, pooling="mean", batch_size=16)
    Xtr = np.concatenate([phys_tr, emb_tr], axis=1)
    Xte = np.concatenate([phys_te, emb_te], axis=1)

    y_log = train["hc50_log_value"].to_numpy(float)
    y_te = test["hc50_log_value"].to_numpy(float)
    censor_type = train["censor_type"].to_numpy()
    lo = train["censor_lower_log"].to_numpy(float)
    hi = train["censor_upper_log"].to_numpy(float)
    groups = train["cluster_id70"].to_numpy()
    exact = censor_type == "exact"
    exact_te = test["censor_type"] == "exact"
    strata = np.where(exact, "exact", "censored")
    _, strata_i = np.unique(strata, return_inverse=True)

    # ---- OOF: Tobit (all censoring, PCA) + SVR (exact only, full) ----
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof_tobit = np.full(len(train), np.nan)
    oof_svr = np.full(len(train), np.nan)
    oof_sig = np.full(len(train), np.nan)

    scaler_full = StandardScaler().fit(Xtr)
    Xs = scaler_full.transform(Xtr)
    pca = PCA(n_components=64, random_state=0).fit(Xs)
    Xp = pca.transform(Xs)

    print("Group CV Tobit(PCA64) + SVR...")
    for fold, (tr, va) in enumerate(cv.split(Xtr, strata_i, groups)):
        # Tobit on all train fold rows in PCA space (vectorized NLL)
        sc_f = StandardScaler().fit(Xtr[tr])
        pca_f = PCA(n_components=64, random_state=0).fit(sc_f.transform(Xtr[tr]))
        Xtr_p = pca_f.transform(sc_f.transform(Xtr[tr]))
        Xva_p = pca_f.transform(sc_f.transform(Xtr[va]))
        tb = TobitGaussian(alpha=1.0, max_iter=120)
        tb.fit(Xtr_p, y_log[tr], censor_type[tr], lo[tr], hi[tr])
        dist = tb.predict_dist(Xva_p)
        oof_tobit[va] = dist.mu
        oof_sig[va] = dist.sigma
        # SVR on exact only within fold
        tr_ex = tr[exact[tr]]
        if len(tr_ex) < 20:
            continue
        svr = Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))])
        svr.fit(Xtr[tr_ex], y_log[tr_ex])
        oof_svr[va] = np.asarray(svr.predict(Xtr[va])).reshape(-1)
        va_ex = exact[va]
        tp = regression_metrics(y_log[va][va_ex], oof_tobit[va][va_ex]).get("pearson_r")
        print(f"  fold {fold}: tobit_exact_pearson={tp}")

    # Blend weight on exact OOF pearson
    ex = exact & np.isfinite(oof_tobit) & np.isfinite(oof_svr)
    best_w, best_p = 0.5, -np.inf
    for w in np.linspace(0, 1, 11):
        blend = w * oof_svr[ex] + (1 - w) * oof_tobit[ex]
        p = regression_metrics(y_log[ex], blend).get("pearson_r") or -np.inf
        if p > best_p:
            best_p, best_w = p, float(w)
    print(f"OOF blend weight SVR={best_w:.2f} Tobit={1-best_w:.2f} pearson={best_p:.3f}")
    oof_blend = best_w * oof_svr + (1 - best_w) * oof_tobit
    # fill nan from whichever head is available
    oof_blend = np.where(np.isfinite(oof_blend), oof_blend,
                         np.where(np.isfinite(oof_svr), oof_svr, oof_tobit))

    nll = censored_gaussian_nll(
        y_log, censor_type, lo, hi, oof_tobit, np.log(np.clip(oof_sig, 1e-6, None))
    )
    print(f"Tobit OOF censored NLL={nll:.4f}")
    print(f"Tobit OOF exact pearson={regression_metrics(y_log[exact], oof_tobit[exact]).get('pearson_r')}")
    print(f"SVR   OOF exact pearson={regression_metrics(y_log[exact & np.isfinite(oof_svr)], oof_svr[exact & np.isfinite(oof_svr)]).get('pearson_r')}")
    print(f"Blend OOF exact pearson={regression_metrics(y_log[exact & np.isfinite(oof_blend)], oof_blend[exact & np.isfinite(oof_blend)]).get('pearson_r')}")

    # ---- Fit final heads on all train ----
    tb_final = TobitGaussian(alpha=1.0, max_iter=150)
    tb_final.fit(Xp, y_log, censor_type, lo, hi)
    svr_models = []
    for seed in SEEDS:
        # seed-varying subsample for mild ensemble diversity
        rng = np.random.default_rng(seed)
        idx = np.where(exact)[0]
        if seed == 0:
            take = idx
        else:
            take = rng.choice(idx, size=int(0.9 * len(idx)), replace=False)
        m = Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))])
        m.fit(Xtr[take], y_log[take])
        svr_models.append(("reg", m, None))

    resid = np.abs(y_log[exact] - oof_blend[exact])
    conf_q = float(conformal_interval(resid[np.isfinite(resid)], alpha=0.1))

    # Classifier on known safe labels
    y_safe = train["y_safe_gt_128"].to_numpy(float)
    known = np.isfinite(y_safe)
    clf_X, clf_y, clf_g = Xtr[known], y_safe[known].astype(int), groups[known]
    oof_p = np.full(len(clf_y), np.nan)
    for tr_i, va_i in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(clf_X, clf_y, clf_g):
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

    # ---- LOCKED TEST once ----
    print("=== LOCKED TEST ===")
    Xs_te = scaler_full.transform(Xte)
    Xp_te = pca.transform(Xs_te)
    mu_tb = tb_final.predict_dist(Xp_te).mu
    mu_svr = np.stack([np.asarray(m.predict(Xte)).reshape(-1) for _, m, _ in svr_models], 0).mean(0)
    mu = best_w * mu_svr + (1 - best_w) * mu_tb
    sigma = np.maximum(np.abs(mu_svr - mu_tb), conf_q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    reg_met = regression_metrics(y_te[exact_te], mu[exact_te])
    print(
        f"reg pearson={reg_met.get('pearson_r'):.3f} spearman={reg_met.get('spearman_rho'):.3f} "
        f"r2={reg_met.get('r2'):.3f} n={int(exact_te.sum())}"
    )
    # right-censored consistency
    right_te = test["censor_type"] == "right"
    if right_te.any():
        floors = test.loc[right_te, "censor_lower_log"].to_numpy(float)
        okf = np.isfinite(floors)
        rc = float(np.mean(mu[right_te][okf] >= floors[okf] - 1e-6)) if okf.any() else float("nan")
    else:
        rc = float("nan")
    cover = float(np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te])))

    p_raw = clf_final.predict_proba(Xte)[:, 1]
    p_cal = calibrator.transform(p_raw)
    p_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.maximum(sigma, 1e-3))
    p_safe = 0.5 * p_cal + 0.5 * p_reg
    y_safe_te = test["y_safe_gt_128"].to_numpy(float)
    known_te = np.isfinite(y_safe_te)
    clf_met = classification_metrics(y_safe_te[known_te], p_safe[known_te])
    print(
        f"clf AUC={clf_met.get('roc_auc'):.3f} PR={clf_met.get('pr_auc'):.3f} "
        f"MCC={clf_met.get('mcc'):.3f} rc_cons={rc:.3f} cover90={cover:.3f}"
    )

    # bootstrap CI
    yt, yp = y_te[exact_te], mu[exact_te]
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

    print("AD...")
    ad = ApplicabilityDomain(tr_seqs, emb_tr, phys_tr)
    ood = ad.score(te_seqs, emb_te, phys_te)

    metrics = {
        "protocol": {
            "selection": f"blend SVR(w={best_w:.2f})+Tobit on physchem+ESM2",
            "oof_blend_pearson": float(best_p),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_exact": int(exact_te.sum()),
            "locked_split_seed": 20260925,
            "note": "Censored Tobit uses all train rows; SVR exact-only; blend weight from OOF.",
        },
        "regression_exact": reg_met,
        "pearson_cluster_bootstrap_mean_lo_hi": pci,
        "classification_safe": clf_met,
        "interval_coverage_90": cover,
        "right_censored_consistency": rc,
        "tobit_oof_censored_nll": nll,
        "calibrator": cal_name,
        "conformal_q": conf_q,
        "blend_svr_weight": best_w,
    }
    (REPORTS / "final_v3_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    pred = test[
        ["sequence", "censor_type", "hc50_value", "hc50_log_value", "censor_lower",
         "y_safe_gt_128", "cluster_id70", "source"]
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
    pred.to_csv(REPORTS / "final_v3_test_predictions.csv", index=False)

    # Bundle: SVR ensemble + Tobit(PCA) blend for predict.py
    bundle = {
        "models": svr_models,
        "tobit": tb_final,
        "tobit_scaler": scaler_full,
        "tobit_pca": pca,
        "blend_svr_weight": best_w,
        "clf": clf_final,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "features": "physchem+esm2",
        "feature_cols": phys_cols,
        "embedding_model": ESM,
        "conformal_q": conf_q,
        "ood_train_sequences": tr_seqs,
        "ood_train_embeddings": emb_tr,
        "ood_train_physchem": phys_tr,
        "physchem_cols": PHYSCHEM_SUMMARY_COLS,
        "log128": LOG128,
        "hc50_safe_threshold_uM": HC50_SAFE_THRESHOLD_UM,
        "selection": {
            "model": f"svr+tobit_pca64_blend_w{best_w:.2f}",
            "features": "physchem+esm2",
            "cv_pearson": float(best_p),
        },
        "metrics_locked_test": metrics,
        "version": "v3_censored_fusion",
    }
    out = FINAL_MODELS / "hc50_bundle.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)
    cfg = {
        "model": bundle["selection"]["model"],
        "features": "physchem+esm2",
        "version": "v3_censored_fusion",
        "blend_svr_weight": best_w,
        "locked_test": {
            "pearson": reg_met.get("pearson_r"),
            "spearman": reg_met.get("spearman_rho"),
            "roc_auc": clf_met.get("roc_auc"),
            "right_censored_consistency": rc,
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved {out}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
