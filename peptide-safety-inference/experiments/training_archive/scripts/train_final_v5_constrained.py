#!/usr/bin/env python
"""HC50 v5: CatBoost/ET blend with Tobit floor for right-censor consistency.

Selection rule (OOF only, fixed a priori):
  maximise exact pearson among blends with Tobit weight >= 0.15 AND
  OOF right-censored consistency >= 0.50.
This addresses the known failure mode where exact-only heads under-predict
right-censored peptides (HC50 floors). Locked test scored once.
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
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

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
MIN_TOBIT_W = 0.15
MIN_RC = 0.50


def collapse(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    part["_pref"] = (part["censor_type"] != "exact").astype(int)
    return (
        part.sort_values(["sequence", "_pref"])
        .drop_duplicates("sequence", keep="first")
        .drop(columns="_pref")
        .reset_index(drop=True)
    )


def rc_consistency(pred, censor_type, lo):
    right = censor_type == "right"
    if not right.any():
        return float("nan")
    floors = lo[right]
    ok = np.isfinite(floors) & np.isfinite(pred[right])
    if not ok.any():
        return float("nan")
    return float(np.mean(pred[right][ok] >= floors[ok] - 1e-6))


def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = collapse(df[df["split"] == "train"])
    test = collapse(df[df["split"] == "test"])
    print(f"train={len(train)} test={len(test)}")

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

    oof_cb = np.full(len(train), np.nan)
    oof_et = np.full(len(train), np.nan)
    oof_tb = np.full(len(train), np.nan)
    oof_sig = np.full(len(train), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    print("Group CV CatBoost + ExtraTrees + Tobit...")
    for fold, (tr, va) in enumerate(cv.split(Xtr, strata_i, groups)):
        tr_ex = tr[exact[tr]]
        cb = CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            random_seed=fold, verbose=False, allow_writing_files=False,
        )
        et = ExtraTreesRegressor(
            n_estimators=600, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=fold
        )
        cb.fit(Xtr[tr_ex], y_log[tr_ex])
        et.fit(Xtr[tr_ex], y_log[tr_ex])
        oof_cb[va] = np.asarray(cb.predict(Xtr[va])).reshape(-1)
        oof_et[va] = np.asarray(et.predict(Xtr[va])).reshape(-1)

        sc = StandardScaler().fit(Xtr[tr])
        pca = PCA(n_components=64, random_state=0).fit(sc.transform(Xtr[tr]))
        tb = TobitGaussian(alpha=1.0, max_iter=120)
        tb.fit(pca.transform(sc.transform(Xtr[tr])), y_log[tr], censor_type[tr], lo[tr], hi[tr])
        dist = tb.predict_dist(pca.transform(sc.transform(Xtr[va])))
        oof_tb[va] = dist.mu
        oof_sig[va] = dist.sigma
        print(f"  fold {fold} done")

    for label, arr in (("catboost", oof_cb), ("extratrees", oof_et), ("tobit", oof_tb)):
        m = exact & np.isfinite(arr)
        met = regression_metrics(y_log[m], arr[m])
        print(f"  OOF {label}: pearson={met.get('pearson_r'):.3f} rc={rc_consistency(arr, censor_type, lo):.3f}")

    # Constrained blend grid
    best = None  # (pearson, a, b, wt, rc)
    fallback = None
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, 1 - a, 21):
            wt = 1 - a - b
            if wt + 1e-9 < MIN_TOBIT_W:
                continue
            blend = a * oof_cb + b * oof_et + wt * oof_tb
            msk = exact & np.isfinite(blend)
            p = regression_metrics(y_log[msk], blend[msk]).get("pearson_r") or -np.inf
            rc = rc_consistency(blend, censor_type, lo)
            cand = (float(p), float(a), float(b), float(wt), float(rc))
            if fallback is None or p > fallback[0]:
                fallback = cand
            if rc >= MIN_RC and (best is None or p > best[0]):
                best = cand
    if best is None:
        print("WARNING: no blend met RC constraint; using best pearson with Tobit floor")
        best = fallback
    print(
        f"OOF blend: cb={best[1]:.2f} et={best[2]:.2f} tobit={best[3]:.2f} "
        f"pearson={best[0]:.3f} rc={best[4]:.3f}"
    )
    oof_blend = best[1] * oof_cb + best[2] * oof_et + best[3] * oof_tb
    nll = censored_gaussian_nll(
        y_log, censor_type, lo, hi, oof_tb, np.log(np.clip(oof_sig, 1e-6, None))
    )

    # Fit finals
    cb_f = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        random_seed=0, verbose=False, allow_writing_files=False,
    )
    et_f = ExtraTreesRegressor(
        n_estimators=600, max_depth=16, min_samples_leaf=3, n_jobs=-1, random_state=0
    )
    cb_f.fit(Xtr[exact], y_log[exact])
    et_f.fit(Xtr[exact], y_log[exact])
    sc_full = StandardScaler().fit(Xtr)
    pca_full = PCA(n_components=64, random_state=0).fit(sc_full.transform(Xtr))
    tb_f = TobitGaussian(alpha=1.0, max_iter=150)
    tb_f.fit(pca_full.transform(sc_full.transform(Xtr)), y_log, censor_type, lo, hi)

    resid = np.abs(y_log[exact] - oof_blend[exact])
    conf_q = float(conformal_interval(resid[np.isfinite(resid)], alpha=0.1))

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

    print("=== LOCKED TEST ===")
    mu_cb = np.asarray(cb_f.predict(Xte)).reshape(-1)
    mu_et = np.asarray(et_f.predict(Xte)).reshape(-1)
    mu_tb = tb_f.predict(pca_full.transform(sc_full.transform(Xte)))
    mu = best[1] * mu_cb + best[2] * mu_et + best[3] * mu_tb
    sigma_ens = np.std(np.stack([mu_cb, mu_et, mu_tb], 0), 0)
    sigma = np.maximum(sigma_ens, conf_q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    reg_met = regression_metrics(y_te[exact_te], mu[exact_te])
    rc = rc_consistency(mu, test["censor_type"].to_numpy(), test["censor_lower_log"].to_numpy(float))
    cover = float(np.mean((y_te[exact_te] >= lo95[exact_te]) & (y_te[exact_te] <= hi95[exact_te])))
    print(
        f"reg pearson={reg_met.get('pearson_r'):.3f} spearman={reg_met.get('spearman_rho'):.3f} "
        f"r2={reg_met.get('r2'):.3f} n={int(exact_te.sum())}"
    )

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
    # Embedding/physchem AD only at train-time (identity vs 6k train is too slow).
    # Full identity AD runs at inference in predict.py.
    ood = ad.score(te_seqs, emb_te, phys_te, identity=False)

    metrics = {
        "protocol": {
            "selection": (
                f"blend catboost={best[1]:.2f}+extratrees={best[2]:.2f}+tobit={best[3]:.2f}; "
                f"constraints tobit>={MIN_TOBIT_W} OOF_rc>={MIN_RC}"
            ),
            "oof_blend_pearson": best[0],
            "oof_rc_consistency": best[4],
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
        "tobit_oof_censored_nll": nll,
        "calibrator": cal_name,
        "conformal_q": conf_q,
        "blend_weights": {"catboost": best[1], "extratrees": best[2], "tobit": best[3]},
    }
    (REPORTS / "final_v5_test_metrics.json").write_text(
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
    pred.to_csv(REPORTS / "final_v5_test_predictions.csv", index=False)

    bundle = {
        "models": [("reg", cb_f, None), ("reg", et_f, None), ("reg", cb_f, None)],
        "blend_members": {"catboost": cb_f, "extratrees": et_f},
        "blend_weights": {"catboost": best[1], "extratrees": best[2], "tobit": best[3]},
        "tobit": tb_f,
        "tobit_scaler": sc_full,
        "tobit_pca": pca_full,
        "blend_svr_weight": None,
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
            "model": metrics["protocol"]["selection"],
            "features": "physchem+esm2",
            "cv_pearson": best[0],
        },
        "metrics_locked_test": metrics,
        "version": "v5_constrained_censored_fusion",
    }
    out = FINAL_MODELS / "hc50_bundle.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)
    cfg = {
        "model": bundle["selection"]["model"],
        "features": "physchem+esm2",
        "version": "v5_constrained_censored_fusion",
        "blend_weights": bundle["blend_weights"],
        "locked_test": {
            "pearson": reg_met.get("pearson_r"),
            "spearman": reg_met.get("spearman_rho"),
            "roc_auc": clf_met.get("roc_auc"),
            "right_censored_consistency": rc,
            "coverage_90": cover,
        },
        "aqueous_solubility": {
            "source": "SolPepBench / PepSol2000",
            "default_model": "solubility_aqueous_pooled.pkl",
            "note": "NOT E. coli expression",
        },
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Saved {out}")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
