#!/usr/bin/env python
"""Freeze final config, evaluate once on locked test, save artifacts."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.calibration import ProbabilityCalibrator  # noqa: E402
from psp.embeddings import DEFAULT_MODEL, embed_sequences  # noqa: E402
from psp.evaluation import (  # noqa: E402
    classification_metrics,
    cluster_bootstrap_ci,
    expected_calibration_error,
    regression_metrics,
)
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.models.censored import TobitGaussian  # noqa: E402
from psp.ood import ApplicabilityDomain  # noqa: E402
from psp.paths import (  # noqa: E402
    CONFIGS,
    EMBEDDINGS,
    HC50_SAFE_THRESHOLD_UM,
    MODELS,
    PROCESSED,
    REPORTS,
    ensure_dirs,
)
from psp.uncertainty import conformal_interval, gaussian_interval  # noqa: E402

LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    sub = df[df["split"] == split].copy()
    sub["_pref"] = (sub["censor_type"] != "exact").astype(int)
    return sub.sort_values(["sequence", "_pref"]).groupby("sequence", as_index=False).first()


def pick_best_from_experiments() -> dict:
    path = REPORTS / "all_experiments.csv"
    if not path.exists():
        return {
            "regressor": "extratrees",
            "features": "physchem",
            "family": "classical",
            "reason": "no experiments yet; default ExtraTrees/physchem",
        }
    exp = pd.read_csv(path)
    if "pearson_r" not in exp.columns and "reg_pearson_r" in exp.columns:
        exp["pearson_r"] = exp["reg_pearson_r"]
    elif "reg_pearson_r" in exp.columns:
        exp["pearson_r"] = exp["pearson_r"].fillna(exp["reg_pearson_r"])
    families = ["classical", "frozen_plm", "hybrid", "censored", "deep", "finetune"]
    reg = exp[exp["family"].isin(families)].copy()
    # Censored runs report their regression sample size under `reg_n`.
    if "reg_n" in reg.columns:
        reg["n"] = reg["n"].fillna(reg["reg_n"]) if "n" in reg.columns else reg["reg_n"]

    # Only models scored under the same protocol may be compared. The classical,
    # PLM and deep runs produce out-of-fold predictions covering every training
    # sequence; the ESM-2 fine-tune was scored on a single grouped holdout
    # because 5-fold fine-tuning was outside the compute budget. Its smaller `n`
    # makes its Pearson neither better nor worse, just not comparable, so it is
    # reported in the comparison table but excluded from selection.
    if "n" in reg.columns and reg["n"].notna().any():
        full_n = reg["n"].max()
        eligible = reg["n"].notna() & (reg["n"] >= 0.9 * full_n)
        reg["eval_protocol"] = np.where(eligible, "groupcv_oof_full", "partial_holdout")
        excluded = reg.loc[~eligible, ["family", "model", "features", "n", "pearson_r"]]
        if len(excluded):
            excluded.to_csv(REPORTS / "model_selection_excluded.csv", index=False)
            print(
                f"Excluded from selection (non-comparable protocol): "
                f"{sorted(set(excluded['model']))}"
            )
        reg = reg[eligible]

    if "pearson_r" not in reg.columns or reg["pearson_r"].dropna().empty:
        return {
            "regressor": "extratrees",
            "features": "physchem",
            "family": "classical",
            "reason": "experiments lacked pearson; default ExtraTrees",
        }
    # Aggregate across seeds
    g = (
        reg.dropna(subset=["pearson_r"])
        .groupby(["family", "model", "features"], dropna=False)["pearson_r"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    g.to_csv(REPORTS / "model_selection_validation.csv", index=False)
    top = g.iloc[0]
    return {
        "regressor": str(top["model"]),
        "features": str(top["features"]),
        "family": str(top["family"]),
        "val_pearson_mean": float(top["mean"]),
        "val_pearson_std": float(top["std"]) if pd.notna(top["std"]) else 0.0,
        "n_seeds": int(top["count"]),
        "reason": "highest mean validation Pearson r among group-disjoint CV runs",
    }


def build_X(sequences, features: str, emb_cache=None):
    feats = featurize_frame(sequences)
    phys_cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    Xp = np.nan_to_num(feats[phys_cols].to_numpy(dtype=float))
    if features == "physchem":
        return Xp, phys_cols, feats
    if features == "aac":
        cols = [c for c in feats.columns if c.startswith("aac_")]
        return np.nan_to_num(feats[cols].to_numpy(dtype=float)), cols, feats
    if features == "full":
        cols = [c for c in feats.columns if not c.startswith("dpc_") and not c.startswith("term_")]
        return np.nan_to_num(feats[cols].to_numpy(dtype=float)), cols, feats
    if "esm2" in features:
        if emb_cache is None:
            emb = embed_sequences(sequences, model_name=DEFAULT_MODEL, pooling="mean", batch_size=8)
        else:
            emb = emb_cache
        if features.startswith("physchem+"):
            return np.concatenate([Xp, emb], axis=1), phys_cols + [f"emb_{i}" for i in range(emb.shape[1])], feats
        return emb, [f"emb_{i}" for i in range(emb.shape[1])], feats
    return Xp, phys_cols, feats


def make_regressor(name: str, seed: int = 0):
    """Rebuild a candidate by name, matching train_baselines.py exactly.

    Any name added to the search in train_baselines.py must be constructible
    here with identical hyperparameters, otherwise the frozen configuration and
    the model actually fitted would silently disagree.
    """
    name = name.lower()
    if name == "svr":
        from sklearn.svm import SVR

        return Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))])
    if name in {"extratrees", "et"}:
        return ExtraTreesRegressor(n_estimators=500, max_depth=14, n_jobs=-1, random_state=seed)
    if name in {"rf", "randomforest"}:
        return RandomForestRegressor(n_estimators=500, max_depth=14, n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05, random_state=seed, verbose=-1
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05, random_seed=seed, verbose=0)
    if name == "tobit_gaussian":
        return "tobit"
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor

        return Pipeline(
            [
                ("sc", StandardScaler()),
                ("m", MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=800, random_state=seed)),
            ]
        )
    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    # Falling back to an arbitrary model here would mean the frozen config names
    # one estimator while a different one is actually fitted and shipped.
    raise ValueError(
        f"Selected regressor {name!r} has no constructor in evaluate.py. "
        "Add it here with the same hyperparameters used in train_baselines.py."
    )


def main():
    ensure_dirs()
    train = load_split("train")
    test = load_split("test")
    print(f"Train={len(train)} Test={len(test)}")

    selection = pick_best_from_experiments()
    print("Selected:", selection)

    # Embeddings if needed
    emb_train = emb_test = None
    if "esm2" in selection["features"]:
        emb_train = embed_sequences(train["sequence"].tolist(), pooling="mean", batch_size=8)
        emb_test = embed_sequences(test["sequence"].tolist(), pooling="mean", batch_size=8)

    Xtr, cols, feats_tr = build_X(train["sequence"].tolist(), selection["features"], emb_train)
    Xte, _, feats_te = build_X(test["sequence"].tolist(), selection["features"], emb_test)

    exact_tr = train["censor_type"] == "exact"
    ytr = train["hc50_log_value"].to_numpy(dtype=float)

    # --- Fit ensemble of 5 seeds on exact ---
    seed_preds = []
    models = []
    for seed in range(5):
        reg = make_regressor(selection["regressor"], seed)
        if reg == "tobit":
            # fit Tobit on all (censored-aware)
            scaler = StandardScaler().fit(Xtr)
            Xs = scaler.transform(Xtr)
            lo = np.where(
                train["censor_lower"].notna() & (train["censor_lower"] > 0),
                np.log(train["censor_lower"].astype(float)),
                np.nan,
            )
            hi = np.where(
                train["censor_upper"].notna() & (train["censor_upper"] > 0),
                np.log(train["censor_upper"].astype(float)),
                np.nan,
            )
            m = TobitGaussian(alpha=1.0, max_iter=200).fit(
                Xs, ytr, train["censor_type"].to_numpy(), lo, hi
            )
            models.append(("tobit", m, scaler))
            seed_preds.append(m.predict(scaler.transform(Xte)))
        else:
            m = reg
            m.fit(Xtr[exact_tr], ytr[exact_tr])
            models.append(("tree", m, None))
            seed_preds.append(np.asarray(m.predict(Xte)).reshape(-1))
    seed_preds = np.stack(seed_preds, axis=0)
    mu = seed_preds.mean(axis=0)
    sigma_ens = seed_preds.std(axis=0)

    # Conformal on OOF residuals within train
    # Quick 5-fold OOF on train exact
    oof = np.full(exact_tr.sum(), np.nan)
    Xex, yex = Xtr[exact_tr], ytr[exact_tr]
    gex = train.loc[exact_tr, "cluster_id70"].to_numpy()
    sex = np.zeros(len(yex), dtype=int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    # strata by HC50 band
    bands = pd.qcut(yex, q=min(4, max(2, len(yex) // 20)), duplicates="drop").codes
    for tr_i, va_i in cv.split(Xex, bands, gex):
        m = make_regressor(selection["regressor"], 0)
        if m == "tobit":
            # Tobit needs the censored columns, so refit it the same way as above
            # on the fold's training rows rather than skipping the fold.
            sc = StandardScaler().fit(Xex[tr_i])
            tb = TobitGaussian(alpha=1.0, max_iter=200).fit(
                sc.transform(Xex[tr_i]),
                yex[tr_i],
                np.array(["exact"] * len(tr_i)),
                np.full(len(tr_i), np.nan),
                np.full(len(tr_i), np.nan),
            )
            oof[va_i] = tb.predict(sc.transform(Xex[va_i]))
            continue
        m.fit(Xex[tr_i], yex[tr_i])
        oof[va_i] = m.predict(Xex[va_i])
    resid = yex - oof
    q = conformal_interval(resid[np.isfinite(resid)], alpha=0.05)
    if not np.isfinite(q):
        finite = resid[np.isfinite(resid)]
        q = float(np.std(finite)) * 1.96 if finite.size else 1.96
    # Combine ensemble std with conformal
    sigma = np.maximum(sigma_ens, q / 1.96)
    lo95, hi95 = gaussian_interval(mu, sigma, 0.05)

    # --- Psafe classification head ---
    known = train["y_safe_gt_128"].notna()
    Xc = Xtr[known]
    yc = train.loc[known, "y_safe_gt_128"].to_numpy(dtype=int)
    gc = train.loc[known, "cluster_id70"].to_numpy()
    clf = RandomForestClassifier(n_estimators=500, max_depth=12, n_jobs=-1, random_state=0)
    # OOF for calibration
    oof_p = np.full(len(yc), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr_i, va_i in cv.split(Xc, yc, gc):
        c = RandomForestClassifier(n_estimators=400, max_depth=12, n_jobs=-1, random_state=0)
        c.fit(Xc[tr_i], yc[tr_i])
        oof_p[va_i] = c.predict_proba(Xc[va_i])[:, 1]
    # Choose the calibrator on *held-out* folds. Scoring a calibrator on the
    # same predictions it was fitted to always favours isotonic regression,
    # which can drive in-sample ECE to ~0 by interpolating the training points.
    # A group-aware inner split keeps the comparison meaningful; the winner is
    # then refitted on all out-of-fold predictions.
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    held = {"platt": np.full(len(yc), np.nan), "isotonic": np.full(len(yc), np.nan)}
    for tr_i, va_i in inner.split(oof_p.reshape(-1, 1), yc, gc):
        for name in held:
            cal = ProbabilityCalibrator(name).fit(yc[tr_i], oof_p[tr_i])
            held[name][va_i] = cal.transform(oof_p[va_i])
    ece_platt = expected_calibration_error(yc, held["platt"])
    ece_iso = expected_calibration_error(yc, held["isotonic"])
    cal_name = "isotonic" if (np.isnan(ece_platt) or ece_iso <= ece_platt) else "platt"
    calibrator = ProbabilityCalibrator(cal_name).fit(yc, oof_p)
    clf.fit(Xc, yc)
    p_raw = clf.predict_proba(Xte)[:, 1]
    p_safe = calibrator.transform(p_raw)

    # Also derive Psafe from regression distribution
    p_safe_reg = 1.0 - stats.norm.cdf((LOG128 - mu) / np.clip(sigma, 1e-6, None))
    # Blend
    p_safe_final = 0.5 * p_safe + 0.5 * p_safe_reg

    # --- OOD ---
    phys_tr = np.nan_to_num(feats_tr[PHYSCHEM_SUMMARY_COLS].to_numpy(dtype=float))
    phys_te = np.nan_to_num(feats_te[PHYSCHEM_SUMMARY_COLS].to_numpy(dtype=float))
    if emb_train is None:
        emb_train = embed_sequences(train["sequence"].tolist(), pooling="mean", batch_size=8)
        emb_test = embed_sequences(test["sequence"].tolist(), pooling="mean", batch_size=8)
    ood = ApplicabilityDomain(
        train["sequence"].tolist(), emb_train, phys_tr
    ).score(test["sequence"].tolist(), emb_test, phys_te)

    # --- Locked test metrics (ONCE) ---
    exact_te = test["censor_type"] == "exact"
    reg_met = regression_metrics(
        test.loc[exact_te, "hc50_log_value"].to_numpy(dtype=float),
        mu[exact_te.to_numpy()],
    )
    known_te = test["y_safe_gt_128"].notna()
    clf_met = classification_metrics(
        test.loc[known_te, "y_safe_gt_128"].to_numpy(dtype=float),
        p_safe_final[known_te.to_numpy()],
    )
    clf_met["ece"] = expected_calibration_error(
        test.loc[known_te, "y_safe_gt_128"].to_numpy(dtype=float),
        p_safe_final[known_te.to_numpy()],
    )

    yt = test.loc[exact_te, "hc50_log_value"].to_numpy(dtype=float)
    yp = mu[exact_te.to_numpy()]
    clus = test.loc[exact_te, "cluster_id70"].to_numpy()

    # Cluster-level bootstrap for the Pearson CI: resample whole clusters so the
    # CI reflects between-cluster variability rather than treating near-identical
    # peptides as independent observations.
    rng = np.random.default_rng(0)
    uniq = np.unique(clus)
    boots = []
    for _ in range(500):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(clus, sample)
        if mask.sum() < 5:
            continue
        if np.std(yt[mask]) < 1e-9 or np.std(yp[mask]) < 1e-9:
            continue
        boots.append(float(stats.pearsonr(yt[mask], yp[mask])[0]))
    if boots:
        pearson_ci = (float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        pearson_ci = (reg_met.get("pearson_r"), float("nan"), float("nan"))

    # Empirical coverage of the nominal 95% interval, on exact observations only.
    lo_e, hi_e = lo95[exact_te.to_numpy()], hi95[exact_te.to_numpy()]
    coverage = float(np.mean((yt >= lo_e) & (yt <= hi_e))) if yt.size else float("nan")

    # Right-censored rows: the interval is "correct" if it admits values above
    # the censoring bound, since the true HC50 is only known to exceed it.
    right_te = (test["censor_type"] == "right").to_numpy()
    if right_te.sum():
        bound = np.log(test.loc[right_te, "censor_lower"].astype(float).to_numpy())
        censored_ok = float(np.mean(hi95[right_te] >= bound))
    else:
        censored_ok = float("nan")

    # Residuals by subgroup, for the failure-mode section of the report.
    sub = pd.DataFrame(
        {
            "residual": yt - yp,
            "abs_residual": np.abs(yt - yp),
            "length": test.loc[exact_te, "sequence"].str.len().to_numpy(),
            "max_train_identity": np.asarray(ood.max_train_identity)[exact_te.to_numpy()],
            "in_domain": np.asarray(ood.in_domain)[exact_te.to_numpy()],
            "hc50_band": pd.cut(np.exp(yt), [0, 32, 128, 512, np.inf],
                                labels=["<32", "32-128", "128-512", ">512"]),
        }
    )
    subgroups = {
        "by_length_bin": sub.groupby(pd.cut(sub["length"], [0, 15, 25, 40, 1000]),
                                     observed=True)["abs_residual"].agg(["mean", "count"]),
        "by_identity_bin": sub.groupby(pd.cut(sub["max_train_identity"], [0, 0.3, 0.5, 0.7]),
                                       observed=True)["abs_residual"].agg(["mean", "count"]),
        "by_hc50_band": sub.groupby("hc50_band", observed=True)["abs_residual"].agg(["mean", "count"]),
        "by_in_domain": sub.groupby("in_domain", observed=True)["abs_residual"].agg(["mean", "count"]),
    }
    subgroup_json = {
        k: {str(i): {"mae": float(r["mean"]), "n": int(r["count"])} for i, r in v.iterrows()}
        for k, v in subgroups.items()
    }
    sub.to_csv(REPORTS / "final_test_residual_analysis.csv", index=False)

    metrics = {
        "selection": selection,
        "calibrator": cal_name,
        "calibrator_ece_val": {"platt": ece_platt, "isotonic": ece_iso},
        "regression_exact": reg_met,
        "interval_coverage_95_exact": coverage,
        "interval_consistent_right_censored": censored_ok,
        "residuals_by_subgroup": subgroup_json,
        "pearson_cluster_bootstrap_95ci": {
            "mean": pearson_ci[0],
            "lo": pearson_ci[1],
            "hi": pearson_ci[2],
        },
        "psafe": clf_met,
        "conformal_q_95": q,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_test_exact": int(exact_te.sum()),
    }
    (REPORTS / "final_test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    # Predictions table
    pred_df = test.copy()
    pred_df["hc50_log_uM_pred"] = mu
    pred_df["hc50_uM_pred"] = np.exp(mu)
    pred_df["hc50_log_lower_95"] = lo95
    pred_df["hc50_log_upper_95"] = hi95
    pred_df["hc50_lower_95_uM"] = np.exp(lo95)
    pred_df["hc50_upper_95_uM"] = np.exp(hi95)
    pred_df["p_hc50_gt_128"] = p_safe_final
    pred_df["hemolysis_risk_prob"] = 1.0 - p_safe_final
    pred_df["hc50_uncertainty"] = sigma
    pred_df["max_train_identity"] = ood.max_train_identity
    pred_df["embedding_ood_score"] = ood.embedding_ood_score
    pred_df["physchem_ood_score"] = ood.physchem_ood_score
    pred_df["in_domain"] = ood.in_domain
    pred_df["prediction_confidence"] = ood.prediction_confidence
    pred_df.to_csv(REPORTS / "final_test_predictions.csv", index=False)

    # Plots
    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if exact_te.sum() >= 5:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yt, yp, alpha=0.6, s=20)
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Observed log HC50")
        ax.set_ylabel("Predicted log HC50")
        ax.set_title(f"Locked test Pearson={reg_met.get('pearson_r', float('nan')):.3f}")
        fig.tight_layout()
        fig.savefig(fig_dir / "pred_vs_obs.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(yt - yp, bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("Residual (log HC50)")
        ax.set_title("Residuals")
        fig.tight_layout()
        fig.savefig(fig_dir / "residuals.png", dpi=120)
        plt.close(fig)

    if known_te.sum() >= 10:
        # reliability
        yt_b = test.loc[known_te, "y_safe_gt_128"].to_numpy(dtype=float)
        yp_b = p_safe_final[known_te.to_numpy()]
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (yp_b >= bins[i]) & (yp_b < bins[i + 1] if i < 9 else yp_b <= bins[i + 1])
            if m.sum() == 0:
                continue
            xs.append(yp_b[m].mean())
            ys.append(yt_b[m].mean())
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--")
        ax.plot(xs, ys, "o-", color="crimson")
        ax.set_xlabel("Predicted P(HC50>128)")
        ax.set_ylabel("Empirical frequency")
        ax.set_title(f"Reliability (ECE={clf_met.get('ece', float('nan')):.3f})")
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=120)
        plt.close(fig)

    # Save final bundle
    bundle = {
        "selection": selection,
        "feature_cols": cols,
        "features": selection["features"],
        "models": models,
        "clf": clf,
        "calibrator": calibrator,
        "calibrator_name": cal_name,
        "conformal_q": q,
        "ood_train_sequences": train["sequence"].tolist(),
        "ood_train_embeddings": emb_train,
        "ood_train_physchem": phys_tr,
        "physchem_cols": PHYSCHEM_SUMMARY_COLS,
        "log128": LOG128,
        "embedding_model": DEFAULT_MODEL,
    }
    with (MODELS / "final" / "hc50_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    # Freeze config
    cfg = {
        "selection": selection,
        "calibrator": cal_name,
        "embedding_model": DEFAULT_MODEL,
        "hc50_safe_threshold_uM": HC50_SAFE_THRESHOLD_UM,
        "n_seeds": 5,
        "locked_test_evaluated": True,
    }
    (CONFIGS / "final.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    print("Saved models/final/hc50_bundle.pkl and reports/final_test_metrics.json")


if __name__ == "__main__":
    main()
