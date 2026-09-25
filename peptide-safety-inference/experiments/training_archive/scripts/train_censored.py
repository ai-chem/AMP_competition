#!/usr/bin/env python
"""Train censored HC50 models (Tobit, AFT, XGB survival, neural)."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.evaluation import (  # noqa: E402
    censored_gaussian_nll,
    classification_metrics,
    regression_metrics,
)
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.models.censored import ConstantCensored, TobitGaussian  # noqa: E402
from psp.paths import HC50_SAFE_THRESHOLD_UM, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

LOG128 = float(np.log(HC50_SAFE_THRESHOLD_UM))


def load_train() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    train = df[df["split"] == "train"].copy()
    train["_pref"] = (train["censor_type"] != "exact").astype(int)
    train = train.sort_values(["sequence", "_pref"]).groupby("sequence", as_index=False).first()
    return train


def main():
    ensure_dirs()
    train = load_train()
    print(f"Train sequences: {len(train)}; censor counts:\n{train['censor_type'].value_counts()}")

    feats = featurize_frame(train["sequence"].tolist())
    cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    X = np.nan_to_num(feats[cols].to_numpy(dtype=float))
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    y_log = train["hc50_log_value"].to_numpy(dtype=float)
    censor_type = train["censor_type"].to_numpy()
    censor_lower_log = train["censor_lower_log"].to_numpy(dtype=float) if "censor_lower_log" in train else np.full(len(train), np.nan)
    if "censor_lower_log" not in train.columns:
        censor_lower_log = np.where(
            train["censor_lower"].notna() & (train["censor_lower"] > 0),
            np.log(train["censor_lower"].astype(float)),
            np.nan,
        )
    censor_upper_log = (
        train["censor_upper_log"].to_numpy(dtype=float)
        if "censor_upper_log" in train.columns
        else np.where(
            train["censor_upper"].notna() & (train["censor_upper"] > 0),
            np.log(train["censor_upper"].astype(float)),
            np.nan,
        )
    )
    groups = train["cluster_id70"].to_numpy()
    strata = np.where(censor_type == "exact", "exact", "censored")
    _, strata_i = np.unique(strata, return_inverse=True)

    results = []
    oof_mu = {}

    # --- Constant baseline ---
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    mu_oof = np.full(len(train), np.nan)
    sig_oof = np.full(len(train), np.nan)
    for tr, va in cv.split(Xs, strata_i, groups):
        m = ConstantCensored().fit(Xs[tr], y_log[tr], censor_type[tr], censor_lower_log[tr], censor_upper_log[tr])
        dist = m.predict_dist(Xs[va])
        mu_oof[va] = dist.mu
        sig_oof[va] = dist.sigma
    exact = censor_type == "exact"
    met = regression_metrics(y_log[exact], mu_oof[exact])
    nll = censored_gaussian_nll(y_log, censor_type, censor_lower_log, censor_upper_log, mu_oof, np.log(sig_oof))
    psafe = 1.0 - __import__("scipy").stats.norm.cdf((LOG128 - mu_oof) / np.clip(sig_oof, 1e-6, None))
    known = train["y_safe_gt_128"].notna().to_numpy()
    clf_met = classification_metrics(train["y_safe_gt_128"].to_numpy()[known], psafe[known]) if known.sum() else {}
    row = {"model": "constant_censored", "family": "censored", "censored_nll": nll, **{f"reg_{k}": v for k, v in met.items()}, **{f"psafe_{k}": v for k, v in clf_met.items()}}
    results.append(row)
    oof_mu["constant_censored"] = mu_oof
    print(f"constant: pearson={met.get('pearson_r')} nll={nll}")

    # --- Tobit ---
    mu_oof = np.full(len(train), np.nan)
    sig_oof = np.full(len(train), np.nan)
    for tr, va in cv.split(Xs, strata_i, groups):
        m = TobitGaussian(alpha=1.0, max_iter=150).fit(
            Xs[tr], y_log[tr], censor_type[tr], censor_lower_log[tr], censor_upper_log[tr]
        )
        dist = m.predict_dist(Xs[va])
        mu_oof[va] = dist.mu
        sig_oof[va] = dist.sigma
    met = regression_metrics(y_log[exact], mu_oof[exact])
    nll = censored_gaussian_nll(y_log, censor_type, censor_lower_log, censor_upper_log, mu_oof, np.log(sig_oof))
    psafe = 1.0 - __import__("scipy").stats.norm.cdf((LOG128 - mu_oof) / np.clip(sig_oof, 1e-6, None))
    clf_met = classification_metrics(train["y_safe_gt_128"].to_numpy()[known], psafe[known]) if known.sum() else {}
    row = {"model": "tobit_gaussian", "family": "censored", "censored_nll": nll, **{f"reg_{k}": v for k, v in met.items()}, **{f"psafe_{k}": v for k, v in clf_met.items()}}
    results.append(row)
    oof_mu["tobit_gaussian"] = mu_oof
    print(f"tobit: pearson={met.get('pearson_r')} nll={nll} psafe_auc={clf_met.get('roc_auc')}")

    # --- lifelines AFT ---
    try:
        from lifelines import LogNormalAFTFitter

        # Build a survival-style frame: duration = exact y or lower bound; event = exact
        mu_oof = np.full(len(train), np.nan)
        for tr, va in cv.split(Xs, strata_i, groups):
            dur = np.where(
                censor_type[tr] == "exact",
                np.exp(y_log[tr]),
                np.where(
                    censor_type[tr] == "right",
                    np.exp(censor_lower_log[tr]),
                    np.nan,
                ),
            )
            event = (censor_type[tr] == "exact").astype(int)
            ok = np.isfinite(dur) & (dur > 0)
            frame = pd.DataFrame(Xs[tr][ok], columns=[f"f{i}" for i in range(Xs.shape[1])])
            frame["duration"] = dur[ok]
            frame["event"] = event[ok]
            # lifelines can be slow / fail with many features — use top PCA-ish subset
            # Keep first 20 features
            use_cols = [f"f{i}" for i in range(min(20, Xs.shape[1]))]
            fitter = LogNormalAFTFitter()
            fitter.fit(frame[use_cols + ["duration", "event"]], duration_col="duration", event_col="event")
            pred = fitter.predict_median(pd.DataFrame(Xs[va][:, : len(use_cols)], columns=use_cols))
            mu_oof[va] = np.log(np.clip(pred.to_numpy().reshape(-1), 1e-6, None))
        met = regression_metrics(y_log[exact], mu_oof[exact])
        row = {"model": "lognormal_aft", "family": "censored", **{f"reg_{k}": v for k, v in met.items()}}
        results.append(row)
        oof_mu["lognormal_aft"] = mu_oof
        print(f"aft: pearson={met.get('pearson_r')}")
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "lognormal_aft", "family": "censored", "error": str(exc)})
        print(f"AFT failed: {exc}")

    # --- XGBoost survival:aft ---
    try:
        import xgboost as xgb

        mu_oof = np.full(len(train), np.nan)
        for tr, va in cv.split(Xs, strata_i, groups):
            # XGB AFT label: (lower, upper) bounds
            y_lower = np.zeros(len(tr))
            y_upper = np.zeros(len(tr))
            for j, i in enumerate(tr):
                ct = censor_type[i]
                if ct == "exact" and np.isfinite(y_log[i]):
                    y_lower[j] = y_upper[j] = float(np.exp(y_log[i]))
                elif ct == "right" and np.isfinite(censor_lower_log[i]):
                    y_lower[j] = float(np.exp(censor_lower_log[i]))
                    y_upper[j] = np.inf
                elif ct == "left" and np.isfinite(censor_upper_log[i]):
                    y_lower[j] = 0.0
                    y_upper[j] = float(np.exp(censor_upper_log[i]))
                elif ct == "interval":
                    y_lower[j] = float(np.exp(censor_lower_log[i])) if np.isfinite(censor_lower_log[i]) else 0.0
                    y_upper[j] = float(np.exp(censor_upper_log[i])) if np.isfinite(censor_upper_log[i]) else np.inf
                else:
                    y_lower[j] = y_upper[j] = 1.0  # dummy
            dtrain = xgb.DMatrix(Xs[tr])
            dtrain.set_float_info("label_lower_bound", y_lower)
            dtrain.set_float_info("label_upper_bound", y_upper)
            dval = xgb.DMatrix(Xs[va])
            params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
                "tree_method": "hist",
                "learning_rate": 0.05,
                "max_depth": 4,
            }
            bst = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
            pred = bst.predict(dval)  # predicted time
            mu_oof[va] = np.log(np.clip(pred, 1e-6, None))
        met = regression_metrics(y_log[exact], mu_oof[exact])
        row = {"model": "xgb_survival_aft", "family": "censored", **{f"reg_{k}": v for k, v in met.items()}}
        results.append(row)
        oof_mu["xgb_survival_aft"] = mu_oof
        print(f"xgb_aft: pearson={met.get('pearson_r')}")
    except Exception as exc:  # noqa: BLE001
        results.append({"model": "xgb_survival_aft", "family": "censored", "error": str(exc)})
        print(f"XGB AFT failed: {exc}")

    # Fit final Tobit on all train for later inference
    final = TobitGaussian(alpha=1.0, max_iter=200).fit(
        Xs, y_log, censor_type, censor_lower_log, censor_upper_log
    )
    bundle = {
        "model": final,
        "scaler": scaler,
        "feature_cols": cols,
        "log128": LOG128,
    }
    import pickle

    out_model = MODELS / "final" / "tobit_gaussian.pkl"
    with out_model.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {out_model}")

    res_df = pd.DataFrame(results)
    res_path = REPORTS / "censored_results.csv"
    res_df.to_csv(res_path, index=False)
    # append to all_experiments
    all_path = REPORTS / "all_experiments.csv"
    if all_path.exists():
        prev = pd.read_csv(all_path)
        # normalize column names for merge
        flat = res_df.copy()
        if "reg_pearson_r" in flat.columns:
            flat["pearson_r"] = flat["reg_pearson_r"]
            flat["spearman_rho"] = flat.get("reg_spearman_rho")
            flat["mae"] = flat.get("reg_mae")
        flat["features"] = "physchem"
        flat["seed"] = 0
        pd.concat([prev, flat], ignore_index=True).to_csv(all_path, index=False)
    else:
        res_df.to_csv(all_path, index=False)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
