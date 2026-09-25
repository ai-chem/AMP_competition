#!/usr/bin/env python
"""Train classical + embedding + hybrid baselines with group-aware CV.

Model selection uses ONLY the training portion (StratifiedGroupKFold).
Locked test is never touched here.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.evaluation import classification_metrics, regression_metrics  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.paths import EMBEDDINGS, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]


def load_train() -> pd.DataFrame:
    path = PROCESSED / "hc50_observations_with_splits.parquet"
    df = pd.read_parquet(path)
    # one row per sequence for modelling (primary exact preferred — already done in splits)
    # Use all train observations but collapse to unique sequences for leakage control
    train = df[df["split"] == "train"].copy()
    # Prefer exact; take first per sequence
    train["_pref"] = (train["censor_type"] != "exact").astype(int)
    train = (
        train.sort_values(["sequence", "_pref"])
        .groupby("sequence", as_index=False)
        .first()
    )
    return train


def build_feature_matrix(sequences, kind: str = "physchem"):
    feats = featurize_frame(sequences)
    if kind == "physchem":
        cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
        return feats[cols].to_numpy(dtype=float), cols
    if kind == "aac":
        cols = [c for c in feats.columns if c.startswith("aac_")]
        return feats[cols].to_numpy(dtype=float), cols
    if kind == "dpc":
        cols = [c for c in feats.columns if c.startswith("dpc_")]
        return feats[cols].to_numpy(dtype=float), cols
    if kind == "full":
        # physchem + aac (skip full dpc for speed in some models)
        cols = [c for c in feats.columns if not c.startswith("dpc_") and not c.startswith("term_")]
        return feats[cols].to_numpy(dtype=float), cols
    raise ValueError(kind)


def get_models(seed: int):
    return {
        "mean": None,  # special
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "elasticnet": Pipeline(
            [("sc", StandardScaler()), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed))]
        ),
        "knn": Pipeline([("sc", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=7))]),
        "pls": Pipeline([("sc", StandardScaler()), ("m", PLSRegression(n_components=5))]),
        "rf": RandomForestRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed
        ),
        "extratrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed
        ),
        "svr": Pipeline(
            [("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))]
        ),
        "mlp": Pipeline(
            [
                ("sc", StandardScaler()),
                (
                    "m",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        max_iter=400,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def try_boosters(seed: int):
    out = {}
    try:
        from xgboost import XGBRegressor

        out["xgboost"] = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        out["lightgbm"] = LGBMRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        out["catboost"] = CatBoostRegressor(
            iterations=400,
            depth=5,
            learning_rate=0.05,
            random_seed=seed,
            verbose=0,
        )
    except Exception:
        pass
    return out


def oof_cv(X, y, groups, strata, model, seed: int, n_splits: int = 5):
    """Return oof predictions and fold metrics (exact observations only for y)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_metrics = []
    for fold, (tr, va) in enumerate(cv.split(X, strata, groups)):
        if model is None:
            pred = np.full(len(va), np.nanmean(y[tr]))
        else:
            m = model
            # clone-ish: re-init from get
            from sklearn.base import clone

            try:
                m = clone(model)
            except Exception:
                m = model
            m.fit(X[tr], y[tr])
            pred = np.asarray(m.predict(X[va])).reshape(-1)
        oof[va] = pred
        fold_metrics.append(regression_metrics(y[va], pred))
    return oof, fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer models/seeds for smoke test")
    parser.add_argument("--with-embeddings", action="store_true", default=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    ensure_dirs()

    train = load_train()
    # Regression target: exact observations only for classical MSE models
    exact_mask = train["censor_type"] == "exact"
    print(f"Train unique sequences: {len(train)}; exact: {exact_mask.sum()}")

    sequences = train["sequence"].tolist()
    y_all = train["hc50_log_value"].to_numpy(dtype=float)
    groups = train["cluster_id70"].to_numpy()
    # stratum for SGKF: combine band + censor
    strata = (
        train["hc50_band"].astype(str) + "_" + train["censor_type"].astype(str)
    ).to_numpy()
    # SGKF needs each stratum to have enough groups — simplify
    strata = np.where(exact_mask, "exact_" + train["hc50_band"].astype(str), "censored").astype(object)
    # Map to ints
    _, strata_i = np.unique(strata, return_inverse=True)

    results = []
    feature_sets = ["physchem", "aac", "full"]
    seeds = SEEDS[:2] if args.quick else SEEDS

    # Precompute features
    feat_cache = {}
    for fs in feature_sets:
        X, cols = build_feature_matrix(sequences, fs)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        feat_cache[fs] = (X, cols)

    # ---- classical on exact only ----
    exact_idx = np.where(exact_mask)[0]
    for fs, (X, cols) in feat_cache.items():
        Xe = X[exact_idx]
        ye = y_all[exact_idx]
        ge = groups[exact_idx]
        se = strata_i[exact_idx]
        for seed in seeds:
            models = get_models(seed)
            models.update(try_boosters(seed))
            if args.quick:
                models = {k: models[k] for k in ["mean", "ridge", "rf", "xgboost"] if k in models}
            for name, model in models.items():
                try:
                    oof, folds = oof_cv(Xe, ye, ge, se, model, seed)
                    met = regression_metrics(ye, oof)
                    met.update(
                        {
                            "model": name,
                            "features": fs,
                            "seed": seed,
                            "family": "classical",
                            "n_features": len(cols),
                        }
                    )
                    # mean fold pearson
                    met["fold_pearson_mean"] = float(
                        np.nanmean([f.get("pearson_r", np.nan) for f in folds])
                    )
                    results.append(met)
                    print(f"[classical] {name}/{fs}/seed{seed}: pearson={met.get('pearson_r')}")
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        {
                            "model": name,
                            "features": fs,
                            "seed": seed,
                            "family": "classical",
                            "error": str(exc),
                        }
                    )

    # ---- embeddings ----
    if args.with_embeddings:
        try:
            from psp.embeddings import DEFAULT_MODEL, embed_sequences

            emb = embed_sequences(
                sequences, model_name=DEFAULT_MODEL, pooling="mean", batch_size=8, device=args.device
            )
            np.savez_compressed(EMBEDDINGS / "train_esm2_35M_mean.npz", embeddings=emb, sequences=np.array(sequences, dtype=object))
            Xe = emb[exact_idx]
            ye = y_all[exact_idx]
            ge = groups[exact_idx]
            se = strata_i[exact_idx]
            for seed in seeds:
                models = {}
                models.update(try_boosters(seed))
                models["ridge"] = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
                models["mlp"] = Pipeline(
                    [
                        ("sc", StandardScaler()),
                        (
                            "m",
                            MLPRegressor(
                                hidden_layer_sizes=(256, 128),
                                max_iter=400,
                                random_state=seed,
                                early_stopping=True,
                            ),
                        ),
                    ]
                )
                if args.quick:
                    models = {k: models[k] for k in list(models)[:3]}
                for name, model in models.items():
                    try:
                        oof, folds = oof_cv(Xe, ye, ge, se, model, seed)
                        met = regression_metrics(ye, oof)
                        met.update(
                            {
                                "model": name,
                                "features": "esm2_t12_35M_mean",
                                "seed": seed,
                                "family": "frozen_plm",
                            }
                        )
                        results.append(met)
                        print(f"[plm] {name}/seed{seed}: pearson={met.get('pearson_r')}")
                    except Exception as exc:  # noqa: BLE001
                        results.append(
                            {
                                "model": name,
                                "features": "esm2_t12_35M_mean",
                                "seed": seed,
                                "family": "frozen_plm",
                                "error": str(exc),
                            }
                        )

            # hybrid: physchem + embedding
            Xh = np.concatenate([feat_cache["physchem"][0][exact_idx], Xe], axis=1)
            for seed in seeds[:3]:
                models = try_boosters(seed)
                models["ridge"] = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
                for name, model in models.items():
                    try:
                        oof, folds = oof_cv(Xh, ye, ge, se, model, seed)
                        met = regression_metrics(ye, oof)
                        met.update(
                            {
                                "model": name,
                                "features": "physchem+esm2_35M",
                                "seed": seed,
                                "family": "hybrid",
                            }
                        )
                        results.append(met)
                        print(f"[hybrid] {name}/seed{seed}: pearson={met.get('pearson_r')}")
                    except Exception as exc:  # noqa: BLE001
                        results.append(
                            {
                                "model": name,
                                "features": "physchem+esm2_35M",
                                "seed": seed,
                                "error": str(exc),
                                "family": "hybrid",
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            print(f"Embedding experiments skipped: {exc}")
            results.append({"family": "frozen_plm", "error": str(exc)})

    # ---- classification head P(HC50>128) on known labels ----
    clf_mask = train["y_safe_gt_128"].notna()
    if clf_mask.sum() >= 30:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        Xc, _ = feat_cache["full"]
        Xc = Xc[clf_mask]
        yc = train.loc[clf_mask, "y_safe_gt_128"].to_numpy(dtype=float)
        gc = groups[clf_mask]
        sc = (yc >= 0.5).astype(int)
        for seed in seeds[:3]:
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            oof = np.full(len(yc), np.nan)
            for tr, va in cv.split(Xc, sc, gc):
                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=10, n_jobs=-1, random_state=seed
                )
                clf.fit(Xc[tr], sc[tr])
                oof[va] = clf.predict_proba(Xc[va])[:, 1]
            met = classification_metrics(yc, oof)
            met.update({"model": "rf", "features": "full", "seed": seed, "family": "psafe_clf"})
            results.append(met)
            print(f"[psafe] rf/seed{seed}: auc={met.get('roc_auc')}")

            oof = np.full(len(yc), np.nan)
            for tr, va in cv.split(Xc, sc, gc):
                clf = Pipeline(
                    [
                        ("sc", StandardScaler()),
                        ("m", LogisticRegression(max_iter=1000, C=1.0)),
                    ]
                )
                clf.fit(Xc[tr], sc[tr])
                oof[va] = clf.predict_proba(Xc[va])[:, 1]
            met = classification_metrics(yc, oof)
            met.update(
                {"model": "logreg", "features": "full", "seed": seed, "family": "psafe_clf"}
            )
            results.append(met)

    out = pd.DataFrame(results)
    out_path = REPORTS / "all_experiments.csv"
    if out_path.exists():
        prev = pd.read_csv(out_path)
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")

    # Save a quick ranking for exact regression
    reg = out[out["family"].isin(["classical", "frozen_plm", "hybrid"]) & out.get("pearson_r", pd.Series(dtype=float)).notna() if "pearson_r" in out.columns else out]
    if "pearson_r" in out.columns:
        ranking = (
            out.dropna(subset=["pearson_r"])
            .sort_values("pearson_r", ascending=False)
            .head(30)
        )
        ranking.to_csv(REPORTS / "baseline_results.csv", index=False)
        print(ranking[["family", "model", "features", "seed", "pearson_r", "spearman_rho", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
