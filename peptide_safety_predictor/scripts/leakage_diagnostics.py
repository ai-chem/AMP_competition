#!/usr/bin/env python
"""Quantify how much a random split flatters the model versus a cluster-disjoint one.

Section 22 of the brief asks for an explicit demonstration of the cases where a
random split looks substantially better than the cluster-disjoint split. Both
evaluations here use the *training portion only* -- the locked test is never
touched -- so this can be run before the final evaluation without spending it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.evaluation import regression_metrics  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.paths import PROCESSED, REPORTS, ensure_dirs  # noqa: E402


def load_train_exact() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hc50_observations_with_splits.parquet")
    df = df[(df["split"] == "train") & (df["censor_type"] == "exact")].copy()
    # One row per sequence so the random split cannot trivially place duplicate
    # measurements of the same peptide on both sides.
    return df.groupby("sequence", as_index=False).first()


def evaluate(X, y, groups, scheme: str, seed: int) -> dict:
    """Run 5-fold CV under either a random or a cluster-disjoint fold assignment."""
    if scheme == "cluster":
        splitter = GroupKFold(n_splits=5).split(X, y, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)

    oof = np.full(len(y), np.nan)
    for tr, va in splitter:
        m = ExtraTreesRegressor(n_estimators=400, max_depth=14, n_jobs=-1, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    met = regression_metrics(y, oof)
    met.update({"scheme": scheme, "seed": seed})
    return met


def main() -> None:
    ensure_dirs()
    df = load_train_exact()
    print(f"Train-portion exact observations: {len(df)}")

    feats = featurize_frame(df["sequence"].tolist())
    cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    X = np.nan_to_num(feats[cols].to_numpy(dtype=float))
    y = df["hc50_log_value"].to_numpy(dtype=float)
    groups = df["cluster_id70"].to_numpy()

    rows = []
    for seed in range(5):
        for scheme in ("random", "cluster"):
            met = evaluate(X, y, groups, scheme, seed)
            rows.append(met)
            print(f"  {scheme}/seed{seed}: pearson={met['pearson_r']:.4f} mae={met['mae']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "leakage_diagnostics.csv", index=False)

    agg = res.groupby("scheme")[["pearson_r", "spearman_rho", "mae", "rmse", "r2"]].agg(["mean", "std"])
    agg.to_csv(REPORTS / "leakage_diagnostics_summary.csv")

    rnd = res[res["scheme"] == "random"]
    clu = res[res["scheme"] == "cluster"]
    summary = {
        "n_sequences": int(len(df)),
        "n_clusters": int(pd.Series(groups).nunique()),
        "random_split": {
            "pearson_mean": float(rnd["pearson_r"].mean()),
            "pearson_std": float(rnd["pearson_r"].std()),
            "mae_mean": float(rnd["mae"].mean()),
            "r2_mean": float(rnd["r2"].mean()),
        },
        "cluster_disjoint_split": {
            "pearson_mean": float(clu["pearson_r"].mean()),
            "pearson_std": float(clu["pearson_r"].std()),
            "mae_mean": float(clu["mae"].mean()),
            "r2_mean": float(clu["r2"].mean()),
        },
        "optimism_gap": {
            "pearson": float(rnd["pearson_r"].mean() - clu["pearson_r"].mean()),
            "mae": float(clu["mae"].mean() - rnd["mae"].mean()),
            "r2": float(rnd["r2"].mean() - clu["r2"].mean()),
        },
        "interpretation": (
            "The random-split numbers are inflated because homologous peptides "
            "appear on both sides of the fold boundary. Only the cluster-disjoint "
            "figures estimate generalization to dissimilar sequences."
        ),
    }
    (REPORTS / "leakage_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
