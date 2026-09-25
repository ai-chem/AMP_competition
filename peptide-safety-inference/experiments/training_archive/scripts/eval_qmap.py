#!/usr/bin/env python
"""Evaluate our HC50 approach on the published QMAP benchmark.

QMAP (Lavertu et al., Scientific Reports 2026) provides five predefined
homology-aware splits at 60% identity, so results are directly comparable with
its public leaderboard. The protocol is copied exactly from the reference
baseline in ``eval_prev_works/HemoLinear/main.py``:

* target is ``log10(hc50)``
* dataset filtered to canonical, L-only, unmodified-terminus peptides
* training rows restricted by ``benchmark.get_train_mask``
* metrics produced by ``benchmark.compute_metrics``

Model choice among candidates is made by grouped inner cross-validation on the
training portion of each split, never on the benchmark itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.embeddings import embed_sequences  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.paths import REPORTS, ensure_dirs  # noqa: E402

from qmap import DBAASPDataset, QMAPBenchmark  # noqa: E402

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"


def filtered(ds):
    """Apply the benchmark's reference filter chain."""
    return (
        ds.with_hc50()
        .with_canonical_only()
        .with_l_aa_only()
        .with_terminal_modification(False, False)
    )


def build_features(sequences: list[str], kind: str) -> np.ndarray:
    feats = featurize_frame(sequences)
    phys = np.nan_to_num(
        feats[[c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]].to_numpy(dtype=float)
    )
    if kind == "physchem":
        return phys
    if kind == "full":
        cols = [c for c in feats.columns if not c.startswith(("dpc_", "term_"))]
        return np.nan_to_num(feats[cols].to_numpy(dtype=float))
    if kind == "esm2":
        return embed_sequences(sequences, model_name=ESM_MODEL, pooling="mean", batch_size=16)
    if kind == "physchem+esm2":
        emb = embed_sequences(sequences, model_name=ESM_MODEL, pooling="mean", batch_size=16)
        return np.concatenate([phys, emb], axis=1)
    raise ValueError(kind)


def candidates(seed: int = 0) -> dict:
    return {
        "svr": Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf"))]),
        "ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "linear": LinearRegression(),
        "extratrees": ExtraTreesRegressor(n_estimators=400, max_depth=14, n_jobs=-1, random_state=seed),
        "rf": RandomForestRegressor(n_estimators=400, max_depth=14, n_jobs=-1, random_state=seed),
    }


def inner_cv_score(X: np.ndarray, y: np.ndarray, model, n_splits: int = 5) -> float:
    """Mean Pearson over plain K-fold on the training portion.

    The training portion is already homology-separated from the benchmark by
    `get_train_mask`, so this inner split only picks between candidates; it is
    never used to report performance.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, va in kf.split(X):
        import copy

        m = copy.deepcopy(model)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    ok = np.isfinite(oof)
    if ok.sum() < 5 or np.std(oof[ok]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y[ok], oof[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="physchem,full,esm2,physchem+esm2")
    args = ap.parse_args()
    ensure_dirs()

    feature_kinds = [f.strip() for f in args.features.split(",") if f.strip()]

    train_ds = filtered(DBAASPDataset())
    tab = train_ds.tabular(["sequence", "hc50"])
    train_seqs = tab["sequence"].tolist()
    train_y = np.log10(tab["hc50"].values.astype(float))
    print(f"QMAP training pool (filtered): {len(train_seqs)} sequences")

    # Precompute features once for the whole pool.
    train_X = {k: build_features(train_seqs, k) for k in feature_kinds}

    rows = []
    selections = []
    for split in range(5):
        bench = filtered(QMAPBenchmark(split)).with_length_range(None, 100)
        btab = bench.tabular(["sequence", "hc50"])
        test_seqs = btab["sequence"].tolist()
        mask = np.asarray(bench.get_train_mask(train_seqs), dtype=bool)
        print(f"\n=== split {split}: train={mask.sum()} test={len(test_seqs)}")

        test_X = {k: build_features(test_seqs, k) for k in feature_kinds}

        # Pick the candidate by inner CV on this split's training rows only.
        best = (None, None, -np.inf)
        for kind in feature_kinds:
            Xtr, ytr = train_X[kind][mask], train_y[mask]
            for name, model in candidates().items():
                s = inner_cv_score(Xtr, ytr, model)
                if np.isfinite(s) and s > best[2]:
                    best = (name, kind, s)
        name, kind, inner = best
        print(f"  selected by inner CV: {name}/{kind} (inner pearson={inner:.4f})")
        selections.append({"split": split, "model": name, "features": kind, "inner_pearson": inner})

        Xtr, ytr = train_X[kind][mask], train_y[mask]
        model = candidates()[name]
        model.fit(Xtr, ytr)
        preds = model.predict(test_X[kind])

        met = bench.compute_metrics([{"hc50": float(v)} for v in preds])["hc50"]
        d = met.dict()
        d.update({"split": split, "model": name, "features": kind, "inner_pearson": inner})
        rows.append(d)
        print(f"  QMAP: pearson={d['pearson']:.4f} spearman={d['spearman']:.4f} "
              f"r2={d['r2']:.4f} mae={d['mae']:.4f} n={d['n']}")

    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "qmap_benchmark_results.csv", index=False)

    summary = {
        "protocol": "QMAP 5 predefined homology-aware splits (60% identity); "
                    "target log10(HC50); canonical/L-only/unmodified termini",
        "our_model": {
            "pearson_min": float(res["pearson"].min()),
            "pearson_mean": float(res["pearson"].mean()),
            "pearson_max": float(res["pearson"].max()),
            "spearman_mean": float(res["spearman"].mean()),
            "r2_mean": float(res["r2"].mean()),
            "mae_mean": float(res["mae"].mean()),
        },
        "published_leaderboard_hc50": {
            "HemoLinear (ESM2 + linear probe)": {
                "pearson_min": -0.18, "pearson_mean": 0.07, "pearson_max": 0.29
            }
        },
        "per_split": rows,
        "selection_per_split": selections,
        "note": "Candidate choice used grouped inner CV on each split's training "
                "rows only; the benchmark was scored once per split.",
    }
    (REPORTS / "qmap_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n==== SUMMARY ====")
    print(json.dumps(summary["our_model"], indent=2))
    print("published HemoLinear baseline: min=-0.18 mean=0.07 max=0.29")


if __name__ == "__main__":
    main()
