#!/usr/bin/env python
"""Solubility and stability proxy models (endpoint-separated)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.features import PHYSCHEM_SUMMARY_COLS, camsol_like_score, featurize_frame  # noqa: E402
from psp.features.clustering import greedy_cluster  # noqa: E402
from psp.paths import EXTERNAL, MODELS, PROCESSED, RAW, REPORTS, ensure_dirs  # noqa: E402


def try_load_peptidebert_solubility():
    """Look for PeptideBERT / PROSO-like solubility labels in cloned repos."""
    candidates = [
        EXTERNAL / "_clones" / "PeptideBERT",
        EXTERNAL / "_clones" / "peptide-dashboard",
        RAW / "solubility",
    ]
    rows = []
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            if "sequence" in cols and any(k in cols for k in ("solubility", "label", "soluble")):
                full = pd.read_csv(p)
                seq_col = cols["sequence"]
                lab_col = cols.get("solubility") or cols.get("soluble") or cols.get("label")
                sub = full[[seq_col, lab_col]].dropna()
                sub.columns = ["sequence", "label"]
                sub["sequence"] = sub["sequence"].astype(str).str.upper().str.strip()
                sub["source_file"] = str(p)
                rows.append(sub)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    # binarize
    def to_bin(v):
        if isinstance(v, (int, float)):
            return int(v >= 0.5)
        s = str(v).lower()
        if s in {"1", "soluble", "yes", "true", "s"}:
            return 1
        if s in {"0", "insoluble", "no", "false", "i"}:
            return 0
        try:
            return int(float(s) >= 0.5)
        except Exception:
            return np.nan

    df["y"] = df["label"].map(to_bin)
    df = df.dropna(subset=["y"])
    df = df.drop_duplicates("sequence")
    return df


def try_load_half_life():
    """PEPlife / PLifePred-style half-life if present; else None."""
    candidates = list((EXTERNAL / "_clones").glob("*")) + list((RAW / "stability").glob("*")) if (RAW / "stability").exists() else list((EXTERNAL / "_clones").glob("*"))
    for root in candidates:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            try:
                df = pd.read_csv(p, nrows=5)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            if "sequence" in cols and any("half" in k or "hl" == k or "time" in k for k in cols):
                full = pd.read_csv(p)
                seq_col = cols["sequence"]
                lab_col = next(cols[k] for k in cols if "half" in k or k in {"hl", "time", "t1/2"})
                sub = full[[seq_col, lab_col]].dropna()
                sub.columns = ["sequence", "half_life"]
                sub["sequence"] = sub["sequence"].astype(str).str.upper().str.strip()
                sub["half_life"] = pd.to_numeric(sub["half_life"], errors="coerce")
                sub = sub.dropna()
                if len(sub) >= 30:
                    return sub
    return None


def main():
    ensure_dirs()
    report = {"solubility": {}, "stability": {}}

    # --- Solubility proxy ---
    sol = try_load_peptidebert_solubility()
    import pickle

    if sol is not None and len(sol) >= 50:
        print(f"Solubility labels: {len(sol)} from {sol['source_file'].iloc[0]}")
        assign = greedy_cluster(sol["sequence"].tolist(), 0.70, 0.80)
        sol["cluster"] = sol["sequence"].map(assign)
        X = featurize_frame(sol["sequence"].tolist())
        cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in X.columns]
        Xv = np.nan_to_num(X[cols].to_numpy(dtype=float))
        y = sol["y"].to_numpy(dtype=int)
        groups = sol["cluster"].to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.full(len(y), np.nan)
        for tr, va in cv.split(Xv, y, groups):
            clf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=0)
            clf.fit(Xv[tr], y[tr])
            oof[va] = clf.predict_proba(Xv[va])[:, 1]
        auc = float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan")
        report["solubility"] = {
            "endpoint": "E. coli soluble-expression proxy (PROSO-II / PeptideBERT-style labels)",
            "n": int(len(sol)),
            "cv_roc_auc": auc,
            "column_name": "solubility_proxy_probability",
        }
        clf = RandomForestClassifier(n_estimators=400, max_depth=10, n_jobs=-1, random_state=0)
        clf.fit(Xv, y)
        with (MODELS / "final" / "solubility_proxy_rf.pkl").open("wb") as f:
            pickle.dump({"model": clf, "feature_cols": cols, "endpoint": report["solubility"]["endpoint"]}, f)
        print(f"Solubility proxy AUC={auc}")
    else:
        report["solubility"] = {
            "endpoint": "CamSol-like physicochemical score only (no labelled dataset found)",
            "n": 0,
            "column_name": "solubility_proxy_score",
            "note": "Labeled solubility data not found in cloned repos; using camsol_like descriptor.",
        }
        print("No labeled solubility dataset; CamSol-like score will be used at inference.")

    # Always save a note about camsol_like availability
    report["solubility"]["camsol_like_available"] = True

    # --- Stability ---
    hl = try_load_half_life()
    if hl is not None:
        print(f"Half-life records: {len(hl)}")
        assign = greedy_cluster(hl["sequence"].tolist(), 0.70, 0.80)
        hl["cluster"] = hl["sequence"].map(assign)
        X = featurize_frame(hl["sequence"].tolist())
        cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in X.columns]
        # add cleavage features already in PHYSCHEM
        Xv = np.nan_to_num(X[cols].to_numpy(dtype=float))
        y = np.log(hl["half_life"].clip(lower=1e-3).to_numpy(dtype=float))
        groups = hl["cluster"].to_numpy()
        cv = GroupKFold(n_splits=5)
        oof = np.full(len(y), np.nan)
        for tr, va in cv.split(Xv, y, groups):
            reg = RandomForestRegressor(n_estimators=300, max_depth=8, n_jobs=-1, random_state=0)
            reg.fit(Xv[tr], y[tr])
            oof[va] = reg.predict(Xv[va])
        r2 = float(r2_score(y, oof))
        report["stability"] = {
            "endpoint": "mammalian blood half-life (log hours) — PEPlife-style",
            "n": int(len(hl)),
            "cv_r2": r2,
            "column_name": "stability_proxy_log_half_life",
        }
        reg = RandomForestRegressor(n_estimators=400, max_depth=8, n_jobs=-1, random_state=0)
        reg.fit(Xv, y)
        with (MODELS / "final" / "stability_proxy_rf.pkl").open("wb") as f:
            pickle.dump({"model": reg, "feature_cols": cols, "endpoint": report["stability"]["endpoint"]}, f)
        print(f"Stability proxy R2={r2}")
    else:
        report["stability"] = {
            "endpoint": "cleavage-site / oxidation physicochemical proxy only; ML_Peptide SIF available as external",
            "n": 0,
            "column_name": "stability_proxy_score",
            "note": "No PEPlife half-life CSV found; inference will emit cleavage-based proxy + optional SIF.",
        }
        print("No half-life dataset found.")

    (REPORTS / "solubility_stability_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
