#!/usr/bin/env python
"""Train aqueous / buffer peptide solubility models from SolPepBench (PepSol2000).

Endpoint (stated explicitly):
  Binary soluble vs insoluble under a named aqueous or buffer condition
  (Ultrapure water, 1X DPBS, 0.1 M PBS, Saline). This is experimental
  peptide–solvent observation data, NOT E. coli expression solubility and
  NOT a calibrated continuous mg/mL aqueous solubility curve.

Evaluation: sequence-cluster-disjoint StratifiedGroupKFold at 70% identity,
plus the benchmark's official sequence-disjoint fold when available.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.features.clustering import greedy_cluster  # noqa: E402
from psp.paths import AA20_SET, FINAL_MODELS, RAW, REPORTS, ensure_dirs  # noqa: E402

PEPSOL = ROOT / "external" / "_clones" / "PepSol2000"
AQUEOUS_SOLVENTS = {
    "Ultrapure water",
    "1X DPBS",
    "0.1 M PBS",
    "Saline",
}
# Default inference condition when the user does not specify one.
DEFAULT_SOLVENT = "0.1 M PBS"


def _phys(sequences: list[str]):
    feats = featurize_frame(sequences)
    cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    return np.nan_to_num(feats[cols].to_numpy(dtype=float)), cols


def load_aqueous() -> pd.DataFrame:
    src = PEPSOL / "data" / "processed" / "long_peptide_solvent.csv"
    if not src.exists():
        raise SystemExit(f"Missing {src}; clone Ascaris-Equi/PepSol2000 first")
    df = pd.read_csv(src)
    df = df[df["solvent_name"].isin(AQUEOUS_SOLVENTS)].copy()
    df["sequence"] = df["sequence_clean"].astype(str).str.upper().str.strip()
    df = df[df["sequence"].map(lambda s: set(s).issubset(AA20_SET))]
    df = df[df["y"].isin([0, 1])]
    df["y"] = df["y"].astype(int)
    # Keep conflicts: same sequence+solvent with both labels → keep both rows
    # for audit; for modelling drop exact duplicate (seq, solvent, y).
    before = len(df)
    df = df.drop_duplicates(subset=["sequence", "solvent_name", "y"]).reset_index(drop=True)
    print(
        f"Aqueous/buffer observations: {len(df)} "
        f"(dropped {before - len(df)} exact dups); "
        f"solvents={df['solvent_name'].value_counts().to_dict()}"
    )
    # Copy into our raw tree for provenance
    out_dir = RAW / "pepsol2000"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "aqueous_buffer_observations.csv", index=False)
    shutil.copy2(src, out_dir / "long_peptide_solvent.csv")
    return df


def train_one(df: pd.DataFrame, name: str, solvent_filter: str | None) -> dict:
    sub = df if solvent_filter is None else df[df["solvent_name"] == solvent_filter].copy()
    # One label per sequence for a single-solvent head: if conflicts, keep both
    # only in multi-observation mode; for single-solvent use majority then drop ties.
    if solvent_filter is not None:
        g = sub.groupby("sequence")["y"]
        maj = g.mean()
        keep = maj[(maj == 0.0) | (maj == 1.0)].index
        n_conflict = int(((maj > 0) & (maj < 1)).sum())
        sub = (
            sub[sub["sequence"].isin(keep)]
            .drop_duplicates("sequence")
            .reset_index(drop=True)
        )
        print(f"  [{name}] conflicts dropped (mixed labels): {n_conflict}; n={len(sub)}")
    else:
        # Pooled aqueous: one row per (sequence, solvent); sequence may appear
        # under multiple solvents with different labels — that is intended.
        n_conflict = 0
        print(f"  [{name}] pooled n={len(sub)} unique_seq={sub['sequence'].nunique()}")

    if len(sub) < 80:
        return {"status": "too_few", "n": int(len(sub)), "name": name}

    assign = greedy_cluster(sub["sequence"].tolist(), 0.70, 0.80)
    sub = sub.copy()
    sub["cluster"] = sub["sequence"].map(assign)
    X, cols = _phys(sub["sequence"].tolist())
    # Condition flag for pooled model
    if solvent_filter is None:
        solv_dummies = pd.get_dummies(sub["solvent_name"], prefix="solv")
        X = np.concatenate([X, solv_dummies.to_numpy(dtype=float)], axis=1)
        cols = cols + list(solv_dummies.columns)

    y = sub["y"].to_numpy(dtype=int)
    g = sub["cluster"].to_numpy()
    n_splits = min(5, sub["cluster"].nunique())
    if n_splits < 2:
        return {"status": "too_few_clusters", "n": int(len(sub)), "name": name}

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, g):
        clf = ExtraTreesClassifier(
            n_estimators=400, max_depth=14, n_jobs=-1, random_state=0
        )
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    ok = np.isfinite(oof)
    auc = float(roc_auc_score(y[ok], oof[ok])) if ok.sum() and len(np.unique(y[ok])) > 1 else float("nan")
    ap = float(average_precision_score(y[ok], oof[ok])) if ok.sum() else float("nan")
    print(f"  [{name}] CV ROC-AUC={auc:.3f} PR-AUC={ap:.3f} clusters={sub['cluster'].nunique()}")

    clf = ExtraTreesClassifier(n_estimators=500, max_depth=14, n_jobs=-1, random_state=0)
    clf.fit(X, y)
    bundle = {
        "model": clf,
        "feature_cols": cols,
        "solvent_filter": solvent_filter,
        "solvent_dummies": solvent_filter is None,
        "aqueous_solvents": sorted(AQUEOUS_SOLVENTS),
        "default_solvent": DEFAULT_SOLVENT,
        "endpoint": (
            "Experimental peptide solubility in aqueous / buffer solvent "
            f"({'pooled: ' + ', '.join(sorted(AQUEOUS_SOLVENTS)) if solvent_filter is None else solvent_filter}). "
            "Binary soluble/insoluble from SolPepBench (PepSol2000). "
            "NOT E. coli expression. NOT continuous mg/mL calibration."
        ),
        "n_train": int(len(sub)),
        "n_clusters": int(sub["cluster"].nunique()),
        "cv_roc_auc": auc,
        "cv_pr_auc": ap,
        "n_label_conflicts_dropped": int(n_conflict),
        "source": "https://github.com/Ascaris-Equi/PepSol2000",
        "positive_means": "soluble under the named aqueous/buffer condition",
    }
    out = FINAL_MODELS / f"solubility_aqueous_{name}.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)
    return {
        "status": "ok",
        "path": str(out),
        "name": name,
        "n": int(len(sub)),
        "n_clusters": int(sub["cluster"].nunique()),
        "cv_roc_auc": auc,
        "cv_pr_auc": ap,
        "endpoint": bundle["endpoint"],
        "column_name": f"solubility_aqueous_{name}_probability",
    }


def evaluate_official_sequence_disjoint(df_aq: pd.DataFrame) -> dict:
    """Score our physchem ExtraTrees on PepSolBench sequence-disjoint fold0 (aqueous only)."""
    split_path = PEPSOL / "splits" / "sequence_disjoint_fold0.csv"
    if not split_path.exists():
        return {"status": "missing_split"}
    sp = pd.read_csv(split_path)
    # Expect columns like sequence_id / split
    print("Official split columns:", list(sp.columns))
    # Merge on sequence_id if present
    if "sequence_id" in sp.columns and "sequence_id" in df_aq.columns:
        m = df_aq.merge(sp, on="sequence_id", how="inner", suffixes=("", "_sp"))
    elif "sequence_clean" in sp.columns:
        m = df_aq.merge(
            sp, left_on="sequence", right_on="sequence_clean", how="inner", suffixes=("", "_sp")
        )
    else:
        # try sequence column
        seq_col = next((c for c in sp.columns if "seq" in c.lower()), None)
        split_col = next((c for c in sp.columns if "split" in c.lower() or c == "fold"), None)
        if seq_col is None or split_col is None:
            return {"status": "unusable_split", "columns": list(sp.columns)}
        m = df_aq.merge(sp, left_on="sequence", right_on=seq_col, how="inner", suffixes=("", "_sp"))

    split_col = next(
        (c for c in m.columns if c in {"split", "fold", "set"} or c.endswith("_split")),
        None,
    )
    if split_col is None:
        split_col = next((c for c in m.columns if "split" in c.lower()), None)
    if split_col is None:
        return {"status": "no_split_col", "columns": list(m.columns)}

    # Use PBS + water pooled for this external check
    m = m[m["solvent_name"].isin(AQUEOUS_SOLVENTS)].copy()
    vals = set(m[split_col].astype(str).str.lower().unique())
    print(f"  split values: {vals}")
    train_mask = m[split_col].astype(str).str.lower().isin({"train", "training", "0", "tr"})
    test_mask = m[split_col].astype(str).str.lower().isin({"test", "testing", "1", "te", "val", "valid"})
    if not train_mask.any() or not test_mask.any():
        # maybe fold id: treat fold0 as test
        if m[split_col].nunique() == 2:
            test_val = m[split_col].value_counts().index[-1]
            test_mask = m[split_col] == test_val
            train_mask = ~test_mask
        else:
            return {"status": "cannot_parse_split", "values": list(vals)}

    tr, te = m[train_mask], m[test_mask]
    Xtr, cols = _phys(tr["sequence"].tolist())
    Xte, _ = _phys(te["sequence"].tolist())
    # add solvent dummies aligned
    all_solv = sorted(AQUEOUS_SOLVENTS)
    def solv_mat(frame):
        mat = np.zeros((len(frame), len(all_solv)), dtype=float)
        for i, s in enumerate(frame["solvent_name"].tolist()):
            if s in all_solv:
                mat[i, all_solv.index(s)] = 1.0
        return mat
    Xtr = np.concatenate([Xtr, solv_mat(tr)], axis=1)
    Xte = np.concatenate([Xte, solv_mat(te)], axis=1)
    clf = ExtraTreesClassifier(n_estimators=500, max_depth=14, n_jobs=-1, random_state=0)
    clf.fit(Xtr, tr["y"].to_numpy(int))
    p = clf.predict_proba(Xte)[:, 1]
    y = te["y"].to_numpy(int)
    return {
        "status": "ok",
        "protocol": "PepSolBench sequence_disjoint_fold0, aqueous solvents only",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)),
    }


def main() -> None:
    ensure_dirs()
    df = load_aqueous()
    results = {}
    # Per-solvent heads
    for solv in ["Ultrapure water", "0.1 M PBS", "1X DPBS"]:
        key = solv.lower().replace(" ", "_").replace(".", "")
        results[key] = train_one(df, key, solv)
    # Pooled aqueous/buffer head (default for inference)
    results["pooled"] = train_one(df, "pooled", None)

    official = evaluate_official_sequence_disjoint(df)
    print("Official sequence-disjoint:", official)

    report = {
        "source": "SolPepBench / PepSol2000 (Ascaris-Equi/PepSol2000)",
        "aqueous_solvents": sorted(AQUEOUS_SOLVENTS),
        "default_inference": "solubility_aqueous_pooled.pkl conditioned on 0.1 M PBS",
        "rejected_endpoint": (
            "PeptideBERT E. coli soluble-expression labels are NOT used for "
            "solubility reporting (user requirement: aqueous/buffer only)."
        ),
        "models": results,
        "official_sequence_disjoint": official,
        "disclaimer": [
            "Labels are binary soluble/insoluble under the named solvent — "
            "not a continuous aqueous solubility in mg/mL.",
            "Solvent matters: 839 sequences have discordant labels across solvents "
            "in the full PepSolBench table; we model condition explicitly.",
        ],
    }
    (REPORTS / "solubility_aqueous_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    # Retire the misleading E. coli model from the default path by renaming.
    ecoli = FINAL_MODELS / "solubility_ecoli_rf.pkl"
    if ecoli.exists():
        retired = FINAL_MODELS / "solubility_ecoli_rf.RETIRED.pkl"
        ecoli.replace(retired)
        print(f"Retired E. coli expression model -> {retired.name}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
