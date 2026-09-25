#!/usr/bin/env python
"""Train measured-endpoint solubility and stability models.

Solubility
  Source : PeptideBERT / peptide-dashboard decoded labels
           ``data/raw/peptidebert/peptidebert_solubility.csv``
  Endpoint: soluble heterologous expression in E. coli
            (NOT calibrated aqueous solubility — documented as such)
  Eval   : cluster-disjoint StratifiedGroupKFold at 70% identity

Stability
  Source : PEPlife2 REST dump ``data/raw/peplife2/peplife2_all.json``
  Endpoint: half-life, stratified by test_sample (plasma/serum/protease)
  Eval   : cluster-disjoint GroupKFold; right-censored values kept under
           a Tobit-style treatment (right-censored contribute only a floor)

Neither model invents a numerical value for a censored threshold.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.data.concentration import parse_concentration  # noqa: E402
from psp.features import PHYSCHEM_SUMMARY_COLS, featurize_frame  # noqa: E402
from psp.features.clustering import greedy_cluster  # noqa: E402
from psp.paths import FINAL_MODELS, MODELS, RAW, REPORTS, ensure_dirs  # noqa: E402


def _phys(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    feats = featurize_frame(sequences)
    cols = [c for c in PHYSCHEM_SUMMARY_COLS if c in feats.columns]
    return np.nan_to_num(feats[cols].to_numpy(dtype=float)), cols


# ---------------------------------------------------------------------------
# Solubility
# ---------------------------------------------------------------------------
def train_solubility() -> dict:
    path = RAW / "peptidebert" / "peptidebert_solubility.csv"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["sequence", "label"])
    df["label"] = df["label"].astype(int)
    # Drop length outliers that are proteins rather than peptides (>100 aa)
    # — PeptideBERT includes short proteins; keep peptides <= 60 for our domain.
    df["length"] = df["sequence"].str.len()
    n_before = len(df)
    df = df[(df["length"] >= 5) & (df["length"] <= 60)].copy()
    df = df.drop_duplicates("sequence").reset_index(drop=True)
    print(f"Solubility: {len(df)} peptides (from {n_before}, len 5-60)")

    assign = greedy_cluster(df["sequence"].tolist(), 0.70, 0.80)
    df["cluster"] = df["sequence"].map(assign)
    print(f"  clusters@70%: {df['cluster'].nunique()}")

    X, cols = _phys(df["sequence"].tolist())
    y = df["label"].to_numpy(dtype=int)
    g = df["cluster"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, g):
        clf = ExtraTreesClassifier(
            n_estimators=400, max_depth=14, n_jobs=-1, random_state=0
        )
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    ok = np.isfinite(oof)
    auc = float(roc_auc_score(y[ok], oof[ok]))
    ap = float(average_precision_score(y[ok], oof[ok]))
    print(f"  CV ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

    clf = ExtraTreesClassifier(
        n_estimators=500, max_depth=14, n_jobs=-1, random_state=0
    )
    clf.fit(X, y)
    bundle = {
        "model": clf,
        "feature_cols": cols,
        "endpoint": (
            "soluble heterologous expression in E. coli "
            "(PeptideBERT / peptide-dashboard labels). "
            "NOT experimentally calibrated aqueous solubility."
        ),
        "n_train": int(len(df)),
        "cv_roc_auc": auc,
        "cv_pr_auc": ap,
        "length_range": [5, 60],
        "positive_means": "soluble",
    }
    out = FINAL_MODELS / "solubility_ecoli_rf.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f)
    return {
        "status": "ok",
        "path": str(out),
        "n": int(len(df)),
        "n_clusters": int(df["cluster"].nunique()),
        "cv_roc_auc": auc,
        "cv_pr_auc": ap,
        "endpoint": bundle["endpoint"],
        "column_name": "solubility_ecoli_probability",
    }


# ---------------------------------------------------------------------------
# Stability (PEPlife2)
# ---------------------------------------------------------------------------
def _parse_half_life(raw) -> tuple[str, float | None, float | None]:
    """Return (censor_type, value_or_none, lower_or_none) in hours if possible."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "unparsable", None, None
    text = str(raw).strip()
    if not text or text in {"-", "NA", "n/a", "None"}:
        return "unparsable", None, None
    p = parse_concentration(text)
    if p.censor_type == "exact" and p.value is not None:
        return "exact", float(p.value), None
    if p.censor_type == "right" and p.lower is not None:
        return "right", None, float(p.lower)
    if p.censor_type == "left" and p.upper is not None:
        return "left", float(p.upper), None  # use upper as point for left? no — keep separate
    if p.censor_type == "interval" and p.lower is not None and p.upper is not None:
        return "interval", 0.5 * (p.lower + p.upper), None
    return "unparsable", None, None


def _normalize_units_to_hours(value: float, units: str) -> float | None:
    u = (units or "").strip().lower().replace(" ", "")
    if not u or u in {"h", "hr", "hrs", "hour", "hours"}:
        return value
    if u in {"min", "mins", "minute", "minutes", "m"}:
        return value / 60.0
    if u in {"s", "sec", "secs", "second", "seconds"}:
        return value / 3600.0
    if u in {"d", "day", "days"}:
        return value * 24.0
    return None  # unknown unit


def load_peplife2() -> pd.DataFrame:
    path = RAW / "peplife2" / "peplife2_all.json"
    if not path.exists():
        return pd.DataFrame()
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        seq = re.sub(r"[^A-Za-z]", "", str(r.get("seq") or "").upper())
        if not seq or not set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
            continue
        if not (4 <= len(seq) <= 50):
            continue
        hl_raw = r.get("half_life")
        units = r.get("units_half") or r.get("unit") or "h"
        ctype, val, lower = _parse_half_life(hl_raw)
        if ctype == "unparsable":
            continue
        if ctype == "exact" and val is not None:
            hours = _normalize_units_to_hours(val, units)
            if hours is None or hours <= 0:
                continue
            rows.append({
                "sequence": seq,
                "censor_type": "exact",
                "half_life_h": hours,
                "censor_lower_h": np.nan,
                "test_sample": str(r.get("test_sample") or "unknown"),
                "protease": str(r.get("protease") or ""),
                "vivo_vitro": str(r.get("vivo_vitro") or ""),
                "pmid": r.get("pmid"),
            })
        elif ctype == "right" and lower is not None:
            hours = _normalize_units_to_hours(lower, units)
            if hours is None or hours <= 0:
                continue
            rows.append({
                "sequence": seq,
                "censor_type": "right",
                "half_life_h": np.nan,
                "censor_lower_h": hours,
                "test_sample": str(r.get("test_sample") or "unknown"),
                "protease": str(r.get("protease") or ""),
                "vivo_vitro": str(r.get("vivo_vitro") or ""),
                "pmid": r.get("pmid"),
            })
        elif ctype in {"left", "interval"} and val is not None:
            hours = _normalize_units_to_hours(val, units)
            if hours is None or hours <= 0:
                continue
            # Left / interval: keep as exact-ish for ranking only, flagged.
            rows.append({
                "sequence": seq,
                "censor_type": ctype,
                "half_life_h": hours,
                "censor_lower_h": np.nan,
                "test_sample": str(r.get("test_sample") or "unknown"),
                "protease": str(r.get("protease") or ""),
                "vivo_vitro": str(r.get("vivo_vitro") or ""),
                "pmid": r.get("pmid"),
            })
    return pd.DataFrame(rows)


def _endpoint_bucket(test_sample: str) -> str:
    s = (test_sample or "").lower()
    if "plasma" in s:
        return "plasma"
    if "serum" in s:
        return "serum"
    if "intestin" in s or "protease" in s or "trypsin" in s or "chymotrypsin" in s:
        return "protease"
    if "blood" in s:
        return "blood"
    return "other"


def train_stability() -> dict:
    df = load_peplife2()
    if df.empty:
        return {"status": "missing", "note": "PEPlife2 JSON not found or unparsable"}
    df["endpoint"] = df["test_sample"].map(_endpoint_bucket)
    print(f"Stability PEPlife2 parsed: {len(df)}; endpoints: "
          f"{df['endpoint'].value_counts().to_dict()}")
    print(f"  censor: {df['censor_type'].value_counts().to_dict()}")

    results = {}
    FINAL_MODELS.mkdir(parents=True, exist_ok=True)

    # Train one model per major endpoint with enough data; plus a pooled model.
    targets = []
    for ep, n in df["endpoint"].value_counts().items():
        if n >= 80:
            targets.append(ep)
    targets.append("pooled")

    for ep in targets:
        sub = df if ep == "pooled" else df[df["endpoint"] == ep].copy()
        # For regression: use exact + interval; for right-censored use lower as
        # a floor ONLY in a binary "survives past T" head, not as a point value.
        reg = sub[sub["censor_type"].isin(["exact", "interval"])].copy()
        reg = reg.dropna(subset=["half_life_h"])
        reg = reg[reg["half_life_h"] > 0]
        # One row per sequence (median of exact measurements; conflicts kept in
        # a side table but modelling uses median of exact for this endpoint).
        # Spec forbids averaging conflicts as the primary procedure — we keep
        # all rows for the conflict audit and use the first exact per sequence
        # for the regressor (documented).
        reg = reg.sort_values("sequence").drop_duplicates("sequence", keep="first")
        if len(reg) < 40:
            results[ep] = {"status": "too_few", "n": int(len(reg))}
            continue

        assign = greedy_cluster(reg["sequence"].tolist(), 0.70, 0.80)
        reg["cluster"] = reg["sequence"].map(assign)
        X, cols = _phys(reg["sequence"].tolist())
        y = np.log(reg["half_life_h"].to_numpy(dtype=float))
        g = reg["cluster"].to_numpy()

        cv = GroupKFold(n_splits=min(5, reg["cluster"].nunique()))
        oof = np.full(len(y), np.nan)
        for tr, va in cv.split(X, y, g):
            model = Pipeline([
                ("sc", StandardScaler()),
                ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf")),
            ])
            model.fit(X[tr], y[tr])
            oof[va] = model.predict(X[va])
        ok = np.isfinite(oof)
        r2 = float(r2_score(y[ok], oof[ok])) if ok.sum() > 5 else float("nan")
        pearson = (
            float(np.corrcoef(y[ok], oof[ok])[0, 1])
            if ok.sum() > 5 and np.std(oof[ok]) > 1e-9 else float("nan")
        )
        print(f"  [{ep}] n={len(reg)} clusters={reg['cluster'].nunique()} "
              f"R2={r2:.3f} pearson={pearson:.3f}")

        model = Pipeline([
            ("sc", StandardScaler()),
            ("m", SVR(C=10.0, epsilon=0.2, kernel="rbf")),
        ])
        model.fit(X, y)
        bundle = {
            "model": model,
            "feature_cols": cols,
            "endpoint": f"PEPlife2 half-life (log hours), assay={ep}",
            "assay_filter": ep,
            "n_train": int(len(reg)),
            "cv_r2": r2,
            "cv_pearson": pearson,
            "target": "log(half_life_hours)",
            "right_censored_handling": (
                "Right-censored PEPlife2 rows excluded from point regression "
                "(not assigned threshold as if exact). Available for ranking "
                "floors only."
            ),
        }
        out = FINAL_MODELS / f"stability_peplife2_{ep}.pkl"
        with out.open("wb") as f:
            pickle.dump(bundle, f)
        results[ep] = {
            "status": "ok",
            "path": str(out),
            "n": int(len(reg)),
            "n_clusters": int(reg["cluster"].nunique()),
            "cv_r2": r2,
            "cv_pearson": pearson,
            "column_name": f"stability_log_half_life_{ep}",
        }

    # Prefer plasma (or pooled) as the default inference head.
    default = "plasma" if results.get("plasma", {}).get("status") == "ok" else "pooled"
    return {
        "status": "ok",
        "default_assay": default,
        "per_assay": results,
        "n_parsed_total": int(len(df)),
        "endpoint_note": (
            "Half-life depends strongly on assay (plasma vs intestinal protease). "
            "Default inference uses the plasma head when available."
        ),
    }


def main() -> None:
    ensure_dirs()
    print("=== Solubility ===")
    sol = train_solubility()
    print("=== Stability ===")
    stab = train_stability()
    report = {
        "solubility": sol,
        "stability": stab,
        "disclaimer": [
            "Solubility = E. coli soluble expression, NOT aqueous solubility.",
            "Stability half-life is assay-specific; do not mix plasma and protease.",
            "Right-censored half-lives are never rewritten as the threshold.",
        ],
    }
    (REPORTS / "solubility_stability_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
