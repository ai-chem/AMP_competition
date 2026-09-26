#!/usr/bin/env python
"""Triage peptides with HC50 + aqueous solubility + stability gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from predict import run as predict_run  # noqa: E402
from psp.paths import REPORTS, ensure_dirs  # noqa: E402


def triage(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    for c, default in (
        ("p_hc50_gt_128", 0.5),
        ("hemolysis_risk_prob", 0.5),
        ("in_domain", True),
        ("solubility_aqueous_probability", np.nan),
        ("stability_half_life_hours", np.nan),
        ("prediction_confidence", 0.5),
    ):
        if c not in df.columns:
            df[c] = default

    df["gate_in_domain"] = df["in_domain"].astype(bool)
    df["gate_low_hemolysis_risk"] = df["hemolysis_risk_prob"] <= 0.40
    df["gate_safe_prob"] = df["p_hc50_gt_128"] >= 0.55
    if df["solubility_aqueous_probability"].notna().any():
        df["gate_soluble"] = df["solubility_aqueous_probability"].fillna(0) >= 0.50
        sol_score = df["solubility_aqueous_probability"].fillna(0.5)
    else:
        df["gate_soluble"] = False
        sol_score = pd.Series(0.5, index=df.index)
    df["gate_stable"] = df["stability_half_life_hours"].fillna(0) >= 1.0

    df["n_gates_passed"] = (
        df["gate_in_domain"].astype(int)
        + df["gate_low_hemolysis_risk"].astype(int)
        + df["gate_safe_prob"].astype(int)
        + df["gate_soluble"].astype(int)
        + df["gate_stable"].astype(int)
    )
    df["triage_score"] = (
        0.45 * df["p_hc50_gt_128"].fillna(0.5)
        + 0.20 * (1.0 - df["hemolysis_risk_prob"].fillna(0.5))
        + 0.15 * sol_score
        + 0.10 * np.clip(np.log10(df["stability_half_life_hours"].fillna(0.1) + 1e-6) / 2.0, 0, 1)
        + 0.10 * df["prediction_confidence"].fillna(0.5)
    )
    df.loc[~df["gate_in_domain"], "triage_score"] = -1.0

    df["triage_decision"] = "reject"
    df.loc[df["n_gates_passed"] >= 3, "triage_decision"] = "review"
    df.loc[
        df["gate_in_domain"]
        & df["gate_low_hemolysis_risk"]
        & df["gate_safe_prob"]
        & (df["n_gates_passed"] >= 4),
        "triage_decision",
    ] = "shortlist"

    return df.sort_values(
        ["triage_decision", "triage_score"],
        ascending=[True, False],
        key=lambda s: s.map({"shortlist": 0, "review": 1, "reject": 2})
        if s.name == "triage_decision"
        else s,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--sequence-col", default="sequence")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    ensure_dirs()

    pred_path = args.output or (REPORTS / "triage_predictions.csv")
    predict_run(args.input, pred_path, args.sequence_col, args.device)
    preds = pd.read_csv(pred_path)
    ranked = triage(preds)
    drop_cols = [c for c in ranked.columns if "ecoli" in c.lower()]
    if drop_cols:
        ranked = ranked.drop(columns=drop_cols)

    out = REPORTS / "triage_ranked.csv"
    ranked.to_csv(out, index=False)
    short = ranked[ranked["triage_decision"] == "shortlist"].head(args.top)
    short_path = REPORTS / "triage_shortlist.csv"
    short.to_csv(short_path, index=False)

    print(f"Decisions: {ranked['triage_decision'].value_counts().to_dict()}")
    print(f"Wrote {out}")
    print(f"Shortlist ({len(short)}): {short_path}")


if __name__ == "__main__":
    main()
