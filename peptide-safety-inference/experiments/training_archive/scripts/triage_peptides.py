#!/usr/bin/env python
"""Real-world peptide triage: HC50 risk + solubility + stability + AD gate.

Ranks an input sequence list into a shortlist suitable for experimental follow-up.
Does NOT invent aqueous solubility or blood half-life when those endpoints are
proxy / assay-specific — every score ships with its endpoint label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from predict import run as predict_run  # noqa: E402
from psp.paths import REPORTS, ensure_dirs  # noqa: E402


def triage(preds: pd.DataFrame) -> pd.DataFrame:
    """Apply documented selection rules and produce a ranked shortlist."""
    df = preds.copy()

    # Required columns with safe defaults
    for c, default in (
        ("p_hc50_gt_128", 0.5),
        ("hemolysis_risk_prob", 0.5),
        ("in_domain", True),
        ("max_train_identity", 0.0),
        ("solubility_aqueous_probability", np.nan),
        ("solubility_ecoli_probability", np.nan),  # retired; ignored if present
        ("stability_half_life_hours", np.nan),
        ("hc50_uM_pred", np.nan),
        ("prediction_confidence", 0.5),
    ):
        if c not in df.columns:
            df[c] = default

    # Gates (documented, not tuned on the locked test)
    df["gate_in_domain"] = df["in_domain"].astype(bool)
    df["gate_low_hemolysis_risk"] = df["hemolysis_risk_prob"] <= 0.40
    df["gate_safe_prob"] = df["p_hc50_gt_128"] >= 0.55
    # Aqueous/buffer solubility only — never E. coli expression.
    sol_col = (
        "solubility_aqueous_probability"
        if "solubility_aqueous_probability" in df.columns
        and df["solubility_aqueous_probability"].notna().any()
        else None
    )
    if sol_col:
        df["gate_soluble"] = df[sol_col].fillna(0) >= 0.50
    else:
        df["gate_soluble"] = False  # do not invent aqueous solubility from proxies
    # Protease half-life: prefer > 1 h when the model is available
    df["gate_stable"] = df["stability_half_life_hours"].fillna(0) >= 1.0

    df["n_gates_passed"] = (
        df["gate_in_domain"].astype(int)
        + df["gate_low_hemolysis_risk"].astype(int)
        + df["gate_safe_prob"].astype(int)
        + df["gate_soluble"].astype(int)
        + df["gate_stable"].astype(int)
    )
    sol_score = df[sol_col].fillna(0.5) if sol_col else pd.Series(0.5, index=df.index)
    # Composite score: safety first, then aqueous solubility, then stability, then AD.
    df["triage_score"] = (
        0.45 * df["p_hc50_gt_128"].fillna(0.5)
        + 0.20 * (1.0 - df["hemolysis_risk_prob"].fillna(0.5))
        + 0.15 * sol_score
        + 0.10 * np.clip(np.log10(df["stability_half_life_hours"].fillna(0.1) + 1e-6) / 2.0, 0, 1)
        + 0.10 * df["prediction_confidence"].fillna(0.5)
    )
    # Hard reject out-of-domain
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
        key=lambda s: s.map({"shortlist": 0, "review": 1, "reject": 2}) if s.name == "triage_decision" else s,
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
    # Never surface the retired E. coli expression column
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
    if len(short):
        cols = [c for c in [
            "sequence", "triage_score", "p_hc50_gt_128", "hemolysis_risk_prob",
            "hc50_uM_pred", "solubility_aqueous_probability", "stability_half_life_hours",
            "in_domain", "solubility_endpoint", "solubility_condition", "stability_endpoint",
        ] if c in short.columns]
        print(short[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
