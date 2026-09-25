#!/usr/bin/env python
"""Build model-comparison tables from existing experiment reports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "comparison_tables"


def load_json(name: str) -> dict:
    p = REPORTS / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt) if pd.notna(v) else "")
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- HC50 locked-test trajectory ---
    rows = []
    specs = [
        ("v2_svr", "final_v2_test_metrics.json", "SVR physchem+ESM-2"),
        ("v3_svr_tobit", "final_v3_test_metrics.json", "SVR + Tobit PCA blend"),
        ("v4_boost", "final_v4_test_metrics.json", "CatBoost + ExtraTrees"),
        ("v5_constrained", "final_v5_test_metrics.json", "CB+ET+Tobit (RC constrained)"),
        ("v7_esm_ft", "final_v7_ft_primary_metrics.json", "ESM-2 35M fine-tune (selected)"),
    ]
    for ver, fname, desc in specs:
        m = load_json(fname)
        if not m:
            continue
        reg = m.get("regression_exact") or m.get("locked_test_blend") or {}
        clf = m.get("classification_safe") or {}
        rows.append(
            {
                "version": ver,
                "description": desc,
                "pearson_r": reg.get("pearson_r"),
                "spearman_rho": reg.get("spearman_rho"),
                "r2": reg.get("r2"),
                "mae": reg.get("mae"),
                "rmse": reg.get("rmse"),
                "roc_auc": clf.get("roc_auc"),
                "pr_auc": clf.get("pr_auc"),
                "mcc": clf.get("mcc"),
                "right_censored_consistency": m.get("right_censored_consistency"),
                "coverage_90": m.get("interval_coverage_90"),
                "n_test_exact": reg.get("n"),
            }
        )
    hc50 = pd.DataFrame(rows)
    hc50.to_csv(OUT / "hc50_locked_test_comparison.csv", index=False)

    # --- OOF / selection heads ---
    oof_rows = []
    v4sel = REPORTS / "final_v4_selection.csv"
    if v4sel.exists():
        sel = pd.read_csv(v4sel)
        for _, r in sel.iterrows():
            oof_rows.append(
                {
                    "stage": "v4_group_cv_oof",
                    "model": r.get("model"),
                    "features": r.get("features"),
                    "pearson_r": r.get("pearson_r"),
                    "spearman_rho": r.get("spearman_rho"),
                    "r2": r.get("r2"),
                    "n": r.get("n"),
                }
            )
    deep = REPORTS / "deep_results.csv"
    if deep.exists():
        d = pd.read_csv(deep)
        for _, r in d.iterrows():
            oof_rows.append(
                {
                    "stage": "deep_oof_or_val",
                    "model": r.get("model"),
                    "features": r.get("features"),
                    "pearson_r": r.get("pearson_r"),
                    "spearman_rho": r.get("spearman_rho"),
                    "r2": r.get("r2"),
                    "n": r.get("n"),
                }
            )
    oof_df = pd.DataFrame(oof_rows)
    if len(oof_df):
        oof_df.to_csv(OUT / "hc50_selection_oof_comparison.csv", index=False)

    # --- Aqueous solubility ---
    aq = load_json("solubility_aqueous_audit.json")
    aq_rows = []
    for name, info in (aq.get("models") or {}).items():
        if info.get("status") != "ok":
            continue
        aq_rows.append(
            {
                "solvent_model": name,
                "n": info.get("n"),
                "n_clusters": info.get("n_clusters"),
                "cv_roc_auc": info.get("cv_roc_auc"),
                "cv_pr_auc": info.get("cv_pr_auc"),
                "endpoint": info.get("endpoint"),
            }
        )
    aq_df = pd.DataFrame(aq_rows)
    if len(aq_df):
        aq_df.to_csv(OUT / "aqueous_solubility_cv_comparison.csv", index=False)

    # --- Markdown summary ---
    lines = [
        "# Model comparison tables",
        "",
        "Locked-test metrics use seed `20260925` and were scored once per protocol version.",
        "Production weights are a post-selection refit on all suitable labels;",
        "they are **not** re-evaluated on the locked test.",
        "",
        "## HC50 — locked test (cluster-disjoint)",
        "",
        _md_table(hc50),
        "",
        "## HC50 — selection / OOF",
        "",
        _md_table(oof_df) if len(oof_df) else "_(none)_",
        "",
        "## Aqueous / buffer solubility — cluster CV",
        "",
        _md_table(aq_df.drop(columns=["endpoint"], errors="ignore")) if len(aq_df) else "_(none)_",
        "",
        "Endpoint: binary soluble/insoluble under named aqueous/buffer solvent",
        "(SolPepBench / PepSol2000). **Not** E. coli expression.",
        "",
        "## Selected production stack",
        "",
        "| Component | Choice | Why |",
        "| --- | --- | --- |",
        "| HC50 regressor | ESM-2 35M fine-tune + censored head | Best val pearson (0.398) vs CatBoost OOF (0.352); best locked RC consistency (0.807) |",
        "| HC50 classifier | ExtraTrees physchem+ESM-2 + isotonic | Calibrated P(HC50 > 128 µM) |",
        "| Solubility | SolPepBench pooled aqueous (default 0.1 M PBS) | User-required aqueous/buffer endpoint; pooled CV AUC 0.785 |",
        "| Stability | PEPlife2 protease RF | Best available labelled half-life assay head |",
        "",
    ]
    (OUT / "comparison_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote tables under {OUT}")
    print("COMPARISON_TABLES_DONE")


if __name__ == "__main__":
    main()
