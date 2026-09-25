#!/usr/bin/env python
"""Build the normalized censored HC50 observation table + data audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.data.dbaasp import extract_hc50_observations  # noqa: E402
from psp.paths import (  # noqa: E402
    AA20_SET,
    EXTERNAL,
    HC50_SAFE_THRESHOLD_UM,
    PROCESSED,
    RAW,
    REPORTS,
    ensure_dirs,
)


def load_consamphemo_secondary() -> pd.DataFrame:
    """ConsAMPHemo curated regression table (averaged HC50; no raw censoring)."""
    from psp.paths import EXTERNAL as EXT

    path = EXT / "_clones" / "ConsAMPHemo" / "Dataset" / "regression" / "Hemo_regression.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, usecols=["Sequence", "Hemo_avg_con", "LN_value"])
    df = df.rename(columns={"Sequence": "sequence", "Hemo_avg_con": "hc50_value"})
    df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
    df["hc50_value"] = pd.to_numeric(df["hc50_value"], errors="coerce")
    df = df.dropna(subset=["hc50_value", "sequence"])
    df["source"] = "ConsAMPHemo"
    df["dataset"] = "consamphemo_regression"
    df["original_value"] = df["hc50_value"].astype(str)
    df["original_units"] = "uM"
    df["normalized_units"] = "uM"
    df["endpoint_raw"] = "HC50 averaged (ConsAMPHemo; censoring collapsed by authors)"
    df["censor_type"] = "exact"
    df["censor_lower"] = np.nan
    df["censor_upper"] = np.nan
    df["right_censored"] = False
    df["chemistry_in_domain"] = df["sequence"].map(lambda s: set(s).issubset(AA20_SET))
    df["processing_decision"] = "consamphemo_secondary_exact_only"
    df["hc50_log_value"] = np.where(df["hc50_value"] > 0, np.log(df["hc50_value"]), np.nan)
    return df


def load_hemopi2_secondary() -> pd.DataFrame:
    frames = []
    for name, split in (
        ("cross_val_dataset.csv", "hemopi2_cv"),
        ("independent_dataset.csv", "hemopi2_ind"),
    ):
        path = EXTERNAL / "hemopi2" / "Dataset" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"SEQUENCE": "sequence", "μM": "hc50_value", "uM": "hc50_value"})
        if "hc50_value" not in df.columns:
            # try second column
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "sequence", cols[1]: "hc50_value"})
        df["sequence"] = df["sequence"].astype(str).str.upper().str.strip()
        df["hc50_value"] = pd.to_numeric(df["hc50_value"], errors="coerce")
        df["source"] = "HemoPI2"
        df["dataset"] = split
        df["original_value"] = df["hc50_value"].astype(str)
        df["original_units"] = "uM"
        df["normalized_units"] = "uM"
        df["endpoint_raw"] = "HC50 (HemoPI2 curated; censoring collapsed by authors)"
        df["censor_type"] = "exact"
        df["censor_lower"] = np.nan
        df["censor_upper"] = np.nan
        df["right_censored"] = False
        df["chemistry_in_domain"] = df["sequence"].map(
            lambda s: set(s).issubset(AA20_SET)
        )
        df["processing_decision"] = "hemopi2_secondary_exact_only"
        df["hc50_log_value"] = np.where(
            df["hc50_value"] > 0, np.log(df["hc50_value"]), np.nan
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ensure_dirs()
    print("Extracting HC50 observations from DBAASP raw JSON...")
    dbaasp = extract_hc50_observations()
    print(f"DBAASP raw activity rows: {len(dbaasp)}")

    usable = dbaasp[dbaasp["processing_decision"] == "ok"].copy()
    print(f"Usable (decision=ok): {len(usable)}")

    # Keep only AA20 for the modelling table (flagged chemistry stays in audit)
    model_ready = usable[usable["chemistry_in_domain"]].copy()
    print(f"Chemistry in-domain: {len(model_ready)}")

    # Binary label for P(HC50 > 128). Exact: compare value. Right-censored at c:
    # if c >= 128 → positive (definitely >128 if c>128; at c==128 treated positive);
    # if c < 128 → unknown (cannot decide) → drop from classification head.
    def safe_label(row) -> float:
        if row["censor_type"] == "exact" and pd.notna(row["hc50_value"]):
            return float(row["hc50_value"] > HC50_SAFE_THRESHOLD_UM)
        if row["censor_type"] == "right" and pd.notna(row["censor_lower"]):
            if row["censor_lower"] >= HC50_SAFE_THRESHOLD_UM:
                return 1.0
            return np.nan
        if row["censor_type"] == "left" and pd.notna(row["censor_upper"]):
            if row["censor_upper"] <= HC50_SAFE_THRESHOLD_UM:
                return 0.0
            return np.nan
        if row["censor_type"] == "interval":
            lo, hi = row["censor_lower"], row["censor_upper"]
            if pd.notna(lo) and lo > HC50_SAFE_THRESHOLD_UM:
                return 1.0
            if pd.notna(hi) and hi <= HC50_SAFE_THRESHOLD_UM:
                return 0.0
            return np.nan
        return np.nan

    model_ready["y_safe_gt_128"] = model_ready.apply(safe_label, axis=1)

    # Secondary HemoPI2 / ConsAMPHemo tables (cross-check; censoring collapsed)
    hemopi2 = load_hemopi2_secondary()
    cons = load_consamphemo_secondary()
    if not hemopi2.empty:
        hemopi2["y_safe_gt_128"] = (hemopi2["hc50_value"] > HC50_SAFE_THRESHOLD_UM).astype(float)
    if not cons.empty:
        cons["y_safe_gt_128"] = (cons["hc50_value"] > HC50_SAFE_THRESHOLD_UM).astype(float)

    # If DBAASP yielded too few usable rows, bootstrap a modelling table from
    # HemoPI2+ConsAMPHemo (exact-only) so the pipeline can proceed; document it.
    if len(model_ready) < 200:
        print(
            f"WARNING: only {len(model_ready)} DBAASP model-ready rows; "
            "augmenting with HemoPI2+ConsAMPHemo exact observations."
        )
        extras = []
        have = set(model_ready["sequence"]) if len(model_ready) else set()
        for extra in (hemopi2, cons):
            if extra.empty:
                continue
            add = extra[~extra["sequence"].isin(have) & extra["chemistry_in_domain"]].copy()
            add["y_safe_gt_128"] = (add["hc50_value"] > HC50_SAFE_THRESHOLD_UM).astype(float)
            extras.append(add)
            have |= set(add["sequence"])
        if extras:
            model_ready = pd.concat([model_ready] + extras, ignore_index=True, sort=False)
            print(f"Augmented model-ready size: {len(model_ready)}")

    # Persist
    out_primary = PROCESSED / "hc50_observations.parquet"
    model_ready.to_parquet(out_primary, index=False)
    dbaasp.to_parquet(PROCESSED / "hc50_observations_all_decisions.parquet", index=False)
    if not hemopi2.empty:
        hemopi2.to_parquet(PROCESSED / "hc50_hemopi2_secondary.parquet", index=False)
    if not cons.empty:
        cons.to_parquet(PROCESSED / "hc50_consamphemo_secondary.parquet", index=False)

    # Also a CSV for easy inspection
    model_ready.to_csv(PROCESSED / "hc50_observations.csv", index=False)

    # ---- audit report ----
    n_exact = int((model_ready["censor_type"] == "exact").sum())
    n_right = int((model_ready["censor_type"] == "right").sum())
    n_left = int((model_ready["censor_type"] == "left").sum())
    n_interval = int((model_ready["censor_type"] == "interval").sum())
    n_unique_seq = model_ready["sequence"].nunique()
    n_safe = int(model_ready["y_safe_gt_128"].sum(skipna=True))
    n_safe_known = int(model_ready["y_safe_gt_128"].notna().sum())

    decision_counts = dbaasp["processing_decision"].value_counts().to_dict() if len(dbaasp) else {}
    unit_counts = (
        dbaasp.loc[dbaasp["processing_decision"] == "ok", "original_units"]
        .value_counts()
        .to_dict()
        if len(dbaasp)
        else {}
    )

    # Exact-duplicate & conflicting measurements
    dup_groups = model_ready.groupby("sequence")
    n_seq_multi = int((dup_groups.size() > 1).sum())
    conflicts = 0
    for seq, g in dup_groups:
        if len(g) < 2:
            continue
        exact = g[g["censor_type"] == "exact"]["hc50_value"].dropna()
        if len(exact) >= 2 and exact.max() / max(exact.min(), 1e-9) > 2.0:
            conflicts += 1

    # Overlap with HemoPI2
    overlap = 0
    if not hemopi2.empty:
        overlap = len(set(model_ready["sequence"]) & set(hemopi2["sequence"]))
    overlap_cons = 0
    if not cons.empty:
        overlap_cons = len(set(model_ready["sequence"]) & set(cons["sequence"]))

    audit = {
        "n_dbaasp_activity_rows": int(len(dbaasp)),
        "n_usable": int(len(usable)),
        "n_model_ready": int(len(model_ready)),
        "n_unique_sequences": int(n_unique_seq),
        "censor_type_counts": {
            "exact": n_exact,
            "right": n_right,
            "left": n_left,
            "interval": n_interval,
        },
        "y_safe_gt_128_known": n_safe_known,
        "y_safe_gt_128_positive": n_safe,
        "sequences_with_multiple_measurements": n_seq_multi,
        "sequences_with_conflicting_exact_gt2x": conflicts,
        "processing_decision_counts": decision_counts,
        "unit_counts_usable": unit_counts,
        "hemopi2_secondary_rows": int(len(hemopi2)),
        "consamphemo_secondary_rows": int(len(cons)),
        "overlap_unique_seq_with_hemopi2": int(overlap),
        "overlap_unique_seq_with_consamphemo": int(overlap_cons),
        "source_counts": model_ready["source"].value_counts().to_dict() if "source" in model_ready.columns else {},
        "hc50_value_quantiles_exact": (
            model_ready.loc[model_ready["censor_type"] == "exact", "hc50_value"]
            .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            .to_dict()
            if n_exact
            else {}
        ),
    }
    (REPORTS / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Data audit — HC50 observations",
        "",
        f"- DBAASP activity rows extracted: **{audit['n_dbaasp_activity_rows']}**",
        f"- Usable (decision=ok): **{audit['n_usable']}**",
        f"- Model-ready (AA20, in-domain chemistry): **{audit['n_model_ready']}**",
        f"- Unique sequences: **{audit['n_unique_sequences']}**",
        f"- Censoring: exact={n_exact}, right={n_right}, left={n_left}, interval={n_interval}",
        f"- P(HC50>128) labels known: {n_safe_known} (positives={n_safe})",
        f"- Sequences with multiple measurements: {n_seq_multi}",
        f"- Sequences with conflicting exact values (>2×): {conflicts}",
        f"- Overlap with HemoPI2 curated unique sequences: {overlap}",
        f"- Overlap with ConsAMPHemo regression: {overlap_cons}",
        f"- Source counts: {audit['source_counts']}",
        "",
        "## Processing decisions",
        "",
        "```",
        json.dumps(decision_counts, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Conflicting duplicate measurements are **retained** (not averaged).",
        "- HemoPI2 / ConsAMPHemo curated sets are secondary/cross-check sources;",
        "  their authors collapsed ranges/censoring, so DBAASP raw records are preferred",
        "  whenever available.",
        "- µg/mL → µM conversion applied only for canonical linear unmodified peptides.",
        "",
    ]
    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))
    print(f"Wrote {out_primary}")


if __name__ == "__main__":
    main()
