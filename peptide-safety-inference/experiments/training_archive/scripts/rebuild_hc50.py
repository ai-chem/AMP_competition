#!/usr/bin/env python
"""Rebuild the expanded HC50 observation table from all available sources.

Sources merged (conflicts retained, never averaged):
  1. DBAASP JSON crawl (existing extractor)
  2. DBAASP hemolytic-and-cytotoxic CSV exports (5 files)
  3. Hemolytik 2.0 complete table (free-text activity parser)
  4. HemoPI2 quantitative uM tables (secondary; authors collapsed censoring)
  5. ConsAMPHemo regression (secondary; collapsed censoring — sequences only
     for overlap audit; NOT mixed into the primary modelling table as targets)

Primary modelling table = sources 1-3 with decision=ok and AA20 chemistry.
Secondary tables are written separately for contamination / cross-check audits.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.data.concentration import (  # noqa: E402
    molecular_weight_da,
    parse_concentration,
    ug_ml_to_uM,
)
from psp.data.dbaasp import extract_hc50_observations  # noqa: E402
from psp.data.hemolysis_text import parse_hemolysis_activity  # noqa: E402
from psp.paths import (  # noqa: E402
    AA20_SET,
    EXTERNAL,
    HC50_SAFE_THRESHOLD_UM,
    PROCESSED,
    RAW,
    REPORTS,
    ensure_dirs,
)

SCHEMA_COLS = [
    "peptide_id", "dbaasp_id", "name", "sequence", "length", "source", "dataset",
    "activity_id", "original_value", "original_units", "normalized_units",
    "endpoint_raw", "endpoint_group", "target_cell", "erythrocyte_species",
    "ph", "ionic_strength", "salt_type", "note", "reference", "pubmed_id",
    "pubmed_ids", "censor_type", "hc50_value", "censor_lower", "censor_upper",
    "right_censored", "n_terminus", "c_terminus", "has_unusual_aa",
    "has_intrachain_bond", "synthesis_type", "chemistry_in_domain", "mw_da",
    "processing_decision", "raw_path", "hc50_log_value", "censor_lower_log",
    "censor_upper_log", "y_safe_gt_128", "lyn_cyc", "ldmix", "cter", "nter",
]


def _canon_seq(s: object) -> str | None:
    if not isinstance(s, str):
        return None
    s = re.sub(r"[^A-Za-z]", "", s.strip().upper())
    return s or None


def _species_from_cell(cell: str) -> str:
    c = (cell or "").lower()
    for sp in ("human", "horse", "sheep", "rabbit", "rat", "mouse", "pig", "cow",
               "bovine", "guinea pig", "dog", "chicken"):
        if sp in c:
            return sp.replace("bovine", "cow").replace("guinea pig", "guinea_pig")
    if "erythrocyte" in c or "rbc" in c:
        return "unknown_rbc"
    return "non_rbc"


def _safe_label(row: pd.Series) -> float:
    thr = HC50_SAFE_THRESHOLD_UM
    if row["censor_type"] == "exact" and pd.notna(row["hc50_value"]):
        return float(row["hc50_value"] > thr)
    if row["censor_type"] == "right" and pd.notna(row["censor_lower"]):
        return 1.0 if row["censor_lower"] >= thr else np.nan
    if row["censor_type"] == "left" and pd.notna(row["censor_upper"]):
        return 0.0 if row["censor_upper"] <= thr else np.nan
    if row["censor_type"] == "interval":
        lo, hi = row["censor_lower"], row["censor_upper"]
        if pd.notna(lo) and lo > thr:
            return 1.0
        if pd.notna(hi) and hi <= thr:
            return 0.0
    return np.nan


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in ("hc50_value", "censor_lower", "censor_upper", "censor_type",
              "processing_decision", "sequence"):
        if c not in df.columns:
            df[c] = np.nan if c != "censor_type" else "unparsable"
    df["length"] = df["sequence"].astype(str).str.len()
    df["mw_da"] = df["sequence"].map(molecular_weight_da)
    df["chemistry_in_domain"] = df["sequence"].map(
        lambda s: isinstance(s, str) and set(s).issubset(AA20_SET)
    )
    df["right_censored"] = df["censor_type"] == "right"
    hv = pd.to_numeric(df["hc50_value"], errors="coerce")
    lo = pd.to_numeric(df["censor_lower"], errors="coerce")
    hi = pd.to_numeric(df["censor_upper"], errors="coerce")
    df["hc50_value"] = hv
    df["censor_lower"] = lo
    df["censor_upper"] = hi
    df["hc50_log_value"] = np.where(hv.notna() & (hv > 0), np.log(hv), np.nan)
    df["censor_lower_log"] = np.where(lo.notna() & (lo > 0), np.log(lo), np.nan)
    df["censor_upper_log"] = np.where(hi.notna() & (hi > 0), np.log(hi), np.nan)
    # Drop literal zeros / negatives — not valid HC50.
    bad = (
        ((df["censor_type"] == "exact") & (hv.notna()) & (hv <= 0))
        | ((df["censor_type"] == "right") & (lo.notna()) & (lo <= 0))
        | ((df["censor_type"] == "left") & (hi.notna()) & (hi <= 0))
    )
    if bad.any():
        df.loc[bad, "processing_decision"] = "nonpositive_value"
        df.loc[bad, "censor_type"] = "unparsable"
    df["y_safe_gt_128"] = df.apply(_safe_label, axis=1)
    for c in SCHEMA_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SCHEMA_COLS]


def _to_uM(value: float | None, unit: str, seq: str) -> float | None:
    if value is None:
        return None
    if unit == "uM":
        return float(value)
    if unit == "ug/mL":
        return ug_ml_to_uM(float(value), seq)
    return None


# ---------------------------------------------------------------------------
# Source 1: DBAASP JSON (reuse existing extractor)
# ---------------------------------------------------------------------------
def load_dbaasp_json() -> pd.DataFrame:
    print("Extracting HC50 from DBAASP JSON crawl...")
    df = extract_hc50_observations()
    if df.empty:
        print("DBAASP JSON: empty")
        return df
    df["erythrocyte_species"] = df["target_cell"].astype(str).map(_species_from_cell)
    df["source"] = "DBAASP"
    df["dataset"] = "dbaasp_json"
    out = _finalize(df)
    print(f"DBAASP JSON rows: {len(out)}; ok="
          f"{int((out['processing_decision']=='ok').sum())}")
    return out


# ---------------------------------------------------------------------------
# Source 2: DBAASP CSV exports
# ---------------------------------------------------------------------------
_HC50_MEASURE = re.compile(
    r"(?i)^(50%\s*hemolysis|hc\s*50|mhc(?:\s*50)?|hd\s*50|lc\s*50|ec\s*50|"
    r"50%\s*lysis|hemolysis\s*50)"
)
_ERYTH = re.compile(r"(?i)(erythrocyte|rbc|red\s*blood)")


def load_dbaasp_csv() -> pd.DataFrame:
    paths = sorted((RAW / "dbaasp").glob("hemolytic*.csv"))
    if not paths:
        print("No DBAASP hemolytic CSVs found")
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    # Dedup exact file-level duplicates across the 5 exports.
    raw = raw.drop_duplicates()
    print(f"DBAASP CSV rows after file-dedup: {len(raw)}")

    rows = []
    for i, r in raw.iterrows():
        seq = _canon_seq(r.get("Peptide Sequence"))
        if not seq:
            continue
        cell = str(r.get("Target Cell") or "")
        if not _ERYTH.search(cell):
            continue
        measure = str(r.get("Activity Measure for Lysis") or "").strip()
        note = str(r.get("Note") or "")
        conc_raw = r.get("Peptide Concentration")
        unit_raw = str(r.get("Unit") or "").strip()

        # Accept explicit HC50-like measures, or "-" / empty with a note that
        # encodes "not active up to X".
        is_hc50 = bool(_HC50_MEASURE.match(measure))
        activity_blob = f"{measure} {note} {conc_raw}"

        unit = None
        u = unit_raw.lower().replace("μ", "µ").replace(" ", "")
        if u in {"µm", "um", "μm"}:
            unit = "uM"
        elif u in {"µg/ml", "ug/ml", "µg/ml"}:
            unit = "ug/mL"

        decision = "ok"
        censor_type = "unparsable"
        hc50 = lower = upper = None
        endpoint_raw = measure

        if is_hc50 and unit and pd.notna(conc_raw) and str(conc_raw).strip():
            parsed = parse_concentration(conc_raw)
            censor_type = parsed.censor_type
            if parsed.censor_type == "exact":
                hc50 = _to_uM(parsed.value, unit, seq)
            elif parsed.censor_type == "right":
                lower = _to_uM(parsed.lower, unit, seq)
            elif parsed.censor_type == "left":
                upper = _to_uM(parsed.upper, unit, seq)
            elif parsed.censor_type == "interval":
                lower = _to_uM(parsed.lower, unit, seq)
                upper = _to_uM(parsed.upper, unit, seq)
                if lower is not None and upper is not None:
                    hc50 = 0.5 * (lower + upper)
            if censor_type == "unparsable" or (
                hc50 is None and lower is None and upper is None
            ):
                decision = "unparsable_concentration"
        else:
            # Try note / free text (e.g. "Not active up to 500 micro M")
            hp = parse_hemolysis_activity(
                f"HC50 {note}" if "not active" in note.lower() else activity_blob
            )
            # Special: "Not active up to X"
            na = re.search(
                rf"(?i)not\s+active\s+up\s+to\s+(\d+(?:\.\d+)?)\s*(µ?u?m|ug/?ml|µg/?ml|micro\s*m)?",
                note,
            )
            if na:
                val = float(na.group(1))
                u2 = (na.group(2) or unit_raw or "uM").lower().replace("μ", "µ")
                unit = "uM" if "m" in u2 and "g" not in u2 else "ug/mL"
                if "micro" in u2:
                    unit = "uM"
                lower = _to_uM(val, unit, seq)
                censor_type = "right"
                endpoint_raw = f"not_active_up_to:{note[:80]}"
                decision = "ok" if lower is not None else "unit_conversion_failed"
            elif hp.censor_type in {"exact", "right", "left", "interval"} and hp.unit:
                unit = hp.unit
                censor_type = hp.censor_type
                hc50 = _to_uM(hp.value, unit, seq) if hp.value is not None else None
                lower = _to_uM(hp.lower, unit, seq) if hp.lower is not None else None
                upper = _to_uM(hp.upper, unit, seq) if hp.upper is not None else None
                endpoint_raw = measure or note[:80]
                decision = "ok"
            else:
                decision = f"skip_measure:{measure[:40]}"

        if decision == "ok" and unit == "ug/mL" and not set(seq).issubset(AA20_SET):
            decision = "ugml_conversion_skipped_modified"
            hc50 = lower = upper = None

        if decision != "ok":
            # Still record for audit with decision flag.
            pass

        rows.append({
            "peptide_id": r.get("Peptide ID"),
            "dbaasp_id": r.get("Peptide ID"),
            "sequence": seq,
            "source": "DBAASP_CSV",
            "dataset": "dbaasp_hemolytic_csv",
            "original_value": str(conc_raw) if pd.notna(conc_raw) else note[:80],
            "original_units": unit_raw,
            "normalized_units": "uM" if decision == "ok" else unit,
            "endpoint_raw": endpoint_raw,
            "endpoint_group": "HC50",
            "target_cell": cell,
            "erythrocyte_species": _species_from_cell(cell),
            "note": note[:200],
            "reference": r.get("Reference"),
            "censor_type": censor_type if decision == "ok" else "unparsable",
            "hc50_value": hc50,
            "censor_lower": lower,
            "censor_upper": upper,
            "processing_decision": decision,
            "raw_path": "data/raw/dbaasp/hemolytic-and-cytotoxic-activities*.csv",
        })
    df = pd.DataFrame(rows)
    print(f"DBAASP CSV extracted rows: {len(df)}; ok="
          f"{int((df['processing_decision']=='ok').sum()) if len(df) else 0}")
    return _finalize(df)


# ---------------------------------------------------------------------------
# Source 3: Hemolytik 2.0
# ---------------------------------------------------------------------------
def load_hemolytik2() -> pd.DataFrame:
    path = RAW / "Hemolytik2" / "Hemolytik2_complete_data.csv"
    if not path.exists():
        print("Hemolytik2 CSV missing")
        return pd.DataFrame()
    raw = pd.read_csv(path)
    print(f"Hemolytik2 raw rows: {len(raw)}")
    rows = []
    for _, r in raw.iterrows():
        seq = _canon_seq(r.get("seq"))
        if not seq:
            continue
        species = str(r.get("source") or "").strip().lower() or "unknown"
        # Only erythrocyte-like species for HC50 modelling.
        if species not in {
            "human", "horse", "sheep", "rabbit", "rat", "mouse", "pig", "cow",
            "guinea pig", "dog", "chicken",
        }:
            # Still keep if activity mentions hemolysis; species unknown_rbc.
            species = species or "unknown"

        activity = r.get("activity")
        hp = parse_hemolysis_activity(activity)

        # Also use non_hem flag as a weak right-censored signal only when no
        # concentration is available — we do NOT invent a threshold number.
        # Flag-only rows are recorded as skip (binary label usable later).
        decision = "ok"
        censor_type = hp.censor_type
        hc50 = lower = upper = None
        unit = hp.unit

        if hp.censor_type in {"exact", "right", "left", "interval"} and unit:
            if hp.censor_type == "exact":
                hc50 = _to_uM(hp.value, unit, seq)
            elif hp.censor_type == "right":
                lower = _to_uM(hp.lower, unit, seq)
            elif hp.censor_type == "left":
                upper = _to_uM(hp.upper, unit, seq)
            elif hp.censor_type == "interval":
                lower = _to_uM(hp.lower, unit, seq)
                upper = _to_uM(hp.upper, unit, seq)
                if lower is not None and upper is not None:
                    hc50 = 0.5 * (lower + upper)
            if hc50 is None and lower is None and upper is None:
                decision = "unit_conversion_failed"
            if decision == "ok" and unit == "ug/mL" and not set(seq).issubset(AA20_SET):
                decision = "ugml_conversion_skipped_modified"
                hc50 = lower = upper = None
        elif hp.endpoint_kind == "non_hemolytic_flag":
            decision = "non_hemolytic_flag_no_concentration"
            censor_type = "unparsable"
        else:
            decision = f"skip:{hp.notes}"
            censor_type = "unparsable"

        rows.append({
            "peptide_id": r.get("id"),
            "name": r.get("name"),
            "sequence": seq,
            "source": "Hemolytik2",
            "dataset": "hemolytik2",
            "original_value": str(activity)[:200] if pd.notna(activity) else "",
            "original_units": unit or "",
            "normalized_units": "uM" if decision == "ok" else unit,
            "endpoint_raw": str(activity)[:200] if pd.notna(activity) else "",
            "endpoint_group": "HC50",
            "target_cell": f"{species} erythrocytes",
            "erythrocyte_species": species,
            "pubmed_id": r.get("pmid"),
            "censor_type": censor_type if decision == "ok" else "unparsable",
            "hc50_value": hc50,
            "censor_lower": lower,
            "censor_upper": upper,
            "n_terminus": r.get("nter"),
            "c_terminus": r.get("cter"),
            "lyn_cyc": r.get("lyn_cyc"),
            "ldmix": r.get("ldmix"),
            "cter": r.get("cter"),
            "nter": r.get("nter"),
            "processing_decision": decision,
            "raw_path": str(path),
            "note": str(r.get("non_hem") or "")[:80],
        })
    df = pd.DataFrame(rows)
    ok = int((df["processing_decision"] == "ok").sum()) if len(df) else 0
    print(f"Hemolytik2 extracted: {len(df)}; ok={ok}")
    return _finalize(df)


# ---------------------------------------------------------------------------
# Secondary: HemoPI2 (collapsed censoring — cross-check only)
# ---------------------------------------------------------------------------
def load_hemopi2_secondary() -> pd.DataFrame:
    frames = []
    for name, split in (
        ("cross_val_dataset.csv", "hemopi2_cv"),
        ("independent_dataset.csv", "hemopi2_ind"),
    ):
        for base in (EXTERNAL / "hemopi2" / "Dataset",
                     EXTERNAL / "_clones" / "HemoPI2" / "Dataset"):
            path = base / name
            if path.exists():
                break
        else:
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={"SEQUENCE": "sequence", "μM": "hc50_value", "uM": "hc50_value"})
        if "hc50_value" not in df.columns:
            cols = list(df.columns)
            df = df.rename(columns={cols[0]: "sequence", cols[1]: "hc50_value"})
        df["sequence"] = df["sequence"].map(_canon_seq)
        df = df.dropna(subset=["sequence", "hc50_value"])
        df["hc50_value"] = pd.to_numeric(df["hc50_value"], errors="coerce")
        df = df.dropna(subset=["hc50_value"])
        df["source"] = "HemoPI2"
        df["dataset"] = split
        df["censor_type"] = "exact"
        df["processing_decision"] = "hemopi2_secondary_exact_only"
        df["endpoint_raw"] = "HC50 (HemoPI2 curated; censoring collapsed)"
        df["normalized_units"] = "uM"
        df["original_units"] = "uM"
        df["original_value"] = df["hc50_value"].astype(str)
        df["erythrocyte_species"] = "human"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _finalize(pd.concat(frames, ignore_index=True))


def main() -> None:
    ensure_dirs()
    dbaasp = load_dbaasp_json()
    dbaasp_csv = load_dbaasp_csv()
    hemolytik = load_hemolytik2()
    hemopi2 = load_hemopi2_secondary()

    all_decisions = pd.concat(
        [dbaasp, dbaasp_csv, hemolytik], ignore_index=True, sort=False
    )
    print(f"All-decision rows: {len(all_decisions)}")

    usable = all_decisions[all_decisions["processing_decision"] == "ok"].copy()
    print(f"Usable (decision=ok): {len(usable)}")

    # Primary modelling: AA20 chemistry only.
    model_ready = usable[usable["chemistry_in_domain"] == True].copy()  # noqa: E712
    print(f"Chemistry in-domain: {len(model_ready)}")

    # Deduplicate *exact* identical observation rows across sources
    # (same sequence, same censor_type, same value/bounds, same pubmed).
    # Conflicting measurements are KEPT.
    before = len(model_ready)
    key_cols = [
        "sequence", "censor_type", "hc50_value", "censor_lower", "censor_upper",
        "pubmed_id", "erythrocyte_species",
    ]
    model_ready = model_ready.drop_duplicates(subset=key_cols, keep="first")
    print(f"Dropped exact-duplicate observations: {before - len(model_ready)}")

    # Persist (cast mixed-type id columns for parquet)
    for col in ("peptide_id", "dbaasp_id", "activity_id", "pubmed_id", "reference"):
        if col in model_ready.columns:
            model_ready[col] = model_ready[col].astype(str).replace({"nan": None, "None": None})
        if col in all_decisions.columns:
            all_decisions[col] = all_decisions[col].astype(str).replace({"nan": None, "None": None})

    out = PROCESSED / "hc50_observations.parquet"
    model_ready.to_parquet(out, index=False)
    model_ready.to_csv(PROCESSED / "hc50_observations.csv", index=False)
    all_decisions.to_parquet(PROCESSED / "hc50_observations_all_decisions.parquet", index=False)
    if not hemopi2.empty:
        hemopi2.to_parquet(PROCESSED / "hc50_hemopi2_secondary.parquet", index=False)

    # Conflict audit
    conflicts = 0
    multi = 0
    for seq, g in model_ready.groupby("sequence"):
        if len(g) < 2:
            continue
        multi += 1
        exact = g.loc[g["censor_type"] == "exact", "hc50_value"].dropna()
        if len(exact) >= 2 and exact.max() / max(exact.min(), 1e-9) > 2.0:
            conflicts += 1

    audit = {
        "n_all_decisions": int(len(all_decisions)),
        "n_usable": int(len(usable)),
        "n_model_ready": int(len(model_ready)),
        "n_unique_sequences": int(model_ready["sequence"].nunique()),
        "censor_type_counts": model_ready["censor_type"].value_counts().to_dict(),
        "source_counts": model_ready["source"].value_counts().to_dict(),
        "species_counts": model_ready["erythrocyte_species"].value_counts().to_dict(),
        "y_safe_gt_128_known": int(model_ready["y_safe_gt_128"].notna().sum()),
        "y_safe_gt_128_positive": int(model_ready["y_safe_gt_128"].sum(skipna=True)),
        "sequences_with_multiple_measurements": multi,
        "sequences_with_conflicting_exact_gt2x": conflicts,
        "processing_decision_counts": all_decisions["processing_decision"]
        .value_counts().head(30).to_dict(),
        "hemopi2_secondary_rows": int(len(hemopi2)),
        "overlap_unique_seq_with_hemopi2": int(
            len(set(model_ready["sequence"]) & set(hemopi2["sequence"]))
            if not hemopi2.empty else 0
        ),
        "hc50_value_quantiles_exact": (
            model_ready.loc[model_ready["censor_type"] == "exact", "hc50_value"]
            .quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            if (model_ready["censor_type"] == "exact").any() else {}
        ),
        "notes": [
            "Conflicting duplicate measurements retained (not averaged).",
            "HemoPI2 kept secondary only — authors collapsed censoring.",
            "ConsAMPHemo regression target NOT used (collapsed censoring pile-up).",
            "Percent-hemolysis-at-concentration converted to censored HC50.",
        ],
    }
    (REPORTS / "data_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )
    lines = [
        "# Data audit — expanded HC50 observations",
        "",
        f"- All-decision rows: **{audit['n_all_decisions']}**",
        f"- Usable (decision=ok): **{audit['n_usable']}**",
        f"- Model-ready (AA20): **{audit['n_model_ready']}** "
        f"({audit['n_unique_sequences']} unique sequences)",
        f"- Censoring: {audit['censor_type_counts']}",
        f"- Sources: {audit['source_counts']}",
        f"- Erythrocyte species: {audit['species_counts']}",
        f"- P(HC50>128) known: {audit['y_safe_gt_128_known']} "
        f"(positives={audit['y_safe_gt_128_positive']})",
        f"- Multi-measurement sequences: {multi}; conflicts (>2x exact): {conflicts}",
        "",
        "## Notes",
        "",
        "- Conflicts retained. No averaging.",
        "- ConsAMPHemo regression targets excluded (collapsed censoring).",
        "- HemoPI2 secondary only.",
        "",
    ]
    (REPORTS / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
