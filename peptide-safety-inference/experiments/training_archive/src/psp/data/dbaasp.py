"""DBAASP download and HC50 observation extraction."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

from psp.data.concentration import (
    ParsedConcentration,
    molecular_weight_da,
    parse_concentration,
    ug_ml_to_uM,
)
from psp.paths import AA20_SET, RAW, ensure_dirs

DBAASP_LIST = "https://dbaasp.org/peptides"
DBAASP_DETAIL = "https://dbaasp.org/peptides/{peptide_id}"
HEADERS = {"Accept": "application/json", "User-Agent": "psp-safety-predictor/0.1"}

# Endpoint strings that map to ~50% hemolysis (HC50 / HC45-60 etc.).
_HC50_PATTERNS = re.compile(
    r"(?i)(50\s*%?\s*hemol|hc\s*50|mhc\s*50|50-60%\s*hemol|hemolysis\s*50)"
)
_HEMOLYSIS_PATTERNS = re.compile(r"(?i)hemol")
_ERYTHROCYTE_PATTERNS = re.compile(r"(?i)(erythrocyte|rbc|red\s*blood)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_peptide_ids(
    limit: int = 200,
    max_ids: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> list[int]:
    """Page through DBAASP search results and return peptide ids.

    Progress is cached to disk so a timed-out run can resume.
    """
    cache_path = cache_path or (RAW / "dbaasp" / "id_list.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    offset = 0
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            ids = list(cached.get("ids", []))
            offset = int(cached.get("offset", len(ids)))
            print(f"Resuming id list from offset={offset} ({len(ids)} cached)")
        except Exception:  # noqa: BLE001
            ids, offset = [], 0
    total = None
    session = requests.Session()
    while True:
        if max_ids is not None and len(ids) >= max_ids:
            return ids[:max_ids]
        ok = False
        for attempt in range(5):
            try:
                r = session.get(
                    DBAASP_LIST,
                    params={"limit": limit, "offset": offset, "complexity.value": "monomer"},
                    headers=HEADERS,
                    timeout=120,
                )
                r.raise_for_status()
                payload = r.json()
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"list offset={offset} attempt {attempt+1}: {exc}")
                time.sleep(2 ** attempt)
        if not ok:
            # Persist what we have and stop; caller can resume.
            cache_path.write_text(
                json.dumps({"ids": ids, "offset": offset, "total": total}),
                encoding="utf-8",
            )
            print(f"Stopping id listing early with {len(ids)} ids (will resume later)")
            return ids
        if total is None:
            total = int(payload.get("totalCount", 0))
            print(f"DBAASP monomer totalCount={total}")
        batch = payload.get("data") or []
        if not batch:
            break
        for item in batch:
            pid = item.get("id")
            if pid is not None:
                ids.append(int(pid))
        offset += len(batch)
        if offset % 1000 < limit:
            cache_path.write_text(
                json.dumps({"ids": ids, "offset": offset, "total": total}),
                encoding="utf-8",
            )
        if total and offset >= total:
            break
        time.sleep(0.05)
    cache_path.write_text(
        json.dumps({"ids": ids, "offset": offset, "total": total, "complete": True}),
        encoding="utf-8",
    )
    return ids


def search_by_sequence(sequence: str) -> list[int]:
    """Return DBAASP peptide ids matching a full sequence."""
    try:
        r = requests.get(
            DBAASP_LIST,
            params={
                "sequence.value": sequence,
                "sequence.option": "full",
                "limit": 20,
            },
            headers=HEADERS,
            timeout=60,
        )
        r.raise_for_status()
        return [int(x["id"]) for x in (r.json().get("data") or []) if x.get("id") is not None]
    except Exception:  # noqa: BLE001
        return []


def _fetch_one(peptide_id: int, out_dir: Path, retries: int = 3) -> Path:
    path = out_dir / f"{peptide_id}.json"
    if path.exists() and path.stat().st_size > 10:
        return path
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(
                DBAASP_DETAIL.format(peptide_id=peptide_id),
                headers=HEADERS,
                timeout=60,
            )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            # DBAASP returns duplicate CTerminus/cTerminus keys; keep raw bytes.
            path.write_bytes(r.content)
            return path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch peptide {peptide_id}: {last_err}")


def download_dbaasp(
    out_dir: Path | None = None,
    workers: int = 8,
    max_ids: Optional[int] = None,
) -> dict[str, Any]:
    """Concurrent resumable download of all DBAASP peptide detail JSONs."""
    ensure_dirs()
    out_dir = out_dir or (RAW / "dbaasp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Listing DBAASP peptide ids...")
    ids = list_peptide_ids(max_ids=max_ids)
    print(f"Found {len(ids)} peptide ids")
    pending = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    print(f"Need to download {len(pending)} (skipping {len(ids) - len(pending)} cached)")
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fetch_one, i, out_dir): i for i in pending}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="DBAASP"):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
    return {
        "n_ids": len(ids),
        "n_downloaded_now": len(pending) - len(errors),
        "n_errors": len(errors),
        "errors_sample": errors[:20],
        "out_dir": str(out_dir),
    }


def _load_json_lenient(path: Path) -> dict:
    """Load JSON that may contain duplicate keys (DBAASP quirk)."""
    # stdlib json keeps the last key; that's fine for our fields of interest.
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _is_hc50_endpoint(activity: dict) -> bool:
    group = (activity.get("activityMeasureForLysisGroup") or {}).get("name") or ""
    value = activity.get("activityMeasureForLysisValue") or ""
    blob = f"{group} {value}"
    if not _HEMOLYSIS_PATTERNS.search(blob):
        return False
    # Prefer explicit ~50% hemolysis; also accept MHC/HC50 labels.
    return bool(_HC50_PATTERNS.search(blob)) or "mhc" in blob.lower()


def _is_erythrocyte(activity: dict) -> bool:
    cell = (activity.get("targetCell") or {}).get("name") or ""
    return bool(_ERYTHROCYTE_PATTERNS.search(cell))


def _unit_name(activity: dict) -> str:
    return ((activity.get("unit") or {}).get("name") or "").strip()


def _normalize_unit(unit: str) -> str:
    u = unit.lower().replace("μ", "µ").replace(" ", "")
    if u in {"µm", "um", "μm", "micromolar", "microm"}:
        return "uM"
    if u in {"µg/ml", "ug/ml", "µg/ml", "ugml", "µgml"}:
        return "ug/mL"
    if u in {"mg/l", "mgl"}:
        return "ug/mL"  # 1 mg/L = 1 µg/mL
    return unit


def _pubmed_ids(peptide: dict) -> list[str]:
    out = []
    for art in peptide.get("articles") or []:
        pm = (art.get("pubmed") or {}).get("pubmedId")
        if pm:
            out.append(str(pm))
    return out


def _canonical_sequence(seq: object) -> Optional[str]:
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper()
    if not s:
        return None
    # Drop non-letter characters that sometimes appear
    s = re.sub(r"[^A-Z]", "", s)
    if not s:
        return None
    return s


def extract_hc50_observations(raw_dir: Path | None = None) -> pd.DataFrame:
    """Walk DBAASP JSON dumps and build the normalized censored observation table."""
    raw_dir = raw_dir or (RAW / "dbaasp")
    rows: list[dict] = []
    files = sorted(raw_dir.glob("*.json"))
    for path in tqdm(files, desc="Extract HC50"):
        try:
            pep = _load_json_lenient(path)
        except Exception:  # noqa: BLE001
            continue
        if pep.get("errors"):
            continue
        seq = _canonical_sequence(pep.get("sequence"))
        if seq is None:
            continue
        chemistry_in_domain = set(seq).issubset(AA20_SET)
        has_unusual = bool(pep.get("unusualAminoAcids"))
        has_bond = bool(pep.get("intrachainBonds")) or bool(pep.get("interchainBonds"))
        nterm = (pep.get("nTerminus") or pep.get("NTerminus") or {}) or {}
        cterm = (pep.get("cTerminus") or pep.get("CTerminus") or {}) or {}
        nterm_name = nterm.get("name") if isinstance(nterm, dict) else None
        cterm_name = cterm.get("name") if isinstance(cterm, dict) else None
        if nterm_name or cterm_name or has_unusual or has_bond:
            # Still keep the row but mark out of native-AA applicability domain
            # unless only C-amide / N-acetyl which many models accept.
            if not (
                (nterm_name in {None, "ACT", "ACE", "Acetyl"} or not nterm_name)
                and (cterm_name in {None, "AMD", "Amide", "NH2"} or not cterm_name)
                and not has_unusual
                and not has_bond
            ):
                chemistry_in_domain = False

        pubmeds = _pubmed_ids(pep)
        pubmed_primary = pubmeds[0] if pubmeds else None
        activities = pep.get("hemoliticCytotoxicActivities") or []
        for act in activities:
            if not _is_hc50_endpoint(act):
                continue
            if not _is_erythrocyte(act):
                # Keep non-RBC cytotoxic records out of the HC50 table.
                continue
            unit_raw = _unit_name(act)
            unit = _normalize_unit(unit_raw)
            conc_raw = act.get("concentration")
            parsed: ParsedConcentration = parse_concentration(conc_raw)
            decision = "ok"
            hc50_uM = None
            lower_uM = None
            upper_uM = None

            def _to_uM(v: Optional[float]) -> Optional[float]:
                if v is None:
                    return None
                if unit == "uM":
                    return float(v)
                if unit == "ug/mL":
                    return ug_ml_to_uM(float(v), seq)
                return None

            if parsed.censor_type == "unparsable":
                decision = f"unparsable_concentration:{parsed.notes}"
            elif unit not in {"uM", "ug/mL"}:
                decision = f"unsupported_unit:{unit_raw}"
            else:
                if parsed.censor_type == "exact":
                    hc50_uM = _to_uM(parsed.value)
                    if hc50_uM is None:
                        decision = "unit_conversion_failed"
                elif parsed.censor_type == "right":
                    lower_uM = _to_uM(parsed.lower)
                    if lower_uM is None:
                        decision = "unit_conversion_failed"
                elif parsed.censor_type == "left":
                    upper_uM = _to_uM(parsed.upper)
                    if upper_uM is None:
                        decision = "unit_conversion_failed"
                elif parsed.censor_type == "interval":
                    lower_uM = _to_uM(parsed.lower)
                    upper_uM = _to_uM(parsed.upper)
                    if lower_uM is None or upper_uM is None:
                        decision = "unit_conversion_failed"
                    else:
                        hc50_uM = 0.5 * (lower_uM + upper_uM)

            if decision == "ok" and unit == "ug/mL" and not chemistry_in_domain:
                # MW conversion for modified peptides is unreliable.
                decision = "ugml_conversion_skipped_modified"
                hc50_uM = lower_uM = upper_uM = None

            rows.append(
                {
                    "peptide_id": pep.get("id"),
                    "dbaasp_id": pep.get("dbaaspId"),
                    "name": pep.get("name"),
                    "sequence": seq,
                    "length": len(seq),
                    "source": "DBAASP",
                    "dataset": "dbaasp_hemolytic",
                    "activity_id": act.get("id"),
                    "original_value": str(conc_raw) if conc_raw is not None else "",
                    "original_units": unit_raw,
                    "normalized_units": "uM",
                    "endpoint_raw": act.get("activityMeasureForLysisValue"),
                    "endpoint_group": (act.get("activityMeasureForLysisGroup") or {}).get(
                        "name"
                    ),
                    "target_cell": (act.get("targetCell") or {}).get("name"),
                    "ph": act.get("ph") or None,
                    "ionic_strength": act.get("ionicStrength") or None,
                    "salt_type": act.get("saltType") or None,
                    "note": act.get("note") or None,
                    "reference": act.get("reference") or None,
                    "pubmed_id": pubmed_primary,
                    "pubmed_ids": ";".join(pubmeds) if pubmeds else None,
                    "censor_type": parsed.censor_type if decision == "ok" else "unparsable",
                    "hc50_value": hc50_uM,
                    "censor_lower": lower_uM,
                    "censor_upper": upper_uM,
                    "right_censored": parsed.censor_type == "right" and decision == "ok",
                    "n_terminus": nterm_name,
                    "c_terminus": cterm_name,
                    "has_unusual_aa": has_unusual,
                    "has_intrachain_bond": has_bond,
                    "synthesis_type": (pep.get("synthesisType") or {}).get("name")
                    if isinstance(pep.get("synthesisType"), dict)
                    else pep.get("synthesisType"),
                    "chemistry_in_domain": chemistry_in_domain and set(seq).issubset(AA20_SET),
                    "mw_da": molecular_weight_da(seq),
                    "processing_decision": decision,
                    "raw_path": str(path),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        # log-space target for exact / interval midpoints
        import numpy as np

        df["hc50_log_value"] = np.where(
            df["hc50_value"].notna() & (df["hc50_value"] > 0),
            np.log(df["hc50_value"].astype(float)),
            np.nan,
        )
        # For right-censored, store log censor threshold in censor_lower (already µM)
        df["censor_lower_log"] = np.where(
            df["censor_lower"].notna() & (df["censor_lower"] > 0),
            np.log(df["censor_lower"].astype(float)),
            np.nan,
        )
        df["censor_upper_log"] = np.where(
            df["censor_upper"].notna() & (df["censor_upper"] > 0),
            np.log(df["censor_upper"].astype(float)),
            np.nan,
        )
    return df
