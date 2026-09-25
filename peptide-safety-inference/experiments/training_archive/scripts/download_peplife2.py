#!/usr/bin/env python
"""Download the full PEPlife 2.0 half-life dataset via its REST API.

The API exposes three query fields (``seq``, ``lin_cyc``, ``org``) rather than a
bulk endpoint, so coverage is obtained by unioning several queries on the ``id``
field. ``seq=Natural`` plus ``seq=Modified`` should already be a partition of the
database; the ``lin_cyc`` and ``org`` queries are issued as a cross-check and to
catch entries whose ``seq`` classification is missing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.paths import RAW, ensure_dirs  # noqa: E402

BASE = "https://webs.iiitd.edu.in/raghava/peplife2/api/api.php"

QUERIES = [
    ("seq", "Natural"),
    ("seq", "Modified"),
    ("lin_cyc", "Linear"),
    ("lin_cyc", "Cyclic"),
]

# Organism / media values listed in the user guide; harmless if some return 404.
ORGANISMS = [
    "Mouse", "dogs", "rats", "Pigs", "Cell line/tissue", "DPP-IV", "RhNEP",
    "Monkeys", "mice", "Cats", "Buffer", "MBP", "PBS", "Human", "Rat", "Dog",
    "Rabbit", "Serum", "Plasma",
]


def fetch(data_type: str, data_value: str, retries: int = 3) -> list[dict]:
    params = {"dataType": data_type, "dataValue": data_value}
    for attempt in range(retries):
        try:
            r = requests.get(BASE, params=params, timeout=120)
            if r.status_code == 404:
                return []
            r.raise_for_status()
            payload = r.json()
            return payload.get("data", []) or []
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  FAILED {data_type}={data_value}: {exc}", flush=True)
                return []
            time.sleep(2 * (attempt + 1))
    return []


def main() -> None:
    ensure_dirs()
    out_dir = RAW / "peplife2"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, dict] = {}
    provenance: dict[str, list[str]] = {}

    for data_type, data_value in QUERIES + [("org", o) for o in ORGANISMS]:
        rows = fetch(data_type, data_value)
        new = 0
        for row in rows:
            rid = str(row.get("id"))
            if rid not in by_id:
                by_id[rid] = row
                new += 1
            provenance.setdefault(rid, []).append(f"{data_type}={data_value}")
        print(f"{data_type}={data_value:20s} rows={len(rows):5d} new={new:5d} total={len(by_id)}", flush=True)

    records = list(by_id.values())
    (out_dir / "peplife2_all.json").write_text(
        json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "peplife2_query_provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} unique PEPlife2 records to {out_dir/'peplife2_all.json'}")


if __name__ == "__main__":
    main()
