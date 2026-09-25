#!/usr/bin/env python
"""Resolve HemoPI2 sequences to DBAASP ids concurrently, then fetch details."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.data.dbaasp import _fetch_one, search_by_sequence  # noqa: E402
from psp.paths import EXTERNAL, RAW, ensure_dirs  # noqa: E402


def main():
    ensure_dirs()
    out = RAW / "dbaasp"
    out.mkdir(parents=True, exist_ok=True)
    seqs = []
    for name in ("cross_val_dataset.csv", "independent_dataset.csv"):
        p = EXTERNAL / "hemopi2" / "Dataset" / name
        df = pd.read_csv(p)
        col = "SEQUENCE" if "SEQUENCE" in df.columns else df.columns[0]
        seqs.extend(df[col].astype(str).str.upper().str.strip().tolist())
    seqs = list(dict.fromkeys(seqs))
    map_path = out / "hemopi2_seq_to_id.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    pending = [s for s in seqs if s not in mapping]
    print(f"sequences={len(seqs)} pending={len(pending)}", flush=True)

    def resolve(s):
        return s, search_by_sequence(s)

    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = [pool.submit(resolve, s) for s in pending]
        done = 0
        for fut in as_completed(futs):
            s, ids = fut.result()
            mapping[s] = ids
            done += 1
            if done % 50 == 0 or done == len(futs):
                map_path.write_text(json.dumps(mapping), encoding="utf-8")
                print(f"resolved {done}/{len(futs)}", flush=True)
    map_path.write_text(json.dumps(mapping), encoding="utf-8")

    ids = sorted({i for v in mapping.values() for i in v})
    print(f"unique dbaasp ids={len(ids)}", flush=True)
    todo = [i for i in ids if not (out / f"{i}.json").exists()]
    print(f"downloading details {len(todo)}", flush=True)
    errors = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_fetch_one, i, out): i for i in todo}
        done = 0
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors <= 5:
                    print("ERR", exc, flush=True)
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"details {done}/{len(futs)} err={errors}", flush=True)
    print(f"DONE json={len(list(out.glob('*.json')))}", flush=True)


if __name__ == "__main__":
    main()
