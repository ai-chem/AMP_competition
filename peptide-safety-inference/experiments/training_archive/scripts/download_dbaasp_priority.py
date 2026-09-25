#!/usr/bin/env python
"""Priority DBAASP fetch: resolve HemoPI2 sequences first, then broaden."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.data.dbaasp import (  # noqa: E402
    _fetch_one,
    download_dbaasp,
    list_peptide_ids,
    search_by_sequence,
)
from psp.paths import EXTERNAL, RAW, ensure_dirs  # noqa: E402


def hemopi2_sequences() -> list[str]:
    seqs = []
    for name in ("cross_val_dataset.csv", "independent_dataset.csv"):
        p = EXTERNAL / "hemopi2" / "Dataset" / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "SEQUENCE" if "SEQUENCE" in df.columns else df.columns[0]
        seqs.extend(df[col].astype(str).str.upper().str.strip().tolist())
    return list(dict.fromkeys(seqs))


def main():
    ensure_dirs()
    out_dir = RAW / "dbaasp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Resolve HemoPI2 sequences → DBAASP ids (high-value, preserves censoring)
    seqs = hemopi2_sequences()
    print(f"HemoPI2 unique sequences: {len(seqs)}")
    map_path = out_dir / "hemopi2_seq_to_id.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    pending = [s for s in seqs if s not in mapping]
    print(f"Need to resolve {len(pending)} sequences")
    for s in tqdm(pending, desc="seq->id"):
        ids = search_by_sequence(s)
        mapping[s] = ids
        if len(mapping) % 50 == 0:
            map_path.write_text(json.dumps(mapping), encoding="utf-8")
    map_path.write_text(json.dumps(mapping), encoding="utf-8")

    priority_ids = sorted({i for ids in mapping.values() for i in ids})
    print(f"Priority DBAASP ids from HemoPI2: {len(priority_ids)}")

    # 2) Download priority details
    todo = [i for i in priority_ids if not (out_dir / f"{i}.json").exists()]
    print(f"Downloading {len(todo)} priority details")
    errors = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_fetch_one, i, out_dir): i for i in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="priority"):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
    print(f"Priority done; errors={len(errors)}")

    # 3) Broader monomer crawl (resumable)
    print("Listing monomer ids (resumable)...")
    ids = list_peptide_ids(limit=200)
    print(f"Monomer ids available: {len(ids)}")
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    print(f"Downloading {len(todo)} remaining monomers")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(_fetch_one, i, out_dir): i for i in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="monomers"):
            try:
                fut.result()
            except Exception:
                pass
    n = len(list(out_dir.glob("*.json")))
    print(f"Total JSON files in {out_dir}: {n}")


if __name__ == "__main__":
    main()
