#!/usr/bin/env python
"""Robust DBAASP harvest with visible progress (unbuffered)."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.data.dbaasp import HEADERS, DBAASP_LIST, DBAASP_DETAIL, _fetch_one  # noqa: E402
from psp.paths import RAW, ensure_dirs  # noqa: E402


def list_ids(out: Path) -> list[int]:
    cache = out / "id_list.json"
    ids: list[int] = []
    offset = 0
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        ids = list(d.get("ids", []))
        offset = int(d.get("offset", 0))
        print(f"resume ids={len(ids)} offset={offset}", flush=True)
        if d.get("complete"):
            return ids
    session = requests.Session()
    total = None
    limit = 100
    while True:
        for attempt in range(6):
            try:
                r = session.get(
                    DBAASP_LIST,
                    params={"limit": limit, "offset": offset},
                    headers=HEADERS,
                    timeout=90,
                )
                r.raise_for_status()
                payload = r.json()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"list fail offset={offset} try={attempt}: {exc}", flush=True)
                time.sleep(min(30, 2 ** attempt))
        else:
            print("giving up listing; returning partial", flush=True)
            cache.write_text(json.dumps({"ids": ids, "offset": offset, "total": total}), encoding="utf-8")
            return ids
        if total is None:
            total = int(payload.get("totalCount", 0))
            print(f"totalCount={total}", flush=True)
        batch = payload.get("data") or []
        if not batch:
            break
        for item in batch:
            if item.get("id") is not None:
                ids.append(int(item["id"]))
        offset += len(batch)
        if offset % 500 < limit:
            cache.write_text(
                json.dumps({"ids": ids, "offset": offset, "total": total}),
                encoding="utf-8",
            )
            print(f"listed {len(ids)}/{total}", flush=True)
        if total and offset >= total:
            break
        time.sleep(0.02)
    cache.write_text(
        json.dumps({"ids": ids, "offset": offset, "total": total, "complete": True}),
        encoding="utf-8",
    )
    print(f"listing complete: {len(ids)}", flush=True)
    return ids


def main():
    ensure_dirs()
    out = RAW / "dbaasp"
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(list_ids(out)))
    print(f"unique ids={len(ids)}", flush=True)
    todo = [i for i in ids if not (out / f"{i}.json").exists()]
    print(f"to download={len(todo)} cached={len(ids)-len(todo)}", flush=True)
    errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(_fetch_one, i, out): i for i in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors <= 10:
                    print("ERR", exc, flush=True)
            done += 1
            if done % 200 == 0 or done == len(futs):
                print(f"details {done}/{len(futs)} errors={errors}", flush=True)
    n = len(list(out.glob("*.json")))
    print(f"DONE json_files={n} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
