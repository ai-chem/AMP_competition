#!/usr/bin/env python
"""Download raw datasets and pin external repositories.

Produces data/raw/... and data/raw/MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.data.dbaasp import download_dbaasp, sha256_file  # noqa: E402
from psp.paths import RAW, EXTERNAL, REPORTS, ensure_dirs  # noqa: E402

PINNED_REPOS = [
    ("raghavagps/HemoPI2", "2b67a5c85422"),
    ("Cpillar/ConsAMPHemo", "950fb333d212"),
    ("plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides", "727fe2d14972"),
    ("adibayaseen/HemoNet", "ba1c948913b4"),
    ("ChakradharG/PeptideBERT", "c6a9a8c40f9a"),
    ("ur-whitelab/peptide-dashboard", "b02d011a1ee6"),
    ("lincubator/LysePred", "3d202a422869"),
    ("FrankWanger/ML_Peptide", "9d23d12b41b0"),
]


def _git_clone_pinned(slug: str, sha_prefix: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{slug}.git"
    if not (dest / ".git").exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        # fetch enough history then checkout the pinned prefix if resolvable
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--depth", "50", "origin"],
            check=False,
            capture_output=True,
            text=True,
        )
        # resolve full sha
        r = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", f"{sha_prefix}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            # fall back to HEAD of default branch
            subprocess.run(
                ["git", "-C", str(dest), "checkout", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            full = r.stdout.strip()
            subprocess.run(
                ["git", "-C", str(dest), "checkout", full],
                check=False,
                capture_output=True,
                text=True,
            )
        # sparse-ish: just checkout all (filter=blob:none already saved bandwidth)
        subprocess.run(
            ["git", "-C", str(dest), "checkout"],
            check=False,
            capture_output=True,
            text=True,
        )
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"slug": slug, "requested_sha_prefix": sha_prefix, "head_sha": head, "path": str(dest)}


def _download_url(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "url": url,
            "path": str(dest),
            "sha256": sha256_file(dest),
            "bytes": dest.stat().st_size,
            "cached": True,
        }
    r = requests.get(url, timeout=120, headers={"User-Agent": "psp-downloader/0.1"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return {
        "url": url,
        "path": str(dest),
        "sha256": sha256_file(dest),
        "bytes": dest.stat().st_size,
        "cached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-dbaasp", type=int, default=None, help="Cap DBAASP ids (debug)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-repos", action="store_true")
    parser.add_argument("--skip-dbaasp", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    manifest: dict = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "entries": [],
    }

    # 1) DBAASP
    if not args.skip_dbaasp:
        info = download_dbaasp(workers=args.workers, max_ids=args.max_dbaasp)
        manifest["entries"].append({"name": "dbaasp", **info})
        # Hash a sample of files for the manifest (full hashing of 25k is slow)
        sample = sorted((RAW / "dbaasp").glob("*.json"))[:20]
        manifest["entries"].append(
            {
                "name": "dbaasp_sample_hashes",
                "files": [
                    {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                    for p in sample
                ],
            }
        )

    # 2) HemoPI2 curated already vendored — record checksums
    for rel in (
        "hemopi2/Dataset/cross_val_dataset.csv",
        "hemopi2/Dataset/independent_dataset.csv",
        "hemopi2/Model/HemoPI2_reg.sav",
        "ml_peptide/SIF_model",
    ):
        p = EXTERNAL / rel
        if p.exists():
            manifest["entries"].append(
                {
                    "name": f"vendored:{rel}",
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
            )

    # 3) Extra public CSVs (HemoPI classic sets, if reachable)
    for url, dest in [
        (
            "https://webs.iiitd.edu.in/raghava/hemopi/HemoPI_datasets.zip",
            RAW / "hemopi" / "HemoPI_datasets.zip",
        ),
    ]:
        try:
            manifest["entries"].append(_download_url(url, dest))
        except Exception as exc:  # noqa: BLE001
            manifest["entries"].append({"url": url, "error": str(exc)})

    # 4) Pin-clone external repos
    if not args.skip_repos:
        clones = EXTERNAL / "_clones"
        for slug, sha in PINNED_REPOS:
            name = slug.split("/")[-1]
            dest = clones / name
            try:
                meta = _git_clone_pinned(slug, sha, dest)
                manifest["entries"].append({"name": f"repo:{slug}", **meta})
            except Exception as exc:  # noqa: BLE001
                manifest["entries"].append({"name": f"repo:{slug}", "error": str(exc)})

    out = RAW / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
