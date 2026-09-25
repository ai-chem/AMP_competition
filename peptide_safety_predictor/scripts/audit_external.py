#!/usr/bin/env python
"""Audit external repositories listed in the analytical review.

Records URL, HEAD SHA, license, data/checkpoint availability and whether
inference is actually reproducible. Writes reports/external_audit.md.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psp.paths import REPORTS, EXTERNAL, ensure_dirs  # noqa: E402

REPOS = [
    {
        "name": "HemoPI2",
        "url": "https://github.com/raghavagps/HemoPI2",
        "slug": "raghavagps/HemoPI2",
        "target": "quantitative HC50 + binary hemolysis (threshold 100 µM)",
        "notes": "Vendored under external/hemopi2 with Model/HemoPI2_reg.sav. "
                 "Trained on DBAASP+Hemolytik → potentially contaminated vs our test.",
        "local_checkpoint": "external/hemopi2/Model/HemoPI2_reg.sav",
    },
    {
        "name": "ConsAMPHemo",
        "url": "https://github.com/Cpillar/ConsAMPHemo",
        "slug": "Cpillar/ConsAMPHemo",
        "target": "HC50 regression + classification",
        "notes": "Siamese/contrastive GRU + ProtBERT + XGBoost. No SPDX license.",
        "local_checkpoint": None,
    },
    {
        "name": "Plisson et al. non-hemolytic",
        "url": "https://github.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides",
        "slug": "plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides",
        "target": "binary non-hemolytic classification (HemoPI benchmarks)",
        "notes": "MIT. .pkl classifiers trained on HemoPI sets.",
        "local_checkpoint": None,
    },
    {
        "name": "HemoNet",
        "url": "https://github.com/adibayaseen/HemoNet",
        "slug": "adibayaseen/HemoNet",
        "target": "binary hemolysis (DBAASP+Hemolytik)",
        "notes": "weights.hdf referenced; license not declared in README.",
        "local_checkpoint": None,
    },
    {
        "name": "PeptideBERT",
        "url": "https://github.com/ChakradharG/PeptideBERT",
        "slug": "ChakradharG/PeptideBERT",
        "target": "hemolysis + solubility + non-fouling (task-specific heads)",
        "notes": "MIT. HuggingFace/PyTorch ecosystem.",
        "local_checkpoint": None,
    },
    {
        "name": "peptide-dashboard / MahLooL",
        "url": "https://github.com/ur-whitelab/peptide-dashboard",
        "slug": "ur-whitelab/peptide-dashboard",
        "target": "E. coli soluble-expression proxy (PROSO-II labels)",
        "notes": "GPL-3.0. Endpoint mismatch with aqueous synthetic-peptide solubility.",
        "local_checkpoint": None,
    },
    {
        "name": "LysePred",
        "url": "https://github.com/lincubator/LysePred",
        "slug": "lincubator/LysePred",
        "target": "binary hemolysis, multiscale CNN",
        "notes": "Repo oriented to retraining; no released trained checkpoint in root.",
        "local_checkpoint": None,
    },
    {
        "name": "ML_Peptide (SGF/SIF)",
        "url": "https://github.com/FrankWanger/ML_Peptide",
        "slug": "FrankWanger/ML_Peptide",
        "target": "SGF/SIF stability classification",
        "notes": "Vendored under external/ml_peptide (SIF_model). No SPDX license.",
        "local_checkpoint": "external/ml_peptide/SIF_model",
    },
    {
        "name": "AmpLyze",
        "url": None,
        "slug": None,
        "target": "quantitative HC50 (ProtT5/ESM2 + cross-attention)",
        "notes": "No reliably identified official code/checkpoint repo. Methodological reference only.",
        "local_checkpoint": None,
    },
]


def github_meta(slug: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "psp-audit"}
    r = requests.get(f"https://api.github.com/repos/{slug}", headers=headers, timeout=30)
    r.raise_for_status()
    info = r.json()
    branch = info.get("default_branch", "main")
    lic = (info.get("license") or {}).get("spdx_id") or "NOASSERTION"
    c = requests.get(
        f"https://api.github.com/repos/{slug}/commits/{branch}",
        headers=headers,
        timeout=30,
    )
    c.raise_for_status()
    sha = c.json()["sha"]
    tree = requests.get(
        f"https://api.github.com/repos/{slug}/git/trees/{branch}?recursive=1",
        headers=headers,
        timeout=60,
    )
    tree.raise_for_status()
    paths = [t["path"] for t in tree.json().get("tree", []) if t.get("type") == "blob"]
    interesting = [
        p
        for p in paths
        if any(
            p.lower().endswith(ext)
            for ext in (
                ".csv",
                ".tsv",
                ".fasta",
                ".pkl",
                ".pt",
                ".h5",
                ".hdf5",
                ".sav",
                ".json",
                ".npy",
                ".joblib",
                ".onnx",
            )
        )
        or any(k in p.lower() for k in ("model", "weight", "checkpoint", "data"))
    ]
    return {
        "default_branch": branch,
        "license": lic,
        "head_sha": sha,
        "n_files": len(paths),
        "interesting_files": interesting[:40],
        "has_checkpoint_candidate": any(
            p.lower().endswith(ext)
            for p in paths
            for ext in (".pkl", ".pt", ".h5", ".hdf5", ".sav", ".joblib", ".onnx")
        ),
    }


def main() -> None:
    ensure_dirs()
    rows = []
    for repo in REPOS:
        row = dict(repo)
        row["audited_at"] = datetime.now(timezone.utc).isoformat()
        if repo["slug"] is None:
            row.update(
                {
                    "default_branch": None,
                    "license": "n/a",
                    "head_sha": None,
                    "n_files": 0,
                    "interesting_files": [],
                    "has_checkpoint_candidate": False,
                    "inference_reproducible": False,
                    "status": "methodological_reference_only",
                }
            )
        else:
            try:
                meta = github_meta(repo["slug"])
                row.update(meta)
                local_ok = False
                if repo.get("local_checkpoint"):
                    local_ok = (ROOT / repo["local_checkpoint"]).exists()
                row["local_checkpoint_present"] = local_ok
                row["inference_reproducible"] = bool(
                    local_ok or meta["has_checkpoint_candidate"]
                )
                row["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "default_branch": None,
                        "license": "unknown",
                        "head_sha": None,
                        "n_files": 0,
                        "interesting_files": [],
                        "has_checkpoint_candidate": False,
                        "inference_reproducible": False,
                        "status": f"audit_failed: {exc}",
                    }
                )
        # Contamination default for DBAASP/Hemolytik-trained hemolysis predictors
        row["potentially_contaminated"] = row["name"] in {
            "HemoPI2",
            "ConsAMPHemo",
            "HemoNet",
            "LysePred",
            "PeptideBERT",
        }
        rows.append(row)

    out_json = REPORTS / "external_audit.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# External repository audit",
        "",
        f"Audited at {datetime.now(timezone.utc).isoformat()}",
        "",
        "AmpLyze has no reliably identified official checkpoint and is treated as a "
        "methodological reference only. Predictors trained on DBAASP/Hemolytik are marked "
        "`potentially_contaminated` relative to our DBAASP-derived locked test.",
        "",
    ]
    for r in rows:
        lines.append(f"## {r['name']}")
        lines.append("")
        lines.append(f"- URL: {r.get('url')}")
        lines.append(f"- HEAD SHA: `{r.get('head_sha')}`")
        lines.append(f"- License: {r.get('license')}")
        lines.append(f"- Target: {r.get('target')}")
        lines.append(f"- Files in tree: {r.get('n_files')}")
        lines.append(f"- Checkpoint candidate in repo: {r.get('has_checkpoint_candidate')}")
        lines.append(f"- Local checkpoint present: {r.get('local_checkpoint_present', False)}")
        lines.append(f"- Inference reproducible: {r.get('inference_reproducible')}")
        lines.append(f"- Potentially contaminated vs our test: {r.get('potentially_contaminated')}")
        lines.append(f"- Status: {r.get('status')}")
        lines.append(f"- Notes: {r.get('notes')}")
        if r.get("interesting_files"):
            lines.append("- Interesting files (first 40):")
            for p in r["interesting_files"][:40]:
                lines.append(f"  - `{p}`")
        lines.append("")

    (REPORTS / "external_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {REPORTS / 'external_audit.md'}")


if __name__ == "__main__":
    main()
