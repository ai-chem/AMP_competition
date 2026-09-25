#!/usr/bin/env python
"""Extract PeptideBERT's three labelled datasets into plain sequence CSVs.

PeptideBERT (github.com/ChakradharG/PeptideBERT) does not ship sequences; its
``data/download_data.py`` pulls integer-encoded arrays from the
``ur-whitelab/peptide-dashboard`` repository. This script downloads those arrays
and decodes them back to amino-acid strings using the same alphabet PeptideBERT
uses in ``data/convert_encodings.py`` (the ``m1`` list), writing one CSV per task.

Endpoints, stated explicitly because they are easy to misread:

* ``solubility``  - soluble vs insoluble *heterologous expression in E. coli*.
  This is an expression-solubility label, NOT a calibrated aqueous solubility.
* ``hemolysis``   - binary hemolytic vs non-hemolytic, thresholded upstream.
* ``nonfouling``  - resistance to non-specific protein adsorption.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "peptidebert"

BASE = "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/"

# PeptideBERT's `m1` alphabet: index -> amino acid, 0 is padding.
ALPHABET = [
    "[PAD]", "A", "R", "N", "D", "C", "Q", "E", "G", "H",
    "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

TASKS = {
    "hemolysis": {
        "positive": "hemo-positive.npz",
        "negative": "hemo-negative.npz",
        "endpoint": "hemolytic (binary, thresholded upstream by peptide-dashboard)",
        "positive_means": "hemolytic",
    },
    "solubility": {
        "positive": "soluble.npz",
        "negative": "insoluble.npz",
        "endpoint": "soluble heterologous expression in E. coli (NOT aqueous solubility)",
        "positive_means": "soluble",
    },
    "nonfouling": {
        "positive": "human-positive.npz",
        "negative": "human-negative.npz",
        "endpoint": "non-fouling / resistant to non-specific adsorption",
        "positive_means": "non-fouling",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(BASE + name, dest)
    return dest


def decode(path: Path) -> list[str]:
    """Turn one integer-encoded array into amino-acid strings, dropping padding."""
    data = np.load(path, allow_pickle=True)
    key = "arr_0" if "arr_0" in data else "seqs"
    arr = np.asarray(data[key]).astype(int)
    seqs = []
    for row in arr:
        s = "".join(ALPHABET[i] for i in row if 0 < i < len(ALPHABET))
        if s:
            seqs.append(s)
    return seqs


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for task, cfg in TASKS.items():
        frames = []
        files = {}
        for label_name in ("positive", "negative"):
            src = cfg[label_name]
            dest = RAW / src
            fetch(src, dest)
            files[src] = {"sha256": sha256(dest), "bytes": dest.stat().st_size}
            seqs = decode(dest)
            frames.append(
                pd.DataFrame(
                    {"sequence": seqs, "label": 1 if label_name == "positive" else 0}
                )
            )
            print(f"{task:11s} {label_name:8s} {src:20s} -> {len(seqs)} sequences")

        df = pd.concat(frames, ignore_index=True)
        df["length"] = df["sequence"].str.len()

        # A sequence appearing under both labels is a genuine conflict; keep one
        # row per (sequence, label) but record how many such conflicts exist.
        before = len(df)
        df = df.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)
        conflicts = int(df["sequence"].duplicated(keep=False).sum())

        out = RAW / f"peptidebert_{task}.csv"
        df.to_csv(out, index=False)

        manifest[task] = {
            "endpoint": cfg["endpoint"],
            "positive_means": cfg["positive_means"],
            "output": out.name,
            "n_rows": int(len(df)),
            "n_unique_sequences": int(df["sequence"].nunique()),
            "n_positive": int((df["label"] == 1).sum()),
            "n_negative": int((df["label"] == 0).sum()),
            "n_exact_duplicate_rows_dropped": int(before - len(df)),
            "n_rows_in_label_conflict": conflicts,
            "length_min": int(df["length"].min()),
            "length_max": int(df["length"].max()),
            "length_median": float(df["length"].median()),
            "source_files": files,
        }
        print(
            f"  -> {out.name}: {len(df)} rows, {df['sequence'].nunique()} unique, "
            f"{conflicts} in label conflict, len {df['length'].min()}-{df['length'].max()}"
        )

    meta = {
        "source_repo": "https://github.com/ChakradharG/PeptideBERT",
        "data_origin": "https://github.com/ur-whitelab/peptide-dashboard (ml/data)",
        "decoding": "PeptideBERT data/convert_encodings.py `m1` alphabet",
        "tasks": manifest,
    }
    (RAW / "MANIFEST.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {RAW / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
