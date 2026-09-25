"""Path helpers for the peptide_safety_predictor package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # peptide_safety_predictor/
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
FINAL_MODELS = MODELS / "final"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
EXTERNAL = ROOT / "external"
EMBEDDINGS = INTERIM / "embeddings"

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = set(AA20)

# Contest safety threshold used for P(HC50 > 128 µM).
HC50_SAFE_THRESHOLD_UM = 128.0


def ensure_dirs() -> None:
    for p in (RAW, INTERIM, PROCESSED, SPLITS, FINAL_MODELS, REPORTS, EMBEDDINGS):
        p.mkdir(parents=True, exist_ok=True)
