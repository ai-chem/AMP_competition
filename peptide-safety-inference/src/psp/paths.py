"""Path helpers for peptide-safety-inference."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
FINAL_MODELS = MODELS
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"
EXAMPLES = ROOT / "examples"
# Embedding cache (optional)
DATA = ROOT / "data"
INTERIM = DATA / "interim"
EMBEDDINGS = INTERIM / "embeddings"

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = set(AA20)
HC50_SAFE_THRESHOLD_UM = 128.0


def ensure_dirs() -> None:
    for p in (MODELS, REPORTS, CONFIGS, EXAMPLES, EMBEDDINGS):
        p.mkdir(parents=True, exist_ok=True)
