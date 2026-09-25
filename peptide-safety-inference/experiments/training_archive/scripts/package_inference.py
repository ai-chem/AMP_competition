#!/usr/bin/env python
"""Assemble peptide-safety-inference/ with final models only; archive experiments."""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]  # peptide_safety_predictor (still)
REPO = SRC.parent
DST = REPO / "peptide-safety-inference"
EXP = DST / "experiments"
ARCHIVE = EXP / "training_archive"

PSP_KEEP = [
    "__init__.py",
    "calibration/__init__.py",
    "embeddings/__init__.py",
    "features/__init__.py",
    "features/clustering.py",
    "ood/__init__.py",
    "uncertainty/__init__.py",
]

FINAL_MODELS = [
    "hc50_bundle.pkl",
    "esm2_35M_ft.pt",
    "solubility_aqueous_pooled.pkl",
    "stability_peplife2_protease.pkl",
]

DROP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".cursor",
    "_superseded_smalldata",
}
DROP_FILE_SUFFIXES = {".pyc", ".log"}
DROP_NAME_SUBSTR = ("_diag_", "RETIRED", "agent-transcript")


def write_paths(dst_psp: Path) -> None:
    dst_psp.mkdir(parents=True, exist_ok=True)
    (dst_psp / "paths.py").write_text(
        '''"""Path helpers for peptide-safety-inference."""

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
''',
        encoding="utf-8",
    )


def write_readme(dst: Path) -> None:
    (dst / "README.md").write_text(
        """# Peptide Safety Inference

Predict peptide **hemolytic safety (HC50)**, **aqueous/buffer solubility**, and
**protease half-life** from amino-acid sequence.

## Install

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\pip install -r requirements.txt
# Linux / macOS
.venv/bin/pip install -r requirements.txt
```

Optional GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

First run downloads `facebook/esm2_t12_35M_UR50D` from Hugging Face (~140 MB).

## Run

```bash
python predict.py --input examples/controllability_sequences.csv --output predictions.csv --sequence-col sequence --device auto

python triage.py --input examples/controllability_sequences.csv --top 30 --device auto
```

## Model

| Head | Architecture | Data |
| --- | --- | --- |
| HC50 | ESM-2 35M fine-tune (censored Gaussian head) + ExtraTrees P(HC50 > 128 µM) | All suitable HC50 observations |
| Solubility | ExtraTrees physicochemical features; pooled aqueous solvents | SolPepBench / PepSol2000 (water, PBS, DPBS, saline) — not E. coli expression |
| Stability | Random forest log half-life | PEPlife2 protease assay |

Selection-phase locked test (cluster-disjoint, seed 20260925, before full-data refit):

| Metric | Value |
| --- | ---: |
| Exact pearson / spearman | 0.317 / 0.383 |
| Safe-class ROC-AUC / MCC | 0.837 / 0.545 |
| Right-censored consistency | 0.807 |
| 90% interval coverage | 0.926 |

Weights in `models/` are a post-selection refit on all suitable labels.

### Main output columns

| Column | Meaning |
| --- | --- |
| `hc50_uM_pred` | Predicted HC50 (µM) |
| `p_hc50_gt_128` | Calibrated P(HC50 > 128 µM) |
| `hemolysis_risk_prob` | `1 - p_hc50_gt_128` |
| `hc50_*_95` | Uncertainty interval |
| `in_domain` | Applicability-domain flag |
| `solubility_aqueous_probability` | P(soluble) aqueous/buffer (default 0.1 M PBS) |
| `stability_half_life_hours` | PEPlife2 protease half-life |

### Limits

- Exact HC50 correlation is moderate under cluster-disjoint evaluation; use µM ranks as a guide, not an assay substitute.
- Solubility is binary under buffer conditions, not continuous mg/mL.
- Stability is protease-assay specific, not in vivo blood half-life.

Experiment history: `experiments/`.
""",
        encoding="utf-8",
    )


def write_requirements(dst: Path) -> None:
    (dst / "requirements.txt").write_text(
        """numpy>=1.26,<3
pandas>=2.2
scipy>=1.11
scikit-learn>=1.4
biopython>=1.83
torch>=2.2
transformers>=4.40
pyyaml>=6.0
tqdm>=4.66
""",
        encoding="utf-8",
    )


def write_final_yaml(dst: Path) -> None:
    cfg = dst / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "final.yaml").write_text(
        """model: esm2_35M_ft
features: esm2_t12_35M_finetune
version: production_full_data
checkpoint: models/esm2_35M_ft.pt
selection_locked_test:
  pearson: 0.317
  spearman: 0.383
  roc_auc: 0.837
  right_censored_consistency: 0.807
  coverage_90: 0.926
aqueous_solubility:
  source: SolPepBench / PepSol2000
  default_model: solubility_aqueous_pooled.pkl
  cv_roc_auc_pooled: 0.785
  note: NOT E. coli expression
stability:
  source: PEPlife2
  default_model: stability_peplife2_protease.pkl
""",
        encoding="utf-8",
    )


def should_delete(path: Path) -> bool:
    name = path.name
    if name in DROP_DIR_NAMES:
        return True
    if path.suffix.lower() in DROP_FILE_SUFFIXES:
        return True
    low = name.lower()
    return any(s.lower() in low for s in DROP_NAME_SUBSTR)


def clean_tree(root: Path) -> None:
    """Remove LLM/Cursor/cache/log artifacts under root."""
    # Walk bottom-up so dirs can be removed
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        try:
            if should_delete(p):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    for req in FINAL_MODELS:
        if not (SRC / "models" / "final" / req).exists():
            sys.exit(f"Missing models/final/{req}")

    if DST.exists():
        print(f"Removing previous {DST}")
        shutil.rmtree(DST)

    # Staging: create DST skeleton first in a temp sibling, then move archive
    DST.mkdir(parents=True)
    EXP.mkdir(parents=True)
    (EXP / "README.md").write_text(
        "# Experiments archive\n\nTraining history, intermediate models, and comparison tables.\n"
        "Not required for inference.\n\nSee `comparison_tables/`.\n",
        encoding="utf-8",
    )
    write_readme(DST)
    write_requirements(DST)
    write_final_yaml(DST)

    # Inference library
    for rel in PSP_KEEP:
        src_f = SRC / "src" / "psp" / rel
        if src_f.exists():
            dst_f = DST / "src" / "psp" / rel
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)
    write_paths(DST / "src" / "psp")
    (DST / "src" / "psp" / "__init__.py").write_text(
        '"""Peptide safety inference library."""\n', encoding="utf-8"
    )

    # Models
    (DST / "models").mkdir(parents=True, exist_ok=True)
    for name in FINAL_MODELS:
        shutil.copy2(SRC / "models" / "final" / name, DST / "models" / name)

    with (DST / "models" / "hc50_bundle.pkl").open("rb") as f:
        bundle = pickle.load(f)
    bundle["esm_ft_ckpt"] = "models/esm2_35M_ft.pt"
    bundle["version"] = "production_full_data"
    with (DST / "models" / "hc50_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    # Examples + CLI
    ex = SRC / "examples" / "controllability_sequences.csv"
    if ex.exists():
        (DST / "examples").mkdir(exist_ok=True)
        shutil.copy2(ex, DST / "examples" / "controllability_sequences.csv")
    shutil.copy2(SRC / "scripts" / "_package_predict.py", DST / "predict.py")
    shutil.copy2(SRC / "scripts" / "_package_triage.py", DST / "triage.py")

    # Comparison tables
    ct = SRC / "reports" / "comparison_tables"
    if ct.exists():
        shutil.copytree(ct, EXP / "comparison_tables")

    # Move entire old tree into experiments/training_archive
    print(f"Moving {SRC} -> {ARCHIVE}")
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    shutil.move(str(SRC), str(ARCHIVE))

    # Scrub artifacts inside archive
    print("Cleaning LLM/Cursor/log artifacts from experiments...")
    clean_tree(ARCHIVE)

    # Drop non-final model weights from archive models/final (keep history reports)
    arch_models = ARCHIVE / "models" / "final"
    if arch_models.exists():
        keep = set(FINAL_MODELS)
        for p in arch_models.iterdir():
            if p.is_file() and p.name not in keep and p.name != ".gitkeep":
                # keep comparison artifacts but remove intermediate nets / retired
                if p.suffix in {".pt", ".pkl"} and p.name not in keep:
                    p.unlink(missing_ok=True)

    meta = {
        "inference_root": str(DST),
        "experiments": str(EXP),
        "final_models": FINAL_MODELS,
        "n_hc50_production": bundle.get("trained_on", {}),
    }
    (EXP / "package_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Packaged {DST}")
    print("PACKAGE_DONE")


if __name__ == "__main__":
    main()
