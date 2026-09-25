# Peptide Safety Inference

Predict peptide **hemolytic safety (HC50)**, **aqueous/buffer solubility**, and
**protease half-life** from amino-acid sequence.

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
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
