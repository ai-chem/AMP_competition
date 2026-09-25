# Final predictor v2 (expanded data)

## How to run

```bash
python peptide_safety_predictor/predict.py \
  --input your_sequences.csv \
  --output predictions.csv \
  --sequence-col sequence \
  --device auto

# Optional triage / shortlist
python peptide_safety_predictor/scripts/triage_peptides.py \
  --input your_sequences.csv \
  --top 30
```

Bundle: `peptide_safety_predictor/models/final/hc50_bundle.pkl`

## What it predicts

| Column | Meaning |
| --- | --- |
| `hc50_uM_pred` | Predicted HC50 (µM), log-space SVR ensemble |
| `p_hc50_gt_128` | P(HC50 > 128 µM) — calibrated classifier blended with regression |
| `hemolysis_risk_prob` | 1 − p_hc50_gt_128 |
| `hc50_*_95` | Conformal/ensemble uncertainty interval |
| `in_domain` | Applicability-domain flag |
| `solubility_ecoli_probability` | P(soluble in E. coli expression) — **not aqueous solubility** |
| `stability_half_life_hours` | PEPlife2 protease-assay half-life — **assay-specific** |

## Locked-test result (seed 20260925, scored once)

Expanded dataset: 14,504 observations / 6,948 sequences → 6,052 train / 896 test
(max train↔test identity 0.697, 0 leakage violations after repair).

| Metric | Value |
| --- | --- |
| Model | SVR / physchem+ESM-2 35M (5-seed) |
| CV selection pearson | 0.389 |
| **Test pearson** | **0.312** [0.231, 0.395] cluster bootstrap |
| Test spearman | 0.340 |
| Test R² | 0.065 |
| P(HC50>128) ROC-AUC | 0.796 |
| P(HC50>128) PR-AUC | 0.788 |
| Interval coverage (nominal 90%) | 0.589 — **under-covered, treat intervals as relative** |
| Right-censored consistency | 0.415 — **weak; do not rely on floor consistency** |

## Controllability demo

319 sequences → 71 shortlist / 17 review / 231 reject.
Top shortlist: `reports/triage_shortlist.csv`.

## Honest limits

- Homology-controlled HC50 regression remains hard (QMAP published baseline mean PCC 0.07).
- Solubility is E. coli expression, not aqueous solubility.
- Stability is PEPlife2 protease half-life; plasma head is weak (CV R²≈0).
- Previous seed-42 locked test is retired (was observed 3×).
