# Final predictor v7

## How to run

```bash
python peptide_safety_predictor/predict.py \
  --input your_sequences.csv \
  --output predictions.csv \
  --sequence-col sequence \
  --device auto

python peptide_safety_predictor/scripts/triage_peptides.py \
  --input your_sequences.csv \
  --top 30
```

Bundle: `peptide_safety_predictor/models/final/hc50_bundle.pkl`  
Checkpoint: `models/final/esm2_35M_ft.pt`

## What it predicts

| Column | Meaning |
| --- | --- |
| `hc50_uM_pred` | Predicted HC50 (µM) from ESM-2 35M fine-tune (censored Gaussian head) |
| `p_hc50_gt_128` | P(HC50 > 128 µM) — ExtraTrees classifier + regression CDF blend |
| `hemolysis_risk_prob` | 1 − p_hc50_gt_128 |
| `hc50_*_95` | Uncertainty interval (model σ + conformal floor) |
| `in_domain` | Applicability-domain flag |
| `solubility_aqueous_probability` | P(soluble) in aqueous/buffer (SolPepBench; default 0.1 M PBS) |
| `solubility_endpoint` | Explicit solvent label — **not** E. coli expression |
| `stability_half_life_hours` | PEPlife2 protease-assay half-life — assay-specific |

## Selection (before locked test)

| Candidate | Selection metric | Value |
| --- | --- | --- |
| CatBoost physchem+ESM (group CV OOF) | exact pearson | 0.352 |
| ESM-2 35M FT + censored head (GroupShuffleSplit val) | exact pearson | **0.398** |

ESM-FT selected as primary regressor. Classifier/calibrator from v5 ExtraTrees on physchem+ESM.

## Locked-test result (seed 20260925, scored once)

6,052 train / 896 test sequences (cluster-disjoint, max identity repaired).

| Metric | Value |
| --- | --- |
| Exact pearson / spearman / R² | 0.317 / 0.383 / −0.138 |
| Safe AUC / PR-AUC / MCC | **0.837** / — / **0.545** |
| Right-censored consistency | **0.807** |
| 90% interval coverage | 0.926 |

### Trajectory vs earlier heads (same locked split)

| Version | Pearson | Spearman | AUC | RC cons. |
| --- | ---: | ---: | ---: | ---: |
| v2 SVR | 0.312 | 0.340 | 0.796 | 0.415 |
| v3 SVR+Tobit | 0.313 | 0.357 | 0.811 | 0.483 |
| v5 CB+ET+Tobit | 0.302 | 0.320 | 0.803 | 0.407 |
| **v7 ESM-FT** | **0.317** | **0.383** | **0.837** | **0.807** |

## Aqueous solubility (SolPepBench / PepSol2000)

E. coli expression labels are **retired** (`solubility_ecoli_rf.RETIRED.pkl`).

| Solvent model | n | CV ROC-AUC |
| --- | ---: | ---: |
| Ultrapure water | 1326 | 0.782 |
| 0.1 M PBS | 635 | 0.773 |
| 1X DPBS | 667 | 0.779 |
| Pooled (default inference) | 2664 | 0.785 |

Binary soluble/insoluble under the named solvent — not continuous mg/mL.

## Honest limits

- Exact HC50 pearson ~0.32 under cluster-disjoint eval — hemolysis is hard; do not claim quantitative µM accuracy.
- Safety classification (AUC 0.84) and right-censor consistency (0.81) are the stronger operational signals.
- Solubility is binary aqueous/buffer, not expression titer.
- Stability is PEPlife2 protease assay, not in vivo blood half-life.
