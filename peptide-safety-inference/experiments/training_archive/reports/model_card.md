# Model card — Peptide Safety Predictor (HC50)

## Summary

A censoring-aware model of peptide hemolytic activity, trained on DBAASP-derived
HC50 measurements. It outputs a predicted HC50, a 95% prediction interval, and a
calibrated probability that HC50 exceeds 128 µM, together with applicability
domain diagnostics.

**Use it to rank peptides by hemolysis risk. Do not use it as a quantitative
HC50 assay substitute.** On held-out, dissimilar peptides the point regression
has R² ≈ 0 while the 128 µM risk classification has ROC-AUC 0.78 with ECE 0.07.

## Intended use

| Supported | Not supported |
| --- | --- |
| Ranking/triaging candidate peptides by hemolysis risk | Replacing an experimental HC50 measurement |
| Screening against the 128 µM safety threshold | Quantitative HC50 values for regulatory or safety-critical decisions |
| Flagging peptides outside the training distribution | Peptides with D-amino acids, cyclisation, or terminal modifications |
| Canonical linear L-peptides, ~5–50 residues | Proteins, peptoids, heavily modified or conjugated peptides |

## Model

- **Architecture:** Support Vector Regression on 33 deterministic physicochemical descriptors, ensembled over 5 seeds
- **Target:** natural-log HC50 in µM
- **Safety head:** RandomForest classifier for `P(HC50 > 128 µM)`, Platt-calibrated on held-out inner folds, blended 50/50 with the probability implied by the regression distribution
- **Intervals:** conformal, from out-of-fold absolute residuals
- **Applicability domain:** max training sequence identity, ESM-2 (`facebook/esm2_t12_35M_UR50D`) embedding nearest-neighbour distance, descriptor-space distance

No protein language model is needed for the HC50 prediction itself; ESM-2 is
used only for the out-of-distribution score. CPU and GPU results agree to
2.7×10⁻¹⁵.

## Training data

- **Source:** DBAASP v3 REST API, complete crawl of 25,544 peptide records
- **Endpoint:** ~50% hemolysis on erythrocytes only (`50-60% Hemolysis`, `HC50`, `MHC50`). IC50, LD50, EC50 and non-erythrocyte cytotoxicity are excluded.
- **Size:** 2,840 observations / 2,446 unique sequences; 2,003 training sequences
- **Censoring:** 37% right-censored. Right-censored rows enter the likelihood as `P(Y > c)`; they are never assigned the threshold value and fitted with MSE.
- **Excluded chemistry:** peptides with non-canonical residues, terminal modifications or intrachain bonds are excluded from training and flagged as a separate applicability region.
- **Conflicts retained:** 226 sequences have multiple measurements; 89 have exact measurements disagreeing by >2×. These are kept, not averaged, and represent an irreducible noise floor.

## Evaluation protocol

Sequences are clustered into single-linkage connected components at 70% identity
(≥80% coverage of the shorter sequence) and whole clusters are assigned to
splits. The locked test (443 sequences, 18% of clusters) was created before
model selection and scored once.

**Independently audited max train→test identity: 0.697. Violations: 0.**

Random splits were not used for any reported result. For reference, a random
split inflates Pearson from 0.352 to 0.600 on this data.

## Performance on the locked test

### Quantitative HC50 (272 exact observations)

| Metric | Value |
| --- | --- |
| Pearson r | 0.299 (95% CI [0.194, 0.412]) |
| Spearman ρ | 0.368 |
| MAE (log µM) | 1.297 |
| RMSE (log µM) | 1.739 |
| R² | −0.020 |

### P(HC50 > 128 µM) (393 observations)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.780 |
| PR-AUC | 0.760 |
| MCC | 0.451 |
| Balanced accuracy | 0.723 |
| Brier | 0.193 |
| ECE | 0.069 |

### Interval calibration

| Check | Value |
| --- | --- |
| Empirical coverage of nominal 95% interval | 90.8% |
| Right-censored rows whose interval admits values above the bound | 99.4% |

## Known limitations

1. **The point regression does not beat the mean on dissimilar peptides** (R² = −0.020). Use `p_hc50_gt_128`, not `hc50_uM_pred`, for decisions.
2. **Regression to the mean at the extremes.** MAE by true HC50 band: 1.72 (<32 µM), 0.76 (32–128 µM), 1.10 (128–512 µM), **3.09 (>512 µM)**. The model is least reliable precisely for the safest peptides a design campaign is looking for.
3. **Error grows with length:** MAE 1.12 (≤15 residues) → 1.42 (26–40 residues); the >40 bin holds only 7 peptides and is not interpretable.
4. **Error grows with dissimilarity:** MAE 1.54 at 0.3–0.5 max train identity vs 1.14 at 0.5–0.7.
5. **Assay heterogeneity is unmodelled.** Training data pools different erythrocyte sources, pH and ionic strengths from hundreds of publications.
6. **Modified chemistry is unsupported.** Predictions for peptides with D-amino acids, cyclisation or terminal modifications are extrapolation.
7. **The 128 µM threshold is a convention**, not a biological constant.

## Solubility and stability columns — read this

`solubility_proxy_score` and `stability_proxy_score` are **unvalidated
physicochemical heuristics**, not trained models.

No labelled dataset for either endpoint could be obtained: PEPlife offers no
bulk download and the PeptideBERT clone contains no data. Therefore:

- `solubility_proxy_score` is a CamSol-like descriptor. It is **not**
  experimentally calibrated aqueous solubility, **not** E. coli soluble-expression
  probability, and has **no** measured accuracy.
- `stability_proxy_score` is a protease-cleavage/oxidation-site propensity score.
  It is **not** a half-life, and not specific to plasma, serum, SGF or SIF.

Neither should be used for any decision that assumes calibration against
experiment.

## Comparison with external predictors

HemoPI2 is the only external predictor that could be executed. It scores
Pearson 0.914 on locked-test sequences present in its own training data and
0.262 on sequences absent from it — a memorisation gap of 0.652. Our model, which
has seen no test sequence, scores 0.299 overall (95% CI [0.194, 0.412]) — a
difference from HemoPI2's unseen-subset score that is not statistically
meaningful.

The honest conclusion is that both models predict HC50 poorly on genuinely novel
peptides, and that published performance differences in this field are dominated
by evaluation protocol rather than model quality.

## Ethical and safety considerations

This model must not be the sole basis for deciding a peptide is safe for
clinical, therapeutic or in-vivo use. A low predicted hemolysis risk is a
screening signal for prioritising experiments, not evidence of safety. Given the
elevated error for high-HC50 peptides (limitation 2), false reassurance is the
most likely harmful failure mode.

## Reproducibility

Python 3.11.9, torch 2.6.0+cu124, GTX 1060 6 GB. Exact versions in
`reports/environment.json`; external repo SHAs in `reports/external_audit.json`;
raw data checksums in `data/raw/MANIFEST.json`; split assignments in
`data/splits/`.

The locked test was scored three times. Both deviations — an invalid calibrator
selection, and a code path that fitted ExtraTrees while the frozen config said
SVR — are documented in full in section N of `reports/model_comparison.md`.
