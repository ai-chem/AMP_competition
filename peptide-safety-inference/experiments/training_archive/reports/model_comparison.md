# HC50 prediction: model comparison and honest generalization assessment

This report covers sections A–Q of the brief. Every number here was produced by
the scripts in `scripts/`; nothing is quoted from a paper as if it were our own
result. Where a method could not be run, the reason is stated rather than
substituted with an estimate.

**The single most important result in this report** is that a random
train/test split makes the same model look roughly twice as good as it is, and
that the best published external predictor collapses from Pearson 0.91 to 0.26
once you score it on peptides it did not train on. Section E and section I
give the numbers.

---

## A. Datasets actually obtained

| Source | What we got | Rows used | Censoring preserved? |
| --- | --- | --- | --- |
| DBAASP v3 REST API (`https://dbaasp.org/peptides`) | 25,544 peptide detail JSONs, complete crawl, 0 errors | 2,840 model-ready HC50 observations | **Yes** — raw `concentration` strings such as `>128` retained |
| HemoPI2 (vendored repo `Dataset/`) | 1,926 curated sequences | secondary / cross-check only | No — authors collapsed censoring |
| ConsAMPHemo (`Dataset/regression/Hemo_regression.csv`) | 1,355 quantitative rows | secondary / cross-check only | No — authors averaged to a single value |
| Hemolytik / Hemolytik 2.0 | not available as a bulk download | — | — |
| PEPlife (half-life) | website reachable, **no bulk download endpoint**; browse-only PHP interface | 0 | — |
| PeptideBERT solubility data | clone contains only `requirements.txt`; data is behind an external drive link | 0 | — |

DBAASP is the only source that preserves right-censoring together with
per-measurement provenance (PubMed ID, assay pH, ionic strength, target cell),
which is why it is the primary source and the other two are used only for
cross-checking.

### Endpoint filtering

Of 25,544 peptide records, the hemolytic/cytotoxic activity tables contain many
endpoints that are *not* HC50. We keep only ~50% hemolysis on erythrocytes:

| Endpoint in DBAASP | Kept? |
| --- | --- |
| `50-60% Hemolysis`, `HC50`, `MHC50` | yes |
| `IC50`, `LD50`, `EC50`, `50% Cell death` | no — different endpoint |
| `0-10%`, `10-20%`, `90-100% Hemolysis` | no — not a 50% endpoint |
| non-erythrocyte target cells (HEK293, HUVEC, HaCat, fibroblasts) | no — cytotoxicity, not hemolysis |

### Processing outcomes

| Decision | Rows |
| --- | --- |
| `ok` | 3,851 |
| `unit_conversion_failed` | 400 |
| `ugml_conversion_skipped_modified` | 279 |
| `unparsable_concentration` | 49 |

µg/mL is converted to µM only when the molecular weight is computable from a
canonical linear sequence. For peptides with non-canonical residues, terminal
modifications or cyclisation the conversion is **skipped rather than guessed**,
and those rows are excluded from the modelling table and flagged as a separate
applicability region.

Final modelling table: **2,840 observations / 2,446 unique sequences**.

---

## B. Censoring: distribution and treatment

| Censor type | Train obs | Test obs |
| --- | --- | --- |
| exact | 1,462 | 301 |
| right (`>128`, `>=256`, …) | 858 | 186 |
| interval (ranges) | 15 | 3 |
| left (`<x`) | 14 | 1 |

**37% of the data is right-censored.** Assigning those rows the numeric value of
the censoring threshold and fitting ordinary MSE — which is what a naive
pipeline does — is not done anywhere in this project. Right-censored rows
contribute `P(Y > c) = 1 - Φ((c-µ)/σ)` to the likelihood.

Exact HC50 quantiles (µM): 5% = 3.5, 25% = 24.1, **median = 64.9**, 75% = 168.2, 95% = 630.9.

The `P(HC50 > 128 µM)` label is recomputed from the quantitative values and the
censoring bounds, **not** inherited from HemoPI2's 100 µM threshold. A row
censored at `>128` is a known positive; a row censored at `>64` is *unknown*
with respect to the 128 µM threshold and is left as NaN rather than guessed.
2,510 of 2,840 rows have a determinable label.

Conflicting measurements are kept, not averaged: 226 sequences have multiple
measurements and 89 have exact measurements disagreeing by more than 2×. This
disagreement is an irreducible noise floor for any model on this data.

---

## C. Leakage control and split construction

Sequences are clustered by **single-linkage connected components** at 70%
identity with ≥80% coverage of the shorter sequence. Single linkage is used
deliberately: with greedy centroid clustering (the first implementation here),
two mutually-similar sequences can attach to different centroids and end up on
opposite sides of the split. Connected components make "no test sequence
exceeds the threshold against any training sequence" true *by construction*
rather than something to hope for.

Identity is `matches / total alignment columns including gaps`. This matters:
under the more common `matches / ungapped columns` definition a 7-mer that is a
subsequence of a 30-mer scores 100% identity and chains unrelated clusters
together. Our first attempt with that definition collapsed 2,446 sequences into
95 giant clusters.

| Quantity | Value |
| --- | --- |
| Clusters @70% identity | 809 |
| Clusters @80% identity (sensitivity) | 1,105 |
| Train sequences / observations | 2,003 / 2,349 |
| Locked test sequences / observations | 443 / 491 (18% of clusters) |
| **Max train→test identity (independently audited)** | **0.697** |
| **Violations of the 70% threshold** | **0** |

The audit in `make_splits.py` is an independent all-vs-all check that does not
reuse the cluster assignments, and it uses only an exact length gate — no k-mer
heuristic — so it cannot inherit a blind spot from the clusterer. `make_splits.py`
exits non-zero on any violation; `reports/split_audit.csv` lists the nearest
training neighbour, identity, coverage, same-source and same-publication flags
for all 443 test sequences.

The locked test was built **before** any model was selected and was scored
exactly once (see section T for the one disclosed exception).

---

## D. Approaches evaluated

All validation numbers below are out-of-fold predictions from
`StratifiedGroupKFold` **inside the training portion only**, grouped by 70%
identity cluster. 223 runs are logged in `reports/all_experiments.csv`.

### Top 20 by mean validation Pearson r

| Family | Model | Features | Pearson (mean ± sd) | MAE | Seeds |
| --- | --- | --- | --- | --- | --- |
| classical | **SVR** | **physchem** | **0.4308 ± 0.0113** | 1.168 | 5 |
| hybrid | XGBoost | physchem + ESM-2 | 0.4283 ± 0.0050 | 1.118 | 3 |
| hybrid | CatBoost | physchem + ESM-2 | 0.4252 ± 0.0045 | 1.123 | 3 |
| classical | SVR | full | 0.4231 ± 0.0100 | 1.173 | 5 |
| *finetune* | *ESM-2 35M fine-tune* | *ESM-2* | *0.4179 (single holdout)* | *1.104* | *1* |
| frozen_plm | CatBoost | ESM-2 mean | 0.4148 ± 0.0314 | 1.138 | 5 |
| classical | XGBoost | full | 0.4065 ± 0.0319 | 1.149 | 5 |
| hybrid | LightGBM | physchem + ESM-2 | 0.4064 ± 0.0219 | 1.136 | 5 |
| censored | XGBoost `survival:aft` | physchem | 0.4011 | 1.283 | 1 |
| frozen_plm | MLP | ESM-2 mean | 0.3988 ± 0.0158 | 1.185 | 5 |
| frozen_plm | XGBoost | ESM-2 mean | 0.3974 ± 0.0261 | 1.146 | 5 |
| classical | LightGBM | full | 0.3948 ± 0.0192 | 1.168 | 5 |
| classical | XGBoost | physchem | 0.3916 ± 0.0258 | 1.162 | 5 |
| frozen_plm | LightGBM | ESM-2 mean | 0.3913 ± 0.0222 | 1.149 | 5 |
| classical | CatBoost | full | 0.3910 ± 0.0265 | 1.149 | 5 |
| classical | CatBoost | physchem | 0.3902 ± 0.0249 | 1.150 | 5 |
| classical | LightGBM | physchem | 0.3889 ± 0.0192 | 1.171 | 5 |
| classical | RF | physchem | 0.3706 ± 0.0125 | 1.158 | 5 |
| classical | RF | full | 0.3688 ± 0.0137 | 1.155 | 5 |
| classical | ExtraTrees | physchem | 0.3686 ± 0.0109 | 1.155 | 5 |

**The top four entries are not distinguishable.** SVR/physchem at 0.4308 ± 0.0113
and hybrid XGBoost at 0.4283 ± 0.0050 overlap well within one standard
deviation across seeds. Declaring SVR "the best model" would overstate what
this data can resolve; it was chosen as the top-ranked entry under a
pre-specified rule, and its simplicity (33 deterministic descriptors, no PLM
dependency at prediction time) is a tie-breaker in its favour, not evidence of
superiority.

### Censored-likelihood models

| Model | Censored NLL ↓ | Pearson (exact rows) | Psafe AUC |
| --- | --- | --- | --- |
| Constant baseline | 1.786 | −0.136 | 0.471 |
| **Tobit Gaussian** | **1.520** | 0.279 | 0.740 |
| Log-normal AFT (lifelines) | — | 0.277 | — |
| XGBoost `survival:aft` | — | **0.401** | — |

The Tobit model improves censored NLL substantially over the constant baseline
(1.520 vs 1.786), confirming the censored likelihood is doing real work. Its
point-prediction Pearson is lower than the tree models because it is linear —
these two facts are not in conflict, they are measuring different things.
XGBoost `survival:aft` is the strongest censoring-aware point predictor at 0.401.

### Sequence-composition and deep models

| Model | Pearson |
| --- | --- |
| 1D CNN | 0.089 |
| Multiscale CNN (LysePred-style, kernels 2–32) | 0.149 |
| BiLSTM | 0.317 |
| AAC-only (best: SVR) | 0.251 |

Trained-from-scratch sequence models underperform descriptors on this dataset
size (2,003 training sequences). This is an honest negative result, not a tuning
failure to be hidden: with ~2k sequences there is not enough signal to learn a
representation that beats 33 hand-computed physicochemical descriptors.

### ESM-2 fine-tuning (the one documented attempt)

The ESM-2 35M fine-tune reached validation Pearson **0.456** at its best epoch
with censored NLL still decreasing (1.984 → 1.275 over 8 epochs) — the best
single validation number observed anywhere in this project.

**It was nonetheless excluded from model selection**, and this is the most
consequential judgement call in the report. It was scored on a single grouped
holdout (n = 173) because 5-fold fine-tuning was outside the compute budget,
while every other candidate was scored on full out-of-fold predictions
(n = 1,223). A single-holdout Pearson and a 5-fold out-of-fold Pearson are not
the same statistic, and the holdout is small enough that 0.456 carries a wide
confidence interval. Picking it because its number is highest would be exactly
the "choose a model for a nice metric" error the brief forbids. The exclusion is
enforced in code (`evaluate.py` requires ≥90% out-of-fold coverage) and recorded
in `reports/model_selection_excluded.csv`.

The fine-tune remains the most promising direction for future work, and the
still-falling NLL suggests it was under-trained rather than converged.

---

## E. Leakage diagnostics: random split vs cluster-disjoint split

Same model (ExtraTrees), same data, same 5 seeds — only the fold assignment
differs. Both run inside the training portion, so the locked test is untouched.

| Split scheme | Pearson r | MAE | R² |
| --- | --- | --- | --- |
| **Random 5-fold** | **0.600 ± 0.010** | 0.957 | 0.359 |
| **Cluster-disjoint 5-fold (70% id)** | **0.352 ± 0.005** | 1.165 | 0.110 |
| **Optimism gap** | **+0.248** | −0.208 | **+0.249** |

A random split inflates Pearson by 0.25 and R² by 0.25 — it makes a model that
explains 11% of variance look like it explains 36%. The seed-to-seed spread
(±0.01) is 25× smaller than the gap, so this is not noise.

**Any hemolysis paper reporting random-split metrics should be assumed to be
reporting something closer to the 0.600 column than the 0.352 column.**

---

## F. Locked test evaluation (scored once)

Selected configuration, frozen to `configs/final.yaml` before evaluation: SVR on
33 physicochemical descriptors, 5-seed ensemble, Platt-calibrated P(HC50>128)
head blended with the regression-derived probability, conformal 95% intervals.

### Quantitative HC50 (272 exact test observations)

| Metric | Value |
| --- | --- |
| Pearson r | **0.299** |
| Pearson 95% CI (cluster bootstrap, 500×) | **[0.194, 0.412]** |
| Spearman ρ | 0.368 |
| MAE (log µM) | 1.297 |
| RMSE (log µM) | 1.739 |
| R² | **−0.020** |

### Binary P(HC50 > 128 µM) (393 test observations with a determinable label)

| Metric | Value |
| --- | --- |
| ROC-AUC | **0.780** |
| PR-AUC | 0.760 |
| MCC | 0.451 |
| Balanced accuracy | 0.723 |
| Brier score | 0.193 |
| ECE | 0.069 |

### Uncertainty calibration

| Check | Value |
| --- | --- |
| Empirical coverage of nominal 95% interval | **90.8%** |
| Right-censored rows whose interval admits values above the bound | **99.4%** |
| Conformal 95% quantile (log µM) | 2.898 |

**Read these two blocks together.** The point regression on the locked test is
weak: R² is slightly negative, meaning it does not beat predicting the mean, and
the confidence interval on Pearson reaches down to 0.19. The binary safety
classification is genuinely useful (AUC 0.780, MCC 0.451, ECE 0.069), and the
intervals are close to honest at 90.8% empirical coverage against a 95% nominal
target — slightly over-confident, so the stated intervals are if anything a
little too narrow.

**The practical conclusion is that this system should be used as a calibrated
risk-ranking tool for the 128 µM safety threshold, not as a quantitative HC50
regressor.** The predicted HC50 value should be treated as a rough
order-of-magnitude indication and always read together with its (wide) interval.

Spearman (0.368) is meaningfully higher than Pearson (0.299): the model ranks
peptides better than it places them on an absolute scale, which is consistent
with recommending it for triage rather than measurement.

Validation Pearson was 0.431 and locked-test Pearson is 0.299. That drop is
itself informative: the locked test is stratified and cluster-disjoint, and the
gap shows that even cluster-disjoint cross-validation retains some optimism.

---

## G. Comparison with external predictors

Results are kept strictly per-predictor. No metric here is compared against a
number quoted from a paper, because no two papers use the same split, the same
endpoint definition, or the same censoring treatment.

| Predictor | Ran on our locked test? | Pearson | Status |
| --- | --- | --- | --- |
| **HemoPI2** (regressor, vendored `.sav`) | **yes** | 0.667 | **potentially contaminated** |
| ConsAMPHemo | no | — | no checkpoint artifacts in repo |
| PeptideBERT | no | — | clone has no data and no checkpoint |
| LysePred | no | — | only a token dictionary; retraining-oriented |
| HemoNet | no | — | `weights.hdf` present, inference API not wired |
| ML-guided non-hemolytic peptides | no | — | 9 binary classifiers, not HC50 regressors |
| AmpLyze | no | — | **methodological reference only**, no checkpoint released |
| Plisson et al. | no | — | no usable released checkpoint |

Only HemoPI2 could be executed end-to-end. Every other predictor is recorded
with a specific reason rather than a fabricated score. Details and commit SHAs
are in `reports/external_audit.md`.

---

## H. Contamination audit — and what it actually shows

HemoPI2 was trained on DBAASP + Hemolytik. Our locked test is DBAASP-derived.
The overlap is therefore **structural, not accidental**: 289 of our 443 locked
test sequences (65%) appear verbatim in HemoPI2's curated training files.

Because we know exactly which sequences those are, the contamination can be
*measured* rather than merely flagged. Splitting HemoPI2's own predictions on
our locked test:

| Subset | n | Pearson r | R² | MAE |
| --- | --- | --- | --- | --- |
| Sequences **in** HemoPI2's training data | 161 | **0.914** | 0.819 | 0.456 |
| Sequences **not** in HemoPI2's training data | 111 | **0.262** | 0.031 | 1.405 |
| **Memorisation gap** | | **0.652** | 0.788 | |

This is the clearest single result in the project. HemoPI2 looks like a
near-perfect HC50 predictor (r = 0.91, R² = 0.82) on peptides it has seen, and
essentially stops working (r = 0.26, R² = 0.03) on peptides it has not. Its
headline-grade performance is memorisation.

For reference, our own model scores r = 0.299 (95% CI [0.194, 0.412]) across the
whole locked test, having never seen *any* test sequence. HemoPI2 scores 0.262
on the subset it did not train on. Those two figures are statistically
indistinguishable — the CI on our estimate comfortably contains 0.262, and
HemoPI2's 0.262 is itself computed on only 111 observations.

The honest reading is therefore not "we beat HemoPI2". It is **that both models
predict HC50 about equally poorly on genuinely novel peptides, and that the
large apparent quality gap one would infer from published metrics is an
artefact of evaluation protocol rather than a real difference in capability.**

---

## I. Failure modes and subgroup analysis

Absolute residuals on the locked test (log µM):

**By peptide length** — error grows with length up to 40 residues:

| Length | MAE | n |
| --- | --- | --- |
| ≤15 | 1.118 | 76 |
| 16–25 | 1.361 | 147 |
| 26–40 | 1.422 | 42 |
| >40 | 1.172 | 7 |

The >40 bin contains only 7 peptides, so its apparent recovery is not meaningful.

**By similarity to training data** — closer is better, as expected:

| Max train identity | MAE | n |
| --- | --- | --- |
| 0.3–0.5 | 1.540 | 108 |
| 0.5–0.7 | 1.137 | 164 |

**By true HC50 band** — the model regresses toward the middle:

| HC50 band (µM) | MAE | n |
| --- | --- | --- |
| <32 (most toxic) | 1.724 | 101 |
| 32–128 | 0.764 | 98 |
| 128–512 | 1.096 | 61 |
| >512 (safest) | **3.095** | 12 |

**By applicability domain:**

| In domain | MAE | n |
| --- | --- | --- |
| True | 1.295 | 269 |
| False | 1.538 | 3 |

The dominant failure mode is **regression to the mean at the extremes**: the
model is accurate in the 32–128 µM band where most training data sits and is
badly wrong for the safest peptides (>512 µM, MAE 3.09 in log space — roughly a
20× error). Since the safest peptides are exactly the ones a design campaign
wants to find, this is a serious practical limitation and a direct argument for
using the calibrated `p_hc50_gt_128` output instead of the point estimate.

The OOD flag points the right way (MAE 1.54 out of domain vs 1.29 in domain) but
only 3 of 272 test peptides fell outside the domain, so this is too small a
sample to claim the applicability domain has been validated.

---

## J. Solubility and stability

**No labelled dataset for either endpoint could be obtained.** PEPlife has no
bulk download endpoint, and the PeptideBERT clone contains no data files. No
solubility or half-life model was trained.

What `predict.py` emits instead:

| Column | What it is | What it is **not** |
| --- | --- | --- |
| `solubility_proxy_score` | A CamSol-like physicochemical score computed from the sequence | Not experimentally calibrated aqueous solubility. Not an E. coli soluble-expression prediction. Not validated against any measurement. |
| `stability_proxy_score` | A protease-cleavage-site / oxidation-site propensity score | Not a measured half-life. Not plasma, serum, SGF or SIF stability. Not validated. |

Both are named `proxy`, both carry an `endpoint` column documenting what they
are, and neither has an associated accuracy claim because there is no ground
truth to measure accuracy against. See `reports/solubility_stability_audit.json`.

---

## K. MIC

Out of scope for this run by explicit decision. No MIC head was trained and no
AMP/non-AMP classifier is presented as a MIC predictor.

---

## L. Excluded methods and why

| Method | Reason for exclusion |
| --- | --- |
| Hemolytik / Hemolytik 2.0 | no programmatic bulk download |
| PEPlife half-life | browse-only PHP interface, no bulk export |
| PeptideBERT solubility/hemolysis | clone contains no data and no checkpoint |
| ConsAMPHemo inference | no checkpoint artifacts in the repository |
| HemoNet, LysePred, ML_Peptide inference | checkpoints exist but the inference API/environment could not be wired up reliably |
| AmpLyze | methodological reference only; no checkpoint released |
| mmseqs2 / CD-HIT clustering | not available on Windows; replaced with an in-process aligner-based connected-components clusterer |
| ESM-2 650M and ProtT5/ProtBERT | 6 GB VRAM (GTX 1060, Pascal sm_61) budget |
| 5-fold ESM-2 fine-tuning | compute budget; one single-holdout run was done and is reported |
| MIC heads | explicitly out of scope for this run |

---

## M. Reproducibility

- Python 3.11.9, torch 2.6.0+cu124, CUDA available on a GTX 1060 6 GB (sm_61)
- Exact package versions: `reports/environment.json`
- External repo commit SHAs: `reports/external_audit.json`
- Raw data checksums and retrieval timestamps: `data/raw/MANIFEST.json`
- PLM: `facebook/esm2_t12_35M_UR50D`, frozen, mean pooling, embeddings cached by content hash
- Split assignments saved to `data/splits/` so the locked test is reconstructible
- Seeds fixed; the 5 best models were run with 5 seeds each

---

## N. Disclosed deviations from single-evaluation discipline

The locked test was scored **three times**, not once. Both deviations are
disclosed here rather than quietly presented as a single clean evaluation, and
every intermediate metrics file is preserved.

### Deviation 1 — invalid calibrator selection

The first run compared Platt against isotonic calibration by computing ECE on
the same out-of-fold predictions they had been fitted to. Isotonic can
interpolate those points, so its apparent ECE was 5×10⁻¹⁷ and it won
automatically. Fixed by selecting the calibrator on held-out inner folds, after
which Platt was chosen.

This changed nothing about the regression model — test Pearson was identical at
0.2364 before and after — and the corrected procedure produced a slightly
*worse* test ECE (0.0748 → 0.0796), which is what removing an optimistic bias
should do. Pre-fix metrics: `reports/final_test_metrics_preCalibFix.json`.

### Deviation 2 — the frozen config did not match the fitted model

The more serious one. Model selection chose `svr`, but `make_regressor()` in
`evaluate.py` had no `svr` branch and ended in a bare
`return ExtraTreesRegressor(...)` fallback. The first two locked-test
evaluations therefore **fitted and scored ExtraTrees while the frozen config,
the bundle metadata and the report all said SVR.** The bundle's 257 MB size —
five 500-tree forests, not five SVRs — is what exposed it.

Fixing this means the deployed model now matches the pre-specified selection
rule. The silent fallback has been replaced with an explicit `ValueError`, so a
selected-but-unconstructible model now fails loudly instead of shipping the
wrong estimator.

Effect on locked-test results:

| Metric | Run 2 (ExtraTrees, mislabelled) | Run 3 (SVR, as selected) |
| --- | --- | --- |
| Pearson r | 0.236 | **0.299** |
| Spearman ρ | 0.194 | **0.368** |
| R² | −0.026 | −0.020 |
| Psafe ROC-AUC | 0.761 | **0.780** |
| Psafe MCC | 0.399 | **0.451** |
| Psafe ECE | 0.080 | **0.069** |
| Interval coverage | 92.3% | 90.8% |

Pre-fix metrics: `reports/final_test_metrics_preSvrFix.json`.

### How to read this

The reported figures improved after the second fix, and readers are entitled to
be sceptical of a bug whose correction happened to help. Two points are
relevant. First, the selection of SVR was made on validation data before any
locked-test evaluation, and is recorded in `reports/model_selection_validation.csv`
— run 3 implements that pre-existing decision rather than substituting a
better-scoring model discovered afterwards. Second, no feature set,
hyperparameter, ensemble weight, split, or selection rule was altered at any
point after the locked test was first viewed; the only changes were making the
code execute the configuration that had already been frozen.

The conservative reading is that the locked test has now been observed three
times and has accordingly lost some of its status as a never-seen holdout. A
fully independent confirmation would require a fresh test set.

---

## O. Final choice and justification

**SVR on 33 deterministic physicochemical descriptors**, 5-seed ensemble, with:
- a Platt-calibrated `P(HC50 > 128 µM)` head blended 50/50 with the probability implied by the regression distribution,
- conformal 95% prediction intervals (empirical coverage 92.3%),
- an applicability domain combining max training identity, ESM-2 embedding distance and descriptor-space distance.

Chosen because it was top-ranked under a pre-specified, leakage-controlled,
multi-seed rule; because it is stable across seeds (±0.0113, among the lowest in
the table); and because it needs no PLM at prediction time, which makes CPU-only
inference fast and fully deterministic (CPU and GPU agree to 2.7×10⁻¹⁵).

It was **not** chosen because it is the best conceivable model. The hybrid
ESM-2 + descriptor models are statistically tied with it, and the ESM-2
fine-tune may well be better but was not evaluated comparably.

---

## P. What this system should and should not be used for

**Appropriate:** ranking candidate peptides by hemolysis risk against the 128 µM
threshold; triaging a design library; flagging peptides that fall outside the
training distribution.

**Not appropriate:** treating `hc50_uM_pred` as a quantitative assay
substitute (R² ≈ 0 on held-out dissimilar peptides); predicting the HC50 of
peptides with D-amino acids, cyclisation or terminal modifications (excluded
from training); any solubility or stability decision based on the proxy columns.

---

## Q. Artefacts

| File | Contents |
| --- | --- |
| `reports/all_experiments.csv` | all 223 validation runs |
| `reports/model_selection_validation.csv` | aggregated across seeds |
| `reports/model_selection_excluded.csv` | candidates excluded for non-comparable protocol |
| `reports/leakage_diagnostics.{csv,json}` | random vs cluster-disjoint gap |
| `reports/split_audit.csv` | nearest training neighbour for all 443 test sequences |
| `reports/contamination_audit.json` | HemoPI2 overlap and seen-vs-unseen breakdown |
| `reports/external_baseline_results.csv` | per-predictor external results |
| `reports/final_test_metrics.json` | the locked-test evaluation |
| `reports/final_test_predictions.csv` | per-sequence locked-test predictions |
| `reports/final_test_residual_analysis.csv` | per-sequence residuals and subgroup keys |
| `reports/figures/` | prediction-vs-observation, residuals, reliability curve |
| `reports/data_audit.md` | dataset provenance and censoring statistics |
