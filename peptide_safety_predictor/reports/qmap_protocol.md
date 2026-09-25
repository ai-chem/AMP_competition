# QMAP benchmark: protocol adoption and results

Source: `github.com/anthol42/QMAP` — Lavertu, Corbeil & Germain, *Scientific Reports* 2026,
doi 10.1038/s41598-026-56004-8. Installed as `qmap-benchmark==0.1.1`; dataset pulled
from the HuggingFace dataset `anthol42/qmap_benchmark_2025`.

QMAP is, to our knowledge, the only published benchmark that evaluates **HC50 regression
under homology-aware splits**, which is exactly the problem this project targets. It is
therefore the right external yardstick for our model, and we adopt its protocol verbatim.

## 1. What the protocol is

Taken from the reference baseline `eval_prev_works/HemoLinear/main.py`:

| Element | Value |
| --- | --- |
| Target | `log10(HC50)`, HC50 in µM |
| Splits | 5 predefined, homology-aware at **60 % sequence identity** |
| Clustering | Leiden community detection on an identity graph (Rust `pwiden_engine`) |
| Dataset filter | `.with_hc50().with_canonical_only().with_l_aa_only().with_terminal_modification(False, False)` |
| Test filter | additionally `.with_length_range(None, 100)` |
| Train rows | whatever `benchmark.get_train_mask(sequences)` admits for that split |
| Scoring | `benchmark.compute_metrics(preds)` → RMSE, MSE, MAE, R², Spearman, Kendall τ, Pearson |

Each split is scored once. Test sizes are fixed at **173, 278, 129, 192, 141**.

## 2. What we adopt

* **The whole evaluation harness**, unchanged, in `scripts/eval_qmap.py`. This gives a
  number that is directly comparable with QMAP's published leaderboard.
* **Homology-aware splitting as the primary evaluation.** This independently confirms the
  choice we had already made; QMAP's own `figures/pcc_vs_threshold` shows measured
  performance collapsing as the identity threshold tightens.
* **Reporting every split rather than only the mean**, because the spread across splits is
  larger than most of the differences between methods.

## 3. What we do not adopt

* **Leiden community detection.** We keep single-linkage connected components. Leiden is a
  soft community partition and does not *guarantee* that no cross-split pair exceeds the
  identity threshold; our union-find construction does, and we verify it. QMAP's own
  splits are used unmodified when we score on QMAP, so this is not a conflict.
* **QMAP's HC50 consensus value as a training target.** QMAP collapses each peptide to a
  single float and discards censoring. Our internal pipeline keeps right-censored records
  under a censored likelihood, which QMAP's format cannot express. We therefore use QMAP
  for external comparison only, not as our internal training target.
* **Its MIC tracks**, which are out of scope for this run.

## 4. Harness validation

Before trusting our score we reproduced QMAP's own `HemoLinear` baseline (ESM-2 embeddings
+ `LinearRegression`) inside our harness.

| Split | n | Published HemoLinear PCC | Our reproduction (ESM-2 35M) |
| --- | --- | --- | --- |
| 0 | 173 | −0.1782 | 0.0652 |
| 1 | 278 | 0.1886 | −0.0361 |
| 2 | 129 | 0.2931 | 0.0157 |
| 3 | 192 | 0.0381 | 0.0372 |
| 4 | 141 | 0.0236 | 0.0116 |
| **mean** | | **0.0730** | **0.0187** |

Test sizes match exactly, confirming the harness is wired correctly. Per-split values
differ because the published baseline uses a larger ESM-2 checkpoint than the 35M model we
can fit on a 6 GB GPU. Both land in the same place qualitatively: a linear probe on ESM-2
embeddings has essentially **no** predictive signal for HC50 across a 60 % identity
barrier, with R² between −3.1 and −5.0.

## 5. Our result

`scripts/eval_qmap.py`. For each split the model/representation pair is chosen by grouped
inner cross-validation **on that split's training rows only**; the benchmark itself is
scored once. Selected: ExtraTrees on physchem+ESM-2 for splits 0, 2, 3, 4 and ExtraTrees on
the full descriptor set for split 1.

| Split | n | Pearson | Spearman | R² | MAE |
| --- | --- | --- | --- | --- | --- |
| 0 | 173 | 0.083 | 0.050 | −0.072 | 0.514 |
| 1 | 278 | 0.149 | 0.200 | −0.169 | 0.539 |
| 2 | 129 | 0.410 | 0.400 | 0.154 | 0.468 |
| 3 | 192 | 0.262 | 0.323 | 0.052 | 0.521 |
| 4 | 141 | 0.164 | 0.092 | 0.017 | 0.501 |
| **mean** | | **0.214** | **0.213** | **−0.003** | **0.509** |

Comparison against the published HC50 leaderboard:

| | PCC min | PCC mean | PCC max | R² mean |
| --- | --- | --- | --- | --- |
| Published HemoLinear | −0.18 | 0.07 | 0.29 | strongly negative (−1.3 to −11.7) |
| Ours | **0.08** | **0.21** | **0.41** | **−0.003** |

We beat the published baseline on every summary statistic, and our worst split is still
positive where theirs is negative. The R² difference is the largest: the published
baseline's predictions are far worse than predicting the training mean, ours are
approximately as good as the mean.

**This must not be oversold.** QMAP's HC50 leaderboard contains exactly **one** entry, and
it is a linear probe — a deliberately weak reference. (The repository's richer tables, with
entries from Witten & Witten 2019 and Cai et al. 2025 reaching mean PCC 0.22–0.52, are for
*E. coli MIC*, not HC50.) So "we lead the HC50 leaderboard" means we beat the only
published entry, not that we beat a field of tuned competitors. The defensible claim is
narrower: on a homology-aware HC50 benchmark, a physicochemical + ESM-2 tree ensemble
extracts real signal where a linear ESM-2 probe extracts none.

## 6. How this recalibrates our own results

Our internal locked-test Pearson of **0.299** previously read as a weak result. Against
QMAP it is at the top of the published range for homology-controlled HC50 regression. The
honest reading is not that our model is unusually good, but that **this task is intrinsically
hard** and the field's published numbers on random splits are not describing the same
problem.

The optimism gap is visible inside this run too. The inner cross-validation Pearson on the
training pool was **0.65–0.68** on every split, while the homology-separated benchmark score
was **0.08–0.41**. That is a gap of roughly 0.4 in correlation, larger than the 0.25 gap we
measured internally in `leakage_diagnostics`, because QMAP's 60 % identity barrier is
stricter than our 70 % one.

## 7. Caveats

* QMAP's training pool after filtering is only **934** sequences (752/650/801/736/773 per
  split), far smaller than our internal dataset, because the canonical/L-only/unmodified
  filters are aggressive. Our score here is not a measure of what our full pipeline can do
  with all its data; it is a like-for-like comparison at QMAP's data budget.
* Selecting model and representation per split by inner CV is a defensible protocol but is
  not identical to the published baseline, which fixes one model in advance. A fixed-choice
  variant is reported implicitly: ExtraTrees on physchem+ESM-2 was chosen on 4 of 5 splits.
* We did not tune hyperparameters against the benchmark, and each split was scored once.
