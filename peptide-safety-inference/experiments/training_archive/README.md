# Peptide Safety Predictor

Censoring-aware prediction of peptide **hemolytic activity (HC50)**, with
explicitly-labelled solubility and stability proxies, an applicability domain,
and calibrated uncertainty.

The headline number this project optimises is generalization to **dissimilar**
peptides. All model selection is done under cluster-disjoint cross-validation at
70% sequence identity; the locked test set was created before any model was
chosen and scored exactly once.

## Quick start

Predict on a CSV containing a `sequence` column:

```bash
python predict.py --input examples/controllability_sequences.csv --output predictions.csv --sequence-col sequence --device auto
```

`--device auto` uses CUDA when available and falls back to CPU. Every column of
the input file is preserved, and row order is unchanged.

## Installation

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# Windows / CUDA 12.4 build of torch (see reports/environment.json for the exact pin):
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
```

Exact versions actually used for the published results are recorded in
[`reports/environment.json`](reports/environment.json).

## What the output columns mean

| Column | Meaning |
| --- | --- |
| `hc50_log_uM_pred` | Predicted HC50 in natural-log µM (the space the model is trained in) |
| `hc50_uM_pred` | Same prediction exponentiated back to µM |
| `hc50_log_lower_95`, `hc50_log_upper_95` | 95% prediction interval in log space |
| `hc50_lower_95_uM`, `hc50_upper_95_uM` | The same interval in µM |
| `hc50_uncertainty` | Predictive standard deviation in log space |
| `p_hc50_gt_128` | Calibrated probability that HC50 exceeds 128 µM |
| `hemolysis_risk_prob` | `1 - p_hc50_gt_128` |
| `max_train_identity` | Highest sequence identity to any training peptide |
| `embedding_ood_score`, `physchem_ood_score` | Distance to the training distribution in ESM-2 and descriptor space |
| `in_domain` | Whether the peptide falls inside the applicability domain |
| `prediction_confidence` | Combined confidence, low when the peptide is out of domain |
| `solubility_proxy_score` | **Proxy only** — a CamSol-like physicochemical score, not calibrated aqueous solubility |
| `stability_proxy_score` | **Proxy only** — a cleavage/oxidation-site score, not a measured half-life |

Read `reports/model_card.md` before acting on any of these numbers; the
solubility and stability columns in particular are unvalidated proxies because
no labelled dataset for those endpoints was obtainable (see
`reports/solubility_stability_audit.json`).

## Reproducing the pipeline

Each stage writes its outputs to disk, so an interrupted run can be resumed by
re-running from the stage that failed.

```bash
python scripts/audit_external.py            # audit external repos/checkpoints
python scripts/download_data.py             # crawl DBAASP, clone pinned repos
python scripts/prepare_data.py              # build the censored observation table
python scripts/make_splits.py               # cluster at 70% id, build locked test
python scripts/train_baselines.py           # descriptor / composition / PLM matrix
python scripts/train_censored.py            # Tobit, AFT, xgboost survival:aft
python scripts/train_deep.py                # CNN, multiscale CNN, BiLSTM, ESM-2 fine-tune
python scripts/leakage_diagnostics.py       # random vs cluster-disjoint gap
python scripts/train_solubility_stability.py
python scripts/external_baselines.py        # score external predictors + contamination audit
python scripts/evaluate.py                  # freeze config, score locked test ONCE
```

`make_splits.py` aborts with a non-zero exit code if any locked-test sequence
reaches 70% identity to a training sequence, so a leak cannot pass silently.

## Key reports

- `reports/model_comparison.md` — full comparison, protocol, and leakage evidence
- `reports/model_card.md` — intended use, endpoints, and limitations
- `reports/data_audit.md` — dataset provenance and censoring statistics
- `reports/split_audit.csv` — nearest training neighbour for every test sequence
- `reports/leakage_diagnostics.json` — how much a random split inflates results
- `reports/external_audit.md` — availability and reproducibility of external predictors
- `reports/final_test_metrics.json` — the single locked-test evaluation
