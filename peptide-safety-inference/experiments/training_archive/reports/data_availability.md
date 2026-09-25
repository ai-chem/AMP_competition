# Data and checkpoint availability — verified sweep

Every source named in the project brief and in the user's follow-up message was re-checked
by actually fetching it, not by assuming. Each row below says what was *done*, not what was
*expected*. HTTP status codes were observed on the dates of this run.

**This report supersedes the earlier "not obtainable" statements in `model_card.md` and
`model_comparison.md`.** Two of those statements were wrong: PEPlife and PeptideBERT data
are both obtainable and are now in hand. Those files will be corrected.

## A. Obtained — in hand and usable

| Source | What we got | Size | Where |
| --- | --- | --- | --- |
| DBAASP REST API | full crawl, 0 errors | 25,544 records | `data/raw/dbaasp/` |
| DBAASP CSV exports (user-supplied) | hemolytic + cytotoxic activities, 5 files | 8,886 rows / 8,882 unique | `data/raw/dbaasp/hemolytic-and-cytotoxic-*.csv` |
| Hemolytik 2.0 (user-supplied) | full table, 1,645 PMIDs | 13,215 rows / 6,849 unique seqs | `data/raw/Hemolytik2/` |
| **PEPlife 2 REST API** | half-life records via 4 query axes | 4,500 records / 1,852 unique seqs | `data/raw/peplife2/peplife2_all.json` |
| **PeptideBERT — hemolysis** | decoded from `peptide-dashboard` arrays | 6,541 rows / 6,076 unique | `data/raw/peptidebert/peptidebert_hemolysis.csv` |
| **PeptideBERT — solubility** | soluble vs insoluble E. coli expression | 18,453 rows, 0 label conflicts | `data/raw/peptidebert/peptidebert_solubility.csv` |
| **PeptideBERT — nonfouling** | decoded | 17,182 rows | `data/raw/peptidebert/peptidebert_nonfouling.csv` |
| ConsAMPHemo — regression set | `Sequence, Hemo_avg_con, LN_value` + 555 descriptors | 1,355 seqs, len 10–49 | `external/_clones/ConsAMPHemo/Dataset/regression/` |
| ConsAMPHemo — S1/S2/S3 | classification sets with predefined splits | 1,104 / 2,557 / 7,179 | `external/_clones/ConsAMPHemo/Dataset/S*/` |
| **ConsAMPHemo — trained models** | 3 × PyTorch + 1 × XGBoost | 43 MB each, 660 KB | `.../S*/model/*.pl`, `.../regression/model/XGB_model_Hemo.joblib` |
| HemoNet — datasets | hemo/non-hemo, 70 % and 90 % non-redundant variants, external set | 2,109 + 3,158 seqs (all), 544/716 at 70 % | `external/_clones/HemoNet/` |
| HemoPI2 — datasets | **quantitative µM column**, cross-val + independent | 1,540 + 386 | `external/_clones/HemoPI2/Dataset/` |
| LysePred — datasets | 8 FASTA sets with `id\|label\|split` headers | HemoPI1 1104, HemoPI2 1014, HemoPI3 1623, hlppredfuse 3518, rathore 1540, rathore2025_external 386, rnnamp 2312/2557 | `external/_clones/LysePred/data/` |
| ML-guided non-hemolytic peptides | `complete_df_with_activity.csv` + APD/HemoPI FASTA | 3,081 × 39 | `external/_clones/ML-guided-.../Data/` |
| **ML_Peptide — GI stability (FigShare)** | `peptide_stability.csv`, measured % drug remaining at 30 min / 2 h | 109 rows | `data/raw/ml_peptide/peptide_stability.csv` |
| QMAP benchmark | pip `qmap-benchmark==0.1.1` + HF `anthol42/qmap_benchmark_2025` | 18,033 samples, 2,797 with HC50 | installed |
| ESM-2 checkpoints | `esm2_t12_35M_UR50D` | — | HuggingFace cache |

## B. Reachable, but needs a deliberate extra step

| Item | Status | What is needed |
| --- | --- | --- |
| LysePred `pretrain/` | Directories exist but are **empty** — the repo ships no BERT weights | LysePred is built on ProtBert. `Rostlab/prot_bert` is live on HuggingFace (verified HTTP 200), so LysePred is runnable after fetching it. No author checkpoint is needed to retrain. |
| DBAASP hemolytic predictor (*Brief. Bioinform.* 23, bbac233) | Web tool only — `dbaasp.org/prediction` returns HTTP 200 but serves no model artifact | There is no downloadable model. Its **training definition** is reusable though, and we adopt it: Active = hemolysis > 40 % at < 40 µg/ml; Non-Active = explicitly marked Non-Active in DBAASP; length 4–30 aa. We can reconstruct an equivalent label from our own DBAASP crawl. |
| HemoPI2 / Hemolytik2 web servers | Both live (HTTP 200) | Model weights are not published; only code and datasets. Retraining from the shipped datasets is the path. |

## C. Genuinely gone — confirmed dead, not assumed

| Item | Link | Evidence |
| --- | --- | --- |
| **HemoNet trained weights** | `HemoNet/weights.hdf` is an 84-byte text file containing `https://drive.google.com/file/d/1k1RbSnjTjS7u0dFibxUltwGoxRMH2q_T/view` | Google Drive returns **HTTP 404** with "file does not exist" on both the `/uc?id=` and `/file/d/.../view` endpoints, and via `gdown`. This is a deleted file, not a permissions problem. |

HemoNet's **data** is fully present, so the model is reproducible by retraining; only the
authors' exact checkpoint is unrecoverable. Per the brief, this is recorded rather than
worked around, and no HemoNet-derived numbers will be reported as if the original
checkpoint had been used.

## D. Nothing is currently blocking us

The user offered to fetch specific files manually. As of this sweep **there is no file we
need that we cannot get.** The single dead artifact (HemoNet's `weights.hdf`) is not
load-bearing: we have its entire training corpus and can retrain, and a third-party copy of
someone else's checkpoint would not be verifiable anyway.

If that changes, the blocked item would be listed here with its exact filename and the URL
that failed.

## E. Data-quality warnings found during acquisition

These matter more than the availability question, because they affect whether a dataset can
be used honestly.

1. **ConsAMPHemo's regression target has collapsed censoring.** `Hemo_avg_con` piles up on
   round numbers — 200 µM (91×), 100 (54×), 500 (53×), 128 (36×), 400 (31×), 300 (25×) —
   and 50 % of all rows sit at ≥ 100. This is the signature of right-censored records
   (">200 µM") being rewritten as the threshold value and then fitted with ordinary
   regression. The project brief forbids exactly this. We may reuse their **sequences**,
   but their target is not adoptable as-is; censoring must be re-derived from the primary
   sources.
2. **PeptideBERT's hemolysis labels contain contradictions.** 930 rows (465 sequences)
   appear with both labels. The solubility set, by contrast, is clean: 18,453 sequences,
   zero label conflicts.
3. **PeptideBERT solubility is not aqueous solubility.** The endpoint is soluble
   *heterologous expression in E. coli*. It will be documented as such and never presented
   as experimentally calibrated aqueous solubility.
4. **Hemolytik 2.0 activity is free text** — `"50 % Hemolysis at >200 µM"`, `"LC50 >200 µM"`,
   `"MHC >1 µg/ml"`. 1,507 rows contain `>` and 1,104 contain `<`, so roughly 20 % of the
   table is censored and must be parsed into the censored schema rather than dropped or
   point-valued.
5. **Erythrocyte species varies** across Hemolytik 2.0: human 9,255, horse 1,062, sheep 840,
   rabbit 485, rat 426. Mixing species silently would confound the target; species becomes
   a recorded covariate.
6. **QMAP discards censoring by design** — it collapses each peptide to one consensus float.
   That is fine for external comparison but makes it unusable as our internal training
   target.

## F. Method reuse (no download required)

* **QMAP's evaluation protocol** is adopted; see `qmap_protocol.md` for what we take and
  what we decline.
* **The DBAASP predictor's label definition** is adopted as a labelling rule applied to our
  own crawl, which is the reusable part of a web-only tool.
