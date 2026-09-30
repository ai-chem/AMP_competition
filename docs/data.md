# Training data, external databases, and filters

## Reference used at generation time

`data/external/antibacterial.fasta` is the organizer file from
[amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027) at
commit `5c8a5d8e2551c8cf572d3d3bfcfe7633b109d91e`. The SHA-256 and record count
are in `data/external/antibacterial.manifest.json`. Generation treats this file
as an immutable exclusion list. It is not a training label.

`data/raw/antibacterial_clean.csv` is that FASTA after `scripts/audit_fasta.py`:
unique sequences, length 8–50, alphabet of 20 amino acids. The tracked table
has 39,448 data rows, the same count as the FASTA. Columns include
`source_databases` and `activity_tags` copied from the organizer headers.
LoRA V1 and the conditional V2 adapter were trained on this table
(`configs/train_lora.yaml`, `configs/train_cond.yaml`), with a length-stratified
10% validation split at seed 42 (`src/amp_competition/data/peptides.py`).

The `source_databases` values name public AMP resources already attached to
the organizer records, including DRAMP, DBAASP, dbAMP, SATPdb, CAMP, APD,
AMPDB, and DADP. No separate private peptide collection is added for the
generator.

## MIC ranker

Inference loads the committed files in `models/`: `lgbm_cv_ranker.txt`,
`pca_cv.pkl`, `robust_scaler_cv.pkl`, and `physchem_cols.pkl`.

The training table is built from GRAMPA, the public file at
<https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/master/data/grampa.csv>
(Witten and Witten, Antimicrobial Peptides Database / GRAMPA). The raw file
tracked here is `data/raw/GRAMPA/grampa.csv`. Rebuild the ranker table with:

```bash
uv run python src/amp_competition/data/build_MIC_dataset.py
```

That writes `data/processed/enriched_ranked_dataset.csv`. Sequences must match
`ACDEFGHIKLMNPQRSTVWY` and have length 8–100. MIC values are averaged within
each sequence–bacterium pair, strains with fewer than 5 pairs are dropped, and
sequences are clustered with a greedy Levenshtein similarity of 0.75 so the
ranker groups do not leak near-duplicates. ESM-2 embeddings for training are
produced by `src/amp_competition/features/extract_esm.py`. The ranker itself is
`src/amp_competition/predictors/train_ranker.py`.

## HC50 model

`peptide-safety-inference/models/esm2_35M_ft.pt` and `hc50_bundle.pkl` are the
inference weights. The observation table used to fit them is
`peptide-safety-inference/experiments/training_archive/data/processed/hc50_observations.csv`
(14,504 data rows). Observations come from DBAASP hemolysis records
(endpoint around 50% hemolysis, normalized to µM, with censoring flags).
DBAASP is public for research use; this file is the project’s normalized extract of those records so the training set can be inspected. DBAASP’s own terms still apply to the underlying database.
The model card and comparison notes live under
`peptide-safety-inference/`.

Solubility and protease-stability weights in the same directory are not used
by `uv run generate`.

## Filters applied to the submitted library

Applied while sampling, before scoring:

- length 8–50;
- 20 standard amino acids;
- N→C only;
- exact duplicate removal inside the run;
- exact match removal against `data/external/antibacterial.fasta`.

Applied after scoring, before writing `library.fasta` and `top.fasta`:

- `Levenshtein.ratio` > 0.80 against the same reference drops the peptide.

No terminal modifications, noncanonical residues, or hand-picked replacements
are introduced. Generation hyperparameters are in `configs/generate.yaml`
(seed 42, temperature 0.8, top-p 0.9, batch size 32).

## Physicochemical condition selection

The two V2 conditioning variables were selected by comparing the organizer AMP dataset with putative non-AMP sequences retrieved from UniProtKB. Background sequences were restricted to the canonical 20-amino-acid alphabet and lengths of 8–50 residues, and sequences carrying antimicrobial or related annotations were excluded. Exact duplicates and exact matches to the organizer AMP set were removed.

To control for the strong difference in sequence length between the two populations, the primary condition-selection analysis used exact length-matched cohorts of 35,627 AMP and 35,627 putative non-AMP sequences. The analysis selected net charge at pH 7.4 and Wimley–White interfacial hydrophobicity at pH 8 as the V2 conditioning variables. Their joint AMP enrichment was estimated using two-dimensional Gaussian KDE, and five representative generation conditions were selected from strongly AMP-enriched regions supported by observed AMP sequences.

The UniProt-derived background is used only for physicochemical analysis and condition selection.
