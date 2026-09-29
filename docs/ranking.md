# Ranking

`uv run generate` builds one scored list and then cuts it.

1. Sample 50,000 accepted peptides at each row of
   `configs/selected_generation_conditions.csv` (five points, 250,000 peptides).
   A peptide is accepted only when all of the following hold:
   - length 8–50;
   - alphabet `ACDEFGHIKLMNPQRSTVWY`;
   - ProtGPT3 direction token N→C;
   - not already accepted in this run;
   - not an exact string match to `data/external/antibacterial.fasta`.
2. Score every accepted peptide.
   - MIC: `models/lgbm_cv_ranker.txt`, ESM-2 8M mean embedding
     (`facebook/esm2_t6_8M_UR50D`, revision
     `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`) reduced by the committed PCA,
     plus physicochemical descriptors. Higher means a better predicted potency
     rank.
   - HC50: ESM-2 35M fine-tune in `peptide-safety-inference/models/esm2_35M_ft.pt`,
     reported in µM. Higher means less hemolytic.
3. Combined score = `rank(MIC) / N + rank(log HC50) / N`, with average ranks
   and `N` the size of the scored pool. Ties in a single score share the
   average rank.
4. Sort by combined score descending, then by sequence ascending.
5. Drop any peptide with `Levenshtein.ratio` strictly above 0.80 against any
   sequence in `data/external/antibacterial.fasta`. This is the function from
   the `python-Levenshtein` package, the same one as
   `scripts/verify_submission.py`.
6. Write `generate/library.fasta` as the first 50,000 remaining peptides and
   `generate/top.fasta` as the first 100 of that file. Headers are `amp1`…
   and `top1`… in that same order. The top-100 is a prefix of the library.

No peptide is inserted by hand after this sort. The batch size is fixed for
the whole run so a retry cannot change the sample stream.
