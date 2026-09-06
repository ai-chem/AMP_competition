# Scripts

All command-line checks live here. Library code is under `src/amp_competition/`.

Run from the repo root with `PYTHONPATH=src` (or `uv run python …` after `uv sync`). GPU jobs need a compute node (`sbatch` / `srun`).

| Script | Config | What it does |
|---|---|---|
| `check_protgpt3.py` | `configs/check_protgpt3.yaml` | Tokenizer + a few ProtGPT3 samples |
| `check_lora.py` | `configs/train_lora.yaml` | LoRA trains, base stays frozen |
| `bench_generate.py` | `configs/bench_generate.yaml` | Speed, VRAM, same-seed reproducibility |
| `verify_submission.py` | — | Official AMP Challenge 2027 verifier |
| `audit_fasta.py` | — | Audit a FASTA and write cleaned/rejected CSV files plus a JSON report |

```bash
PYTHONPATH=src python3 scripts/check_protgpt3.py --tokenizer-only
PYTHONPATH=src python3 scripts/check_lora.py
PYTHONPATH=src python3 scripts/bench_generate.py
uv run python scripts/verify_submission.py <github-url>
uv run python scripts/audit_fasta.py
```

Audit the organizer reference after it has been fetched:

```bash
uv run python scripts/audit_fasta.py
```

The default outputs are `data/processed/antibacterial_clean.csv`,
`data/processed/antibacterial_rejected.csv`, and
`data/processed/antibacterial_audit.json`. The audit rejects empty sequences,
sequences outside 8–50 residues, non-canonical amino acids, and duplicate
occurrences. It also reports reproducibly sampled pairwise Levenshtein
similarity; use `--sample-pairs 0` to skip that calculation.

The cleaned CSV columns are `record_id`, `charge`, `disulfide`,
`source_databases`, `activity_tags`, `header`, `sequence`, and
`sequence_length`. The two tag columns are JSON arrays in CSV cells so that
their individual values remain machine-readable.

Verifier source: https://github.com/szczurek-lab/amp-challenge-2027/blob/main/scripts/verify_submission.py
