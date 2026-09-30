# AMP Challenge 2027

Reproducible pipeline for de novo antimicrobial peptide design
([AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027)).

Python is pinned to **3.11**. Seed is **42** (`configs/default.yaml`).
Dependencies are locked with [`uv`](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

On an A100 / A6000 node:

```bash
nvidia-smi
```

Linux installs CUDA 12.6 PyTorch wheels by default. If the driver is newer or older, change the index in `pyproject.toml` and run `uv lock && uv sync`.

## Configs

| File | Role |
|---|---|
| `configs/default.yaml` | Base: seed, model, generation, LoRA, train |
| `configs/generate.yaml` | 100k library sampling |
| `configs/train_lora.yaml` | LoRA on antibacterial peptides; best adapter in `checkpoint/lora_antibacterial/best` |
| `configs/check_protgpt3.yaml` | Tokenizer / generation smoke |
| `configs/bench_generate.yaml` | Speed / VRAM / reproducibility |
| `configs/physchem_analysis.yaml` | AMP vs putative non-AMP physicochemical comparison |

`extends: default.yaml` merges the parent. CLI `--config` and `--seed` override YAML.
Each run writes `run.json` + `config.resolved.yaml` next to outputs and under `runs/<UTC>_<command>/`.

## Generate

`uv run generate` is the submission entry point. It needs a CUDA GPU and, on a
machine without the Hugging Face cache, a network connection to download the
pinned base models. With the defaults it writes:

```bash
uv run generate
```

| File | Contents |
|---|---|
| `generate/library.fasta` | 50,000 unique peptides, the highest-scoring survivors |
| `generate/top.fasta` | first 100 of that library, same order |
| `generate/passed_scores.csv` | every peptide that passed the reference screen, with MIC, HC50, and the combined score |

Sampling draws 50,000 accepted peptides at each of the five V2 condition points
(250,000 before the reference screen). Acceptance during sampling is length,
the 20-letter alphabet, N→C direction, within-run duplicates, and exact copies
of `data/external/antibacterial.fasta`. MIC and HC50 are scored on that pool.
The combined score is the sum of the average rank of MIC and the average rank
of `log(HC50)`. Peptides with `Levenshtein.ratio` above 0.80 against the
organizer reference are removed, then the library and the top-100 are the head
of what remains.

| Flag | Default | Description |
|---|---|---|
| `--config` | `generate.yaml` | YAML in `configs/` |
| `--n-sequences` | `50000` | Alias of `--library-size` |
| `--library-size` | `50000` | Peptides written to `library.fasta` |
| `--n-per-cluster` | `50000` | Accepted peptides sampled at each condition point |
| `--top-k` | `100` | Ranked shortlist |
| `--min-length` | `8` | Minimum peptide length |
| `--max-length` | `50` | Maximum peptide length |
| `--seed` | `42` | RNG seed |
| `--batch-size` | `32` | Fixed sampling batch. A mid-run reduction would change the sample stream, so an out-of-memory error stops the run |
| `--checkpoint` | `checkpoint/lora_cond_v2/best` | V2 adapter |
| `--conditions` | `configs/selected_generation_conditions.csv` | Target (Q, H) points |

The seed is fixed. Token sampling uses a CPU `torch.Generator`, TF32 and flash
attention are off, and deterministic CUDA algorithms are required. Run the
command twice on the same GPU: `library.fasta` and `top.fasta` should be
byte-identical. Details of the rank and the training tables are in
[docs/ranking.md](docs/ranking.md) and [docs/data.md](docs/data.md). A short
method summary is in [docs/abstract.md](docs/abstract.md).

FASTAs are gitignored. The generator is conditional ProtGPT3-1.3B V2, not the base model.

## LoRA

```bash
PYTHONPATH=src python3 scripts/train_lora.py --config train_lora.yaml
PYTHONPATH=src python3 scripts/eval_lora_generate.py
```

Best adapter: `checkpoint/lora_antibacterial/best/` (see that folder’s README for specs, val curve, and EOS / copy-check). The training table is `data/raw/antibacterial_clean.csv`.

## Physicochemical analysis

```bash
uv sync --extra analysis
PYTHONPATH=src python3 scripts/run_physchem_analysis.py --config physchem_analysis.yaml
```

Compares organizer AMP sequences to putative non-AMP UniProtKB negatives (unmatched and length-matched cohorts). Descriptor code lives in `src/amp_competition/features/` for reuse in conditional generation. Outputs go to `outputs/physchem/` (gitignored).

## Scripts

See **[scripts/README.md](scripts/README.md)**.

## Layout

```
├── configs/                      # YAML run configs (seed 42)
├── src/amp_competition/
│   ├── generate.py               # `uv run generate`
│   ├── config.py                 # load YAML, seed_everything, run.json
│   ├── generator/
│   │   ├── protgpt3.py           # load / tokenize / sample
│   │   ├── sample.py             # mass library 8–50 AA
│   │   └── lora.py               # LoRA on frozen base
│   ├── data/                     # peptide CSV loader / split
│   ├── features/                 # physchem descriptors, cohorts, comparison
│   ├── predictors/               # MIC ranker used by `uv run generate`
│   └── filters/
├── data/                         # raw / processed / external
├── generate/                     # library.fasta (gitignored)
├── legacy3/                      # byte-matched 50k library and top-100
├── checkpoint/                   # LoRA adapters (`lora_antibacterial/best` is tracked)
├── scripts/                      # checks, bench, submission verifier
├── runs/                         # run archives (gitignored)
└── outputs/                      # bench / slurm logs (gitignored)
```

## License

MIT
