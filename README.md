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
| `configs/train_lora.yaml` | Frozen-base LoRA check |
| `configs/check_protgpt3.yaml` | Tokenizer / generation smoke |
| `configs/bench_generate.yaml` | Speed / VRAM / reproducibility |

`extends: default.yaml` merges the parent. CLI `--config` and `--seed` override YAML.
Each run writes `run.json` + `config.resolved.yaml` next to outputs and under `runs/<UTC>_<command>/`.

## Generate

```bash
PYTHONPATH=src python3 -m amp_competition.generate --config generate.yaml
# or: uv run generate --config generate.yaml
```

Writes `generate/library.fasta` (100,000 unique peptides, length 8–50) and `generate/top.fasta` (placeholder ranking until predictors are wired).

| Flag | Default | Description |
|---|---|---|
| `--config` | `generate.yaml` | YAML in `configs/` |
| `--n-sequences` | `100000` | Library size |
| `--top-k` | `100` | Ranked shortlist |
| `--min-length` | `8` | Minimum peptide length |
| `--max-length` | `50` | Maximum peptide length |
| `--seed` | `42` | RNG seed |

FASTAs are gitignored. The generator is ProtGPT3-1.3B (`AI4PD/ProtGPT3-1.3B`), not the placeholder.

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
│   ├── data/                     # corpus prep (not wired yet)
│   ├── features/
│   ├── predictors/               # MIC / hemolysis (not wired yet)
│   └── filters/
├── data/                         # raw / processed / external
├── generate/                     # library.fasta (gitignored)
├── checkpoint/                   # LoRA adapters (gitignored)
├── scripts/                      # checks, bench, submission verifier
├── runs/                         # run archives (gitignored)
└── outputs/                      # bench / slurm logs (gitignored)
```

## License

MIT
