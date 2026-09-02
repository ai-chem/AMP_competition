# AMP Challenge 2027

Reproducible pipeline for de novo antimicrobial peptide design
([AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027)).

Python is pinned to **3.11**. Dependencies are locked with [`uv`](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

On an A100 server, confirm the driver first:

```bash
nvidia-smi
```

Linux installs CUDA 12.6 PyTorch wheels by default. If the driver is newer or older, change the index in `pyproject.toml`:

- CUDA 12.4 → `https://download.pytorch.org/whl/cu124`
- CUDA 12.8 → `https://download.pytorch.org/whl/cu128`

Then run `uv lock` and `uv sync` again.

## Generate

```bash
uv run generate
```

Writes `generate/library.fasta` (50,000 sequences) and `generate/top.fasta` (top 100).
All extra CLI flags have defaults:

| Flag | Default | Description |
|---|---|---|
| `--n-sequences` | `50000` | Library size |
| `--top-k` | `100` | Ranked shortlist size |
| `--min-length` | `8` | Minimum peptide length |
| `--max-length` | `50` | Maximum peptide length |
| `--seed` | `42` | Random seed |

The current command uses a **placeholder generator**. ProtGPT3-1.3B will replace it in tasks 1.3–1.4.

## Layout

```
├── configs/                 # seed, model, generation, LoRA
├── src/amp_competition/
│   ├── generate.py          # `uv run generate`
│   ├── generator/           # ProtGPT3 + LoRA
│   ├── data/                # corpus prep
│   ├── features/            # physicochemical conditions
│   ├── predictors/          # MIC / hemolysis
│   └── filters/             # validity, novelty, diversity
├── data/                    # raw / processed / external corpora
├── checkpoint/              # model weights and LoRA adapters
├── scripts/                 # submission verifier
└── notebooks/
```

## License

MIT
