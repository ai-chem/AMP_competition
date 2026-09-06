# Scripts

All command-line checks live here. Library code is under `src/amp_competition/`.

Run from the repo root with `PYTHONPATH=src` (or `uv run python …` after `uv sync`). GPU jobs need a compute node (`sbatch` / `srun`).

| Script | Config | What it does |
|---|---|---|
| `check_protgpt3.py` | `configs/check_protgpt3.yaml` | Tokenizer + a few ProtGPT3 samples |
| `check_lora.py` | `configs/train_lora.yaml` | LoRA trains, base stays frozen |
| `bench_generate.py` | `configs/bench_generate.yaml` | Speed, VRAM, same-seed reproducibility |
| `verify_submission.py` | — | Official AMP Challenge 2027 verifier |

```bash
PYTHONPATH=src python3 scripts/check_protgpt3.py --tokenizer-only
PYTHONPATH=src python3 scripts/check_lora.py
PYTHONPATH=src python3 scripts/bench_generate.py
uv run python scripts/verify_submission.py <github-url>
```

Verifier source: https://github.com/szczurek-lab/amp-challenge-2027/blob/main/scripts/verify_submission.py
