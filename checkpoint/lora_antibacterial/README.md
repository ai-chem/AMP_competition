# ProtGPT3-1.3B LoRA (antibacterial)

Frozen-base LoRA on `data/raw/antibacterial_clean.csv`. Best adapter is `best/` (val loss **2.131**, epoch 5). `last/` is not committed (same weights).

## Training

| | |
|---|---|
| Base | `AI4PD/ProtGPT3-1.3B`, bfloat16 |
| Format | `<\|bos\|> 1 SEQ <\|eos\|>` |
| Data | 39 448 unique peptides, length 8–50, 20 AA |
| Split | 35 498 train / 3 950 val, seed 42, stratified by length |
| LoRA | r=16, α=32, dropout 0.05, `q/k/v/o_proj` |
| Trainable | 1 810 432 / 1 330 070 528 (**0.136%**) |
| Optim | AdamW lr `1e-4`, cosine + 5% warmup, wd 0.01, grad clip 1.0 |
| Batch / epochs | 64 / 5 (2775 steps), patience 2 |
| Hardware | 1× RTX A6000, 679 s |

Config: `configs/train_lora.yaml`. Run: `PYTHONPATH=src python3 scripts/train_lora.py`.

Val history: 2.36 → 2.22 → 2.16 → 2.14 → **2.13**.

## Generation check (no length forcing)

`max_new_tokens=50`, prompt `1`, T=0.8, top-p=0.9. Base 256 vs LoRA 512.

| | Base | LoRA |
|---|---|---|
| EOS | 0% | **97.7%** |
| Length | all 50 | 8–50, mean **17.82** (train 18.72) |
| Exact train copies | 0 | **10%** (51/512, 27 unique) |

Details: `eval_generate.json`, `metrics.json`. Re-run: `PYTHONPATH=src python3 scripts/eval_lora_generate.py`.
