# ProtGPT3-1.3B conditional LoRA (V2)

Soft-prompt on charge (`charge_pH7_4`) and interface hydrophobicity (`hydrophobicity_interfaceScale_pH8`). Starts from AMP-LoRA V1. Best adapter is `best/` (val loss **1.976**, epoch 3). `last/` is not committed.

Full write-up: [`controllability.md`](controllability.md).

## Training

| | |
|---|---|
| Base | V1 adapter on `AI4PD/ProtGPT3-1.3B` |
| Format | `<\|bos\|> 1 [cond×4] SEQ <\|eos\|>` |
| Conditions | z-scored train Q/H → MLP 2→128→4×1024 |
| Split | same as V1: 35 498 / 3 950 |
| Trainable | LoRA 1 810 432 + MLP 528 768 |
| Optim | AdamW lr `1e-4`, cosine + 5% warmup, 3 epochs, batch 64 |
| Hardware | 1× RTX A6000, 422 s |

Config: `configs/train_cond.yaml`. Run: `PYTHONPATH=src python3 scripts/train_cond_lora.py --config train_cond.yaml`.

Val history: 2.053 → 1.991 → **1.976**.

## Controllability

64 samples × 5 train-percentile targets, descriptors recomputed with `descriptors.py`.

- Pearson(Q*, Q) **0.86**
- Pearson(H*, H) **0.89**
- high-Q > low-Q and high-H > low-H

Generate: `PYTHONPATH=src python3 scripts/generate_cond.py --charge 4.0 --hydrophobicity -0.2 --n 32`.
