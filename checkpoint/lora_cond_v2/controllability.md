# V2: условная генерация пептидов

Цель — генератор, которому можно задать заряд и гидрофобность, а не только «сэмплируй AMP».

## Что сделано

Стартовали с **AMP-LoRA V1** (`checkpoint/lora_antibacterial/best`, ProtGPT3-1.3B, val loss 2.131). Базовые веса ProtGPT3 заморожены. Доучивали LoRA (`r=16`, α=32, q/k/v/o) и небольшой MLP условия.

Для каждого пептида из `data/raw/antibacterial_clean.csv` посчитали два дескриптора теми же функциями, что в `src/amp_competition/features/descriptors.py`:

- `charge_pH7_4` — заряд при pH 7.4 (шкала Lehninger);
- `hydrophobicity_interfaceScale_pH8` — интерфейсная гидрофобность Wimley–White (более отрицательное = сильнее тянется в интерфейс).

Сплит тот же, что у V1: **35 498 train / 3 950 val**. По train посчитали z-score (среднее и std) и сохранили в `normalizer.json`.

Условие `c = [Q, H]` после нормализации идёт в MLP (2 → 128 GELU → 4×1024) и становится **четырьмя soft-prompt токенами**. Они вставляются **после** `<|bos|> 1` и **перед аминокислотами**:

```
[<|bos|>, 1, cond, cond, cond, cond, SEQ, <|eos|>]
```

В ТЗ было `[emb(cond), emb(seq)]` прямо перед всей последовательностью, включая BOS. Так Mixtral/PEFT давал NaN, поэтому BOS и направление оставили первыми, а condition — сразу перед SEQ. Эмбеддинги condition подменяются через hook по `input_ids`, а не подаются голым `inputs_embeds`.

Loss — обычный causal LM. На слотах condition метка `-100`. Последний слой MLP инициализирован N(0, 0.02), не нулями.

## Обучение

GPU RTX A6000, 3 эпохи, batch 64, lr 1e-4, cosine + warmup 5%, weight decay 0.01. Учится ~1.81M LoRA + ~0.53M MLP.

| эпоха | train loss | val loss |
|------:|----------:|---------:|
| 1 | 2.138 | 2.053 |
| 2 | 1.971 | 1.991 |
| 3 | 1.922 | **1.976** |

Лучший адаптер: `checkpoint/lora_cond_v2/best/` (7 минут, 1665 шагов).

Смоук на одном батче до полного прогона: `smoke_loss=3.21` (конечный). Если лосс становится NaN, прогон считается неуспешным и `best/` не пишется.

## Controllability

Генерировали по 64 пептида (длина 8–50, алфавит челленджа) на пять целей: 10/50/90 перцентили train по Q и H. Для каждой валидной последовательности заново считали дескрипторы через `descriptors.py`.

- Pearson(цель Q, факт Q): **0.86**
- Pearson(цель H, факт H): **0.89**
- средний заряд high-Q > low-Q: **да**
- средняя гидрофобность high-H > low-H: **да**

| цель | Q* | Q факт | ΔQ | H* | H факт | ΔH | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| lowQ_lowH | −1.005 | −0.586 | 0.419 | −0.389 | −0.424 | −0.034 | 63 |
| lowQ_highH | −1.005 | −1.015 | −0.010 | 0.126 | 0.170 | 0.045 | 64 |
| meanQ_meanH | 2.849 | 3.128 | 0.279 | −0.101 | −0.109 | −0.008 | 64 |
| highQ_lowH | 6.990 | 7.176 | 0.187 | −0.389 | −0.366 | 0.023 | 64 |
| highQ_highH | 6.990 | 8.981 | 1.991 | 0.126 | 0.077 | −0.049 | 64 |

Модель хорошо ловит заряд. По гидрофобности знак и порядок целей тоже верные, но на high-Q/high-H H чуть недотягивает до цели (0.077 против 0.126).

InterfaceScale — это ΔG: **более отрицательное** значение значит сильнее интерфейс. «Высокая H» в таблице — 90-й перцентиль этой знаковой шкалы, не «более гидрофобный» в бытовом смысле.

## Как пользоваться

```bash
PYTHONPATH=src python3 scripts/generate_cond.py --charge 4.0 --hydrophobicity -0.2 --n 32
PYTHONPATH=src python3 scripts/eval_controllability.py
```

Код: `src/amp_competition/generator/conditioning.py`, `scripts/train_cond_lora.py`, `scripts/generate_cond.py`, `scripts/eval_controllability.py`, конфиг `configs/train_cond.yaml`. Сырые цифры этого отчёта — `controllability.json`.
