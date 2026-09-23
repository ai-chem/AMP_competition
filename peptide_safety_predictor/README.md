# Предсказание гемолиза и стабильности пептидов

Локальный инференс по CSV со столбцом `sequence` (20 стандартных аминокислот). Веса лежат в `external/`.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Нужен Python 3.11.

## Запуск

Из каталога `peptide_safety_predictor`:

```bash
python predict.py --input examples/controllability_sequences.csv --output predictions.csv
```

В выходном CSV сохраняются все исходные столбцы и добавляются предсказания.

## Ограничения

Гемолиз, HemoPI2. Число `hemopi2_hc50_uM` — оценка HC50 в мкМ (концентрация 50% лизиса эритроцитов), не измерение. Последовательности длиннее 40 остатков обрезаются до первых 40; такие строки помечены `hemopi2_truncated_to_40`. `hemopi2_class_if_hc50_lt_100` — правило HemoPI2: Hemolytic при HC50 < 100 мкМ. Это не порог 128 мкМ. `hemopi2_tree_fraction_hc50_gt_128` — доля деревьев леса, у которых оценка HC50 > 128 мкМ; это не откалиброванная вероятность.

Стабильность, ML_Peptide SIF. `ml_peptide_sif_class` — класс в simulated intestinal fluid: Not Stable, Partly Stable, Stable. Это не период полужизни в крови и не стабильность в буфере MIC или гемолиза. Последовательность считается линейным немодифицированным all-L пептидом. `ml_peptide_sif_p_not_stable`, `ml_peptide_sif_p_partly_stable`, `ml_peptide_sif_p_stable` — вероятности трёх классов.
