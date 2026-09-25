# Data audit — expanded HC50 observations

- All-decision rows: **24295**
- Usable (decision=ok): **17053**
- Model-ready (AA20): **14504** (6948 unique sequences)
- Censoring: {'right': 8417, 'exact': 4988, 'left': 1068, 'interval': 31}
- Sources: {'Hemolytik2': 9486, 'DBAASP': 3267, 'DBAASP_CSV': 1751}
- Erythrocyte species: {'human': 10679, 'horse': 849, 'sheep': 669, 'mouse': 650, 'rat': 635, 'rabbit': 382, 'pig': 252, 'fish': 68, 'chicken': 56, 'guinea-pig': 54, 'porcine': 54, 'unknown_rbc': 29, 'murine': 27, 'cattle': 21, 'cow': 19, 'bovine': 19, 'canine': 13, 'nan': 7, 'dog': 6, 'lizard': 4, 'pigeons': 4, 'goat': 3, 'swiss mouse': 2, 'hagfish': 1, 'mus musculus': 1}
- P(HC50>128) known: 8932 (positives=4398)
- Multi-measurement sequences: 3067; conflicts (>2x exact): 388

## Notes

- Conflicts retained. No averaging.
- ConsAMPHemo regression targets excluded (collapsed censoring).
- HemoPI2 secondary only.
