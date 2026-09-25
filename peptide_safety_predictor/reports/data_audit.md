# Data audit — HC50 observations

- DBAASP activity rows extracted: **4579**
- Usable (decision=ok): **3851**
- Model-ready (AA20, in-domain chemistry): **2840**
- Unique sequences: **2446**
- Censoring: exact=1763, right=1044, left=15, interval=18
- P(HC50>128) labels known: 2510 (positives=1271)
- Sequences with multiple measurements: 226
- Sequences with conflicting exact values (>2×): 89
- Overlap with HemoPI2 curated unique sequences: 1589
- Overlap with ConsAMPHemo regression: 1150
- Source counts: {'DBAASP': 2840}

## Processing decisions

```
{
  "ok": 3851,
  "unit_conversion_failed": 400,
  "ugml_conversion_skipped_modified": 279,
  "unparsable_concentration:unmatched": 49
}
```

## Notes

- Conflicting duplicate measurements are **retained** (not averaged).
- HemoPI2 / ConsAMPHemo curated sets are secondary/cross-check sources;
  their authors collapsed ranges/censoring, so DBAASP raw records are preferred
  whenever available.
- µg/mL → µM conversion applied only for canonical linear unmodified peptides.
