# Data

Keep large FASTA/CSV dumps out of git. Commit only small samples, schemas, and download scripts.

| Path | Contents |
|---|---|
| `raw/` | Unmodified downloads. `antibacterial_clean.csv` is the cleaned organizer reference and is tracked |
| `processed/` | Cleaned AMP corpus and train/val splits. GRAMPA is rebuilt here by `build_MIC_dataset.py` |
| `external/` | Reference files such as `antibacterial.fasta` from the organizers |

The organizer reference set is fetched as an immutable input together with a
provenance manifest:

```bash
uv run python scripts/fetch_organizer_reference.py
```

This writes `external/antibacterial.fasta` and
`external/antibacterial.manifest.json`. The script pins a full upstream Git
commit and records the FASTA SHA-256; do not replace either file manually.
