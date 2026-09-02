# Data

Keep large FASTA/CSV dumps out of git. Commit only small samples, schemas, and download scripts.

| Path | Contents |
|---|---|
| `raw/` | Unmodified downloads (DBAASP, DRAMP, GRAMPA, organizer files) |
| `processed/` | Cleaned AMP corpus and train/val splits |
| `external/` | Reference files such as `antibacterial.fasta` from the organizers |

The organizer reference set belongs here as `external/antibacterial.fasta` once task 2.1 is done.
