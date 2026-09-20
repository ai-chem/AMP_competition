"""AMP sequences plus charge / interface-hydrophobicity conditions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from amp_competition.data.peptides import load_peptide_csv, split_train_val
from amp_competition.features.descriptors import compute_descriptor_for_sequence
from amp_competition.features.physchem_constants import HYDROPHOBICITY_SCALES

CHARGE_KEY = "charge_pH7_4"
HYDROPHOBICITY_KEY = "hydrophobicity_interfaceScale_pH8"
CONDITION_KEYS = (CHARGE_KEY, HYDROPHOBICITY_KEY)

if HYDROPHOBICITY_KEY not in {f"hydrophobicity_{scale}" for scale in HYDROPHOBICITY_SCALES}:
    raise RuntimeError(f"{HYDROPHOBICITY_KEY} is not produced by descriptors.py")


@dataclass(frozen=True)
class ConditionedPeptide:
    sequence: str
    charge_pH7_4: float
    hydrophobicity_interfaceScale_pH8: float

    def vector(self) -> tuple[float, float]:
        return self.charge_pH7_4, self.hydrophobicity_interfaceScale_pH8


def conditions_for_sequence(sequence: str) -> tuple[float, float]:
    """Charge and interface hydrophobicity from ``descriptors.py``."""
    row = compute_descriptor_for_sequence(sequence)
    return float(row[CHARGE_KEY]), float(row[HYDROPHOBICITY_KEY])


def annotate_sequences(sequences: Iterable[str]) -> list[ConditionedPeptide]:
    rows: list[ConditionedPeptide] = []
    for sequence in sequences:
        charge, hydrophobicity = conditions_for_sequence(sequence)
        rows.append(
            ConditionedPeptide(
                sequence=sequence,
                charge_pH7_4=charge,
                hydrophobicity_interfaceScale_pH8=hydrophobicity,
            )
        )
    return rows


def load_split_sequences(path: Path | str) -> tuple[list[str], list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    train = list(payload["train"])
    val = list(payload["val"])
    if not train or not val:
        raise ValueError(f"Empty train/val split in {path}")
    return train, val


def build_conditioned_split(
    csv_path: Path | str,
    *,
    split_path: Path | str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    min_length: int = 8,
    max_length: int = 50,
) -> tuple[list[ConditionedPeptide], list[ConditionedPeptide], dict[str, Any]]:
    """Reuse V1 split when present; otherwise length-stratify the CSV."""
    if split_path is not None and Path(split_path).is_file():
        train_seqs, val_seqs = load_split_sequences(split_path)
        data_stats = {"source": "split", "split": str(split_path)}
    else:
        sequences, data_stats = load_peptide_csv(
            csv_path, min_length=min_length, max_length=max_length
        )
        train_seqs, val_seqs = split_train_val(sequences, val_fraction=val_fraction, seed=seed)
        data_stats = {"source": "csv", **data_stats}

    train = annotate_sequences(train_seqs)
    val = annotate_sequences(val_seqs)
    data_stats.update({"n_train": len(train), "n_val": len(val)})
    return train, val, data_stats
