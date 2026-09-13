"""Physicochemical descriptors for peptides (reusable in analysis and generation)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import peptides

from amp_competition.features.physchem_constants import (
    ACIDIC_RESIDUES,
    AROMATIC_RESIDUES,
    BASIC_RESIDUES,
    FLEXIBILITY_RESIDUES,
    HYDROPHOBICITY_SCALES,
    HYDROPHOBIC_RESIDUES,
)
if TYPE_CHECKING:
    import pandas as pd

FastaRecord = tuple[str, str]


def safe_hydrophobic_moment(peptide: peptides.Peptide, sequence: str, angle: int) -> float:
    # peptides.py requires window <= sequence length.
    window = min(11, len(sequence))
    return peptide.hydrophobic_moment(window=window, angle=angle)


def descriptor_row(header: str, sequence: str, label: str) -> dict[str, Any]:
    """Compute one descriptor row for a single peptide sequence."""
    peptide = peptides.Peptide(sequence)
    frequencies = peptide.frequencies()

    row: dict[str, Any] = {
        "id": header.split()[0],
        "sequence": sequence,
        "class": label,
        "length": len(sequence),
        "molecular_weight": peptide.molecular_weight(),
        "charge_pH7_4": peptide.charge(pH=7.4, pKscale="Lehninger"),
        "isoelectric_point": peptide.isoelectric_point(pKscale="EMBOSS"),
        "fraction_basic_KR": sum(frequencies.get(aa, 0.0) for aa in BASIC_RESIDUES),
        "fraction_acidic_DE": sum(frequencies.get(aa, 0.0) for aa in ACIDIC_RESIDUES),
        "fraction_hydrophobic": sum(
            frequencies.get(aa, 0.0) for aa in HYDROPHOBIC_RESIDUES
        ),
        "fraction_aromatic": sum(
            frequencies.get(aa, 0.0) for aa in AROMATIC_RESIDUES
        ),
        "fraction_GP": sum(
            frequencies.get(aa, 0.0) for aa in FLEXIBILITY_RESIDUES
        ),
        "fraction_C": frequencies.get("C", 0.0),
        "boman_index": peptide.boman(),
        "hydrophobic_moment_alpha": safe_hydrophobic_moment(
            peptide, sequence, angle=100
        ),
        "hydrophobic_moment_beta": safe_hydrophobic_moment(
            peptide, sequence, angle=160
        ),
    }

    for scale in HYDROPHOBICITY_SCALES:
        row[f"hydrophobicity_{scale}"] = peptide.hydrophobicity(scale=scale)

    return row


def compute_descriptor_for_sequence(sequence: str, *, label: str = "AMP") -> dict[str, Any]:
    """Compute descriptors for one sequence (for generation / filtering reuse)."""
    return descriptor_row("seq1", sequence, label)


def compute_descriptors(records: list[FastaRecord], label: str) -> "pd.DataFrame":
    import pandas as pd

    rows: list[dict[str, Any]] = []

    for index, (header, sequence) in enumerate(records, start=1):
        rows.append(descriptor_row(header, sequence, label))

        if index % 10000 == 0:
            print(f"Computed descriptors for {index:,} {label} sequences")

    return pd.DataFrame(rows)


def compute_descriptors_for_sequences(
    sequences: list[str],
    label: str,
    *,
    id_prefix: str = "seq",
) -> "pd.DataFrame":
    """Compute descriptor table from bare sequences (for generation / filtering reuse)."""
    records = [(f"{id_prefix}{index}", sequence) for index, sequence in enumerate(sequences, start=1)]
    return compute_descriptors(records, label)


def descriptor_columns(df: "pd.DataFrame") -> list[str]:
    import pandas as pd

    excluded = {"id", "sequence", "class"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
