"""Constants for physicochemical AMP vs non-AMP analysis."""

from __future__ import annotations

from amp_competition.constants import STANDARD_AMINO_ACIDS

CANONICAL_AA = set(STANDARD_AMINO_ACIDS)

# UniAMP-inspired exclusion terms, with challenge length limits.
UNIPROT_QUERY = (
    "length:[8 TO 50] "
    "NOT antimicrobial "
    "NOT antibiotic "
    "NOT antiviral "
    "NOT antifungal "
    "NOT fungicide "
    "NOT secreted "
    "NOT secretory "
    "NOT excreted "
    "NOT effector "
    "NOT defensin"
)

# Representative hydrophobicity scales with distinct physical interpretations.
HYDROPHOBICITY_SCALES = [
    "Eisenberg",
    "KyteDoolittle",
    "interfaceScale_pH8",
    "octanolScale_pH8",
    "oiScale_pH8",
]

HYDROPHOBIC_RESIDUES = set("AVILMFWY")
AROMATIC_RESIDUES = set("FWY")
BASIC_RESIDUES = set("KR")
ACIDIC_RESIDUES = set("DE")
FLEXIBILITY_RESIDUES = set("GP")
