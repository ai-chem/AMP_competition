"""Deterministic sequence-only physicochemical + composition features."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from psp.paths import AA20

# --- scales -----------------------------------------------------------------
# Kyte-Doolittle hydrophobicity
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Eisenberg consensus
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
# Wimley-White interfacial (pH ~8) — matches the CSV metadata scale family
WW_INTERFACE = {
    "A": 0.17, "R": 0.81, "N": 0.42, "D": 1.23, "C": -0.24,
    "Q": 0.58, "E": 2.02, "G": 0.01, "H": 0.96, "I": -0.31,
    "L": -0.56, "K": 0.99, "M": -0.23, "F": -1.13, "P": 0.45,
    "S": 0.13, "T": 0.14, "W": -1.85, "Y": -0.94, "V": 0.07,
}
# Hopp-Woods hydrophilicity
HOPP_WOODS = {
    "A": -0.5, "R": 3.0, "N": 0.2, "D": 3.0, "C": -1.0,
    "Q": 0.2, "E": 3.0, "G": 0.0, "H": -0.5, "I": -1.8,
    "L": -1.8, "K": 3.0, "M": -1.3, "F": -2.5, "P": 0.0,
    "S": 0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
}
# Janin / transfer free energy-ish for membrane propensity
JANIN = {
    "A": 0.3, "R": -1.4, "N": -0.5, "D": -0.6, "C": 0.9,
    "Q": -0.7, "E": -0.7, "G": 0.3, "H": -0.1, "I": 0.7,
    "L": 0.5, "K": -1.8, "M": 0.4, "F": 0.5, "P": -0.3,
    "S": -0.1, "T": -0.2, "W": 0.3, "Y": -0.4, "V": 0.6,
}
# Helical propensity (Chou-Fasman-ish normalized)
HELIX = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
# Side-chain pKa for charge at given pH
PKA_SIDE = {"D": 3.9, "E": 4.3, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
PKA_NTERM = 9.0
PKA_CTERM = 2.2

# Boman (potential protein-binding) residues average values
BOMAN = {
    "A": 0.17, "R": 0.81, "N": 0.42, "D": 0.50, "C": -0.24,
    "Q": 0.58, "E": 0.50, "G": 0.01, "H": 0.96, "I": -0.31,
    "L": -0.56, "K": 0.99, "M": -0.23, "F": -1.13, "P": 0.45,
    "S": 0.13, "T": 0.14, "W": -1.85, "Y": -0.94, "V": 0.07,
}

# Guruprasad instability dipeptide weights (subset — full table is large;
# we use a residue-level proxy + known unstable dipeptides count).
_UNSTABLE_DI = {
    "DG", "EG", "DW", "DP", "NG", "NS", "NT", "NN", "NW", "NP",
    "QG", "QS", "QT", "QN", "QW", "QP", "GG", "GS", "GT", "GN",
    "GW", "GP", "SG", "SS", "ST", "SN", "SW", "SP", "TG", "TS",
    "TT", "TN", "TW", "TP", "WG", "WS", "WT", "WN", "WW", "WP",
    "PG", "PS", "PT", "PN", "PW", "PP",
}

# Protease cleavage motifs (simple pattern counts)
_CLEAVAGE = {
    "trypsin_KR": ("K", "R"),
    "chymotrypsin_FWY": ("F", "W", "Y"),
    "pepsin_FL": ("F", "L"),
    "gluC_DE": ("D", "E"),
}

_AA_MW = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "Q": 128.1307, "E": 129.1155, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}


def _mean_scale(seq: str, scale: Dict[str, float]) -> float:
    if not seq:
        return 0.0
    return float(np.mean([scale.get(a, 0.0) for a in seq]))


def _hydrophobic_moment(seq: str, scale: Dict[str, float], angle_deg: float = 100.0) -> float:
    """Eisenberg hydrophobic moment for an assumed α-helix (100°) or beta (180°)."""
    if not seq:
        return 0.0
    angle = np.deg2rad(angle_deg)
    sx = sy = 0.0
    for i, a in enumerate(seq):
        h = scale.get(a, 0.0)
        sx += h * np.cos(i * angle)
        sy += h * np.sin(i * angle)
    return float(np.sqrt(sx * sx + sy * sy) / len(seq))


def net_charge(seq: str, pH: float = 7.4) -> float:
    """Henderson-Hasselbalch net charge including termini."""
    if not seq:
        return 0.0
    charge = 1.0 / (1.0 + 10 ** (pH - PKA_NTERM))  # +N-terminus
    charge -= 1.0 / (1.0 + 10 ** (PKA_CTERM - pH))  # -C-terminus
    for a in seq:
        if a in {"K", "R"}:
            charge += 1.0 / (1.0 + 10 ** (pH - PKA_SIDE[a]))
        elif a == "H":
            charge += 1.0 / (1.0 + 10 ** (pH - PKA_SIDE["H"]))
        elif a in {"D", "E"}:
            charge -= 1.0 / (1.0 + 10 ** (PKA_SIDE[a] - pH))
        elif a == "C":
            charge -= 1.0 / (1.0 + 10 ** (PKA_SIDE["C"] - pH))
        elif a == "Y":
            charge -= 1.0 / (1.0 + 10 ** (PKA_SIDE["Y"] - pH))
    return float(charge)


def isoelectric_point(seq: str, lo: float = 0.0, hi: float = 14.0, tol: float = 1e-4) -> float:
    """Binary-search pI where net charge ≈ 0."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = net_charge(seq, mid)
        if abs(c) < tol:
            return mid
        if c > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def molecular_weight(seq: str) -> float:
    return float(sum(_AA_MW.get(a, 0.0) for a in seq) + 18.01528)


def aliphatic_index(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    a = seq.count("A") / n * 100
    v = seq.count("V") / n * 100
    i = seq.count("I") / n * 100
    l = seq.count("L") / n * 100
    return a + 2.9 * v + 3.9 * (i + l)


def instability_index(seq: str) -> float:
    """Residue-level proxy + unstable-dipeptide enrichment (Guruprasad-inspired)."""
    if len(seq) < 2:
        return 0.0
    di = sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] in _UNSTABLE_DI)
    return 10.0 * di / (len(seq) - 1) * 100.0 / 10.0  # scaled dipeptide fraction * 100


def camsol_like_score(seq: str) -> float:
    """Lightweight CamSol-inspired intrinsic solubility proxy.

    Combines charge (favours solubility), hydrophobicity (penalises), and
    helical propensity of hydrophobic stretches. Not the official CamSol.
    """
    if not seq:
        return 0.0
    q = abs(net_charge(seq, 7.4))
    h = _mean_scale(seq, EISENBERG)
    arom = sum(seq.count(a) for a in "FWY") / len(seq)
    # Higher score ⇒ more soluble
    return float(0.5 * q - 1.2 * h - 0.8 * arom)


def featurize_sequence(seq: str) -> Dict[str, float]:
    """Return a flat dict of deterministic features for one uppercase AA20 sequence."""
    seq = seq.upper()
    n = max(len(seq), 1)
    feats: Dict[str, float] = {}
    feats["length"] = float(len(seq))
    feats["mw"] = molecular_weight(seq)
    feats["charge_pH7_4"] = net_charge(seq, 7.4)
    feats["charge_pH5"] = net_charge(seq, 5.0)
    feats["pI"] = isoelectric_point(seq)
    feats["charge_density"] = feats["charge_pH7_4"] / n
    feats["hydrophobicity_KD"] = _mean_scale(seq, KD)
    feats["hydrophobicity_Eisenberg"] = _mean_scale(seq, EISENBERG)
    feats["hydrophobicity_WW"] = _mean_scale(seq, WW_INTERFACE)
    feats["hydrophobicity_HoppWoods"] = _mean_scale(seq, HOPP_WOODS)
    feats["hydrophobicity_Janin"] = _mean_scale(seq, JANIN)
    feats["hydrophobic_moment_helix"] = _hydrophobic_moment(seq, EISENBERG, 100.0)
    feats["hydrophobic_moment_beta"] = _hydrophobic_moment(seq, EISENBERG, 180.0)
    feats["aromatic_frac"] = sum(seq.count(a) for a in "FWY") / n
    feats["trp_frac"] = seq.count("W") / n
    feats["phe_frac"] = seq.count("F") / n
    feats["tyr_frac"] = seq.count("Y") / n
    feats["lys_arg_frac"] = (seq.count("K") + seq.count("R")) / n
    feats["acidic_frac"] = (seq.count("D") + seq.count("E")) / n
    feats["hydrophobic_frac"] = sum(seq.count(a) for a in "AILMFVW") / n
    feats["aliphatic_index"] = aliphatic_index(seq)
    feats["boman_index"] = _mean_scale(seq, BOMAN)
    feats["instability_index"] = instability_index(seq)
    feats["helical_propensity"] = _mean_scale(seq, HELIX)
    feats["aggregation_propensity"] = (
        feats["hydrophobicity_Eisenberg"] + 0.5 * feats["aromatic_frac"] - 0.3 * abs(feats["charge_pH7_4"])
    )
    feats["amphipathicity"] = feats["hydrophobic_moment_helix"]
    # Terminal composition
    for pos, label in ((0, "n1"), (1, "n2"), (-1, "c1"), (-2, "c2")):
        if abs(pos) < len(seq) or (pos >= 0 and pos < len(seq)):
            try:
                aa = seq[pos]
            except IndexError:
                aa = ""
        else:
            aa = ""
        for a in AA20:
            feats[f"term_{label}_{a}"] = 1.0 if aa == a else 0.0
    # AAC
    for a in AA20:
        feats[f"aac_{a}"] = seq.count(a) / n
    # DPC (400 dims — useful but large; keep)
    for a in AA20:
        for b in AA20:
            key = f"dpc_{a}{b}"
            if len(seq) < 2:
                feats[key] = 0.0
            else:
                feats[key] = sum(1 for i in range(len(seq) - 1) if seq[i] == a and seq[i + 1] == b) / (
                    len(seq) - 1
                )
    # Cleavage / chemical stability
    for name, residues in _CLEAVAGE.items():
        feats[f"cleavage_{name}"] = sum(seq.count(r) for r in residues) / n
    feats["oxidation_sensitive"] = (seq.count("M") + seq.count("C") + seq.count("W")) / n
    feats["deamidation_prone"] = (seq.count("N") + seq.count("Q")) / n
    feats["camsol_like"] = camsol_like_score(seq)
    return feats


# Columns that are "physchem summary" (exclude AAC/DPC/term one-hots) for OOD.
PHYSCHEM_SUMMARY_COLS = [
    "length", "mw", "charge_pH7_4", "charge_pH5", "pI", "charge_density",
    "hydrophobicity_KD", "hydrophobicity_Eisenberg", "hydrophobicity_WW",
    "hydrophobicity_HoppWoods", "hydrophobicity_Janin",
    "hydrophobic_moment_helix", "hydrophobic_moment_beta",
    "aromatic_frac", "trp_frac", "phe_frac", "tyr_frac",
    "lys_arg_frac", "acidic_frac", "hydrophobic_frac",
    "aliphatic_index", "boman_index", "instability_index",
    "helical_propensity", "aggregation_propensity", "amphipathicity",
    "cleavage_trypsin_KR", "cleavage_chymotrypsin_FWY", "cleavage_pepsin_FL",
    "cleavage_gluC_DE", "oxidation_sensitive", "deamidation_prone", "camsol_like",
]


def featurize_frame(sequences: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = [featurize_sequence(s) for s in sequences]
    return pd.DataFrame(rows)
