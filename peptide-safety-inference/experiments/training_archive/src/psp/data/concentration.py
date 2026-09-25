"""Parse concentration strings from DBAASP / literature into censored observations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

_NUM = r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_PM = r"(?:±|\+/-|\+\/\-)"


@dataclass
class ParsedConcentration:
    censor_type: str  # exact | right | left | interval | unparsable
    value: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    original: str
    notes: str = ""


def _f(x: str) -> float:
    return float(x)


def parse_concentration(raw: object) -> ParsedConcentration:
    """Parse a free-text concentration into a censored observation."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ParsedConcentration("unparsable", None, None, None, "", "empty")
    text = str(raw).strip()
    original = text
    if not text or text.lower() in {"na", "n/a", "nan", "none", "-", "."}:
        return ParsedConcentration("unparsable", None, None, None, original, "empty")

    text = (
        text.replace("≥", ">=")
        .replace("≤", "<=")
        .replace("＞", ">")
        .replace("＜", "<")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace(",", ".")
        .replace(" ", "")
    )
    text = text.replace("approx.", "~").replace("approximately", "~")
    text = re.sub(r"(?i)(about|ca\.?|circa)", "~", text)
    text = re.sub(r"(?i)(µ?u?m|ug/?ml|µg/?ml|mg/?l|mg/?ml)$", "", text)

    m = re.fullmatch(rf"{_NUM}-{_NUM}", text)
    if m:
        a, b = _f(m.group(1)), _f(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return ParsedConcentration("interval", 0.5 * (lo + hi), lo, hi, original)

    m = re.fullmatch(rf"{_NUM}{_PM}{_NUM}", text)
    if m:
        a, b = _f(m.group(1)), _f(m.group(2))
        return ParsedConcentration(
            "exact", a, max(0.0, a - b), a + b, original, f"pm={b}"
        )

    m = re.fullmatch(rf"(?:>=|>){_NUM}", text)
    if m:
        c = _f(m.group(1))
        return ParsedConcentration("right", None, c, None, original)

    m = re.fullmatch(rf"(?:<=|<){_NUM}", text)
    if m:
        c = _f(m.group(1))
        return ParsedConcentration("left", None, None, c, original)

    m = re.fullmatch(rf"[~≈]{_NUM}", text)
    if m:
        a = _f(m.group(1))
        return ParsedConcentration("exact", a, None, None, original, "approximate")

    m = re.fullmatch(_NUM, text)
    if m:
        a = _f(m.group(1))
        return ParsedConcentration("exact", a, None, None, original)

    return ParsedConcentration("unparsable", None, None, None, original, "unmatched")


_AA_MW = {
    "A": 71.0788, "R": 156.1875, "N": 114.1038, "D": 115.0886, "C": 103.1388,
    "Q": 128.1307, "E": 129.1155, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "L": 113.1594, "K": 128.1741, "M": 131.1926, "F": 147.1766, "P": 97.1167,
    "S": 87.0782, "T": 101.1051, "W": 186.2132, "Y": 163.1760, "V": 99.1326,
}
_WATER = 18.01528


def molecular_weight_da(sequence: str) -> Optional[float]:
    seq = sequence.upper()
    if not seq or any(a not in _AA_MW for a in seq):
        return None
    return sum(_AA_MW[a] for a in seq) + _WATER


def ug_ml_to_uM(value_ug_ml: float, sequence: str) -> Optional[float]:
    mw = molecular_weight_da(sequence)
    if mw is None or mw <= 0:
        return None
    return value_ug_ml / mw * 1000.0
