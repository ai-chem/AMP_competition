"""Parse free-text hemolysis activity strings into censored HC50 observations.

Handles Hemolytik 2.0-style strings such as:
  ``HC50 >200 μg/ml``, ``50% Hemolysis at >300 µM``,
  ``<15% Hemolysis at 500 µM``, ``0% Hemolysis at 100 μg/ml``,
  ``LC50 = 1.4±0.2 µM``, ``HD50 = 11.7 µM``, ``MHC >1 μg/ml``.

Percent-hemolysis-at-concentration is converted to a censored HC50:
* %hemolysis < 50 at C  -> HC50 > C  (right)
* %hemolysis > 50 at C  -> HC50 < C  (left)
* %hemolysis ≈ 50 at C  -> HC50 ≈ C  (exact)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from psp.data.concentration import ParsedConcentration, parse_concentration

_NUM = r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_UNIT = r"(µ?u?m|μ?m|ug/?ml|µg/?ml|μg/?ml|mg/?l|mg/?ml)"

# Explicit ~50% hemolysis endpoints.
_HC50_LABEL = re.compile(
    r"(?i)\b(hc\s*50|hd\s*50|mhc(?:\s*50)?|lc\s*50|ec\s*50)\b"
)
_PCT_HEMOL = re.compile(
    rf"(?i)(?:(~|>|<|>=|<=)?\s*{_NUM}\s*%?\s*%?\s*hemol|"
    rf"hemolysis\s*(?:of|=|:)?\s*(~|>|<|>=|<=)?\s*{_NUM}\s*%?)"
)
_AT_CONC = re.compile(
    rf"(?i)(?:at|@|=|:)\s*(>=|<=|>|<|~)?\s*{_NUM}\s*{_UNIT}"
)
_BARE_CONC = re.compile(rf"(?i)(>=|<=|>|<|~)?\s*{_NUM}\s*{_UNIT}")


@dataclass
class HemolysisParse:
    censor_type: str  # exact | right | left | interval | unparsable | skip
    value: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    unit: Optional[str]  # uM | ug/mL
    original: str
    endpoint_kind: str  # hc50_label | pct_at_conc | non_hemolytic_flag | skip
    notes: str = ""


def _norm_unit(u: str) -> str:
    u = u.lower().replace("μ", "µ").replace(" ", "")
    if u in {"µm", "um", "μm", "micromolar"}:
        return "uM"
    if u in {"µg/ml", "ug/ml", "µg/ml", "ugml", "µgml", "mg/l", "mgl"}:
        return "ug/mL"
    if u in {"mg/ml", "mgml"}:
        return "ug/mL"  # will be scaled 1000x by caller if needed; rare
    return u


def parse_hemolysis_activity(raw: object) -> HemolysisParse:
    """Parse a Hemolytik/DBAASP free-text hemolysis activity string."""
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return HemolysisParse("unparsable", None, None, None, None, "", "skip", "empty")
    text = str(raw).strip()
    original = text
    if not text or text.lower() in {"na", "n/a", "nan", "-", ".", "none"}:
        return HemolysisParse("unparsable", None, None, None, None, original, "skip", "empty")

    # Soft non-hemolytic free-text without a concentration — not usable as HC50.
    soft = text.lower()
    if soft in {"poor hemolytic", "non-hemolytic", "non hemolytic", "low hemolytic",
                "nonhemolytic", "negligible hemolysis"}:
        return HemolysisParse("unparsable", None, None, None, None, original,
                              "non_hemolytic_flag", "no_concentration")

    # ---- Explicit HC50 / HD50 / MHC / LC50 / EC50 labels ----
    if _HC50_LABEL.search(text):
        # Strip the label, leave the concentration part.
        rest = _HC50_LABEL.sub(" ", text)
        rest = re.sub(r"(?i)(hemolysis|hemolytic|activity)", " ", rest)
        m = _BARE_CONC.search(rest)
        if m:
            op, num, unit = m.group(1) or "", float(m.group(2)), _norm_unit(m.group(3))
            if unit not in {"uM", "ug/mL"}:
                return HemolysisParse("unparsable", None, None, None, None, original,
                                      "hc50_label", f"bad_unit:{unit}")
            if op in {">", ">="}:
                return HemolysisParse("right", None, num, None, unit, original, "hc50_label")
            if op in {"<", "<="}:
                return HemolysisParse("left", None, None, num, unit, original, "hc50_label")
            # Try ± form via the shared concentration parser on the numeric blob.
            blob = rest[m.start(): m.end()]
            # Rebuild a parseable concentration string without unit.
            conc_str = f"{op or ''}{m.group(2)}"
            # Check for ± in the surrounding text
            pm = re.search(rf"{re.escape(m.group(2))}\s*[±\+\-/\s]*(\d+(?:\.\d+)?)", rest)
            if "±" in rest or "+/-" in rest.replace(" ", ""):
                pm2 = re.search(
                    rf"{re.escape(m.group(2))}\s*(?:±|\+/-)\s*{_NUM}", rest
                )
                if pm2:
                    a, b = float(m.group(2)), float(pm2.group(1))
                    return HemolysisParse("exact", a, max(0.0, a - b), a + b, unit,
                                          original, "hc50_label", f"pm={b}")
            parsed = parse_concentration(conc_str)
            if parsed.censor_type == "exact":
                return HemolysisParse("exact", parsed.value, parsed.lower, parsed.upper,
                                      unit, original, "hc50_label", parsed.notes)
            if parsed.censor_type != "unparsable":
                return HemolysisParse(parsed.censor_type, parsed.value, parsed.lower,
                                      parsed.upper, unit, original, "hc50_label",
                                      parsed.notes)
        # Fall through: maybe "HC50 = 100" without unit — reject.
        return HemolysisParse("unparsable", None, None, None, None, original,
                              "hc50_label", "no_unit_or_value")

    # ---- Percent hemolysis at a concentration ----
    pct_m = re.search(
        rf"(?i)(~|>|<|>=|<=)?\s*{_NUM}\s*%\s*(?:hemol\w*)", text
    )
    if not pct_m:
        pct_m = re.search(
            rf"(?i)(?:hemolysis\s*(?:of|=|:)?\s*)(~|>|<|>=|<=)?\s*{_NUM}\s*%", text
        )
    conc_m = _AT_CONC.search(text) or _BARE_CONC.search(text)

    if pct_m and conc_m:
        pct_op = (pct_m.group(1) or "").strip()
        pct = float(pct_m.group(2))
        conc_op = (conc_m.group(1) or "").strip()
        conc = float(conc_m.group(2))
        unit = _norm_unit(conc_m.group(3))
        if unit not in {"uM", "ug/mL"}:
            return HemolysisParse("unparsable", None, None, None, None, original,
                                  "pct_at_conc", f"bad_unit:{unit}")

        # Effective percent bound.
        # "<15% Hemolysis at 500" -> pct_upper=15 -> definitely <50 -> HC50 > 500
        # ">50% Hemolysis at 80"  -> pct_lower=50 -> at/above 50 -> HC50 <= 80
        # "50% Hemolysis at >300" -> exact-ish at bound with right censor on conc
        if pct_op in {"<", "<="} or (pct_op == "" and pct < 45):
            # Hemolysis below 50% at this concentration => HC50 > conc
            # If concentration itself is censored (">300"), keep right at that floor.
            floor = conc
            if conc_op in {">", ">="}:
                floor = conc
            return HemolysisParse("right", None, floor, None, unit, original,
                                  "pct_at_conc", f"pct={pct_op}{pct}")
        if pct_op in {">", ">="} or (pct_op == "" and pct > 55):
            ceil = conc
            if conc_op in {"<", "<="}:
                ceil = conc
            return HemolysisParse("left", None, None, ceil, unit, original,
                                  "pct_at_conc", f"pct={pct_op}{pct}")
        # ~50%
        if conc_op in {">", ">="}:
            return HemolysisParse("right", None, conc, None, unit, original,
                                  "pct_at_conc", "approx50_right_conc")
        if conc_op in {"<", "<="}:
            return HemolysisParse("left", None, None, conc, unit, original,
                                  "pct_at_conc", "approx50_left_conc")
        return HemolysisParse("exact", conc, None, None, unit, original,
                              "pct_at_conc", "approx50")

    # Percent without concentration, or concentration without percent — skip.
    if "hemol" in soft:
        return HemolysisParse("unparsable", None, None, None, None, original,
                              "skip", "hemolysis_without_usable_pair")
    return HemolysisParse("unparsable", None, None, None, None, original,
                          "skip", "unmatched")
