"""Unit tests for concentration parsing and deterministic features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psp.data.concentration import parse_concentration, ug_ml_to_uM
from psp.features import featurize_sequence, net_charge


def test_parse_exact():
    p = parse_concentration("128")
    assert p.censor_type == "exact" and p.value == 128


def test_parse_right():
    p = parse_concentration(">128")
    assert p.censor_type == "right" and p.lower == 128 and p.value is None
    p = parse_concentration(">=256")
    assert p.censor_type == "right" and p.lower == 256


def test_parse_left():
    p = parse_concentration("<4")
    assert p.censor_type == "left" and p.upper == 4


def test_parse_interval():
    p = parse_concentration("128-256")
    assert p.censor_type == "interval" and p.lower == 128 and p.upper == 256


def test_parse_pm():
    p = parse_concentration("0.5±0.2")
    assert p.censor_type == "exact" and abs(p.value - 0.5) < 1e-9


def test_ug_ml_conversion():
    # Magainin-ish short peptide
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    v = ug_ml_to_uM(100.0, seq)
    assert v is not None and v > 0


def test_features_deterministic():
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    a = featurize_sequence(seq)
    b = featurize_sequence(seq)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k], k


def test_net_charge_sign():
    # Poly-K should be strongly positive at pH 7.4
    assert net_charge("KKKKKK", 7.4) > 4
    # Poly-D should be strongly negative
    assert net_charge("DDDDDD", 7.4) < -4
