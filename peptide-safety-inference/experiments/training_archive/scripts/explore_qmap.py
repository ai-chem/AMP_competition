#!/usr/bin/env python
"""Inspect the QMAP benchmark metrics API and HC50 value semantics."""

import inspect

import numpy as np

from qmap import DBAASPDataset, QMAPBenchmark
from qmap.benchmark.dataset import QMAP_metrics
from qmap.benchmark import benchmark as bench_mod

print("==== QMAP_metrics ====")
print(inspect.getsource(QMAP_metrics))
print("==== benchmark.py ====")
print(inspect.getsource(bench_mod))

ds = DBAASPDataset()
h = ds.with_hc50()
cons = np.array([s.hc50.consensus for s in h.samples], dtype=float)
mn = np.array([s.hc50.min_hc50 for s in h.samples], dtype=float)
mx = np.array([s.hc50.max_hc50 for s in h.samples], dtype=float)
print("\nHC50 consensus quantiles:", np.nanpercentile(cons, [1, 5, 25, 50, 75, 95, 99]).round(3))
print("min==max (single measurement):", int((mn == mx).sum()), "of", len(cons))
print("range>2x:", int((mx > 2 * mn).sum()))
print("looks like log scale?", cons.min(), cons.max())

print("\nterminals:", {
    "nterm_ACT": sum(s.nterminal == "ACT" for s in h.samples),
    "cterm_AMD": sum(s.cterminal == "AMD" for s in h.samples),
    "with_bonds": sum(len(s.bonds) > 0 for s in h.samples),
})
alpha = set("".join(s.sequence for s in h.samples))
print("alphabet:", "".join(sorted(alpha)))
