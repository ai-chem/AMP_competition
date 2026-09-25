"""Sequence-identity clustering and leakage-controlled splits."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from tqdm import tqdm


def _aligner() -> PairwiseAligner:
    aln = PairwiseAligner()
    aln.mode = "global"
    aln.match_score = 1.0
    aln.mismatch_score = 0.0
    aln.open_gap_score = -1.0
    aln.extend_gap_score = -0.5
    return aln


def pairwise_identity(a: str, b: str, aligner: Optional[PairwiseAligner] = None) -> Tuple[float, float]:
    """Return (identity, coverage) for a global alignment.

    identity = matched residues / total alignment columns (gaps included).
    coverage = ungapped aligned columns / length of the shorter sequence.

    Counting gap columns in the identity denominator is what keeps a short
    peptide from looking identical to a long one that merely contains it: a
    7-mer inside a 30-mer scores 7/30 = 23% identity, not 100%. It also makes
    `identity >= t` imply `min_len/max_len >= t`, which gives the exact length
    gate used to prune candidates below.
    """
    if a == b:
        return 1.0, 1.0
    if not a or not b:
        return 0.0, 0.0

    aln = aligner or _aligner()
    best = next(iter(aln.align(a, b)))
    seqA, seqB = best[0], best[1]

    matches = sum(1 for x, y in zip(seqA, seqB) if x == y and x != "-")
    aligned = sum(1 for x, y in zip(seqA, seqB) if x != "-" and y != "-")
    aln_len = max(len(seqA), len(seqB))
    if aln_len == 0 or aligned == 0:
        return 0.0, 0.0
    identity = matches / aln_len
    coverage = aligned / min(len(a), len(b))
    return float(identity), float(min(1.0, coverage))


def _kmer_set(seq: str, k: int = 3) -> set:
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def length_gate(a: str, b: str, identity_threshold: float) -> bool:
    """Exact necessary condition for `identity >= identity_threshold`.

    With identity = matches / alignment_length, matches cannot exceed the
    shorter length and the alignment cannot be shorter than the longer one,
    so identity <= min_len / max_len. Pairs failing this can be skipped with
    no risk of a false negative.
    """
    if not a or not b:
        return False
    lo, hi = (len(a), len(b)) if len(a) <= len(b) else (len(b), len(a))
    return lo / hi >= identity_threshold


def candidate_pairs(
    uniq: Sequence[str],
    identity_threshold: float,
    k: int = 3,
) -> Iterable[Tuple[int, int]]:
    """Yield index pairs worth aligning, via a k-mer inverted index.

    Prunes with the exact `length_gate` plus a deliberately loose shared-k-mer
    floor, leaving the real accept/reject decision to the aligner. This avoids
    the full O(n^2) alignment cost; the independent audit in make_splits.py
    re-checks the resulting split all-vs-all.
    """
    kmers = [_kmer_set(s, k) for s in uniq]
    index: Dict[str, List[int]] = {}
    for i, km in enumerate(kmers):
        for m in km:
            index.setdefault(m, []).append(i)

    for i in range(len(uniq)):
        shared: Dict[int, int] = {}
        for m in kmers[i]:
            for j in index[m]:
                if j > i:
                    shared[j] = shared.get(j, 0) + 1
        for j, n_shared in shared.items():
            if not length_gate(uniq[i], uniq[j], identity_threshold):
                continue
            if n_shared < _min_shared_kmers(uniq[i], uniq[j], identity_threshold, k):
                continue
            yield i, j


def _min_shared_kmers(a: str, b: str, identity_threshold: float, k: int) -> int:
    """Loose lower bound on shared k-mers for a pair at the identity cutoff.

    Each mismatch destroys at most k k-mers. Halved for safety margin, since
    indels shift k-mer frames in ways this bound does not model.
    """
    lo = min(len(a), len(b))
    hi = max(len(a), len(b))
    n_kmers = max(1, lo - k + 1)
    mismatches = (1.0 - identity_threshold) * hi
    return max(1, int(0.5 * (n_kmers - k * mismatches)))


def cluster_sequences(
    sequences: Sequence[str],
    identity_threshold: float = 0.70,
    coverage_threshold: float = 0.80,
) -> Dict[str, int]:
    """Single-linkage (connected-component) clustering for leakage control.

    Any two sequences with identity >= `identity_threshold` over >=
    `coverage_threshold` of the shorter sequence end up in the same component.
    Because whole components are assigned to a split, no test sequence can
    exceed the threshold against any training sequence *by construction* --
    unlike greedy centroid clustering, where two mutually-similar sequences can
    join different centroids and leak across the split.
    """
    aligner = _aligner()
    uniq = list(dict.fromkeys(sequences))
    n = len(uniq)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = list(candidate_pairs(uniq, identity_threshold))
    for i, j in tqdm(pairs, desc=f"cluster@{identity_threshold:.0%}"):
        if find(i) == find(j):
            continue
        ident, cov = pairwise_identity(uniq[i], uniq[j], aligner)
        if ident >= identity_threshold and cov >= coverage_threshold:
            union(i, j)

    roots: Dict[int, int] = {}
    assignment: Dict[str, int] = {}
    for i, seq in enumerate(uniq):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        assignment[seq] = roots[r]
    return assignment


# Backwards-compatible alias; the greedy variant is no longer used because it
# does not guarantee transitive closure.
greedy_cluster = cluster_sequences


def max_train_identity(
    query: str,
    train_seqs: Sequence[str],
    aligner: Optional[PairwiseAligner] = None,
    coverage_threshold: float = 0.80,
) -> Tuple[float, float, str]:
    """Return (identity, coverage, nearest train sequence).

    The nearest neighbour is the training sequence with the highest identity
    among those aligning over >= `coverage_threshold` of the shorter sequence,
    i.e. exactly the criterion used to build clusters. This is an independent
    all-vs-all check, deliberately not reusing the cluster assignments.
    """
    aligner = aligner or _aligner()
    best_id, best_cov, best_seq = 0.0, 0.0, ""
    for t in train_seqs:
        # Only the exact length gate is applied here -- no k-mer heuristic --
        # so the audit cannot inherit a blind spot from the clusterer.
        if not length_gate(query, t, 0.5):
            continue
        ident, cov = pairwise_identity(query, t, aligner)
        if cov >= coverage_threshold and ident > best_id:
            best_id, best_cov, best_seq = ident, cov, t
    return best_id, best_cov, best_seq


def assign_locked_split(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    test_fraction: float = 0.18,
    seed: int = 42,
    stratify_cols: Optional[List[str]] = None,
) -> pd.Series:
    """Assign whole clusters to train or locked test.

    Stratifies on the first stratify column of the cluster (mode).
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    if stratify_cols:
        # Build a cluster-level stratum label
        labels = []
        for c in clusters:
            sub = df[df[cluster_col] == c]
            parts = []
            for col in stratify_cols:
                mode = sub[col].mode()
                parts.append(str(mode.iloc[0]) if len(mode) else "NA")
            labels.append("|".join(parts))
        # Group clusters by stratum and sample within each
        from collections import defaultdict

        by_stratum: Dict[str, List] = defaultdict(list)
        for c, lab in zip(clusters, labels):
            by_stratum[lab].append(c)
        test_clusters = set()
        for lab, members in by_stratum.items():
            members = list(members)
            rng.shuffle(members)
            n_test = max(1, int(round(len(members) * test_fraction))) if len(members) >= 4 else (
                1 if len(members) >= 2 and rng.random() < test_fraction else 0
            )
            test_clusters.update(members[:n_test])
    else:
        clusters = list(clusters)
        rng.shuffle(clusters)
        n_test = max(1, int(round(len(clusters) * test_fraction)))
        test_clusters = set(clusters[:n_test])

    return df[cluster_col].map(lambda c: "test" if c in test_clusters else "train")
