"""Applicability domain / OOD scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from Bio.Align import PairwiseAligner

from psp.features.clustering import max_train_identity


@dataclass
class OODScores:
    max_train_identity: np.ndarray
    embedding_ood_score: np.ndarray
    physchem_ood_score: np.ndarray
    in_domain: np.ndarray
    prediction_confidence: np.ndarray


class ApplicabilityDomain:
    def __init__(
        self,
        train_sequences: Sequence[str],
        train_embeddings: np.ndarray,
        train_physchem: np.ndarray,
        identity_threshold: float = 0.40,
        emb_percentile: float = 95.0,
        phys_percentile: float = 95.0,
    ):
        self.train_sequences = list(train_sequences)
        self.train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
        self.train_physchem = np.asarray(train_physchem, dtype=np.float32)
        # Normalize physchem
        self.phys_mean = self.train_physchem.mean(axis=0)
        self.phys_std = self.train_physchem.std(axis=0) + 1e-6
        train_phys_z = (self.train_physchem - self.phys_mean) / self.phys_std
        # Pairwise distances within train for thresholds
        # Use distance to nearest OTHER train point (leave-one-ish subsample)
        emb_nn = self._nn_distances(self.train_embeddings)
        phys_nn = self._nn_distances(train_phys_z)
        self.emb_thresh = float(np.percentile(emb_nn, emb_percentile))
        self.phys_thresh = float(np.percentile(phys_nn, phys_percentile))
        self.identity_threshold = identity_threshold

    @staticmethod
    def _nn_distances(X: np.ndarray) -> np.ndarray:
        # For large N this is O(N^2); subsample if needed
        n = len(X)
        if n > 800:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=800, replace=False)
            X = X[idx]
            n = len(X)
        # squared euclidean via (x^2 - 2x·y + y^2)
        x2 = np.sum(X * X, axis=1, keepdims=True)
        d2 = x2 + x2.T - 2 * X @ X.T
        np.fill_diagonal(d2, np.inf)
        return np.sqrt(np.clip(d2.min(axis=1), 0, None))

    def score(
        self,
        sequences: Sequence[str],
        embeddings: np.ndarray,
        physchem: np.ndarray,
        identity: bool = True,
        identity_train_cap: int = 2000,
    ) -> OODScores:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        physchem = np.asarray(physchem, dtype=np.float32)
        # identity (optional / capped — full pairwise vs 6k train is very slow)
        max_id = np.zeros(len(sequences), dtype=np.float32)
        if identity:
            train_seqs = self.train_sequences
            if len(train_seqs) > identity_train_cap:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(train_seqs), identity_train_cap, replace=False)
                train_seqs = [train_seqs[i] for i in idx]
            for i, s in enumerate(sequences):
                max_id[i], _, _ = max_train_identity(s, train_seqs)
        else:
            # embedding cosine proxy when identity skipped (train-time eval)
            train_emb = self.train_embeddings
            if len(train_emb) > 1500:
                rng = np.random.default_rng(0)
                train_emb = train_emb[rng.choice(len(train_emb), 1500, replace=False)]
            # cosine similarity -> pseudo-identity in [0,1]
            a = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            b = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
            max_id = np.clip((a @ b.T).max(axis=1), 0, 1).astype(np.float32)
        # embedding NN distance
        # distance to nearest train embedding
        # (subsample train if huge)
        train_emb = self.train_embeddings
        if len(train_emb) > 1500:
            rng = np.random.default_rng(0)
            train_emb = train_emb[rng.choice(len(train_emb), 1500, replace=False)]
        x2 = np.sum(embeddings ** 2, axis=1, keepdims=True)
        y2 = np.sum(train_emb ** 2, axis=1, keepdims=True).T
        d2 = x2 + y2 - 2 * embeddings @ train_emb.T
        emb_dist = np.sqrt(np.clip(d2.min(axis=1), 0, None))
        emb_ood = emb_dist / max(self.emb_thresh, 1e-6)

        phys_z = (physchem - self.phys_mean) / self.phys_std
        px2 = np.sum(phys_z ** 2, axis=1, keepdims=True)
        train_z = (self.train_physchem - self.phys_mean) / self.phys_std
        if len(train_z) > 1500:
            rng = np.random.default_rng(0)
            train_z = train_z[rng.choice(len(train_z), 1500, replace=False)]
        py2 = np.sum(train_z ** 2, axis=1, keepdims=True).T
        pd2 = px2 + py2 - 2 * phys_z @ train_z.T
        phys_dist = np.sqrt(np.clip(pd2.min(axis=1), 0, None))
        phys_ood = phys_dist / max(self.phys_thresh, 1e-6)

        in_domain = (
            (max_id >= self.identity_threshold * 0.5)  # soft
            & (emb_ood <= 1.5)
            & (phys_ood <= 1.5)
        )
        # confidence: high when in-domain and high identity
        conf = np.clip(
            0.4 * max_id + 0.3 * (1.0 / (1.0 + emb_ood)) + 0.3 * (1.0 / (1.0 + phys_ood)),
            0,
            1,
        )
        return OODScores(
            max_train_identity=max_id,
            embedding_ood_score=emb_ood.astype(np.float32),
            physchem_ood_score=phys_ood.astype(np.float32),
            in_domain=in_domain.astype(bool),
            prediction_confidence=conf.astype(np.float32),
        )
