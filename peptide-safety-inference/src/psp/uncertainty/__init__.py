"""Uncertainty estimation helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def conformal_interval(
    residuals: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Absolute residual quantile for split conformal prediction."""
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return float("nan")
    q = min(1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
    return float(np.quantile(np.abs(residuals), q))


def gaussian_interval(mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * sigma, mu + z * sigma


def ensemble_stats(preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """preds: (n_models, n_samples) → mean, std."""
    return preds.mean(axis=0), preds.std(axis=0)
