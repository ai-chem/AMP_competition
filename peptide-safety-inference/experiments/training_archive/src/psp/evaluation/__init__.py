"""Evaluation metrics for exact and censored HC50 predictions."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return {"n": float(len(yt))}
    pearson = float(stats.pearsonr(yt, yp)[0]) if len(yt) > 2 else float("nan")
    spearman = float(stats.spearmanr(yt, yp)[0]) if len(yt) > 2 else float("nan")
    return {
        "n": float(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    yt, yp = y_true[mask].astype(int), y_prob[mask]
    if len(yt) < 3 or len(np.unique(yt)) < 2:
        return {"n": float(len(yt))}
    yhat = (yp >= 0.5).astype(int)
    out = {
        "n": float(len(yt)),
        "roc_auc": float(roc_auc_score(yt, yp)),
        "pr_auc": float(average_precision_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yhat)),
        "brier": float(brier_score_loss(yt, yp)),
    }
    return out


def censored_gaussian_nll(
    y_log: np.ndarray,
    censor_type: np.ndarray,
    censor_lower_log: np.ndarray,
    censor_upper_log: np.ndarray,
    mu: np.ndarray,
    log_sigma: np.ndarray,
) -> float:
    """Mean NLL under log-normal / Gaussian-in-log-space with censoring."""
    sigma = np.exp(np.clip(log_sigma, -5, 5))
    nll = np.zeros(len(mu))
    for i in range(len(mu)):
        m, s = mu[i], sigma[i]
        ct = censor_type[i]
        if ct == "exact" and np.isfinite(y_log[i]):
            z = (y_log[i] - m) / s
            nll[i] = 0.5 * z * z + np.log(s) + 0.5 * np.log(2 * np.pi)
        elif ct == "right" and np.isfinite(censor_lower_log[i]):
            # P(Y > c) = 1 - Φ((c-μ)/σ)
            z = (censor_lower_log[i] - m) / s
            surv = 1.0 - stats.norm.cdf(z)
            nll[i] = -np.log(max(surv, 1e-12))
        elif ct == "left" and np.isfinite(censor_upper_log[i]):
            z = (censor_upper_log[i] - m) / s
            cdf = stats.norm.cdf(z)
            nll[i] = -np.log(max(cdf, 1e-12))
        elif ct == "interval" and np.isfinite(censor_lower_log[i]) and np.isfinite(
            censor_upper_log[i]
        ):
            z1 = (censor_lower_log[i] - m) / s
            z2 = (censor_upper_log[i] - m) / s
            p = max(stats.norm.cdf(z2) - stats.norm.cdf(z1), 1e-12)
            nll[i] = -np.log(p)
        else:
            nll[i] = np.nan
    return float(np.nanmean(nll))


def concordance_index_censored(
    y_log: np.ndarray,
    censor_type: np.ndarray,
    censor_lower_log: np.ndarray,
    mu: np.ndarray,
) -> float:
    """Simple pairwise concordance treating right-censored as lower bounds."""
    # Build comparable point estimates for ranking: exact → y, right → lower+ε
    scores = []
    events = []
    for i in range(len(mu)):
        if censor_type[i] == "exact" and np.isfinite(y_log[i]):
            scores.append(y_log[i])
            events.append(1)
        elif censor_type[i] == "right" and np.isfinite(censor_lower_log[i]):
            scores.append(censor_lower_log[i])
            events.append(0)
        else:
            continue
    if len(scores) < 5:
        return float("nan")
    try:
        from lifelines.utils import concordance_index

        # lifelines: event observed=True means not censored
        return float(
            concordance_index(scores, mu[: len(scores)] if False else mu, events)
        )
    except Exception:
        # Fallback: pairwise among exact only
        exact_idx = [i for i, ct in enumerate(censor_type) if ct == "exact" and np.isfinite(y_log[i])]
        if len(exact_idx) < 5:
            return float("nan")
        conc = ties = 0
        for a in range(len(exact_idx)):
            for b in range(a + 1, len(exact_idx)):
                i, j = exact_idx[a], exact_idx[b]
                if y_log[i] == y_log[j]:
                    continue
                truth = np.sign(y_log[i] - y_log[j])
                pred = np.sign(mu[i] - mu[j])
                if pred == 0:
                    ties += 1
                elif pred == truth:
                    conc += 1
        total = conc + ties + (len(exact_idx) * (len(exact_idx) - 1) // 2 - conc - ties)
        # recompute properly
        total = 0
        conc = 0
        for a in range(len(exact_idx)):
            for b in range(a + 1, len(exact_idx)):
                i, j = exact_idx[a], exact_idx[b]
                if y_log[i] == y_log[j]:
                    continue
                total += 1
                if np.sign(mu[i] - mu[j]) == np.sign(y_log[i] - y_log[j]):
                    conc += 1
        return float(conc / total) if total else float("nan")


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    yt, yp = y_true[mask].astype(float), y_prob[mask]
    if len(yt) < n_bins:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (yp >= bins[i]) & (yp < bins[i + 1] if i < n_bins - 1 else yp <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(yp)) * abs(yp[m].mean() - yt[m].mean())
    return float(ece)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    stat_fn,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
):
    """Bootstrap at the cluster level. values aligned with clusters."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    stats_ = []
    for _ in range(n_boot):
        sample_clusters = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(clusters, sample_clusters)
        # handle multiplicity: take all rows belonging to sampled clusters
        if mask.sum() < 3:
            continue
        stats_.append(stat_fn(values[mask]))
    if not stats_:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(stats_, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))
