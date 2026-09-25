"""Censored regression models for log-HC50."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


@dataclass
class CensoredPrediction:
    mu: np.ndarray
    sigma: np.ndarray

    @property
    def point(self) -> np.ndarray:
        return self.mu

    def p_gt(self, threshold_log: float) -> np.ndarray:
        return 1.0 - stats.norm.cdf((threshold_log - self.mu) / np.clip(self.sigma, 1e-6, None))


class TobitGaussian(BaseEstimator, RegressorMixin):
    """Linear Tobit / censored Gaussian in feature space (log-HC50).

    Fits μ = X β by maximising the censored Gaussian likelihood with a shared σ.
    Right-censored observations contribute -log P(Y > c).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 200):
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.log_sigma_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        censor_type: np.ndarray,
        censor_lower_log: np.ndarray,
        censor_upper_log: Optional[np.ndarray] = None,
    ):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        # init with ridge on exact observations
        exact = np.asarray(censor_type) == "exact"
        if exact.sum() >= 5:
            ridge = Ridge(alpha=self.alpha).fit(X[exact], y_log[exact])
            beta0 = np.concatenate([[ridge.intercept_], ridge.coef_])
            resid = y_log[exact] - ridge.predict(X[exact])
            log_sigma0 = float(np.log(np.std(resid) + 1e-3))
        else:
            beta0 = np.zeros(d + 1)
            log_sigma0 = 0.0

        def pack(beta, log_sigma):
            return np.concatenate([beta, [log_sigma]])

        def unpack(theta):
            return theta[:-1], theta[-1]

        y_log = np.asarray(y_log, dtype=float)
        censor_type = np.asarray(censor_type)
        censor_lower_log = np.asarray(censor_lower_log, dtype=float)
        if censor_upper_log is None:
            censor_upper_log = np.full(n, np.nan)
        else:
            censor_upper_log = np.asarray(censor_upper_log, dtype=float)

        is_exact = censor_type == "exact"
        is_right = censor_type == "right"
        is_left = censor_type == "left"
        is_interval = censor_type == "interval"

        def nll(theta):
            beta, log_sigma = unpack(theta)
            mu = beta[0] + X @ beta[1:]
            sigma = float(np.exp(np.clip(log_sigma, -5, 5)))
            total = 0.0
            if is_exact.any():
                m = mu[is_exact]
                y = y_log[is_exact]
                ok = np.isfinite(y)
                z = (y[ok] - m[ok]) / sigma
                total += float(np.sum(0.5 * z * z + log_sigma))
            if is_right.any():
                m = mu[is_right]
                c = censor_lower_log[is_right]
                ok = np.isfinite(c)
                z = (c[ok] - m[ok]) / sigma
                surv = np.clip(1.0 - stats.norm.cdf(z), 1e-12, 1.0)
                total += float(-np.sum(np.log(surv)))
            if is_left.any():
                m = mu[is_left]
                c = censor_upper_log[is_left]
                ok = np.isfinite(c)
                z = (c[ok] - m[ok]) / sigma
                cdf = np.clip(stats.norm.cdf(z), 1e-12, 1.0)
                total += float(-np.sum(np.log(cdf)))
            if is_interval.any():
                m = mu[is_interval]
                c1 = censor_lower_log[is_interval]
                c2 = censor_upper_log[is_interval]
                ok = np.isfinite(c1) & np.isfinite(c2)
                z1 = (c1[ok] - m[ok]) / sigma
                z2 = (c2[ok] - m[ok]) / sigma
                p = np.clip(stats.norm.cdf(z2) - stats.norm.cdf(z1), 1e-12, 1.0)
                total += float(-np.sum(np.log(p)))
            total += 0.5 * self.alpha * float(np.sum(beta[1:] ** 2))
            return total / n

        theta0 = pack(beta0, log_sigma0)
        res = optimize.minimize(
            nll, theta0, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        beta, log_sigma = unpack(res.x)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.log_sigma_ = float(log_sigma)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def predict_dist(self, X: np.ndarray) -> CensoredPrediction:
        mu = self.predict(X)
        sigma = np.full(len(mu), np.exp(self.log_sigma_))
        return CensoredPrediction(mu=mu, sigma=sigma)


class ConstantCensored(BaseEstimator, RegressorMixin):
    """Baseline: global mean of exact observations + shared sigma."""

    def fit(self, X, y_log, censor_type, censor_lower_log, censor_upper_log=None):
        exact = np.asarray(censor_type) == "exact"
        vals = np.asarray(y_log)[exact]
        vals = vals[np.isfinite(vals)]
        self.mu_ = float(np.mean(vals)) if len(vals) else 0.0
        self.sigma_ = float(np.std(vals) + 1e-3) if len(vals) else 1.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)

    def predict_dist(self, X):
        mu = self.predict(X)
        return CensoredPrediction(mu=mu, sigma=np.full(len(mu), self.sigma_))
