"""Calibration helpers for P(HC50 > 128)."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self._model = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        mask = np.isfinite(y_true) & np.isfinite(y_prob)
        yt, yp = y_true[mask].astype(int), y_prob[mask].reshape(-1, 1)
        if len(np.unique(yt)) < 2:
            self._model = None
            return self
        if self.method == "platt":
            self._model = LogisticRegression(max_iter=1000)
            self._model.fit(yp, yt)
        else:
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(yp.ravel(), yt)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self._model is None:
            return y_prob
        if self.method == "platt":
            return self._model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return self._model.transform(y_prob)
