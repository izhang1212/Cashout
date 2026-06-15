"""Probability calibration layer.

The standard for every probability this system emits: when the model says 70%,
the event must happen ~70% of the time on held-out seasons. Verified with
reliability curves and Brier score; corrected with isotonic regression fitted
on OUT-OF-SAMPLE predictions and applied on top of the raw model.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


def brier_score(p_pred: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_pred) - np.asarray(outcomes)) ** 2))


def reliability_curve(p_pred: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Returns (bin_mean_pred, bin_empirical_freq, bin_count) for plotting."""
    p_pred, outcomes = np.asarray(p_pred), np.asarray(outcomes)
    bins = np.clip((p_pred * n_bins).astype(int), 0, n_bins - 1)
    mean_pred, emp_freq, counts = [], [], []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        mean_pred.append(p_pred[mask].mean())
        emp_freq.append(outcomes[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(emp_freq), np.array(counts)


class Calibrator:
    """Isotonic recalibration: fit on out-of-sample preds, apply at inference."""

    def __init__(self):
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required: pip install scikit-learn")
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.fitted = False

    def fit(self, p_oos: np.ndarray, outcomes: np.ndarray) -> "Calibrator":
        self._iso.fit(np.asarray(p_oos), np.asarray(outcomes))
        self.fitted = True
        return self

    def apply(self, p_raw: float | np.ndarray):
        if not self.fitted:
            return p_raw  # identity until fitted -- explicit and safe
        return self._iso.predict(np.atleast_1d(np.asarray(p_raw, dtype=float)))
