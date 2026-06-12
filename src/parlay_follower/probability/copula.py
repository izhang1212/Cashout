"""Gaussian copula for joint leg resolution.

Legs are marginally Bernoulli(p_i); joint dependence comes from a Gaussian
copula with correlation matrix R. Leg i hits on a draw Z ~ MVN(0, R) iff
Phi(z_i) < p_i.

Upgrade path (state-conditional correlation): estimate R conditional on game
state buckets (lead x time), since e.g. a star's scoring prop and his team
winning are more correlated in a close game than a blowout. `CorrelationTable`
holds that mapping; it degrades gracefully to a single default rho.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def make_corr_matrix(n_legs: int, rho: float) -> np.ndarray:
    R = np.full((n_legs, n_legs), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    return R


def nearest_psd(R: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so Cholesky always succeeds."""
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-10, None)
    R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def joint_hit_samples(probs: np.ndarray, R: np.ndarray, n_paths: int,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """(n_paths, n_legs) boolean array of joint leg outcomes."""
    rng = rng or np.random.default_rng()
    L = np.linalg.cholesky(nearest_psd(R))
    Z = rng.standard_normal((n_paths, len(probs))) @ L.T
    return norm.cdf(Z) < probs[None, :]


class CorrelationTable:
    """State-conditional pairwise correlation, bucketed by (|lead|, tau)."""

    def __init__(self, default_rho: float = 0.35):
        self.default_rho = default_rho
        self.table: dict[tuple[str, str, str], float] = {}
        # key: (leg_kind_a, leg_kind_b, state_bucket) -> rho, fit from history

    @staticmethod
    def bucket(abs_lead: float, tau_min: float) -> str:
        lead_b = "close" if abs_lead <= 6 else ("mid" if abs_lead <= 14 else "blowout")
        time_b = "late" if tau_min <= 12 else ("mid" if tau_min <= 30 else "early")
        return f"{lead_b}-{time_b}"

    def rho(self, kind_a: str, kind_b: str, abs_lead: float, tau_min: float) -> float:
        key = (min(kind_a, kind_b), max(kind_a, kind_b), self.bucket(abs_lead, tau_min))
        return self.table.get(key, self.default_rho)
