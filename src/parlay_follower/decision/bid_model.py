"""Empirical bid model: how far below fair value does Kalshi's combo bid sit?

    M(t) = F_mm(t) * (1 - h(tau, p, k)) + noise

where F_mm is a market-maker-style fair value (market-implied legs + crude
correlation), and the haircut h depends on time remaining tau, combo
probability p, and number of live legs k. Parametric form:

    h = a + b * (1 - p) * exp(-c * tau_frac) * sqrt(k)

so the haircut grows as the game runs down (tau_frac -> 0), as the combo gets
less likely, and with more live legs -- matching the observed reality that
mid-game exits on partially-hit combos "usually lose money."

THE PARAMETERS SHIPPED IN settings.yaml ARE PLACEHOLDERS. They MUST be refit
from logged real bids (market_data/bid_logger.py + fit() below) before the
exercise boundary is trusted. This calibration is where most of the project's
empirical value lives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGULATION_MIN = 48.0


@dataclass
class HaircutParams:
    a: float = 0.03
    b: float = 0.25
    c: float = 3.0


class BidModel:
    def __init__(self, params: HaircutParams | None = None):
        self.params = params or HaircutParams()
        self.residual_std: float = 0.0   # set by fit(); >0 => add bid as DP state

    def haircut(self, tau_min: float, p_combo: float, k_live: int) -> float:
        tau_frac = np.clip(tau_min / REGULATION_MIN, 0.0, 1.0)
        h = (self.params.a
             + self.params.b * (1.0 - p_combo) * np.exp(-self.params.c * tau_frac)
             * np.sqrt(max(k_live, 1)))
        return float(np.clip(h, 0.0, 0.95))

    def bid(self, fair_value_mm: float, tau_min: float, p_combo: float, k_live: int) -> float:
        return float(np.clip(fair_value_mm * (1.0 - self.haircut(tau_min, p_combo, k_live)), 0.0, 1.0))

    # ---------- calibration from logged data ----------
    def fit(self, tau_min: np.ndarray, p_combo: np.ndarray, k_live: np.ndarray,
            fv_mm: np.ndarray, observed_bid: np.ndarray) -> "BidModel":
        """Least-squares fit of (a, b, c) to logged (state, bid) observations."""
        from scipy.optimize import least_squares

        tau_min, p_combo = np.asarray(tau_min, float), np.asarray(p_combo, float)
        k_live, fv_mm = np.asarray(k_live, float), np.asarray(fv_mm, float)
        observed_bid = np.asarray(observed_bid, float)

        mask = fv_mm > 1e-6
        h_obs = 1.0 - observed_bid[mask] / fv_mm[mask]
        tau_frac = np.clip(tau_min[mask] / REGULATION_MIN, 0.0, 1.0)
        oneminus_p, sqrt_k = 1.0 - p_combo[mask], np.sqrt(np.maximum(k_live[mask], 1))

        def resid(theta):
            a, b, c = theta
            return (a + b * oneminus_p * np.exp(-c * tau_frac) * sqrt_k) - h_obs

        sol = least_squares(resid, x0=[self.params.a, self.params.b, self.params.c],
                            bounds=([0.0, 0.0, 0.1], [0.5, 2.0, 20.0]))
        self.params = HaircutParams(*sol.x)
        self.residual_std = float(np.std(sol.fun))
        return self
