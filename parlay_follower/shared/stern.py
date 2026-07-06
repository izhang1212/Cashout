"""Stern (1994) Brownian-motion win-probability model.

Score differential D(t) (home minus away) is modeled as Brownian motion with
drift: dD = mu dt + sigma dW, with t in minutes. Then with tau minutes left
and current lead d:

    P(home wins) = Phi( (d + mu * tau) / (sigma * sqrt(tau)) )

This is the basketball analogue of Black-Scholes: closed-form, transparent,
and -- crucially for the DP -- it gives closed-form Gaussian TRANSITION
probabilities for the score differential, which is what the backward sweep
consumes.

Calibration:
  * mu (per minute) from the pregame spread: mu = -spread / 48 under the
    convention that a home line of -6.5 means home favored by 6.5.
  * sigma_per_min from historical score-diff increments (~1.7 => ~11.8 pts
    of full-game standard deviation). Re-estimate from data; do not trust
    the default with money on the line.

Known mis-calibration zones (endgame foul-fests, garbage time, last-possession
value) are corrected by the isotonic re-calibration layer, not here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class SternModel:
    def __init__(self, sigma_per_min: float = 1.7, pregame_spread: float = 0.0):
        self.sigma = sigma_per_min
        self.mu = -pregame_spread / 48.0  # home line -6.5 -> +drift for home

    # ---------- win probability ----------
    def home_win_prob(self, score_diff: float, tau_min: float) -> float:
        if tau_min <= 0:
            return 1.0 if score_diff > 0 else (0.5 if score_diff == 0 else 0.0)
        z = (score_diff + self.mu * tau_min) / (self.sigma * np.sqrt(tau_min))
        return float(norm.cdf(z))

    def win_prob(self, score_diff: float, tau_min: float, side: str = "home") -> float:
        p = self.home_win_prob(score_diff, tau_min)
        return p if side == "home" else 1.0 - p

    # ---------- terminal total / margin distributions ----------
    def terminal_diff_distribution(self, score_diff: float, tau_min: float) -> tuple[float, float]:
        """Final margin ~ Normal(mean, std) from the current state."""
        return score_diff + self.mu * tau_min, self.sigma * np.sqrt(max(tau_min, 1e-9))

    # ---------- transition kernel for the DP grid ----------
    def transition_matrix(self, diff_grid: np.ndarray, dt_min: float) -> np.ndarray:
        """P[i, j] = P(D(t+dt) in bin j | D(t) = diff_grid[i]).

        Increment over dt is N(mu*dt, sigma^2*dt), discretized onto the grid by
        integrating the Gaussian density over each bin (mass beyond the edges is
        absorbed into the boundary bins, so rows sum to exactly 1).
        """
        n = len(diff_grid)
        step = float(diff_grid[1] - diff_grid[0])
        edges = np.concatenate(([-np.inf], diff_grid[:-1] + step / 2.0, [np.inf]))
        m, s = self.mu * dt_min, self.sigma * np.sqrt(dt_min)

        P = np.empty((n, n))
        for i, d in enumerate(diff_grid):
            cdf = norm.cdf(edges, loc=d + m, scale=s)
            P[i] = np.diff(cdf)
        return P

    # ---------- path simulation (Monte Carlo layer) ----------
    def simulate_paths(self, score_diff: float, tau_min: float,
                       n_paths: int, n_steps: int,
                       rng: np.random.Generator | None = None) -> np.ndarray:
        """(n_paths, n_steps+1) score-differential paths via exact GBM-style update."""
        rng = rng or np.random.default_rng()
        dt = tau_min / n_steps
        increments = rng.normal(self.mu * dt, self.sigma * np.sqrt(dt), size=(n_paths, n_steps))
        paths = np.concatenate(
            [np.full((n_paths, 1), float(score_diff)), increments], axis=1
        ).cumsum(axis=1)
        return paths
