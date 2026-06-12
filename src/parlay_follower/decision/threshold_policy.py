"""Parametric threshold baseline: the floor every fancier method must beat.

SELL when  executable_bid >= alpha * fair_value  AND  tau < beta_minutes.
Tune (alpha, beta) by grid search over backtest P&L / Sharpe.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class ThresholdPolicy:
    alpha: float = 0.92
    beta_minutes: float = 48.0   # 48 => time gate inactive

    def should_sell(self, executable_bid: float, fair_value: float, tau_min: float) -> bool:
        if fair_value <= 0:
            return executable_bid > 0          # salvage anything on a dead combo
        return executable_bid >= self.alpha * fair_value and tau_min < self.beta_minutes


def grid_search(replay_fn, alphas=None, betas=None) -> tuple["ThresholdPolicy", float]:
    """replay_fn(policy) -> mean P&L (or Sharpe) over the backtest set."""
    alphas = alphas if alphas is not None else np.arange(0.80, 1.01, 0.02)
    betas = betas if betas is not None else [6, 12, 24, 36, 48]
    best, best_score = ThresholdPolicy(), -np.inf
    for a, bm in product(alphas, betas):
        pol = ThresholdPolicy(alpha=float(a), beta_minutes=float(bm))
        score = replay_fn(pol)
        if score > best_score:
            best, best_score = pol, score
    return best, best_score
