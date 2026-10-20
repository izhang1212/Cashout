"""
solver.py (LSM) — Longstaff-Schwartz fit for a multi-leg combo.

Simulates paths (prediction backbone), then does the backward regression: at each
step, regress the utility of the currently-carried value on state features across
paths, and let each path exercise where cashing out (at the MODELED price) beats
the fitted continuation. The stored artifact is the per-step coefficient vector
beta_t -- LSM's analogue of the DP table. Runtime evaluates beta_t against the
ACTUAL Kalshi bid (see policy.py).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config
from data.schemas import Combo, GameState
from prediction.backbone import simulate_paths
from prediction.marginals import combo_fair_prob
from execution.lsm.basis import combo_features


def _modeled_reward(combo: Combo, state: GameState, ref: dict) -> float:
    """Efficient-market cash-out value used to value future exercise in the fit."""
    bid = min(max(combo_fair_prob(combo, state, ref) - config.HALF_SPREAD, 0.0), 1.0)
    return bid - config.kalshi_fee(bid)


@dataclass
class LSMSolution:
    steps: int
    betas: list                       # betas[t] (np.ndarray) for t in 0..T-2; None at terminal

    def continuation(self, step: int, features: list[float]) -> float:
        return float(np.dot(features, self.betas[step]))


def fit_lsm(combo: Combo, timeline: list[GameState], ref: dict,
            wealth: float = config.BANKROLL_W, payout: float = 1.0,
            n_sims: int = config.N_SIMS, seed: int = config.RNG_SEED) -> LSMSolution:
    paths = simulate_paths(combo, timeline, ref, n_sims=n_sims, seed=seed)
    T = len(timeline)

    # value carried per path: dollars realized from the position under the policy
    # so far. Initialize to settlement (hold-to-buzzer).
    value = np.array([payout if combo.resolves_yes(p[-1]) else 0.0 for p in paths])
    betas: list = [None] * T

    for t in range(T - 2, -1, -1):
        X = np.array([combo_features(combo, p[t], ref) for p in paths])   # N x F
        y = np.log(np.maximum(wealth + value, 1e-9))                      # vectorized U (log)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[t] = beta

        cont_hat = X @ beta                                              # est. continuation utility
        R = np.array([_modeled_reward(combo, p[t], ref) for p in paths])
        ex_util = np.log(np.maximum(wealth + R, 1e-9))
        exercise = ex_util >= cont_hat
        value = np.where(exercise, R, value)                            # optimal value from step t on

    return LSMSolution(steps=T, betas=betas)