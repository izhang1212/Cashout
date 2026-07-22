"""General n-leg joint simulation -> LSMC exercise boundary.

The exact-DP grid (decision/exact_dp.py) is provably optimal but only tractable
for a SINGLE game-outcome leg, whose state is the 1-D score differential. For an
arbitrary n-leg combo the joint state is high-dimensional, so we fall back to the
general method: simulate joint forward paths and fit a Longstaff-Schwartz
continuation-value regression. Same Bellman logic, approximated -- and it handles
any number/mix of legs.

Approach (honest v1):
  * The score-differential path is the shared latent driving the game. It is
    simulated once per path from the Stern diffusion.
  * Each leg's per-step marginal probability is computed from that path (closed
    form for game-outcome legs; supplied model probs for props).
  * Joint resolution at each step is drawn with the Gaussian copula so legs stay
    correlated.
  * The LSMC regression state is a compact, leg-count-agnostic summary:
    (tau, score_diff, n_completed, model_combo_prob, momentum) -- so the SAME
    boundary representation works for n = 1, 2, 3, ... legs.

Momentum feature
----------------
The 5th state dimension is the recent score-diff change rate (pts/min over the
last ~2 simulated steps). This allows the LSMC boundary to distinguish:
  * A team building a lead steadily (neutral momentum, hold per model)
  * A team that just went on a run (high momentum, SELL at the spike is correct
    because momentum tends to mean-revert and the bid has temporarily peaked)

At inference time the follower passes the live MomentumDetector's urgency score
(0..1) into should_sell(), which is scaled to a comparable unit and inserted
into the same position in the state vector.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...shared.copula import make_corr_matrix, nearest_psd
from ..bid_model import BidModel
from .basis import basis

# How many simulation steps back to look for the momentum feature.
_MOMENTUM_LOOKBACK = 4   # ~2 minutes at 96-step / 48-min resolution


@dataclass
class NLegBoundary:
    """Fitted LSMC boundary, queried live by summarizing the current state."""
    coeffs: list[np.ndarray | None]
    times_min: np.ndarray              # tau at each step (descending)
    bid_model: BidModel
    k_legs: int

    def _step_index(self, tau_min: float) -> int:
        return int(np.argmin(np.abs(self.times_min - tau_min)))

    def should_sell(self, tau_min: float, score_diff: float,
                    n_completed: int, combo_prob: float,
                    exit_price: float,
                    momentum_score: float = 0.0) -> tuple[bool, float]:
        """Returns (sell?, estimated continuation value).

        Parameters
        ----------
        momentum_score : float in [0, 1]
            Live run urgency from MomentumDetector.  Inserted as the 5th
            feature so it matches the dimension used during LSMC training.
            Defaults to 0.0 (neutral) for backwards-compatible callers.
        """
        t = self._step_index(tau_min)
        c = self.coeffs[t]
        if c is None:
            return False, exit_price  # no fitted rule here -> default HOLD
        # Scale momentum_score to a pts/min-like unit so it's on the same
        # order of magnitude as the simulated d_momentum feature.
        momentum_scaled = momentum_score * 5.0
        state = np.array([tau_min, score_diff, float(n_completed),
                          combo_prob, momentum_scaled])
        cont = float((basis(state[None, :]) @ c)[0])
        return exit_price >= cont, cont


def simulate_joint(
    stern,
    leg_marginal_fns: list[Callable[[float, float], float]],
    rho: float,
    tau_start_min: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate joint leg resolution along score-diff paths.

    leg_marginal_fns[i](score_diff, tau) -> P(leg i eventually hits | state).

    Returns:
      diff_paths:   (n_paths, n_steps+1) score differential
      completed:    (n_paths, n_steps+1) count of legs already clinched by step
      final_payoff: (n_paths,) 1 if ALL legs hit at the end else 0
    """
    rng = rng or np.random.default_rng()
    k = len(leg_marginal_fns)
    diff_paths = stern.simulate_paths(0.0, tau_start_min, n_paths, n_steps, rng=rng)
    times = np.linspace(tau_start_min, 0.0, n_steps + 1)

    R = nearest_psd(make_corr_matrix(k, rho)) if k > 1 else np.array([[1.0]])
    L = np.linalg.cholesky(R)

    # One correlated uniform draw per path, reused to decide each leg's final
    # outcome against its terminal marginal (keeps legs correlated).
    Z = rng.standard_normal((n_paths, k)) @ L.T
    from scipy.stats import norm
    U = norm.cdf(Z)  # (n_paths, k) correlated uniforms

    final_marg = np.empty((n_paths, k))
    for i, fn in enumerate(leg_marginal_fns):
        final_marg[:, i] = [fn(d, 1e-6) for d in diff_paths[:, -1]]
    final_hits = U < final_marg
    final_payoff = final_hits.all(axis=1).astype(float)

    # Completed-count proxy: a leg counts as "clinched by step t" once its
    # running marginal crosses ~0.999 (e.g. a total already exceeded, a prop
    # already met). For game-outcome legs this naturally fires near the buzzer.
    completed = np.zeros((n_paths, n_steps + 1))
    for t in range(n_steps + 1):
        cnt = np.zeros(n_paths)
        for i, fn in enumerate(leg_marginal_fns):
            marg_t = np.array([fn(d, max(times[t], 1e-6)) for d in diff_paths[:, t]])
            cnt += (marg_t > 0.999).astype(float)
        completed[:, t] = cnt
    return diff_paths, completed, final_payoff


def build_nleg_boundary(
    stern,
    bid_model: BidModel,
    leg_marginal_fns: list[Callable[[float, float], float]],
    rho: float,
    tau_start_min: float,
    n_paths: int = 20000,
    n_steps: int = 96,
    risk_aversion: float = 0.0,
    rng: np.random.Generator | None = None,
) -> NLegBoundary:
    """Fit the LSMC boundary for an arbitrary n-leg combo.

    State vector for the regression (5 features):
        [tau, score_diff, n_completed, combo_prob, d_momentum]

    d_momentum is the score-diff change rate over the last _MOMENTUM_LOOKBACK
    steps (pts/min), capturing whether the game is swinging. The regression
    can use this to fire the SELL signal earlier during a scoring run.
    """
    rng = rng or np.random.default_rng()
    k = len(leg_marginal_fns)
    times = np.linspace(tau_start_min, 0.0, n_steps + 1)
    diff_paths, completed, final_payoff = simulate_joint(
        stern, leg_marginal_fns, rho, tau_start_min, n_paths, n_steps, rng=rng)

    # Per-path, per-step model combo probability and modeled exit price.
    combo_prob = np.empty((n_paths, n_steps + 1))
    bids = np.empty((n_paths, n_steps + 1))
    for t in range(n_steps + 1):
        tau = max(float(times[t]), 1e-6)
        p = np.ones(n_paths)
        for fn in leg_marginal_fns:
            p *= np.array([fn(d, tau) for d in diff_paths[:, t]])
        combo_prob[:, t] = p
        bids[:, t] = [bid_model.bid(p[j], tau, p[j], k) for j in range(n_paths)]

    # Pre-compute time step width for momentum in pts/min units.
    dt_min = tau_start_min / n_steps if n_steps > 0 else 1.0

    # Backward LSMC over the 5-feature state summary.
    cashflow = final_payoff.copy()
    coeffs: list[np.ndarray | None] = [None] * (n_steps + 1)
    for t in range(n_steps, -1, -1):
        # Momentum: score-diff change rate over the last _MOMENTUM_LOOKBACK steps
        lb = min(_MOMENTUM_LOOKBACK, t)
        if lb > 0:
            d_momentum = (diff_paths[:, t] - diff_paths[:, t - lb]) / (lb * dt_min)
        else:
            d_momentum = np.zeros(n_paths)

        state = np.column_stack([
            np.full(n_paths, times[t]),
            diff_paths[:, t],
            completed[:, t],
            combo_prob[:, t],
            d_momentum,
        ])
        X = basis(state)
        itm = bids[:, t] > 1e-4
        if itm.sum() >= X.shape[1] * 3:
            beta, *_ = np.linalg.lstsq(X[itm], cashflow[itm], rcond=None)
            coeffs[t] = beta
            cont_hat = X @ beta
            exercise = itm & (bids[:, t] >= cont_hat)
            cashflow[exercise] = bids[exercise, t]

    return NLegBoundary(coeffs=coeffs, times_min=times, bid_model=bid_model, k_legs=k)
