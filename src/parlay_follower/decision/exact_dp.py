"""Exact dynamic programming: the workhorse optimizer.

Because the core state -- (time remaining, score differential, leg-status
combination) -- is low-dimensional and discrete, the Bellman equation is solved
EXACTLY by backward induction. By Bellman's principle of optimality the
resulting policy is provably the best possible GIVEN THE MODEL. Live operation
is a table lookup: the entire exercise boundary is precomputed before tip-off.

    V(T, s)  = terminal payoff (1 if all legs hit else 0)
    V(t, s)  = max( bid(t, s),  E_adj[ V(t+1, s') | s ] )

where E_adj is the (optionally risk-adjusted) expectation and the score-diff
transition kernel comes from the Stern diffusion in closed form.

v1 scope (documented approximation): the diffusion-driven leg is the moneyline
on the followed game; other LIVE legs are folded in as a multiplicative
survival probability q_other(t) assumed conditionally independent of the score
diff given time (their dependence is carried by the Monte Carlo layer's fair
value, which the bid model consumes). Removing this approximation = adding leg
state dimensions to the grid (cheap for totals) or switching to LSMC (props).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..probability.stern import SternModel
from .bid_model import BidModel


@dataclass
class DPResult:
    time_grid_min: np.ndarray      # decision times, descending tau (T..0), shape (nt,)
    diff_grid: np.ndarray          # score-diff bins, shape (nd,)
    value: np.ndarray              # V table, shape (nt, nd)
    exercise: np.ndarray           # boolean boundary mask, True = SELL, shape (nt, nd)
    bid_surface: np.ndarray        # modeled bid at each node, shape (nt, nd)

    def lookup(self, tau_min: float, score_diff: float) -> tuple[bool, float, float]:
        """Live operation: (sell?, continuation value proxy, modeled bid)."""
        it = int(np.argmin(np.abs(self.time_grid_min - tau_min)))
        idx = int(np.argmin(np.abs(self.diff_grid - score_diff)))
        return bool(self.exercise[it, idx]), float(self.value[it, idx]), float(self.bid_surface[it, idx])


def risk_adjusted_expectation(values: np.ndarray, probs: np.ndarray,
                              risk_aversion: float) -> float:
    """E[V] if risk_aversion == 0, else exponential-utility certainty equivalent:

        CE = -(1/lambda) * log( E[ exp(-lambda * V) ] )

    For a payoff that can hit $0 on one missed free throw, lambda > 0 shifts
    the boundary toward earlier exits -- the optimal policy for the user's
    actual preferences, not a worse one.
    """
    ev = float(probs @ values)
    if risk_aversion <= 0:
        return ev
    lam = risk_aversion
    return float(-np.log(probs @ np.exp(-lam * values)) / lam)


def solve(
    stern: SternModel,
    bid_model: BidModel,
    tau_start_min: float,
    moneyline_side: str,                      # "home" | "away"
    q_other: Callable[[float], float],        # survival prob of all other LIVE legs at tau
    k_live: int,                              # number of live legs (for the haircut)
    dt_min: float = 0.5,
    diff_range: int = 45,
    risk_aversion: float = 0.0,
) -> DPResult:
    """Backward sweep from the final buzzer to now. Returns the full boundary."""
    nt = int(np.ceil(tau_start_min / dt_min)) + 1
    time_grid = np.linspace(tau_start_min, 0.0, nt)          # descending tau
    diff_grid = np.arange(-diff_range, diff_range + 1, dtype=float)
    nd = len(diff_grid)

    P = stern.transition_matrix(diff_grid, dt_min)            # (nd, nd)

    sign = 1.0 if moneyline_side == "home" else -1.0
    V = np.zeros((nt, nd))
    exercise = np.zeros((nt, nd), dtype=bool)
    bids = np.zeros((nt, nd))

    # ---- terminal condition (tau = 0) ----
    won = (sign * diff_grid) > 0
    tied = diff_grid == 0
    V[-1] = np.where(won, q_other(0.0), np.where(tied, 0.5 * q_other(0.0), 0.0))
    bids[-1] = V[-1]                                          # at the buzzer, value is value
    exercise[-1] = True

    # ---- backward sweep ----
    for it in range(nt - 2, -1, -1):
        tau = float(time_grid[it])
        cont = np.array([
            risk_adjusted_expectation(V[it + 1], P[i], risk_aversion)
            for i in range(nd)
        ])
        # Modeled executable bid at each node: F_mm proxied by the node's
        # model probability (combo prob = P(ml wins from here) * q_other).
        p_ml = np.array([stern.win_prob(d, tau, side=moneyline_side) for d in diff_grid])
        p_combo = p_ml * q_other(tau)
        node_bids = np.array([
            bid_model.bid(p_combo[i], tau, p_combo[i], k_live) for i in range(nd)
        ])
        V[it] = np.maximum(node_bids, cont)
        exercise[it] = node_bids >= cont
        bids[it] = node_bids

    return DPResult(time_grid_min=time_grid, diff_grid=diff_grid,
                    value=V, exercise=exercise, bid_surface=bids)


def boundary_heatmap(result: DPResult, out_path: str) -> None:
    """The headline deliverable: exercise region in (time remaining, lead) space --
    the basketball analogue of an American option's early-exercise region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        result.exercise.T.astype(int),
        aspect="auto", origin="lower", cmap="RdYlGn_r",
        extent=[result.time_grid_min[0], result.time_grid_min[-1],
                result.diff_grid[0], result.diff_grid[-1]],
    )
    ax.set_xlabel("Minutes remaining")
    ax.set_ylabel("Score differential (home - away)")
    ax.set_title("Exercise boundary: SELL (red) vs HOLD (green)")
    fig.colorbar(im, ax=ax, label="1 = SELL")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
