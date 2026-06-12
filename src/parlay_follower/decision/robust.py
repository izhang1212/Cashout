"""Robust (model-risk-aware) stopping: the humility layer.

All probabilities are estimates, so the stopping rule must know it might be
wrong. Implementation: run the DP under an ENSEMBLE of models (bootstrap-
resampled calibrations; Stern vs. GBM variants) and require holding to survive
the pessimistic member:

    V = max( bid,  min over ensemble of E[V] )   =>   HOLD only if ALL members say hold.

This biases toward earlier, safer exits -- the correct direction to be wrong in
with real money.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..probability.stern import SternModel
from .bid_model import BidModel
from .exact_dp import DPResult, solve


@dataclass
class RobustDecision:
    sell: bool
    votes_sell: int
    n_members: int
    members: list[DPResult]


def make_ensemble(base_sigma: float, pregame_spread: float, n_members: int = 5,
                  sigma_jitter: float = 0.15, mu_jitter_pts: float = 1.0) -> list[SternModel]:
    """Parameter-perturbation ensemble. Replace with bootstrap-refit models once
    historical training data is wired in -- the interface stays the same."""
    import numpy as np
    rng = np.random.default_rng(7)
    members = [SternModel(sigma_per_min=base_sigma, pregame_spread=pregame_spread)]
    for _ in range(n_members - 1):
        s = base_sigma * float(1 + rng.uniform(-sigma_jitter, sigma_jitter))
        spr = pregame_spread + float(rng.uniform(-mu_jitter_pts, mu_jitter_pts))
        members.append(SternModel(sigma_per_min=s, pregame_spread=spr))
    return members


def robust_solve(models: list[SternModel], bid_model: BidModel, *,
                 tau_start_min: float, moneyline_side: str, q_other, k_live: int,
                 dt_min: float = 0.5, risk_aversion: float = 0.0) -> list[DPResult]:
    return [
        solve(m, bid_model, tau_start_min=tau_start_min, moneyline_side=moneyline_side,
              q_other=q_other, k_live=k_live, dt_min=dt_min, risk_aversion=risk_aversion)
        for m in models
    ]


def robust_lookup(results: list[DPResult], tau_min: float, score_diff: float) -> RobustDecision:
    votes = [r.lookup(tau_min, score_diff)[0] for r in results]
    n_sell = sum(votes)
    # HOLD requires unanimity: any member voting SELL => SELL.
    return RobustDecision(sell=n_sell > 0, votes_sell=n_sell,
                          n_members=len(results), members=results)
