"""Decision engine: dispatch to the right optimizer for the position's shape.

This is the single entry point the live follower and backtester call. It hides
the method choice behind one `recommend()` call so the rest of the system is
leg-count-agnostic.

Dispatch rule:
  * Exactly one leg, game-outcome (moneyline/total)  -> EXACT DP (provably
    optimal; 1-D score-diff state). With robust_ensemble_size > 1 this becomes
    ROBUST DP: solve under an ensemble and HOLD only if every member agrees.
  * Anything else (2+ legs, or a single prop leg)     -> LSMC n-leg boundary
    (general; handles any number/mix of legs).

All paths return an identical Recommendation, so callers never branch on which
optimizer ran.

Boundary freshness
------------------
The boundary is rebuilt when any of three conditions are true:
  1. A leg changes status (clinched / dead) -- always rebuild.
  2. 5 game-minutes have elapsed since the last build -- keeps q_other and
     the totals pace estimate fresh even if no status change fires.
  3. invalidate_boundary() was called explicitly (e.g. after a live mu update).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm as _norm

from ..data_gathering.nba.game_state import GameState, Leg, LegStatus
from ..models.nba.stern import SternModel
from .pricing.copula import CorrelationTable
from .pricing.monte_carlo import leg_live_prob, price_combo

from .bellman.exact_dp import DPResult, solve
from .bellman.robust import make_ensemble, robust_lookup, robust_solve
from .bid_model import BidModel
from .lsm.lsm import NLegBoundary, build_nleg_boundary

# Rebuild the exercise boundary every N game-minutes to keep q_other and
# pace-based totals fresh (even when no leg status changes).
_REBUILD_INTERVAL_MIN = 2.0


@dataclass
class Recommendation:
    sell: bool
    fair_value: float
    continuation_value: float
    method: str                  # "exact_dp" | "robust_dp" | "lsmc_nleg"
    per_leg_probs: dict[str, float]
    ensemble_votes_sell: int = 0
    ensemble_size: int = 0


GAME_OUTCOME = {"moneyline", "total_over", "total_under"}


def _single_game_outcome(legs: list[Leg]) -> Leg | None:
    live = [l for l in legs if l.status is LegStatus.LIVE]
    if len(live) == 1 and live[0].kind in GAME_OUTCOME:
        return live[0]
    return None


class DecisionEngine:
    """Builds the boundary once (per leg-status change or staleness interval), then cheap lookups."""

    def __init__(self, stern: SternModel, bid_model: BidModel,
                 corr: CorrelationTable, *, mc_paths: int = 20000,
                 dt_min: float = 0.5, risk_aversion: float = 0.0,
                 pregame_spread: float = 0.0, robust_ensemble_size: int = 1, use_exact_dp: bool = True):
        self.stern = stern
        self.bid_model = bid_model
        self.corr = corr
        self.mc_paths = mc_paths
        self.dt_min = dt_min
        self.risk_aversion = risk_aversion
        self.pregame_spread = pregame_spread
        self.robust_ensemble_size = robust_ensemble_size
        self.use_exact_dp = use_exact_dp
        self._dp: DPResult | None = None
        self._dp_ensemble: list[DPResult] | None = None
        self._nleg: NLegBoundary | None = None
        self._method: str = ""
        self._built_for: tuple | None = None
        self._last_build_tau: float = float("inf")   # tau_minutes at last build

    def invalidate_boundary(self) -> None:
        """Force boundary rebuild on the next recommend() call.

        Call this after updating stern.mu from a live market price so the DP
        boundary immediately reflects the new drift estimate.
        """
        self._built_for = None
        self._last_build_tau = float("inf")

    # ---------- marginal-prob closures for the n-leg simulator ----------

    def _leg_marginal_fn(self, leg: Leg, prop_probs: dict[str, float] | None,
                         gs: GameState) -> Callable[[float, float], float]:
        """Return a closure (score_diff, tau_min) -> P(leg hits | simulated state).

        NOTE: tau passed to these closures is in MINUTES.
        """
        stern = self.stern

        if leg.kind == "moneyline":
            side = leg.params.get("side", "home")
            return lambda d, tau, side=side: stern.win_prob(d, tau, side=side)

        if leg.kind in ("total_over", "total_under"):
            line = float(leg.params["line"])
            kind = leg.kind
            # Anchor pace to the live game state at build time.
            current_total = gs.home_score + gs.away_score
            elapsed_min = max(48.0 - gs.tau_minutes, 1e-6)
            current_pace = current_total / elapsed_min   # pts/min
            sig = stern.sigma

            def _total(d: float, tau: float, line: float = line, kind: str = kind,
                       c_total: float = current_total, c_pace: float = current_pace,
                       sigma: float = sig) -> float:
                mean = c_total + c_pace * tau
                std = 1.6 * sigma * np.sqrt(max(tau, 1e-9))
                p_over = float(1.0 - _norm.cdf(line, loc=mean, scale=std))
                return float(np.clip(p_over if kind == "total_over" else 1.0 - p_over,
                                     0.01, 0.99))
            return _total

        if prop_probs and leg.leg_id in prop_probs:
            p = float(prop_probs[leg.leg_id])
            return lambda d, tau, p=p: p

        return lambda d, tau: 0.5

    # ---------- boundary construction ----------

    def _ensure_boundary(self, legs: list[Leg], gs: GameState,
                         prop_probs: dict[str, float] | None) -> None:
        statuses = tuple(l.status for l in legs)
        time_stale = (self._last_build_tau - gs.tau_minutes) >= _REBUILD_INTERVAL_MIN

        if self._built_for == statuses and (self._dp or self._nleg) and not time_stale:
            return

        self._last_build_tau = gs.tau_minutes

        single = _single_game_outcome(legs) if self.use_exact_dp else None
        if single is not None and single.kind == "moneyline":
            # Other legs (if any) fold in as a constant survival probability.
            others = [l for l in legs if l is not single and l.status is LegStatus.LIVE]
            q = 1.0
            for o in others:
                q *= leg_live_prob(o, gs, self.stern, prop_probs)
            ml_side = single.params.get("side", "home")
            k_live = 1 + len(others)

            if self.robust_ensemble_size > 1:
                models = make_ensemble(self.stern.sigma, self.pregame_spread,
                                       n_members=self.robust_ensemble_size)
                self._dp_ensemble = robust_solve(
                    models, self.bid_model, tau_start_min=gs.tau_minutes,
                    moneyline_side=ml_side, q_other=lambda tau, q=q: q,
                    k_live=k_live, dt_min=self.dt_min,
                    risk_aversion=self.risk_aversion)
                self._dp = self._dp_ensemble[0]
                self._method = "robust_dp"
            else:
                self._dp = solve(
                    self.stern, self.bid_model, tau_start_min=gs.tau_minutes,
                    moneyline_side=ml_side, q_other=lambda tau, q=q: q,
                    k_live=k_live, dt_min=self.dt_min,
                    risk_aversion=self.risk_aversion)
                self._dp_ensemble = None
                self._method = "exact_dp"
            self._nleg = None
        else:
            live = [l for l in legs if l.status is LegStatus.LIVE]
            fns = [self._leg_marginal_fn(l, prop_probs, gs) for l in live]
            rho = self.corr.default_rho
            self._nleg = build_nleg_boundary(
                self.stern, self.bid_model, fns, rho,
                tau_start_min=gs.tau_minutes, n_paths=self.mc_paths,
                risk_aversion=self.risk_aversion)
            self._dp, self._method = None, "lsmc_nleg"
            self._dp_ensemble = None

        self._built_for = statuses

    # ---------- the one call the rest of the system uses ----------

    def recommend(self, legs: list[Leg], gs: GameState, exit_price: float,
                  prop_probs: dict[str, float] | None = None,
                  momentum_score: float = 0.0) -> Recommendation:
        """Return a HOLD/SELL recommendation.

        Parameters
        ----------
        momentum_score : float in [0, 1]
            Current run urgency from MomentumDetector.  Forwarded to the LSMC
            boundary so the continuation-value regression can distinguish
            low-momentum states (boundary moves later) from hot-run states
            (boundary moves earlier, easier to SELL).  Defaults to 0.0 so
            existing callers (backtester, tests) are unaffected.
        """
        # Dead combo short-circuit.
        if any(l.status is LegStatus.FAILED for l in legs):
            return Recommendation(True, 0.0, 0.0, "dead_combo",
                                  {l.leg_id: 0.0 for l in legs})

        val = price_combo(legs, gs, self.stern, self.corr,
                          n_paths=self.mc_paths, prop_probs=prop_probs)
        self._ensure_boundary(legs, gs, prop_probs)
        n_completed = sum(l.status is LegStatus.COMPLETED for l in legs)
        votes_sell = ens_size = 0

        if self._method == "robust_dp" and self._dp_ensemble is not None:
            rd = robust_lookup(self._dp_ensemble, gs.tau_minutes, gs.score_diff)
            _, cont, _ = self._dp.lookup(gs.tau_minutes, gs.score_diff)
            sell = rd.sell or (exit_price >= cont)
            votes_sell, ens_size = rd.votes_sell, rd.n_members
        elif self._method == "exact_dp" and self._dp is not None:
            sell, cont, _ = self._dp.lookup(gs.tau_minutes, gs.score_diff)
            sell = sell or (exit_price >= cont)
        elif self._nleg is not None:
            sell, cont = self._nleg.should_sell(
                gs.tau_minutes, gs.score_diff, n_completed,
                val.fair_value, exit_price, momentum_score=momentum_score)
        else:
            sell, cont = False, val.fair_value

        return Recommendation(sell=sell, fair_value=val.fair_value,
                              continuation_value=cont, method=self._method or "none",
                              per_leg_probs=dict(val.per_leg_probs),
                              ensemble_votes_sell=votes_sell, ensemble_size=ens_size)
