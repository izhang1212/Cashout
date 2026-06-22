"""MLB game context aggregator: produces per-leg probabilities with full situational awareness.

Wraps the per-leg probability dispatch for MLB games. Each tick:
  1. Moneyline: MLBWinModel.win_prob_from_state() conditions on inning/outs/runners/score.
  2. Totals: Bayesian blend of current-game run rate with team season rates.
  3. Player batting props: Bayesian blend of game rate with season baseline.
  4. Pitcher props: innings-remaining projection.

All computation is wrapped in try/except so bugs never crash a live session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..game_feed.game_state import Leg, LegStatus
from .game_state import MLBGameState
from .player_model import (
    batter_hits_over_prob, batter_home_run_prob,
    batter_rbi_over_prob, batter_total_bases_over_prob,
    pitcher_strikeouts_over_prob,
)
from .stats import MLBStatsCache
from .win_model import MLBWinModel, _LEAGUE_AVG_RUNS_PER_HALF_INN


@dataclass
class MLBContextualProbs:
    per_leg: dict[str, float]
    notes: list[str] = field(default_factory=list)

    # Stub for shared-code compatibility (NBA has MomentumSignal; provide a neutral one)
    @property
    def momentum(self):
        from types import SimpleNamespace
        return SimpleNamespace(sell_urgency=0.0)


class MLBGameContext:
    """Computes context-enriched per-leg probabilities for MLB games."""

    def __init__(self, win_model: MLBWinModel, stats_cache: MLBStatsCache):
        self.model = win_model
        self.stats = stats_cache

    def compute(self, legs: list[Leg], gs: MLBGameState) -> MLBContextualProbs:
        try:
            return self._compute(legs, gs)
        except Exception as exc:
            fallback = {l.leg_id: self._fallback(l, gs) for l in legs}
            return MLBContextualProbs(per_leg=fallback,
                                      notes=[f"context error (fallback): {exc}"])

    def _compute(self, legs: list[Leg], gs: MLBGameState) -> MLBContextualProbs:
        notes: list[str] = []
        per_leg: dict[str, float] = {}

        for leg in legs:
            if leg.status is LegStatus.COMPLETED:
                per_leg[leg.leg_id] = 1.0
            elif leg.status is LegStatus.FAILED:
                per_leg[leg.leg_id] = 0.0
            else:
                per_leg[leg.leg_id] = self._live_prob(leg, gs, notes)

        return MLBContextualProbs(per_leg=per_leg, notes=notes)

    def _live_prob(self, leg: Leg, gs: MLBGameState, notes: list[str]) -> float:
        k = leg.kind
        p = leg.params

        if k == "moneyline":
            return self.model.win_prob_from_state(gs)

        if k in ("total_over", "total_under"):
            return self._total(leg, gs)

        if k == "hits_over":
            return batter_hits_over_prob(p.get("player", ""), float(p["line"]), gs, self.stats)

        if k == "home_runs":
            return batter_home_run_prob(p.get("player", ""), gs, self.stats)

        if k == "total_bases_over":
            return batter_total_bases_over_prob(p.get("player", ""), float(p["line"]), gs, self.stats)

        if k == "rbi_over":
            return batter_rbi_over_prob(p.get("player", ""), float(p["line"]), gs, self.stats)

        if k == "strikeouts_over":
            return pitcher_strikeouts_over_prob(p.get("player", ""), float(p["line"]), gs, self.stats)

        return 0.5

    def _total(self, leg: Leg, gs: MLBGameState) -> float:
        """Bayesian pace-aware totals probability."""
        current_total = gs.home_score + gs.away_score
        half_innings_played = max(1, (gs.inning - 1) * 2 + (1 if gs.half == "bottom" else 0))
        game_rate_per_hi = current_total / half_innings_played   # runs/half-inning so far

        # Team season rates
        h_team = self.stats.team(gs.home_team_id)
        a_team = self.stats.team(gs.away_team_id)
        home_rate = h_team.runs_per_inning if h_team else _LEAGUE_AVG_RUNS_PER_HALF_INN
        away_rate = a_team.runs_per_inning if a_team else _LEAGUE_AVG_RUNS_PER_HALF_INN
        season_rate = (home_rate + away_rate) / 2.0

        # Bayesian blend: trust game rate more as game progresses
        blend = min(half_innings_played / (half_innings_played + 4.0), 0.75)
        blended_rate = blend * game_rate_per_hi + (1.0 - blend) * season_rate

        # Expected remaining half-innings
        half_innings_remaining = gs.outs_remaining / 3.0
        projected_total = current_total + blended_rate * half_innings_remaining

        # Variance: Poisson-ish, sigma ~ sqrt(lambda * half_innings)
        sigma = max(np.sqrt(blended_rate * half_innings_remaining) * 1.2, 0.5)

        line = float(leg.params["line"])
        p_over = float(1.0 - norm.cdf(line, loc=projected_total, scale=sigma))
        if leg.kind == "total_over":
            return float(np.clip(p_over, 0.01, 0.99))
        return float(np.clip(1.0 - p_over, 0.01, 0.99))

    def _fallback(self, leg: Leg, gs: MLBGameState) -> float:
        if leg.status is LegStatus.COMPLETED:
            return 1.0
        if leg.status is LegStatus.FAILED:
            return 0.0
        if leg.kind == "moneyline":
            return self.model.win_prob(gs.score_diff, gs.tau_minutes)
        return 0.5
