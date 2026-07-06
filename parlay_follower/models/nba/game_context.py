"""Game context aggregator: the single call-site that produces context-aware per-leg probs.

This replaces the manual `leg_live_prob -> shrink` loop in the follower with a
richer computation that incorporates:

  * Stern diffusion win probability (existing, unchanged)
  * Foul trouble: star in foul trouble -> team's win prob adjusted down
  * Scoring momentum: run in progress -> mean-reversion nudge on model prob
  * Pace-aware totals: team-season pace anchors the projection; Q4 dynamics
    (clock management in close games; garbage-time fouling in blowouts) adjust
    the effective pace for the rest of the game
  * Player prop projection: season pts/min baseline blended with live game pace,
    adjusted for remaining minutes and foul trouble

Output is a ContextualProbs with per_leg dict (leg_id -> float) that the follower
passes to shrink() and then to the decision engine. The existing shrinkage layer
remains unchanged: unproven models are still heavily blended toward the market.

The compute() method is wrapped defensively so any bug in the new models falls
back to the existing leg_live_prob behavior rather than crashing a live session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ...data_gathering.nba.stats import NBAStatsCache
from ...shared.game_feed.game_state import GameState, Leg, LegStatus
from ...shared.stern import SternModel
from .foul_model import FoulTroubleImpact, FoulTroubleModel
from .momentum import MomentumDetector, MomentumSignal
from .player_model import player_pts_over_prob

# League-average points per minute when we have no team-specific data
_LEAGUE_AVG_PTS_PER_MIN = 110.0 / 48.0


@dataclass
class ContextualProbs:
    per_leg: dict[str, float]
    momentum: MomentumSignal
    foul_impact: FoulTroubleImpact
    notes: list[str] = field(default_factory=list)


class GameContext:
    """Computes context-enriched per-leg probabilities from all available signals.

    Instantiated once per session; compute() called every tick.
    """

    def __init__(self, stern: SternModel, stats_cache: NBAStatsCache,
                 momentum: MomentumDetector, foul_model: FoulTroubleModel):
        self.stern = stern
        self.stats = stats_cache
        self.momentum = momentum
        self.foul_model = foul_model

    def compute(self, legs: list[Leg], gs: GameState) -> ContextualProbs:
        """Return context-aware probability for every leg.

        Catches all exceptions and falls back to Stern/neutral defaults so
        a bug in the new models never crashes the live session.
        """
        try:
            return self._compute(legs, gs)
        except Exception as exc:
            # Fallback: use basic Stern probs so the follower keeps running
            fallback = {l.leg_id: self._stern_fallback(l, gs) for l in legs}
            from .momentum import _NEUTRAL as _MOM_NEUTRAL
            from .foul_model import FoulTroubleImpact as _FTI
            return ContextualProbs(
                per_leg=fallback,
                momentum=_MOM_NEUTRAL,
                foul_impact=_FTI(0.0, 0.0),
                notes=[f"context error (fallback): {exc}"],
            )

    def _compute(self, legs: list[Leg], gs: GameState) -> ContextualProbs:
        mom_sig = self.momentum.signal()
        foul_impact = self.foul_model.assess(gs)
        notes: list[str] = []

        if mom_sig.run is not None:
            notes.append(mom_sig.note)
        if foul_impact.troubled_players:
            notes.append("foul trouble: " + ", ".join(foul_impact.troubled_players))

        per_leg: dict[str, float] = {}
        for leg in legs:
            if leg.status is LegStatus.COMPLETED:
                per_leg[leg.leg_id] = 1.0
            elif leg.status is LegStatus.FAILED:
                per_leg[leg.leg_id] = 0.0
            else:
                per_leg[leg.leg_id] = self._live_prob(leg, gs, mom_sig, foul_impact)

        return ContextualProbs(per_leg=per_leg, momentum=mom_sig,
                               foul_impact=foul_impact, notes=notes)

    # ---------- per-leg probability dispatch ----------

    def _live_prob(self, leg: Leg, gs: GameState,
                   mom: MomentumSignal, foul: FoulTroubleImpact) -> float:
        k = leg.kind
        if k == "moneyline":
            return self._moneyline(leg, gs, mom, foul)
        if k in ("total_over", "total_under"):
            return self._total(leg, gs, foul)
        if k == "player_points_over":
            return self._player_pts(leg, gs)
        # Unknown leg kind: return neutral
        return 0.5

    def _moneyline(self, leg: Leg, gs: GameState,
                   mom: MomentumSignal, foul: FoulTroubleImpact) -> float:
        side = leg.params.get("side", "home")

        # Base: Stern diffusion win probability
        base = self.stern.win_prob(gs.score_diff, gs.tau_minutes, side=side)

        # Foul trouble adjustment: adjust from both sides
        # If home stars are in foul trouble -> home win prob drops; away win prob rises
        # We model the NET effect on the specific side being bet on.
        if side == "home":
            base = float(np.clip(base + foul.home_delta - foul.away_delta, 0.01, 0.99))
        else:
            # Betting on away: their foul trouble hurts them; home foul trouble helps them
            base = float(np.clip(base - foul.away_delta + foul.home_delta, 0.01, 0.99))

        # Momentum mean-reversion nudge
        base = self.momentum.win_prob_adjustment(base, side)

        return float(np.clip(base, 0.01, 0.99))

    def _total(self, leg: Leg, gs: GameState, foul: FoulTroubleImpact) -> float:
        """Pace-aware totals probability.

        Improvements over the original constant-pace approximation:
          1. Blend current-game pace with team season pace (Bayesian anchor).
          2. Q4 close-game adjustment: pace slows in the final 12 min when the
             game is within 8 pts (clock management, more deliberate possessions).
          3. Blowout late adjustment: intentional fouling speeds pace slightly.
          4. Foul trouble reduces effective scoring efficiency (fewer quality
             possessions when key players sit).
        """
        elapsed_min = max(48.0 - gs.tau_minutes, 1e-6)
        current_total = gs.home_score + gs.away_score

        # Current pace in this game (pts/min)
        game_pace = current_total / elapsed_min

        # Season-average expected combined pace from both teams
        season_pace = self._combined_season_pace(gs)

        # Bayesian blend: current game pace vs. season baseline
        # At elapsed=0 we use pure season; at elapsed=24 min we trust game pace ~66%
        blend = min(elapsed_min / (elapsed_min + 12.0), 0.80)
        blended_pace = blend * game_pace + (1.0 - blend) * season_pace

        # Q4 pace dynamics
        abs_diff = abs(gs.score_diff)
        tau = gs.tau_minutes
        if tau <= 12.0 and abs_diff <= 8:
            # Close game in the 4th: clock management slows pace ~8%
            blended_pace *= 0.92
        elif tau <= 6.0 and abs_diff >= 10:
            # Blowout: intentional fouling / garbage time speeds pace ~5%
            blended_pace *= 1.05

        # Foul trouble pace discount: key players fouled out -> fewer possessions used well
        foul_discount = (abs(foul.home_delta) + abs(foul.away_delta)) * 0.25
        blended_pace *= (1.0 - foul_discount)

        mean_final = current_total + blended_pace * gs.tau_minutes

        # Variance: Stern sigma governs the score-diff diffusion; for the total we
        # use 1.6σ which accounts for both teams' independent scoring variance.
        # (The 1.6 is empirically calibrated; upgrade: separate variance model.)
        std_final = 1.6 * self.stern.sigma * np.sqrt(max(gs.tau_minutes, 1e-9))

        p_over = float(1.0 - norm.cdf(leg.params["line"], loc=mean_final, scale=std_final))
        if leg.kind == "total_over":
            return float(np.clip(p_over, 0.01, 0.99))
        return float(np.clip(1.0 - p_over, 0.01, 0.99))

    def _player_pts(self, leg: Leg, gs: GameState) -> float:
        player = leg.params.get("player", "")
        line = float(leg.params.get("line", 0.0))
        if not player:
            return 0.5
        if self.stats.player(player) is not None:
            return player_pts_over_prob(player, line, gs, self.stats)
        return 0.5   # no historical data -> neutral; shrinkage defers to market

    # ---------- helpers ----------

    def _combined_season_pace(self, gs: GameState) -> float:
        """Season-average combined scoring rate (pts/min) for this matchup."""
        h_team = self.stats.team(gs.home_team_id)
        a_team = self.stats.team(gs.away_team_id)
        h_rate = h_team.pts_per_min if h_team else _LEAGUE_AVG_PTS_PER_MIN
        a_rate = a_team.pts_per_min if a_team else _LEAGUE_AVG_PTS_PER_MIN
        return h_rate + a_rate

    def _stern_fallback(self, leg: Leg, gs: GameState) -> float:
        """Basic Stern-only probability for use when context computation fails."""
        if leg.status is LegStatus.COMPLETED:
            return 1.0
        if leg.status is LegStatus.FAILED:
            return 0.0
        k = leg.kind
        if k == "moneyline":
            return self.stern.win_prob(
                gs.score_diff, gs.tau_minutes, side=leg.params.get("side", "home"))
        if k in ("total_over", "total_under"):
            elapsed = max(48.0 - gs.tau_minutes, 1e-6)
            pace = (gs.home_score + gs.away_score) / elapsed
            mean = gs.home_score + gs.away_score + pace * gs.tau_minutes
            std = 1.6 * self.stern.sigma * np.sqrt(max(gs.tau_minutes, 1e-9))
            p_over = float(1.0 - norm.cdf(leg.params["line"], loc=mean, scale=std))
            return float(np.clip(p_over if k == "total_over" else 1.0 - p_over, 0.01, 0.99))
        return 0.5
