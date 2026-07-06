"""Foul trouble model: win-probability impact of key players in foul trouble.

A player in serious foul trouble is likely to sit for some portion of remaining
time, reducing their team's offensive efficiency. The impact depends on:

  * How many fouls they have (4 fouls vs. 3 fouls is materially different)
  * How much time is left (4 fouls with 2 min left is fine; with 15 min left is not)
  * How important that player is to their team (star vs. bench player)

Player importance is derived from season usage_pct × minutes_share. A 30%-usage
player who averages 36 min contributes ~22.5% of team offense while on the floor.
If they sit for 40% of remaining time, that's ~9% of remaining team offense at risk.

Calibration constant _IMPORTANCE_TO_WIN_PROB_DELTA is intentionally conservative:
losing an entire game's worth of a star player is worth roughly 6-8% win probability
(empirical NBA estimate). The model caps each team's total impact at 15pp so
extreme cases don't produce implausible swings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...data_gathering.nba.stats import NBAStatsCache
from ...shared.game_feed.game_state import GameState

# A fully removed star (importance=1.0) for the entire game is worth ~7pp win prob.
_IMPORTANCE_TO_WIN_PROB_DELTA = 0.07

# Maximum per-team delta so the model doesn't produce absurd outputs.
_MAX_TEAM_DELTA = 0.15


@dataclass
class FoulTroubleImpact:
    home_delta: float                    # delta to home win prob (negative = hurts home)
    away_delta: float                    # delta to away win prob (negative = hurts away)
    troubled_players: list[str] = field(default_factory=list)


def _foul_sit_risk(n_fouls: int, tau_min: float) -> float:
    """Probability the player will have meaningfully reduced minutes from foul trouble.

    Thresholds match standard NBA coaching conventions:
      2 fouls: minor concern; coaches may sit briefly early in game
      3 fouls: moderate; likely to sit for stretches if early/mid game
      4 fouls: serious; will sit until late unless team is desperate
      5 fouls: on the brink of fouling out; coaches minimize exposure
    """
    if n_fouls <= 1:
        return 0.0
    if n_fouls == 2:
        # Only a concern if very early (lots of game left)
        return 0.10 if tau_min > 30 else 0.05
    if n_fouls == 3:
        return 0.30 if tau_min > 18 else (0.20 if tau_min > 6 else 0.08)
    if n_fouls == 4:
        return 0.65 if tau_min > 12 else (0.45 if tau_min > 4 else 0.20)
    # 5 fouls
    return 0.80 if tau_min > 6 else 0.50


def _minutes_at_risk_fraction(n_fouls: int, tau_min: float) -> float:
    """Expected fraction of remaining game minutes the player will miss."""
    risk = _foul_sit_risk(n_fouls, tau_min)
    # Players in more trouble sit for longer stretches when they do sit
    if n_fouls <= 2:
        return risk * 0.25
    if n_fouls == 3:
        return risk * 0.38
    if n_fouls == 4:
        return risk * 0.55
    return risk * 0.75   # 5 fouls: coaches keep them glued to the bench


def foul_minutes_at_risk(n_fouls: int, tau_min: float) -> float:
    """Public API used by player_model.py for per-player projection."""
    return _minutes_at_risk_fraction(n_fouls, tau_min)


class FoulTroubleModel:
    """Assess team-level win-probability impact from foul trouble across the roster."""

    def __init__(self, stats_cache: NBAStatsCache):
        self.cache = stats_cache

    def assess(self, gs: GameState) -> FoulTroubleImpact:
        home_raw = 0.0
        away_raw = 0.0
        troubled: list[str] = []
        tau_min = gs.tau_minutes

        for name, stats in gs.player_stats.items():
            n_fouls = int(stats.get("fouls", 0))
            if n_fouls < 3:
                continue   # 0-2 fouls: no material impact modeled

            team = str(stats.get("team", ""))
            hist = self.cache.player(name)

            if hist is not None:
                importance = hist.importance
            else:
                # Unknown player: assume league-average bench importance
                importance = 0.12

            at_risk = _minutes_at_risk_fraction(n_fouls, tau_min)
            delta = importance * at_risk * _IMPORTANCE_TO_WIN_PROB_DELTA

            if team == "home":
                home_raw -= delta
            elif team == "away":
                away_raw -= delta

            if at_risk > 0.20 and importance > 0.08:
                troubled.append(f"{name} ({n_fouls}F/{team})")

        # Cap and clip: impacts are always negative (hurts the team)
        home_delta = float(np.clip(home_raw, -_MAX_TEAM_DELTA, 0.0))
        away_delta = float(np.clip(away_raw, -_MAX_TEAM_DELTA, 0.0))

        return FoulTroubleImpact(
            home_delta=home_delta,
            away_delta=away_delta,
            troubled_players=troubled,
        )
