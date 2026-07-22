"""Player prop probability model: P(player exceeds stat line by game end).

For a prop like "LeBron James points over 27.5", this answers:
  P(final_pts > 27.5 | current game state)

Approach
--------
1. Estimate remaining minutes (season avg, adjusted for foul trouble and game time).
2. Blend the player's current per-minute rate in this game with their season baseline.
   Bayesian blend: early in the game the season rate dominates; late game the actual
   pace matters more.
3. Project game-end stat ~ Normal(projected_total, sigma).
4. P(over line) = 1 - Normal.cdf(line, projected_total, sigma).

The shrinkage layer in the follower then blends this model probability toward the
market-implied probability, weighted by how much forward edge this model has
demonstrated. Until proven otherwise, the market gets heavy weight.

Known approximations
--------------------
* Single-stat model (points only for now; same skeleton works for reb/ast).
* Gaussian terminal distribution (slightly fat-tailed in reality; upgrade: NegBinomial).
* Minutes projection doesn't model game script (team is down 20 => star plays
  garbage time fewer minutes). Upgrade: add score_diff to the minutes model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ...data_gathering.nba.stats import NBAStatsCache
from ...data_gathering.nba.game_state import GameState
from .foul_model import foul_minutes_at_risk

# Bayesian prior strength in minutes: season rate weighted as if we saw N minutes.
_PRIOR_MINUTES = 18.0


def projected_minutes_remaining(player_name: str, gs: GameState,
                                stats_cache: NBAStatsCache) -> float:
    """Estimate remaining playing time for this player.

    Uses season average minutes as the baseline, adjusts for:
      - Minutes already played this game
      - Foul trouble (via foul_minutes_at_risk)
      - Hard cap at remaining game time
    """
    live_stats = gs.player_stats.get(player_name, {})
    n_fouls = int(live_stats.get("fouls", 0))
    min_played = float(live_stats.get("min", 0.0))

    hist = stats_cache.player(player_name)
    season_min = hist.min_per_game if hist is not None else 28.0

    # How many more minutes does this player typically play in a game?
    baseline_remaining = max(0.0, season_min - min_played)
    # Cap at actual game time remaining
    remaining_game_min = gs.tau_minutes
    expected = min(baseline_remaining, remaining_game_min)

    # Foul trouble discount
    if n_fouls >= 3:
        at_risk_frac = foul_minutes_at_risk(n_fouls, gs.tau_minutes)
        expected *= (1.0 - at_risk_frac)

    return float(max(0.0, expected))


def player_pts_over_prob(player_name: str, line: float,
                         gs: GameState, stats_cache: NBAStatsCache) -> float:
    """P(player finishes with > line points in this game).

    Returns a float in [0.01, 0.99]. Falls back to 0.5 if no historical data
    is available (treated as "no model" and heavily shrunk toward market).
    """
    live = gs.player_stats.get(player_name, {})
    current_pts = float(live.get("pts", 0.0))

    # Already over the line
    if current_pts > line:
        return 0.99

    hist = stats_cache.player(player_name)
    if hist is None:
        return 0.5   # no historical data; neutral

    ppg = hist.pts_per_game
    pts_per_min_season = hist.pts_per_min
    pts_sigma = hist.pts_std   # game-to-game std dev from nba_stats

    rem_min = projected_minutes_remaining(player_name, gs, stats_cache)
    min_played = float(live.get("min", 0.0))

    if rem_min <= 0.0:
        # Game effectively over for this player
        return 0.99 if current_pts > line else 0.01

    # Bayesian blend of current game rate and season rate
    if min_played > 1.0:
        game_rate = current_pts / min_played   # pts/min so far this game
        # Weight the game rate by minutes played; season rate weighted by _PRIOR_MINUTES
        game_weight = min_played / (min_played + _PRIOR_MINUTES)
        blended_rate = game_weight * game_rate + (1.0 - game_weight) * pts_per_min_season
    else:
        blended_rate = pts_per_min_season

    projected_remaining = blended_rate * rem_min
    projected_total = current_pts + projected_remaining

    # Scale sigma to remaining game: longer remaining → more variance
    # Season std dev is calibrated for a full game; scale by sqrt of time fraction
    time_fraction = rem_min / max(hist.min_per_game, 1.0)
    sigma = pts_sigma * np.sqrt(max(time_fraction, 0.0))
    sigma = max(sigma, 1.0)   # floor: always some uncertainty

    p_over = float(1.0 - norm.cdf(line, loc=projected_total, scale=sigma))
    return float(np.clip(p_over, 0.01, 0.99))
