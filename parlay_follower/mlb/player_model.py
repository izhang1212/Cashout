"""MLB player prop probability model.

Answers questions like:
  P(Shohei Ohtani gets > 1.5 hits today)
  P(Paul Skenes strikes out > 6.5 batters today)
  P(Aaron Judge hits a home run today)
  P(player gets > 2.5 total bases)

Approach (Bayesian blend + Normal projection):
  1. Count what the player has already done this game.
  2. Estimate remaining at-bats / innings pitched.
  3. Blend the player's current per-AB rate with season baseline.
     Early: season dominates. Late: game rate dominates.
  4. Project game-end stat ~ Normal(projected_total, sigma).
  5. P(over line) = 1 - Normal.cdf(line, projected_total, sigma).

All probabilities are clipped to [0.01, 0.99]. The shrinkage layer in the
follower blends these toward market prices, so the model doesn't need to be
perfectly calibrated to be useful.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .game_state import MLBGameState
from .stats import MLBStatsCache

_PRIOR_AB = 12.0   # season rate weighted as if we saw N at-bats


def _remaining_ab(player_name: str, gs: MLBGameState,
                  stats: MLBStatsCache) -> float:
    """Expected remaining at-bats for a batter."""
    live = gs.player_stats.get(player_name, {})
    ab_so_far = float(live.get("ab", 0))
    hist = stats.batter(player_name)
    season_ab_pg = hist.ab_per_game if hist else 3.8
    # Remaining based on season average minus already taken
    remaining_fraction = 1.0 - min(gs.outs_completed / 27.0, 1.0)
    expected_total = season_ab_pg * (1.0 / max(1 - remaining_fraction, 0.01))
    return max(0.0, min(season_ab_pg - ab_so_far, season_ab_pg * remaining_fraction))


def _remaining_innings(pitcher_name: str, gs: MLBGameState,
                        stats: MLBStatsCache) -> float:
    """Expected remaining innings for the current pitcher."""
    hist = stats.pitcher(pitcher_name)
    ips = hist.innings_per_start if hist else 5.5
    # Innings pitched so far (approximate from outs)
    outs_done = gs.outs_completed
    ip_done = outs_done / 3.0
    return max(0.0, min(ips - ip_done, gs.outs_remaining / 3.0))


def batter_hits_over_prob(player_name: str, line: float,
                          gs: MLBGameState, stats: MLBStatsCache) -> float:
    """P(player finishes with > line hits)."""
    live = gs.player_stats.get(player_name, {})
    current = float(live.get("hits", 0))
    if current > line:
        return 0.99

    hist = stats.batter(player_name)
    season_avg = hist.avg if hist else 0.260
    rem_ab = _remaining_ab(player_name, gs, stats)
    if rem_ab <= 0:
        return 0.01

    ab_done = float(live.get("ab", 0))
    game_avg = (current / ab_done) if ab_done > 1 else season_avg
    w = ab_done / (ab_done + _PRIOR_AB)
    blended_avg = w * game_avg + (1 - w) * season_avg

    projected_hits = current + blended_avg * rem_ab
    sigma = max(np.sqrt(rem_ab * blended_avg * (1 - blended_avg)), 0.5)
    return float(np.clip(1 - norm.cdf(line, projected_hits, sigma), 0.01, 0.99))


def batter_total_bases_over_prob(player_name: str, line: float,
                                  gs: MLBGameState, stats: MLBStatsCache) -> float:
    """P(player finishes with > line total bases)."""
    live = gs.player_stats.get(player_name, {})
    current = float(live.get("tb", 0))
    if current > line:
        return 0.99

    hist = stats.batter(player_name)
    slg = hist.slg if hist else 0.420
    rem_ab = _remaining_ab(player_name, gs, stats)
    if rem_ab <= 0:
        return 0.01

    ab_done = float(live.get("ab", 0))
    tb_done = current
    game_slg = (tb_done / ab_done) if ab_done > 1 else slg
    w = ab_done / (ab_done + _PRIOR_AB)
    blended_slg = w * game_slg + (1 - w) * slg

    projected_tb = current + blended_slg * rem_ab
    sigma = max(blended_slg * np.sqrt(rem_ab), 0.8)
    return float(np.clip(1 - norm.cdf(line, projected_tb, sigma), 0.01, 0.99))


def batter_home_run_prob(player_name: str, gs: MLBGameState,
                          stats: MLBStatsCache) -> float:
    """P(player hits >= 1 HR today)."""
    live = gs.player_stats.get(player_name, {})
    if float(live.get("hr", 0)) >= 1:
        return 0.99

    hist = stats.batter(player_name)
    hr_rate = hist.hr_rate if hist else 0.035   # HRs per AB
    rem_ab = _remaining_ab(player_name, gs, stats)
    if rem_ab <= 0:
        return 0.01

    # P(at least 1 HR in rem_ab ABs) = 1 - P(0 HRs) = 1 - (1-hr_rate)^rem_ab
    p_no_hr = (1.0 - hr_rate) ** rem_ab
    return float(np.clip(1.0 - p_no_hr, 0.01, 0.99))


def pitcher_strikeouts_over_prob(pitcher_name: str, line: float,
                                  gs: MLBGameState, stats: MLBStatsCache) -> float:
    """P(pitcher finishes with > line strikeouts)."""
    live = gs.player_stats.get(pitcher_name, {})
    current_k = float(live.get("k", 0))
    if current_k > line:
        return 0.99

    hist = stats.pitcher(pitcher_name)
    k_per_inn = (hist.k_per_9 / 9.0) if hist else (8.7 / 9.0)
    rem_inn = _remaining_innings(pitcher_name, gs, stats)
    if rem_inn <= 0:
        return 0.01

    # Ks per inning is roughly Poisson
    projected_k = current_k + k_per_inn * rem_inn
    sigma = max(np.sqrt(k_per_inn * rem_inn), 0.8)
    return float(np.clip(1 - norm.cdf(line, projected_k, sigma), 0.01, 0.99))


def batter_rbi_over_prob(player_name: str, line: float,
                          gs: MLBGameState, stats: MLBStatsCache) -> float:
    """P(player finishes with > line RBI)."""
    live = gs.player_stats.get(player_name, {})
    current = float(live.get("rbi", 0))
    if current > line:
        return 0.99

    hist = stats.batter(player_name)
    rbi_pg = hist.rbi if hist else 0.60
    rem_fraction = gs.outs_remaining / 27.0
    projected = current + rbi_pg * rem_fraction
    sigma = max(np.sqrt(rbi_pg * rem_fraction), 0.5)
    return float(np.clip(1 - norm.cdf(line, projected, sigma), 0.01, 0.99))
