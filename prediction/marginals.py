"""
marginals.py — per-leg analytic probability that a leg finishes YES, given the
current state. This is the model side (a leg's marginal). artificial_odds keeps
its own copy for the synthetic *market* price; this stays separate as the *model*.

Moneyline/spread lean by team strength (config.team_strength_lean). Player props
now use live in-game signals rather than blind season-rate extrapolation:
  - hot/cold: shrink tonight's realized rate toward the season prior,
  - blowout benching: trim projected remaining minutes when the game is decided.
These are heuristic (no ML) and live only in the marginal for now -- the backbone
sim and synthetic market stay on season rates so the oracle-mode eval stays clean.
"""
from __future__ import annotations
import math

import config
from data.schemas import LegType, Side, Leg, Combo, GameState, STAT_KEY

# ---- player-prop live-signal knobs (heuristic placeholders; tune/learn later) ----
HOT_COLD_K = 24.0        # court-minutes of season prior in the hot/cold blend (higher = trust tonight less)
BLOWOUT_MARGIN = 18.0    # |margin| (pts) beyond which benching risk begins
BLOWOUT_SCALE = 16.0     # pts over the threshold for full blowout severity
BLOWOUT_MINS = 12.0      # minutes-left under which benching starts to bite
MAX_BENCH = 0.65         # cap on the fractional cut to a star's remaining minutes


def _poisson_sf(k: int, lam: float) -> float:
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    cdf, term = 0.0, math.exp(-lam)
    for n in range(k):
        if n > 0:
            term *= lam / n
        cdf += term
    return max(0.0, 1.0 - cdf)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def minutes_left(state: GameState) -> float:
    per_min = config.PERIOD_SECONDS / 60.0
    if state.period <= config.REGULATION_PERIODS:
        return (config.REGULATION_PERIODS - state.period) * per_min + state.clock / 60.0
    return state.clock / 60.0


def avg_pace(ref: dict) -> float:
    teams = ref.get("teams", {})
    return sum(t.get("pace", 100) for t in teams.values()) / max(1, len(teams))


def _player_prop_prob(leg: Leg, state: GameState, ref: dict, mins: float) -> float:
    """P(player reaches the line) using live in-game signals (hot/cold + benching)."""
    stat = STAT_KEY[leg.type]
    have = state.player_stats.get(leg.subject, {}).get(stat, 0)
    need = leg.line - have
    if need <= 0:                                   # already locked
        return 1.0

    season_rate = ref.get("players", {}).get(leg.subject, {}).get(stat, 0.0)  # per court-minute
    share = min(1.0, ref.get("minutes", {}).get(leg.subject, 34) / config.GAME_MINUTES)

    # (1) hot/cold: shrink tonight's realized rate toward the season prior. Weight
    # on tonight grows with court-minutes played, so early noise barely moves it.
    elapsed_court = max(0.0, (config.GAME_MINUTES - mins) * share)
    if elapsed_court > 1e-6:
        realized = have / elapsed_court
        w = elapsed_court / (elapsed_court + HOT_COLD_K)
        rate = w * realized + (1.0 - w) * season_rate
    else:
        rate = season_rate

    # (2)/(3) projected remaining court-minutes, trimmed for blowout benching --
    # only bites when the margin is large AND time is short (real garbage time).
    margin = abs(state.home_score - state.away_score)
    blowout = max(0.0, margin - BLOWOUT_MARGIN) / BLOWOUT_SCALE
    timegate = max(0.0, BLOWOUT_MINS - mins) / BLOWOUT_MINS
    bench_cut = min(MAX_BENCH, blowout * timegate)
    proj_court = mins * share * (1.0 - bench_cut)

    return _poisson_sf(int(math.ceil(need)), rate * proj_court)


def leg_fair_prob(leg: Leg, state: GameState, ref: dict) -> float:
    mins = minutes_left(state)
    if mins <= 1e-6:
        met = leg.condition_met(state)
        yes = 1.0 if met else 0.0
        return yes if leg.side is Side.YES else 1.0 - yes

    if leg.type in STAT_KEY:
        p = _player_prop_prob(leg, state, ref, mins)
    elif leg.type is LegType.TOTAL:
        proj = state.total_points() + (avg_pace(ref) * 2 * 1.05 / config.GAME_MINUTES) * mins
        sd = max(1.0, 3.0 * math.sqrt(mins))
        p = 1.0 - _normal_cdf((leg.line - proj) / sd)
    elif leg.type in (LegType.MONEYLINE, LegType.SPREAD):
        line = leg.line if leg.type is LegType.SPREAD else 0.0
        margin = state.score_of(leg.subject) - state.opponent_score_of(leg.subject)
        lean = config.team_strength_lean(leg.subject, state.home_team, state.away_team, ref)
        exp_margin = margin + lean * (mins / config.GAME_MINUTES)
        sd = max(1.0, 3.2 * math.sqrt(mins))
        p = 1.0 - _normal_cdf((-line - exp_margin) / sd)
    else:
        p = 0.5

    return p if leg.side is Side.YES else (1.0 - p)


def combo_fair_prob(combo: Combo, state: GameState, ref: dict) -> float:
    p = 1.0
    for leg in combo.legs:
        p *= leg_fair_prob(leg, state, ref)
    return p