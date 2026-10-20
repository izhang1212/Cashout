from __future__ import annotations
import math
import random
from dataclasses import dataclass

import config
from data.schemas import LegType, Side, Leg, Combo, GameState, STAT_KEY


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


def _minutes_left(state: GameState) -> float:
    per_min = config.PERIOD_SECONDS / 60.0
    if state.period <= config.REGULATION_PERIODS:
        return (config.REGULATION_PERIODS - state.period) * per_min + state.clock / 60.0
    return state.clock / 60.0


def _leg_prob(leg: Leg, state: GameState, ref: dict) -> float:
    mins = _minutes_left(state)
    if mins <= 1e-6:
        met = leg.condition_met(state)
        yes = 1.0 if met else 0.0
        return yes if leg.side is Side.YES else 1.0 - yes

    if leg.type in STAT_KEY:
        stat = STAT_KEY[leg.type]
        have = state.player_stats.get(leg.subject, {}).get(stat, 0)
        need = leg.line - have
        if need <= 0:
            p = 1.0
        else:
            rate = ref.get("players", {}).get(leg.subject, {}).get(stat, 0.0)
            share = min(1.0, ref.get("minutes", {}).get(leg.subject, 34) / config.GAME_MINUTES)
            p = _poisson_sf(int(math.ceil(need)), rate * mins * share)
    elif leg.type is LegType.TOTAL:
        teams = ref.get("teams", {})
        pace = sum(t.get("pace", 100) for t in teams.values()) / max(1, len(teams))
        proj = state.total_points() + (pace * 2 * 1.05 / config.GAME_MINUTES) * mins
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


def combo_fair(combo: Combo, state: GameState, ref: dict) -> float:
    p = 1.0
    for leg in combo.legs:
        p *= _leg_prob(leg, state, ref)
    return p


@dataclass
class Quote:
    step: int
    fair: float
    bid: float


def generate_kalshi_series(combo: Combo, timeline: list[GameState], ref: dict,
                           half_spread: float = config.HALF_SPREAD,
                           noise_std: float = config.NOISE_STD,
                           rng_seed: int = config.RNG_SEED) -> list[Quote]:
    rng = random.Random(rng_seed)
    quotes, noise = [], 0.0
    for i, state in enumerate(timeline):
        fair = combo_fair(combo, state, ref)
        noise = 0.6 * noise + rng.gauss(0.0, noise_std)
        bid = min(max(fair - half_spread + noise, 0.0), 1.0)
        quotes.append(Quote(step=i, fair=fair, bid=bid))
    return quotes