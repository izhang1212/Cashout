"""
backbone.py — the Monte Carlo game simulator (the state-transition model).

Rolls the game forward from a start state to the buzzer, N times, snapshotting
the combo-relevant state at each timeline step. Reduced-form for v1:
  - team scores drift by team strength (rates leaned so margin sd matches the
    moneyline marginal),
  - player counting stats are Poisson (matching the player-prop marginal), so
    prod stays calibrated against simulated outcomes,
  - a player's Poisson rate is scaled by its own team's scoring via COUPLE_RHO,
    so correlation across legs is captured natively (0 = independent baseline).
Possession-level detail is a roadmap upgrade. Paths feed LSM.
"""
from __future__ import annotations
import math
from copy import deepcopy

import numpy as np

import config
from data.schemas import Combo, GameState, STAT_KEY
from prediction.marginals import minutes_left, avg_pace

MARGIN_VOL = 3.2                       # per sqrt-min; matches the moneyline marginal
TEAM_VOL = MARGIN_VOL / math.sqrt(2.0)  # so (away - home) sd == MARGIN_VOL
COUPLE_RHO = 0.0                       # player<->team scoring coupling (0=indep baseline; >0 adds correlation)


def simulate_paths(combo: Combo, timeline: list[GameState], ref: dict,
                   n_sims: int = config.N_SIMS, start_step: int = 0,
                   seed: int = config.RNG_SEED) -> list[list[GameState]]:
    rng = np.random.default_rng(seed)
    start = timeline[start_step]
    home, away = start.home_team, start.away_team

    base_rate = avg_pace(ref) * 1.05 / config.GAME_MINUTES   # pts/min per team
    home_lean = config.team_strength_lean(home, home, away, ref)  # expected full-game (home - away)
    drift = home_lean / config.GAME_MINUTES                       # margin drift per minute
    home_rate = base_rate + drift / 2.0
    away_rate = base_rate - drift / 2.0
    players = ref.get("players", {})
    mins_ref = ref.get("minutes", {})

    tracked = []                       # (player, stat, team, rate, share)
    for leg in combo.legs:
        if leg.type in STAT_KEY:
            stat = STAT_KEY[leg.type]
            tracked.append((
                leg.subject, stat,
                players.get(leg.subject, {}).get("team", home),
                players.get(leg.subject, {}).get(stat, 0.0),
                min(1.0, mins_ref.get(leg.subject, 34) / config.GAME_MINUTES),
            ))

    T = len(timeline)
    paths: list[list[GameState]] = []

    for _ in range(n_sims):
        h_score = float(start.home_score)
        a_score = float(start.away_score)
        pstats = {p: dict(st) for p, st in start.player_stats.items()}
        snaps = [start]

        for t in range(start_step + 1, T):
            dt = max(minutes_left(timeline[t - 1]) - minutes_left(timeline[t]), 0.0)
            mu_h = max(0.0, home_rate * dt)
            mu_a = max(0.0, away_rate * dt)
            sd = TEAM_VOL * math.sqrt(dt) if dt > 0 else 0.0
            h_add = max(0.0, rng.normal(mu_h, sd))
            a_add = max(0.0, rng.normal(mu_a, sd))

            for (player, stat, team, rate, share) in tracked:
                team_add = h_add if team == home else a_add
                mu_team = mu_h if team == home else mu_a
                scale = 1.0 + COUPLE_RHO * ((team_add / mu_team) - 1.0) if mu_team > 0 else 1.0
                lam = max(0.0, rate * share * dt * max(0.0, scale))
                add = int(rng.poisson(lam)) if lam > 0 else 0
                pstats.setdefault(player, {})
                pstats[player][stat] = pstats[player].get(stat, 0) + add

            h_score += h_add
            a_score += a_add
            snaps.append(GameState(
                period=timeline[t].period, clock=timeline[t].clock,
                home_team=home, away_team=away,
                home_score=round(h_score), away_score=round(a_score),
                possession=None, player_stats=deepcopy(pstats),
            ))
        paths.append(snaps)

    return paths