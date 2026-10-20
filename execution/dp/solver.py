# exact backward induction for a single counting-stat leg (e.g. single leg on a player stat or game outcome)
    # state is (stat_so_far, time_step); remaining stat follows a Poisson counting process, so teh problem is small and solved exactly 
    # with all "won" states collasped to `line`, so the grid is tiny (e.g. 3 states x 4 steps for Pritchard)
    # produces a continuation-value surface C[t][m] the harness compares the live cash-out price against
# design note: the recursion needs a cash-out value at EVERY grid state to value future exercise, but we only observe the real Kalshi price along the realized path. So the solve MODELS future exercise as (fair_prob - half_spread) - an efficient-market assumption. At runtime the policy compares the ACTUAL bid against this surface; that gap is where real decisions come from.

from __future__ import annotations
import math
from dataclasses import dataclass

import config
from data.schemas import Leg, GameState, STAT_KEY


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)

# P(X >= k) for X ~ Poisson(lam).  (1 - CDF, or survival function)
def _poisson_sf(k: int, lam: float) -> float:
    if k <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    return 1.0 - sum(_poisson_pmf(i, lam) for i in range(k))

# Minutes left in the game
def minutes_left(state: GameState) -> float:
    per_min = config.PERIOD_SECONDS / 60.0
    if state.period <= config.REGULATION_PERIODS:
        return (config.REGULATION_PERIODS - state.period) * per_min + state.clock / 60.0
    return state.clock / 60.0

# Per-minute-played rate for this leg's stat, and the player's court share.
def leg_rate_share(leg: Leg, ref: dict) -> tuple[float, float]:
    stat = STAT_KEY[leg.type]
    rate = ref.get("players", {}).get(leg.subject, {}).get(stat, 0.0)
    share = min(1.0, ref.get("minutes", {}).get(leg.subject, 34) / config.GAME_MINUTES)
    return rate, share

# So far, how many makes have we seen for this leg's stat? (0..line)
def realized_makes(leg: Leg, state: GameState) -> int:
    return int(state.player_stats.get(leg.subject, {}).get(STAT_KEY[leg.type], 0))

# Return format 
@dataclass
class DPSolution:
    line: int
    steps: int
    # cont[t][m], continuation value (utility)
    cont: list[list[float]]           

    # collapse all "won" states to `line`
    def continuation(self, step: int, makes: int) -> float:
        m = min(makes, self.line)     
        return self.cont[step][m]

# Model probabilty teh leg finshes >= line, given makes & minutes left.  (Poisson counting process)
def fair_prob(makes: int, mins: float, rate: float, share: float, line: int) -> float:
    if makes >= line:
        return 1.0
    lam = rate * mins * share
    return _poisson_sf(line - makes, lam)


def solve_dp(leg: Leg, timeline: list[GameState], ref: dict, wealth: float = config.BANKROLL_W, payout: float = 1.0) -> DPSolution:
    line = int(leg.line)
    T = len(timeline)
    rate, share = leg_rate_share(leg, ref)
    states = range(line + 1)          # m = 0..line (line = "won")

    V = [[0.0] * (line + 1) for _ in range(T)]
    C = [[0.0] * (line + 1) for _ in range(T)]

    # terminal step: settle (no decision)
    for m in states:
        V[T - 1][m] = config.utility(wealth + payout * (1.0 if m >= line else 0.0))

    # backward induction over decision steps T-2 .. 0
    for t in range(T - 2, -1, -1):
        dt = max(minutes_left(timeline[t]) - minutes_left(timeline[t + 1]), 0.0)
        lam = rate * dt * share
        mins_here = minutes_left(timeline[t])
        for m in states:
            # already won, stays won
            if m >= line:
                cont = V[t + 1][line]                 
            else:
                cont = 0.0
                # additional makes staying < line
                for k in range(line - m):             
                    cont += _poisson_pmf(k, lam) * V[t + 1][m + k]
                cont += _poisson_sf(line - m, lam) * V[t + 1][line]   # reaches line
            C[t][m] = cont

            fair = fair_prob(m, mins_here, rate, share, line)
            modeled_bid = min(max(fair - config.HALF_SPREAD, 0.0), 1.0)
            exercise = config.utility(wealth + (modeled_bid - config.kalshi_fee(modeled_bid)))

            V[t][m] = max(exercise, cont)

    # return the DP solution surface (continuation values) for this leg
    return DPSolution(line=line, steps=T, cont=C)