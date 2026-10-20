"""
metrics.py — scoring for the exit policies, plus the baseline policies to grade
against.

A single game gives one data point, so metrics are computed over many simulated
realizations of the game (the backbone is the ground-truth generator here -- the
"oracle mode" evaluation): each realization is a full game path with its own
Kalshi price series; every policy runs on it; results aggregate into rates and a
risk-adjusted value.

Primary metric is the certainty-equivalent (expected log-utility mapped back to
dollars). % of oracle is the intuitive secondary. Baselines (hold-to-expiry,
fixed-threshold, immediate) bracket the space so the adaptive DP/LSM policy has
something to beat. (Baselines live here rather than execution/ because they only
exist for comparison.)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import config
from data.schemas import Combo, GameState
from data.artificial_odds import Quote
from execution.stopping import reward, RunResult


# ---- baseline policies (return the same RunResult as run_dp / run_lsm) -------
def hold_to_expiry(combo: Combo, timeline: list[GameState], quotes: list[Quote],
                   wealth: float = config.BANKROLL_W, payout: float = 1.0) -> RunResult:
    won = combo.resolves_yes(timeline[-1])
    return RunResult(combo.id, [], None, payout * (1.0 if won else 0.0), won)


def fixed_threshold(combo: Combo, timeline: list[GameState], quotes: list[Quote],
                    theta: float = 0.85, wealth: float = config.BANKROLL_W,
                    payout: float = 1.0) -> RunResult:
    for t in range(len(timeline) - 1):
        if quotes[t].bid >= theta:
            return RunResult(combo.id, [], t, reward(quotes[t].bid),
                             combo.resolves_yes(timeline[-1]))
    won = combo.resolves_yes(timeline[-1])
    return RunResult(combo.id, [], None, payout * (1.0 if won else 0.0), won)


def immediate_cashout(combo: Combo, timeline: list[GameState], quotes: list[Quote],
                      wealth: float = config.BANKROLL_W, payout: float = 1.0) -> RunResult:
    return RunResult(combo.id, [], 0, reward(quotes[0].bid),
                     combo.resolves_yes(timeline[-1]))


# ---- oracle (best hindsight outcome for one realization) --------------------
def oracle_value(quotes: list[Quote], won: bool, payout: float = 1.0) -> float:
    """Max over {cash out at any step, hold to settlement}. Settlement pays full
    $1 with no spread/fee, so on winners it beats every cash-out."""
    best_cash = max((reward(q.bid) for q in quotes), default=0.0)
    return max(best_cash, payout * (1.0 if won else 0.0))


# ---- aggregation ------------------------------------------------------------
@dataclass
class PolicyMetrics:
    name: str
    n: int
    bust_rate: float        # fraction that walked away with $0
    win_rate: float         # fraction that walked away with > $0
    mean_value: float       # E[realized value]
    certainty_equiv: float  # exp(E[log(W+value)]) - W  (risk-adjusted $)
    pct_oracle: float       # mean(realized / hindsight-best)


def evaluate(name: str, run_fn, pairs, combo: Combo,
             wealth: float = config.BANKROLL_W, payout: float = 1.0) -> PolicyMetrics:
    """run_fn(timeline, quotes) -> RunResult, applied to each (timeline, quotes) pair."""
    vals, oracles = [], []
    for timeline, quotes in pairs:
        res = run_fn(timeline, quotes)
        won = combo.resolves_yes(timeline[-1])
        vals.append(res.realized_value)
        oracles.append(oracle_value(quotes, won, payout))

    vals = np.asarray(vals, dtype=float)
    oracles = np.asarray(oracles, dtype=float)
    ce = float(np.exp(np.mean(np.log(np.maximum(wealth + vals, 1e-9)))) - wealth)
    pct = float(np.mean(np.divide(vals, oracles, out=np.zeros_like(vals), where=oracles > 0)))
    return PolicyMetrics(
        name=name, n=len(vals),
        bust_rate=float(np.mean(vals == 0.0)),
        win_rate=float(np.mean(vals > 0.0)),
        mean_value=float(np.mean(vals)),
        certainty_equiv=ce, pct_oracle=pct,
    )


def format_table(rows: list[PolicyMetrics]) -> str:
    head = f"    {'policy':<18}{'bust%':>7}{'win%':>7}{'mean$':>9}{'CE$':>9}{'%oracle':>9}"
    lines = [head]
    for m in rows:
        lines.append(f"    {m.name:<18}{m.bust_rate*100:>7.1f}{m.win_rate*100:>7.1f}"
                     f"{m.mean_value:>9.3f}{m.certainty_equiv:>9.3f}{m.pct_oracle*100:>9.1f}")
    return "\n".join(lines)