"""
policy.py (LSM) — runtime evaluation of the fitted continuation surface.

Walks the realized timeline; at each decision step, builds the state features,
evaluates beta_t -> continuation value, and asks the shared harness to compare it
against the ACTUAL Kalshi bid. Stops at the first cash-out; otherwise settles.
Same shape as the DP policy -- both feed one stopping.py.
"""
from __future__ import annotations

import config
from data.schemas import Combo, GameState
from data.artificial_odds import Quote
from execution.stopping import decide, reward, Action, RunResult
from execution.lsm.solver import LSMSolution
from execution.lsm.basis import combo_features


def run_lsm(combo: Combo, timeline: list[GameState], quotes: list[Quote],
            solution: LSMSolution, ref: dict, wealth: float = config.BANKROLL_W,
            payout: float = 1.0) -> RunResult:
    T = len(timeline)
    decisions = []

    for t in range(T - 1):                    # decision steps (last step settles)
        feats = combo_features(combo, timeline[t], ref)
        cont = solution.continuation(t, feats)
        step = decide(t, quotes[t].bid, cont, wealth)
        decisions.append(step)
        if step.action is Action.CASH_OUT:
            return RunResult(
                combo_id=combo.id, decisions=decisions, exit_step=t,
                realized_value=reward(quotes[t].bid),
                won=combo.resolves_yes(timeline[-1]),
            )

    won = combo.resolves_yes(timeline[-1])
    return RunResult(
        combo_id=combo.id, decisions=decisions, exit_step=None,
        realized_value=payout * (1.0 if won else 0.0), won=won,
    )