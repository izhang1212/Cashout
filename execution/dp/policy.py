# runtime lookup: 
    # walk the realized timeline
    # compare the actaul Kalshi bid against pre-computed continuation value at realized state
    # emit hold/cash-out, stop at first cash-out

from __future__ import annotations
from dataclasses import dataclass

import config
from data.schemas import Leg, Combo, GameState
from data.artificial_odds import Quote
from execution.dp.solver import DPSolution, realized_makes
from execution.stopping import decide, reward, Action, StepDecision, RunResult

# Result format for DP
@dataclass
class RunResult:
    combo_id: str
    decisions: list[StepDecision]
    # step we cashed out at (or None if settled)
    exit_step: int | None      
    # cash-out reward (dollars) or settlement payout (dollars)    
    realized_value: float     
    # did the leg's condition hold at the buzzer     
    won: bool                      

    @property
    def settled(self) -> bool:
        return self.exit_step is None


def run_dp(combo: Combo, timeline: list[GameState], quotes: list[Quote], solution: DPSolution, wealth: float = config.BANKROLL_W, payout: float = 1.0) -> RunResult:
    leg = combo.legs[0]                       # single-leg combo
    T = len(timeline)
    decisions: list[StepDecision] = []

    # decision steps (len of timeline - 1, since last step is settle)
    for t in range(T - 1):                    
        makes = realized_makes(leg, timeline[t])
        cont = solution.continuation(t, makes)
        step = decide(t, quotes[t].bid, cont, wealth)
        decisions.append(step)
        if step.action is Action.CASH_OUT:
            return RunResult(
                combo_id=combo.id, decisions=decisions, exit_step=t,
                realized_value=reward(quotes[t].bid),
                won=leg.resolves_yes(timeline[-1]),
            )

    won = leg.resolves_yes(timeline[-1])
    return RunResult(
        combo_id=combo.id, decisions=decisions, exit_step=None,
        realized_value=payout * (1.0 if won else 0.0), won=won,
    )