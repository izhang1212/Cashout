# shared, solver-agnostic (either DP or LSM) decision harness
    # given continuation value (utility) from either solver (DP and LSM give same continuation value in units, so we can share)
    # computes the reward (R_t), does comparision, and emits hold/cash-out

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum

import config

# Actions we take (either hold or cash-out)
class Action(str, Enum):
    HOLD = "hold"
    CASH_OUT = "cash_out"

# Cash-out now reward (bid minus Kalshi fee)
def reward(bid: float) -> float:
    return bid - config.kalshi_fee(bid)

# Utility of cashing out now at `bid` (log utility of wealth + cash-out reward)
def exercise_utility(bid: float, wealth: float = config.BANKROLL_W) -> float:
    return config.utility(wealth + reward(bid))

# Certainty-equivalent dollar value of holding until next period
    # Inverts log utility: cont_value = U(W + CE)  ->  CE = exp(cont_value) - W
def continuation_ce(cont_value: float, wealth: float = config.BANKROLL_W) -> float:
    return math.exp(cont_value) - wealth

# Decision for a single step, considering the given factors
@dataclass
class StepDecision:
    step: int
    # action we will take
    action: Action
    # cash-out reward (dollars) if we cash out now
    cash_out_reward: float     
    # certainty-equivalent value (dollars) of holding until next step
    hold_value_ce: float      

# Cash-out IFF cashing out now is at least as good (in utility) as holding
    # `cont_value` is the precomputed continuation value at the current state, from whichever solver (DP table lookup or LSM regression).
def decide(step: int, bid: float, cont_value: float,
           wealth: float = config.BANKROLL_W) -> StepDecision:
    cash_out = exercise_utility(bid, wealth) >= cont_value
    return StepDecision(
        step=step,
        action=Action.CASH_OUT if cash_out else Action.HOLD,
        cash_out_reward=reward(bid),
        hold_value_ce=continuation_ce(cont_value, wealth),
    )

# Outcome of running a policy (DP or LSM) over one combo's realized game.
@dataclass
class RunResult:
    combo_id: str
    decisions: list[StepDecision]
    exit_step: int | None          # step we cashed out, or None if settled
    realized_value: float          # cash-out reward, or settlement payout
    won: bool                      # did the combo's condition hold at the buzzer

    @property
    def settled(self) -> bool:
        return self.exit_step is None