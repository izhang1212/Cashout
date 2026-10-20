# Execution

## Overview

**Purpose**: This is the final stage of this project; since we have determined probabilities, we are now left with executing within a live game. This final section determines whether to cash-out or hold at a given step within the game, where at each time step, a "hold" or "cash-out" signal is emitted.


## Process
1. Use either DP or LSM to solve for continuation value (in utility units)
2. `stopping.py` takes in the CV; since same format for both strategies, and everything after that is the same, we can use one engine to decide
3. Does two conversions:
    1. Convert current cash-out value (dollars) to equivilant utility amount (utility)
    2. Converts continuation value (utility) and converts it into certainty-equivilant dollar value (dollar)
4. Makes decision
    - cash-out now IFF cash-out-now utility >= continuation value
    - Otherwise, hold
5. Returns:
    - Action (cash-out or hold)
    - Cash-out reward (dollar value of cash-out now)
    - Hold Value (dollar value of holding using CE)
