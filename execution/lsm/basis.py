# basis functions

"""
basis.py (LSM) — the regression basis for the Longstaff-Schwartz continuation fit.

Split out from the solver so the feature choice lives in one place: the fit
(solver.py) and the runtime evaluation (policy.py) both import combo_features
from here, guaranteeing they build the exact same feature vector.

For an all-or-nothing combo, E[U(W + settle) | state] is ~linear in the combo
probability, so [1, prod, prod^2] is the right basis (prod^2 absorbs the mild
curvature from the exercise option). Per-leg probs are collinear with their
product and just destabilize the fit. Add richer features here (e.g. minutes
left) if a combo's continuation stops being well summarized by prod alone.
"""
from __future__ import annotations

from data.schemas import Combo, GameState
from prediction.marginals import leg_fair_prob


def combo_features(combo: Combo, state: GameState, ref: dict) -> list[float]:
    prod = 1.0
    for leg in combo.legs:
        prod *= leg_fair_prob(leg, state, ref)
    return [1.0, prod, prod * prod]