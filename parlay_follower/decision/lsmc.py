"""Longstaff-Schwartz Monte Carlo: the branch for high-dimensional states.

Used only when player-prop legs expand the state beyond what a grid can hold
(stat pace, minutes, foul trouble are continuous). Same Bellman logic as
exact_dp, approximated: simulate forward paths, regress realized continuation
payoffs on basis functions of the state, use the fitted regression as the
exercise boundary. This is the established industry technique (Bermudan
swaptions) for exactly this situation.
"""
from __future__ import annotations

import numpy as np


def basis(states: np.ndarray) -> np.ndarray:
    """Polynomial basis: [1, x, x^2, pairwise xy]. states: (n_paths, n_features)."""
    n, f = states.shape
    cols = [np.ones(n)]
    cols += [states[:, j] for j in range(f)]
    cols += [states[:, j] ** 2 for j in range(f)]
    cols += [states[:, a] * states[:, b] for a in range(f) for b in range(a + 1, f)]
    return np.column_stack(cols)
