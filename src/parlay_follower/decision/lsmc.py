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


def lsmc_policy(state_paths: np.ndarray, bid_paths: np.ndarray,
                terminal_payoff: np.ndarray) -> list[np.ndarray | None]:
    """Backward LSMC over simulated paths.

    state_paths:     (n_paths, n_steps, n_features)
    bid_paths:       (n_paths, n_steps)   modeled executable bid along each path
    terminal_payoff: (n_paths,)           0/1 at T

    Returns per-step regression coefficient vectors (None where regression was
    skipped); at run time, SELL when current bid >= basis(state) @ coeffs.
    """
    n_paths, n_steps, _ = state_paths.shape
    cashflow = terminal_payoff.astype(float).copy()
    coeffs: list[np.ndarray | None] = [None] * n_steps

    for t in range(n_steps - 1, -1, -1):
        X = basis(state_paths[:, t, :])
        # Regress on in-the-money-ish paths (positive bid) per LS convention.
        itm = bid_paths[:, t] > 1e-4
        if itm.sum() < X.shape[1] * 3:
            continue
        beta, *_ = np.linalg.lstsq(X[itm], cashflow[itm], rcond=None)
        coeffs[t] = beta
        cont_hat = X @ beta
        exercise = itm & (bid_paths[:, t] >= cont_hat)
        cashflow[exercise] = bid_paths[exercise, t]

    return coeffs


def lsmc_should_sell(coeffs_t: np.ndarray | None, state: np.ndarray, bid: float) -> bool:
    if coeffs_t is None:
        return False
    cont = float(basis(state[None, :]) @ coeffs_t)
    return bid >= cont
