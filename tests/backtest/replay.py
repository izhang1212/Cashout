"""Tick-by-tick replay engine.

Replays a game's state series against logged (preferred) or modeled bids, runs
a policy, and records terminal P&L. Used both for the historical backtest and
for synthetic-game studies before real bid logs exist.

A "tick" is a plain dict so policies stay decoupled from internal classes:
    {"tau_min", "score_diff", "legs_completed", "legs_live",
     "executable_bid", "fair_value", "entry_price"}
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from parlay_follower.cashout.bid_model import BidModel
from parlay_follower.models.nba.stern import SternModel

Policy = Callable[[dict], bool]   # tick -> sell?


def synthetic_game_ticks(stern: SternModel, bid_model: BidModel, *,
                         q_other_fn: Callable[[float], float],
                         k_live: int, entry_price: float,
                         moneyline_side: str = "home",
                         n_steps: int = 96,
                         rng: np.random.Generator | None = None) -> tuple[list[dict], int]:
    """Simulate one game under the model; returns (ticks, combo_won 0/1).

    The "other legs" (folded into q_other) resolve at a uniformly random tick
    during the game -- mirroring tests/cpp_backtest/include/backtest_engine.hpp's
    prop_resolve_tick, so `legs_completed` actually flips from 0 to 1 mid-game
    instead of staying 0 for the whole tick stream. If the other legs fail,
    the combo is dead: ticks stop at that point and the caller's terminal
    fallback (payoff 0, since combo_won is False) settles it, matching the
    C++ engine's forced-exit-without-consulting-the-policy behavior. Before
    this, q_other_fn(tau) was evaluated every tick as if the other legs were
    perpetually still-live at their pregame probability, decoupled from
    whether they actually hit -- so the bid/fair_value stream a policy saw
    never reflected the game's own eventual outcome.

    Used to sanity-check policies before real logged data exists. Note the
    epistemic status: a backtest on model-generated games can only show a
    policy is optimal UNDER THE MODEL -- the paper-trading gate is what tests
    the model against reality.
    """
    rng = rng or np.random.default_rng()
    tau0 = 48.0
    path = stern.simulate_paths(0.0, tau0, n_paths=1, n_steps=n_steps, rng=rng)[0]
    sign = 1.0 if moneyline_side == "home" else -1.0

    q_prop = float(q_other_fn(0.0))
    prop_won = bool(rng.random() < q_prop)
    prop_resolve_tick = int(rng.integers(0, n_steps))   # uniform over [0, n_steps-1]

    ml_won = sign * path[-1] > 0
    combo_won = int(ml_won and prop_won)

    ticks = []
    for i, d in enumerate(path):
        if i == prop_resolve_tick and not prop_won:
            break   # other legs died: combo is dead, no more ticks/decisions

        tau = tau0 * (1 - i / n_steps)
        legs_completed = 1 if i >= prop_resolve_tick else 0
        q_eff = 1.0 if legs_completed else q_prop
        p_ml = stern.win_prob(d, tau, side=moneyline_side)
        p_combo = p_ml * q_eff
        bid = bid_model.bid(p_combo, tau, p_combo, k_live)
        ticks.append({
            "tau_min": tau, "score_diff": float(d),
            "legs_completed": legs_completed, "legs_live": k_live - legs_completed,
            "executable_bid": bid, "fair_value": p_combo,
            "entry_price": entry_price,
        })
    return ticks, combo_won


def run_policy(ticks: Iterable[dict], combo_won: int, entry_price: float,
               policy: Policy) -> float:
    """P&L per contract: sell proceeds (or terminal payoff) minus entry."""
    for tick in ticks:
        if policy(tick):
            return tick["executable_bid"] - entry_price
    return (1.0 if combo_won else 0.0) - entry_price


def run_dp_policy(ticks: list[dict], combo_won: int, entry_price: float,
                  dp_lookup: Callable[[float, float], bool]) -> float:
    """Same, but the policy is a precomputed DP boundary lookup(tau, diff).

    Skips the terminal tick (tau=0): the DP's exercise=True sentinel at
    tau=0 is a boundary condition, not a tradeable signal — at game end the
    position settles at the true payoff (1 or 0), not the market bid.
    """
    for tick in ticks:
        if tick["tau_min"] <= 0.0:
            break   # terminal — settle at true payoff below
        if dp_lookup(tick["tau_min"], tick["score_diff"]):
            return tick["executable_bid"] - entry_price
    return (1.0 if combo_won else 0.0) - entry_price


def monte_carlo_study(n_games: int, stern: SternModel, bid_model: BidModel, *,
                      q_other_fn, k_live: int, entry_price: float,
                      policies: dict[str, Policy],
                      dp_lookup: Callable[[float, float], bool] | None = None,
                      seed: int = 0) -> dict[str, np.ndarray]:
    """Run all policies on the same simulated games (common random numbers)."""
    rng = np.random.default_rng(seed)
    results: dict[str, list[float]] = {name: [] for name in policies}
    if dp_lookup is not None:
        results["exact_dp"] = []

    for _ in range(n_games):
        ticks, won = synthetic_game_ticks(
            stern, bid_model, q_other_fn=q_other_fn, k_live=k_live,
            entry_price=entry_price, rng=rng,
        )
        for name, pol in policies.items():
            results[name].append(run_policy(ticks, won, entry_price, pol))
        if dp_lookup is not None:
            results["exact_dp"].append(run_dp_policy(ticks, won, entry_price, dp_lookup))

    return {k: np.array(v) for k, v in results.items()}
