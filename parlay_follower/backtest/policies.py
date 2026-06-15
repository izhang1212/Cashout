"""Comparison policies for the backtest: the DP must beat all of these."""
from __future__ import annotations


def hold_to_resolution(_tick) -> bool:
    return False


def sell_on_first_leg_complete(tick) -> bool:
    return tick["legs_completed"] >= 1


def sell_at_halftime(tick) -> bool:
    return tick["tau_min"] <= 24.0


def sell_at_profit_multiple(multiple: float = 2.0):
    def _policy(tick) -> bool:
        return tick["executable_bid"] >= multiple * tick["entry_price"]
    return _policy
