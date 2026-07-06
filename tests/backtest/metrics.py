"""Backtest metrics: mean P&L, Sharpe-style ratio, busts avoided, signal hit rate."""
from __future__ import annotations

import numpy as np


def summarize(pnls: np.ndarray, entry_prices: float | np.ndarray | None = None) -> dict:
    """entry_prices: cost basis per game -- a scalar if it's the same for every
    game in `pnls`, or an array aligned with `pnls` if it varies per game (e.g.
    the historical backtest, where each game's entry is its own tip-off model
    probability). A "bust" is a near-total loss of THAT stake:
    pnl <= -0.999 * entry_price. This project's entry prices are fractions of
    a $1 contract (e.g. 0.30), not $1 itself, so a fixed -0.999 threshold
    would never fire and silently report 0% busts regardless of the real loss
    rate. If entry_prices is omitted, bust_rate falls back to that fixed
    -0.999 dollars (correct only if entry_price really is ~$1).
    """
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {}
    mean = float(pnls.mean())
    std = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    if entry_prices is None:
        bust_threshold = -0.999
    else:
        bust_threshold = -0.999 * np.asarray(entry_prices, dtype=float)
    return {
        "n": int(len(pnls)),
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe_like": mean / std if std > 0 else float("inf") if mean > 0 else 0.0,
        "bust_rate": float((pnls <= bust_threshold).mean()),
        "win_rate": float((pnls > 0).mean()),
        "p5": float(np.percentile(pnls, 5)),
        "p95": float(np.percentile(pnls, 95)),
    }


def sell_signal_hit_rate(records: list[dict]) -> float | None:
    """Of SELL signals taken: was the realized continuation value below the bid?
    records: [{"bid_taken": float, "realized_if_held": float}, ...]"""
    if not records:
        return None
    hits = sum(1 for r in records if r["bid_taken"] >= r["realized_if_held"])
    return hits / len(records)
