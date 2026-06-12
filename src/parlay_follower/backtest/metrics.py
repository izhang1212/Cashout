"""Backtest metrics: mean P&L, Sharpe-style ratio, busts avoided, signal hit rate."""
from __future__ import annotations

import numpy as np


def summarize(pnls: np.ndarray) -> dict:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {}
    mean = float(pnls.mean())
    std = float(pnls.std(ddof=1)) if len(pnls) > 1 else 0.0
    return {
        "n": int(len(pnls)),
        "mean_pnl": mean,
        "std_pnl": std,
        "sharpe_like": mean / std if std > 0 else float("inf") if mean > 0 else 0.0,
        "bust_rate": float((pnls <= -0.999).mean()),   # per-$1-cost convention
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
