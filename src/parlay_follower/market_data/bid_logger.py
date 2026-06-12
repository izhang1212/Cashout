"""Persistent logger: live combo/leg bids alongside synchronized game state.

This growing dataset calibrates the empirical bid (haircut) model and is one of
the most valuable assets the project produces. Append-only CSV per game.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

FIELDS = [
    "ts", "game_clock_sec_remaining", "score_diff",
    "combo_ticker", "combo_best_bid", "combo_exec_avg_px",
    "legs_total", "legs_live", "legs_completed",
    "model_fair_value", "synthetic_fair_value", "per_leg_market_probs",
]


class BidLogger:
    def __init__(self, log_dir: str | Path, game_id: str):
        self.path = Path(log_dir) / f"{game_id}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def log(self, **kw) -> None:
        row = [kw.get(k, "") for k in FIELDS]
        row[0] = row[0] or f"{time.time():.3f}"
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
