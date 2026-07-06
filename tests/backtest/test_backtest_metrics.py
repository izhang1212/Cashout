"""Regression test for the bust_rate fix in backtest/metrics.py.

summarize() used to flag a "bust" as pnl <= -0.999 dollars unconditionally.
Every entry price in this project is a fraction of a $1 contract (e.g. 0.30),
so the worst possible pnl (-entry_price) never got near -0.999 and bust_rate
silently read 0% regardless of the real loss rate. Fixed by scaling the
threshold to the actual entry price(s).
"""
from __future__ import annotations

import numpy as np

from .metrics import summarize


class TestBustRate:
    def test_scalar_entry_price(self):
        # entry=0.30: a full loss is pnl=-0.30, a win is pnl=+0.70.
        pnls = np.array([-0.30, -0.30, 0.70, 0.70, -0.30])
        s = summarize(pnls, entry_prices=0.30)
        assert s["bust_rate"] == 3 / 5

    def test_scalar_entry_price_ignores_partial_losses(self):
        # A partial drawdown (sold early at a small loss) is not a bust.
        pnls = np.array([-0.05, -0.30, 0.10])
        s = summarize(pnls, entry_prices=0.30)
        assert s["bust_rate"] == 1 / 3

    def test_per_game_entry_prices(self):
        # Heterogeneous entry prices (e.g. historical backtest): each game's
        # own stake, not a single global number.
        pnls          = np.array([-0.40, -0.20, 0.60, -0.35])
        entry_prices  = np.array([0.40,  0.20,  0.40,  0.50])
        # game 0: full loss of its own 0.40 stake -> bust
        # game 1: full loss of its own 0.20 stake -> bust
        # game 2: a win, not a bust
        # game 3: lost 0.35 of a 0.50 stake -- a real loss, but not a bust
        s = summarize(pnls, entry_prices=entry_prices)
        assert s["bust_rate"] == 2 / 4

    def test_without_entry_prices_falls_back_to_dollar_threshold(self):
        # Documented fallback: only correct if entry price really is ~$1.
        pnls = np.array([-1.0, -0.30, 0.70])
        s = summarize(pnls)
        assert s["bust_rate"] == 1 / 3

    def test_old_default_would_have_hidden_this_backtests_real_loss_rate(self):
        # Reproduces the exact bug: entry=0.30 across many games, ~60% of
        # which are a full loss. The unfixed code (fixed -0.999 threshold)
        # would report 0% here.
        rng = np.random.default_rng(0)
        losses = -0.30 * np.ones(600)
        wins = 0.70 * np.ones(400)
        pnls = np.concatenate([losses, wins])
        rng.shuffle(pnls)

        broken = float((pnls <= -0.999).mean())
        fixed = summarize(pnls, entry_prices=0.30)["bust_rate"]
        assert broken == 0.0
        assert fixed == 0.6
