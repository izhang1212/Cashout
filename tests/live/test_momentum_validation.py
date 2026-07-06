"""Tests for the momentum reversion-measurement harness itself.

These do NOT validate the momentum feature's empirical claim (that requires
real historical data pulled via scripts/validate_momentum.py, which needs
network access this test suite doesn't assume). They validate that the
measurement code correctly distinguishes a run that reverts from one that
keeps going, on synthetic sequences with a known ground truth -- i.e. that
the harness is trustworthy once pointed at real data.
"""
from __future__ import annotations

from parlay_follower.shared.game_feed.game_state import GameState

from ..backtest.momentum_validation import (
    RunReversionResult, measure_run_reversion, summarize_reversion)

TICK_SEC = 15.0
START_REMAINING = 48 * 60.0


def _build_states(home_deltas: list[int], away_deltas: list[int]) -> list[GameState]:
    """One GameState per TICK_SEC-second tick; scores accumulate from the deltas."""
    assert len(home_deltas) == len(away_deltas)
    states = []
    h = a = 0
    for i, (dh, da) in enumerate(zip(home_deltas, away_deltas)):
        h += dh
        a += da
        states.append(GameState(seconds_remaining=START_REMAINING - i * TICK_SEC,
                                home_score=h, away_score=a))
    return states


def _flat(n: int) -> tuple[list[int], list[int]]:
    return [0] * n, [0] * n


def _scoring(n: int, pts_per_tick: int) -> list[int]:
    return [pts_per_tick] * n


class TestMeasureRunReversion:
    def test_detects_reverting_run(self):
        # 10 flat ticks (fills the window with a quiet baseline), then home
        # goes on a run. The detector fires partway *into* the run (as soon as
        # the trailing window crosses min_run_pts), so the measured post-window
        # still overlaps a few ticks of the still-ongoing home run before away
        # claws back -- the reversal has to be strong enough to outweigh that
        # overlap, which is realistic: a partial reversal can still net negative.
        flat_h, flat_a = _flat(10)
        run_h, run_a = _scoring(10, 2), _flat(10)[1]
        reversal_h, reversal_a = _flat(10)[0], _scoring(10, 5)

        states = _build_states(flat_h + run_h + reversal_h,
                               flat_a + run_a + reversal_a)
        results = measure_run_reversion(states, window_sec=150.0, min_run_pts=7)

        assert len(results) >= 1
        first = results[0]
        assert first.hot_team == "home"
        assert first.reverted is True
        assert first.post_net_pts < 0   # away outscored home in the post-window

    def test_detects_continuing_run(self):
        # Same run, but home keeps scoring at the same rate afterward instead
        # of the opponent responding.
        flat_h, flat_a = _flat(10)
        run_h, run_a = _scoring(10, 2), _flat(10)[1]
        continuation_h, continuation_a = _scoring(10, 2), _flat(10)[1]

        states = _build_states(flat_h + run_h + continuation_h,
                               flat_a + run_a + continuation_a)
        results = measure_run_reversion(states, window_sec=150.0, min_run_pts=7)

        assert len(results) >= 1
        first = results[0]
        assert first.hot_team == "home"
        assert first.reverted is False
        assert first.post_net_pts > 0

    def test_no_run_below_threshold(self):
        # Small back-and-forth scoring never crosses min_run_pts.
        h_deltas = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3
        a_deltas = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3
        states = _build_states(h_deltas, a_deltas)
        results = measure_run_reversion(states, window_sec=150.0, min_run_pts=7)
        assert results == []

    def test_one_event_per_run_not_per_tick(self):
        # A run that stays flagged for several consecutive ticks must count
        # once, not once per tick -- otherwise a long hot streak inflates the
        # sample and biases the reversion rate.
        flat_h, flat_a = _flat(5)
        run_h, run_a = _scoring(15, 2), _flat(15)[1]
        states = _build_states(flat_h + run_h, flat_a + run_a)
        results = measure_run_reversion(states, window_sec=150.0, min_run_pts=7)
        assert len(results) == 1


class TestSummarizeReversion:
    def test_aggregates_known_ground_truth(self):
        results = [
            RunReversionResult(0, "home", 10.0, 120.0, 100.0, -5.0, 150.0, True),
            RunReversionResult(0, "home", 10.0, 120.0, 100.0, -3.0, 150.0, True),
            RunReversionResult(0, "away", 8.0, 120.0, 100.0, 4.0, 150.0, False),
            RunReversionResult(0, "home", 12.0, 120.0, 100.0, 6.0, 150.0, False),
        ]
        report = summarize_reversion(results)
        assert report["n"] == 4
        assert report["reversion_rate"] == 0.5
        lo, hi = report["reversion_rate_ci95"]
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0

    def test_empty_input(self):
        assert summarize_reversion([]) == {"n": 0}
