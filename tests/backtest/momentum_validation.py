"""Empirical validation harness for the momentum mean-reversion claim.

parlay_follower/models/nba/momentum.py's win_prob_adjustment() encodes a specific empirical bet:
that a scoring run tends to *stop* once detected, so the model should nudge
its probability toward 0.5 at the moment a run crosses the urgency threshold
(making SELL easier to trigger at what the run detector believes is a
temporary peak).

That bet is not obviously true. Hot-hand / run-persistence is a genuinely
contested question in the sports-analytics literature -- some studies find
short-run persistence rather than reversion. Nothing in this codebase, until
this module, measured whether it holds on this project's own historical data.
The 6pp maximum correction in momentum.py is a hand-tuned prior, not a fitted
parameter, and should be treated as such until validated here.

This module measures the ONLY piece of the claim testable from play-by-play
data alone (no historical market bid data exists in this repo --
data/logs/bids/ is empty): does the score-diff net swing that defines a "run"
actually shrink/reverse in the window immediately following detection,
compared to a run that just keeps going?

It does NOT (and cannot, without real market bid history) validate the
market-overshoot half of the claim -- that the Kalshi bid itself spikes above
fair value at the run's peak. That would need logged real bids bracketing
detected runs (market_data/bid_logger.py's output, once accumulated).

Usage: scripts/validate_momentum.py pulls real NBA play-by-play via
historical_replay.pull_nba_games() and calls summarize_reversion() on the
result. Requires network access to the NBA Stats API (not available in every
environment -- see that script's docstring).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parlay_follower.models.nba.momentum import MomentumDetector
from parlay_follower.shared.game_feed.game_state import GameState


@dataclass
class RunReversionResult:
    game_index: int
    hot_team: str
    run_net_pts: float          # net pts during the detected run (always positive)
    run_window_sec: float
    detected_at_sec: float      # seconds_remaining at the tick the run was flagged
    post_net_pts: float         # hot team's net pts in the post_window immediately after
    post_window_sec: float
    reverted: bool              # post_net_pts <= 0: hot team's run did not continue


def _post_window_net(states: list[GameState], start_idx: int,
                     hot_team: str, window_sec: float) -> tuple[float, float]:
    """Net pts for hot_team in the window after states[start_idx], up to window_sec."""
    if start_idx >= len(states) - 1:
        return 0.0, 0.0
    anchor = states[start_idx]
    cutoff = anchor.seconds_remaining - window_sec
    end_idx = start_idx
    for i in range(start_idx + 1, len(states)):
        if states[i].seconds_remaining < cutoff:
            break
        end_idx = i
    if end_idx == start_idx:
        return 0.0, 0.0
    elapsed = anchor.seconds_remaining - states[end_idx].seconds_remaining
    home_pts = states[end_idx].home_score - anchor.home_score
    away_pts = states[end_idx].away_score - anchor.away_score
    net = (home_pts - away_pts) if hot_team == "home" else (away_pts - home_pts)
    return float(net), float(elapsed)


def measure_run_reversion(states: list[GameState], *, game_index: int = 0,
                          window_sec: float = 150.0, min_run_pts: int = 7,
                          post_window_sec: float | None = None,
                          ) -> list[RunReversionResult]:
    """Replay states through a fresh MomentumDetector; measure post-run reversion.

    Only counts the FIRST tick of each distinct run (a run that stays flagged
    for several consecutive ticks is one event, not N events) so a long hot
    streak doesn't inflate the sample.
    """
    post_window_sec = post_window_sec if post_window_sec is not None else window_sec
    detector = MomentumDetector(window_sec=window_sec, min_run_pts=min_run_pts)

    results: list[RunReversionResult] = []
    was_running = False
    for i, gs in enumerate(states):
        detector.update(gs)
        sig = detector.signal()
        is_running = sig.run is not None
        if is_running and not was_running:
            post_net, post_elapsed = _post_window_net(
                states, i, sig.hot_team, post_window_sec)
            results.append(RunReversionResult(
                game_index=game_index,
                hot_team=sig.hot_team,
                run_net_pts=float(sig.run.net_pts),
                run_window_sec=float(sig.run.window_sec),
                detected_at_sec=float(gs.seconds_remaining),
                post_net_pts=post_net,
                post_window_sec=post_elapsed,
                reverted=post_net <= 0.0,
            ))
        was_running = is_running
    return results


def summarize_reversion(results: list[RunReversionResult]) -> dict:
    """Aggregate reversion results into a report.

    reversion_rate: fraction of detected runs where the hot team's net scoring
    in the following window was <= 0 (run stopped or opponent responded).
    A rate meaningfully above 0.5 supports the mean-reversion claim; a rate
    near or below 0.5 means the claim is not supported by this data and the
    momentum nudge should be reconsidered (shrunk or removed), not kept on
    the strength of the story alone.
    """
    if not results:
        return {"n": 0}

    reverted = np.array([r.reverted for r in results], dtype=float)
    post_net = np.array([r.post_net_pts for r in results], dtype=float)
    run_net = np.array([r.run_net_pts for r in results], dtype=float)

    rng = np.random.default_rng(0)
    boot = np.array([
        reverted[rng.integers(0, len(reverted), len(reverted))].mean()
        for _ in range(5000)
    ])

    return {
        "n": len(results),
        "reversion_rate": float(reverted.mean()),
        "reversion_rate_ci95": (float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))),
        "mean_post_net_pts": float(post_net.mean()),
        "mean_run_net_pts": float(run_net.mean()),
        "mean_continuation_ratio": float((post_net / run_net).mean()),
    }
