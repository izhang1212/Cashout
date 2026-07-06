#!/usr/bin/env python3
"""Empirically validate (or refute) the momentum mean-reversion claim.

parlay_follower/models/nba/momentum.py bets that a detected scoring run tends
to stop -- see tests/backtest/momentum_validation.py for the full context on
what this does and does not test (score-diff reversion only; no historical
bid data exists to test the market-overshoot half of the claim).

Requires network access to the NBA Stats API (nba_api). Not runnable in
network-isolated environments -- this is a data-pull step to run once, from a
machine with internet access, before trusting the momentum feature live.

Usage:
    python3 scripts/validate_momentum.py --n-games 200 --season 2024-25
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from parlay_follower.shared.game_feed.game_state import GameState  # noqa: E402
from tests.backtest.historical_replay import _parse_clock, pull_nba_games  # noqa: E402
from tests.backtest.momentum_validation import (  # noqa: E402
    measure_run_reversion, summarize_reversion)


def _pbp_to_states(pbp_df) -> list[GameState]:
    rows = pbp_df[pbp_df["scoreHome"].notna() & (pbp_df["scoreHome"] != "")]
    states: list[GameState] = []
    for _, row in rows.iterrows():
        tau_min = _parse_clock(str(row["clock"]), int(row["period"]))
        if tau_min is None:
            continue
        try:
            h, a = int(row["scoreHome"]), int(row["scoreAway"])
        except (ValueError, TypeError):
            continue
        states.append(GameState(seconds_remaining=tau_min * 60.0,
                                home_score=h, away_score=a))
    states.sort(key=lambda s: -s.seconds_remaining)
    return states


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-games", type=int, default=200)
    ap.add_argument("--season", default="2024-25")
    ap.add_argument("--window-sec", type=float, default=150.0)
    ap.add_argument("--min-run-pts", type=int, default=7)
    args = ap.parse_args()

    print(f"Pulling {args.n_games} NBA games from season {args.season}...")
    pbp_frames = pull_nba_games(n_games=args.n_games, season=args.season)

    all_results = []
    for i, df in enumerate(pbp_frames):
        states = _pbp_to_states(df)
        if len(states) < 10:
            continue
        all_results.extend(measure_run_reversion(
            states, game_index=i, window_sec=args.window_sec,
            min_run_pts=args.min_run_pts))

    report = summarize_reversion(all_results)
    print("\n=== Momentum reversion report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if report.get("n", 0) == 0:
        print("\nNo runs detected -- check window_sec/min_run_pts or the pulled data.")
        return

    rate = report["reversion_rate"]
    lo, hi = report["reversion_rate_ci95"]
    print()
    if lo > 0.5:
        print(f"Reversion rate {rate:.2f} (95% CI [{lo:.2f}, {hi:.2f}]) is significantly "
              "above 0.5 -- the mean-reversion claim is SUPPORTED by this sample.")
    elif hi < 0.5:
        print(f"Reversion rate {rate:.2f} (95% CI [{lo:.2f}, {hi:.2f}]) is significantly "
              "BELOW 0.5 -- runs tend to CONTINUE, not revert. The momentum nudge's "
              "sign is backwards on this data; do not ship it as-is.")
    else:
        print(f"Reversion rate {rate:.2f} (95% CI [{lo:.2f}, {hi:.2f}]) straddles 0.5 -- "
              "not significantly different from a coin flip on this sample. The claim "
              "is NOT supported; treat the momentum nudge as an unvalidated prior.")


if __name__ == "__main__":
    main()
