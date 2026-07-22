"""Regression tests for synthetic_game_ticks' leg-resolution modeling.

Before this fix, the "other legs" folded into q_other were priced as
perpetually still-live at their pregame probability for the entire game,
decoupled from whether they actually resolved -- legs_completed stayed 0 for
every tick, and the eventual combo_won outcome never showed up in the bid
stream a policy sees. This mirrors
tests/cpp_backtest/include/backtest_engine.hpp's prop_resolve_tick modeling:
the other legs resolve at a random in-game tick, flip legs_completed to 1 if
they hit, or end the game early (dead combo) if they don't.
"""
from __future__ import annotations

import numpy as np

from parlay_follower.cashout.bid_model import BidModel
from parlay_follower.models.nba.stern import SternModel

from .replay import synthetic_game_ticks


def _run_many(n=300, seed=7, q_other=0.6):
    stern = SternModel(sigma_per_min=2.2845, pregame_spread=-4.5)
    bm = BidModel()
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n):
        ticks, won = synthetic_game_ticks(
            stern, bm, q_other_fn=lambda tau: q_other, k_live=2,
            entry_price=0.30, rng=rng)
        games.append((ticks, won))
    return games


class TestSyntheticGameTicksLegResolution:
    def test_legs_completed_transitions_within_games(self):
        games = _run_many()
        saw_transition = any(
            any(t["legs_completed"] == 1 for t in ticks) and
            any(t["legs_completed"] == 0 for t in ticks)
            for ticks, _won in games
        )
        assert saw_transition, "legs_completed should flip 0 -> 1 mid-game in some games"

    def test_dead_combo_ends_early_and_never_wins(self):
        games = _run_many()
        dead_games = [
            (ticks, won) for ticks, won in games
            if won == 0 and (not ticks or ticks[-1]["tau_min"] > 1e-6)
        ]
        assert dead_games, "some games should end early because the other legs died"
        for ticks, won in dead_games:
            assert won == 0
            # No tick in a dead game ever reports legs_completed == 1 -- the
            # other legs died before clinching, they didn't clinch and un-clinch.
            assert all(t["legs_completed"] == 0 for t in ticks)

    def test_full_length_games_reach_the_buzzer(self):
        games = _run_many()
        full_length = [ticks for ticks, won in games if ticks and ticks[-1]["tau_min"] == 0.0]
        assert full_length, "some games should run all the way to tau=0"

    def test_bid_after_leg_completion_uses_q_other_one(self):
        # Once legs_completed flips to 1 (other legs hit), q_eff jumps to 1.0,
        # so fair_value should equal the pure moneyline win_prob from then on --
        # not p_ml * q_other, which is what every tick used before this fix.
        stern = SternModel(sigma_per_min=2.2845, pregame_spread=-4.5)
        bm = BidModel()
        rng = np.random.default_rng(7)
        found = False
        for _ in range(300):
            ticks, won = synthetic_game_ticks(
                stern, bm, q_other_fn=lambda tau: 0.6, k_live=2,
                entry_price=0.30, rng=rng)
            for t in ticks:
                if t["legs_completed"] == 1:
                    p_ml = stern.win_prob(t["score_diff"], t["tau_min"], side="home")
                    assert abs(t["fair_value"] - p_ml) < 1e-9
                    found = True
        assert found, "expected at least one post-completion tick across 300 games"
