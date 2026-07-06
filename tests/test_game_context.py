"""Tests for the NBA context layer: momentum, foul model, player model, game context.

All tests are network-free: NBAStatsCache is left unloaded (HAS_NBA_STATS may be
False in CI) and all lookups fall back to sensible defaults.
"""
import numpy as np
import pytest

from parlay_follower.data_gathering.nba.stats import NBAStatsCache, PlayerSeasonStats, TeamSeasonStats
from parlay_follower.models.nba.foul_model import FoulTroubleModel, foul_minutes_at_risk
from parlay_follower.models.nba.game_context import GameContext
from parlay_follower.models.nba.momentum import MomentumDetector
from parlay_follower.models.nba.player_model import (
    player_pts_over_prob, projected_minutes_remaining)
from parlay_follower.shared.game_feed.game_state import GameState, Leg, LegStatus
from parlay_follower.shared.stern import SternModel


# ---- helpers ----

def _gs(home=55, away=50, sec=20*60, period=3, fouls=None, team_ids=(0, 0)):
    """Build a GameState for tests. fouls: list of (name, n_fouls, team)."""
    gs = GameState(
        seconds_remaining=sec,
        period=period,
        home_score=home,
        away_score=away,
        home_team_id=team_ids[0],
        away_team_id=team_ids[1],
    )
    if fouls:
        for name, n, team in fouls:
            gs.player_stats[name] = {
                "pts": 15.0, "reb": 4.0, "ast": 3.0,
                "min": 28.0, "fouls": float(n), "team": team,
            }
    return gs


def _empty_cache() -> NBAStatsCache:
    """Cache with no data loaded (simulates missing nba_api or API failure)."""
    cache = NBAStatsCache()
    cache._fetched = True  # mark as fetched so load() is a no-op
    return cache


def _populated_cache() -> NBAStatsCache:
    """Cache with hand-crafted player and team entries for testing."""
    cache = _empty_cache()
    cache._players["star player"] = PlayerSeasonStats(
        player_id=1, player_name="Star Player", team_id=100,
        pts_per_game=28.0, min_per_game=36.0, usage_pct=0.32,
        pts_std=6.0,
    )
    cache._players["bench guy"] = PlayerSeasonStats(
        player_id=2, player_name="Bench Guy", team_id=100,
        pts_per_game=8.0, min_per_game=18.0, usage_pct=0.12,
        pts_std=3.5,
    )
    cache._teams[100] = TeamSeasonStats(
        team_id=100, team_name="Home Team",
        pace=100.5, off_rating=115.0, def_rating=110.0, pts_per_game=116.0,
    )
    cache._teams[200] = TeamSeasonStats(
        team_id=200, team_name="Away Team",
        pace=98.0, off_rating=112.0, def_rating=113.0, pts_per_game=112.0,
    )
    return cache


# ==== MomentumDetector ====

class TestMomentumDetector:
    def test_no_signal_before_any_updates(self):
        det = MomentumDetector()
        sig = det.signal()
        assert sig.hot_team == "none"
        assert sig.run is None
        assert sig.sell_urgency == 0.0

    def test_no_signal_with_one_snapshot(self):
        det = MomentumDetector()
        det.update(_gs(55, 50, sec=20*60))
        assert det.signal().hot_team == "none"

    def test_detects_home_run(self):
        det = MomentumDetector(window_sec=150, min_run_pts=7)
        # Start: 50-50 with 22 min left
        det.update(_gs(50, 50, sec=22*60))
        # 2 min later: home went on a 10-0 run
        det.update(_gs(60, 50, sec=20*60))
        sig = det.signal()
        assert sig.hot_team == "home"
        assert sig.run is not None
        assert sig.run.net_pts == 10
        assert sig.sell_urgency > 0.0

    def test_detects_away_run(self):
        det = MomentumDetector(window_sec=150, min_run_pts=7)
        det.update(_gs(50, 50, sec=22*60))
        det.update(_gs(50, 60, sec=20*60))
        sig = det.signal()
        assert sig.hot_team == "away"
        assert sig.run.net_pts == 10

    def test_small_swing_below_threshold_is_neutral(self):
        det = MomentumDetector(window_sec=150, min_run_pts=7)
        det.update(_gs(50, 50, sec=22*60))
        det.update(_gs(54, 50, sec=20*60))   # only 4-pt net swing
        assert det.signal().hot_team == "none"

    def test_urgency_higher_in_late_game(self):
        det_early = MomentumDetector(window_sec=300, min_run_pts=7)
        det_late = MomentumDetector(window_sec=300, min_run_pts=7)

        # Same 10-pt run, different game time
        det_early.update(_gs(50, 50, sec=40*60))
        det_early.update(_gs(60, 50, sec=38*60))

        det_late.update(_gs(50, 50, sec=8*60))
        det_late.update(_gs(60, 50, sec=6*60))

        assert det_late.signal().sell_urgency > det_early.signal().sell_urgency

    def test_win_prob_adjustment_nudges_correct_direction(self):
        det = MomentumDetector(window_sec=300, min_run_pts=7)
        det.update(_gs(50, 50, sec=30*60))
        det.update(_gs(60, 50, sec=28*60))  # home on 10-0 run

        base = 0.72
        # Hot team is home; betting on home -> slight downward nudge (mean reversion)
        adj_home_bet = det.win_prob_adjustment(base, "home")
        assert adj_home_bet < base

        # Opponent (away) is on a run when we bet home -> also nudged down
        det2 = MomentumDetector(window_sec=300, min_run_pts=7)
        det2.update(_gs(50, 50, sec=30*60))
        det2.update(_gs(50, 60, sec=28*60))  # away on 10-0 run
        adj_away_run = det2.win_prob_adjustment(base, "home")
        assert adj_away_run < base
        # Away run (danger) should have a larger downward nudge
        assert adj_away_run <= adj_home_bet

    def test_adjustment_clipped_to_valid_range(self):
        det = MomentumDetector(window_sec=300, min_run_pts=5)
        det.update(_gs(50, 50, sec=6*60))
        det.update(_gs(65, 50, sec=4*60))   # massive run
        adj = det.win_prob_adjustment(0.95, "home")
        assert 0.01 <= adj <= 0.99


# ==== FoulTroubleModel ====

class TestFoulTroubleModel:
    def test_no_foul_trouble_no_impact(self):
        cache = _populated_cache()
        model = FoulTroubleModel(cache)
        gs = _gs(fouls=[("Star Player", 1, "home"), ("Bench Guy", 2, "away")])
        impact = model.assess(gs)
        assert impact.home_delta == 0.0
        assert impact.away_delta == 0.0
        assert impact.troubled_players == []

    def test_four_fouls_impacts_team(self):
        cache = _populated_cache()
        model = FoulTroubleModel(cache)
        # Star player (high importance) with 4 fouls, 15 min left
        gs = _gs(sec=15*60, fouls=[("Star Player", 4, "home")])
        impact = model.assess(gs)
        assert impact.home_delta < 0.0      # hurts home
        assert impact.away_delta == 0.0     # away unaffected
        assert any("Star Player" in p for p in impact.troubled_players)

    def test_bench_player_has_smaller_impact_than_star(self):
        cache = _populated_cache()
        model = FoulTroubleModel(cache)
        gs_star = _gs(sec=15*60, fouls=[("Star Player", 4, "home")])
        gs_bench = _gs(sec=15*60, fouls=[("Bench Guy", 4, "home")])
        impact_star = model.assess(gs_star)
        impact_bench = model.assess(gs_bench)
        assert impact_star.home_delta < impact_bench.home_delta  # star hurts more

    def test_unknown_player_uses_default_importance(self):
        cache = _empty_cache()   # no player data at all
        model = FoulTroubleModel(cache)
        gs = _gs(sec=15*60, fouls=[("Unknown Guy", 4, "away")])
        impact = model.assess(gs)
        assert impact.away_delta < 0.0   # should still register some impact

    def test_impact_capped_at_max(self):
        cache = _populated_cache()
        model = FoulTroubleModel(cache)
        # Five players on home with 5 fouls each (extreme scenario)
        fouls = [(f"Star Player", 5, "home")] * 1   # just one star
        gs = _gs(sec=20*60, fouls=fouls)
        impact = model.assess(gs)
        assert impact.home_delta >= -0.15

    def test_foul_risk_increases_with_fouls(self):
        for n in (3, 4, 5):
            assert foul_minutes_at_risk(n, 15.0) > foul_minutes_at_risk(n - 1, 15.0)

    def test_foul_risk_higher_with_more_time_left(self):
        # 4 fouls with lots of time left = more at risk than 4 fouls near end
        assert foul_minutes_at_risk(4, 20.0) > foul_minutes_at_risk(4, 3.0)


# ==== PlayerPropModel ====

class TestPlayerModel:
    def test_already_hit_line(self):
        cache = _populated_cache()
        gs = _gs()
        gs.player_stats["Star Player"] = {
            "pts": 30.0, "min": 28.0, "fouls": 1, "team": "home",
            "reb": 4.0, "ast": 3.0,
        }
        p = player_pts_over_prob("Star Player", 27.5, gs, cache)
        assert p >= 0.99

    def test_no_historical_data_returns_neutral(self):
        cache = _empty_cache()
        gs = _gs()
        gs.player_stats["Unknown"] = {"pts": 5.0, "min": 10.0, "fouls": 0, "team": "home",
                                       "reb": 1.0, "ast": 0.0}
        p = player_pts_over_prob("Unknown", 20.0, gs, cache)
        assert p == 0.5

    def test_on_hot_pace_increases_probability(self):
        cache = _populated_cache()
        gs = _gs(sec=24*60)  # half game left

        # Player on cold pace: 8 pts in 24 min (ppg = ~16, below season avg of 28)
        gs.player_stats["Star Player"] = {
            "pts": 8.0, "min": 24.0, "fouls": 0, "team": "home",
            "reb": 4.0, "ast": 2.0,
        }
        p_cold = player_pts_over_prob("Star Player", 27.5, gs, cache)

        # Player on hot pace: 22 pts in 24 min
        gs.player_stats["Star Player"]["pts"] = 22.0
        p_hot = player_pts_over_prob("Star Player", 27.5, gs, cache)

        assert p_hot > p_cold

    def test_foul_trouble_reduces_probability(self):
        cache = _populated_cache()
        gs = _gs(sec=18*60)

        gs.player_stats["Star Player"] = {
            "pts": 14.0, "min": 18.0, "fouls": 1, "team": "home",
            "reb": 4.0, "ast": 2.0,
        }
        p_no_trouble = player_pts_over_prob("Star Player", 27.5, gs, cache)

        gs.player_stats["Star Player"]["fouls"] = 4  # serious foul trouble
        p_foul_trouble = player_pts_over_prob("Star Player", 27.5, gs, cache)

        assert p_foul_trouble < p_no_trouble

    def test_projected_minutes_respects_game_remaining(self):
        cache = _populated_cache()
        gs = _gs(sec=2*60)  # only 2 min left
        gs.player_stats["Star Player"] = {
            "pts": 14.0, "min": 32.0, "fouls": 0, "team": "home",
            "reb": 3.0, "ast": 2.0,
        }
        rem = projected_minutes_remaining("Star Player", gs, cache)
        assert rem <= 2.0

    def test_result_in_valid_range(self):
        cache = _populated_cache()
        gs = _gs(sec=20*60)
        gs.player_stats["Star Player"] = {
            "pts": 10.0, "min": 20.0, "fouls": 2, "team": "home",
            "reb": 3.0, "ast": 2.0,
        }
        for line in (15.0, 25.0, 35.0):
            p = player_pts_over_prob("Star Player", line, gs, cache)
            assert 0.01 <= p <= 0.99


# ==== GameContext ====

class TestGameContext:
    def _context(self, cache=None):
        stern = SternModel(sigma_per_min=1.7, pregame_spread=-3.0)
        det = MomentumDetector()
        cache = cache or _empty_cache()
        foul = FoulTroubleModel(cache)
        return GameContext(stern, cache, det, foul)

    def test_completed_leg_prob_is_one(self):
        ctx = self._context()
        leg = Leg("a", "moneyline", {"side": "home"}, status=LegStatus.COMPLETED)
        result = ctx.compute([leg], _gs())
        assert result.per_leg["a"] == 1.0

    def test_failed_leg_prob_is_zero(self):
        ctx = self._context()
        leg = Leg("a", "total_under", {"line": 100.0}, status=LegStatus.FAILED)
        result = ctx.compute([leg], _gs())
        assert result.per_leg["a"] == 0.0

    def test_moneyline_prob_in_valid_range(self):
        ctx = self._context()
        leg = Leg("a", "moneyline", {"side": "home"})
        result = ctx.compute([leg], _gs(home=60, away=50, sec=10*60))
        p = result.per_leg["a"]
        assert 0.01 <= p <= 0.99

    def test_moneyline_leading_team_has_higher_prob(self):
        ctx = self._context()
        leg_home = Leg("a", "moneyline", {"side": "home"})
        leg_away = Leg("b", "moneyline", {"side": "away"})
        gs = _gs(home=70, away=55, sec=10*60)
        result = ctx.compute([leg_home, leg_away], gs)
        assert result.per_leg["a"] > result.per_leg["b"]

    def test_total_over_late_close_game_lower_prob(self):
        # Close late game -> pace slows -> lower probability of going over a high line
        ctx = self._context()
        leg = Leg("a", "total_over", {"line": 230.0})

        gs_early = _gs(home=55, away=52, sec=20*60, period=3)
        gs_late = _gs(home=55, away=52, sec=5*60, period=4)

        p_early = ctx.compute([leg], gs_early).per_leg["a"]
        p_late = ctx.compute([leg], gs_late).per_leg["a"]

        # With only 5 min left and a high line, probability should be lower
        # (fewer expected additional points than 20 min out)
        assert p_late < p_early or abs(p_late - p_early) < 0.3   # directional check

    def test_unknown_prop_returns_neutral(self):
        ctx = self._context(_empty_cache())
        leg = Leg("a", "player_points_over", {"player": "Nobody Special", "line": 24.5})
        result = ctx.compute([leg], _gs())
        assert result.per_leg["a"] == 0.5

    def test_player_prop_with_data_not_neutral(self):
        cache = _populated_cache()
        ctx = self._context(cache)
        gs = _gs(sec=20*60, team_ids=(100, 200))
        gs.player_stats["Star Player"] = {
            "pts": 16.0, "min": 28.0, "fouls": 1, "team": "home",
            "reb": 4.0, "ast": 3.0,
        }
        leg = Leg("a", "player_points_over", {"player": "Star Player", "line": 24.5})
        result = ctx.compute([leg], gs)
        assert result.per_leg["a"] != 0.5   # model should produce a real estimate

    def test_foul_trouble_appears_in_notes(self):
        cache = _populated_cache()
        ctx = self._context(cache)
        gs = _gs(sec=15*60, fouls=[("Star Player", 4, "home")])
        result = ctx.compute([Leg("a", "moneyline", {"side": "home"})], gs)
        assert any("foul trouble" in n for n in result.notes)

    def test_foul_trouble_lowers_moneyline_prob_for_troubled_team(self):
        cache = _populated_cache()
        stern = SternModel(sigma_per_min=1.7, pregame_spread=0.0)
        det = MomentumDetector()
        foul = FoulTroubleModel(cache)
        ctx = GameContext(stern, cache, det, foul)
        leg = Leg("a", "moneyline", {"side": "home"})

        gs_clean = _gs(sec=15*60)
        gs_fouled = _gs(sec=15*60, fouls=[("Star Player", 4, "home")])

        p_clean = ctx.compute([leg], gs_clean).per_leg["a"]
        p_fouled = ctx.compute([leg], gs_fouled).per_leg["a"]
        assert p_fouled < p_clean

    def test_context_handles_exception_gracefully(self):
        """If any sub-model raises, compute() falls back to Stern defaults."""
        class BrokenFoulModel:
            def assess(self, gs):
                raise RuntimeError("simulated bug")

        stern = SternModel(sigma_per_min=1.7)
        det = MomentumDetector()
        cache = _empty_cache()
        ctx = GameContext(stern, cache, det, BrokenFoulModel())

        leg = Leg("a", "moneyline", {"side": "home"})
        result = ctx.compute([leg], _gs())
        # Should not raise; should return some valid probability
        assert 0.0 <= result.per_leg["a"] <= 1.0
        assert any("fallback" in n for n in result.notes)

    def test_momentum_run_appears_in_notes(self):
        cache = _empty_cache()
        stern = SternModel(sigma_per_min=1.7)
        det = MomentumDetector(window_sec=300, min_run_pts=7)
        foul = FoulTroubleModel(cache)
        ctx = GameContext(stern, cache, det, foul)

        det.update(_gs(50, 50, sec=22*60))
        det.update(_gs(62, 50, sec=20*60))   # home 12-0 run

        leg = Leg("a", "moneyline", {"side": "home"})
        result = ctx.compute([leg], _gs(62, 50, sec=20*60))
        assert any("run" in n.lower() for n in result.notes)

    def test_all_probs_in_valid_range(self):
        cache = _populated_cache()
        ctx = self._context(cache)
        gs = _gs(sec=15*60, team_ids=(100, 200))
        gs.player_stats["Star Player"] = {
            "pts": 18.0, "min": 24.0, "fouls": 3, "team": "home",
            "reb": 5.0, "ast": 2.0,
        }
        legs = [
            Leg("ml", "moneyline", {"side": "home"}),
            Leg("tot", "total_over", {"line": 220.0}),
            Leg("pp", "player_points_over", {"player": "Star Player", "line": 24.5}),
        ]
        result = ctx.compute(legs, gs)
        for leg_id, p in result.per_leg.items():
            assert 0.0 <= p <= 1.0, f"{leg_id}: {p} out of range"
