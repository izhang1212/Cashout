from parlay_follower.shared.game_feed.game_state import (GameState, Leg, LegStatus,
                                                          resolve_leg, update_all)


def test_total_over_clinches_early():
    gs = GameState(seconds_remaining=600, home_score=120, away_score=110)
    leg = Leg("x", "total_over", {"line": 224.5})
    assert resolve_leg(leg, gs) is LegStatus.COMPLETED


def test_total_under_dies_when_crossed():
    gs = GameState(seconds_remaining=600, home_score=120, away_score=110)
    leg = Leg("x", "total_under", {"line": 224.5})
    assert resolve_leg(leg, gs) is LegStatus.FAILED


def test_moneyline_waits_for_final():
    gs = GameState(seconds_remaining=30, home_score=99, away_score=98)
    leg = Leg("x", "moneyline", {"side": "home"})
    assert resolve_leg(leg, gs) is LegStatus.LIVE
    gs.final, gs.seconds_remaining = True, 0
    assert resolve_leg(leg, gs) is LegStatus.COMPLETED


def test_player_points_clinch_and_persistence():
    gs = GameState(seconds_remaining=900, player_stats={"Star": {"pts": 26}})
    legs = [Leg("p", "player_points_over", {"player": "Star", "line": 24.5})]
    update_all(legs, gs)
    assert legs[0].status is LegStatus.COMPLETED
    gs.player_stats["Star"]["pts"] = 0  # resolved legs never un-resolve
    update_all(legs, gs)
    assert legs[0].status is LegStatus.COMPLETED
