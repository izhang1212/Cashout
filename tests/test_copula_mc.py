import numpy as np

from parlay_follower.game_feed.game_state import GameState, Leg, LegStatus
from parlay_follower.probability.copula import CorrelationTable
from parlay_follower.probability.monte_carlo import price_combo, synthetic_fair_value
from parlay_follower.probability.stern import SternModel


def test_independent_legs_match_product():
    fv = synthetic_fair_value([0.6, 0.5], rho=0.0, n_paths=200000,
                              rng=np.random.default_rng(1))
    assert abs(fv - 0.30) < 0.01


def test_positive_correlation_raises_joint_prob():
    rng = np.random.default_rng(2)
    fv_ind = synthetic_fair_value([0.6, 0.5], rho=0.0, n_paths=200000, rng=rng)
    fv_cor = synthetic_fair_value([0.6, 0.5], rho=0.6, n_paths=200000,
                                  rng=np.random.default_rng(2))
    assert fv_cor > fv_ind


def test_failed_leg_kills_combo():
    gs = GameState(seconds_remaining=600, home_score=80, away_score=75)
    legs = [
        Leg("a", "moneyline", {"side": "home"}),
        Leg("b", "total_under", {"line": 150.5}, status=LegStatus.FAILED),
    ]
    val = price_combo(legs, gs, SternModel(), CorrelationTable(), n_paths=1000)
    assert val.fair_value == 0.0


def test_clinched_legs_pay_one():
    gs = GameState(seconds_remaining=0, home_score=100, away_score=90, final=True)
    legs = [Leg("a", "moneyline", {"side": "home"}, status=LegStatus.COMPLETED)]
    val = price_combo(legs, gs, SternModel(), CorrelationTable(), n_paths=1000)
    assert val.fair_value == 1.0
