import numpy as np

from parlay_follower.cashout.bellman.exact_dp import solve
from parlay_follower.cashout.bid_model import BidModel, HaircutParams
from parlay_follower.shared.stern import SternModel


def _solve(haircut_a=0.03, haircut_b=0.25, risk_aversion=0.0, q=0.7):
    stern = SternModel(sigma_per_min=1.7, pregame_spread=-3.0)
    bm = BidModel(HaircutParams(a=haircut_a, b=haircut_b, c=3.0))
    return solve(stern, bm, tau_start_min=48.0, moneyline_side="home",
                 q_other=lambda tau: q, k_live=2, dt_min=1.0,
                 risk_aversion=risk_aversion)


def test_value_bounded_and_monotone_in_lead():
    dp = _solve()
    assert np.all(dp.value >= -1e-9) and np.all(dp.value <= 1.0 + 1e-9)
    mid = len(dp.time_grid_min) // 2
    v = dp.value[mid]
    assert v[-1] > v[0]  # bigger home lead -> more valuable home-side combo


def test_huge_haircut_means_never_sell_early():
    # If the bid is always terrible, holding dominates almost everywhere pre-buzzer.
    dp = _solve(haircut_a=0.6, haircut_b=0.9)
    early = dp.exercise[: len(dp.time_grid_min) // 2]
    assert early.mean() < 0.05


def test_risk_aversion_expands_sell_region():
    plain = _solve(risk_aversion=0.0)
    averse = _solve(risk_aversion=3.0)
    assert averse.exercise.mean() >= plain.exercise.mean()


def test_lookup_roundtrip():
    dp = _solve()
    sell, cont, bid = dp.lookup(24.0, 5.0)
    assert isinstance(sell, bool)
    assert 0.0 <= cont <= 1.0 and 0.0 <= bid <= 1.0
