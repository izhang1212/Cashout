import numpy as np

from parlay_follower.account.kalshi_client import Position
from parlay_follower.decision.bid_model import BidModel, HaircutParams
from parlay_follower.decision.engine import DecisionEngine
from parlay_follower.game_feed.game_state import GameState, Leg, LegStatus
from parlay_follower.market_data.exit_quote import (ExitSource, get_exit_quote,
                                                    liquidity_preflight)
from parlay_follower.probability.copula import CorrelationTable
from parlay_follower.probability.nleg_paths import build_nleg_boundary
from parlay_follower.nba.stern import SternModel


# ---- fake client to exercise exit-quote dispatch without the network ----
class FakeClient:
    def __init__(self, book=None, rfq=None, raise_book=False):
        self._book, self._rfq, self._raise = book, rfq, raise_book

    def get_orderbook(self, ticker, depth=10):
        if self._raise:
            raise RuntimeError("boom")
        return self._book or {"yes": []}

    def create_rfq(self, ticker, contracts):
        return self._rfq or {}


def test_single_leg_uses_order_book():
    pos = Position("KXNBA-X-ML", contracts=10, cost_basis_dollars=3.0, is_combo=False)
    client = FakeClient(book={"yes": [[45, 100]]})
    q = get_exit_quote(client, pos)
    assert q.available and q.source is ExitSource.ORDER_BOOK
    assert abs(q.avg_price - 0.45) < 1e-9


def test_combo_uses_rfq():
    pos = Position("KXNBACOMBO-Y", contracts=10, cost_basis_dollars=3.0,
                   is_combo=True, leg_tickers=["a", "b"])
    client = FakeClient(rfq={"yes_bid": 22})
    q = get_exit_quote(client, pos)
    assert q.available and q.source is ExitSource.RFQ
    assert abs(q.avg_price - 0.22) < 1e-9


def test_combo_illiquid_rfq_returns_unavailable():
    pos = Position("KXNBACOMBO-Z", contracts=10, cost_basis_dollars=3.0,
                   is_combo=True, leg_tickers=["a", "b", "c", "d"])
    client = FakeClient(rfq={})  # no quote -> "no one willing to trade"
    q = get_exit_quote(client, pos)
    assert not q.available and q.source is ExitSource.RFQ


def test_preflight_warns_on_many_legs():
    pos = Position("KXNBACOMBO-W", contracts=5, cost_basis_dollars=1.0,
                   is_combo=True, leg_tickers=["a", "b", "c", "d"])
    client = FakeClient(rfq={"price": 10})
    q = liquidity_preflight(client, pos)
    assert "4+ legs" in q.note


def _engine():
    stern = SternModel(sigma_per_min=1.7, pregame_spread=-3.0)
    bm = BidModel(HaircutParams(a=0.05, b=0.3, c=3.0))
    return DecisionEngine(stern, bm, CorrelationTable(default_rho=0.3),
                          mc_paths=3000, dt_min=1.0)


def test_engine_single_leg_uses_exact_dp():
    eng = _engine()
    gs = GameState(seconds_remaining=20 * 60, home_score=55, away_score=50)
    legs = [Leg("a", "moneyline", {"side": "home"})]
    rec = eng.recommend(legs, gs, exit_price=0.7)
    assert rec.method == "exact_dp"
    assert 0.0 <= rec.fair_value <= 1.0


def test_engine_multi_leg_uses_lsmc():
    eng = _engine()
    gs = GameState(seconds_remaining=20 * 60, home_score=55, away_score=50)
    legs = [Leg("a", "moneyline", {"side": "home"}),
            Leg("b", "player_points_over", {"player": "Star", "line": 24.5})]
    rec = eng.recommend(legs, gs, exit_price=0.5, prop_probs={"b": 0.6})
    assert rec.method == "lsmc_nleg"


def test_engine_handles_n_legs_generically():
    eng = _engine()
    gs = GameState(seconds_remaining=30 * 60, home_score=40, away_score=38)
    legs = [Leg(f"l{i}", "player_points_over", {"player": f"P{i}", "line": 20.5})
            for i in range(4)]
    pp = {f"l{i}": 0.7 for i in range(4)}
    rec = eng.recommend(legs, gs, exit_price=0.2, prop_probs=pp)
    assert rec.method == "lsmc_nleg"
    assert isinstance(rec.sell, bool)


def test_dead_leg_forces_sell_signal():
    eng = _engine()
    gs = GameState(seconds_remaining=600, home_score=80, away_score=70)
    legs = [Leg("a", "moneyline", {"side": "home"}),
            Leg("b", "total_under", {"line": 100.5}, status=LegStatus.FAILED)]
    rec = eng.recommend(legs, gs, exit_price=0.0)
    assert rec.sell and rec.method == "dead_combo"


def test_nleg_boundary_lookup_shape():
    stern = SternModel(pregame_spread=-2.0)
    bm = BidModel()
    fns = [lambda d, tau: 0.6, lambda d, tau: 0.55]
    nb = build_nleg_boundary(stern, bm, fns, rho=0.3, tau_start_min=48.0,
                             n_paths=2000, n_steps=48,
                             rng=np.random.default_rng(0))
    sell, cont = nb.should_sell(24.0, 3.0, 0, 0.33, exit_price=0.4)
    assert isinstance(sell, bool) and 0.0 <= cont <= 1.0

def test_engine_robust_mode_reports_votes():
    stern = SternModel(sigma_per_min=1.7, pregame_spread=-3.0)
    bm = BidModel(HaircutParams(a=0.05, b=0.3, c=3.0))
    eng = DecisionEngine(stern, bm, CorrelationTable(default_rho=0.3),
                         mc_paths=2000, dt_min=1.0,
                         pregame_spread=-3.0, robust_ensemble_size=5)
    gs = GameState(seconds_remaining=20 * 60, home_score=55, away_score=50)
    legs = [Leg("a", "moneyline", {"side": "home"})]
    rec = eng.recommend(legs, gs, exit_price=0.7)
    assert rec.method == "robust_dp"
    assert rec.ensemble_size == 5
    assert 0 <= rec.ensemble_votes_sell <= 5