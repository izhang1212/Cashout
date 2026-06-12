"""Live follower: the orchestration loop tying everything together.

ALERT-ONLY BY DEFAULT: the system advises and the human executes. It never
places or cancels orders.

Per tick:
  1. Pull fresh game state; resolve legs (clinch/dead detection).
  2. Pull combo order book -> executable proceeds (depth-weighted, not top-of-book).
  3. Pull per-leg books -> market-implied probs; apply shrinkage to model probs.
  4. Price the combo (copula Monte Carlo) -> fair value + payoff distribution.
  5. Look up the precomputed robust DP boundary -> HOLD / SELL.
  6. Log everything (bid logger feeds the haircut calibration).
"""
from __future__ import annotations

import time

from ..account.kalshi_client import KalshiClient, Position
from ..decision.bid_model import BidModel
from ..decision.robust import make_ensemble, robust_lookup, robust_solve
from ..decision.signal import Signal
from ..game_feed import nba_feed
from ..game_feed.game_state import Leg, LegStatus, update_all
from ..market_data.bid_logger import BidLogger
from ..market_data.orderbook import executable_proceeds, parse_yes_bids
from ..probability.copula import CorrelationTable
from ..probability.monte_carlo import price_combo
from ..probability.shrinkage import shrink


class LiveFollower:
    def __init__(self, *, client: KalshiClient, position: Position, legs: list[Leg],
                 game_id: str, pregame_spread: float, settings: dict,
                 alert_fn=print):
        self.client = client
        self.position = position
        self.legs = legs
        self.game_id = game_id
        self.settings = settings
        self.alert_fn = alert_fn

        m = settings["model"]
        d = settings["decision"]
        self.corr = CorrelationTable(default_rho=m["copula_default_rho"])
        self.bid_model = BidModel()  # refit from logs via `lpf fit-bid-model`
        self.ensemble = make_ensemble(
            base_sigma=m["sigma_per_min"], pregame_spread=pregame_spread,
            n_members=d["robust_ensemble_size"],
        )
        self.risk_aversion = d["risk_aversion"]
        self.dt_min = d["dp_time_step_sec"] / 60.0
        self.mc_paths = m["mc_paths"]
        self.w_market = d["shrinkage_unproven_weight"]
        self.logger = BidLogger(settings["paths"]["bid_log_dir"], game_id)
        self._dp_results = None
        self._last_score = None
        self._last_score_change_ts = 0.0

    # ---------- boundary (precomputed, refreshed when leg statuses change) ----------
    def _refresh_boundary(self, gs, q_other: float, ml_side: str, k_live: int):
        q = lambda tau: q_other  # v1: constant survival prob between refreshes
        self._dp_results = robust_solve(
            self.ensemble, self.bid_model,
            tau_start_min=gs.tau_minutes, moneyline_side=ml_side,
            q_other=q, k_live=k_live,
            dt_min=self.dt_min, risk_aversion=self.risk_aversion,
        )

    # ---------- per-leg market probs ----------
    def _market_prob(self, leg: Leg) -> float | None:
        if not leg.market_ticker:
            return None
        try:
            book = self.client.get_orderbook(leg.market_ticker)
            levels = parse_yes_bids(book)
            return levels[0].price if levels else None
        except Exception:
            return None

    # ---------- main loop ----------
    def run(self, poll_interval: float | None = None):
        interval = poll_interval or self.settings["game_feed"]["poll_interval_sec"]
        prev_statuses = None

        for gs in nba_feed.poll(self.game_id, interval_sec=interval):
            update_all(self.legs, gs)
            statuses = tuple(l.status for l in self.legs)

            # Big-event latency guard: suppress signals fired within seconds of
            # a score swing -- bids move before the poller catches up.
            if self._last_score is not None and (gs.home_score, gs.away_score) != self._last_score:
                self._last_score_change_ts = time.time()
            self._last_score = (gs.home_score, gs.away_score)
            just_after_event = (time.time() - self._last_score_change_ts) < 5.0

            # Dead combo: nothing to optimize, salvage whatever bid exists.
            if any(l.status is LegStatus.FAILED for l in self.legs):
                self.alert_fn(Signal("SELL", 0, 0, 0, 0,
                                     reason="A leg failed; combo is dead -- salvage any bid.").line())
                return

            # Shrunken per-leg probs (model -> market where edge unproven).
            prop_probs = {}
            for leg in self.legs:
                pm = self._market_prob(leg)
                if pm is not None and leg.status is LegStatus.LIVE:
                    # model prob filled in by price_combo; shrink via override
                    prop_probs[leg.leg_id] = None, pm  # placeholder pairing

            # Fair value + payoff distribution (model probs internally).
            val = price_combo(self.legs, gs, self.ensemble[0], self.corr,
                              n_paths=self.mc_paths)
            per_leg = dict(val.per_leg_probs)
            for leg_id, pair in prop_probs.items():
                _, pm = pair
                per_leg[leg_id] = shrink(per_leg[leg_id], pm, self.w_market)

            # Executable exit value from the combo book.
            try:
                book = self.client.get_orderbook(self.position.ticker)
                levels = parse_yes_bids(book)
                proceeds, avg_px = executable_proceeds(levels, self.position.contracts)
            except Exception:
                levels, proceeds, avg_px = [], 0.0, 0.0

            # Refresh the robust boundary if leg statuses changed (or first tick).
            ml = next((l for l in self.legs if l.kind == "moneyline"), None)
            live_other = [l for l in self.legs
                          if l.status is LegStatus.LIVE and l.kind != "moneyline"]
            q_other = 1.0
            for l in live_other:
                q_other *= per_leg[l.leg_id]
            if ml is not None and (statuses != prev_statuses or self._dp_results is None):
                self._refresh_boundary(gs, q_other, ml.params.get("side", "home"),
                                       k_live=1 + len(live_other))
                prev_statuses = statuses

            # Decision.
            if ml is not None and self._dp_results:
                rd = robust_lookup(self._dp_results, gs.tau_minutes, gs.score_diff)
                _, cont, _ = self._dp_results[0].lookup(gs.tau_minutes, gs.score_diff)
                sell = rd.sell and not just_after_event
                sig = Signal(
                    action="SELL" if sell else "HOLD",
                    margin=avg_px - cont,
                    executable_bid=avg_px,
                    fair_value=val.fair_value,
                    continuation_value=cont,
                    ensemble_votes_sell=rd.votes_sell,
                    ensemble_size=rd.n_members,
                    reason=("post-event cooldown" if rd.sell and just_after_event else
                            f"{sum(s is LegStatus.COMPLETED for s in statuses)} leg(s) clinched, "
                            f"{gs.tau_minutes:.1f} min left"),
                )
            else:
                sig = Signal("HOLD", 0, avg_px, val.fair_value, val.fair_value,
                             reason="no moneyline leg -> LSMC branch (not yet wired live)")

            self.alert_fn(sig.line())
            self.logger.log(
                game_clock_sec_remaining=gs.seconds_remaining,
                score_diff=gs.score_diff,
                combo_ticker=self.position.ticker,
                combo_best_bid=levels[0].price if levels else 0.0,
                combo_exec_avg_px=avg_px,
                legs_total=len(self.legs),
                legs_live=sum(s is LegStatus.LIVE for s in statuses),
                legs_completed=sum(s is LegStatus.COMPLETED for s in statuses),
                model_fair_value=val.fair_value,
                synthetic_fair_value="",
                per_leg_market_probs="",
            )

            if sig.action == "SELL":
                return  # one clean signal, then hand control back to the human
