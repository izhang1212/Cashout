"""Live MLB follower: orchestration loop for MLB parlay cash-out timing.

Structurally identical to nba/follower.py but adapted for baseball's discrete
state (innings/outs/runners) instead of a running clock.

Per tick:
  1. Fresh MLBGameState from statsapi play-by-play.
  2. Resolve legs (clinch/dead detection via MLB-specific resolvers).
  3. MLBGameContext.compute() -> per-leg model probs.
  4. Shrink each model prob toward the market-implied (TTL-cached).
  5. Get exit quote. If unavailable, alert and keep watching.
  6. DecisionEngine.recommend() -> HOLD / SELL.
  7. Log everything (score, inning, per-leg probs, synth FV).
"""
from __future__ import annotations

import time

import numpy as np

from ..account.kalshi_client import KalshiClient, Position
from ..decision.bid_model import BidModel
from ..decision.engine import DecisionEngine
from ..decision.signal import Signal
from . import feed as mlb_feed
from ..game_feed.game_state import Leg, LegStatus
from .game_state import MLBGameState, update_mlb_legs
from .stats import MLBStatsCache
from ..market_data.bid_logger import BidLogger
from ..market_data.exit_quote import get_exit_quote, liquidity_preflight
from ..market_data.orderbook import best_bid, parse_yes_bids
from ..probability.copula import CorrelationTable
from .game_context import MLBGameContext
from ..probability.monte_carlo import synthetic_fair_value
from ..probability.shrinkage import shrink
from .win_model import MLBWinModel

_MARKET_CACHE_TTL_SEC = 15.0


class MLBFollower:
    def __init__(self, *, client: KalshiClient, position: Position, legs: list[Leg],
                 game_pk: int | str, pregame_home_advantage_runs: float = 0.15,
                 settings: dict, alert_fn=print):
        self.client = client
        self.position = position
        self.legs = legs
        self.game_pk = game_pk
        self.settings = settings
        self.alert_fn = alert_fn

        m, d = settings["model"], settings["decision"]

        self.stats_cache = MLBStatsCache().load()
        self.win_model = MLBWinModel(
            stats_cache=self.stats_cache,
            pregame_home_advantage_runs=pregame_home_advantage_runs,
        )
        self.corr = CorrelationTable(default_rho=m["copula_default_rho"])
        self.bid_model = BidModel()

        # Use LSMC only (no exact DP for MLB — the DP is designed for BM / NBA)
        self.engine = DecisionEngine(
            self.win_model, self.bid_model, self.corr,
            mc_paths=m["mc_paths"],
            dt_min=1.0,           # 1 "out" per step
            risk_aversion=d["risk_aversion"],
            pregame_spread=0.0,
            robust_ensemble_size=1,
            use_exact_dp=False,   # always LSMC for MLB
        )
        self.w_market = d["shrinkage_unproven_weight"]
        self.logger = BidLogger(settings["paths"]["bid_log_dir"], str(game_pk))

        self.game_context = MLBGameContext(self.win_model, self.stats_cache)
        self._market_cache: dict[str, tuple[float, float]] = {}
        self._last_score: tuple[int, int] | None = None

    def _market_prob(self, leg: Leg) -> float | None:
        if not leg.market_ticker:
            return None
        now = time.time()
        cached = self._market_cache.get(leg.market_ticker)
        if cached and (now - cached[1]) < _MARKET_CACHE_TTL_SEC:
            return cached[0]
        try:
            levels = parse_yes_bids(self.client.get_orderbook(leg.market_ticker))
            prob = best_bid(levels) or None
            if prob is not None:
                self._market_cache[leg.market_ticker] = (prob, now)
            return prob
        except Exception:
            return None

    def preflight(self) -> bool:
        q = liquidity_preflight(self.client, self.position)
        self.alert_fn(f"[PREFLIGHT] exit {'AVAILABLE' if q.available else 'UNAVAILABLE'} "
                      f"via {q.source.value}; "
                      f"{'$%.3f/contract' % q.avg_price if q.available else 'no quote'}"
                      f"{' | ' + q.note if q.note else ''}")
        return q.available

    def run(self, poll_interval: float | None = None, require_preflight: bool = True):
        if require_preflight and not self.preflight():
            self.alert_fn("[PREFLIGHT] No exit liquidity. Will keep watching.")

        interval = poll_interval or self.settings["game_feed"].get("mlb_poll_interval_sec", 10.0)
        n_legs = max(1, len(self.legs))

        for gs in mlb_feed.poll(self.game_pk, interval_sec=interval):
            update_mlb_legs(self.legs, gs)
            statuses = tuple(l.status for l in self.legs)

            market_probs_raw: dict[str, float] = {}
            for leg in self.legs:
                pm = self._market_prob(leg)
                if pm is not None:
                    market_probs_raw[leg.leg_id] = pm

            ctx = self.game_context.compute(self.legs, gs)
            prop_probs: dict[str, float] = {}
            for leg in self.legs:
                p_model = ctx.per_leg.get(leg.leg_id, 0.5)
                pm = market_probs_raw.get(leg.leg_id)
                prop_probs[leg.leg_id] = (
                    shrink(p_model, pm, self.w_market) if pm is not None else p_model
                )

            live_market_probs = [
                market_probs_raw[leg.leg_id]
                for leg in self.legs
                if leg.status is LegStatus.LIVE and leg.leg_id in market_probs_raw
            ]
            synth_fv: float | str = ""
            if live_market_probs:
                synth_fv = round(float(
                    synthetic_fair_value(live_market_probs, self.corr.default_rho,
                                        n_paths=5000)), 4)

            quote = get_exit_quote(self.client, self.position)
            if not quote.available:
                self.alert_fn(Signal("HOLD", 0, 0, 0, 0,
                              reason=f"no exit liquidity; {n_legs} leg(s)").line())
                self._log(gs, statuses, 0.0, 0.0, "", synth_fv)
                if gs.final:
                    return
                continue

            rec = self.engine.recommend(
                self.legs, gs, quote.avg_price,
                prop_probs=prop_probs,
                momentum_score=0.0,   # no momentum model for MLB yet
            )

            ctx_note = "; ".join(ctx.notes) if ctx.notes else ""
            inning_str = f"{'T' if gs.half == 'top' else 'B'}{gs.inning}"
            base_reason = (
                f"{sum(s is LegStatus.COMPLETED for s in statuses)}/{n_legs} legs clinched, "
                f"{inning_str} {gs.outs} out(s), {gs.home_score}-{gs.away_score}, "
                f"via {rec.method}, exit={quote.source.value}"
                + (f" [{quote.note}]" if quote.note else "")
            )
            reason = f"{base_reason} | {ctx_note}" if ctx_note else base_reason

            sig = Signal(
                action="SELL" if rec.sell else "HOLD",
                margin=quote.avg_price - rec.continuation_value,
                executable_bid=quote.avg_price,
                fair_value=rec.fair_value,
                continuation_value=rec.continuation_value,
                ensemble_votes_sell=rec.ensemble_votes_sell,
                ensemble_size=rec.ensemble_size,
                reason=reason,
            )
            self.alert_fn(sig.line())
            self._log(gs, statuses, quote.avg_price, rec.fair_value,
                      ";".join(f"{k}={v:.3f}" for k, v in prop_probs.items()),
                      synth_fv)

            if sig.action == "SELL":
                return

            if any(l.status is LegStatus.FAILED for l in self.legs) or gs.final:
                return

    def _log(self, gs: MLBGameState, statuses, exit_px, fair_value,
             market_probs, synth_fv=""):
        self.logger.log(
            game_clock_sec_remaining=gs.seconds_remaining,
            score_diff=gs.score_diff,
            combo_ticker=self.position.ticker,
            combo_best_bid=exit_px,
            combo_exec_avg_px=exit_px,
            legs_total=len(self.legs),
            legs_live=sum(s is LegStatus.LIVE for s in statuses),
            legs_completed=sum(s is LegStatus.COMPLETED for s in statuses),
            model_fair_value=fair_value,
            synthetic_fair_value=synth_fv,
            per_leg_market_probs=market_probs,
        )
