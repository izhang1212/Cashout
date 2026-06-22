"""Live MLB follower: generalized orchestration for any MLB combo structure.

Automatically handles:
  - Same-game combos: all legs from one game (moneyline, totals, player props).
  - Cross-game totals: each leg is a "total runs over" in a different game.
  - Mixed combos: some legs from one game, others from different games.

Detection is automatic from leg params:
  - Legs with game=GAMEPK in params → routed to that specific game.
  - Legs without game= → routed to the primary game (first in game_pks).
  - Multiple distinct game PKs found → all polled simultaneously.

Correlation:
  - Same-game legs share a single correlated Poisson/BM process — use configured rho.
  - Cross-game legs from different games are independent — use rho ~0.05.
  - Mixed: average of the two (approximation; copula correlation matrix is uniform).

Per tick:
  1. Fresh MLBGameState for every active game (parallel API calls).
  2. Resolve each leg against its own game state.
  3. MLBGameContext.compute() -> per-leg model probs (handles routing internally).
  4. Shrink each model prob toward the market-implied (TTL-cached).
  5. Get exit quote. If unavailable, alert and keep watching.
  6. DecisionEngine.recommend() -> HOLD / SELL (all leg probs via prop_probs).
  7. Log everything.
"""
from __future__ import annotations

import time

from ..account.kalshi_client import KalshiClient, Position
from ..decision.bid_model import BidModel
from ..decision.engine import DecisionEngine
from ..decision.signal import Signal
from . import feed as mlb_feed
from ..game_feed.game_state import Leg, LegStatus
from .game_state import (
    MLBCrossGameState, _leg_game_pk, update_mlb_legs_generalized,
)
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
_CROSS_GAME_DEFAULT_RHO = 0.05   # near-zero for independent games


def _collect_game_pks(game_pks: list[str], legs: list[Leg]) -> list[str]:
    """Merge explicit game_pks with any game= params found in legs.

    Returns deduplicated list preserving order (explicit PKs first).
    """
    seen: dict[str, None] = {}
    for pk in game_pks:
        seen[str(pk)] = None
    for leg in legs:
        pk = _leg_game_pk(leg)
        if pk:
            seen[pk] = None
    return list(seen.keys())


def _effective_rho(all_pks: list[str], legs: list[Leg], same_game_rho: float) -> float:
    """Return a blended rho based on how many legs are cross-game vs same-game."""
    if len(all_pks) <= 1:
        return same_game_rho
    # Count legs with explicit cross-game routing
    n_cross = sum(1 for l in legs if _leg_game_pk(l) and
                  len({_leg_game_pk(l2) for l2 in legs if _leg_game_pk(l2)}) > 1)
    n_same = len(legs) - n_cross
    if n_same == 0:
        return _CROSS_GAME_DEFAULT_RHO
    # Weighted average
    return (n_same * same_game_rho + n_cross * _CROSS_GAME_DEFAULT_RHO) / len(legs)


class MLBFollower:
    def __init__(self, *, client: KalshiClient, position: Position, legs: list[Leg],
                 game_pks: list[int | str],
                 pregame_home_advantage_runs: float = 0.15,
                 settings: dict, alert_fn=print):
        self.client = client
        self.position = position
        self.legs = legs
        self.settings = settings
        self.alert_fn = alert_fn

        m, d = settings["model"], settings["decision"]

        # Auto-detect all game PKs from explicit list + leg params.
        explicit = [str(int(pk)) for pk in game_pks if str(pk).strip()]
        self.game_pks = _collect_game_pks(explicit, legs)
        self.primary_pk = self.game_pks[0] if self.game_pks else None

        # Correlation: blend same-game and cross-game rhos.
        blended_rho = _effective_rho(self.game_pks, legs,
                                      m.get("copula_default_rho", 0.35))

        self.stats_cache = MLBStatsCache().load()
        self.win_model = MLBWinModel(
            stats_cache=self.stats_cache,
            pregame_home_advantage_runs=pregame_home_advantage_runs,
            lambda_per_half_inning=m.get("mlb_lambda_per_half_inning"),
            variance_factor=m.get("mlb_variance_factor"),
        )
        self.corr = CorrelationTable(default_rho=blended_rho)
        self.bid_model = BidModel()

        self.engine = DecisionEngine(
            self.win_model, self.bid_model, self.corr,
            mc_paths=m["mc_paths"],
            dt_min=1.0,
            risk_aversion=d["risk_aversion"],
            pregame_spread=0.0,
            robust_ensemble_size=1,
            use_exact_dp=False,   # always LSMC for MLB
        )
        self.w_market = d["shrinkage_unproven_weight"]
        self.logger = BidLogger(settings["paths"]["bid_log_dir"],
                                "_".join(self.game_pks))
        self.game_context = MLBGameContext(self.win_model, self.stats_cache)
        self._market_cache: dict[str, tuple[float, float]] = {}

        n_games = len(self.game_pks)
        mode = ("cross-game" if n_games > 1 else "same-game")
        self.alert_fn(
            f"[MLB] {mode} combo | {n_games} game(s): {', '.join(self.game_pks)} | "
            f"rho={blended_rho:.2f} | {len(legs)} leg(s)"
        )

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

        for game_states in mlb_feed.poll_multiple(self.game_pks, interval_sec=interval):
            combined = MLBCrossGameState(game_states=game_states)

            update_mlb_legs_generalized(self.legs, game_states, self.primary_pk)
            statuses = tuple(l.status for l in self.legs)

            market_probs_raw: dict[str, float] = {}
            for leg in self.legs:
                pm = self._market_prob(leg)
                if pm is not None:
                    market_probs_raw[leg.leg_id] = pm

            ctx = self.game_context.compute(self.legs, game_states, self.primary_pk)
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
                self._log(combined, statuses, 0.0, 0.0, "", synth_fv)
                if combined.final:
                    return
                continue

            rec = self.engine.recommend(
                self.legs, combined, quote.avg_price,
                prop_probs=prop_probs,
                momentum_score=0.0,
            )

            ctx_note = "; ".join(ctx.notes) if ctx.notes else ""
            scores_str = ", ".join(
                f"G{i+1}:{gs.home_score}-{gs.away_score}"
                for i, gs in enumerate(game_states.values())
            )
            games_live = sum(1 for gs in game_states.values() if not gs.final)
            base_reason = (
                f"{sum(s is LegStatus.COMPLETED for s in statuses)}/{n_legs} legs clinched, "
                f"{games_live}/{len(game_states)} game(s) live, "
                f"scores=[{scores_str}], "
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
            self._log(combined, statuses, quote.avg_price, rec.fair_value,
                      ";".join(f"{k}={v:.3f}" for k, v in prop_probs.items()),
                      synth_fv)

            if sig.action == "SELL":
                return
            if any(l.status is LegStatus.FAILED for l in self.legs) or combined.final:
                return

    def _log(self, combined: MLBCrossGameState, statuses, exit_px, fair_value,
             market_probs, synth_fv=""):
        self.logger.log(
            game_clock_sec_remaining=combined.seconds_remaining,
            score_diff=0,
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
