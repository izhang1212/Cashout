"""Live follower: orchestration loop, now leg-count-agnostic and liquidity-aware.

ALERT-ONLY BY DEFAULT: the system advises; the human executes. It never places
or cancels orders.

Per tick:
  1. Fresh game state; resolve legs (clinch/dead detection).
  2. Update MomentumDetector with new game state.
  3. GameContext.compute() -> per-leg model probs (momentum + foul + player model
     + pace-aware totals).
  4. Shrink each model prob toward the market-implied prob (TTL-cached).
  5. Update the Stern drift mu from the live Kalshi moneyline market price --
     back-solving the implied mu and blending it with the pregame estimate.
     Forces the DP boundary to rebuild when mu shifts materially.
  6. Compute synthetic fair value (market-implied leg probs through the copula)
     and log it alongside every tick for haircut-model calibration.
  7. Get the exit quote (book or RFQ). If unavailable, alert and keep watching.
  8. DecisionEngine.recommend(..., momentum_score=...) -> HOLD / SELL.
  9. Log everything.

Improvements vs. the previous version
---------------------------------------
  * Game-specific sigma: two high-pace teams get a higher diffusion sigma,
    narrowing the DP's exercise boundary in the appropriate direction.
  * Live drift update: the Stern model's drift mu is back-solved from the
    Kalshi moneyline market price every tick and blended with the pregame
    estimate (early game: pregame dominates; mid/late: live market blend
    increases to ~50%).
  * Market prob TTL cache: each leg's Kalshi order-book query is cached for
    15 s to cut API traffic by ~5x without meaningfully staling the signal.
  * synthetic_fair_value logged every tick: gives the haircut model an
    explicit anchor, improving calibration.
  * momentum_score forwarded to the LSMC boundary so the n-leg optimizer
    can fire SELL sooner during a scoring run.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm as _sp_norm

from ..data_gathering.nba import feed as nba_feed
from ..data_gathering.nba.stats import NBAStatsCache
from ..models.nba.foul_model import FoulTroubleModel
from ..models.nba.game_context import GameContext
from ..models.nba.momentum import MomentumDetector
from ..cashout.bid_model import BidModel
from ..cashout.engine import DecisionEngine
from ..execution.account.kalshi_client import KalshiClient, Position
from ..execution.market_data.bid_logger import BidLogger
from ..execution.market_data.exit_quote import get_exit_quote, liquidity_preflight
from ..execution.market_data.orderbook import best_bid, parse_yes_bids
from ..cashout.pricing.copula import CorrelationTable
from ..cashout.pricing.monte_carlo import synthetic_fair_value
from ..data_gathering.nba.game_state import Leg, LegStatus, update_all
from ..models.nba.stern import SternModel
from .shrinkage import shrink
from .signal import Signal

# Market-prob order-book results are cached for this many seconds to avoid
# hammering the API with a separate request per leg per tick.
_MARKET_CACHE_TTL_SEC = 15.0

# Live mu blend ramp: after this many game-minutes elapsed the live market
# signal carries half the weight (caps at _MU_LIVE_MAX_WEIGHT).
#
# CIRCULARITY NOTE: mu_implied is back-solved FROM the same Kalshi moneyline
# price that exit_price (the SELL trigger's other side) comes from. Blending
# stern.mu toward mu_implied pulls the DP's continuation_value toward the
# market's own price -- which shrinks the model-vs-market disagreement that
# SELL edge is made of. This is consistent with this project's shrinkage
# philosophy elsewhere (biasing toward the market is "wrong" in the safe
# direction when the model is unproven), but it means _MU_LIVE_MAX_WEIGHT is
# not just a smoothing knob: it is a cap on how much genuine model-vs-market
# disagreement the DP is even allowed to see late in the game. Raising it
# toward 1.0 doesn't just track the market better, it can erase the DP's
# ability to disagree with it at all. See test_live_mu_blend.py for the
# regression guard on this (asserts residual disagreement survives the blend
# at max weight for a case with genuine pregame/live divergence).
_MU_RAMP_HALFLIFE_MIN = 20.0
_MU_LIVE_MAX_WEIGHT = 0.50

# Only update mu when the market prob is in a reliable range (not near 0 or 1).
_MU_MARKET_CLIP = (0.06, 0.94)


def _implied_mu(p_home_market: float, tau: float, score_diff: float, sigma: float) -> float:
    """Back-solve the drift implied by a market win probability.

    Inverts P(home wins) = Phi((d + mu*tau) / (sigma*sqrt(tau))) for mu.
    Clipped to +-1.0 (~a 50-pt pregame spread) to reject outliers -- a bad
    order-book read should not blow up the drift estimate.
    """
    p_clipped = float(np.clip(p_home_market, *_MU_MARKET_CLIP))
    z = float(_sp_norm.ppf(p_clipped))
    mu = (z * sigma * np.sqrt(tau) - score_diff) / tau
    return float(np.clip(mu, -1.0, 1.0))


def _blend_weight(elapsed_min: float, ramp_halflife_min: float = _MU_RAMP_HALFLIFE_MIN,
                  max_weight: float = _MU_LIVE_MAX_WEIGHT) -> float:
    """Live-market weight in the mu blend: ramps from 0 toward max_weight."""
    return min(elapsed_min / (elapsed_min + ramp_halflife_min), max_weight)


def _blend_mu(mu_pregame: float, mu_implied: float, w_live: float) -> float:
    """Blend pregame and live-implied drift. w_live is capped below 1.0 by
    construction (see _blend_weight), so this always preserves some residual
    disagreement between the model's pregame view and the market's live view
    -- it narrows the gap the DP can trade against, it doesn't close it."""
    return w_live * mu_implied + (1.0 - w_live) * mu_pregame


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

        m, d = settings["model"], settings["decision"]

        # --- load historical stats first (needed for sigma calculation) ---
        self.stats_cache = NBAStatsCache().load()

        # --- game-specific sigma from team pace data ---
        # Take one snapshot to get team IDs before the main loop starts.
        sigma = m["sigma_per_min"]
        try:
            gs0 = nba_feed.snapshot(game_id)
            sigma_adj = self.stats_cache.game_sigma(
                gs0.home_team_id, gs0.away_team_id,
                base_sigma=m["sigma_per_min"],
            )
            if abs(sigma_adj - m["sigma_per_min"]) > 0.01:
                alert_fn(f"[SIGMA] game sigma={sigma_adj:.3f} "
                         f"(base={m['sigma_per_min']:.3f}, "
                         f"home_id={gs0.home_team_id} away_id={gs0.away_team_id})")
            sigma = sigma_adj
        except Exception as exc:
            alert_fn(f"[SIGMA] preflight snapshot failed ({exc}); "
                     f"using base sigma={sigma:.3f}")

        self.stern = SternModel(sigma_per_min=sigma, pregame_spread=pregame_spread)
        self._pregame_mu = self.stern.mu    # stored for blending; mu is updated live

        self.corr = CorrelationTable(default_rho=m["copula_default_rho"])
        self.bid_model = BidModel()
        self.engine = DecisionEngine(
            self.stern, self.bid_model, self.corr,
            mc_paths=m["mc_paths"], dt_min=d["dp_time_step_sec"] / 60.0,
            risk_aversion=d["risk_aversion"], pregame_spread=pregame_spread,
            robust_ensemble_size=d.get("robust_ensemble_size", 1))
        self.w_market = d["shrinkage_unproven_weight"]
        self.logger = BidLogger(settings["paths"]["bid_log_dir"], game_id)

        # --- context models ---
        self.momentum = MomentumDetector(window_sec=150.0, min_run_pts=7)
        self.foul_model = FoulTroubleModel(self.stats_cache)
        self.game_context = GameContext(
            self.stern, self.stats_cache, self.momentum, self.foul_model)

        # --- market prob TTL cache: ticker -> (prob, wall_clock_ts) ---
        self._market_cache: dict[str, tuple[float, float]] = {}

        self._last_score: tuple[int, int] | None = None
        self._last_score_change_ts: float = 0.0

    # ---------- market prob with TTL cache ----------

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

    # ---------- live drift update ----------

    def _update_mu_from_market(self, p_home_market: float,
                                elapsed_min: float, score_diff: float) -> None:
        """Back-solve the market-implied drift and blend it with the pregame mu.

        p_home_market: Kalshi's implied P(home wins), from the moneyline market.
        elapsed_min:   minutes already played (tau_minutes subtracted from 48).
        score_diff:    current home minus away score.
        """
        tau = 48.0 - elapsed_min
        if tau < 1.0:
            return   # no meaningful signal in the last minute

        mu_implied = _implied_mu(p_home_market, tau, score_diff, self.stern.sigma)
        w_live = _blend_weight(elapsed_min)
        old_mu = self.stern.mu
        self.stern.mu = _blend_mu(self._pregame_mu, mu_implied, w_live)

        # Force boundary rebuild if drift shifted materially (>0.02 pts/min ≈ ~1 pt)
        if abs(self.stern.mu - old_mu) > 0.02:
            self.engine.invalidate_boundary()

    # ---------- liquidity preflight ----------

    def preflight(self) -> bool:
        q = liquidity_preflight(self.client, self.position)
        self.alert_fn(f"[PREFLIGHT] exit {'AVAILABLE' if q.available else 'UNAVAILABLE'} "
                      f"via {q.source.value}; "
                      f"{'$%.3f/contract' % q.avg_price if q.available else 'no quote'}"
                      f"{' | ' + q.note if q.note else ''}")
        return q.available

    # ---------- main loop ----------

    def run(self, poll_interval: float | None = None, require_preflight: bool = True):
        if require_preflight and not self.preflight():
            self.alert_fn("[PREFLIGHT] No exit liquidity right now. Will keep watching; "
                          "for combos this is normal pre-game (RFQ opens near tip-off).")

        base_interval = poll_interval or self.settings["game_feed"]["poll_interval_sec"]
        late_interval = self.settings["game_feed"].get("late_game_poll_interval_sec", 1.0)
        late_threshold = self.settings["game_feed"].get("late_game_threshold_min", 3.0)

        def _adaptive_interval(gs) -> float:
            return late_interval if gs.tau_minutes <= late_threshold else base_interval

        n_legs = max(1, len(self.legs))

        for gs in nba_feed.poll(self.game_id, interval_sec=_adaptive_interval):
            update_all(self.legs, gs)
            statuses = tuple(l.status for l in self.legs)

            # Latency guard: suppress SELL signals within 5 s of any score swing
            if self._last_score is not None and \
                    (gs.home_score, gs.away_score) != self._last_score:
                self._last_score_change_ts = time.time()
            self._last_score = (gs.home_score, gs.away_score)
            just_after_event = (time.time() - self._last_score_change_ts) < 5.0

            # Update momentum detector
            self.momentum.update(gs)

            # Fetch all per-leg market probs (TTL-cached)
            market_probs_raw: dict[str, float] = {}
            for leg in self.legs:
                pm = self._market_prob(leg)
                if pm is not None:
                    market_probs_raw[leg.leg_id] = pm

            # Live drift update: back-solve mu from the moneyline market price
            elapsed_min = 48.0 - gs.tau_minutes
            for leg in self.legs:
                if leg.kind == "moneyline" and leg.status is LegStatus.LIVE:
                    pm = market_probs_raw.get(leg.leg_id)
                    if pm and elapsed_min > 1.0:
                        side = leg.params.get("side", "home")
                        p_home = pm if side == "home" else 1.0 - pm
                        self._update_mu_from_market(p_home, elapsed_min, gs.score_diff)
                    break  # one moneyline leg is enough

            # Context-aware per-leg model probs, shrunk toward market
            ctx = self.game_context.compute(self.legs, gs)
            prop_probs: dict[str, float] = {}
            for leg in self.legs:
                p_model = ctx.per_leg.get(leg.leg_id, 0.5)
                pm = market_probs_raw.get(leg.leg_id)
                prop_probs[leg.leg_id] = (
                    shrink(p_model, pm, self.w_market) if pm is not None else p_model
                )

            # Synthetic fair value: market-implied leg probs through the copula.
            # Logged for haircut model calibration (tells us what the market thinks
            # the combo is worth, independent of our model).
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

            # Exit quote (book or RFQ)
            quote = get_exit_quote(self.client, self.position)
            if not quote.available:
                self.alert_fn(Signal("HOLD", 0, 0, 0, 0,
                              reason=f"no exit liquidity ({quote.source.value}: {quote.note}); "
                                     f"{n_legs} leg(s)").line())
                self._log(gs, statuses, 0.0, 0.0, "", synth_fv)
                if gs.final:
                    return
                continue

            rec = self.engine.recommend(
                self.legs, gs, quote.avg_price,
                prop_probs=prop_probs,
                momentum_score=ctx.momentum.sell_urgency,
            )
            sell = rec.sell and not just_after_event

            # Signal reason: base info + context notes
            ctx_note = "; ".join(ctx.notes) if ctx.notes else ""
            base_reason = (
                "post-event cooldown" if rec.sell and just_after_event else
                f"{sum(s is LegStatus.COMPLETED for s in statuses)}/{n_legs} legs clinched, "
                f"{gs.tau_minutes:.1f}m left, via {rec.method}, exit={quote.source.value}"
                + (f" [{quote.note}]" if quote.note else "")
            )
            reason = f"{base_reason} | {ctx_note}" if ctx_note else base_reason

            sig = Signal(
                action="SELL" if sell else "HOLD",
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
                return  # one clean signal; hand control to the human

            if any(l.status is LegStatus.FAILED for l in self.legs) or gs.final:
                return

    def _log(self, gs, statuses, exit_px, fair_value, market_probs, synth_fv=""):
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
