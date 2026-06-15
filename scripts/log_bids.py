#!/usr/bin/env python
"""Standalone bid logger: record live combo + leg bids alongside game state for
games you are NOT holding a position in. This grows the haircut-calibration
dataset faster than only logging your own games.

Usage:
    python scripts/log_bids.py --game-id 0042500404 --combo-ticker KXNBACOMBO-... \
        --leg-ticker KXNBA-...-ML --leg-ticker KXNBA-...-TOTAL --spread -4.5
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parlay_follower.account.auth import KalshiSigner            # noqa: E402
from parlay_follower.account.kalshi_client import KalshiClient    # noqa: E402
from parlay_follower.config import base_url, load_creds, load_settings  # noqa: E402
from parlay_follower.game_feed import nba_feed                    # noqa: E402
from parlay_follower.market_data.bid_logger import BidLogger      # noqa: E402
from parlay_follower.market_data.orderbook import best_bid, parse_yes_bids  # noqa: E402
from parlay_follower.probability.monte_carlo import synthetic_fair_value    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--combo-ticker", required=True)
    ap.add_argument("--leg-ticker", action="append", default=[])
    ap.add_argument("--spread", type=float, default=0.0)
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    logger = BidLogger(settings["paths"]["bid_log_dir"], f"{args.game_id}_{int(time.time())}")
    rho = settings["model"]["copula_default_rho"]

    for gs in nba_feed.poll(args.game_id, interval_sec=args.interval):
        try:
            combo_bid = best_bid(parse_yes_bids(client.get_orderbook(args.combo_ticker)))
        except Exception:
            combo_bid = 0.0
        leg_probs = []
        for lt in args.leg_ticker:
            try:
                leg_probs.append(best_bid(parse_yes_bids(client.get_orderbook(lt))))
            except Exception:
                leg_probs.append(0.0)
        fv_mm = synthetic_fair_value(leg_probs, rho) if leg_probs else 0.0

        logger.log(
            game_clock_sec_remaining=gs.seconds_remaining,
            score_diff=gs.score_diff,
            combo_ticker=args.combo_ticker,
            combo_best_bid=combo_bid,
            combo_exec_avg_px=combo_bid,
            legs_total=len(args.leg_ticker),
            legs_live=len(args.leg_ticker),
            legs_completed=0,
            model_fair_value=fv_mm,
            synthetic_fair_value=fv_mm,
            per_leg_market_probs=";".join(f"{p:.3f}" for p in leg_probs),
        )
        print(f"tau={gs.tau_minutes:5.1f}m diff={gs.score_diff:+3d} "
              f"combo_bid=${combo_bid:.3f} fv_mm=${fv_mm:.3f}")


if __name__ == "__main__":
    main()
