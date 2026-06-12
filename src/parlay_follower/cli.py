"""Command-line interface.

    lpf recon                       Day-one API recon (positions, books, data contract)
    lpf positions                   List discovered Kalshi positions
    lpf backtest                    Synthetic-game policy comparison + boundary heatmap
    lpf fit-bid-model               Refit the haircut model from logged bids
    lpf follow --game-id ... --ticker ...    Live alert-only follower
"""
from __future__ import annotations

import argparse
import glob
import sys


def cmd_recon(_args):
    from scripts_shim import day_one_recon  # noqa: F401  (see scripts/day_one_recon.py)
    print("Run: python scripts/day_one_recon.py   (kept as a standalone script)")


def cmd_positions(_args):
    from .account.auth import KalshiSigner
    from .account.kalshi_client import KalshiClient
    from .config import base_url, load_creds, load_settings

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    for p in client.discover_positions():
        kind = "COMBO" if p.is_combo else "single"
        print(f"{p.ticker:40s} {kind:6s} x{p.contracts:<5d} cost=${p.cost_basis_dollars:.2f}")


def cmd_backtest(args):
    import numpy as np

    from .backtest import policies as pol
    from .backtest.metrics import summarize
    from .backtest.replay import monte_carlo_study
    from .config import load_settings
    from .decision.bid_model import BidModel
    from .decision.exact_dp import boundary_heatmap, solve
    from .probability.stern import SternModel

    settings = load_settings()
    m, d = settings["model"], settings["decision"]
    stern = SternModel(sigma_per_min=m["sigma_per_min"], pregame_spread=args.spread)
    bid_model = BidModel()
    q_other = lambda tau: args.q_other

    dp = solve(stern, bid_model, tau_start_min=48.0, moneyline_side="home",
               q_other=q_other, k_live=args.k_live,
               dt_min=d["dp_time_step_sec"] / 60.0,
               risk_aversion=d["risk_aversion"])
    boundary_heatmap(dp, "exercise_boundary.png")
    print("Saved exercise_boundary.png")

    comparison = {
        "hold_to_resolution": pol.hold_to_resolution,
        "sell_at_halftime": pol.sell_at_halftime,
        "sell_at_2x": pol.sell_at_profit_multiple(2.0),
    }
    results = monte_carlo_study(
        n_games=args.n_games, stern=stern, bid_model=bid_model,
        q_other_fn=q_other, k_live=args.k_live, entry_price=args.entry,
        policies=comparison, dp_lookup=lambda t, s: dp.lookup(t, s)[0],
    )
    print(f"\n{'policy':24s} {'mean':>8s} {'sharpe':>8s} {'bust%':>7s} {'win%':>7s}")
    for name, pnls in results.items():
        s = summarize(pnls)
        print(f"{name:24s} {s['mean_pnl']:8.4f} {s['sharpe_like']:8.3f} "
              f"{100*s['bust_rate']:6.1f}% {100*s['win_rate']:6.1f}%")
    print("\nNOTE: synthetic games test the policy UNDER THE MODEL. "
          "The paper-trading gate tests the model against reality.")


def cmd_fit_bid_model(args):
    import numpy as np
    import pandas as pd

    from .config import load_settings
    from .decision.bid_model import BidModel

    settings = load_settings()
    files = glob.glob(f"{settings['paths']['bid_log_dir']}/*.csv")
    if not files:
        sys.exit("No bid logs yet. Run the follower (or scripts/log_bids.py) on live games first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df["model_fair_value"] > 0) & (df["combo_best_bid"] > 0)]
    bm = BidModel().fit(
        tau_min=df["game_clock_sec_remaining"].to_numpy() / 60.0,
        p_combo=df["model_fair_value"].to_numpy(),
        k_live=df["legs_live"].to_numpy(),
        fv_mm=df["model_fair_value"].to_numpy(),
        observed_bid=df["combo_best_bid"].to_numpy(),
    )
    p = bm.params
    print(f"Fitted haircut: a={p.a:.4f} b={p.b:.4f} c={p.c:.4f} "
          f"residual_std={bm.residual_std:.4f}  (n={len(df)})")
    print("Update config/settings.yaml [bid_model] with these values.")
    if bm.residual_std > 0.05:
        print("residual_std is large -> consider adding the current bid as a DP state variable.")


def cmd_follow(args):
    from .account.auth import KalshiSigner
    from .account.kalshi_client import KalshiClient
    from .config import base_url, load_creds, load_settings
    from .game_feed.game_state import Leg
    from .live.alerts import loud_console_alert
    from .live.follower import LiveFollower

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    positions = {p.ticker: p for p in client.discover_positions()}
    if args.ticker not in positions:
        sys.exit(f"Ticker {args.ticker} not found in your positions: {list(positions)}")

    # v1: legs supplied on the command line until multivariate lookup is recon'd.
    # Format: kind:param=value[,param=value][@leg_market_ticker]
    legs = []
    for j, spec in enumerate(args.leg or []):
        market = ""
        if "@" in spec:
            spec, market = spec.split("@", 1)
        kind, _, params_str = spec.partition(":")
        params = {}
        for kv in filter(None, params_str.split(",")):
            k, v = kv.split("=")
            params[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
        legs.append(Leg(leg_id=f"leg{j}", kind=kind, params=params, market_ticker=market))

    LiveFollower(
        client=client, position=positions[args.ticker], legs=legs,
        game_id=args.game_id, pregame_spread=args.spread,
        settings=settings, alert_fn=loud_console_alert,
    ).run()


def main():
    p = argparse.ArgumentParser(prog="lpf", description="Live Parlay Follower")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("recon").set_defaults(fn=cmd_recon)
    sub.add_parser("positions").set_defaults(fn=cmd_positions)

    bt = sub.add_parser("backtest")
    bt.add_argument("--n-games", type=int, default=2000)
    bt.add_argument("--spread", type=float, default=-4.5, help="pregame home line")
    bt.add_argument("--q-other", type=float, default=0.6, help="survival prob of non-ML legs")
    bt.add_argument("--k-live", type=int, default=2)
    bt.add_argument("--entry", type=float, default=0.30, help="combo entry price $")
    bt.set_defaults(fn=cmd_backtest)

    sub.add_parser("fit-bid-model").set_defaults(fn=cmd_fit_bid_model)

    fl = sub.add_parser("follow")
    fl.add_argument("--ticker", required=True, help="combo ticker from `lpf positions`")
    fl.add_argument("--game-id", required=True, help="NBA game id (nba_api format)")
    fl.add_argument("--spread", type=float, required=True, help="pregame home line")
    fl.add_argument("--leg", action="append",
                    help="moneyline:side=home@TICKER | total_over:line=224.5@TICKER | "
                         "player_points_over:player=Name,line=24.5@TICKER")
    fl.set_defaults(fn=cmd_follow)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
