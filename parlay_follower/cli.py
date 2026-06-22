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
from re import sub
import sys
from .backtest import policies as pol
from .backtest.metrics import summarize
from .backtest.replay import monte_carlo_study, run_policy, synthetic_game_ticks
from .config import load_settings
from .decision.bid_model import BidModel
from .decision.exact_dp import boundary_heatmap, solve
from .decision.threshold_policy import grid_search
from .probability.stern import SternModel
import numpy as np


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

    # Tune the threshold baseline on a held-out batch of synthetic games, so the
    # comparison includes the floor the DP must beat (not just naive rules).
    tune_rng = np.random.default_rng(123)
    tune_games = [synthetic_game_ticks(stern, bid_model, q_other_fn=q_other,
                                       k_live=args.k_live, entry_price=args.entry,
                                       rng=tune_rng)
                  for _ in range(min(args.n_games, 800))]

    def _thresh_policy(policy):
        return lambda tick: policy.should_sell(
            tick["executable_bid"], tick["fair_value"], tick["tau_min"])

    def _replay(policy):
        pnls = [run_policy(ticks, won, args.entry, _thresh_policy(policy))
                for ticks, won in tune_games]
        arr = np.array(pnls)
        s = arr.std()
        return arr.mean() / s if s > 0 else arr.mean()

    best_thresh, _ = grid_search(_replay)
    print(f"Tuned threshold baseline: alpha={best_thresh.alpha:.2f} "
          f"beta={best_thresh.beta_minutes:.0f}m")

    comparison = {
        "threshold_baseline": (lambda tick, p=best_thresh: p.should_sell(
            tick["executable_bid"], tick["fair_value"], tick["tau_min"])),
        "hold_to_resolution": pol.hold_to_resolution,
        "sell_on_first_leg_complete": pol.sell_on_first_leg_complete,
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
def cmd_inspect(args):
    """Show a full snapshot of one combo position: size, cost, payout, cash-out, per-leg probs."""
    from .account.auth import KalshiSigner
    from .account.kalshi_client import KalshiClient
    from .config import base_url, load_creds, load_settings
    from .market_data.exit_quote import get_exit_quote
    from .market_data.orderbook import best_bid, parse_yes_bids

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    positions = {p.ticker: p for p in client.discover_positions()}
    if args.ticker not in positions:
        sys.exit(f"{args.ticker} not in positions: {list(positions)}")

    pos = positions[args.ticker]
    payout = pos.contracts * 1.00        # Kalshi contracts pay $1.00 each if they win
    pnl_if_win  = payout - pos.cost_basis_dollars

    print(f"\n{'='*55}")
    print(f"  COMBO: {pos.ticker}")
    print(f"{'='*55}")
    print(f"  Contracts held : {pos.contracts}")
    print(f"  Paid (cost)    : ${pos.cost_basis_dollars:.2f}")
    print(f"  Payout if wins : ${payout:.2f}   (P&L +${pnl_if_win:.2f})")

    # --- current exit (cash-out) value ---
    print(f"\n  --- Exit quote ---")
    q = get_exit_quote(client, pos)
    if q.available:
        pnl_now = q.proceeds - pos.cost_basis_dollars
        pnl_sign = "+" if pnl_now >= 0 else ""
        print(f"  Cash-out now   : ${q.proceeds:.2f}  "
              f"(${q.avg_price:.3f}/contract, P&L {pnl_sign}${pnl_now:.2f})")
        print(f"  Source         : {q.source.value}"
              + (f"  [{q.note}]" if q.note else ""))
    else:
        print(f"  Cash-out now   : UNAVAILABLE ({q.source.value}: {q.note})")

    # --- leg structure from the market endpoint ---
    print(f"\n  --- Legs & market-implied probabilities ---")
    try:
        market_info = client.get_market(pos.ticker)
    except Exception as e:
        market_info = {}
        print(f"  (could not fetch market metadata: {e})")

    # Kalshi may embed leg tickers under different field names — try several.
    legs_raw = (
        market_info.get("legs") or
        market_info.get("multileg_structure") or
        market_info.get("selected_markets") or
        []
    )
    if not legs_raw:
        # Print raw keys so we can identify the right field during RECON
        print(f"  RECON: market fields = {list(market_info.keys())}")
        print(f"  Re-run after confirming field name; or pass --leg tickers manually.")
    else:
        for i, leg in enumerate(legs_raw):
            leg_ticker = leg.get("ticker") or leg.get("market_ticker") or str(leg)
            side = leg.get("side", "yes")
            try:
                levels = parse_yes_bids(client.get_orderbook(leg_ticker))
                prob = best_bid(levels)
                prob_str = f"{prob*100:.1f}%" if prob else "no bids"
            except Exception as e:
                prob_str = f"error ({e})"
            print(f"  Leg {i+1}: {leg_ticker}  side={side}  implied prob ≈ {prob_str}")

    print(f"{'='*55}\n")


def cmd_check_liquidity(args):
    from .account.auth import KalshiSigner
    from .account.kalshi_client import KalshiClient
    from .config import base_url, load_creds, load_settings
    from .market_data.exit_quote import liquidity_preflight

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    positions = {p.ticker: p for p in client.discover_positions()}
    if args.ticker not in positions:
        sys.exit(f"{args.ticker} not in positions: {list(positions)}")
    q = liquidity_preflight(client, positions[args.ticker])
    print(f"exit {'AVAILABLE' if q.available else 'UNAVAILABLE'} via {q.source.value}")
    if q.available:
        print(f"  ~${q.avg_price:.3f}/contract  total ${q.proceeds:.2f}  depth_ok={q.depth_ok}")
    if q.note:
        print(f"  note: {q.note}")
    
def _parse_legs(leg_specs: list[str]) -> list:
    from .game_feed.game_state import Leg
    legs = []
    for j, spec in enumerate(leg_specs or []):
        market = ""
        if "@" in spec:
            spec, market = spec.split("@", 1)
        kind, _, params_str = spec.partition(":")
        params = {}
        for kv in filter(None, params_str.split(",")):
            k, v = kv.split("=")
            params[k] = float(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v
        # Cross-game MLB totals: rename kind so the engine falls through to prop_probs
        # (which are computed per-game by MLBGameContext.compute_cross_game).
        if kind in ("total_over", "total_under") and "game" in params:
            kind = "cross_" + kind
        legs.append(Leg(leg_id=f"leg{j}", kind=kind, params=params, market_ticker=market))
    return legs


def cmd_follow(args):
    from .account.auth import KalshiSigner
    from .account.kalshi_client import KalshiClient
    from .config import base_url, load_creds, load_settings

    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))
    positions = {p.ticker: p for p in client.discover_positions()}
    if args.ticker not in positions:
        sys.exit(f"Ticker {args.ticker} not found in your positions: {list(positions)}")

    legs = _parse_legs(args.leg)

    if args.sport == "mlb":
        from .live.alerts import loud_console_alert
        from .mlb.follower import MLBFollower
        # Start with explicitly listed game IDs, then let the follower auto-add
        # any game= params found in legs (_collect_game_pks inside MLBFollower).
        explicit_pks = [pk.strip() for pk in args.game_id.split(",") if pk.strip()]
        MLBFollower(
            client=client, position=positions[args.ticker], legs=legs,
            game_pks=explicit_pks,
            pregame_home_advantage_runs=float(args.spread) if args.spread else 0.15,
            settings=settings, alert_fn=loud_console_alert,
        ).run()
    else:
        from .live.alerts import loud_console_alert
        from .nba.follower import LiveFollower
        LiveFollower(
            client=client, position=positions[args.ticker], legs=legs,
            game_id=args.game_id, pregame_spread=args.spread or 0.0,
            settings=settings, alert_fn=loud_console_alert,
        ).run()


def cmd_historical_backtest(args):
    """Run the decision policy on real historical NBA/MLB games and compare vs benchmarks."""
    from .backtest.historical_replay import (
        historical_policy_comparison, pull_nba_games, pull_mlb_games,
        nba_ticks_from_pbp, mlb_ticks_from_innings,
    )
    from .backtest.metrics import summarize
    from .decision.bid_model import BidModel
    from .decision.exact_dp import solve
    from .probability.stern import SternModel

    settings = load_settings()
    m, d = settings["model"], settings["decision"]
    bid_model = BidModel()

    if args.sport == "nba":
        print(f"Pulling {args.n_games} NBA games ({settings.get('season','2024-25')})...")
        stern = SternModel(sigma_per_min=m["sigma_per_min"],
                           pregame_spread=args.spread)
        frames = pull_nba_games(n_games=args.n_games)
        tick_games = []
        rng = np.random.default_rng(42)
        for df in frames:
            result = nba_ticks_from_pbp(df, stern, bid_model,
                                        q_prop=args.q_prop, k_live=2)
            if result:
                tick_games.append(result)
        print(f"Built {len(tick_games)} tick series.\n")

        q_other = lambda tau: args.q_prop
        dp = solve(stern, bid_model, tau_start_min=48.0,
                   moneyline_side="home", q_other=q_other, k_live=2,
                   dt_min=d["dp_time_step_sec"] / 60.0,
                   risk_aversion=d["risk_aversion"])
        dp_lookup = lambda t, s: dp.lookup(t, s)[0]

    else:  # mlb
        print(f"Pulling {args.n_games} MLB games...")
        tick_games = []
        for innings in pull_mlb_games(n_games=args.n_games):
            result = mlb_ticks_from_innings(innings, None, bid_model, line=args.line)
            if result:
                tick_games.append(result)
        print(f"Built {len(tick_games)} tick series.\n")
        dp_lookup = None

    results = historical_policy_comparison(tick_games, dp_lookup)

    print(f"{'policy':28s} {'n':>5s} {'mean P&L':>9s} {'sharpe':>8s} "
          f"{'bust%':>7s} {'win%':>7s} {'p5':>7s} {'p95':>7s}")
    print("-" * 85)
    for name, pnls in sorted(results.items(),
                              key=lambda x: -summarize(x[1])["mean_pnl"]):
        s = summarize(pnls)
        print(f"{name:28s} {s['n']:5d} {s['mean_pnl']:+9.4f} {s['sharpe_like']:8.3f} "
              f"{100*s['bust_rate']:6.1f}% {100*s['win_rate']:6.1f}% "
              f"{s['p5']:+7.3f} {s['p95']:+7.3f}")
    print()
    if "exact_dp" in results:
        dp_mean = summarize(results["exact_dp"])["mean_pnl"]
        baselines = {k: v for k, v in results.items() if k != "exact_dp"}
        best_baseline = max(baselines, key=lambda k: summarize(baselines[k])["mean_pnl"])
        best_mean = summarize(baselines[best_baseline])["mean_pnl"]
        edge = dp_mean - best_mean
        print(f"DP edge over best baseline ({best_baseline}): {edge:+.4f} per contract")


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

    hb = sub.add_parser("historical-backtest",
                         help="Run decision policy on real historical games")
    hb.add_argument("--sport", choices=["nba", "mlb"], default="nba")
    hb.add_argument("--n-games", type=int, default=100,
                    help="Number of real games to pull (default 100)")
    hb.add_argument("--spread", type=float, default=-4.5,
                    help="NBA: pregame spread used at entry (default -4.5)")
    hb.add_argument("--q-prop", type=float, default=0.65,
                    help="Probability of the synthetic prop leg winning (default 0.65)")
    hb.add_argument("--line", type=float, default=8.5,
                    help="MLB: total runs line (default 8.5)")
    hb.set_defaults(fn=cmd_historical_backtest)

    ins = sub.add_parser("inspect", help="Show position size, cost, payout, cash-out value, and per-leg probabilities")
    ins.add_argument("--ticker", required=True, help="combo ticker from `lpf positions`")
    ins.set_defaults(fn=cmd_inspect)

    cl = sub.add_parser("check-liquidity")
    cl.add_argument("--ticker", required=True, help="position ticker to probe")
    cl.set_defaults(fn=cmd_check_liquidity)
    
    fl = sub.add_parser("follow")
    fl.add_argument("--ticker", required=True, help="combo ticker from `lpf positions`")
    fl.add_argument("--game-id", required=True,
                    help="NBA: nba_api game id. "
                         "MLB: primary gamePk (required for legs without game= param). "
                         "Multiple games are auto-detected from game= params in --leg specs; "
                         "also accepts comma-separated list, e.g. 745528,745529,745530")
    fl.add_argument("--sport", choices=["nba", "mlb"], default="nba", help="sport (default: nba)")
    fl.add_argument("--spread", type=float, default=None,
                    help="NBA: pregame home line (e.g. -4.5). MLB: home advantage in runs (default 0.15)")
    fl.add_argument("--leg", action="append",
                    help="NBA: moneyline:side=home@TICKER | total_over:line=224.5@TICKER | "
                         "player_points_over:player=Name,line=24.5@TICKER  "
                         "MLB cross-game totals: total_over:line=8.5,game=745528@TICKER  "
                         "(each leg specifies its own gamePk via game=; "
                         "moneyline and player props work the same as before)")
    fl.set_defaults(fn=cmd_follow)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
