"""Historical replay: run the decision policy against real NBA game data.

Uses actual play-by-play scores as the state sequence. Bids are synthetic
(BidModel applied to model fair value) since real Kalshi combo bids are not
available historically.

This tests whether the model's probability estimates are accurate enough to
generate edge over naive policies, using the REAL game outcomes as ground truth.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

Policy = Callable[[dict], bool]


# ─────────────────────────────────────────────────────────────────────────────
# NBA: build tick series from PlayByPlayV3 data
# ─────────────────────────────────────────────────────────────────────────────

def _parse_clock(clock_str: str, period: int) -> float | None:
    """Convert NBA clock string + period → tau_min remaining (regulation only)."""
    if period > 4:
        return None
    try:
        s = clock_str.replace("PT", "").replace("S", "")
        m, sec = s.split("M")
        elapsed = (period - 1) * 12 + (12 - float(m) - float(sec) / 60)
        return float(np.clip(48.0 - elapsed, 0.0, 48.0))
    except Exception:
        return None


def nba_ticks_from_pbp(pbp_df, stern, bid_model, *,
                        entry_prob: float | None = None,
                        q_prop: float = 0.65,
                        k_live: int = 2) -> tuple[list[dict], int] | None:
    """Build tick sequence for one NBA game from a PlayByPlayV3 DataFrame.

    Returns (ticks, combo_won) or None if the game data is unusable.
    entry_prob: if None, uses the model probability at tip-off.
    q_prop: P(prop leg wins) — represents a second leg in the synthetic combo.
    """
    rows = pbp_df[pbp_df["scoreHome"].notna() & (pbp_df["scoreHome"] != "")]
    if len(rows) < 10:
        return None

    # Build (tau_min, score_diff) series from scored plays
    points: list[tuple[float, float]] = []
    for _, row in rows.iterrows():
        tau = _parse_clock(str(row["clock"]), int(row["period"]))
        if tau is None:
            continue
        try:
            d = float(int(row["scoreHome"]) - int(row["scoreAway"]))
        except (ValueError, TypeError):
            continue
        points.append((tau, d))

    if not points:
        return None

    # Sort descending by tau (tip-off → final)
    points.sort(key=lambda x: -x[0])
    # Add tip-off if missing
    if points[0][0] < 47.9:
        points.insert(0, (48.0, 0.0))

    # Combo: moneyline (home) × prop leg resolved at game end
    final_diff = points[-1][1]
    ml_won = final_diff > 0
    prop_won = np.random.default_rng(int(abs(final_diff))).random() < q_prop
    combo_won = int(ml_won and prop_won)

    # Entry price at tip-off
    tau0, d0 = points[0]
    p_ml_0 = stern.win_prob(d0, tau0, side="home")
    entry = entry_prob if entry_prob is not None else float(p_ml_0 * q_prop)
    if entry <= 0:
        return None

    ticks = []
    for tau, d in points:
        p_ml = stern.win_prob(d, tau, side="home")
        p_prop = q_prop  # prop leg treated as fixed (not state-dependent)
        p_combo = float(p_ml * p_prop)
        bid = bid_model.bid(p_combo, tau, p_combo, k_live)
        ticks.append({
            "tau_min": tau, "score_diff": d,
            "legs_completed": 0, "legs_live": k_live,
            "executable_bid": bid, "fair_value": p_combo,
            "entry_price": entry,
        })

    return ticks, combo_won


def pull_nba_games(n_games: int = 100, season: str = "2024-25",
                   verbose: bool = True) -> list:
    """Pull completed NBA game play-by-play data. Returns list of DataFrames."""
    from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    games_df = finder.get_data_frames()[0]
    # Unique game IDs (each game appears twice, once per team)
    game_ids = games_df["GAME_ID"].unique()[:n_games * 2]
    seen: set[str] = set()

    pbp_frames = []
    for gid in game_ids:
        if gid in seen or len(pbp_frames) >= n_games:
            continue
        seen.add(gid)
        try:
            time.sleep(0.65)
            df = playbyplayv3.PlayByPlayV3(game_id=gid).get_data_frames()[0]
            if len(df) > 50:
                pbp_frames.append(df)
                if verbose and len(pbp_frames) % 25 == 0:
                    print(f"  NBA: pulled {len(pbp_frames)} games...")
        except Exception as e:
            if verbose:
                print(f"  NBA: skip {gid}: {e}")
            continue

    if verbose:
        print(f"NBA: {len(pbp_frames)} games loaded.")
    return pbp_frames


# ─────────────────────────────────────────────────────────────────────────────
# Policy runner (same as replay.py but named separately for clarity)
# ─────────────────────────────────────────────────────────────────────────────

def run_policy_on_ticks(ticks: list[dict], combo_won: int,
                         policy: Policy) -> float:
    """P&L per contract for a policy on a tick series."""
    for tick in ticks:
        if policy(tick):
            return tick["executable_bid"] - tick["entry_price"]
    return (1.0 if combo_won else 0.0) - ticks[-1]["entry_price"]


def historical_policy_comparison(tick_games: list[tuple[list[dict], int]],
                                  dp_lookup: Callable[[float, float], bool] | None,
                                  extra_policies: dict[str, Policy] | None = None,
                                  ) -> dict[str, np.ndarray]:
    """Run all standard policies on a list of (ticks, combo_won) tuples.

    Returns dict[policy_name -> P&L array].
    """
    from .policies import (hold_to_resolution, sell_on_first_leg_complete,
                           sell_at_halftime, sell_at_profit_multiple)

    policies: dict[str, Policy] = {
        "hold_to_resolution": hold_to_resolution,
        "sell_first_leg_complete": sell_on_first_leg_complete,
        "sell_at_halftime": sell_at_halftime,
        "sell_at_2x": sell_at_profit_multiple(2.0),
    }
    if extra_policies:
        policies.update(extra_policies)

    results: dict[str, list[float]] = {k: [] for k in policies}
    if dp_lookup is not None:
        results["exact_dp"] = []

    for ticks, won in tick_games:
        for name, pol in policies.items():
            results[name].append(run_policy_on_ticks(ticks, won, pol))
        if dp_lookup is not None:
            sell = False
            for tick in ticks:
                if tick["tau_min"] <= 0.0:
                    break   # terminal tick — settle at true payoff, not market bid
                if dp_lookup(tick["tau_min"], tick["score_diff"]):
                    results["exact_dp"].append(
                        tick["executable_bid"] - tick["entry_price"])
                    sell = True
                    break
            if not sell:
                results["exact_dp"].append(
                    (1.0 if won else 0.0) - ticks[-1]["entry_price"])

    return {k: np.array(v) for k, v in results.items()}
