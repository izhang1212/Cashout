"""
calibrate.py — Fits NBA Stern BM sigma and MLB overdispersion from real game data.

Usage:
    python scripts/calibrate.py

Outputs:
    - Fitted values printed to stdout
    - config/settings.yaml updated with model.sigma_per_min,
      model.mlb_lambda_per_half_inning, model.mlb_variance_factor,
      and game_feed.mlb_poll_interval_sec
"""
from __future__ import annotations

import os
import sys
import time
import math
import pathlib

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


# ===========================================================================
# NBA calibration
# ===========================================================================

def parse_nba_clock(clock_str: str, period: int) -> float | None:
    """Convert 'PT10M30.00S' + period to minutes elapsed in the game.

    Returns elapsed minutes from tip-off, or None if unparseable.
    Filters out overtime (period > 4).
    """
    if period > 4:
        return None
    if not clock_str or not clock_str.startswith("PT"):
        return None
    try:
        inner = clock_str[2:]  # remove "PT"
        m_idx = inner.index("M")
        s_idx = inner.index("S")
        minutes = float(inner[:m_idx])
        seconds = float(inner[m_idx + 1:s_idx])
        # elapsed within the period = 12 - remaining
        period_elapsed = 12.0 - minutes - seconds / 60.0
        elapsed_total = (period - 1) * 12.0 + period_elapsed
        return elapsed_total
    except (ValueError, IndexError):
        return None


def calibrate_nba(n_games: int = 150) -> float:
    """Pull completed NBA 2024-25 regular-season games, fit Stern BM sigma via MLE.

    sigma = sqrt(sum(delta_d^2) / sum(delta_t))
    """
    from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
    from nba_api.stats.endpoints.playbyplayv3 import PlayByPlayV3

    print("\n=== NBA calibration ===")
    print(f"Fetching {n_games} completed 2024-25 regular-season games...")

    finder = LeagueGameFinder(
        season_nullable="2024-25",
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    time.sleep(0.6)

    games_df = finder.get_data_frames()[0]
    # Keep unique GAME_IDs for home teams only to avoid duplicates
    games_df = games_df[games_df["MATCHUP"].str.contains(" vs\. ", regex=True)]
    game_ids = games_df["GAME_ID"].unique()[:n_games]

    print(f"Found {len(game_ids)} unique home-team game IDs. Processing...")

    sum_dd2 = 0.0   # numerator: sum of (delta_diff)^2
    sum_dt  = 0.0   # denominator: sum of delta_t in minutes
    success_count = 0

    for idx, game_id in enumerate(game_ids):
        if idx > 0 and idx % 25 == 0:
            print(f"  Progress: {idx}/{len(game_ids)} games processed "
                  f"({success_count} succeeded)")

        try:
            pbp = PlayByPlayV3(game_id=str(game_id))
            time.sleep(0.6)

            df = pbp.get_data_frames()[0]
            # Filter rows with valid scores
            df = df[
                df["scoreHome"].notna() &
                df["scoreAway"].notna() &
                (df["scoreHome"] != "") &
                (df["scoreAway"] != "")
            ].copy()

            if df.empty:
                continue

            # Compute elapsed time and score diff
            elapsed = []
            score_diff = []

            for _, row in df.iterrows():
                period = int(row["period"])
                if period > 4:
                    continue
                t = parse_nba_clock(str(row["clock"]), period)
                if t is None:
                    continue
                try:
                    d = int(row["scoreHome"]) - int(row["scoreAway"])
                except (ValueError, TypeError):
                    continue
                elapsed.append(t)
                score_diff.append(d)

            if len(elapsed) < 2:
                continue

            # Sort by elapsed time
            pairs = sorted(zip(elapsed, score_diff), key=lambda x: x[0])
            times = [p[0] for p in pairs]
            diffs = [p[1] for p in pairs]

            # Accumulate MLE numerator / denominator
            for i in range(1, len(times)):
                dt = times[i] - times[i - 1]
                dd = diffs[i] - diffs[i - 1]
                if dt > 0:
                    sum_dd2 += dd ** 2
                    sum_dt  += dt

            success_count += 1

        except Exception as exc:
            # Skip games with API or parsing errors
            pass

    print(f"NBA: {success_count}/{len(game_ids)} games used for fitting.")

    if sum_dt == 0:
        raise RuntimeError("NBA: no valid time increments collected.")

    sigma = math.sqrt(sum_dd2 / sum_dt)
    print(f"NBA fitted sigma_per_min = {sigma:.4f}")
    return sigma


# ===========================================================================
# MLB calibration
# ===========================================================================

def calibrate_mlb(n_games: int = 150) -> tuple[float, float]:
    """Pull completed MLB 2025 regular-season games and fit:
      - lambda = mean runs per half-inning
      - overdispersion = Var(runs) / Mean(runs)  [clamped to [1.0, 2.0]]

    Returns (lambda, overdispersion).
    """
    import statsapi

    print("\n=== MLB calibration ===")
    print(f"Fetching completed MLB 2025 games from Apr-Jun 2025...")

    schedule = statsapi.schedule(start_date="2025-04-01", end_date="2025-06-01")
    finished = [g for g in schedule if g.get("status") == "Final"]
    print(f"Found {len(finished)} Final games. Sampling up to {n_games}...")

    # Take first n_games
    finished = finished[:n_games]

    all_half_inning_runs: list[int] = []
    success_count = 0

    for idx, game in enumerate(finished):
        if idx > 0 and idx % 25 == 0:
            print(f"  Progress: {idx}/{len(finished)} games processed "
                  f"({success_count} succeeded)")

        gid = game.get("game_id")
        if not gid:
            continue

        try:
            data = statsapi.get("game", {"gamePk": gid})
            time.sleep(0.3)

            innings = (
                data.get("liveData", {})
                    .get("linescore", {})
                    .get("innings", [])
            )

            if not innings:
                continue

            for inn in innings:
                # home half-inning
                home_runs = inn.get("home", {}).get("runs")
                away_runs = inn.get("away", {}).get("runs")
                if home_runs is not None:
                    all_half_inning_runs.append(int(home_runs))
                if away_runs is not None:
                    all_half_inning_runs.append(int(away_runs))

            success_count += 1

        except Exception as exc:
            pass

    print(f"MLB: {success_count}/{len(finished)} games used for fitting.")
    print(f"MLB: {len(all_half_inning_runs)} half-inning run samples collected.")

    if not all_half_inning_runs:
        raise RuntimeError("MLB: no half-inning run samples collected.")

    samples = np.array(all_half_inning_runs, dtype=float)
    mu = float(np.mean(samples))
    var_empirical = float(np.var(samples, ddof=1))

    # Poisson: Var = Mean, so ratio > 1 means overdispersed
    overdispersion = var_empirical / mu
    overdispersion = float(np.clip(overdispersion, 1.0, 2.0))

    print(f"MLB fitted lambda_per_half_inning = {mu:.4f}")
    print(f"MLB fitted overdispersion (variance factor) = {overdispersion:.4f}")
    return mu, overdispersion


# ===========================================================================
# Settings update
# ===========================================================================

def update_settings(sigma: float, mlb_lambda: float, mlb_variance: float) -> None:
    """Read config/settings.yaml, add fitted model params, write back."""
    with open(SETTINGS_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure top-level sections exist
    if "model" not in cfg or cfg["model"] is None:
        cfg["model"] = {}
    if "game_feed" not in cfg or cfg["game_feed"] is None:
        cfg["game_feed"] = {}

    cfg["model"]["sigma_per_min"] = round(sigma, 4)
    cfg["model"]["mlb_lambda_per_half_inning"] = round(mlb_lambda, 4)
    cfg["model"]["mlb_variance_factor"] = round(mlb_variance, 4)

    # Add MLB poll interval if absent
    if "mlb_poll_interval_sec" not in cfg["game_feed"]:
        cfg["game_feed"]["mlb_poll_interval_sec"] = 10

    with open(SETTINGS_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nconfig/settings.yaml updated at: {SETTINGS_PATH}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("Parlay Follower — Model Calibration Script")
    print("=" * 60)

    # NBA
    try:
        sigma = calibrate_nba(n_games=150)
    except Exception as exc:
        print(f"\nERROR during NBA calibration: {exc}")
        sigma = 1.7  # keep current default
        print(f"Using fallback sigma = {sigma}")

    # MLB
    try:
        mlb_lambda, mlb_variance = calibrate_mlb(n_games=150)
    except Exception as exc:
        print(f"\nERROR during MLB calibration: {exc}")
        mlb_lambda = 0.473   # league average ~0.47 runs/half-inning
        mlb_variance = 1.2   # current hardcoded value
        print(f"Using fallback lambda = {mlb_lambda}, variance_factor = {mlb_variance}")

    # Results summary
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"{'Parameter':<40}  {'Value':>10}")
    print("-" * 52)
    print(f"{'NBA sigma_per_min (Stern BM)':<40}  {sigma:>10.4f}")
    print(f"{'MLB lambda_per_half_inning':<40}  {mlb_lambda:>10.4f}")
    print(f"{'MLB variance_factor (overdispersion)':<40}  {mlb_variance:>10.4f}")
    print("=" * 60)

    # Write to settings.yaml
    update_settings(sigma=sigma, mlb_lambda=mlb_lambda, mlb_variance=mlb_variance)
    print("\nDone.")


if __name__ == "__main__":
    main()
