"""
calibrate.py — Fits NBA Stern BM sigma from real game data.

Usage:
    python scripts/calibrate.py

Outputs:
    - Fitted values printed to stdout
    - config/settings.yaml updated with model.sigma_per_min
"""
from __future__ import annotations

import time
import math
import pathlib

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
# Settings update
# ===========================================================================

def update_settings(sigma: float) -> None:
    """Read config/settings.yaml, add fitted model params, write back."""
    with open(SETTINGS_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure top-level sections exist
    if "model" not in cfg or cfg["model"] is None:
        cfg["model"] = {}

    cfg["model"]["sigma_per_min"] = round(sigma, 4)

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

    try:
        sigma = calibrate_nba(n_games=150)
    except Exception as exc:
        print(f"\nERROR during NBA calibration: {exc}")
        sigma = 1.7  # keep current default
        print(f"Using fallback sigma = {sigma}")

    # Results summary
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"{'Parameter':<40}  {'Value':>10}")
    print("-" * 52)
    print(f"{'NBA sigma_per_min (Stern BM)':<40}  {sigma:>10.4f}")
    print("=" * 60)

    # Write to settings.yaml
    update_settings(sigma=sigma)
    print("\nDone.")


if __name__ == "__main__":
    main()
