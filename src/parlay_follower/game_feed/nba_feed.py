"""Live NBA feed: polls play-by-play / box score and maintains the GameState.

Uses `nba_api` (free). The import is guarded so the rest of the package works
in environments without it (tests, backtests on cached data).

Latency note: a few seconds of lag is acceptable for this decision horizon,
but clutch scoring events can move bids before a poller catches up -- prefer
the shortest safe poll interval and treat signals fired within seconds of a
major event with extra caution (handled in live/follower.py).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

from .game_state import REGULATION_SEC, GameState

try:
    from nba_api.live.nba.endpoints import boxscore as _boxscore
    HAS_NBA_API = True
except ImportError:  # pragma: no cover
    HAS_NBA_API = False


def _parse_clock(iso_clock: str) -> float:
    """NBA live API clock format 'PT06M12.00S' -> seconds remaining in period."""
    try:
        body = iso_clock.removeprefix("PT")
        mins, rest = body.split("M")
        secs = rest.rstrip("S")
        return int(mins) * 60 + float(secs)
    except Exception:
        return 0.0


def snapshot(game_id: str) -> GameState:
    """One-shot pull of the current game state."""
    if not HAS_NBA_API:
        raise RuntimeError("nba_api not installed: pip install nba_api")
    box = _boxscore.BoxScore(game_id).get_dict()["game"]

    period = int(box.get("period", 1))
    clock_in_period = _parse_clock(box.get("gameClock", "PT00M00.00S"))
    periods_left_after = max(0, 4 - period)
    seconds_remaining = clock_in_period + periods_left_after * 12 * 60
    status = int(box.get("gameStatus", 1))  # 1 scheduled, 2 live, 3 final

    gs = GameState(
        seconds_remaining=min(seconds_remaining, REGULATION_SEC),
        period=period,
        home_score=int(box["homeTeam"]["score"]),
        away_score=int(box["awayTeam"]["score"]),
        final=(status == 3),
    )
    for side in ("homeTeam", "awayTeam"):
        for pl in box[side].get("players", []):
            st = pl.get("statistics", {})
            gs.player_stats[pl.get("name", "")] = {
                "pts": float(st.get("points", 0)),
                "reb": float(st.get("reboundsTotal", 0)),
                "ast": float(st.get("assists", 0)),
                "min": 0.0,
                "fouls": float(st.get("foulsPersonal", 0)),
            }
    return gs


def poll(game_id: str, interval_sec: float = 3.0,
         until: Callable[[GameState], bool] | None = None) -> Iterator[GameState]:
    """Generator yielding fresh GameStates until the game is final (or `until`)."""
    while True:
        gs = snapshot(game_id)
        yield gs
        if gs.final or (until and until(gs)):
            return
        time.sleep(interval_sec)
