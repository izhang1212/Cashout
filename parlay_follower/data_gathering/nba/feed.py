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


def _parse_minutes(iso_min: str) -> float:
    """NBA live API minutes format 'PT28M30.00S' -> decimal minutes (28.5)."""
    try:
        body = iso_min.removeprefix("PT")
        mins, rest = body.split("M")
        secs = rest.rstrip("S")
        return int(mins) + float(secs) / 60.0
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
        home_team_id=int(box["homeTeam"].get("teamId", 0)),
        away_team_id=int(box["awayTeam"].get("teamId", 0)),
    )
    for side, label in (("homeTeam", "home"), ("awayTeam", "away")):
        for pl in box[side].get("players", []):
            st = pl.get("statistics", {})
            name = pl.get("name", "")
            if not name:
                continue
            gs.player_stats[name] = {
                "pts": float(st.get("points", 0)),
                "reb": float(st.get("reboundsTotal", 0)),
                "ast": float(st.get("assists", 0)),
                "min": _parse_minutes(st.get("minutesCalculated", "PT00M00.00S")),
                "fouls": float(st.get("foulsPersonal", 0)),
                "team": label,
            }
    return gs


def poll(game_id: str,
         interval_sec: float | Callable[[GameState], float] = 3.0,
         until: Callable[[GameState], bool] | None = None) -> Iterator[GameState]:
    """Generator yielding fresh GameStates until the game is final (or `until`).

    interval_sec may be a fixed float or a callable (GameState -> float) for
    adaptive polling — e.g. tighter intervals in late-game crunch time.
    """
    while True:
        gs = snapshot(game_id)
        yield gs
        if gs.final or (until and until(gs)):
            return
        sleep_dur = interval_sec(gs) if callable(interval_sec) else interval_sec
        time.sleep(sleep_dur)
