"""Live MLB feed: polls the MLB Stats API for game state and play-by-play.

Uses `statsapi` (mlb-statsapi package, free, no auth). The live game endpoint
returns the full linescore (inning, outs, runners on base, score) plus the
complete play-by-play for the current game. This is checked every tick to keep
the MLBGameState current.

Endpoint used:
    statsapi.get('game', {'gamePk': game_pk, 'fields': 'liveData,gameData'})

Key fields consumed:
    liveData.linescore.currentInning        -- inning number
    liveData.linescore.inningHalf           -- "Top" | "Bottom"
    liveData.linescore.outs                 -- outs in current half-inning
    liveData.linescore.teams.home.runs      -- home score
    liveData.linescore.teams.away.runs      -- away score
    liveData.linescore.offense.{first,second,third}  -- runner on base?
    liveData.boxscore.teams.{home,away}.players      -- player stats
    gameData.status.abstractGameState       -- "Live" | "Final" | "Preview"
"""
from __future__ import annotations

import time
from collections.abc import Iterator

from .game_state import MLBGameState

try:
    import statsapi
    HAS_STATSAPI = True
except ImportError:
    HAS_STATSAPI = False


def snapshot(game_pk: int | str) -> MLBGameState:
    """One-shot pull of the current MLB game state."""
    if not HAS_STATSAPI:
        raise RuntimeError("mlb-statsapi not installed: pip install mlb-statsapi")

    data = statsapi.get("game", {
        "gamePk": int(game_pk),
        "fields": "liveData,gameData",
    })

    live = data.get("liveData", {})
    game_data = data.get("gameData", {})

    status = game_data.get("status", {}).get("abstractGameState", "Preview")
    final = status == "Final"

    ls = live.get("linescore", {})
    inning = int(ls.get("currentInning", 1))
    half_raw = ls.get("inningHalf", "Top")
    half = "top" if half_raw.lower() == "top" else "bottom"
    outs = int(ls.get("outs", 0))

    teams = ls.get("teams", {})
    home_score = int(teams.get("home", {}).get("runs", 0))
    away_score = int(teams.get("away", {}).get("runs", 0))

    offense = ls.get("offense", {})
    runners = 0
    if offense.get("first"):
        runners |= 0b001
    if offense.get("second"):
        runners |= 0b010
    if offense.get("third"):
        runners |= 0b100

    # Team IDs
    home_team = game_data.get("teams", {}).get("home", {})
    away_team = game_data.get("teams", {}).get("away", {})
    home_team_id = int(home_team.get("id", 0))
    away_team_id = int(away_team.get("id", 0))

    gs = MLBGameState(
        game_id=str(game_pk),
        inning=inning,
        half=half,
        outs=outs,
        runners=runners,
        home_score=home_score,
        away_score=away_score,
        final=final,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )

    # Player stats from box score
    box = live.get("boxscore", {}).get("teams", {})
    for side, label in (("home", "home"), ("away", "away")):
        team_box = box.get(side, {})
        # Current pitcher
        bullpen = team_box.get("bullpen", [])
        pitchers = team_box.get("pitchers", [])
        all_players = team_box.get("players", {})

        if side == "home":
            gs.away_pitcher = _current_pitcher(pitchers, all_players)
        else:
            gs.home_pitcher = _current_pitcher(pitchers, all_players)

        for pid_str, p in all_players.items():
            name = p.get("person", {}).get("fullName", "")
            if not name:
                continue
            batting = p.get("stats", {}).get("batting", {})
            pitching = p.get("stats", {}).get("pitching", {})
            gs.player_stats[name] = {
                "hits": int(batting.get("hits", 0)),
                "hr": int(batting.get("homeRuns", 0)),
                "tb": int(batting.get("totalBases", 0)),
                "rbi": int(batting.get("rbi", 0)),
                "sb": int(batting.get("stolenBases", 0)),
                "ab": int(batting.get("atBats", 0)),
                "k": int(pitching.get("strikeOuts", 0)),
                "team": label,
            }

    return gs


def _current_pitcher(pitcher_ids: list[int], all_players: dict) -> str:
    """Return the full name of the last pitcher listed (active pitcher)."""
    if not pitcher_ids:
        return ""
    last_id = pitcher_ids[-1]
    p_key = f"ID{last_id}"
    return all_players.get(p_key, {}).get("person", {}).get("fullName", "")


def poll(game_pk: int | str, interval_sec: float = 10.0) -> Iterator[MLBGameState]:
    """Generator yielding fresh MLBGameStates until the game is final.

    MLB is slower-paced than NBA so a 10-second default poll interval is fine;
    tighten to 5 s if you want to catch pitching changes faster.
    """
    while True:
        gs = snapshot(game_pk)
        yield gs
        if gs.final:
            return
        time.sleep(interval_sec)
