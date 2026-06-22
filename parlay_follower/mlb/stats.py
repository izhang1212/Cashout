"""Historical MLB season stats: batter averages, pitcher stats, and team offense/defense.

Fetched once at session start via the MLB Stats API (statsapi package).
Used by the player model (prop projections) and win model (team run rates).
Falls back to league-average defaults when the API is unavailable.

Stats fetched:
  Batters: AVG, OBP, SLG, HR, H, RBI, SB, AB, PA per game/season
  Pitchers: ERA, K/9, BB/9, WHIP, IP, K total per season
  Teams: runs scored per game, runs allowed per game (for win model)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import statsapi
    HAS_STATSAPI = True
except ImportError:
    HAS_STATSAPI = False

CURRENT_SEASON = 2025
_RATE_LIMIT_SEC = 0.65


@dataclass
class BatterSeasonStats:
    player_id: int
    player_name: str
    team_id: int
    avg: float           # batting average
    obp: float           # on-base percentage
    slg: float           # slugging percentage
    hr: int              # home runs
    hits: float          # hits per game
    rbi: float           # RBI per game
    sb: float            # stolen bases per game
    ab_per_game: float   # at-bats per game
    hr_rate: float       # HR per at-bat

    @property
    def total_bases_per_game(self) -> float:
        return self.slg * self.ab_per_game

    @property
    def importance(self) -> float:
        """Batting importance: OPS normalized to 0-1 scale."""
        ops = self.obp + self.slg
        return float(min(ops / 1.8, 1.0))


@dataclass
class PitcherSeasonStats:
    player_id: int
    player_name: str
    team_id: int
    era: float
    k_per_9: float
    bb_per_9: float
    whip: float
    innings_per_start: float   # average IP when starting
    k_per_game: float          # strikeouts per appearance


@dataclass
class TeamSeasonStats:
    team_id: int
    team_name: str
    runs_per_game: float       # offensive runs scored
    runs_allowed_per_game: float  # pitching+defense quality
    hr_per_game: float

    @property
    def runs_per_inning(self) -> float:
        return self.runs_per_game / 9.0


class MLBStatsCache:
    """In-memory cache of season averages. Call load() once at session start."""

    # League-average fallbacks (2024-25 season approximations)
    _DEFAULT_RUNS_PER_GAME = 4.5
    _DEFAULT_K_PER_9 = 8.7

    def __init__(self, season: int = CURRENT_SEASON):
        self.season = season
        self._batters: dict[str, BatterSeasonStats] = {}   # lowercase name -> stats
        self._pitchers: dict[str, PitcherSeasonStats] = {}
        self._teams: dict[int, TeamSeasonStats] = {}
        self._fetched = False

    def load(self) -> "MLBStatsCache":
        if self._fetched or not HAS_STATSAPI:
            if not HAS_STATSAPI:
                print("[mlb_stats] mlb-statsapi not installed; using league defaults")
            self._fetched = True
            return self
        try:
            self._fetch_teams()
            self._fetch_batters()
            self._fetch_pitchers()
            print(f"[mlb_stats] loaded {len(self._batters)} batters, "
                  f"{len(self._pitchers)} pitchers, {len(self._teams)} teams "
                  f"(season {self.season})")
        except Exception as e:
            print(f"[mlb_stats] fetch failed (league defaults in use): {e}")
        self._fetched = True
        return self

    def batter(self, name: str) -> BatterSeasonStats | None:
        return self._batters.get(name.lower())

    def pitcher(self, name: str) -> PitcherSeasonStats | None:
        return self._pitchers.get(name.lower())

    def team(self, team_id: int) -> TeamSeasonStats | None:
        return self._teams.get(team_id)

    def team_runs_per_inning(self, team_id: int) -> float:
        t = self._teams.get(team_id)
        return t.runs_per_inning if t else self._DEFAULT_RUNS_PER_GAME / 9.0

    def _fetch_teams(self) -> None:
        time.sleep(_RATE_LIMIT_SEC)
        teams = statsapi.get("teams", {"sportId": 1})  # MLB = sportId 1
        for t in teams.get("teams", []):
            tid = int(t.get("id", 0))
            # Fetch team stats for the season
            time.sleep(_RATE_LIMIT_SEC)
            try:
                ts = statsapi.get("team_stats", {
                    "teamId": tid, "season": self.season,
                    "group": "hitting,pitching", "type": "season",
                })
                hitting = {}
                pitching = {}
                for grp in ts.get("stats", []):
                    if grp.get("group", {}).get("displayName") == "hitting":
                        hitting = grp.get("splits", [{}])[0].get("stat", {})
                    elif grp.get("group", {}).get("displayName") == "pitching":
                        pitching = grp.get("splits", [{}])[0].get("stat", {})
                self._teams[tid] = TeamSeasonStats(
                    team_id=tid,
                    team_name=t.get("name", ""),
                    runs_per_game=float(hitting.get("runs", self._DEFAULT_RUNS_PER_GAME * 162)
                                        ) / 162,
                    runs_allowed_per_game=float(pitching.get("runs",
                                                             self._DEFAULT_RUNS_PER_GAME * 162)
                                               ) / 162,
                    hr_per_game=float(hitting.get("homeRuns", 1.0 * 162)) / 162,
                )
            except Exception:
                self._teams[tid] = TeamSeasonStats(
                    team_id=tid, team_name=t.get("name", ""),
                    runs_per_game=self._DEFAULT_RUNS_PER_GAME,
                    runs_allowed_per_game=self._DEFAULT_RUNS_PER_GAME,
                    hr_per_game=1.0,
                )

    def _fetch_batters(self) -> None:
        time.sleep(_RATE_LIMIT_SEC)
        data = statsapi.get("stats", {
            "stats": "season", "group": "hitting",
            "sportId": 1, "season": self.season,
            "limit": 500,
        })
        for s in data.get("stats", [{}])[0].get("splits", []):
            pl = s.get("player", {})
            name = pl.get("fullName", "")
            pid = int(pl.get("id", 0))
            team_id = int(s.get("team", {}).get("id", 0))
            st = s.get("stat", {})
            games = max(int(st.get("gamesPlayed", 1)), 1)
            ab = float(st.get("atBats", 0))
            ab_pg = ab / games
            self._batters[name.lower()] = BatterSeasonStats(
                player_id=pid,
                player_name=name,
                team_id=team_id,
                avg=float(st.get("avg", ".250").lstrip(".").__class__(st.get("avg", 0.250))
                          if isinstance(st.get("avg"), float) else 0.250),
                obp=float(st.get("obp", 0.320)),
                slg=float(st.get("slg", 0.400)),
                hr=int(st.get("homeRuns", 0)),
                hits=float(st.get("hits", 0)) / games,
                rbi=float(st.get("rbi", 0)) / games,
                sb=float(st.get("stolenBases", 0)) / games,
                ab_per_game=ab_pg,
                hr_rate=float(st.get("homeRuns", 0)) / max(ab, 1),
            )

    def _fetch_pitchers(self) -> None:
        time.sleep(_RATE_LIMIT_SEC)
        data = statsapi.get("stats", {
            "stats": "season", "group": "pitching",
            "sportId": 1, "season": self.season,
            "limit": 300,
        })
        for s in data.get("stats", [{}])[0].get("splits", []):
            pl = s.get("player", {})
            name = pl.get("fullName", "")
            pid = int(pl.get("id", 0))
            team_id = int(s.get("team", {}).get("id", 0))
            st = s.get("stat", {})
            ip = float(st.get("inningsPitched", 0) or 0)
            games = max(int(st.get("gamesPlayed", 1)), 1)
            k_total = int(st.get("strikeOuts", 0))
            self._pitchers[name.lower()] = PitcherSeasonStats(
                player_id=pid,
                player_name=name,
                team_id=team_id,
                era=float(st.get("era", 4.50)),
                k_per_9=float(st.get("strikeoutsPer9Inn", self._DEFAULT_K_PER_9)),
                bb_per_9=float(st.get("walksPer9Inn", 3.0)),
                whip=float(st.get("whip", 1.30)),
                innings_per_start=ip / games if games > 0 else 5.0,
                k_per_game=k_total / games,
            )
