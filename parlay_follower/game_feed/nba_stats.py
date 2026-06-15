"""Historical NBA season stats: player averages + team pace/ratings.

Fetched once at session start via nba_api stats endpoints (public, no auth).
Consumed by the foul model (player importance weighting) and the player prop
model (projection baseline). Falls back silently to neutral defaults when the
API is unavailable so nothing downstream crashes.

Upgrade path: cache to disk (pickle/parquet) so repeated runs skip the fetch;
switch to a proper data warehouse (BigQuery, S3) when logging at scale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:
    from nba_api.stats.endpoints import (
        LeagueDashPlayerStats,
        LeagueDashTeamStats,
    )
    HAS_NBA_STATS = True
except ImportError:
    HAS_NBA_STATS = False

CURRENT_SEASON = "2024-25"
_RATE_LIMIT_SEC = 0.65   # nba_api unofficial rate limit; be polite


@dataclass
class PlayerSeasonStats:
    player_id: int
    player_name: str
    team_id: int
    pts_per_game: float
    min_per_game: float
    usage_pct: float       # fraction of team possessions used while on floor (0..1)
    pts_std: float = 5.5   # game-to-game scoring std dev (default; see player_model.py)

    @property
    def importance(self) -> float:
        """Composite importance score: usage × minutes share of a full game.

        Roughly captures "what fraction of team offense disappears if this
        player sits". A 30%-usage player who plays 36 min = 0.30 × 36/48 = 0.225.
        """
        return self.usage_pct * (self.min_per_game / 48.0)

    @property
    def pts_per_min(self) -> float:
        return self.pts_per_game / max(self.min_per_game, 1.0)


@dataclass
class TeamSeasonStats:
    team_id: int
    team_name: str
    pace: float          # possessions per 48 min (high pace = more scoring opportunities)
    off_rating: float    # points per 100 possessions on offense
    def_rating: float    # points allowed per 100 possessions on defense
    pts_per_game: float

    @property
    def pts_per_min(self) -> float:
        return self.pts_per_game / 48.0


class NBAStatsCache:
    """In-memory cache of season averages. Call load() once at session start."""

    def __init__(self, season: str = CURRENT_SEASON):
        self.season = season
        self._players: dict[str, PlayerSeasonStats] = {}   # lowercase name -> stats
        self._teams: dict[int, TeamSeasonStats] = {}       # team_id -> stats
        self._fetched = False

    # ---------- public interface ----------

    def load(self) -> "NBAStatsCache":
        """Fetch from nba_api. Tolerates any API failure (logs and continues)."""
        if self._fetched or not HAS_NBA_STATS:
            if not HAS_NBA_STATS:
                print("[nba_stats] nba_api not installed; using neutral defaults")
            self._fetched = True
            return self
        try:
            self._fetch_players()
            self._fetch_teams()
            print(f"[nba_stats] loaded {len(self._players)} players, "
                  f"{len(self._teams)} teams ({self.season})")
        except Exception as e:
            print(f"[nba_stats] fetch failed (neutral defaults in use): {e}")
        self._fetched = True
        return self

    def player(self, name: str) -> PlayerSeasonStats | None:
        """Lookup by player name (case-insensitive). Returns None if unknown."""
        return self._players.get(name.lower())

    def team(self, team_id: int) -> TeamSeasonStats | None:
        return self._teams.get(team_id)

    def all_players(self) -> dict[str, PlayerSeasonStats]:
        return dict(self._players)

    def game_sigma(self, home_team_id: int, away_team_id: int,
                   base_sigma: float = 1.7,
                   baseline_pace: float = 98.5) -> float:
        """Return a game-specific score-diff sigma based on both teams' pace.

        Higher-pace matchups produce more possessions per minute, which means
        more scoring variance per minute (σ scales with sqrt(pace)).

        Falls back to base_sigma when team data is unavailable.

        Parameters
        ----------
        baseline_pace : float
            League-average pace the base_sigma was calibrated against (~98.5
            possessions per 48 min for the 2024-25 season).
        """
        h = self._teams.get(home_team_id)
        a = self._teams.get(away_team_id)
        if h is None or a is None:
            return base_sigma
        avg_pace = (h.pace + a.pace) / 2.0
        adjusted = base_sigma * float(np.sqrt(avg_pace / baseline_pace))
        # Keep within ±30% of the base to avoid extreme values from data errors
        return float(np.clip(adjusted, base_sigma * 0.70, base_sigma * 1.30))

    # ---------- fetchers (called at load time only) ----------

    def _fetch_players(self) -> None:
        time.sleep(_RATE_LIMIT_SEC)
        base = LeagueDashPlayerStats(
            season=self.season,
            per_mode_simple="PerGame",
            measure_type_simple="Base",
        ).get_data_frames()[0]

        time.sleep(_RATE_LIMIT_SEC)
        adv = LeagueDashPlayerStats(
            season=self.season,
            per_mode_simple="PerGame",
            measure_type_simple="Advanced",
        ).get_data_frames()[0]

        # Build usage map from advanced table
        usg_map: dict[int, float] = {}
        for _, row in adv.iterrows():
            pid = int(row["PLAYER_ID"])
            raw = row.get("USG_PCT", 0.0)
            # nba_api returns USG_PCT as a fraction (e.g. 0.273) or percentage (27.3)
            usg = float(raw) if raw is not None else 0.0
            usg_map[pid] = usg / 100.0 if usg > 1.0 else usg

        for _, row in base.iterrows():
            name = str(row["PLAYER_NAME"])
            pid = int(row["PLAYER_ID"])
            ppg = float(row.get("PTS", 0) or 0)
            mpg = float(row.get("MIN", 0) or 0)
            usg = usg_map.get(pid, 0.20)
            self._players[name.lower()] = PlayerSeasonStats(
                player_id=pid,
                player_name=name,
                team_id=int(row.get("TEAM_ID", 0) or 0),
                pts_per_game=ppg,
                min_per_game=mpg,
                usage_pct=usg,
                # Rough scoring std dev: sqrt(ppg) is Poisson-like; scale by 1.35
                pts_std=max(1.35 * (ppg ** 0.5), 2.0),
            )

    def _fetch_teams(self) -> None:
        time.sleep(_RATE_LIMIT_SEC)
        base = LeagueDashTeamStats(
            season=self.season,
            per_mode_simple="PerGame",
            measure_type_simple="Base",
        ).get_data_frames()[0]

        time.sleep(_RATE_LIMIT_SEC)
        adv = LeagueDashTeamStats(
            season=self.season,
            per_mode_simple="PerGame",
            measure_type_simple="Advanced",
        ).get_data_frames()[0]

        pace_map: dict[int, float] = dict(zip(
            adv["TEAM_ID"].astype(int), adv["PACE"].astype(float)
        ))
        off_map: dict[int, float] = dict(zip(
            adv["TEAM_ID"].astype(int), adv["OFF_RATING"].astype(float)
        ))
        def_map: dict[int, float] = dict(zip(
            adv["TEAM_ID"].astype(int), adv["DEF_RATING"].astype(float)
        ))

        for _, row in base.iterrows():
            tid = int(row["TEAM_ID"])
            self._teams[tid] = TeamSeasonStats(
                team_id=tid,
                team_name=str(row.get("TEAM_NAME", "")),
                pace=float(pace_map.get(tid, 98.5)),
                off_rating=float(off_map.get(tid, 112.0)),
                def_rating=float(def_map.get(tid, 112.0)),
                pts_per_game=float(row.get("PTS", 110.0) or 110.0),
            )
