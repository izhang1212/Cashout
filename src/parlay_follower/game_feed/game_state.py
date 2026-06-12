"""Live game state vector + leg definitions and resolvers.

Each leg maps to a deterministic resolver over the state vector:
COMPLETED the instant its condition is met, FAILED/clinched-dead the instant it
becomes impossible, LIVE otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

REGULATION_SEC = 48 * 60


@dataclass
class GameState:
    seconds_remaining: float = REGULATION_SEC   # regulation; OT handled as extra time
    period: int = 1
    home_score: int = 0
    away_score: int = 0
    home_possession: bool | None = None
    final: bool = False
    player_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # player_stats["LeBron James"] = {"pts": 22, "reb": 6, "ast": 7, "min": 28, "fouls": 3}

    @property
    def score_diff(self) -> int:
        """Home minus away."""
        return self.home_score - self.away_score

    @property
    def tau_minutes(self) -> float:
        return self.seconds_remaining / 60.0


class LegStatus(Enum):
    LIVE = "live"
    COMPLETED = "completed"   # leg has hit (or is mathematically clinched)
    FAILED = "failed"         # leg is mathematically dead


@dataclass
class Leg:
    leg_id: str
    kind: str                  # "moneyline" | "total_over" | "total_under" | "player_points_over"
    params: dict               # e.g. {"side": "home"} or {"line": 224.5} or {"player": "...", "line": 24.5}
    market_ticker: str = ""    # per-leg Kalshi ticker (for market-implied prob)
    status: LegStatus = LegStatus.LIVE


def resolve_leg(leg: Leg, gs: GameState) -> LegStatus:
    """Deterministic resolution from the current state. Pure function."""
    k, p = leg.kind, leg.params

    if k == "moneyline":
        if not gs.final:
            return LegStatus.LIVE
        home_won = gs.score_diff > 0
        want_home = p.get("side", "home") == "home"
        return LegStatus.COMPLETED if home_won == want_home else LegStatus.FAILED

    if k == "total_over":
        total = gs.home_score + gs.away_score
        if total > p["line"]:
            return LegStatus.COMPLETED          # clinched: totals only go up
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k == "total_under":
        total = gs.home_score + gs.away_score
        if total > p["line"]:
            return LegStatus.FAILED             # dead the moment the line is crossed
        return LegStatus.COMPLETED if gs.final else LegStatus.LIVE

    if k == "player_points_over":
        pts = gs.player_stats.get(p["player"], {}).get("pts", 0.0)
        if pts > p["line"]:
            return LegStatus.COMPLETED          # clinched
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    raise ValueError(f"Unknown leg kind: {k}")


def update_all(legs: list[Leg], gs: GameState) -> list[Leg]:
    for leg in legs:
        if leg.status is LegStatus.LIVE:        # resolved legs never un-resolve
            leg.status = resolve_leg(leg, gs)
    return legs
