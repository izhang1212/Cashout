"""
schemas.py — shared data types for the whole project.

Both the prediction and execution halves import from here, so these are the
canonical definitions. Dependency-free (stdlib dataclasses/enums).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LegType(str, Enum):
    MONEYLINE = "moneyline"   # team to win        (subject=team,   line=None)
    SPREAD    = "spread"      # team covers a line  (subject=team,   line=±pts)
    TOTAL     = "total"       # game total points   (subject=None,   line=pts)
    POINTS    = "points"      # player points       (subject=player, line=pts)
    REBOUNDS  = "rebounds"    # player rebounds     (subject=player, line=reb)
    ASSISTS   = "assists"     # player assists      (subject=player, line=ast)
    THREES    = "threes"      # player 3-pointers   (subject=player, line=3PM)


class Side(str, Enum):
    YES = "yes"   # condition happens (over the line / team wins)
    NO  = "no"    # condition does not happen


# stat key each player-prop leg accumulates in GameState.player_stats
STAT_KEY = {
    LegType.POINTS: "points",
    LegType.REBOUNDS: "rebounds",
    LegType.ASSISTS: "assists",
    LegType.THREES: "threes",
}


@dataclass(frozen=True)
class Leg:
    """One binary Kalshi contract held as part of a combo."""
    type: LegType
    side: Side = Side.YES
    subject: Optional[str] = None   # team abbrev or player name; None for TOTAL
    line: Optional[float] = None    # threshold; None for MONEYLINE

    def condition_met(self, state: "GameState") -> bool:
        """Is the YES condition satisfied at this (usually final) state?
        Ignores `side` — this is the raw event, not whether the held side won."""
        if self.type is LegType.MONEYLINE:
            return state.score_of(self.subject) > state.opponent_score_of(self.subject)
        if self.type is LegType.SPREAD:
            margin = state.score_of(self.subject) - state.opponent_score_of(self.subject)
            return margin > -self.line                 # covers signed spread `line`
        if self.type is LegType.TOTAL:
            return state.total_points() > self.line
        stat = STAT_KEY[self.type]                      # player counting prop
        return state.player_stats.get(self.subject, {}).get(stat, 0) >= self.line

    def resolves_yes(self, state: "GameState") -> bool:
        """Did the *held side* win at this (final) state?"""
        met = self.condition_met(state)
        return met if self.side is Side.YES else (not met)

    def describe(self) -> str:
        pre = "" if self.side is Side.YES else "NO "
        if self.type is LegType.MONEYLINE:
            return f"{pre}{self.subject} win"
        if self.type is LegType.TOTAL:
            return f"{pre}total over {self.line:g}"
        if self.type is LegType.SPREAD:
            return f"{pre}{self.subject} {self.line:+g}"
        return f"{pre}{self.subject} {self.line:g}+ {self.type.value}"


@dataclass
class Combo:
    """An all-or-nothing multi-leg position (n=1 is a single-leg combo)."""
    id: str
    legs: list[Leg]
    entry_price: float          # what we paid (0..1 per $1 payout)
    payout: float = 1.0         # $1/contract if every leg hits

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def is_single_leg(self) -> bool:
        return self.n_legs == 1

    def resolves_yes(self, state: "GameState") -> bool:
        """Combo wins only if every leg's side wins."""
        return all(leg.resolves_yes(state) for leg in self.legs)

    def validate(self) -> None:
        """<=5 legs and no directly contradictory legs; raises on failure.
        Does NOT judge correlation (correlated legs are fine) — only logical
        contradictions, mirroring Kalshi rejecting mutually exclusive baskets."""
        if not (1 <= self.n_legs <= 5):
            raise ValueError(f"combo {self.id}: {self.n_legs} legs (must be 1-5)")
        for i, a in enumerate(self.legs):
            for b in self.legs[i + 1:]:
                if _contradict(a, b):
                    raise ValueError(
                        f"combo {self.id}: contradictory legs "
                        f"'{a.describe()}' and '{b.describe()}'")


def _contradict(a: Leg, b: Leg) -> bool:
    if a.type == b.type and a.subject == b.subject and a.line == b.line:
        return a.side != b.side                         # same market, opposite sides
    if a.type is LegType.MONEYLINE and b.type is LegType.MONEYLINE:
        if a.subject != b.subject and a.side is Side.YES and b.side is Side.YES:
            return True                                 # both teams to win
    return False                                        # (kept simple for v1)


@dataclass
class GameState:
    """The game at one point in time. One entry per game_timeline step."""
    period: int                       # 1-4 regulation, 5+ overtime
    clock: float                      # seconds left in the current period
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    possession: Optional[str] = None
    player_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    # ISO wall-clock time of this state (for real-odds alignment)
    timestamp: Optional[str] = None   

    def score_of(self, team: str) -> int:
        return self.home_score if team == self.home_team else self.away_score

    def opponent_score_of(self, team: str) -> int:
        return self.away_score if team == self.home_team else self.home_score

    def total_points(self) -> int:
        return self.home_score + self.away_score


@dataclass
class Paths:
    """Simulated forward paths — produced by the MC backbone, consumed by the
    marginal/correlation read-out and by LSM. Placeholder for the data slice;
    fleshed out with the simulation backbone. `finals` = terminal state per path."""
    finals: list[GameState] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.finals)


@dataclass
class Seed:
    """Everything loaded from seed_data.yaml for one game."""
    game_id: str
    home_team: str
    away_team: str
    date: str
    combos: list[Combo]
    timeline: list[GameState]
    reference_data: dict
    real_odds: Optional[dict] = None   # de-vigged benchmark, filled by odds_api

    @property
    def final_state(self) -> GameState:
        return self.timeline[-1]