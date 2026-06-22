"""MLB live game state and leg resolution.

Parallel to game_feed/game_state.py but for baseball's discrete structure:
innings, outs, and base-runner state instead of a running clock.

The shared Leg / LegStatus types are reused unchanged. Only the state vector
and leg resolvers are sport-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..game_feed.game_state import Leg, LegStatus

# A regulation MLB game has 27 outs per team = 54 total.
TOTAL_OUTS = 54


@dataclass
class MLBGameState:
    game_id: str = ""
    inning: int = 1
    half: str = "top"      # "top" (away batting) | "bottom" (home batting)
    outs: int = 0          # 0-2 within the current half-inning
    runners: int = 0       # bitmask: bit0=1B, bit1=2B, bit2=3B
    home_score: int = 0
    away_score: int = 0
    final: bool = False
    player_stats: dict[str, dict] = field(default_factory=dict)
    # player_stats["Shohei Ohtani"] = {"hits": 2, "hr": 1, "tb": 5, "rbi": 2,
    #                                   "sb": 0, "ab": 4, "team": "home",
    #                                   "k": 0}  # for pitchers: "k" = strikeouts
    home_team_id: int = 0
    away_team_id: int = 0
    home_pitcher: str = ""   # current pitcher name
    away_pitcher: str = ""

    # ---------- derived properties ----------

    @property
    def score_diff(self) -> int:
        """Home minus away runs (positive = home leading)."""
        return self.home_score - self.away_score

    @property
    def outs_completed(self) -> int:
        """Total outs recorded so far in this game."""
        half_innings_done = (self.inning - 1) * 2
        if self.half == "bottom":
            half_innings_done += 1
        return half_innings_done * 3 + self.outs

    @property
    def outs_remaining(self) -> int:
        """Approximate outs remaining. Walk-off situations undercount by a few."""
        return max(0, TOTAL_OUTS - self.outs_completed)

    @property
    def tau_minutes(self) -> float:
        """Map outs remaining to a 'time' value for shared LSMC/engine code.

        Convention: 1 out ≈ 1 minute (a 27-out game ≈ 27 min per team ≈ 3 hr).
        This keeps the LSMC time grid in the same order of magnitude as NBA.
        """
        return float(self.outs_remaining)

    @property
    def seconds_remaining(self) -> float:
        """Rough wall-clock estimate for logging (not used in decision math)."""
        return self.outs_remaining * 60.0

    @property
    def runner_on_first(self) -> bool:
        return bool(self.runners & 0b001)

    @property
    def runner_on_second(self) -> bool:
        return bool(self.runners & 0b010)

    @property
    def runner_on_third(self) -> bool:
        return bool(self.runners & 0b100)

    @property
    def run_expectancy_index(self) -> int:
        """Base-out state index (0-23) into the run expectancy table."""
        return self.runners * 3 + self.outs   # 8 base states × 3 out states


@dataclass
class MLBCrossGameState:
    """Aggregate view across multiple independent MLB games for cross-game combos.

    Each leg in a cross-game combo belongs to a specific game identified by
    `leg.params["game"]` (the gamePk). This state provides the engine with
    the combined time horizon and a neutral score_diff (games are independent).
    """
    game_states: dict  # str(game_pk) -> MLBGameState

    @property
    def tau_minutes(self) -> float:
        active = [gs.tau_minutes for gs in self.game_states.values() if not gs.final]
        return max(active, default=0.0)

    @property
    def score_diff(self) -> int:
        return 0  # not meaningful across independent games

    @property
    def seconds_remaining(self) -> float:
        return self.tau_minutes * 60.0

    @property
    def final(self) -> bool:
        return all(gs.final for gs in self.game_states.values())

    @property
    def cumulative_runs(self) -> int:
        return sum(gs.home_score + gs.away_score for gs in self.game_states.values())

    @property
    def game_id(self) -> str:
        return ",".join(self.game_states.keys())


def _leg_game_pk(leg: Leg) -> str:
    """Return the leg's game pk as a string, or '' if none specified."""
    raw = leg.params.get("game")   # None when absent (no default 0)
    if raw is None:
        return ""
    try:
        return str(int(float(raw)))
    except (ValueError, TypeError):
        return str(raw)


def resolve_mlb_leg(leg: Leg, gs: MLBGameState) -> LegStatus:
    """Deterministic leg resolution from the current MLB game state."""
    k, p = leg.kind, leg.params

    if k == "moneyline":
        if not gs.final:
            return LegStatus.LIVE
        return LegStatus.COMPLETED if gs.score_diff > 0 else LegStatus.FAILED

    if k in ("total_over", "cross_total_over"):
        total = gs.home_score + gs.away_score
        if total > p["line"]:
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k in ("total_under", "cross_total_under"):
        total = gs.home_score + gs.away_score
        if total > p["line"]:
            return LegStatus.FAILED
        return LegStatus.COMPLETED if gs.final else LegStatus.LIVE

    # Player prop legs — count accumulated so far
    player = p.get("player", "")
    stats = gs.player_stats.get(player, {})

    if k == "hits_over":
        val = float(stats.get("hits", 0))
        if val > p["line"]:
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k == "home_runs":
        val = float(stats.get("hr", 0))
        if val >= p.get("line", 1):
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k == "total_bases_over":
        val = float(stats.get("tb", 0))
        if val > p["line"]:
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k == "rbi_over":
        val = float(stats.get("rbi", 0))
        if val > p["line"]:
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    if k == "strikeouts_over":   # pitcher prop
        val = float(stats.get("k", 0))
        if val > p["line"]:
            return LegStatus.COMPLETED
        return LegStatus.FAILED if gs.final else LegStatus.LIVE

    raise ValueError(f"Unknown MLB leg kind: {k}")


def update_mlb_legs(legs: list[Leg], gs: MLBGameState) -> list[Leg]:
    for leg in legs:
        if leg.status is LegStatus.LIVE:
            leg.status = resolve_mlb_leg(leg, gs)
    return legs


def update_mlb_legs_cross_game(legs: list[Leg],
                                game_states: dict) -> list[Leg]:
    """Resolve legs across multiple games; each leg carries its gamePk in params."""
    return update_mlb_legs_generalized(legs, game_states)


def update_mlb_legs_generalized(legs: list[Leg], game_states: dict,
                                 primary_pk: str | None = None) -> list[Leg]:
    """Resolve legs for same-game, cross-game, or mixed combos.

    Legs with game= param route to that specific game state.
    Legs without game= param use primary_pk (defaults to first game in dict).
    """
    _primary = primary_pk or next(iter(game_states.keys()), None)
    for leg in legs:
        if leg.status is LegStatus.LIVE:
            pk = _leg_game_pk(leg) or _primary
            gs = game_states.get(str(pk)) if pk else next(iter(game_states.values()), None)
            if gs is not None:
                leg.status = resolve_mlb_leg(leg, gs)
    return legs
