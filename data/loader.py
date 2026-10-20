"""loader.py — read seed_data.yaml into typed objects and validate it."""
from __future__ import annotations
import yaml
from pathlib import Path

import config
from .schemas import LegType, Side, Leg, Combo, GameState, Seed


def load_seed(path: str | Path = config.SEED_PATH) -> Seed:
    with open(path) as f:
        raw = yaml.safe_load(f)

    g = raw["game"]
    home, away = g["home_team"], g["away_team"]

    combos = [_parse_combo(c) for c in raw["combos"]]
    for c in combos:
        c.validate()

    timeline = [_parse_state(s, home, away) for s in raw["game_timeline"]]

    return Seed(
        game_id=g["game_id"], home_team=home, away_team=away, date=g.get("date", ""),
        combos=combos, timeline=timeline, reference_data=raw.get("reference_data", {}),
    )


def _parse_combo(c: dict) -> Combo:
    return Combo(
        id=c["id"], legs=[_parse_leg(l) for l in c["legs"]],
        entry_price=float(c["entry_price"]), payout=float(c.get("payout", 1.0)),
    )


def _parse_leg(l: dict) -> Leg:
    line = l.get("line")
    side = l.get("side", "yes")
    if isinstance(side, bool):            # YAML parses bare yes/no as booleans
        side = "yes" if side else "no"
    return Leg(
        type=LegType(l["type"]), side=Side(str(side).lower()),
        subject=l.get("subject"), line=float(line) if line is not None else None,
    )

def _parse_state(s: dict, home: str, away: str) -> GameState:
    return GameState(
        period=int(s["period"]), clock=float(s["clock"]),
        home_team=home, away_team=away,
        home_score=int(s["home_score"]), away_score=int(s["away_score"]),
        possession=s.get("possession"),
        player_stats={p: dict(st) for p, st in s.get("player_stats", {}).items()},
        timestamp=s.get("timestamp"),
    )