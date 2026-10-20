"""
benchmark.py — grade the prediction model against the real-odds line.

For each timeline state, compares our model's marginal to the de-vigged
sportsbook consensus at that state's wall-clock timestamp. Moneyline first: it's
the one leg with no line, so P(team wins) compares directly.

Honesty guards:
 - the API returns the snapshot AT-OR-BEFORE the requested time, so we surface
   the actual snapshot timestamp and flag rows where it's badly stale;
 - deterministic rows (model prob 0 or 1, e.g. the final buzzer) and stale rows
   are excluded from the summary so a misaligned snapshot can't poison the stats.

Needs a seed whose timeline carries real `timestamp`s and a working ODDS_API_KEY.
The fetch is injectable so the alignment logic is unit-tested against a mock.
"""
from __future__ import annotations
from datetime import datetime

import numpy as np

from data.schemas import Leg, LegType, Side
from data.odds_api import try_featured_snapshot, parse_game_markets
from prediction.marginals import leg_fair_prob

STALE_MIN = 20.0            # a snapshot more than this far from the request is "stale"


def _minutes_apart(iso_a: str, iso_b: str):
    try:
        a = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
        return abs((a - b).total_seconds()) / 60.0
    except Exception:
        return None


def _deterministic(p: float) -> bool:
    return p <= 1e-3 or p >= 1.0 - 1e-3


def _excluded(st, model_p, actual_ts):
    """Reason this row is excluded from the summary, or '' if it counts."""
    if _deterministic(model_p):
        return "deterministic"
    if st.timestamp and actual_ts:
        d = _minutes_apart(st.timestamp, actual_ts)
        if d is not None and d > STALE_MIN:
            return f"stale {d:.0f}m"
    return ""


def benchmark_moneyline(seed, team: str, fetch=None) -> list[tuple]:
    """Returns [(state, model_p, market_p_or_None, actual_snapshot_ts_or_None), ...]."""
    fetch = fetch or try_featured_snapshot
    home, away = seed.home_team, seed.away_team
    ml = Leg(type=LegType.MONEYLINE, subject=team, side=Side.YES)

    rows = []
    for st in seed.timeline:
        model_p = leg_fair_prob(ml, st, seed.reference_data)
        market_p, actual_ts = None, None
        if st.timestamp:
            snap = fetch(st.timestamp)
            if snap:
                actual_ts = snap.get("timestamp")
                parsed = parse_game_markets(snap.get("data", []), home, away)
                if parsed and "h2h" in parsed:
                    market_p = parsed["h2h"].get(team)
        rows.append((st, model_p, market_p, actual_ts))
    return rows


def summarize(rows: list[tuple]) -> dict | None:
    good = [(m, r) for st, m, r, ats in rows
            if r is not None and not _excluded(st, m, ats)]
    if not good:
        return None
    m = np.array([g[0] for g in good])
    r = np.array([g[1] for g in good])
    return {
        "n": len(good),
        "mae": float(np.mean(np.abs(m - r))),
        "bias": float(np.mean(m - r)),
        "corr": float(np.corrcoef(m, r)[0, 1]) if len(good) > 1 else float("nan"),
    }


def format_rows(rows: list[tuple], team: str) -> str:
    lines = [f"  {'requested':<21}{'snapshot':<21}{'model':>7}{'mkt':>7}{'gap':>8}  note"]
    for st, m, r, ats in rows:
        req = st.timestamp or f"P{st.period} {int(st.clock)}s"
        snap = ats or "—"
        note = _excluded(st, m, ats)
        if r is None:
            lines.append(f"  {req:<21}{snap:<21}{m:>7.3f}{'—':>7}{'—':>8}  {note}")
        else:
            lines.append(f"  {req:<21}{snap:<21}{m:>7.3f}{r:>7.3f}{m - r:>+8.3f}  {note}")
    return "\n".join(lines)