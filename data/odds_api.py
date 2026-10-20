"""
odds_api.py — real sportsbook odds for the seed game, de-vigged to implied
probabilities. This is the BENCHMARK the prediction model is graded against.

Source: The Odds API v4. Featured markets (h2h, totals, spreads) come from the
historical/odds endpoint; player props come one game at a time from the
historical event-odds endpoint (Business tier). Requires ODDS_API_KEY.

The HTTP fetch is the one untested seam (needs a key + network). The parse/de-vig
below is unit-tested against the documented response schema.

Caveats baked into how you use the output:
 - h2h -> P(team wins): directly comparable to our moneyline marginal (no line).
 - totals/spreads/props: the book's LINE differs from our combo's line, so compare
   at the book's line, not ours.
"""
from __future__ import annotations
import os
from statistics import mean
from typing import Optional

try:
    import requests
except ImportError:
    requests = None

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# The API names teams in full ("Boston Celtics"); our seed uses abbreviations.
NBA_TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def _team_matches(api_name: str, ours: str) -> bool:
    """True if the API's full team name refers to our (abbrev or full) team."""
    full = NBA_TEAMS.get(ours, ours).lower()
    api = (api_name or "").lower()
    return api == full or api in full or full in api or ours.lower() in api

def _find_event(data: list[dict], home: str, away: str) -> Optional[dict]:
    return next((e for e in data
                 if _team_matches(e.get("home_team", ""), home)
                 and _team_matches(e.get("away_team", ""), away)), None)

# ---- odds math --------------------------------------------------------------
def american_to_prob(american: float) -> float:
    """Raw (vig-inclusive) implied probability from American odds."""
    a = float(american)
    return (-a) / ((-a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def devig(probs: list[float]) -> list[float]:
    """Normalize a market's raw implied probabilities so they sum to 1."""
    s = sum(probs)
    return [p / s for p in probs] if s > 0 else probs


# ---- parsing (unit-tested) --------------------------------------------------
def _find_event(data: list[dict], home: str, away: str) -> Optional[dict]:
    return next((e for e in data
                 if _team_matches(e.get("home_team", ""), home)
                 and _team_matches(e.get("away_team", ""), away)), None)


def parse_game_markets(data: list[dict], home: str, away: str) -> Optional[dict]:
    """One snapshot's `data` -> consensus de-vigged probs for h2h and totals.
    Averages the de-vigged prob across bookmakers. Returns None if game absent."""
    ev = _find_event(data, home, away)
    if ev is None:
        return None

    h2h_home, h2h_away = [], []
    tot_over, tot_under, tot_line = [], [], []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "h2h":
                probs = {o["name"]: american_to_prob(o["price"]) for o in mk["outcomes"]}
                hn = next((n for n in probs if _team_matches(n, home)), None)
                an = next((n for n in probs if _team_matches(n, away)), None)
                if hn and an:
                    ph, pa = devig([probs[hn], probs[an]])
                    h2h_home.append(ph); h2h_away.append(pa)
            elif mk["key"] == "totals":
                over = next((o for o in mk["outcomes"] if o["name"] == "Over"), None)
                under = next((o for o in mk["outcomes"] if o["name"] == "Under"), None)
                if over and under:
                    po, pu = devig([american_to_prob(over["price"]),
                                    american_to_prob(under["price"])])
                    tot_over.append(po); tot_under.append(pu)
                    tot_line.append(over.get("point"))

    out: dict = {}
    if h2h_home:
        out["h2h"] = {home: mean(h2h_home), away: mean(h2h_away)}
    if tot_over:
        out["totals"] = {"line": mean(tot_line), "over": mean(tot_over),
                         "under": mean(tot_under)}
    return out


def parse_player_props(event: dict, market_key: str) -> dict:
    """Historical event-odds payload -> {player: {line, over, under}} for one
    prop market (e.g. 'player_points'), de-vigged and averaged across books.
    NOTE: the book's line rarely equals our combo's threshold -- compare at the
    book's line."""
    per_player: dict[str, dict[str, list]] = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != market_key:
                continue
            overs = {o["description"]: o for o in mk["outcomes"] if o.get("name") == "Over"}
            unders = {o["description"]: o for o in mk["outcomes"] if o.get("name") == "Under"}
            for player, o in overs.items():
                u = unders.get(player)
                if not u:
                    continue
                po, pu = devig([american_to_prob(o["price"]), american_to_prob(u["price"])])
                slot = per_player.setdefault(player, {"line": [], "over": [], "under": []})
                slot["line"].append(o.get("point")); slot["over"].append(po); slot["under"].append(pu)
    return {p: {"line": mean(v["line"]), "over": mean(v["over"]), "under": mean(v["under"])}
            for p, v in per_player.items()}


# ---- HTTP fetch (untested seam: needs ODDS_API_KEY + network) ----------------
def _key(api_key: Optional[str]) -> str:
    k = api_key or os.environ.get("ODDS_API_KEY")
    if not k:
        raise RuntimeError("ODDS_API_KEY not set (see .env)")
    if requests is None:
        raise RuntimeError("`requests` not installed")
    return k


def fetch_featured_snapshot(date_iso: str, markets: str = "h2h,totals",
                            regions: str = "us", api_key: Optional[str] = None) -> dict:
    """Historical featured-markets snapshot at/just before date_iso.
    Returns the full wrapper {timestamp, previous_timestamp, next_timestamp, data}."""
    resp = requests.get(f"{BASE}/historical/sports/{SPORT}/odds",
                        params={"apiKey": _key(api_key), "regions": regions,
                                "markets": markets, "oddsFormat": "american",
                                "date": date_iso}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_event_snapshot(event_id: str, date_iso: str, markets: str = "player_points",
                         regions: str = "us", api_key: Optional[str] = None) -> dict:
    """Historical event-odds snapshot (player props etc.). Business tier."""
    resp = requests.get(f"{BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
                        params={"apiKey": _key(api_key), "regions": regions,
                                "markets": markets, "oddsFormat": "american",
                                "date": date_iso}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def try_featured_snapshot(date_iso: str, **kw) -> Optional[dict]:
    """Best-effort; None on any failure so callers fall back to the model."""
    try:
        return fetch_featured_snapshot(date_iso, **kw)
    except Exception as e:
        print(f"[odds_api] real odds unavailable, falling back to model: {e}")
        return None