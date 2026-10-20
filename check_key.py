"""
check_key.py — verify ODDS_API_KEY works with one cheap current-odds call.
Reads the key from .env (via config). Run from the project root:
    python check_key.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: F401  (loads .env into the environment)
import requests

KEY = os.environ.get("ODDS_API_KEY", "")
if not KEY or KEY.startswith("PASTE_"):
    print("ODDS_API_KEY is not set — paste your key into .env first.")
    sys.exit(1)

try:
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds",
        params={"apiKey": KEY, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
        timeout=20,
    )
    print("status:", r.status_code)
    print("quota remaining:", r.headers.get("x-requests-remaining"),
          "| used:", r.headers.get("x-requests-used"))
    if r.ok:
        games = r.json()
        print(f"OK — key works. {len(games)} NBA games in the current feed.")
        if games:
            g = games[0]
            print(f"  sample: {g['away_team']} @ {g['home_team']}")
    else:
        print("ERROR body:", r.text[:300])
except Exception as e:
    print("request failed:", e)