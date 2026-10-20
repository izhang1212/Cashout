"""
config.py — project-wide constants and shared economic primitives.

Two things live here: (1) numeric knobs you'd tune without touching logic, and
(2) the economic primitives that must be identical across the whole system --
the Kalshi fee and the utility function. Both the data and execution layers
import these, so they sit at the root where everything can depend on them
without cross-layer imports.

Also loads .env (no dependency) so secrets like ODDS_API_KEY reach the
environment without ever living in code.
"""
from __future__ import annotations
import math
import os
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).parent
SEED_PATH = ROOT / "data" / "seed_data.yaml"


# ---- .env loader (minimal, no dependency) ----
def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ. A real exported env var
    always wins (setdefault), so this never clobbers your shell."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()

# ---- objective (execution) ----
UTILITY = "log"            # total-wealth log utility -- the only v1 choice
BANKROLL_W = 1000.0        # outside wealth W; parameterizes the stopping boundary

# ---- stopping engine ----
N_SIMS = 20_000            # Monte Carlo paths for the LSM backbone
REFIT_ON_PERIOD_START = True   # re-fit the policy at each quarter boundary (4x/game)

# ---- game clock ----
REGULATION_PERIODS = 4
PERIOD_SECONDS = 12 * 60   # 720s; OT periods are 5min, handled via the timeline
GAME_MINUTES = 48

# ---- frictions (cornerstone: realized cash-out = bid - exit_fee, never mid) ----
FEE_COEFF = 0.07           # placeholder -- set from Kalshi's current fee schedule
HALF_SPREAD = 0.02         # bid sits ~half_spread below mid
NOISE_STD = 0.015          # AR(1) noise on the synthetic Kalshi price (test realism)
RNG_SEED = 0


# ---- team strength (our own prior; independent of the market line) ----
HOME_COURT_ADV = 2.8       # points; standard NBA home-court edge


def _team_net(team: str, ref: dict) -> float:
    t = ref.get("teams", {}).get(team, {})
    return t.get("off_rtg", 110.0) - t.get("def_rtg", 110.0)


def team_strength_lean(subject: str, home_team: str, away_team: str, ref: dict) -> float:
    """Expected full-game margin (subject - opponent) from net ratings + home court.
    This is OUR prior from team quality -- NOT the market line, so using it doesn't
    anchor the model to the odds we're grading against."""
    opp = away_team if subject == home_team else home_team
    lean = _team_net(subject, ref) - _team_net(opp, ref)
    lean += HOME_COURT_ADV if subject == home_team else -HOME_COURT_ADV
    return lean


# ---- shared economic primitives ----
def kalshi_fee(price: float, contracts: int = 1, coeff: float = FEE_COEFF) -> float:
    """Per-trade Kalshi fee, price-dependent, peaks near $0.50: coeff*p*(1-p).
    Shared by the data layer (price display) and execution (reward R_t)."""
    price = min(max(price, 0.0), 1.0)
    return coeff * contracts * price * (1.0 - price)


def utility(wealth: float) -> float:
    """Total-wealth log utility U(W)=log(W). Guarded against non-positive W."""
    return math.log(max(wealth, 1e-9))