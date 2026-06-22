"""Monte Carlo combo pricing: full terminal payoff distribution of the parlay.

From the current state, simulate joint leg resolution under the Gaussian copula.
Mean of the payoff is the fair value F(t); the full distribution feeds the risk
adjustment in the decision engine.

Already-resolved legs are short-circuited: COMPLETED legs contribute probability
1, FAILED legs make the combo worth exactly 0 regardless of simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..game_feed.game_state import GameState, Leg, LegStatus
from .copula import CorrelationTable, joint_hit_samples, make_corr_matrix, nearest_psd


@dataclass
class ComboValuation:
    fair_value: float            # E[payoff] in dollars per contract
    p_all_hit: float             # same number, but named for what it is
    payoff_samples: np.ndarray   # (n_paths,) of 0/1 -- feeds CVaR / utility adjustment
    per_leg_probs: dict[str, float]


def leg_live_prob(leg: Leg, gs: GameState, stern,
                  prop_probs: dict[str, float] | None = None) -> float:
    """Per-leg probability for LIVE legs.

    Game-outcome legs come from the Stern diffusion in closed form. Prop legs
    (and anything the diffusion can't price) are supplied by the production
    LightGBM model via `prop_probs`; absent that, a neutral 0.5 placeholder is
    used and should be treated as UNPROVEN by the shrinkage layer.
    """
    if leg.status is LegStatus.COMPLETED:
        return 1.0
    if leg.status is LegStatus.FAILED:
        return 0.0

    if leg.kind == "moneyline":
        return stern.win_prob(gs.score_diff, gs.tau_minutes, side=leg.params.get("side", "home"))

    if leg.kind in ("total_over", "total_under"):
        # Pace-aware totals: blend current-game pace with league-average anchor,
        # then apply Q4 adjustments for game situation.
        from scipy.stats import norm
        elapsed_min = max(48.0 - gs.tau_minutes, 1e-6)
        current_total = gs.home_score + gs.away_score
        game_pace = current_total / elapsed_min
        # League-average combined pace fallback (~4.58 pts/min = 220 pts/48 min)
        season_pace = 220.0 / 48.0
        # Bayesian blend: weight game pace by elapsed time, prior by 12 minutes
        blend = min(elapsed_min / (elapsed_min + 12.0), 0.80)
        blended_pace = blend * game_pace + (1.0 - blend) * season_pace
        # Q4 game-situation adjustments
        abs_diff = abs(gs.score_diff)
        if gs.tau_minutes <= 12.0 and abs_diff <= 8:
            blended_pace *= 0.92   # close game: clock management slows pace
        elif gs.tau_minutes <= 6.0 and abs_diff >= 10:
            blended_pace *= 1.05   # blowout: intentional fouling speeds pace
        mean_total = current_total + blended_pace * gs.tau_minutes
        std_total = 1.6 * stern.sigma * np.sqrt(max(gs.tau_minutes, 1e-9))
        p_over = 1.0 - norm.cdf(leg.params["line"], loc=mean_total, scale=std_total)
        return float(np.clip(p_over if leg.kind == "total_over" else 1.0 - p_over,
                             0.01, 0.99))

    if prop_probs and leg.leg_id in prop_probs:
        return prop_probs[leg.leg_id]
    return 0.5  # explicit "no model" placeholder


def price_combo(legs: list[Leg], gs: GameState, stern,
                corr: CorrelationTable, n_paths: int = 20000,
                prop_probs: dict[str, float] | None = None,
                rng: np.random.Generator | None = None) -> ComboValuation:
    # Short-circuit: any FAILED leg kills the combo.
    if any(l.status is LegStatus.FAILED for l in legs):
        z = np.zeros(n_paths)
        return ComboValuation(0.0, 0.0, z, {l.leg_id: leg_live_prob(l, gs, stern, prop_probs) for l in legs})

    live = [l for l in legs if l.status is LegStatus.LIVE]
    per_leg = {l.leg_id: leg_live_prob(l, gs, stern, prop_probs) for l in legs}

    if not live:  # everything clinched
        return ComboValuation(1.0, 1.0, np.ones(n_paths), per_leg)

    probs = np.array([per_leg[l.leg_id] for l in live])

    # Pairwise state-conditional correlations -> matrix (projected to PSD).
    n = len(live)
    R = make_corr_matrix(n, corr.default_rho)
    for a in range(n):
        for b in range(a + 1, n):
            R[a, b] = R[b, a] = corr.rho(
                live[a].kind, live[b].kind, abs(gs.score_diff), gs.tau_minutes
            )
    R = nearest_psd(R)

    hits = joint_hit_samples(probs, R, n_paths, rng=rng)   # (n_paths, n_live)
    payoff = hits.all(axis=1).astype(float)                # $1 if every live leg hits
    fv = float(payoff.mean())
    return ComboValuation(fv, fv, payoff, per_leg)


def synthetic_fair_value(per_leg_market_probs: list[float], rho: float,
                         n_paths: int = 20000,
                         rng: np.random.Generator | None = None) -> float:
    """Market-maker-style fair value F_mm: market-implied legs + crude correlation.

    This is the anchor the empirical haircut model is measured against.
    """
    probs = np.array(per_leg_market_probs, dtype=float)
    if len(probs) == 0:
        return 0.0
    if len(probs) == 1:
        return float(probs[0])
    R = make_corr_matrix(len(probs), rho)
    hits = joint_hit_samples(probs, R, n_paths, rng=rng)
    return float(hits.all(axis=1).mean())
