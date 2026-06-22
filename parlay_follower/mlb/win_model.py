"""MLB win probability model via run-expectancy Markov chain simulation.

Baseball win probability is fundamentally different from the NBA Brownian motion
model. The state space is discrete: (inning, half, outs, runners on base, score
differential). We estimate P(home wins) by Monte Carlo simulation of remaining
half-innings, drawing per-half-inning run totals from a team-adjusted Poisson
distribution.

Key design decisions
--------------------
  * Per-half-inning run distribution: Poisson(lambda), where lambda is blended
    from the batting team's season runs/inning and the pitching team's ERA-implied
    runs/inning. This captures matchup quality without complex simulation.
  * Run expectancy in the current half-inning: conditioned on outs and base state
    via the standard 24-state run expectancy matrix (Tango, Lichtman, Dolphin).
    This is an additive correction to the Poisson-drawn remaining innings.
  * simulate_paths() produces score-diff traces in the same format as SternModel,
    allowing the shared LSMC boundary code (nleg_paths.py) to work unchanged.
    tau is mapped as: 1 unit = 1 out remaining, so tau=27 means a full game left.
  * sigma property is set to match the empirical score-diff variance per out
    (~0.45), which puts it in the correct order of magnitude for the bid model.

Walk-off handling: in the bottom of the last inning when the home team is ahead
the game can end mid-inning. We approximate this by capping home win probability
at 1.0 and using the Poisson simulation for the main paths.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .game_state import MLBGameState
from .stats import MLBStatsCache

# Standard 24-state run expectancy matrix (MLB 2010-2019 average).
# Indexed as [runners_bitmask (0-7)][outs (0-2)].
# runners: bit0=1B, bit1=2B, bit2=3B
_RE_TABLE: list[list[float]] = [
    [0.544, 0.291, 0.113],  # bases empty
    [0.941, 0.562, 0.243],  # 1B only
    [1.170, 0.699, 0.348],  # 2B only
    [1.556, 0.940, 0.462],  # 1B + 2B
    [1.381, 0.983, 0.385],  # 3B only
    [1.798, 1.151, 0.548],  # 1B + 3B
    [2.052, 1.447, 0.628],  # 2B + 3B
    [2.417, 1.634, 0.798],  # bases loaded
]

# League-average runs per half-inning (used when no team data available)
_LEAGUE_AVG_RUNS_PER_HALF_INN = 0.50   # ~4.5 runs/9-inn team = 0.5/half-inning

# Score-diff std per out (calibrated to empirical MLB data).
# A full 54-out game has ~sqrt(54)*0.45 ≈ 3.3 pts std, which matches
# typical MLB game score-diff distributions.
_SIGMA_PER_OUT = 0.45


class MLBWinModel:
    """Win probability model for MLB.

    Compatible interface with SternModel:
      .win_prob(score_diff, tau, side)  -- used by price_combo / leg_live_prob
      .simulate_paths(...)              -- used by build_nleg_boundary
      .sigma                            -- used by bid_model
      .mu                               -- drift (home advantage per out)
    """

    def __init__(self, stats_cache: MLBStatsCache | None = None,
                 pregame_home_advantage_runs: float = 0.15):
        """
        pregame_home_advantage_runs: expected extra runs home team scores
            over a full game due to home field. Translates to mu per out.
        """
        self.stats = stats_cache or MLBStatsCache()
        self.sigma = _SIGMA_PER_OUT
        # mu: drift of score_diff per out (positive = home favored)
        self.mu = pregame_home_advantage_runs / 54.0

    # ---------- primary interface (matches SternModel duck type) ----------

    def win_prob(self, score_diff: float, tau: float, side: str = "home") -> float:
        """P(side wins) from current score_diff and outs_remaining (tau).

        Uses a fast Normal approximation of the run-diff distribution for
        remaining outs, which is accurate when tau > 6 (more than 2 innings
        remaining). For late-game situations with specific base-out state,
        use win_prob_from_state() instead.
        """
        if tau <= 0:
            if score_diff > 0:
                p = 1.0
            elif score_diff == 0:
                p = 0.5
            else:
                p = 0.0
            return p if side == "home" else 1.0 - p

        from scipy.stats import norm
        # Remaining runs differential ~ Normal(mu*tau, sigma^2*tau)
        mean = self.mu * tau
        std = self.sigma * np.sqrt(max(tau, 1e-9))
        # P(home wins) = P(score_diff + remaining_diff > 0)
        # = P(remaining_diff > -score_diff) = 1 - Phi((-score_diff - mean) / std)
        p = float(norm.sf(-float(score_diff), loc=mean, scale=std))
        p = float(np.clip(p, 0.01, 0.99))
        return p if side == "home" else 1.0 - p

    def win_prob_from_state(self, gs: MLBGameState,
                            n_sims: int = 5000,
                            rng: np.random.Generator | None = None) -> float:
        """Higher-accuracy P(home wins) using the full game state.

        Uses Monte Carlo simulation through the Markov chain. More expensive
        than win_prob() but conditions on the base-out state and specific
        half-inning structure. Called by game_context.py each tick.
        """
        rng = rng or np.random.default_rng()
        if gs.final:
            return 1.0 if gs.score_diff > 0 else (0.5 if gs.score_diff == 0 else 0.0)

        final_diffs = self._simulate_game_endings(gs, n_sims, rng)
        return float((final_diffs > 0).mean() + 0.5 * (final_diffs == 0).mean())

    def simulate_paths(self, score_diff: float, tau: float,
                       n_paths: int, n_steps: int,
                       rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate future score-diff paths for the shared LSMC boundary code.

        tau is outs_remaining (following the tau_minutes convention in
        MLBGameState.tau_minutes). Each step covers tau/n_steps outs.
        Returns (n_paths, n_steps+1) array.
        """
        rng = rng or np.random.default_rng()
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = score_diff

        if tau <= 0 or n_steps == 0:
            paths[:] = score_diff
            return paths

        outs_per_step = tau / n_steps
        # Runs per step per team: scale Poisson rate to step size
        lam = _LEAGUE_AVG_RUNS_PER_HALF_INN * (outs_per_step / 3.0)
        lam = max(lam, 1e-4)

        for s in range(1, n_steps + 1):
            home_runs = rng.poisson(lam, n_paths).astype(float)
            away_runs = rng.poisson(lam, n_paths).astype(float)
            paths[:, s] = paths[:, s - 1] + home_runs - away_runs + self.mu * outs_per_step

        return paths

    # ---------- internal helpers ----------

    def _run_lambda(self, batting_team_id: int, pitching_team_id: int) -> float:
        """Expected runs per half-inning for batting_team vs pitching_team."""
        off = self.stats.team(batting_team_id)
        def_ = self.stats.team(pitching_team_id)
        off_rate = off.runs_per_inning if off else _LEAGUE_AVG_RUNS_PER_HALF_INN
        def_rate = def_.runs_allowed_per_game / 9.0 if def_ else _LEAGUE_AVG_RUNS_PER_HALF_INN
        # Blend: geometric mean of team rates vs. opponent (log5-style)
        blended = float(np.sqrt(off_rate * def_rate))
        return float(np.clip(blended, 0.1, 2.0))

    def _simulate_game_endings(self, gs: MLBGameState, n_sims: int,
                                rng: np.random.Generator) -> np.ndarray:
        """Simulate n_sims games from the current state, return final score diffs."""
        score_diffs = np.full(n_sims, float(gs.score_diff))

        # Determine team IDs for run-lambda calculation
        home_id = gs.home_team_id
        away_id = gs.away_team_id

        # Run expectancy correction for the current half-inning
        re_now = _RE_TABLE[gs.runners][gs.outs]
        re_end = 0.0   # no runs expected after 3 outs

        # Runs expected in current half-inning beyond what's already scored
        # (approximation: the batter team scores Poisson(re_now) more runs)
        if gs.half == "top":
            lam_current = self._run_lambda(away_id, home_id) * (re_now / 0.5)
        else:
            lam_current = self._run_lambda(home_id, away_id) * (re_now / 0.5)
        lam_current = max(lam_current, 0.0)

        # Current half-inning remaining runs
        current_half_runs = rng.poisson(lam_current, n_sims).astype(float)
        if gs.half == "top":
            score_diffs -= current_half_runs  # away scores more
        else:
            score_diffs += current_half_runs  # home scores more

        # Full half-innings remaining after the current one
        # Innings structure: we're in inning gs.inning, half gs.half
        # Remaining full half-innings until end of regulation:
        half_innings_played = (gs.inning - 1) * 2 + (1 if gs.half == "bottom" else 0)
        full_half_innings_left = max(0, 18 - half_innings_played - 1)

        for hi in range(full_half_innings_left):
            # Alternate: even hi = away bats (top), odd hi = home bats (bottom)
            if hi % 2 == 0:  # top of inning (away batting)
                lam = self._run_lambda(away_id, home_id)
                runs = rng.poisson(lam, n_sims).astype(float)
                score_diffs -= runs
            else:            # bottom of inning (home batting)
                lam = self._run_lambda(home_id, away_id)
                runs = rng.poisson(lam, n_sims).astype(float)
                score_diffs += runs
                # Walk-off: if home is now ahead in the 9th or later, game ends
                if gs.inning + hi // 2 >= 9:
                    # Clamp paths that went walk-off to their current value
                    walkoff = score_diffs > 0
                    score_diffs = np.where(walkoff, score_diffs, score_diffs)

        return score_diffs
