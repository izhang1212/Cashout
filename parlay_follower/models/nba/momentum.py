"""Scoring run / momentum detector.

A "run" is one team outscoring the other by a meaningful margin within a short
rolling window. Runs matter for two distinct sell-signal mechanisms:

1. THE COMEBACK RUN (minimize loss): Your team is down big but suddenly goes on
   a 9-2 run. The market reprices the exit bid UPWARD temporarily. Your model,
   anchored to the score diff + Stern fundamentals, knows this might mean-revert.
   The gap between the spiked market bid and the model's stable continuation value
   crosses the SELL threshold => the signal fires at the temporary peak.

2. THE OPPONENT RUN (protect winnings): Your team is ahead, but the opponent
   just went on a 10-0 run. The Stern model captures the narrowing score diff,
   but momentum adds a forward-looking warning: runs often cluster. The model
   probability is nudged down slightly, lowering the continuation value and
   making the SELL threshold easier to cross before the lead evaporates.

Implementation note: the mean-reversion nudge is intentionally small (max ~6pp).
The main mechanism is already in the DP/LSMC via the changing score diff; this
is a correction for the lag between when a run starts and when the diffusion
model fully prices it in.

UNVALIDATED EMPIRICAL CLAIM: "runs tend to stop" is a specific, contested bet
-- the hot-hand / run-persistence literature does not settle it either way,
and nothing here fit the 6pp figure to data before this comment was written.
Treat it as a hand-tuned prior, not a calibrated parameter, until it clears
the check in tests/backtest/momentum_validation.py (run via
scripts/validate_momentum.py against real historical games). If that check
shows runs continuing at least as often as they revert, shrink or remove this
adjustment rather than keeping it on the strength of the story alone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ...shared.game_feed.game_state import GameState


@dataclass
class ScoringRun:
    hot_team: str          # "home" | "away": which team is on the run
    net_pts: int           # net point swing in the window (always positive)
    home_pts: int          # raw pts by home in the window
    away_pts: int          # raw pts by away in the window
    window_sec: float      # elapsed seconds of the window


@dataclass
class MomentumSignal:
    run: ScoringRun | None
    hot_team: str          # "home" | "away" | "none"
    sell_urgency: float    # 0.0 .. 1.0; higher = stronger SELL pressure
    note: str              # human-readable summary for the signal line


_NEUTRAL = MomentumSignal(run=None, hot_team="none", sell_urgency=0.0, note="")


class MomentumDetector:
    """Stateful: call update(gs) each tick, then query signal() or win_prob_adjustment().

    Parameters
    ----------
    window_sec : rolling window length for run detection (default 2.5 minutes)
    min_run_pts : minimum net point swing in the window to flag a run (default 7)
    """

    def __init__(self, window_sec: float = 150.0, min_run_pts: int = 7):
        self.window_sec = window_sec
        self.min_run_pts = min_run_pts
        # Each entry: (seconds_remaining, home_score, away_score)
        self._history: deque[tuple[float, int, int]] = deque()
        self._latest: GameState | None = None

    def update(self, gs: GameState) -> None:
        """Ingest a new game state snapshot."""
        self._latest = gs
        self._history.append((gs.seconds_remaining, gs.home_score, gs.away_score))
        # Trim snapshots that have left the rolling window.
        # Note: seconds_remaining DECREASES as the game progresses.
        cutoff_sec = gs.seconds_remaining + self.window_sec
        while self._history and self._history[0][0] > cutoff_sec:
            self._history.popleft()

    def signal(self) -> MomentumSignal:
        """Return the current momentum signal based on the rolling window."""
        if self._latest is None or len(self._history) < 2:
            return _NEUTRAL

        gs = self._latest
        oldest_sec, old_home, old_away = self._history[0]

        home_pts_in_window = gs.home_score - old_home
        away_pts_in_window = gs.away_score - old_away
        net = home_pts_in_window - away_pts_in_window   # positive = home on a run
        window_elapsed = oldest_sec - gs.seconds_remaining

        if abs(net) < self.min_run_pts or window_elapsed <= 0:
            return _NEUTRAL

        hot = "home" if net > 0 else "away"
        run = ScoringRun(
            hot_team=hot,
            net_pts=abs(net),
            home_pts=home_pts_in_window,
            away_pts=away_pts_in_window,
            window_sec=window_elapsed,
        )

        # Urgency scales with run magnitude and game lateness.
        # A 7-pt run with 1 min left is far more impactful than with 20 min left.
        tau_min = gs.tau_minutes
        pts_factor = min(abs(net) / 15.0, 1.0)           # 15-pt run = maximal
        time_factor = max(0.25, 1.0 - tau_min / 48.0)    # late game weights more
        urgency = float(pts_factor * time_factor)

        score_diff = gs.score_diff
        note = (
            f"{hot.upper()} on a {abs(net)}-pt run "
            f"(H {home_pts_in_window}-{away_pts_in_window} A, "
            f"last {window_elapsed/60:.1f}m); "
            f"score={gs.home_score}-{gs.away_score} ({score_diff:+d}); "
            f"urgency={urgency:.2f}"
        )
        return MomentumSignal(run=run, hot_team=hot, sell_urgency=urgency, note=note)

    def win_prob_adjustment(self, base_prob: float, bet_side: str) -> float:
        """Return a momentum-adjusted win probability.

        The adjustment is a mean-reversion correction: runs tend to stop, so
        we nudge our model probability slightly toward 0.5 (away from extremes)
        when a run is in progress. This makes the SELL threshold easier to cross
        at the temporary market-price peak.

        When the HOT TEAM is the team we bet on: the market has overpriced us.
        We nudge down slightly (so exit_price > cont_value is more likely => SELL).

        When the HOT TEAM is the OPPONENT: the danger run nudges us down more.
        """
        sig = self.signal()
        if sig.run is None or sig.sell_urgency < 0.05:
            return base_prob

        # Max correction: 6pp in the most urgent situation
        strength = sig.sell_urgency * 0.06

        if sig.hot_team == bet_side:
            # Our side is running hot; market temporarily overpriced us
            adj = -strength * (base_prob - 0.5)
        else:
            # Opponent running hot; danger signal, nudge harder
            adj = -strength * 1.6 * (base_prob - 0.5)

        return float(max(0.01, min(0.99, base_prob + adj)))
