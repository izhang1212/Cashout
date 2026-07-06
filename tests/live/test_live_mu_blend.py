"""Regression guard on the live mu blend's circularity tradeoff.

nba_follower.py's _update_mu_from_market() back-solves mu_implied from the same
Kalshi moneyline price that later gets compared against the DP's
continuation_value in the SELL decision. Blending stern.mu toward mu_implied
therefore pulls the model's fair value toward the market's own price -- the
_MU_LIVE_MAX_WEIGHT cap is what keeps that from fully collapsing model-vs-
market disagreement (see the comment at its definition in nba_follower.py).

These tests don't assert the blend is "correct" (that's a strategy choice,
not a bug) -- they pin down the two properties that make the tradeoff safe:
1. Residual disagreement always survives, even at max blend weight, when the
   pregame and live-implied views genuinely differ.
2. The weight is monotonically increasing in elapsed time and never reaches
   1.0 (a hard cap of exactly 1.0 would let the live market fully own mu,
   erasing the model's ability to disagree with it late in the game).
"""
from __future__ import annotations

from parlay_follower.live.nba_follower import (
    _MU_LIVE_MAX_WEIGHT, _MU_RAMP_HALFLIFE_MIN, _blend_mu, _blend_weight, _implied_mu)


class TestBlendWeight:
    def test_ramps_from_zero(self):
        assert _blend_weight(0.0) == 0.0

    def test_monotone_increasing(self):
        weights = [_blend_weight(m) for m in [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]]
        assert all(b >= a for a, b in zip(weights, weights[1:]))

    def test_capped_below_one(self):
        # Even after a full 48-minute game, live weight never reaches 1.0 --
        # the model always keeps a nonzero say in its own drift.
        assert _blend_weight(48.0) <= _MU_LIVE_MAX_WEIGHT
        assert _blend_weight(48.0) < 1.0

    def test_reaches_cap_by_ramp_halflife(self):
        # elapsed/(elapsed+halflife) = 0.5 at elapsed=halflife, which is also
        # this project's current _MU_LIVE_MAX_WEIGHT -- so the cap binds right
        # at the halflife point. Before it, weight should be meaningfully lower.
        w_before = _blend_weight(_MU_RAMP_HALFLIFE_MIN / 4.0)
        w_at = _blend_weight(_MU_RAMP_HALFLIFE_MIN)
        assert w_before < w_at
        assert w_at == _MU_LIVE_MAX_WEIGHT


class TestBlendMu:
    def test_residual_disagreement_survives_at_max_weight(self):
        # Genuine divergence: pregame model likes home much more than the
        # live market-implied drift does.
        mu_pregame, mu_implied = 0.30, -0.10
        blended = _blend_mu(mu_pregame, mu_implied, _MU_LIVE_MAX_WEIGHT)

        # At the max blend weight the model still contributes (1 - max_weight)
        # of the pregame view -- disagreement narrows, it does not vanish.
        expected = _MU_LIVE_MAX_WEIGHT * mu_implied + (1 - _MU_LIVE_MAX_WEIGHT) * mu_pregame
        assert blended == expected
        assert abs(blended - mu_implied) > 0.01, (
            "blended mu collapsed onto the market-implied value -- the DP has "
            "lost the ability to disagree with the market it trades against")

    def test_zero_weight_is_pure_pregame(self):
        assert _blend_mu(0.30, -0.10, 0.0) == 0.30

    def test_full_weight_is_pure_implied(self):
        # Not reachable via _blend_weight() in practice (capped), but the
        # blend function itself should be a plain linear interpolation.
        assert _blend_mu(0.30, -0.10, 1.0) == -0.10


class TestImpliedMu:
    def test_matches_pregame_style_inversion(self):
        # If the market prob equals the model's own win_prob at (d, tau), the
        # implied mu should recover very close to the drift that produced it.
        import numpy as np
        from scipy.stats import norm

        sigma, tau, d, mu_true = 2.2845, 24.0, 3.0, 0.09375
        p = float(norm.cdf((d + mu_true * tau) / (sigma * np.sqrt(tau))))
        recovered = _implied_mu(p, tau, d, sigma)
        assert abs(recovered - mu_true) < 1e-6

    def test_clipped_to_reasonable_range(self):
        # A near-certain market price shouldn't imply an absurd drift.
        assert _implied_mu(0.999, 10.0, 0.0, 2.2845) <= 1.0
        assert _implied_mu(0.001, 10.0, 0.0, 2.2845) >= -1.0
