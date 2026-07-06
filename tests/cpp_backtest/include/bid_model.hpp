#pragma once
#include <cmath>
#include <algorithm>

// Market-maker bid model: M(t) = fv * (1 - h)
//
// Haircut formula (mirrors parlay_follower/cashout/bid_model.py exactly):
//   h = a + b * (1 - p_combo) * exp(-c * tau_frac) * sqrt(k_live)
//
// where tau_frac = tau_min / 48.0.
//
// Interpretation:
//   a              — irreducible execution cost (bid-ask half-spread floor)
//   b * (1-p)      — uncertainty premium: wider spread for lower-probability combos
//   exp(-c*τ)      — haircut widens late-game (liquidity dries up, adverse-selection peaks)
//   sqrt(k_live)   — more live legs → wider spread (independent uncertainty scaling)
//
// Parameters are placeholders; calibrate from real Kalshi bid data via
// `lpf fit-bid-model` after collecting logs with `scripts/log_bids.py`.
struct BidModelParams {
    float a = 0.03f;   // base haircut
    float b = 0.25f;   // uncertainty premium coefficient
    float c = 3.00f;   // time-decay rate
};

struct BidModel {
    BidModelParams p;

    explicit BidModel(BidModelParams params = {}) : p(params) {}

    // Returns the executable bid price for a combo position.
    //   p_combo   : model fair-value probability
    //   tau_min   : minutes remaining in game
    //   fv_mm     : market-maker fair value (usually same as p_combo here)
    //   k_live    : number of live legs (sqrt scaling on haircut — matches Python BidModel)
    float bid(float p_combo, float tau_min, float fv_mm, int k_live) const {
        const float tau_frac = std::clamp(tau_min / 48.0f, 0.0f, 1.0f);
        const float sqrt_k   = std::sqrt(static_cast<float>(std::max(k_live, 1)));
        const float h = std::clamp(
            p.a + p.b * (1.0f - p_combo) * std::exp(-p.c * tau_frac) * sqrt_k,
            0.0f, 0.95f);
        return std::clamp(fv_mm * (1.0f - h), 0.0f, 1.0f);
    }

    // Fair value haircut at given state (k_live=1 convenience overload for DP solver).
    float haircut(float p_combo, float tau_min, int k_live = 1) const {
        const float tau_frac = std::clamp(tau_min / 48.0f, 0.0f, 1.0f);
        const float sqrt_k   = std::sqrt(static_cast<float>(std::max(k_live, 1)));
        return std::clamp(
            p.a + p.b * (1.0f - p_combo) * std::exp(-p.c * tau_frac) * sqrt_k,
            0.0f, 0.95f);
    }
};
