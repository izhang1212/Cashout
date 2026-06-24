#pragma once
#include "dp_solver.hpp"
#include <string>

enum class Action { HOLD, SELL };

// Read-only view of one simulation tick passed to strategy callbacks.
// q_other: current survival probability of the prop leg.
//   = q_prop   before the prop resolves (e.g. 0.65)
//   = 1.0      after the prop wins (legs_completed >= 1 and combo still live)
//   Dead-combo ticks (q_other=0) are never passed to on_tick(); the engine
//   short-circuits and records a loss before calling the strategy.
struct TickView {
    float tau_min;
    float score_diff;
    float fair_value;
    float executable_bid;
    float entry_price;
    float q_other;         // prop-leg survival probability at this tick
    int   legs_live;
    int   legs_completed;
    int   legs_total;
};

// Abstract strategy interface.
class Strategy {
public:
    virtual ~Strategy()                           = default;
    virtual Action on_tick(const TickView& tick)  = 0;
    virtual void   on_game_start(uint32_t, float) {}  // game_id, entry_price
    virtual void   on_game_end(uint32_t, bool)    {}  // game_id, combo_won
    virtual std::string name() const              = 0;
};

// ── Concrete strategies ────────────────────────────────────────────────────

// Never exits early; realizes full terminal payoff (or total loss).
struct HoldToResolution : Strategy {
    Action on_tick(const TickView&) override { return Action::HOLD; }
    std::string name() const override { return "hold_to_resolution"; }
};

// Sell when fewer than 24 minutes remain (halftime in the decision framework).
struct SellAtHalftime : Strategy {
    Action on_tick(const TickView& t) override {
        return t.tau_min <= 24.0f ? Action::SELL : Action::HOLD;
    }
    std::string name() const override { return "sell_at_halftime"; }
};

// Sell when bid reaches N× the entry price.
struct SellAtProfitMultiple : Strategy {
    explicit SellAtProfitMultiple(float multiple = 2.0f) : _mult(multiple) {}
    Action on_tick(const TickView& t) override {
        return t.executable_bid >= _mult * t.entry_price ? Action::SELL : Action::HOLD;
    }
    std::string name() const override { return "sell_at_2x"; }
private:
    float _mult;
};

// Sell as soon as the first leg has completed (legs_completed >= 1).
struct SellFirstLegComplete : Strategy {
    Action on_tick(const TickView& t) override {
        return t.legs_completed >= 1 ? Action::SELL : Action::HOLD;
    }
    std::string name() const override { return "sell_first_leg"; }
};

// Optimal-stopping strategy using a precomputed Bellman DP boundary.
// Sells when bid > expected continuation value at (tau, score_diff).
//
// Single-grid mode (backward-compat): uses one boundary throughout the game.
//
// Dual-grid mode (matches Python live engine):
//   grid_pre  — solved with q_other=q_prop (before prop leg resolves)
//   grid_post — solved with q_other=1.0    (after prop leg wins)
//   Switches to grid_post when legs_completed >= 1, i.e. the prop has clinched.
//   Dead-combo ticks never reach on_tick() (engine short-circuits first).
//
// This mirrors DecisionEngine._ensure_boundary() in the Python live engine,
// which rebuilds with an updated q_other whenever a leg status changes.
struct DPBoundaryStrategy : Strategy {
    // Single-grid (static q_other for the whole game).
    explicit DPBoundaryStrategy(DPGrid grid)
        : _grid_pre(std::move(grid)), _dynamic(false) {}

    // Dual-grid (dynamic q_other — matches Python live engine behaviour).
    DPBoundaryStrategy(DPGrid grid_pre, DPGrid grid_post)
        : _grid_pre(std::move(grid_pre))
        , _grid_post(std::move(grid_post))
        , _dynamic(true) {}

    Action on_tick(const TickView& t) override {
        // After the prop clinches, switch to the post-resolution boundary
        // (q_other=1.0 instead of q_prop).  Dead combos (q_other=0) never
        // reach here — the engine exits before calling on_tick().
        const DPGrid& g = (_dynamic && t.legs_completed >= 1) ? _grid_post : _grid_pre;
        return g.should_sell(t.tau_min, t.score_diff) ? Action::SELL : Action::HOLD;
    }
    std::string name() const override {
        return _dynamic ? "dp_boundary_dynamic" : "dp_boundary";
    }
    const DPGrid& grid()      const { return _grid_pre;  }
    const DPGrid& grid_post() const { return _grid_post; }

private:
    DPGrid _grid_pre;
    DPGrid _grid_post;   // only valid when _dynamic == true
    bool   _dynamic;
};
