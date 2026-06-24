#pragma once
#include "event.hpp"
#include "event_queue.hpp"
#include "portfolio.hpp"
#include "stern_model.hpp"
#include "bid_model.hpp"
#include "strategy.hpp"
#include <chrono>
#include <random>
#include <vector>

// Synthetic game produced by the path generator.
struct SyntheticGame {
    std::vector<double> score_diffs;  // length = n_ticks + 1 (index 0 = tip-off)
    std::vector<float>  taus;         // tau_min at each tick (descending)
    float               entry_price;
    bool                combo_won;
    // Prop leg resolution: the prop resolves at this tick index (0-indexed).
    // Before that tick, legs_completed = 0; at and after, legs_completed = 1.
    int                 prop_resolve_tick;
    bool                prop_won;
};

// Configuration for a backtest run.
struct BacktestConfig {
    int      n_games           = 10000;
    int      n_ticks_per_game  = 96;     // ticks from tau=48 → tau=0, step=0.5 min
    float    tau_start         = 48.0f;
    float    pregame_spread    = -4.5f;  // home line (negative = home favored)
    float    q_prop            = 0.65f;  // P(prop leg wins) at game start
    float    entry_discount    = 0.80f;  // entry = discount × model FV (simulate alpha)
    uint64_t seed              = 42;
};

// Full report for one strategy over all games.
struct BacktestReport {
    StrategyStats stats;
    double        wall_sec;
    uint64_t      total_events;
};

// ── Engine ─────────────────────────────────────────────────────────────────

class BacktestEngine {
public:
    BacktestEngine(SternModel stern, BidModel bid_model, BacktestConfig cfg)
        : _stern(stern), _bm(bid_model), _cfg(cfg), _rng(cfg.seed) {}

    // Generate n_games synthetic game paths (common random numbers).
    // Must be called before run() or run_all().
    void generate_games() {
        _games.clear();
        _games.reserve(_cfg.n_games);
        std::bernoulli_distribution prop_dist(_cfg.q_prop);
        std::uniform_int_distribution<int> tick_dist(0, _cfg.n_ticks_per_game - 1);

        for (int g = 0; g < _cfg.n_games; ++g) {
            auto path = _stern.simulate_path(
                0.0, _cfg.tau_start, _cfg.n_ticks_per_game, _rng);

            const float p_ml_0 = static_cast<float>(
                _stern.win_prob(path[0], _cfg.tau_start, true));
            const float fv    = p_ml_0 * _cfg.q_prop;
            const float entry = _cfg.entry_discount * fv;

            const double final_d  = path.back();
            const bool   ml_won   = final_d > 0.0;
            const bool   prop_won = prop_dist(_rng);
            const int    prop_tick = tick_dist(_rng);  // prop resolves at random tick

            SyntheticGame sg;
            sg.score_diffs.reserve(path.size());
            for (double d : path) sg.score_diffs.push_back(d);

            sg.taus.reserve(_cfg.n_ticks_per_game + 1);
            const float dt = _cfg.tau_start / _cfg.n_ticks_per_game;
            for (int i = 0; i <= _cfg.n_ticks_per_game; ++i)
                sg.taus.push_back(_cfg.tau_start - i * dt);

            sg.entry_price       = entry;
            sg.combo_won         = ml_won && prop_won;
            sg.prop_resolve_tick = prop_tick;
            sg.prop_won          = prop_won;
            _games.push_back(std::move(sg));
        }
    }

    const std::vector<SyntheticGame>& games() const { return _games; }

    // Run one strategy over all pre-generated games.
    //
    // q_other handling (matches Python live engine):
    //   Before prop_resolve_tick : q_other = q_prop  (prop still live)
    //   At prop_resolve_tick, prop won  : q_other = 1.0  (pure moneyline remains)
    //   At prop_resolve_tick, prop lost : combo is DEAD — engine records the loss
    //     immediately WITHOUT calling strategy.on_tick(), then moves to next game.
    //     This mirrors DecisionEngine._ensure_boundary() rebuilding with q_other=0.
    BacktestReport run(Strategy& strategy) {
        auto t0 = std::chrono::high_resolution_clock::now();

        StrategyStats stats;
        stats.strategy_name = strategy.name();
        stats.results.reserve(_games.size());

        uint64_t ev_count = 0;

        for (uint32_t g = 0; g < static_cast<uint32_t>(_games.size()); ++g) {
            const SyntheticGame& sg = _games[g];
            strategy.on_game_start(g, sg.entry_price);

            GameResult result;
            result.sold_early = false;
            result.combo_won  = sg.combo_won;

            bool exited = false;
            for (int ti = 0; ti <= _cfg.n_ticks_per_game; ++ti) {
                ++ev_count;
                const float tau = sg.taus[ti];
                const float d   = static_cast<float>(sg.score_diffs[ti]);

                // Determine current prop-leg survival probability.
                float q_eff;
                if (ti < sg.prop_resolve_tick) {
                    q_eff = _cfg.q_prop;   // prop not yet resolved
                } else if (sg.prop_won) {
                    q_eff = 1.0f;          // prop won — pure moneyline remains
                } else {
                    // Prop failed — combo is dead.  Record a loss at the bid
                    // (approximately 0) without consulting the strategy.
                    const float dead_bid = _bm.bid(0.0f, tau, 0.0f, 1);
                    result.pnl       = dead_bid - sg.entry_price;
                    result.sell_tau  = tau;
                    result.sold_early = false;  // forced exit, not a strategy decision
                    exited = true;
                    break;
                }

                const float p_ml  = static_cast<float>(_stern.win_prob(d, tau, true));
                const float fv    = p_ml * q_eff;
                const float bid   = _bm.bid(fv, tau, fv, 2);
                const int   legs_c = (ti >= sg.prop_resolve_tick) ? 1 : 0;

                TickView tick{tau, d, fv, bid, sg.entry_price,
                              q_eff,          // q_other — informational for strategies
                              2 - legs_c, legs_c, 2};

                if (tau > 0.0f && strategy.on_tick(tick) == Action::SELL) {
                    result.pnl        = bid - sg.entry_price;
                    result.sell_tau   = tau;
                    result.sold_early = true;
                    exited = true;
                    break;
                }
            }

            if (!exited) {
                result.pnl      = (sg.combo_won ? 1.0f : 0.0f) - sg.entry_price;
                result.sell_tau = 0.0f;
            }

            stats.add(result);
            strategy.on_game_end(g, sg.combo_won);
        }

        auto t1  = std::chrono::high_resolution_clock::now();
        double ws = std::chrono::duration<double>(t1 - t0).count();

        return BacktestReport{std::move(stats), ws, ev_count};
    }

    // Run multiple strategies with common random numbers (fair comparison).
    std::vector<BacktestReport> run_all(std::vector<Strategy*> strategies) {
        std::vector<BacktestReport> reports;
        reports.reserve(strategies.size());
        for (auto* s : strategies)
            reports.push_back(run(*s));
        return reports;
    }

private:
    SternModel                 _stern;
    BidModel                   _bm;
    BacktestConfig             _cfg;
    std::mt19937_64            _rng;
    std::vector<SyntheticGame> _games;
};
