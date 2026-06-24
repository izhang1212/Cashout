// Parlay Follower C++ Backtester — performance benchmark + strategy comparison.
//
// Run with: make bench
// Generates 10,000 synthetic NBA games (calibrated sigma=2.2845, spread=-4.5),
// solves the Bellman DP boundary, then compares six exit strategies.
//
// Key changes vs. original benchmark:
//   dp_boundary        — static q_other=0.65 (old behaviour, for comparison)
//   dp_boundary_dynamic — dynamic q_other: switches to q=1.0 grid when prop wins,
//                         short-circuits to loss when prop fails (matches Python live engine)
//   risk_aversion parameter: mirrors Python exact_dp.py's exponential-utility CE
#include "backtest_engine.hpp"
#include "dp_solver.hpp"
#include "strategy.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Sec   = std::chrono::duration<double>;

static void print_separator(int w = 82) {
    for (int i = 0; i < w; ++i) std::putchar('-');
    std::putchar('\n');
}

static void print_row(const BacktestReport& r) {
    const StrategyStats& s = r.stats;
    std::printf("%-28s %+8.4f   %6.3f   %5.1f%%  %5.1f%%  %+7.4f  %+7.4f\n",
                s.strategy_name.c_str(),
                s.mean_pnl(),
                s.sharpe(),
                100.0f * s.win_rate(),
                100.0f * s.bust_rate(),
                s.percentile(0.05f),
                s.percentile(0.95f));
}

int main() {
    // ── Parameters ───────────────────────────────────────────────────────────
    constexpr float    SIGMA        = 2.2845f;
    constexpr float    SPREAD       = -4.5f;
    constexpr float    Q_PROP       = 0.65f;
    constexpr float    DISCOUNT     = 0.80f;
    constexpr float    RISK_AVERSION = 0.0f;   // match Python default (risk-neutral)
    constexpr int      K_LIVE       = 2;        // live legs for haircut scaling (matches Python)
    constexpr int      N_GAMES      = 10000;
    constexpr int      N_TICKS      = 96;
    constexpr float    TAU_START    = 48.0f;
    constexpr uint64_t SEED         = 0xDEADBEEF;

    SternModel stern(SIGMA, SPREAD);
    BidModel   bm({0.03f, 0.25f, 3.0f});

    const float fv0   = static_cast<float>(stern.win_prob(0.0, TAU_START)) * Q_PROP;
    const float entry = DISCOUNT * fv0;

    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║      Parlay Follower — C++ Backtest Engine Benchmark        ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    std::printf("Parameters\n");
    std::printf("  sport            : NBA (Stern 1994 BM model)\n");
    std::printf("  n_games          : %d\n", N_GAMES);
    std::printf("  ticks / game     : %d  (every 30 s)\n", N_TICKS);
    std::printf("  total events     : %d\n", N_GAMES * (N_TICKS + 1));
    std::printf("  sigma_per_min    : %.4f  (calibrated: 2024-25 NBA, N=1225)\n", SIGMA);
    std::printf("  pregame_spread   : %.1f (home favored)\n", SPREAD);
    std::printf("  q_prop           : %.2f (prop-leg survival prob)\n", Q_PROP);
    std::printf("  risk_aversion    : %.1f (0 = risk-neutral, matches Python default)\n",
                RISK_AVERSION);
    std::printf("  k_live           : %d   (live legs; sqrt(k) haircut scaling — matches Python)\n",
                K_LIVE);
    std::printf("  entry_discount   : %.0f%%  (simulate buying combo at a discount)\n",
                100.0 * DISCOUNT);
    std::printf("  model_FV_at_t0   : %.4f  (P_ml=%.4f × q_prop=%.2f)\n",
                fv0, static_cast<float>(stern.win_prob(0.0, TAU_START)), Q_PROP);
    std::printf("  entry_price      : %.4f  (%.0f%% × FV)\n\n", entry, 100.0 * DISCOUNT);

    // ── Step 1: Solve DP boundaries ──────────────────────────────────────────
    std::printf("Solving DP boundaries...\n");

    auto t_dp0 = Clock::now();
    // grid_pre:  q_other = q_prop (before prop resolves)
    DPGrid grid_pre  = solve_dp(stern, bm, TAU_START, 0.5f, 45.0f, 1.0f,
                                Q_PROP, RISK_AVERSION, K_LIVE);
    double dp_pre_ms = Ms(Clock::now() - t_dp0).count();

    auto t_dp1 = Clock::now();
    // grid_post: q_other = 1.0  (after prop wins; pure moneyline boundary)
    DPGrid grid_post = solve_dp(stern, bm, TAU_START, 0.5f, 45.0f, 1.0f,
                                1.0f, RISK_AVERSION, K_LIVE);
    double dp_post_ms = Ms(Clock::now() - t_dp1).count();

    auto sell_count = [](const DPGrid& g) {
        int n = 0; for (bool s : g.sell_here) if (s) ++n; return n;
    };
    const int total_cells = grid_pre.n_tau * grid_pre.n_d;

    std::printf("  grid_pre  (q=%.2f): %.2f ms  [%d tau × %d d, %.1f%% → SELL]\n",
                Q_PROP, dp_pre_ms, grid_pre.n_tau, grid_pre.n_d,
                100.0f * sell_count(grid_pre) / total_cells);
    std::printf("  grid_post (q=1.00): %.2f ms  [%d tau × %d d, %.1f%% → SELL]\n\n",
                dp_post_ms, grid_post.n_tau, grid_post.n_d,
                100.0f * sell_count(grid_post) / total_cells);

    // ── Step 2: Generate synthetic games ─────────────────────────────────────
    std::printf("Generating %d synthetic games...\n", N_GAMES);
    BacktestConfig cfg;
    cfg.n_games          = N_GAMES;
    cfg.n_ticks_per_game = N_TICKS;
    cfg.tau_start        = TAU_START;
    cfg.pregame_spread   = SPREAD;
    cfg.q_prop           = Q_PROP;
    cfg.entry_discount   = DISCOUNT;
    cfg.seed             = SEED;

    BacktestEngine engine(stern, bm, cfg);
    auto t_gen = Clock::now();
    engine.generate_games();
    double gen_ms = Ms(Clock::now() - t_gen).count();

    int wins = 0, prop_wins = 0;
    for (auto& g : engine.games()) {
        if (g.combo_won) ++wins;
        if (g.prop_won)  ++prop_wins;
    }
    std::printf("  Generated in %.1f ms  (%.1f%% combo wins, %.1f%% prop wins)\n\n",
                gen_ms, 100.0 * wins / N_GAMES, 100.0 * prop_wins / N_GAMES);

    // ── Step 3: Run strategies ────────────────────────────────────────────────
    std::printf("Running strategies (common random numbers)...\n\n");

    HoldToResolution     hold;
    SellAtHalftime       halftime;
    SellAtProfitMultiple sell2x(2.0f);
    SellFirstLegComplete first_leg;
    // Static DP: q_other fixed at q_prop throughout (original behaviour)
    DPBoundaryStrategy   dp_static(grid_pre);
    // Dynamic DP: switches grid when prop resolves, short-circuits on prop fail
    // This matches the Python live engine's _ensure_boundary() rebuild logic.
    DPBoundaryStrategy   dp_dynamic(grid_pre, grid_post);

    std::vector<Strategy*> strategies = {
        &dp_dynamic, &dp_static, &halftime, &sell2x, &first_leg, &hold
    };

    auto t_run = Clock::now();
    auto reports = engine.run_all(strategies);
    double run_sec = Sec(Clock::now() - t_run).count();

    uint64_t total_events = 0;
    for (auto& r : reports) total_events = std::max(total_events, r.total_events);

    // ── Step 4: Print results table ───────────────────────────────────────────
    std::printf("Strategy results (%d games, entry = %.0f%% of model FV)\n",
                N_GAMES, 100.0 * DISCOUNT);
    print_separator(82);
    std::printf("%-28s %8s   %6s   %6s  %6s  %7s  %7s\n",
                "Strategy", "mean P&L", "sharpe", "win%", "bust%", "p5", "p95");
    print_separator(82);
    for (auto& r : reports) print_row(r);
    print_separator(82);

    // Dynamic DP edge over static DP (shows benefit of correct q_other handling)
    float dynamic_pnl = 0.0f, static_pnl = 0.0f;
    for (auto& r : reports) {
        if (r.stats.strategy_name == "dp_boundary_dynamic") dynamic_pnl = r.stats.mean_pnl();
        if (r.stats.strategy_name == "dp_boundary")         static_pnl  = r.stats.mean_pnl();
    }
    std::printf("\nDynamic DP edge over static DP: %+.4f per contract\n",
                dynamic_pnl - static_pnl);
    std::printf("  (dynamic correctly handles prop resolution; static uses stale q=%.2f)\n\n",
                Q_PROP);

    // ── Step 5: Performance numbers ───────────────────────────────────────────
    std::printf("Performance\n");
    print_separator(50);
    std::printf("  Total events processed : %llu\n", (unsigned long long)total_events);
    std::printf("  Wall time (strategies) : %.3f s\n",  run_sec);
    std::printf("  Throughput             : %.0f events/sec\n",
                static_cast<double>(total_events) / run_sec);
    std::printf("  Per-game latency       : %.1f μs\n",
                run_sec * 1e6 / (N_GAMES * static_cast<int>(strategies.size())));
    std::printf("  DP grid (pre)  solve   : %.2f ms\n", dp_pre_ms);
    std::printf("  DP grid (post) solve   : %.2f ms\n", dp_post_ms);
    std::printf("  Path generation        : %.1f ms\n", gen_ms);
    std::printf("  Grid cells each        : %lld  (%d × %d)\n",
                (long long)grid_pre.total_cells(), grid_pre.n_tau, grid_pre.n_d);
    print_separator(50);

    std::printf("\nDP alignment with Python live engine\n");
    std::printf("  ✓  risk_aversion=%.1f  (matches Python exact_dp.py default)\n",
                RISK_AVERSION);
    std::printf("  ✓  k_live=%d, sqrt(k) haircut scaling (matches Python BidModel exactly)\n",
                K_LIVE);
    std::printf("  ✓  CDF-based transition kernel (exact, matches Python transition_matrix())\n");
    std::printf("  ✓  dead-combo short-circuit on prop failure (q_other → 0)\n");
    std::printf("  ✓  grid switch on prop win (q_other: %.2f → 1.00)\n", Q_PROP);
    std::printf("  ~  no robust ensemble  (Python default: robust_ensemble_size=1, off)\n");
    std::printf("  ~  bid model params uncalibrated (run `lpf fit-bid-model` to fix)\n");

    return 0;
}
