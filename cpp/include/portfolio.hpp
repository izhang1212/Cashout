#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// Per-game P&L record produced by the backtest engine.
struct GameResult {
    float pnl;          // realized P&L per unit contract
    float sell_tau;     // tau_min at which position was exited (48.0 = held to end)
    bool  sold_early;   // true if strategy triggered a SELL before game over
    bool  combo_won;    // terminal outcome (for hold_to_resolution reference)
};

// Aggregate statistics for one strategy over N games.
struct StrategyStats {
    std::string           strategy_name;
    std::vector<GameResult> results;

    void add(GameResult r) { results.push_back(r); }
    int  n() const { return static_cast<int>(results.size()); }

    float mean_pnl() const {
        if (results.empty()) return 0.0f;
        float s = 0.0f;
        for (auto& r : results) s += r.pnl;
        return s / static_cast<float>(results.size());
    }

    float std_pnl() const {
        if (results.size() < 2) return 0.0f;
        const float m = mean_pnl();
        float sq = 0.0f;
        for (auto& r : results) sq += (r.pnl - m) * (r.pnl - m);
        return std::sqrt(sq / static_cast<float>(results.size() - 1));
    }

    float sharpe() const {
        const float s = std_pnl();
        return s > 0.0f ? mean_pnl() / s : 0.0f;
    }

    float win_rate() const {
        if (results.empty()) return 0.0f;
        int w = 0;
        for (auto& r : results) if (r.pnl > 0.0f) ++w;
        return static_cast<float>(w) / static_cast<float>(results.size());
    }

    // "Bust" = held to end and combo lost (maximum loss realized).
    float bust_rate() const {
        if (results.empty()) return 0.0f;
        int b = 0;
        for (auto& r : results) if (!r.sold_early && !r.combo_won) ++b;
        return static_cast<float>(b) / static_cast<float>(results.size());
    }

    float percentile(float pct) const {
        if (results.empty()) return 0.0f;
        std::vector<float> v;
        v.reserve(results.size());
        for (auto& r : results) v.push_back(r.pnl);
        std::sort(v.begin(), v.end());
        const int idx = std::clamp(
            static_cast<int>(pct * static_cast<float>(v.size())),
            0, static_cast<int>(v.size()) - 1);
        return v[idx];
    }
};
