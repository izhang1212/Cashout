#pragma once
#include <cmath>
#include <random>
#include <vector>

// Normal CDF via erfc — portable across all C++17 implementations.
inline double normal_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Stern (1994) Brownian motion model for in-game NBA score differential.
//
// D(t) = score_diff at time t, evolves as:
//   dD = mu * dt + sigma * dW,   D(0) = pregame_expected_margin
//
// win_prob returns P(home wins | D(t)=score_diff, tau_min remaining).
//
// Parameters calibrated from 2024-25 NBA regular season (N=1225 games):
//   sigma = 2.2845  pts/min  (game-level final-margin std / sqrt(48))
//   mu    = -spread / 48.0   (spread=-4.5 → mu=0.09375 pts/min home advantage)
struct SternModel {
    double sigma;   // score-diff volatility in pts/min
    double mu;      // drift in pts/min (positive → home favored)

    // pregame_spread: Vegas home line (negative = home favored, e.g. -4.5).
    SternModel(double sigma_per_min, double pregame_spread)
        : sigma(sigma_per_min)
        , mu(-pregame_spread / 48.0) {}

    // P(home wins | current state).
    double win_prob(double score_diff, double tau_min,
                    bool home_side = true) const {
        if (tau_min <= 0.0) {
            double p = score_diff > 0.0 ? 1.0 : (score_diff < 0.0 ? 0.0 : 0.5);
            return home_side ? p : 1.0 - p;
        }
        double mean   = score_diff + mu * tau_min;
        double stddev = sigma * std::sqrt(tau_min);
        double p      = normal_cdf(mean / stddev);
        return home_side ? p : 1.0 - p;
    }

    // Euler-Maruyama BM path: returns score_diff at n_steps+1 points.
    // path[0] = d0 (at tau_start), path[n_steps] = final diff.
    std::vector<double> simulate_path(double d0, double tau_start, int n_steps,
                                      std::mt19937_64& rng) const {
        const double dt      = tau_start / n_steps;
        const double sqrt_dt = std::sqrt(dt);
        std::normal_distribution<double> dist(0.0, 1.0);

        std::vector<double> path(n_steps + 1);
        path[0] = d0;
        for (int i = 1; i <= n_steps; ++i)
            path[i] = path[i-1] + mu * dt + sigma * sqrt_dt * dist(rng);
        return path;
    }
};
