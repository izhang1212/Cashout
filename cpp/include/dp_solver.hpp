#pragma once
#include "stern_model.hpp"
#include "bid_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Precomputed Bellman boundary for optimal early exit of an n-leg combo.
//
// The DP solves: V(τ, d) = max( bid(τ, p(τ,d)), E_adj[V(τ-dt, d')] )
// by backward induction on a (tau, score_diff) grid.
//
// Combo modeled as: moneyline × independent prop leg with survival prob q_other.
//   p_combo(τ, d) = P_ml(d, τ) × q_other
//
// When the prop leg resolves mid-game, q_other jumps from the pregame estimate
// to 1.0 (won) or 0.0 (dead).  The correct approach — matching the Python live
// engine — is to precompute two grids (q_other=q_prop before resolution,
// q_other=1.0 after it wins) and switch between them at prop_resolve_tick.
// See DPBoundaryStrategy in strategy.hpp.
//
// risk_aversion: mirrors Python exact_dp.py's risk_adjusted_expectation.
//   0.0 → risk-neutral E[V]  (default, matches plain expected value)
//   λ>0 → certainty-equivalent CE = -(1/λ) * log(E[exp(-λV)])
//          (moves sell boundary earlier; same formula as Python)
//
// Grid parameters:
//   tau:  0 to tau_start in steps of dt_min  →  n_tau = tau_start/dt_min + 1
//   d:   -d_max to +d_max in steps of d_step →  n_d  = 2*d_max/d_step + 1
struct DPGrid {
    int   n_tau;        // number of time steps (index 0 = game over)
    int   n_d;          // score-diff grid size
    float dt_min;       // minutes per time step
    float d_min;        // lowest score_diff on grid
    float d_step;       // grid spacing in points
    float q_other;      // prop-leg survival probability used in solve
    float risk_aversion;

    std::vector<float> values;      // V[tau_idx * n_d + d_idx]
    std::vector<bool>  sell_here;   // true → sell at (tau_idx, d_idx)

    // ── lookup helpers ─────────────────────────────────────────────────────

    int tau_idx(float tau_min) const {
        return std::clamp(static_cast<int>(std::round(tau_min / dt_min)), 0, n_tau - 1);
    }
    int d_idx(float d) const {
        return std::clamp(static_cast<int>(std::round((d - d_min) / d_step)), 0, n_d - 1);
    }

    bool should_sell(float tau_min, float score_diff) const {
        return sell_here[tau_idx(tau_min) * n_d + d_idx(score_diff)];
    }
    float value_at(float tau_min, float score_diff) const {
        return values[tau_idx(tau_min) * n_d + d_idx(score_diff)];
    }

    // Total grid cells (for reporting).
    int64_t total_cells() const { return static_cast<int64_t>(n_tau) * n_d; }
};

// Solve the DP by backward induction.
//
// Transition kernel: exact CDF-bin method, matching Python's
//   SternModel.transition_matrix() (scipy.stats.norm.cdf over bin edges).
//
//   P[i, j] = Phi((edge_hi[j] - mean_i) / sig_step)
//            - Phi((edge_lo[j] - mean_i) / sig_step)
//   where mean_i = d_i + mu*dt, edge_lo[0]=-inf, edge_hi[-1]=+inf.
//
// This correctly absorbs probability mass beyond the grid edges into the
// boundary bins (same as Python) and eliminates the systematic under-estimation
// that the previous 61-point Gaussian quadrature caused near bin boundaries.
//
// risk_aversion matches Python exact_dp.py's risk_adjusted_expectation():
//   0.0 → simple expectation (fast path, no exp/log)
//   λ>0 → exponential-utility certainty equivalent (same formula as Python)
inline DPGrid solve_dp(const SternModel& stern,
                       const BidModel&   bid_model,
                       float tau_start    = 48.0f,
                       float dt_min       = 0.5f,
                       float d_max        = 45.0f,
                       float d_step       = 1.0f,
                       float q_other      = 0.65f,
                       float risk_aversion = 0.0f,
                       int   k_live       = 2)
{
    const int n_tau = static_cast<int>(std::round(tau_start / dt_min)) + 1;
    const int n_d   = static_cast<int>(std::round(2.0f * d_max / d_step)) + 1;

    DPGrid grid;
    grid.n_tau        = n_tau;
    grid.n_d          = n_d;
    grid.dt_min       = dt_min;
    grid.d_min        = -d_max;
    grid.d_step       = d_step;
    grid.q_other      = q_other;
    grid.risk_aversion = risk_aversion;
    grid.values   .assign(n_tau * n_d, 0.0f);
    grid.sell_here.assign(n_tau * n_d, false);

    // ── τ = 0: terminal values (no sell decision possible) ─────────────────
    for (int di = 0; di < n_d; ++di) {
        const float d    = grid.d_min + di * d_step;
        const float p_ml = d > 0.0f ? 1.0f : (d < 0.0f ? 0.0f : 0.5f);
        grid.values[di]  = p_ml * q_other;  // expected terminal payoff
    }

    // ── Precompute exact CDF-based transition matrix (same as Python) ──────
    // P[i*n_d + j] = P(D(t+dt) in bin j | D(t) = grid_i).
    // Computed once; reused for all n_tau-1 backward steps.
    //
    // Bin edges: edge[0]=-inf, edge[j]=d_min+(j-0.5)*d_step for j=1..n_d-1,
    //            edge[n_d]=+inf.  Matches np.concatenate([-inf, d[:-1]+0.5, +inf]).
    const double sigma_step = stern.sigma * std::sqrt(static_cast<double>(dt_min));
    const double mu_step    = stern.mu    * static_cast<double>(dt_min);

    std::vector<double> trans(static_cast<size_t>(n_d) * n_d);
    for (int i = 0; i < n_d; ++i) {
        const double mean_i = (grid.d_min + i * d_step) + mu_step;
        // Precompute n_d+1 edge CDF values for row i.
        // edge[j] for j=0..n_d: -inf, d_min+0.5, d_min+1.5, ..., d_max-0.5, +inf
        double prev_cdf = 0.0;  // CDF(-inf) = 0
        for (int j = 0; j < n_d; ++j) {
            const double edge_hi = (j == n_d - 1)
                ? 1e300
                : (grid.d_min + (j + 0.5) * d_step);
            const double curr_cdf = (j == n_d - 1) ? 1.0
                                  : normal_cdf((edge_hi - mean_i) / sigma_step);
            trans[i * n_d + j] = curr_cdf - prev_cdf;
            prev_cdf = curr_cdf;
        }
    }

    // ── backward induction ─────────────────────────────────────────────────
    const bool   use_ra = (risk_aversion > 0.0f);
    const double lam    = static_cast<double>(risk_aversion);

    for (int ti = 1; ti < n_tau; ++ti) {
        const float tau = ti * dt_min;
        const float* prev = grid.values.data() + (ti - 1) * n_d;

        for (int di = 0; di < n_d; ++di) {
            const float  d       = grid.d_min + di * d_step;
            const float  p_ml    = static_cast<float>(stern.win_prob(d, tau, true));
            const float  p_combo = p_ml * q_other;
            const float  cur_bid = bid_model.bid(p_combo, tau, p_combo, k_live);

            // E_adj[V(ti-1, d')] via exact transition row.
            // Risk-neutral (default): E[V]
            // Risk-averse (λ>0): CE = -(1/λ) * log(E[exp(-λV)])  — same as Python
            const double* row = trans.data() + di * n_d;
            double E_cont;
            if (!use_ra) {
                E_cont = 0.0;
                for (int j = 0; j < n_d; ++j)
                    E_cont += row[j] * prev[j];
            } else {
                double exp_sum = 0.0;
                for (int j = 0; j < n_d; ++j)
                    exp_sum += row[j] * std::exp(-lam * prev[j]);
                E_cont = -std::log(std::max(exp_sum, 1e-300)) / lam;
            }

            const bool sell  = cur_bid > static_cast<float>(E_cont);
            grid.values   [ti * n_d + di] = sell ? cur_bid : static_cast<float>(E_cont);
            grid.sell_here[ti * n_d + di] = sell;
        }
    }

    return grid;
}
