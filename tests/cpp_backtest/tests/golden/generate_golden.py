"""Generate the cross-language DP golden reference file.

Solves the Python exact_dp boundary with the same canonical parameters used by
tests/cpp_backtest/tests/test_main.cpp and tests/cpp_backtest/bench/bench_main.cpp
(sigma=2.2845, spread=-4.5, tau_start=48, dt=0.5min, diff_range=45, q_other=0.65,
risk_aversion=0.0, k_live=2, moneyline_side="home"), then dumps the full
(tau, score_diff) grid to tests/cpp_backtest/tests/golden/dp_grid_golden.csv.

tests/cpp_backtest/tests/test_main.cpp reads this file and asserts that C++'s
solve_dp() produces the same V grid and exercise boundary -- the automated
check that replaces the old single hand-verified spot check
(V(24,0) = 0.3760 in both).

Run via `make golden` from tests/cpp_backtest/, or directly:
    python3 tests/cpp_backtest/tests/golden/generate_golden.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from parlay_follower.cashout.bellman.exact_dp import solve  # noqa: E402
from parlay_follower.cashout.bid_model import BidModel  # noqa: E402
from parlay_follower.shared.stern import SternModel  # noqa: E402

# ── Canonical parameters (must match test_main.cpp / bench_main.cpp) ─────────
SIGMA          = 2.2845
SPREAD         = -4.5
TAU_START_MIN  = 48.0
DT_MIN         = 0.5
D_MAX          = 45.0
D_STEP         = 1.0
Q_OTHER        = 0.65
RISK_AVERSION  = 0.0
K_LIVE         = 2
MONEYLINE_SIDE = "home"

OUT_PATH = os.path.join(os.path.dirname(__file__), "dp_grid_golden.csv")


def main() -> None:
    stern = SternModel(sigma_per_min=SIGMA, pregame_spread=SPREAD)
    bid_model = BidModel()

    result = solve(
        stern, bid_model,
        tau_start_min=TAU_START_MIN,
        moneyline_side=MONEYLINE_SIDE,
        q_other=lambda tau: Q_OTHER,
        k_live=K_LIVE,
        dt_min=DT_MIN,
        diff_range=int(D_MAX),
        risk_aversion=RISK_AVERSION,
    )

    n_tau = int(round(TAU_START_MIN / DT_MIN)) + 1
    n_d = int(round(2.0 * D_MAX / D_STEP)) + 1

    with open(OUT_PATH, "w") as f:
        # Header: parameters the C++ side must reconstruct identically.
        f.write(f"{n_tau},{n_d},{DT_MIN},{-D_MAX},{D_STEP},{Q_OTHER},"
                f"{RISK_AVERSION},{K_LIVE},{SIGMA},{stern.mu}\n")
        # Rows in C++ storage order: idx = tau_idx * n_d + d_idx,
        # tau_idx=0 -> tau=0 (terminal), d_idx=0 -> d=-D_MAX.
        for tau_idx in range(n_tau):
            tau = tau_idx * DT_MIN
            for d_idx in range(n_d):
                d = -D_MAX + d_idx * D_STEP
                sell, value, _bid = result.lookup(tau, d)
                f.write(f"{value:.10g},{int(sell)}\n")

    print(f"Wrote {n_tau * n_d} grid cells to {OUT_PATH}")
    sell0, cont0, _ = result.lookup(24.0, 0.0)
    print(f"Sanity check: V(24, 0) = {cont0:.4f} (sell={sell0})")


if __name__ == "__main__":
    main()
