"""
main.py — project entry point. Runs whatever's toggled on below:
  RUN_EVAL       -> offline policy evaluation over simulated realizations (free)
  RUN_BENCHMARK  -> prediction vs real-odds benchmark (needs a REAL-game seed;
                    ~10 quota per timestamp on The Odds API historical endpoint)
Run from the root:  python main.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make folders importable (no __init__)

import config
from data.loader import load_seed
from data.artificial_odds import generate_kalshi_series
from execution.dp.solver import solve_dp
from execution.dp.policy import run_dp
from execution.lsm.solver import fit_lsm
from execution.lsm.policy import run_lsm
from prediction.backbone import simulate_paths
from prediction.benchmark import benchmark_moneyline, summarize, format_rows
import metrics

# ---- what to run ----
RUN_EVAL = True            # offline; always safe, no network
RUN_BENCHMARK = True      
BENCHMARK_TEAM = "NYK"

# ---- eval settings ----
M_EVAL = 4000
EVAL_SEED = 2024
W_GRID = (2.0, 25.0, 1000.0)


# ================= offline policy evaluation =================
def _fit_runner(combo, seed, wealth):
    ref = seed.reference_data
    if combo.is_single_leg:
        sol = solve_dp(combo.legs[0], seed.timeline, ref, wealth)
        return "DP", (lambda tl, q, s=sol, w=wealth: run_dp(combo, tl, q, s, w))
    sol = fit_lsm(combo, seed.timeline, ref, wealth)
    return "LSM", (lambda tl, q, s=sol, w=wealth: run_lsm(combo, tl, q, s, ref, w))


def _eval_pairs(combo, seed):
    reals = simulate_paths(combo, seed.timeline, seed.reference_data,
                           n_sims=M_EVAL, seed=EVAL_SEED)
    return [(r, generate_kalshi_series(combo, r, seed.reference_data, rng_seed=i))
            for i, r in enumerate(reals)]


def run_eval(seed) -> None:
    print("=== Offline policy evaluation "
          f"({M_EVAL} simulated realizations) ===")
    for combo in seed.combos:
        kind = "DP (single leg)" if combo.is_single_leg else f"LSM ({combo.n_legs} legs)"
        print(f"\n[{combo.id}] {kind}: " + ", ".join(l.describe() for l in combo.legs))
        pairs = _eval_pairs(combo, seed)
        for W in W_GRID:
            label, run_fn = _fit_runner(combo, seed, W)
            rows = [
                metrics.evaluate(label, run_fn, pairs, combo, W),
                metrics.evaluate("hold-to-expiry",
                                 lambda tl, q, w=W: metrics.hold_to_expiry(combo, tl, q, w),
                                 pairs, combo, W),
                metrics.evaluate("threshold(0.85)",
                                 lambda tl, q, w=W: metrics.fixed_threshold(combo, tl, q, 0.85, w),
                                 pairs, combo, W),
                metrics.evaluate("immediate",
                                 lambda tl, q, w=W: metrics.immediate_cashout(combo, tl, q, w),
                                 pairs, combo, W),
            ]
            print(f"  W = {W:g}")
            print(metrics.format_table(rows))


# ================= real-odds prediction benchmark =================
def run_benchmark(seed) -> None:
    print(f"\n=== Prediction benchmark vs real odds (moneyline, {BENCHMARK_TEAM}) ===")
    rows = benchmark_moneyline(seed, BENCHMARK_TEAM)   # uses the real fetch by default
    print(format_rows(rows, BENCHMARK_TEAM))
    s = summarize(rows)
    if s is None:
        print("  no real odds returned for these timestamps. Check:")
        print("   - does your Odds API tier include HISTORICAL odds?  (a 403 means it does NOT)")
        print("   - do these timestamps have snapshots?  (in-play coverage can be sparse; the")
        print("     pregame row is the most reliable, so that gap alone is the key result)")
    else:
        print("  summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def run(seed_path=config.SEED_PATH) -> None:
    seed = load_seed(seed_path)
    final = seed.final_state
    print(f"Game {seed.game_id}: {seed.away_team} @ {seed.home_team} ({seed.date})")
    print(f"  final {seed.home_team} {final.home_score}-{final.away_score} {seed.away_team}\n")

    if RUN_EVAL:
        run_eval(seed)
    if RUN_BENCHMARK:
        run_benchmark(seed)
    else:
        print("\n[benchmark] skipped — set RUN_BENCHMARK=True (needs a real-game seed; "
              "costs ~10 quota/timestamp).")


if __name__ == "__main__":
    run()