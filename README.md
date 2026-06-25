# Live Parlay Follower
Real-time optimal cash-out timing for Kalshi sports combos (NBA and MLB) via exact dynamic programming.

## What it does

Given a multi-leg position ("combo") held on Kalshi, this system:

1. **Auto-discovers** the position — contracts, cost basis, leg structure, and current exit price — directly from the Kalshi API. No manual entry.
2. **Follows the live game** tick-by-tick from tip-off to final buzzer (NBA) or first pitch to last out (MLB), tracking each leg's status in real time.
3. **Re-prices the position** at each tick: calibrated per-leg win probabilities, joint combo fair value, the live executable exit bid, and the expected value of continuing to hold.
4. **Emits a SELL signal** the instant the market bid crosses the precomputed, provably optimal exercise boundary — with a one-line explanation and live P&L.

Alert-only by default. The system advises; the human executes.

## Inspiration

Holding a tradeable parlay is an **optimal stopping problem** — the same mathematical category as American option early exercise. At every instant the holder owns a claim with some model fair value, faces a live market bid (the cash-out offer), and must decide whether the bid exceeds the expected value of continuing to hold.

Because the game state is low-dimensional and discrete, this stopping problem can be solved **exactly** by dynamic programming (Bellman backward induction on a state grid) — no approximation needed for the core case. The result is a precomputed exercise boundary: a surface in (time remaining, score lead) space that tells you, for every possible game state, whether the current bid is worth taking. Approximation (Longstaff–Schwartz Monte Carlo) enters only when player-prop legs push the state space beyond what a grid can hold.

## Data sources

| Source | What we pull | Used for |
|---|---|---|
| **Kalshi REST API** (authenticated, RSA-PSS signed) | Open positions, contracts, cost basis, combo leg tickers, live order-book bids and depth, RFQ exit quotes | Position discovery, live exit pricing, haircut model calibration |
| **NBA Stats API** (`nba_api`, free) | Live box scores, play-by-play, player minutes/stats/fouls, team pace and ratings | Live game state, per-leg win probabilities, foul trouble and momentum adjustments |
| **MLB Stats API** (`mlb-statsapi`, free) | Live linescore (inning, outs, runners on base, score), play-by-play, player batting/pitching stats | Live MLB game state, win expectancy, player prop projections |

## Methods and models

### Win probability

**NBA — Stern (1994) Brownian motion model.** Score differential is modeled as Brownian motion with drift calibrated to the pregame spread: `D(t) ~ BM(μ, σ)`. Win probability is a closed-form normal CDF in (lead, time remaining). Supplies closed-form transition probabilities for the DP grid and is augmented by:
- **Foul trouble weighting** — player importance (usage × minutes share) scales the win-probability impact of star players sitting with fouls.
- **Scoring run detection** — a rolling 2.5-minute window flags momentum runs ≥ 7 net points and nudges the model toward mean reversion, triggering SELL at the temporary market-price peak.
- **Pace-aware totals** — Bayesian blend of current-game scoring rate with team season pace; Q4 adjustments for clock management (close games) and intentional fouling (blowouts).
- **Live drift update** — each tick back-solves an implied drift `μ` from the Kalshi moneyline market price and blends it with the pregame estimate, forcing a boundary rebuild when the shift is material.

**MLB — Per-game run total model.** Each leg in an MLB combo is a "total runs over" bet on a single game. For each game the system polls the live linescore (inning, outs, runners, score) and projects the final run total as: current runs + blended per-half-inning scoring rate × half-innings remaining, where the blending weight shifts from season team pace toward the current-game pace as innings accumulate. The projection is treated as Normal with Poisson-scaled variance, giving `P(final total > line)` in closed form. All games are polled simultaneously; legs from different games are treated as independent (copula ρ ≈ 0.05).

### Joint modeling (correlation)

Same-game legs are correlated. Joint resolution is modeled with a **Gaussian copula** whose correlation matrix is estimated from historical co-resolution of comparable leg pairs. State-conditional correlation (close vs. blowout, early vs. late) is maintained as a lookup table.

Monte Carlo simulation of joint forward paths yields the combo's full terminal payoff distribution. The mean is the model fair value; the full distribution feeds the risk adjustment.

### Bid model

The live exit bid is modeled as:

```
M(t) = F_mm(t) × (1 − h(τ, p, k))
```

where `F_mm` is a market-maker fair value (market-implied legs through the copula), and `h` is a haircut function of time remaining `τ`, combo probability `p`, and number of live legs `k`. Parameters are fit from logged real combo bids. The policy always optimizes against **depth-weighted executable proceeds** for the actual position size, not the top-of-book price.

### Decision engine

**Exact DP (single game-outcome leg).** The Bellman equation is solved by backward induction over a discretized (time, score-diff) grid:

```
V(t, s) = max( M(t, s),  E[ V(t+1, s′) | s ] )
```

with transitions from the Stern diffusion (closed form). The result is a precomputed exercise boundary — live operation is a fast table lookup.

**LSMC (n-leg or prop combos).** When player-prop legs add continuous dimensions, the system switches to Longstaff–Schwartz Monte Carlo: simulate joint forward paths, regress realized continuation payoffs on a polynomial basis of the state vector `[τ, score_diff, n_completed, combo_prob, momentum]`, and use the fitted regression as the boundary. The same Bellman logic, approximated — the industry technique for Bermudan swaption pricing.

**Robust ensemble.** The DP runs under a small ensemble of perturbed models. HOLD requires unanimity; any SELL from any ensemble member triggers a SELL. This is the system's built-in acknowledgment that model probabilities are estimates.

**Shrinkage.** Each leg's model probability is shrunk toward the Kalshi market-implied probability in proportion to demonstrated edge. Until the paper-trading log proves the model beats the market on a leg type, the market gets the greater weight — biasing toward earlier, safer exits.

## Output

Every tick the system prints a status line. When the boundary is crossed it emits a SELL signal:

```
[12:34:07] HOLD  | fv=0.412  bid=0.380  cont=0.431  margin=-0.051
           2/3 legs clinched, 8.2m left, via lsmc_nleg, exit=order_book
           HOME on a 9-pt run (H 9-0 A, last 2.1m); urgency=0.74

[12:41:22] SELL  | fv=0.389  bid=0.445  cont=0.391  margin=+0.054
           2/3 legs clinched, 4.1m left, via lsmc_nleg, exit=order_book
           Leg 2 clinched; executable bid exceeds continuation value;
           5/5 ensemble members agree. P&L: +$0.82 on $0.50 cost basis.
```

Fields:
- `fv` — model fair value of the combo
- `bid` — current executable exit price per contract
- `cont` — estimated continuation value (expected value of holding one more step)
- `margin` — `bid − cont` (positive = SELL is favored)

## Architecture

The project has two independent components that share the same mathematical models but serve different purposes:

**C++ offline engine** (`cpp/`) — throughput-bound batch work, no Kalshi dependency:
- Bellman DP solver: precomputes the exercise boundary on a 97 × 91 (time, score-diff) grid in **0.89 ms** using an exact CDF-based transition matrix (matches Python's `SternModel.transition_matrix()` exactly).
- Backtester: replays 10K synthetic games against six strategies at **~10M events/sec** on a single core.
- Verified to produce identical DP values as Python: V(24, 0) = 0.3760 in both.
- 60 unit tests, no external framework.
- Used for offline strategy validation and parameter calibration.

**Python live engine** (`parlay_follower/`) — network-bound orchestration of a single real game:
- Watches one live game tick-by-tick with **adaptive polling**: 1 s in the final 3 game-minutes (crunch time), 2 s otherwise.
- Runs its own DP/LSMC in Python — correct choice because the live session is I/O-bound, the DP solve takes ~1 ms in NumPy, and the Python DP has session-specific behaviour the offline C++ solver doesn't need: it rebuilds every 5 game-minutes as legs resolve and as live Kalshi moneyline prices update the drift estimate `μ`.
- Fires HOLD/SELL alerts; the human executes on Kalshi.

The two components deliberately remain independent. The Python DP rebuilds dynamically mid-game (not a static precomputed table), uses risk-adjusted expectations and a robust model ensemble, and starts from the current game clock rather than tip-off — features that belong in the live session, not the offline benchmarker. The shared ground is `config/settings.yaml` (same calibrated parameters) and identical mathematical formulas.

### C++ strategy comparison

Results from `make bench` (N=10K games, σ=2.2845, spread=−4.5, entry=80% of model FV, k\_live=2):

| Strategy | mean P&L | Sharpe | win% | bust% |
|---|---|---|---|---|
| dp\_boundary\_dynamic | +0.0853 | 0.174 | 40.3% | 53.7% |
| dp\_boundary | +0.0853 | 0.174 | 40.3% | 53.7% |
| sell\_at\_2x | +0.0648 | 0.163 | 49.4% | 50.6% |
| sell\_first\_leg | +0.0615 | 0.171 | 52.2% | 35.0% |
| sell\_at\_halftime | +0.0579 | 0.206 | 57.1% | 17.7% |
| hold\_to\_resolution | +0.0853 | 0.174 | 40.3% | 59.7% |

`dp_boundary_dynamic` switches from the pre-resolution grid (q\_prop=0.65) to a post-resolution grid (q=1.0) when the prop leg wins, and short-circuits to loss when it fails — matching the Python live engine's grid-rebuild logic.

With the current placeholder bid-model parameters, the DP correctly identifies no profitable mid-game exits and matches `hold_to_resolution` exactly. The boundary becomes non-trivial once real Kalshi bid logs are used to calibrate the haircut parameters via `lpf fit-bid-model`.

## Project layout

```
live-parlay-follower/
├── cpp/                   # C++ offline backtester + DP solver (no Kalshi dependency)
│   ├── include/           # Header-only library: event queue, Stern model, bid model,
│   │   │                  #   DP solver (Bellman backward induction), strategy interface,
│   │   └──                #   portfolio stats, backtest engine
│   ├── tests/             # 60 unit tests (no external framework)
│   ├── bench/             # Benchmark: 10K games, ~10M events/sec, 6-strategy comparison
│   ├── Makefile           # Build: make test / make bench
│   └── CMakeLists.txt     # CMake config (pybind11 binding stubbed for future use)
├── parlay_follower/
│   ├── nba/               # NBA-specific: feed, stats, Stern model, foul/momentum/player context, follower
│   ├── mlb/               # MLB-specific: feed, stats, win model, player props, context, follower
│   ├── probability/       # Shared math: copula, shrinkage, Monte Carlo, LSMC paths, Stern model
│   ├── decision/          # Shared optimizer: exact DP, LSMC, robust ensemble, bid model, signal
│   ├── game_feed/         # Shared game state, Leg/LegStatus types, resolvers
│   ├── account/           # Kalshi auth (RSA-PSS) and REST client
│   ├── market_data/       # Order-book pricing, RFQ, exit quote dispatch, bid logger
│   ├── backtest/          # Tick-by-tick replay, policy comparison, P&L metrics
│   └── cli.py             # lpf command-line entry point
├── config/settings.yaml   # Model, decision, and bid-model parameters
├── scripts/               # Standalone bid logger, day-one API recon
└── tests/
```

## Quickstart

```bash
pip install -e .
cp .env.example .env      # fill in KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH

lpf positions             # discover your combo ticker and cost basis
lpf inspect --ticker KXNBACOMBO-...   # show exit value, leg probs, P&L

# Follow a live NBA game
lpf follow --sport nba --ticker KXNBACOMBO-... --game-id 0042500404 --spread -4.5 \
    --leg "moneyline:side=home@KXNBA-...-ML" \
    --leg "total_over:line=224.5@KXNBA-...-TOTAL"

# Follow MLB cross-game totals (typical use: total runs over in 3-4 separate games)
lpf follow --sport mlb --ticker KXMLBCOMBO-... \
    --game-id 745528,745529,745530 \
    --leg "total_over:line=8.5,game=745528@KXMLB-...-G1TOTAL" \
    --leg "total_over:line=7.5,game=745529@KXMLB-...-G2TOTAL" \
    --leg "total_over:line=9.0,game=745530@KXMLB-...-G3TOTAL"
```

## References
- [Bellman DP](https://en.wikipedia.org/wiki/Bellman_equation) 
- [Longstaff-Schwartz LSMC](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf)
