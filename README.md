# Live Parlay Follower

Optimal cash-out timing for Kalshi NBA combos via **exact dynamic programming**.

Given a multi-leg NBA position ("combo") on Kalshi, this system follows the game
live, re-prices the position every few seconds, and tells you the optimal moment
to sell — solving the same optimal stopping problem as American option exercise,
exactly, by Bellman backward induction. Full design rationale lives in
`docs/Live_Parlay_Follower_Project_Spec_v2.docx`.

**Alert-only by default.** The system advises; the human executes. It never
places orders.

## Project layout

```
live-parlay-follower/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example                  # Kalshi API credentials template
├── config/
│   └── settings.yaml             # model, decision, bid-model parameters
├── docs/
│   └── Live_Parlay_Follower_Project_Spec_v2.docx
├── data/logs/                    # bid + game logs (gitignored; calibration fuel)
├── scripts/
│   ├── day_one_recon.py          # lock the Kalshi data contract (run FIRST)
│   └── log_bids.py               # standalone bid logger for games you don't hold
├── src/parlay_follower/
│   ├── config.py                 # settings.yaml + .env loader
│   ├── cli.py                    # `lpf` entry point
│   ├── account/
│   │   ├── auth.py               # RSA-PSS request signing (3 headers, ms timestamps)
│   │   └── kalshi_client.py      # positions, fills, order books, multivariate, RFQ
│   ├── market_data/
│   │   ├── orderbook.py          # depth-weighted executable proceeds (not top-of-book)
│   │   └── bid_logger.py         # persistent bid+state logging -> haircut calibration
│   ├── game_feed/
│   │   ├── game_state.py         # state vector, Leg, deterministic resolvers
│   │   └── nba_feed.py           # nba_api live polling
│   ├── probability/
│   │   ├── stern.py              # Brownian-motion win prob + DP transition kernel
│   │   ├── calibration.py        # isotonic recalibration, Brier, reliability curves
│   │   ├── copula.py             # Gaussian copula, state-conditional correlation
│   │   ├── monte_carlo.py        # combo fair value + full payoff distribution
│   │   └── shrinkage.py          # shrink-to-market + EdgeLedger (paper-trading gate)
│   ├── decision/
│   │   ├── bid_model.py          # empirical haircut h(tau, p, k); fit from logs
│   │   ├── threshold_policy.py   # tuned baseline the DP must beat
│   │   ├── exact_dp.py           # Bellman backward sweep -> precomputed boundary
│   │   ├── lsmc.py               # Longstaff–Schwartz branch for player-prop legs
│   │   ├── robust.py             # ensemble DP: HOLD requires unanimity
│   │   └── signal.py             # HOLD/SELL signal + explanation
│   ├── backtest/
│   │   ├── replay.py             # tick-by-tick replay; common-random-number studies
│   │   ├── policies.py           # hold-to-end, first-leg, halftime, profit-multiple
│   │   └── metrics.py            # mean P&L, Sharpe-like, bust rate, signal hit rate
│   └── live/
│       ├── follower.py           # orchestration loop (alert-only)
│       └── alerts.py             # console / loud-console sinks
└── tests/                        # stern, DP, order book, copula/MC, leg resolvers
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Generate an API key pair at Kalshi: Settings -> API.
# Save the private key PEM, fill in .env. Start with KALSHI_ENV=demo.
```

## Order of operations

**1. Day-one recon (do this before anything else).**

```bash
python scripts/day_one_recon.py
```

Answers the project's one open question — are combo order books queryable via
public market-data endpoints, or RFQ-only? Lock the findings into
`docs/data_contract.md` and resolve every `RECON:` comment in the code.

**2. Sanity-check the engine on synthetic games.**

```bash
lpf backtest --spread -4.5 --q-other 0.6 --entry 0.30 --n-games 5000
```

Prints a policy comparison table (exact DP vs. hold-to-end vs. naive rules) and
saves `exercise_boundary.png` — the SELL/HOLD region in (time remaining, lead)
space, the basketball analogue of an American option's early-exercise region.
Note the epistemic status printed at the bottom: synthetic backtests prove the
policy is optimal *under the model*; only forward data tests the model.

**3. Start logging real bids (every game night, even games you don't hold).**

```bash
python scripts/log_bids.py --game-id 0042500404 \
    --combo-ticker KXNBACOMBO-... --leg-ticker KXNBA-...-ML --spread -4.5
```

**4. Refit the haircut model once logs accumulate.**

```bash
lpf fit-bid-model     # then update config/settings.yaml [bid_model]
```

The shipped haircut parameters are placeholders. The exercise boundary is only
as honest as this calibration.

**5. Follow a live position.**

```bash
lpf positions   # discover your combo ticker + cost basis
lpf follow --ticker KXNBACOMBO-... --game-id 0042500404 --spread -4.5 \
    --leg "moneyline:side=home@KXNBA-...-ML" \
    --leg "total_over:line=224.5@KXNBA-...-TOTAL"
```

## Tests

```bash
pytest -q
```

## The paper-trading gate (read before risking money)

No stopping algorithm can rescue a probability model without edge. Before any
real capital relies on a SELL signal:

1. Run the follower in alert-only mode for several weeks across many games.
2. Use `probability/shrinkage.EdgeLedger.edge_report()` per leg type: were the
   model's disagreements with the market right more often than wrong?
3. Leg types without demonstrated forward edge stay shrunk toward market
   probabilities (`shrinkage_unproven_weight` in settings) — which correctly
   biases toward earlier, safer exits.

Size anything live as tuition, not income, until the forward log says otherwise.
This software is a research and decision-support tool, not financial advice.

## Known v1 approximations (each has a planned upgrade path)

- The DP grid models the moneyline leg via the score-diff diffusion; other live
  legs enter as a multiplicative survival probability between boundary
  refreshes. Upgrade: add leg dimensions to the grid (totals) or use the LSMC
  branch (props).
- Totals are priced with a Normal-around-pace approximation. Upgrade: LightGBM
  totals model with isotonic recalibration.
- The robust ensemble uses parameter perturbation. Upgrade: bootstrap-refit
  models on historical seasons (interface unchanged).
- Player-prop probabilities default to 0.5 until the LightGBM prop model is
  trained — and the shrinkage layer treats them as unproven accordingly.
