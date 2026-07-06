#!/usr/bin/env python
"""DAY-ONE RECON: lock the Kalshi data contract before trusting anything.

Answers the project's one open question: are combo order books queryable via
the standard public market-data endpoints, or does combo exit pricing surface
only through RFQ?

Procedure (run after buying a small NBA combo in demo or production):
  1. Auth check (balance)
  2. Dump raw GetPositions          -> how do combo positions appear? field names?
  3. Dump fills for each position   -> cost basis fields?
  4. Dump the combo ticker's order book -> queryable? shape? depth?
  5. If (4) fails, attempt an RFQ   -> Plan B confirmed?

Paste the JSON outputs into docs/data_contract.md and update every `RECON:`
comment in parlay_follower/execution/account/kalshi_client.py and
parlay_follower/execution/market_data/orderbook.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parlay_follower.execution.account.auth import KalshiSigner          # noqa: E402
from parlay_follower.execution.account.kalshi_client import KalshiClient  # noqa: E402
from parlay_follower.shared.config import base_url, load_creds, load_settings  # noqa: E402


def dump(label, obj):
    print(f"\n========== {label} ==========")
    print(json.dumps(obj, indent=2, default=str)[:4000])


def main():
    settings, creds = load_settings(), load_creds()
    client = KalshiClient(base_url(settings, creds.env),
                          KalshiSigner(creds.key_id, creds.private_key_path))

    dump("1. BALANCE (auth check)", client.get_balance())

    positions = client.get_positions()
    dump("2. RAW POSITIONS", positions)

    for p in positions:
        ticker = p.get("ticker", "")
        if not ticker:
            continue
        dump(f"3. FILLS for {ticker}", client.get_fills(ticker=ticker)[:5])
        try:
            dump(f"4. ORDER BOOK for {ticker}", client.get_orderbook(ticker))
        except Exception as e:
            print(f"\n4. ORDER BOOK for {ticker} FAILED: {e}")
            print("   -> combo book may not be public-endpoint queryable; trying RFQ (Plan B)")
            try:
                dump(f"5. RFQ for {ticker}", client.create_rfq(ticker, contracts=1))
            except Exception as e2:
                print(f"5. RFQ FAILED too: {e2}  -> inspect docs.kalshi.com/communications")

    print("\nRecon complete. Lock findings into docs/data_contract.md and "
          "resolve every `RECON:` comment in the codebase.")


if __name__ == "__main__":
    main()
