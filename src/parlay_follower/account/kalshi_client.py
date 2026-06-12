"""Thin Kalshi REST client.

Covers the endpoints the project needs:
  * portfolio: positions, fills, settlements, balance  (account discovery, cost basis)
  * markets:   market metadata, order book              (live bid + depth)
  * multivariate / communications (RFQ)                 (combo structure + Plan-B quotes)

NOTE -- DAY-ONE RECON: exact response field names for combo (multivariate) positions
and combo order books must be confirmed against the live API (scripts/day_one_recon.py)
before the rest of the pipeline trusts this data contract. Anything marked
`RECON:` below is a best-effort guess to be locked in then.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests

from .auth import KalshiSigner


@dataclass
class Position:
    ticker: str
    contracts: int
    cost_basis_dollars: float       # total paid
    is_combo: bool = False
    leg_tickers: list[str] = field(default_factory=list)  # RECON: from multivariate lookup


class KalshiClient:
    def __init__(self, base_url: str, signer: KalshiSigner, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.timeout = timeout
        self._session = requests.Session()

    # ---------- low-level ----------
    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        sign_path = urlsplit(url).path  # path only, NO query params
        headers = self.signer.headers("GET", sign_path)
        r = self._session.get(url, headers=headers, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict) -> dict:
        url = f"{self.base_url}{path}"
        sign_path = urlsplit(url).path
        headers = self.signer.headers("POST", sign_path)
        r = self._session.post(url, headers=headers, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---------- portfolio (authenticated) ----------
    def get_positions(self) -> list[dict]:
        return self._get("/portfolio/positions").get("market_positions", [])

    def get_fills(self, ticker: str | None = None) -> list[dict]:
        params = {"ticker": ticker} if ticker else None
        return self._get("/portfolio/fills", params=params).get("fills", [])

    def get_settlements(self) -> list[dict]:
        return self._get("/portfolio/settlements").get("settlements", [])

    def get_balance(self) -> dict:
        return self._get("/portfolio/balance")

    # ---------- market data (public) ----------
    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """RECON: returns yes/no bid levels; confirm shape for combo tickers."""
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth}).get("orderbook", {})

    # ---------- multivariate (combos) ----------
    def get_multivariate_lookup(self, collection_ticker: str, selected: list[dict]) -> dict:
        """RECON: resolve a combo's constituent legs / created market ticker."""
        return self._post(
            f"/multivariate_event_collections/{collection_ticker}/lookup",
            json={"selected_markets": selected},
        )

    # ---------- RFQ fallback (Plan B for combo exit quotes) ----------
    def create_rfq(self, ticker: str, contracts: int) -> dict:
        """RECON: communications endpoint group; request a firm quote on our combo."""
        return self._post("/communications/rfqs", json={"ticker": ticker, "contracts": contracts})

    # ---------- convenience ----------
    def discover_positions(self) -> list[Position]:
        """Turn raw portfolio data into Position objects with cost basis from fills."""
        out: list[Position] = []
        for p in self.get_positions():
            ticker = p.get("ticker", "")
            contracts = int(p.get("position", 0))
            if contracts == 0:
                continue
            fills = self.get_fills(ticker=ticker)
            # Cost basis: sum of buy-side fill notionals. RECON: confirm field names
            # ("price_dollars"/"yes_price" and "count") against live responses.
            cost = 0.0
            for f in fills:
                px = float(f.get("price_dollars", f.get("yes_price", 0)) or 0)
                if px > 1.5:  # value came back in cents
                    px /= 100.0
                cost += px * int(f.get("count", 0))
            is_combo = "MULTI" in ticker.upper() or bool(p.get("is_multivariate"))  # RECON
            out.append(Position(ticker=ticker, contracts=contracts,
                                cost_basis_dollars=cost, is_combo=is_combo))
        return out
