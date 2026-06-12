"""Order-book utilities: the policy optimizes REALIZED proceeds, not top-of-book."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BookLevel:
    price: float   # dollars per contract, 0..1
    size: int      # contracts available


def parse_yes_bids(raw_orderbook: dict) -> list[BookLevel]:
    """Kalshi books quote in cents; normalize to dollars, best bid first.

    RECON: confirm the raw shape for combo tickers ('yes' = [[price_cents, size], ...]).
    """
    levels = raw_orderbook.get("yes", []) or []
    out = [BookLevel(price=lvl[0] / 100.0, size=int(lvl[1])) for lvl in levels]
    return sorted(out, key=lambda l: l.price, reverse=True)


def best_bid(levels: list[BookLevel]) -> float:
    return levels[0].price if levels else 0.0


def executable_proceeds(levels: list[BookLevel], contracts: int) -> tuple[float, float]:
    """Walk the book selling `contracts`.

    Returns (total_proceeds_dollars, avg_price). If the book is too thin to absorb
    the full position, the unfilled remainder is valued at 0 -- a deliberately
    pessimistic convention so thin books push the policy toward HOLD-with-eyes-open
    rather than imaginary exits.
    """
    remaining = contracts
    proceeds = 0.0
    for lvl in levels:
        take = min(remaining, lvl.size)
        proceeds += take * lvl.price
        remaining -= take
        if remaining == 0:
            break
    avg = proceeds / contracts if contracts else 0.0
    return proceeds, avg
