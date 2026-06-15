"""Unified exit pricing: the one place that knows HOW to get out of a position.

Kalshi has two completely different exit mechanisms, and conflating them is what
makes a cash-out silently fail:

  * SINGLE-LEG markets trade on a visible order book. The exit price is the
    depth-weighted bid -- pollable any time, almost always present.

  * COMBOS (multi-leg) are priced via RFQ (request-for-quote). There is no
    resting bid to poll; you submit the combo and institutional market makers
    *may* respond with a price within seconds -- or may not respond at all.
    "No one willing to trade" is an RFQ that came back empty. Liquidity thins
    hard beyond 3-4 legs.

Every exit decision in the system goes through `get_exit_quote`, which returns
an ExitQuote that explicitly carries whether an exit is even AVAILABLE right
now. Downstream logic must handle `available == False` rather than assuming a
number always exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..account.kalshi_client import KalshiClient, Position
from .orderbook import executable_proceeds, parse_yes_bids


class ExitSource(Enum):
    ORDER_BOOK = "order_book"   # single-leg: pollable bid
    RFQ = "rfq"                 # combo: requested quote
    NONE = "none"               # no liquidity / no quote returned


@dataclass
class ExitQuote:
    available: bool             # can we exit right now, at all?
    proceeds: float             # total $ for the whole position if we sell now
    avg_price: float            # per-contract proceeds (0..1)
    source: ExitSource
    depth_ok: bool = True       # False if the book/quote can't absorb full size
    note: str = ""

    @classmethod
    def unavailable(cls, source: ExitSource, note: str) -> "ExitQuote":
        return cls(False, 0.0, 0.0, source, depth_ok=False, note=note)


def get_exit_quote(client: KalshiClient, position: Position,
                   rfq_for_combos: bool = True) -> ExitQuote:
    """Single dispatch point for 'what can I sell this for, right now?'."""
    if not position.is_combo:
        return _order_book_quote(client, position)
    if rfq_for_combos:
        return _rfq_quote(client, position)
    # Fallback: try the combo ticker as if it had a book (works only if Kalshi
    # ever exposes one; otherwise returns unavailable cleanly).
    q = _order_book_quote(client, position)
    return q if q.available else ExitQuote.unavailable(
        ExitSource.NONE, "combo has no pollable book; enable rfq_for_combos")


def _order_book_quote(client: KalshiClient, position: Position) -> ExitQuote:
    try:
        levels = parse_yes_bids(client.get_orderbook(position.ticker))
    except Exception as e:
        return ExitQuote.unavailable(ExitSource.ORDER_BOOK, f"orderbook error: {e}")
    if not levels:
        return ExitQuote.unavailable(ExitSource.ORDER_BOOK, "empty book (no bids)")
    proceeds, avg = executable_proceeds(levels, position.contracts)
    depth = sum(l.size for l in levels)
    return ExitQuote(
        available=proceeds > 0, proceeds=proceeds, avg_price=avg,
        source=ExitSource.ORDER_BOOK, depth_ok=depth >= position.contracts,
        note="" if depth >= position.contracts else
             f"thin book: depth {depth} < position {position.contracts}",
    )


def _rfq_quote(client: KalshiClient, position: Position) -> ExitQuote:
    """Request a firm quote to SELL the combo. RECON: confirm the create_rfq
    response shape (field carrying the quoted price/size) on day one."""
    try:
        resp = client.create_rfq(position.ticker, position.contracts)
    except Exception as e:
        return ExitQuote.unavailable(ExitSource.RFQ, f"rfq error: {e}")
    # RECON: replace these guessed field names once the live response is dumped.
    price = resp.get("yes_bid") or resp.get("price") or resp.get("best_bid")
    if not price:
        return ExitQuote.unavailable(ExitSource.RFQ, "no quote returned (illiquid)")
    px = float(price) / (100.0 if float(price) > 1.5 else 1.0)
    return ExitQuote(
        available=True, proceeds=px * position.contracts, avg_price=px,
        source=ExitSource.RFQ, depth_ok=True,
        note="rfq quote (valid for seconds; re-request before acting)",
    )


def liquidity_preflight(client: KalshiClient, position: Position) -> ExitQuote:
    """Run BEFORE following a game so you never discover mid-game that you're
    trapped. Same call as a live quote, but framed as a go/no-go check."""
    q = get_exit_quote(client, position)
    n_legs = max(1, len(position.leg_tickers))
    if not q.available:
        q.note = (f"PREFLIGHT: cannot exit {n_legs}-leg position right now "
                  f"({q.note}). Combos rely on RFQ and may be illiquid; "
                  f"consider 2-3 legs or single-leg markets.")
    elif n_legs >= 4:
        q.note += " | PREFLIGHT WARNING: 4+ legs -- expect wide spreads / thin RFQ."
    return q