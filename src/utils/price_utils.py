"""Price/quote helpers: tick rounding, spread filter, quote validity."""

from __future__ import annotations

import math
from typing import Optional

from config.constants import TICK_SIZE_ABOVE_ONE, TICK_SIZE_BELOW_ONE


def tick_size_for(price: float) -> float:
    """Return the legal NMS tick size for a given price level.

    Equities >= $1.00 trade in $0.01 increments; sub-$1 names trade in $0.0001.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    return TICK_SIZE_ABOVE_ONE if price >= 1.0 else TICK_SIZE_BELOW_ONE


def round_to_tick(price: float, *, mode: str = "nearest") -> float:
    """Round `price` to a legal tick.

    `mode` ∈ {"nearest", "down", "up"}:
      - "down" rounds toward zero (useful for buy limits below current bid)
      - "up"   rounds away from zero (useful for sell limits above current ask)
      - "nearest" rounds to the closest tick.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    tick = tick_size_for(price)
    units = price / tick
    if mode == "down":
        rounded = math.floor(units) * tick
    elif mode == "up":
        rounded = math.ceil(units) * tick
    elif mode == "nearest":
        rounded = round(units) * tick
    else:
        raise ValueError(f"unknown rounding mode: {mode!r}")
    # Avoid float drift residue like 12.730000000000002.
    return round(rounded, 4 if tick < 0.01 else 2)


def mid_price(bid: float, ask: float) -> float:
    """Return (bid + ask) / 2 with bound checks."""
    if bid <= 0 or ask <= 0 or ask <= bid:
        raise ValueError(f"invalid quote: bid={bid}, ask={ask}")
    return (bid + ask) / 2.0


def spread_pct(bid: float, ask: float) -> float:
    """Compute (ask - bid) / mid, the canonical relative spread used as a filter."""
    if bid <= 0 or ask <= 0 or ask <= bid:
        raise ValueError(f"invalid quote: bid={bid}, ask={ask}")
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid


def is_valid_quote(
    bid: Optional[float],
    ask: Optional[float],
    *,
    quote_age_seconds: Optional[float] = None,
    max_age_seconds: Optional[float] = None,
) -> bool:
    """Return True iff bid/ask are well-formed and the quote is fresh enough.

    `quote_age_seconds` and `max_age_seconds` are both required to enforce the
    staleness check; pass `None` to skip the freshness component.
    """
    if bid is None or ask is None:
        return False
    if bid <= 0 or ask <= 0:
        return False
    if ask <= bid:
        return False
    if quote_age_seconds is not None and max_age_seconds is not None:
        if quote_age_seconds > max_age_seconds:
            return False
    return True
