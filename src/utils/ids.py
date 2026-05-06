"""Identifier helpers, primarily for idempotent client_order_id values."""

from __future__ import annotations

import re
import uuid

from .time_utils import utc_compact_timestamp

_VALID_PIECE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def short_uuid(length: int = 8) -> str:
    """Return a short, URL-safe random ID derived from uuid4."""
    if length <= 0 or length > 32:
        raise ValueError("short_uuid length must be in [1, 32]")
    return uuid.uuid4().hex[:length]


def _sanitize(piece: str) -> str:
    """Replace any character that is not safe for an order id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", piece)


def generate_client_order_id(
    strategy: str,
    symbol: str,
    side: str,
    *,
    timestamp: str | None = None,
    short_id: str | None = None,
) -> str:
    """Build a unique client_order_id with format:

        {strategy}-{symbol}-{side}-{utc_compact_ts}-{short_uuid}

    Length-bounded so it stays well under Alpaca's typical 128-char ceiling.
    """
    if not strategy or not symbol or not side:
        raise ValueError("strategy, symbol, and side are required")

    ts = timestamp or utc_compact_timestamp()
    sid = short_id or short_uuid(8)

    pieces = [
        _sanitize(strategy)[:24],
        _sanitize(symbol.upper())[:12],
        _sanitize(side.lower())[:6],
        _sanitize(ts)[:24],
        _sanitize(sid)[:12],
    ]

    coid = "-".join(pieces)
    if len(coid) > 96:
        coid = coid[:96]
    return coid
