"""High-beta correlation gate (SPY ↔ QQQ) for duplicate factor exposure."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.constants import LOGGER_RISK
from config.settings import Settings
from core.account import PositionSnapshot
from core.market_data import BarFetcher

_LOG = logging.getLogger(LOGGER_RISK)


def pearson_from_closes(a: pd.Series, b: pd.Series) -> float:
    """Pearson r on intersection of indices; NaN if insufficient overlap."""
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(how="any")
    if len(aligned) < 3:
        return float("nan")
    r = aligned["a"].corr(aligned["b"])
    return float(r) if r == r else float("nan")


def _has_long(sym: str, positions: list[PositionSnapshot]) -> bool:
    sup = sym.upper()
    for p in positions:
        if p.symbol.upper() != sup:
            continue
        if p.side.lower() != "long":
            continue
        if abs(p.qty) > 1e-9:
            return True
    return False


def correlation_block_reason(
    settings: Settings,
    *,
    follower_symbol: str,
    positions: list[PositionSnapshot],
    bar_fetcher: BarFetcher,
) -> Optional[str]:
    """Return a skip token when leader + follower overlap is too correlated.

    Uses the last ``CORRELATION_LOOKBACK_CALENDAR_DAYS`` overlapping *calendar*
    observations of 1D closes (trading days returned by Alpaca bars).
    """
    if not settings.CORRELATION_BREAKER_ENABLED:
        return None

    leader = settings.CORRELATION_LEADER_SYMBOL.upper()
    follower = follower_symbol.strip().upper()

    if follower not in {s.upper() for s in settings.correlation_follower_symbols_list}:
        return None

    if not _has_long(leader, positions):
        return None

    look = settings.CORRELATION_LOOKBACK_CALENDAR_DAYS

    try:
        la = bar_fetcher.fetch_bars(leader, "1Day", lookback_bars=look + 15)
        fb = bar_fetcher.fetch_bars(follower, "1Day", lookback_bars=look + 15)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("correlation breaker: bar fetch failed %s", exc)
        return None

    if la is None or fb is None or la.empty or fb.empty:
        _LOG.info("correlation breaker: insufficient bars leader=%s follower=%s", leader, follower)
        return None

    a = la["close"].astype(float).tail(look)
    b = fb["close"].astype(float).tail(look)
    corr = pearson_from_closes(a, b)
    if corr != corr:
        return None
    if corr > float(settings.CORRELATION_BREAKER_THRESHOLD):
        _LOG.info(
            "event=correlation_breaker leader=%s follower=%s corr=%.4f thresh=%.4f",
            leader,
            follower,
            corr,
            settings.CORRELATION_BREAKER_THRESHOLD,
        )
        return f"correlation_breaker_leader_{leader}_follower_{follower}"
    return None


__all__ = ["correlation_block_reason", "pearson_from_closes"]
