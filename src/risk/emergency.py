"""Global flash-crash style monitor (rolling peak-to-last drawdown).

Compares latest mid against the rolling **peak** midpoint inside the deque window.
Triggered when ``(last / peak - 1) <= -threshold_pct``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Optional, Tuple

__all__ = ["PricePoint", "SpyFlashCrashMonitor", "detect_black_swan_drop"]


@dataclass(frozen=True)
class PricePoint:
    """One mid-price observation in UTC."""

    t: datetime
    mid: float


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def detect_black_swan_drop(
    symbol: str,
    price_window: Sequence[PricePoint],
    *,
    threshold_pct: float = 0.03,
    window_minutes: Optional[int] = None,
) -> bool:
    """Return True iff the last mid is at least *threshold_pct* below the rolling peak.

    When *window_minutes* is set, only points within that many minutes of the
    last observation are considered (mirrors the rolling window in production).

    ``symbol`` is for call-site clarity and tests; detection is price-path only.
    """

    _ = symbol
    pts = list(price_window)
    if len(pts) < 2:
        return False
    if window_minutes is not None:
        anchor = _to_utc(pts[-1].t)
        cutoff = anchor - timedelta(minutes=int(window_minutes))
        filtered: list[PricePoint] = []
        for p in pts:
            t = _to_utc(p.t)
            if t >= cutoff:
                filtered.append(p)
        pts = filtered
        if len(pts) < 2:
            return False
    peak = max(float(p.mid) for p in pts if p.mid > 0)
    last = float(pts[-1].mid)
    if peak <= 0 or last <= 0:
        return False
    return (last / peak) - 1.0 <= -float(threshold_pct)


class SpyFlashCrashMonitor:
    """Maintain a deque of `(timestamp_utc, mid_price)` observations."""

    __slots__ = ("_sym", "_drop_pct", "_window", "_points")

    def __init__(self, *, symbol: str, drop_pct: float, window_minutes: int) -> None:
        self._sym = str(symbol).upper()
        self._drop_pct = float(drop_pct)
        self._window = timedelta(minutes=int(window_minutes))
        self._points: Deque[Tuple[datetime, float]] = deque()

    def observe(self, now_utc: datetime, mid_price: float) -> None:
        if mid_price <= 0 or not isinstance(now_utc, datetime):
            return
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        now_utc = now_utc.astimezone(timezone.utc)
        self._points.append((now_utc, float(mid_price)))
        cutoff = now_utc - self._window
        while self._points and self._points[0][0] < cutoff:
            self._points.popleft()

    def triggered(self) -> bool:
        window_pts = tuple(PricePoint(t=t, mid=p) for t, p in self._points)
        wm = int(self._window.total_seconds() // 60)
        return detect_black_swan_drop(
            self._sym,
            window_pts,
            threshold_pct=self._drop_pct,
            window_minutes=wm,
        )

    def reset(self) -> None:
        self._points.clear()
