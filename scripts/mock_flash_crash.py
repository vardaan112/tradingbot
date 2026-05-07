#!/usr/bin/env python3
"""Offline SPY flash-crash demonstration for the Black Swan detector (no Alpaca).

Run from the repository root::

    python scripts/mock_flash_crash.py

Feeds a synthetic SPY midpoint path that declines ~10% over 15 minutes into the
same ``SpyFlashCrashMonitor`` used in ``services.orchestrator``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def main() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    sys.path.insert(0, str(src))

    from risk.emergency import (  # noqa: PLC0415
        PricePoint,
        SpyFlashCrashMonitor,
        detect_black_swan_drop,
    )

    t0 = datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)

    pts: list[PricePoint] = []
    spy = SpyFlashCrashMonitor(symbol="SPY", drop_pct=0.03, window_minutes=15)
    price = 450.0
    spy.observe(t0, price)
    pts.append(PricePoint(t=t0, mid=price))
    for minute in range(1, 16):
        price -= 3.0
        ts = t0 + timedelta(minutes=minute)
        spy.observe(ts, price)
        pts.append(PricePoint(t=ts, mid=price))

    window_flat = tuple(
        PricePoint(t=p.t, mid=p.mid) for p in pts
    )
    pure = detect_black_swan_drop("SPY", window_flat, threshold_pct=0.03, window_minutes=15)

    print(
        f"path: 450 -> {price:.2f} over 15m "
        f"detect_black_swan_drop={pure} monitor.triggered={spy.triggered()}"
    )


if __name__ == "__main__":
    main()
