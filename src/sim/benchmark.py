"""Buy-and-hold benchmark curve (default SPY)."""

from __future__ import annotations

import pandas as pd


def buy_hold_equity_curve(close: pd.Series, *, initial_equity: float) -> pd.Series:
    """Scale ``initial_equity`` by total return of ``close`` from its first valid value."""

    c = close.astype(float)
    c = c[c > 0]
    if c.empty:
        return pd.Series(dtype=float)
    start = float(c.iloc[0])
    return float(initial_equity) * (c / start)


__all__ = ["buy_hold_equity_curve"]
