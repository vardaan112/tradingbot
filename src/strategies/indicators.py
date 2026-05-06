"""Pure-Python indicators (RSI, ATR) implemented with pandas/numpy.

Wilder's smoothing is used so behavior matches Alpaca / TradingView defaults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI computed on a close-price series.

    Returns a Series aligned to the input index. Initial values are NaN until
    enough samples are available.
    """
    if length < 2:
        raise ValueError("RSI length must be >= 2")
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    rsi_series = rsi_series.fillna(100.0).where(avg_loss.notna() | (avg_gain > 0))
    rsi_series.name = f"rsi_{length}"
    return rsi_series


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Wilder's ATR using true range with EWM smoothing."""
    if length < 2:
        raise ValueError("ATR length must be >= 2")
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    atr_series.name = f"atr_{length}"
    return atr_series
