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


def bollinger_bands(
    close: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return basis, upper, lower, and width pct for classic Bollinger bands."""

    if length < 2:
        raise ValueError("Bollinger length must be >= 2")
    if num_std <= 0:
        raise ValueError("Bollinger std multiplier must be > 0")
    c = close.astype(float)
    basis = c.rolling(window=length, min_periods=length).mean()
    sigma = c.rolling(window=length, min_periods=length).std(ddof=0)
    upper = basis + float(num_std) * sigma
    lower = basis - float(num_std) * sigma
    width = (upper - lower) / basis.replace(0.0, np.nan)
    basis.name = f"bb_basis_{length}"
    upper.name = f"bb_upper_{length}"
    lower.name = f"bb_lower_{length}"
    width.name = f"bb_width_pct_{length}"
    return basis, upper, lower, width


def rolling_vwap_bands(
    close: pd.Series,
    volume: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return rolling VWAP, upper/lower deviation bands, and distance pct."""

    if length < 2:
        raise ValueError("VWAP length must be >= 2")
    if num_std <= 0:
        raise ValueError("VWAP std multiplier must be > 0")
    c = close.astype(float)
    v = volume.astype(float).clip(lower=0.0)
    vol_sum = v.rolling(window=length, min_periods=length).sum()
    vwap = (c * v).rolling(window=length, min_periods=length).sum() / vol_sum.replace(0.0, np.nan)
    dev = (c - vwap).rolling(window=length, min_periods=length).std(ddof=0)
    upper = vwap + float(num_std) * dev
    lower = vwap - float(num_std) * dev
    distance_pct = (c - vwap) / vwap.replace(0.0, np.nan)
    vwap.name = f"rolling_vwap_{length}"
    upper.name = f"vwap_upper_{length}"
    lower.name = f"vwap_lower_{length}"
    distance_pct.name = f"vwap_distance_pct_{length}"
    return vwap, upper, lower, distance_pct


def rolling_vwap_zscore_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    *,
    length: int = 20,
    z_threshold: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return rolling VWAP, z-score bands, z-score, distance pct, and deviation.

    VWAP uses typical price ``(high + low + close) / 3`` weighted by bar volume.
    The deviation estimate is the rolling standard deviation of price-VWAP
    residuals. This keeps the calculation deterministic on completed 5-minute
    bars and avoids relying on broker-side partial-bar VWAP fields.
    """

    if length < 2:
        raise ValueError("VWAP length must be >= 2")
    if z_threshold <= 0:
        raise ValueError("VWAP z threshold must be > 0")

    h = high.astype(float)
    lo = low.astype(float)
    c = close.astype(float)
    v = volume.astype(float).clip(lower=0.0)
    typical = (h + lo + c) / 3.0
    vol_sum = v.rolling(window=length, min_periods=length).sum()
    vwap = (typical * v).rolling(window=length, min_periods=length).sum() / vol_sum.replace(
        0.0,
        np.nan,
    )
    residual = typical - vwap
    dev = residual.rolling(window=length, min_periods=length).std(ddof=0)
    upper = vwap + float(z_threshold) * dev
    lower = vwap - float(z_threshold) * dev
    zscore = residual / dev.replace(0.0, np.nan)
    distance_pct = residual / vwap.replace(0.0, np.nan)

    vwap.name = f"rolling_vwap_{length}"
    upper.name = f"vwap_upper_z{z_threshold:g}_{length}"
    lower.name = f"vwap_lower_z{z_threshold:g}_{length}"
    zscore.name = f"vwap_zscore_{length}"
    distance_pct.name = f"vwap_distance_pct_{length}"
    dev.name = f"vwap_deviation_{length}"
    return vwap, upper, lower, zscore, distance_pct, dev
