"""Regime filtering: pure, deterministic computations on OHLC bars.

No logging, network, or I/O. Intended for use inside strategy evaluation after
completed bars have been sliced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import Settings

RegimeKind = Literal["Trending", "Range"]


@dataclass(frozen=True)
class RegimeSnapshot:
    """Point-in-time regime inputs for RSI entry gating and sizing cues."""

    adx: float
    adx_length: int
    sma200: float
    sma_length: int
    sma_slope: float
    sma_slope_lookback: int
    price_above_sma200: bool
    regime_type: RegimeKind
    high_conviction: bool  # semantic: price_above_sma200 per product spec
    allow_rsi_long: bool
    reason: str


def _ensure_float_series(series: pd.Series) -> pd.Series:
    return series.astype(float)


def _wilders_smoothed_tr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (+DM_smooth, -DM_smooth, TR_smooth) aligned to input via Wilder's EWM.

    Uses pandas ewm with alpha=1/length across TR, plus/minus directional movement,
    identical to Wilder's iterative smoothing convention used elsewhere for ATR.
    """
    hi = _ensure_float_series(high)
    lo = _ensure_float_series(low)
    cl = _ensure_float_series(close)

    hi_prev = hi.shift(1)
    lo_prev = lo.shift(1)

    plus_dm_raw = hi.diff()
    minus_dm_raw = (-lo.diff())

    plus_dm_raw = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm_raw = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)

    prev_close = cl.shift(1)
    tr = pd.concat(
        [
            (hi - lo).abs(),
            (hi - prev_close).abs(),
            (lo - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / float(length)
    tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    pp_s = plus_dm_raw.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    mn_s = minus_dm_raw.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return pp_s, mn_s, tr_s


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-style Average Directional Index (ADX), consistent with indicator style.

    Returns a Series indexed like `close` with NaNs until sufficiently warm,
    capped at sensible bounds above 100.
    """
    if length < 2:
        raise ValueError("ADX length must be >= 2")
    pp_s, mn_s, tr_s = _wilders_smoothed_tr(high, low, close, length)
    tr_safe = tr_s.replace(0.0, np.nan)

    plus_di = 100.0 * (pp_s / tr_safe).fillna(0.0)
    minus_di = 100.0 * (mn_s / tr_safe).fillna(0.0)
    denom_di = (plus_di + minus_di).replace(0.0, np.nan)
    dx_series = (
        100.0 * (plus_di - minus_di).abs() / denom_di
    ).fillna(0.0)

    alpha = 1.0 / float(length)
    adx_series = dx_series.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    adx_series.name = f"adx_{length}"
    return adx_series


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    if length < 1:
        raise ValueError("SMA length must be >= 1")
    return close.astype(float).rolling(window=length, min_periods=length).mean()


def compute_regime_snapshot(
    *,
    bars: pd.DataFrame,
    settings: Settings,
) -> RegimeSnapshot | None:
    """Build a regime snapshot from the last closed bar slice.

    Entry gating: ``allow_rsi_long`` when (200-)SMA slope is positive **or**
    ADX is strictly below ``settings.ADX_RANGE_MAX`` (default 25 ⇒ "ADX below 25").
    Otherwise long mean-reversion entries are blocked while ADX reads as trending.

    Returns None if bars are insufficient to compute SMA/ADX cleanly.
    """
    req = settings.SMA_FILTER_LENGTH + settings.SMA_SLOPE_LOOKBACK_BARS + 5
    if bars is None or bars.empty or len(bars) < req:
        return None

    adx_len = settings.ADX_LENGTH
    sma_len = settings.SMA_FILTER_LENGTH
    slope_lb = settings.SMA_SLOPE_LOOKBACK_BARS

    highs = bars["high"]
    lows = bars["low"]
    closes = bars["close"]

    adx_series = adx(highs, lows, closes, length=adx_len)
    sma_series = sma(closes, length=sma_len)

    last_adx_raw = float(adx_series.iloc[-1])
    last_adx = last_adx_raw if not np.isnan(last_adx_raw) else float("nan")
    last_close = float(closes.iloc[-1])
    last_sma_raw = float(sma_series.iloc[-1])
    if np.isnan(last_sma_raw) or np.isnan(last_adx) or np.isnan(last_close):
        return None

    # Slope defined as change in SMA over `slope_lb` periods (backward-looking).
    if len(bars) < sma_len + slope_lb:
        return None
    past_sma = float(sma_series.iloc[-1 - slope_lb])
    if np.isnan(past_sma):
        return None
    sma_slope = last_sma_raw - past_sma

    regime_type: RegimeKind = (
        "Range" if float(last_adx) < settings.ADX_RANGE_MAX else "Trending"
    )
    price_above = last_close > last_sma_raw
    sma_slope_positive = sma_slope > 0.0
    adx_relaxed = float(last_adx) < settings.ADX_RANGE_MAX

    allow = sma_slope_positive or adx_relaxed
    price_above_bool = price_above

    if allow:
        if sma_slope_positive and adx_relaxed:
            reason = "ok:sma_positive_and_adx_below"
        elif sma_slope_positive:
            reason = "ok:sma_slope_positive"
        else:
            reason = "ok:adx_below_threshold"
    else:
        parts: list[str] = []
        if not sma_slope_positive:
            parts.append("non_positive_slope")
        if not adx_relaxed:
            parts.append(f"adx>={settings.ADX_RANGE_MAX}")
        reason = "blocked:" + "+".join(parts)

    return RegimeSnapshot(
        adx=float(last_adx),
        adx_length=adx_len,
        sma200=float(last_sma_raw),
        sma_length=sma_len,
        sma_slope=float(sma_slope),
        sma_slope_lookback=slope_lb,
        price_above_sma200=price_above_bool,
        regime_type=regime_type,
        high_conviction=price_above_bool,
        allow_rsi_long=allow,
        reason=reason,
    )


__all__ = ["RegimeSnapshot", "adx", "sma", "compute_regime_snapshot"]
