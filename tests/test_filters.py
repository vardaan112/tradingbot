"""Unit tests for deterministic regime filters (ADX, SMA, entry gating)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.filters import adx, compute_regime_snapshot, sma


def _ohlc_from_close(idx: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    c = pd.Series(close, index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c.iloc[0]),
            "high": c + 0.75,
            "low": c - 0.75,
            "close": c,
            "volume": np.full(len(c), 1_000_000.0),
        }
    )


def test_adx_low_in_choppy_series(make_settings_factory):
    """Sideways chop should print a relatively low ADX vs a clean trend."""
    settings = make_settings_factory(
        SMA_FILTER_LENGTH=20,
        SMA_SLOPE_LOOKBACK_BARS=3,
        ADX_LENGTH=14,
    )
    idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
    chop = 100.0 + np.sin(np.linspace(0, 80, len(idx)))
    trend = np.linspace(50.0, 250.0, len(idx))
    bars_chop = _ohlc_from_close(idx, chop)
    bars_trend = _ohlc_from_close(idx, trend)
    adx_chop = float(adx(bars_chop["high"], bars_chop["low"], bars_chop["close"], length=14).iloc[-1])
    adx_tr = float(adx(bars_trend["high"], bars_trend["low"], bars_trend["close"], length=14).iloc[-1])
    assert adx_chop < adx_tr
    snap = compute_regime_snapshot(bars=bars_chop, settings=settings)
    assert snap is not None
    assert snap.adx < adx_tr


def test_compute_regime_snapshot_classifies_trending_when_adx_high(make_settings_factory):
    settings = make_settings_factory(
        SMA_FILTER_LENGTH=50,
        SMA_SLOPE_LOOKBACK_BARS=5,
        ADX_LENGTH=14,
        ADX_RANGE_MAX=25.0,
    )
    idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
    trend = np.linspace(80.0, 220.0, len(idx))
    bars = _ohlc_from_close(idx, trend)
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.regime_type == "Trending"
    assert snap.adx >= settings.ADX_RANGE_MAX - 1e-6


def test_allow_rsi_long_when_adx_relaxed_even_if_slope_negative(make_settings_factory):
    """From spec: allow entries when ADX is below the range threshold OR SMA slope > 0."""
    settings = make_settings_factory(SMA_FILTER_LENGTH=30, SMA_SLOPE_LOOKBACK_BARS=3, ADX_RANGE_MAX=100.0)
    idx = pd.date_range("2026-01-01", periods=200, freq="5min", tz="UTC")
    # Slow drift down (negative SMA slope) but ADX stays tame in this gentle series.
    drift = 120.0 - np.linspace(0, 3.0, len(idx))
    bars = _ohlc_from_close(idx, drift)
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.allow_rsi_long is True


def test_block_rsi_long_when_trending_and_negative_slope(make_settings_factory):
    settings = make_settings_factory(
        SMA_FILTER_LENGTH=50,
        SMA_SLOPE_LOOKBACK_BARS=5,
        ADX_RANGE_MAX=25.0,
    )
    idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
    trend_down = np.linspace(220.0, 70.0, len(idx))
    bars = _ohlc_from_close(idx, trend_down)
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.regime_type == "Trending"
    assert snap.sma_slope < 0
    assert snap.allow_rsi_long is False
    assert "blocked" in snap.reason


def test_allow_rsi_long_when_trending_with_positive_slope(make_settings_factory):
    """Even in a Trending (high ADX) regime, positive 200-SMA slope permits mean-revert longs."""
    settings = make_settings_factory(
        SMA_FILTER_LENGTH=50,
        SMA_SLOPE_LOOKBACK_BARS=5,
        ADX_RANGE_MAX=25.0,
    )
    idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
    trend_up = np.linspace(70.0, 220.0, len(idx))
    bars = _ohlc_from_close(idx, trend_up)
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.regime_type == "Trending"
    assert snap.sma_slope > 0
    assert snap.allow_rsi_long is True
    assert "ok:" in snap.reason


def test_high_conviction_tracks_price_above_sma(make_settings_factory):
    settings = make_settings_factory(SMA_FILTER_LENGTH=40, SMA_SLOPE_LOOKBACK_BARS=3)
    idx = pd.date_range("2026-01-01", periods=200, freq="5min", tz="UTC")
    lvl = np.full(len(idx), 150.0)
    bars = _ohlc_from_close(idx, lvl)
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.price_above_sma200 is False
    assert snap.high_conviction is False

    jump = np.full(len(idx), 150.0)
    jump[-5:] = 200.0
    bars2 = _ohlc_from_close(idx, jump)
    snap2 = compute_regime_snapshot(bars=bars2, settings=settings)
    assert snap2 is not None
    assert snap2.price_above_sma200 is True
    assert snap2.high_conviction is True


def test_sma_length_enforced():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        sma(closes, length=0)
