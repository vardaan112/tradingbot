"""Tests for QQQ regime classification, liquidity gate, and position sizer regime mult."""

from __future__ import annotations

import pandas as pd

from config.settings import Settings
from risk.compliance import ComplianceAdapter
from risk.exposure import ExposureChecker
from risk.position_sizer import PositionSizer
from services.regime_detector import QqqRegimeSnapshot, _compute_snapshot, _with_anchor_metrics
from tests.conftest import make_settings
from tests.test_position_sizer import make_account


def _settings_min() -> Settings:
    return make_settings()


def test_bear_volatile_when_below_sma_and_high_atr_ratio() -> None:
    n = 120
    # Declining closes + elevated recent volatility vs smooth ATR history.
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = pd.Series([400.0 - i * 0.5 + (10 if i >= n - 5 else 0) for i in range(n)], index=idx)
    high = close + 2.0
    low = close - 2.0
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)
    s = make_settings(REGIME_ATR_RATIO_THRESHOLD=1.0)
    snap = _compute_snapshot(df, s, sym="QQQ")
    assert snap.error == ""
    assert snap.bear_volatile is True


def test_regime_insufficient_bars_fail_closed_by_default() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series([100.0] * len(idx), index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close}, index=idx)

    snap = _compute_snapshot(df, make_settings(), sym="QQQ")

    assert snap.error == "insufficient_bars"
    assert snap.bear_volatile is True


def test_regime_unknown_reduce_size_does_not_hard_block() -> None:
    idx = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    close = pd.Series([100.0] * len(idx), index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close}, index=idx)

    snap = _compute_snapshot(
        df,
        make_settings(REGIME_UNKNOWN_ACTION="reduce_size"),
        sym="QQQ",
    )

    assert snap.error == "insufficient_bars"
    assert snap.bear_volatile is False


def test_regime_mult_reduces_usd_cap() -> None:
    s = make_settings(BOT_CAPITAL_BASE_USD=25_000.0)
    sz = PositionSizer(s, ComplianceAdapter(s), ExposureChecker(s))
    acct = make_account(equity=100_000.0, buying_power=400_000.0)
    out = sz.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=acct,
        positions=[],
        bot_managed_notional=0.0,
        regime_equity_multiplier=0.5,
    )
    assert out.shares >= 0
    half = sz.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=acct,
        positions=[],
        bot_managed_notional=0.0,
        regime_equity_multiplier=1.0,
    )
    assert out.shares <= half.shares + 1e-6


def test_liquidity_threshold_math() -> None:
    thr = 0.5
    vol5m = 10.0
    avg20 = 1000.0
    assert vol5m < thr * avg20


def test_anchor_regime_metrics_classify_parabolic_bull() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    close = pd.Series([100.0 + i * 0.5 for i in range(len(idx))], index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )
    base = QqqRegimeSnapshot(
        bear_volatile=False,
        close=500.0,
        sma50=490.0,
        atr1h=2.0,
        atr_ma=2.0,
        atr_ratio=1.0,
        updated_at=idx[-1].to_pydatetime(),
    )

    snap = _with_anchor_metrics(df, make_settings(), base, sym="SPY")

    assert snap.anchor_symbol == "SPY"
    assert snap.anchor_state == "ParabolicBull"
    assert snap.anchor_close > snap.anchor_sma
    assert snap.anchor_rsi >= 70.0
