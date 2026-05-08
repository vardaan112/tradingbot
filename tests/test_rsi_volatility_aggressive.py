"""Volatility-tiered RSI and aggressive SMA bypass behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.account import AccountSnapshot
from core.market_data import Quote
from strategies.base import SignalAction, StrategyContext
from strategies.filters import RegimeSnapshot
from strategies.indicators import rolling_vwap_zscore_bands
from strategies.rsi_strategy import RSIMeanReversionStrategy


def _account(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity * 2.0,
        regt_buying_power=equity * 2.0,
        portfolio_value=equity,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=2.0,
        status="ACTIVE",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
    )


def _quote(symbol: str = "SHOP", bid: float = 99.95, ask: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=10.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _bars(n: int = 260, close_px: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    close = pd.Series(np.full(n, close_px), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000.0), index=idx),
        },
    )


def _ctx(symbol: str, bars: pd.DataFrame, quote: Quote) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        bars=bars,
        quote=quote,
        account=_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )


def _allow_regime() -> RegimeSnapshot:
    return RegimeSnapshot(
        adx=12.0,
        adx_length=14,
        sma200=101.0,
        sma_length=200,
        sma_slope=0.5,
        sma_slope_lookback=5,
        price_above_sma200=False,
        regime_type="Range",
        high_conviction=False,
        allow_rsi_long=True,
        reason="ok",
    )


def _blocked_sma_regime() -> RegimeSnapshot:
    return RegimeSnapshot(
        adx=12.0,
        adx_length=14,
        sma200=101.0,
        sma_length=200,
        sma_slope=-0.5,
        sma_slope_lookback=5,
        price_above_sma200=False,
        regime_type="Range",
        high_conviction=False,
        allow_rsi_long=False,
        reason="blocked:non_positive_slope",
    )


def _patch_compute(monkeypatch: pytest.MonkeyPatch, strat: RSIMeanReversionStrategy, *, rsi_v: float, atr_v: float):
    def _fake_compute(bars: pd.DataFrame):
        return (
            pd.Series(np.full(len(bars), rsi_v), index=bars.index, dtype=float),
            pd.Series(np.full(len(bars), atr_v), index=bars.index, dtype=float),
        )

    monkeypatch.setattr(strat, "_compute", _fake_compute)


def test_tiered_rsi_thresholds_by_atr_pct(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(
        DEFAULT_RSI_ENTRY=30.0,
        HIGH_VOL_RSI_ENTRY=35.0,
        HIGH_VOL_ATR_PCT_THRESHOLD=0.05,
    )
    bars = _bars(close_px=100.0)
    q = _quote("SHOP", bid=99.95, ask=100.0)
    ctx = _ctx("SHOP", bars, q)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _allow_regime())

    low_vol = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, low_vol, rsi_v=32.0, atr_v=3.0)  # atr_pct=3%
    low_actions = [s.action for s in low_vol.evaluate(ctx)]
    assert SignalAction.ENTER_LONG not in low_actions

    high_vol = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, high_vol, rsi_v=32.0, atr_v=6.0)  # atr_pct=6%
    high_actions = [s.action for s in high_vol.evaluate(ctx)]
    assert SignalAction.ENTER_LONG in high_actions


def test_dynamic_rsi_threshold_scales_with_atr(make_settings_factory):
    settings = make_settings_factory(
        DYNAMIC_RSI_ENABLED=True,
        DYNAMIC_RSI_BASE=30.0,
        DYNAMIC_RSI_K=2.0,
        DYNAMIC_RSI_MIN=20.0,
        DYNAMIC_RSI_MAX=35.0,
    )
    strat = RSIMeanReversionStrategy(settings)

    high_thr, _, high_tier, high_ratio = strat._resolve_rsi_entry_threshold(
        atr_value=6.0,
        atr_mean=3.0,
        last_close=100.0,
    )
    low_thr, _, low_tier, low_ratio = strat._resolve_rsi_entry_threshold(
        atr_value=1.5,
        atr_mean=3.0,
        last_close=100.0,
    )

    assert high_ratio == pytest.approx(2.0)
    assert low_ratio == pytest.approx(0.5)
    assert high_thr > 30.0
    assert low_thr < 30.0
    assert high_tier == "DYNAMIC_HIGH_ATR"
    assert low_tier == "DYNAMIC_LOW_ATR"


def test_bollinger_rsi_requires_outer_band_extreme(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(
        BOLLINGER_ENABLED=True,
        BOLLINGER_LENGTH=20,
        BOLLINGER_STD=2.0,
        BOLLINGER_MIN_WIDTH_PCT=0.0,
        BOLLINGER_REQUIRE_TOUCH=True,
    )
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _allow_regime())

    not_extreme = _bars(close_px=100.0)
    not_extreme.iloc[-1, not_extreme.columns.get_loc("close")] = 101.0
    not_extreme.iloc[-1, not_extreme.columns.get_loc("high")] = 101.05
    not_extreme.iloc[-1, not_extreme.columns.get_loc("low")] = 100.95
    fail = RSIMeanReversionStrategy(settings)
    monkeypatch.setattr(
        fail,
        "_bollinger_snapshot",
        lambda _bars: {
            "basis": 100.0,
            "upper": 102.0,
            "lower": 98.0,
            "width_pct": 0.04,
            "price": 101.0,
            "price_below_lower": False,
            "price_above_upper": False,
            "touch": False,
            "require_touch": True,
            "passed": False,
            "min_width_pct": 0.0,
        },
    )
    _patch_compute(monkeypatch, fail, rsi_v=25.0, atr_v=3.0)
    assert SignalAction.ENTER_LONG not in [s.action for s in fail.evaluate(_ctx("SHOP", not_extreme, _quote("SHOP")))]

    extreme = _bars(close_px=100.0)
    extreme.iloc[-1, extreme.columns.get_loc("close")] = 80.0
    extreme.iloc[-1, extreme.columns.get_loc("high")] = 80.05
    extreme.iloc[-1, extreme.columns.get_loc("low")] = 79.95
    ok = RSIMeanReversionStrategy(settings)
    monkeypatch.setattr(
        ok,
        "_bollinger_snapshot",
        lambda _bars: {
            "basis": 100.0,
            "upper": 102.0,
            "lower": 98.0,
            "width_pct": 0.04,
            "price": 80.0,
            "price_below_lower": True,
            "price_above_upper": False,
            "touch": True,
            "require_touch": True,
            "passed": True,
            "min_width_pct": 0.0,
        },
    )
    _patch_compute(monkeypatch, ok, rsi_v=25.0, atr_v=3.0)
    assert SignalAction.ENTER_LONG in [s.action for s in ok.evaluate(_ctx("SHOP", extreme, _quote("SHOP", bid=79.95, ask=80.0)))]


def test_vwap_zscore_calculation_detects_lower_tail():
    n = 45
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    close = pd.Series(np.full(n, 100.0), index=idx)
    close.iloc[-8:-1] = np.linspace(99.8, 99.2, 7)
    close.iloc[-1] = 94.0
    high = close + 0.05
    low = close - 0.05
    volume = pd.Series(np.full(n, 1_000_000.0), index=idx)

    vwap, _upper, lower, zscore, distance, deviation = rolling_vwap_zscore_bands(
        high,
        low,
        close,
        volume,
        length=20,
        z_threshold=2.0,
    )

    assert float(vwap.iloc[-1]) > float(close.iloc[-1])
    assert float(zscore.iloc[-1]) < -2.0
    assert float(close.iloc[-1]) < float(lower.iloc[-1])
    assert float(distance.iloc[-1]) < 0.0
    assert float(deviation.iloc[-1]) > 0.0


def test_bollinger_bandwidth_skip_reason(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    caplog: pytest.LogCaptureFixture,
):
    settings = make_settings_factory(
        BOLLINGER_ENABLED=True,
        BOLLINGER_LENGTH=20,
        BOLLINGER_STD=2.0,
        BOLLINGER_MIN_WIDTH_PCT=0.02,
        BOLLINGER_REQUIRE_TOUCH=False,
    )
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _allow_regime())
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=25.0, atr_v=3.0)

    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        actions = [s.action for s in strat.evaluate(_ctx("SHOP", _bars(close_px=100.0), _quote("SHOP")))]

    assert SignalAction.ENTER_LONG not in actions
    assert any("skip_code=BOLLINGER_FILTER_FAIL" in r.getMessage() for r in caplog.records)
    assert any("skip_bollinger" in r.getMessage() for r in caplog.records)


def test_adx_high_requires_deeper_vwap_zscore(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    caplog: pytest.LogCaptureFixture,
):
    def _high_adx_regime() -> RegimeSnapshot:
        return RegimeSnapshot(
            adx=45.0,
            adx_length=14,
            sma200=95.0,
            sma_length=200,
            sma_slope=0.5,
            sma_slope_lookback=5,
            price_above_sma200=True,
            regime_type="Trending",
            high_conviction=True,
            allow_rsi_long=True,
            reason="ok:sma_slope_positive",
        )

    settings = make_settings_factory(
        VWAP_STRATEGY_ENABLED=True,
        VWAP_Z_THRESHOLD=2.0,
        ADX_HIGH=40.0,
        ADX_LOW=20.0,
    )
    strat = RSIMeanReversionStrategy(settings)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _high_adx_regime())
    monkeypatch.setattr(
        strat,
        "_vwap_snapshot",
        lambda _bars: {
            "vwap": 100.0,
            "upper": 102.0,
            "lower": 98.0,
            "zscore": -2.1,
            "distance_pct": -0.021,
            "deviation": 1.0,
            "price": 97.9,
            "z_threshold": 2.0,
            "passed": True,
        },
    )
    _patch_compute(monkeypatch, strat, rsi_v=25.0, atr_v=3.0)

    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        actions = [s.action for s in strat.evaluate(_ctx("SHOP", _bars(close_px=100.0), _quote("SHOP")))]

    assert SignalAction.ENTER_LONG not in actions
    assert any("skip_code=ADX_FILTER_FAIL" in r.getMessage() for r in caplog.records)
    assert any("vwap_deep_z_required" in r.getMessage() for r in caplog.records)


def test_volatility_gate_logging_includes_threshold_and_tier(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    caplog: pytest.LogCaptureFixture,
):
    settings = make_settings_factory(
        DEFAULT_RSI_ENTRY=30.0,
        HIGH_VOL_RSI_ENTRY=35.0,
        HIGH_VOL_ATR_PCT_THRESHOLD=0.05,
    )
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=32.0, atr_v=6.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _allow_regime())

    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        list(strat.evaluate(_ctx("SHOP", _bars(), _quote("SHOP"))))
    line = next((r.getMessage() for r in caplog.records if "event=strategy_volatility_gate" in r.getMessage()), "")
    assert "volatility_tier=HIGH_VOL" in line
    assert "rsi_threshold=35.0000" in line


def test_aggressive_mode_sma_bypass_cases(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    caplog: pytest.LogCaptureFixture,
):
    bars = _bars(close_px=100.0)
    q = _quote("MARA", bid=99.95, ask=100.0)
    ctx = _ctx("MARA", bars, q)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _blocked_sma_regime())

    s_off = make_settings_factory(AGGRESSIVE_MODE=False, AGGRESSIVE_RSI_BYPASS_THRESHOLD=20.0)
    strat_off = RSIMeanReversionStrategy(s_off)
    _patch_compute(monkeypatch, strat_off, rsi_v=18.0, atr_v=3.0)
    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        off_actions = [s.action for s in strat_off.evaluate(ctx)]
    assert SignalAction.ENTER_LONG not in off_actions
    assert any("code=SMA_FILTER_FAIL" in r.getMessage() for r in caplog.records)
    caplog.clear()

    s_on_no_bypass = make_settings_factory(AGGRESSIVE_MODE=True, AGGRESSIVE_RSI_BYPASS_THRESHOLD=20.0)
    strat_on_no_bypass = RSIMeanReversionStrategy(s_on_no_bypass)
    _patch_compute(monkeypatch, strat_on_no_bypass, rsi_v=24.0, atr_v=3.0)
    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        no_bypass_actions = [s.action for s in strat_on_no_bypass.evaluate(ctx)]
    assert SignalAction.ENTER_LONG not in no_bypass_actions
    assert any("code=SMA_FILTER_FAIL" in r.getMessage() for r in caplog.records)
    caplog.clear()

    s_on_bypass = make_settings_factory(AGGRESSIVE_MODE=True, AGGRESSIVE_RSI_BYPASS_THRESHOLD=20.0)
    strat_on_bypass = RSIMeanReversionStrategy(s_on_bypass)
    _patch_compute(monkeypatch, strat_on_bypass, rsi_v=18.0, atr_v=3.0)
    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        bypass_actions = [s.action for s in strat_on_bypass.evaluate(ctx)]
    assert SignalAction.ENTER_LONG in bypass_actions
    assert any("code=AGGRESSIVE_SMA_BYPASS" in r.getMessage() for r in caplog.records)
