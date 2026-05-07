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
