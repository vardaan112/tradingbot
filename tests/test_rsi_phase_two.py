"""Phase Two: regime skip logging, synthetic trailing-profit, persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config.constants import LOGGER_STRATEGY
from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote
from core.state_store import StateStore
from strategies.base import SignalAction, StrategyContext
from strategies.rsi_strategy import RSIMeanReversionStrategy


def _account(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity * 2,
        regt_buying_power=equity * 2,
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


def _quote(bid: float, ask: float) -> Quote:
    return Quote(
        symbol="SPY",
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _ctx(*, bars, position=None, bid=99.9, ask=100.0):
    return StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=bid, ask=ask),
        account=_account(),
        positions_by_symbol={"SPY": position} if position else {},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )


def _bear_trend_bars(n: int = 320) -> pd.DataFrame:
    """Strong down-trend ending deeply oversold (regime typically blocks mean-reversion entries)."""
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=30),
        periods=n,
        freq="5min",
    )
    t = np.linspace(240.0, 75.0, n)
    close = pd.Series(t, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.25,
            "low": close - 1.25,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _trail_bars(*, last_close: float, n: int = 260) -> pd.DataFrame:
    """Volatile recent window so ATR is material; last close controls unrealized PnL %."""
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=30),
        periods=n,
        freq="5min",
    )
    closes = np.full(n, 100.0)
    for i in range(n - 25, n - 1):
        closes[i] = 100.0 + 1.1 * ((-1) ** (i - (n - 25)))
    closes[-1] = last_close
    c = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c.iloc[0]),
            "high": c + 2.5,
            "low": c - 2.5,
            "close": c,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def test_regime_skip_emits_structured_log(make_settings_factory, caplog):
    settings = make_settings_factory(ADX_RANGE_MAX=25.0)
    strat = RSIMeanReversionStrategy(settings)
    bars = _bear_trend_bars()
    ctx = _ctx(bars=bars, bid=float(bars["close"].iloc[-1]) - 0.05, ask=float(bars["close"].iloc[-1]))
    with caplog.at_level(logging.INFO, logger=LOGGER_STRATEGY):
        signals = list(strat.evaluate(ctx))
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)
    assert any("event=strategy_skip_regime" in r.getMessage() for r in caplog.records)


def test_signal_log_includes_regime_and_trailing_fields(make_settings_factory, caplog):
    settings = make_settings_factory(ADX_RANGE_MAX=100.0)
    strat = RSIMeanReversionStrategy(settings)
    bars = _bear_trend_bars()
    # Relax quote to still valid spread around last close
    lc = float(bars["close"].iloc[-1])
    ctx = _ctx(bars=bars, bid=lc - 0.05, ask=lc + 0.05)
    with caplog.at_level(logging.INFO, logger=LOGGER_STRATEGY):
        list(strat.evaluate(ctx))
    hit = [
        r.getMessage()
        for r in caplog.records
        if "event=strategy_signal" in r.getMessage() and "enter_long" in r.getMessage()
    ]
    assert hit
    msg = hit[0]
    assert "regime_type=" in msg
    assert "trailing_stop_active=false" in msg


def test_trailing_profit_activate_and_exit_on_breach(make_settings_factory):
    settings = make_settings_factory(
        ADX_RANGE_MAX=100.0,
        TRAIL_TRIGGER_PCT=0.01,
        TRAIL_LOCKED_PROFIT_PCT=0.005,
        TRAIL_ATR_MULTIPLIER=1.5,
        ATR_STOP_MULTIPLIER=2.0,
        RSI_EXIT=95.0,
    )
    strat = RSIMeanReversionStrategy(settings)
    bars = _trail_bars(last_close=101.2)
    last_close = float(bars["close"].iloc[-1])
    pos = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=100.0,
        side="long",
        market_value=1012.0,
        cost_basis=1000.0,
        unrealized_pl=12.0,
        current_price=last_close,
    )
    ctx = _ctx(bars=bars, position=pos, bid=last_close - 0.05, ask=last_close + 0.05)
    assert list(strat.evaluate(ctx)) == []

    trail = strat._trails_by_symbol.get("SPY")
    assert trail is not None
    assert trail.trailing_stop_active is True
    locked = 100.0 * (1.0 + settings.TRAIL_LOCKED_PROFIT_PCT)
    assert trail.trailing_stop_price >= locked - 1e-9

    breach_close = trail.trailing_stop_price - 0.02
    bars2 = _trail_bars(last_close=breach_close, n=len(bars))
    lc2 = float(bars2["close"].iloc[-1])
    ctx2 = _ctx(bars=bars2, position=pos, bid=lc2 - 0.05, ask=lc2 + 0.05)
    sigs = list(strat.evaluate(ctx2))
    ex = [s for s in sigs if s.action == SignalAction.EXIT_LONG]
    assert ex
    assert "trailing_profit_breach" in ex[0].reason
    md = ex[0].metadata
    assert md.get("trailing_stop_active") is True
    assert md.get("target_a_hit") is True
    assert float(md.get("trailing_stop_price", 0.0)) >= locked - 1e-9
    assert float(md.get("highest_close_since_activation", 0.0)) >= 101.15


def test_catastrophic_stop_still_emergency_before_trailing(make_settings_factory):
    settings = make_settings_factory(
        ADX_RANGE_MAX=100.0,
        ATR_STOP_MULTIPLIER=1.0,
        TRAIL_TRIGGER_PCT=0.01,
    )
    strat = RSIMeanReversionStrategy(settings)
    bars = _trail_bars(last_close=74.0, n=260)
    lc = float(bars["close"].iloc[-1])
    pos = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=100.0,
        side="long",
        market_value=740.0,
        cost_basis=1000.0,
        unrealized_pl=-260.0,
        current_price=lc,
    )
    ctx = _ctx(bars=bars, position=pos, bid=lc - 0.05, ask=lc + 0.05)
    sigs = list(strat.evaluate(ctx))
    assert any(s.action == SignalAction.EMERGENCY_EXIT_LONG for s in sigs)
    assert not any(s.action == SignalAction.EXIT_LONG for s in sigs)


def test_trailing_state_persists_and_reloads(make_settings_factory, tmp_path):
    settings = make_settings_factory(
        ADX_RANGE_MAX=100.0,
        TRAIL_TRIGGER_PCT=0.01,
        TRAIL_LOCKED_PROFIT_PCT=0.005,
        TRAIL_ATR_MULTIPLIER=1.5,
        ATR_STOP_MULTIPLIER=2.0,
        RSI_EXIT=95.0,
    )
    store = StateStore(tmp_path)
    strat = RSIMeanReversionStrategy(settings, state_store=store)
    bars = _trail_bars(last_close=101.2)
    last_close = float(bars["close"].iloc[-1])
    pos = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=100.0,
        side="long",
        market_value=1012.0,
        cost_basis=1000.0,
        unrealized_pl=12.0,
        current_price=last_close,
    )
    list(strat.evaluate(_ctx(bars=bars, position=pos, bid=last_close - 0.05, ask=last_close + 0.05)))
    assert (tmp_path / "trail_trailing_state.json").exists()

    strat2 = RSIMeanReversionStrategy(settings, state_store=store)
    st = strat2._trails_by_symbol.get("SPY")
    assert st is not None
    assert st.trailing_stop_active is True
