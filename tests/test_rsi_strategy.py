"""Tests for the RSI mean reversion strategy and emergency flatten construction."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.account import AccountSnapshot
from core.market_data import Quote
from strategies.base import SignalAction, StrategyContext
from strategies.indicators import atr, rsi
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from utils.price_utils import round_to_tick


def _make_account(equity=100_000.0) -> AccountSnapshot:
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


def _make_bars(n: int, *, drop_last_pct: float = 0.0) -> pd.DataFrame:
    """Build a deterministic price series. If drop_last_pct > 0, force the
    final segment to dive so RSI prints oversold."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz=timezone.utc)
    rng = np.random.default_rng(42)
    base = 100.0 + np.cumsum(rng.normal(0, 0.05, size=n))
    if drop_last_pct > 0:
        # Force a sharp decline over the final 30 bars.
        decline = np.linspace(0, -drop_last_pct * base[-30], 30)
        base[-30:] = base[-30:] + decline
    close = pd.Series(base, index=idx)
    high = close + 0.05
    low = close - 0.05
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.full(n, 1_000_000), index=idx)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _make_quote(symbol: str = "AAPL", bid: float = 99.95, ask: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def test_rsi_indicator_known_values():
    closes = pd.Series([float(x) for x in range(1, 30)])  # monotonic up
    r = rsi(closes, length=14)
    # Monotonic increase -> RSI saturates near 100.
    assert r.iloc[-1] > 70


def test_atr_indicator_positive():
    n = 50
    closes = pd.Series(np.linspace(100, 110, n))
    highs = closes + 0.5
    lows = closes - 0.5
    a = atr(highs, lows, closes, length=14)
    assert a.dropna().iloc[-1] > 0


def test_rsi_strategy_no_signal_when_not_oversold(settings):
    strat = RSIMeanReversionStrategy(settings)
    bars = _make_bars(200)  # no forced drop
    ctx = StrategyContext(
        symbol="AAPL",
        bars=bars,
        quote=_make_quote(),
        account=_make_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = list(strat.evaluate(ctx))
    # In a stable random-walk seeded series, there shouldn't be a clean oversold.
    actions = [s.action for s in signals]
    assert SignalAction.ENTER_LONG not in actions or all(
        s.metadata.get("rsi", 100) < settings.RSI_OVERSOLD for s in signals if s.metadata
    )


def test_rsi_strategy_emits_enter_when_oversold(settings):
    strat = RSIMeanReversionStrategy(settings)
    # Construct a clearly oversold tail by appending a strong decline.
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz=timezone.utc)
    base = np.full(n, 100.0)
    # Drop fast in the last 50 bars to push RSI into oversold.
    base[-50:] = np.linspace(100, 80, 50)
    close = pd.Series(base, index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000), index=idx),
        }
    )
    ctx = StrategyContext(
        symbol="AAPL",
        bars=bars,
        quote=_make_quote(bid=79.95, ask=80.0),
        account=_make_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = list(strat.evaluate(ctx))
    actions = [s.action for s in signals]
    assert SignalAction.ENTER_LONG in actions


def test_rsi_strategy_blocks_entry_when_open_order_present(settings):
    strat = RSIMeanReversionStrategy(settings)
    n = 200
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz=timezone.utc)
    base = np.full(n, 100.0)
    base[-50:] = np.linspace(100, 80, 50)
    close = pd.Series(base, index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000), index=idx),
        }
    )
    ctx = StrategyContext(
        symbol="AAPL",
        bars=bars,
        quote=_make_quote(),
        account=_make_account(),
        positions_by_symbol={},
        open_order_symbols={"AAPL"},
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = list(strat.evaluate(ctx))
    assert all(s.action != SignalAction.ENTER_LONG for s in signals)


def test_emergency_flatten_long_constructs_below_bid():
    """Emergency long flatten = sell limit IOC at price <= current bid."""
    quote = _make_quote(bid=100.00, ask=100.10)
    aggressiveness_pct = 0.001  # 10 bps
    mid = (quote.bid + quote.ask) / 2.0
    raw = quote.bid - aggressiveness_pct * mid
    rounded = round_to_tick(raw, mode="down")
    assert rounded <= quote.bid
    assert rounded > 0


def test_emergency_flatten_short_constructs_above_ask():
    """Emergency short flatten = buy limit IOC at price >= current ask."""
    quote = _make_quote(bid=100.00, ask=100.10)
    aggressiveness_pct = 0.001
    mid = (quote.bid + quote.ask) / 2.0
    raw = quote.ask + aggressiveness_pct * mid
    rounded = round_to_tick(raw, mode="up")
    assert rounded >= quote.ask
