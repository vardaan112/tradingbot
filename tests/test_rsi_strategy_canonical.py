"""Tests for the canonical RSI mean-reversion strategy.

Covers:
- enter_long emission when RSI is oversold and gates pass
- exit_long emission on rsi_exit / tp_hit / time_exit
- emergency_exit_long emission on ATR stop breach
- no-lookahead behavior: only completed bars influence the signal
- backward compat: importing from rsi_mean_reversion still works
- enriched Signal.metadata contents
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote
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


def _quote(symbol: str = "SPY", bid: float = 99.95, ask: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _bars_with_drop(n: int = 200, *, drop_to: float = 80.0) -> pd.DataFrame:
    """Bar series flat at 100 then linearly declining to drop_to over the last 50 bars."""
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    base = np.full(n, 100.0)
    base[-50:] = np.linspace(100.0, drop_to, 50)
    close = pd.Series(base, index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000), index=idx),
        }
    )


def _flat_bars(n: int = 200, *, level: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    base = np.full(n, level)
    close = pd.Series(base, index=idx)
    # introduce tiny noise so ATR is non-zero
    rng = np.random.default_rng(7)
    close = close + rng.normal(0, 0.02, size=n)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000), index=idx),
        }
    )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_legacy_import_path_still_works():
    """Existing imports from strategies.rsi_mean_reversion must still resolve."""
    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy as Legacy
    assert Legacy is RSIMeanReversionStrategy


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def test_enter_long_when_rsi_oversold(settings):
    strat = RSIMeanReversionStrategy(settings)
    bars = _bars_with_drop()
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=79.95, ask=80.0),
        account=_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = list(strat.evaluate(ctx))
    assert any(s.action == SignalAction.ENTER_LONG for s in signals)


def test_signal_metadata_includes_required_fields(settings):
    strat = RSIMeanReversionStrategy(settings)
    bars = _bars_with_drop()
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=79.95, ask=80.0),
        account=_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = [s for s in strat.evaluate(ctx) if s.action == SignalAction.ENTER_LONG]
    assert signals
    md = signals[0].metadata
    for key in (
        "rsi",
        "atr",
        "last_close",
        "bar_timestamp",
        "quote_bid",
        "quote_ask",
        "spread_pct",
        "quote_age_seconds",
        "strategy_name",
    ):
        assert key in md, f"missing metadata key: {key}"
    assert md["strategy_name"] == "rsi_meanrev"


def test_no_signal_when_open_order_present(settings):
    strat = RSIMeanReversionStrategy(settings)
    bars = _bars_with_drop()
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=79.95, ask=80.0),
        account=_account(),
        positions_by_symbol={},
        open_order_symbols={"SPY"},
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    actions = [s.action for s in strat.evaluate(ctx)]
    assert SignalAction.ENTER_LONG not in actions


def test_no_signal_in_flat_market(settings):
    strat = RSIMeanReversionStrategy(settings)
    bars = _flat_bars()
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(),
        account=_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    actions = [s.action for s in strat.evaluate(ctx)]
    # In a near-flat series RSI should sit in the middle band; no entries.
    assert SignalAction.ENTER_LONG not in actions


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


def test_emergency_exit_on_atr_stop_breach(make_settings_factory):
    settings = make_settings_factory(ATR_STOP_MULTIPLIER=1.0)
    strat = RSIMeanReversionStrategy(settings)
    # Build bars where price stayed near 100 then sharply dropped.
    n = 200
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    base = np.full(n, 100.0)
    base[-1] = 80.0  # last completed bar dropped hard
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

    position = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=100.0,
        side="long",
        market_value=800.0,
        cost_basis=1000.0,
        unrealized_pl=-200.0,
        current_price=80.0,
    )
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=79.95, ask=80.0),
        account=_account(),
        positions_by_symbol={"SPY": position},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    actions = [s.action for s in strat.evaluate(ctx)]
    assert SignalAction.EMERGENCY_EXIT_LONG in actions


def test_exit_on_rsi_revert_above_threshold(settings):
    strat = RSIMeanReversionStrategy(settings)
    # Climb from 80 to 110 in last 50 bars -> RSI saturates high.
    n = 200
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    base = np.full(n, 100.0)
    base[-50:] = np.linspace(80.0, 110.0, 50)
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

    position = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=85.0,
        side="long",
        market_value=1100.0,
        cost_basis=850.0,
        unrealized_pl=250.0,
        current_price=110.0,
    )
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=109.95, ask=110.05),
        account=_account(),
        positions_by_symbol={"SPY": position},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    signals = [s for s in strat.evaluate(ctx)]
    actions = [s.action for s in signals]
    # Expect EXIT_LONG (rsi_exit OR tp_hit) - both are normal exits, not emergency.
    assert SignalAction.EXIT_LONG in actions
    assert SignalAction.EMERGENCY_EXIT_LONG not in actions


def test_exit_on_max_hold_time(make_settings_factory):
    settings = make_settings_factory(MAX_HOLD_BARS=2)
    strat = RSIMeanReversionStrategy(settings)
    bars = _flat_bars(n=200, level=100.0)
    position = PositionSnapshot(
        symbol="SPY",
        qty=10,
        avg_entry_price=100.0,
        side="long",
        market_value=1000.0,
        cost_basis=1000.0,
        unrealized_pl=0.0,
        current_price=100.0,
    )
    ctx = StrategyContext(
        symbol="SPY",
        bars=bars,
        quote=_quote(),
        account=_account(),
        positions_by_symbol={"SPY": position},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )
    # No prior entry index recorded -> defaults to len(bars)-1, held_bars=0.
    # Hand-record the entry index so the time-exit path triggers deterministically.
    strat._entry_bar_index["SPY"] = len(bars) - 5  # held 4 bars >= MAX_HOLD_BARS=2
    signals = list(strat.evaluate(ctx))
    actions = [s.action for s in signals]
    assert SignalAction.EXIT_LONG in actions


# ---------------------------------------------------------------------------
# No-lookahead
# ---------------------------------------------------------------------------


def test_in_progress_trailing_bar_is_dropped(settings):
    """If the trailing bar's timestamp is newer than 'now - timeframe', it must
    be considered in-progress and dropped before computing the signal."""
    strat = RSIMeanReversionStrategy(settings)

    # Build a bar series where the last bar is timestamped 'just now' so it is
    # still in progress for a 5-minute timeframe. The penultimate bar is
    # neutral so RSI is not in oversold territory; an erroneous lookahead
    # implementation that uses the (manipulated) last bar would produce a
    # signal, while the correct implementation must not.
    n = 200
    now = datetime.now(timezone.utc)
    idx_completed = pd.date_range(
        end=now - timedelta(minutes=10),
        periods=n - 1,
        freq="5min",
    )
    last_ts_in_progress = now - timedelta(seconds=5)
    idx = idx_completed.append(pd.DatetimeIndex([last_ts_in_progress]))

    # First n-1 bars: flat at 100. Final bar (in-progress): crash to 60 to
    # force RSI oversold IF it were used.
    closes = np.full(n, 100.0)
    closes[-1] = 60.0
    close = pd.Series(closes, index=idx)
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
        symbol="SPY",
        bars=bars,
        quote=_quote(bid=99.95, ask=100.0),
        account=_account(),
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=now,
        feed="iex",
    )
    signals = list(strat.evaluate(ctx))
    # The dropped in-progress bar must NOT influence the signal: RSI on the
    # remaining flat series stays mid-band so we expect NO entry.
    assert all(s.action != SignalAction.ENTER_LONG for s in signals)
