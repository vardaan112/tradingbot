"""Pure helpers for dashboard Live Watchlist (no Streamlit session)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.conftest import make_settings
from utils.dashboard import (
    WATCHLIST_DEFAULT_SYMBOLS,
    dashboard_drop_inprogress_bar,
    watchlist_rsi_signal_label,
    watchlist_symbols,
)


def test_watchlist_symbols_uses_defaults_when_symbols_empty() -> None:
    from types import SimpleNamespace

    empty = SimpleNamespace(symbols_list=[])
    assert watchlist_symbols(empty) == list(WATCHLIST_DEFAULT_SYMBOLS)  # type: ignore[arg-type]


def test_watchlist_symbols_parses_config():
    s = make_settings(SYMBOLS="spy, tsla")
    assert watchlist_symbols(s) == ["SPY", "TSLA"]


def test_rsi_signal_buckets():
    assert watchlist_rsi_signal_label(
        20.0, rsi_ready=True, oversold=30.0, overbought=70.0,
    ) == "Oversold"
    assert watchlist_rsi_signal_label(
        50.0, rsi_ready=True, oversold=30.0, overbought=70.0,
    ) == "Neutral"
    assert watchlist_rsi_signal_label(
        80.0, rsi_ready=True, oversold=30.0, overbought=70.0,
    ) == "Overbought"
    assert watchlist_rsi_signal_label(
        float("nan"), rsi_ready=False, oversold=30.0, overbought=70.0,
    ) == "Warming Up"


def test_drop_inprogress_bar_removes_partial_tail():
    # Last bar timestamp is recent -> should drop forming bar when delta is 5m
    ts = datetime.now(timezone.utc) - timedelta(minutes=1)
    ix = pd.DatetimeIndex([ts - timedelta(minutes=30), ts], tz="UTC")
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=ix)
    out = dashboard_drop_inprogress_bar(df, "5Min")
    assert len(out) == 1


def test_drop_inprogress_bar_keeps_completed_history():
    ts = datetime.now(timezone.utc) - timedelta(minutes=30)
    ix = pd.DatetimeIndex([ts - timedelta(minutes=10), ts], tz="UTC")
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=ix)
    out = dashboard_drop_inprogress_bar(df, "5Min")
    assert len(out) == 2
