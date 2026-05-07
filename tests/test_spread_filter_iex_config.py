"""Settings and universe behaviour for optional IEX-specific spread cap."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from core.market_data import Quote
from strategies.universe import UniverseFilter
from tests.conftest import make_settings


def test_spread_filter_pct_for_feed_default_all_feeds() -> None:
    s = make_settings(SPREAD_FILTER_PCT=0.001)
    assert s.spread_filter_pct_for_feed("iex") == pytest.approx(0.001)
    assert s.spread_filter_pct_for_feed("sip") == pytest.approx(0.001)
    assert s.spread_filter_pct_for_feed(None) == pytest.approx(0.001)


def test_spread_filter_pct_for_feed_iex_override() -> None:
    s = make_settings(SPREAD_FILTER_PCT=0.0005, SPREAD_FILTER_PCT_IEX=0.002)
    assert s.spread_filter_pct_for_feed("iex") == pytest.approx(0.002)
    assert s.spread_filter_pct_for_feed("sip") == pytest.approx(0.0005)


def test_spread_filter_pct_iex_invalid_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(SPREAD_FILTER_PCT_IEX=0.0)
    with pytest.raises(ValidationError):
        make_settings(SPREAD_FILTER_PCT_IEX=0.06)


def test_spread_filter_pct_iex_empty_string_is_none() -> None:
    s = make_settings(SPREAD_FILTER_PCT=0.0005, SPREAD_FILTER_PCT_IEX="")  # type: ignore[arg-type]
    assert s.SPREAD_FILTER_PCT_IEX is None


def test_universe_marginal_quote_passes_with_iex_override() -> None:
    """~6 bps relative spread: fails 5 bps default but passes 1% IEX cap."""
    s = make_settings(SPREAD_FILTER_PCT=0.0005, SPREAD_FILTER_PCT_IEX=0.01)
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="SPY",
        bid=100.0,
        ask=100.06,
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    r = uf.is_eligible("SPY", quote=q, bars=pd.DataFrame(), has_position=False, has_open_order=False)
    assert r.eligible


def test_universe_same_quote_fails_on_sip_with_iex_override_only() -> None:
    s = make_settings(SPREAD_FILTER_PCT=0.0005, SPREAD_FILTER_PCT_IEX=0.01)
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="SPY",
        bid=100.0,
        ask=100.06,
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="sip",
    )
    r = uf.is_eligible("SPY", quote=q, bars=pd.DataFrame(), has_position=False, has_open_order=False)
    assert not r.eligible
    assert "spread" in r.reason


def test_universe_elastic_spread_allows_marginal_iex_low_price_quote() -> None:
    s = make_settings(
        SPREAD_FILTER_PCT=0.0005,
        SPREAD_FILTER_ELASTIC_ENABLED=True,
        SPREAD_FILTER_MAX_PCT=0.005,
        SPREAD_FILTER_LOW_PRICE_THRESHOLD=25.0,
        SPREAD_FILTER_LOW_PRICE_MULTIPLIER=1.5,
        SPREAD_FILTER_IEX_ELASTIC_MULTIPLIER=1.75,
        SPREAD_FILTER_SPARSE_QUOTE_MULTIPLIER=1.25,
        SPREAD_FILTER_FRESH_QUOTE_MULTIPLIER=1.15,
    )
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="SPY",
        bid=10.00,
        ask=10.018,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    r = uf.is_eligible("SPY", quote=q, bars=pd.DataFrame(), has_position=False, has_open_order=False)
    assert r.eligible


def test_universe_elastic_spread_still_blocks_unsafe_quote() -> None:
    s = make_settings(
        SPREAD_FILTER_PCT=0.0005,
        SPREAD_FILTER_ELASTIC_ENABLED=True,
        SPREAD_FILTER_MAX_PCT=0.005,
    )
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="SPY",
        bid=10.0,
        ask=10.20,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    r = uf.is_eligible("SPY", quote=q, bars=pd.DataFrame(), has_position=False, has_open_order=False)
    assert not r.eligible
    assert r.code == "SPREAD_TOO_WIDE"
