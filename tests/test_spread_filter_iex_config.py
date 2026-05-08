"""Settings and universe behaviour for optional IEX-specific spread cap."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from core.market_data import Quote
from strategies.universe import UniverseFilter
from tests.conftest import make_settings


def _bars(*, close: float = 100.0, volume: float = 1_000.0, rows: int = 30) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=5),
        periods=rows,
        freq="5min",
    )
    closes = pd.Series([close] * rows, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.10,
            "low": closes - 0.10,
            "close": closes,
            "volume": pd.Series([volume] * rows, index=idx, dtype=float),
        },
        index=idx,
    )


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
    s = make_settings(
        SPREAD_FILTER_PCT=0.0005,
        SPREAD_FILTER_PCT_IEX=0.01,
        MIN_SPREAD_THRESHOLD_PERCENT=0.0,
    )
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


def test_elastic_spread_aliases_and_profit_budget_cap() -> None:
    s = make_settings(
        SPREAD_FILTER_PCT=0.01,
        ELASTIC_SPREAD_ENABLED=True,
        ELASTIC_SPREAD_HARD_MAX_PCT=0.02,
        ELASTIC_SPREAD_TARGET_PROFIT_PCT=0.015,
        ELASTIC_SPREAD_MAX_COST_FRACTION=0.35,
        SPREAD_FILTER_IEX_ELASTIC_MULTIPLIER=1.0,
        SPREAD_FILTER_FRESH_QUOTE_MULTIPLIER=1.0,
    )
    assert s.SPREAD_FILTER_ELASTIC_ENABLED is True
    assert s.SPREAD_FILTER_MAX_PCT == pytest.approx(0.02)

    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="AMD",
        bid=100.0,
        ask=100.60,  # ~60 bps > target-profit budget cap of 52.5 bps.
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    r = uf.is_eligible("AMD", quote=q, bars=pd.DataFrame(), has_position=False, has_open_order=False)
    assert not r.eligible
    assert r.code == "SPREAD_TOO_WIDE"


def test_adv_uses_daily_projection_for_five_minute_bars() -> None:
    s = make_settings(
        BAR_TIMEFRAME="5Min",
        MIN_AVG_DOLLAR_VOLUME=5_000_000.0,
        MIN_PRICE=1.0,
        SPREAD_FILTER_PCT=0.0001,
    )
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="NVDA",
        bid=100.0,
        ask=100.01,
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

    # Per-bar dollar volume is only 100k, but projected 5m daily ADV is 7.8m.
    r = uf.is_eligible("NVDA", quote=q, bars=_bars(close=100.0, volume=1_000.0), has_position=False, has_open_order=False)

    assert r.eligible


def test_min_spread_threshold_floor_allows_low_vol_penny_spread() -> None:
    s = make_settings(
        SPREAD_FILTER_PCT=0.0001,
        SPREAD_FILTER_PCT_IEX=0.0001,
        SPREAD_FILTER_ELASTIC_ENABLED=False,
        MIN_SPREAD_THRESHOLD_PERCENT=0.0008,
        MIN_AVG_DOLLAR_VOLUME=0.0,
        MIN_PRICE=1.0,
    )
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="PYPL",
        bid=100.00,
        ask=100.07,  # ~7 bps, above 1 bp calculated cap but below 8 bps floor.
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

    r = uf.is_eligible("PYPL", quote=q, bars=_bars(close=100.0), has_position=False, has_open_order=False)

    assert r.eligible


def test_stale_quote_grace_allows_within_spread_multiplier() -> None:
    s = make_settings(
        QUOTE_MAX_AGE_SECONDS=10.0,
        QUOTE_STRICT_MAX_AGE_SECONDS=3.0,
        QUOTE_STALE_SPREAD_MULTIPLIER=1.5,
        SPREAD_FILTER_PCT=0.001,
        SPREAD_FILTER_PCT_IEX=0.001,
        SPREAD_FILTER_ELASTIC_ENABLED=False,
        MIN_SPREAD_THRESHOLD_PERCENT=0.0008,
        MIN_AVG_DOLLAR_VOLUME=0.0,
        MIN_PRICE=1.0,
    )
    uf = UniverseFilter(s, strategy_name="rsi_meanrev")
    q = Quote(
        symbol="COIN",
        bid=100.00,
        ask=100.14,  # ~14 bps, within 1.5x the 10 bps base threshold.
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=5),
        feed="iex",
    )

    r = uf.is_eligible("COIN", quote=q, bars=_bars(close=100.0), has_position=False, has_open_order=False)

    assert r.eligible
