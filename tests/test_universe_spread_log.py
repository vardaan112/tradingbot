"""Tests for structured spread-skip logging in the universe filter."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from core.market_data import Quote
from strategies.universe import UniverseFilter


def _wide_quote() -> Quote:
    # 200 bps spread on a $100 stock -> well above the default 5 bps filter.
    return Quote(
        symbol="QQQ",
        bid=100.0,
        ask=102.0,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def test_spread_skip_emits_structured_log_with_required_fields(settings, caplog):
    uf = UniverseFilter(settings, strategy_name="rsi_meanrev")
    bars = pd.DataFrame()
    quote = _wide_quote()
    with caplog.at_level(logging.INFO, logger="tradingbot.strategy"):
        result = uf.is_eligible(
            "QQQ",
            quote=quote,
            bars=bars,
            has_position=False,
            has_open_order=False,
        )

    assert not result.eligible
    assert "spread_" in result.reason

    # Find the structured spread-skip log line.
    skip_lines = [r.message for r in caplog.records if "event=strategy_skip_spread" in r.message]
    assert skip_lines, "no event=strategy_skip_spread log emitted"
    msg = skip_lines[0]
    for needle in (
        "symbol=QQQ",
        "bid=100.0000",
        "ask=102.0000",
        "mid=",
        "spread_pct=",
        f"spread_threshold={settings.SPREAD_FILTER_PCT:.6f}",
        "feed=iex",
        "quote_age_seconds=",
        "strategy=rsi_meanrev",
        "timestamp=",
    ):
        assert needle in msg, f"missing {needle!r} in spread skip log: {msg}"


def test_spread_skip_not_logged_when_eligible(settings, caplog):
    """Tight spread should pass through without a spread-skip log line."""
    uf = UniverseFilter(settings, strategy_name="rsi_meanrev")
    quote = Quote(
        symbol="QQQ",
        bid=100.00,
        ask=100.02,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    bars = pd.DataFrame()
    with caplog.at_level(logging.INFO, logger="tradingbot.strategy"):
        result = uf.is_eligible(
            "QQQ",
            quote=quote,
            bars=bars,
            has_position=False,
            has_open_order=False,
        )

    skip_lines = [r.message for r in caplog.records if "event=strategy_skip_spread" in r.message]
    assert not skip_lines
    # Eligibility may still fail on price/volume gates depending on settings,
    # but it must not be the spread filter rejection.
    assert "spread" not in result.reason
