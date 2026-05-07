"""Unit tests for entry skip diagnostic helpers."""

from __future__ import annotations

import logging

from datetime import UTC, datetime

import pytest

from core.market_data import Quote
from strategies.filters import RegimeSnapshot
from strategies.skip_diagnostics import (
    SkipCodes,
    SkipDiagnosticsThrottle,
    SkipReason,
    emit_skip_diagnostic,
    format_skip_log_line,
    regime_skip_reason,
    skip_reason_to_discord_spec,
)
from tests.conftest import make_settings


def test_format_skip_log_line_contains_symbol_and_code() -> None:
    sr = SkipReason(
        code="test_code",
        message="hello world",
        symbol="SPY",
        rsi=28.5,
        metadata={"foo": 1},
    )
    line = format_skip_log_line("strategy_entry_skip", sr, strategy_name="rsi_meanrev")
    assert "event=strategy_entry_skip" in line
    assert "code=test_code" in line
    assert "symbol=SPY" in line
    assert "rsi=28.5" in line
    assert "meta_foo=1" in line


def test_skip_throttle_log_and_discord_independent() -> None:
    th = SkipDiagnosticsThrottle()
    assert th.allow_discord("SPY", "x", 10.0) is True
    assert th.allow_discord("SPY", "x", 10.0) is False
    assert th.allow_log("SPY", "x", 10.0) is True
    assert th.allow_log("SPY", "x", 10.0) is False


def test_emit_skip_respects_noisy_log_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    log = logging.getLogger("t.skip")
    th = SkipDiagnosticsThrottle()
    records: list[str] = []

    def capture(*args: object, **_: object) -> None:
        records.append(str(args[0]))

    monkeypatch.setattr(log, "info", capture)
    s = make_settings(
        SKIP_DIAGNOSTICS_DISCORD_COOLDOWN_SECONDS=9999.0,
        SKIP_DIAGNOSTICS_NOISY_LOG_THROTTLE_SECONDS=3600.0,
    )
    sr = SkipReason(
        code=SkipCodes.RSI_NOT_OVERSOLD,
        message="x",
        symbol="X",
    )
    emit_skip_diagnostic(
        settings=s,
        logger=log,
        log_event="e",
        sr=sr,
        discord_enqueue=None,
        throttle=th,
    )
    emit_skip_diagnostic(
        settings=s,
        logger=log,
        log_event="e",
        sr=sr,
        discord_enqueue=None,
        throttle=th,
    )
    assert len(records) == 1


def test_regime_skip_reason_maps_regime_fields() -> None:
    r = RegimeSnapshot(
        adx=30.0,
        adx_length=14,
        sma200=400.0,
        sma_length=200,
        sma_slope=0.01,
        sma_slope_lookback=5,
        price_above_sma200=True,
        regime_type="Trending",
        high_conviction=True,
        allow_rsi_long=False,
        reason="adx_range_market",
    )
    q = Quote(
        symbol="IWM",
        bid=100.0,
        ask=100.1,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(UTC),
        feed="sip",
    )
    sr = regime_skip_reason(
        symbol="IWM",
        regime_reason=r.reason,
        rsi_value=25.0,
        last_close=400.0,
        regime=r,
        quote=q,
    )
    assert sr.code == SkipCodes.REGIME_FILTER
    assert sr.adx == 30.0
    assert sr.sma_200 == 400.0
    assert sr.spread_pct is not None
    spec = skip_reason_to_discord_spec(sr, title="T")
    assert spec["title"] == "T"
    assert any(f"code={SkipCodes.REGIME_FILTER}" in x for x in spec["lines"])
