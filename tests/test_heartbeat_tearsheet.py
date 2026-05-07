"""Heartbeat service should embed tearsheet metrics best-effort."""

from __future__ import annotations

import logging
from datetime import date
from itertools import chain, repeat
from pathlib import Path

import pytest

from config.constants import LOGGER_HEARTBEAT
from config.settings import Settings
from core.state_store import StateStore
from risk.compliance import ComplianceAdapter
from risk.killswitch import KillSwitch
from services.heartbeat import HeartbeatService


@pytest.fixture(autouse=True)
def _silence_heartbeat_chaos_hooks(monkeypatch):
    """Heartbeat runs WS/battery hooks; these tests only assert heartbeat/tearsheet text."""

    monkeypatch.setattr(
        "services.heartbeat.evaluate_stream_websocket_notifications",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "services.heartbeat.log_local_resource_check",
        lambda *args, **kwargs: None,
    )


class _MarketClockStub:
    def get_session(self):
        class S:
            is_open = False

        return S()


class _QuoteCacheStub:
    feed = "iex"

    def latest_age_seconds(self):
        return None


class _HealthStub:
    trading_ok = True
    market_ok = True


def _minimal_settings(**kwargs) -> Settings:
    base = dict(
        ALPACA_API_KEY="k",
        ALPACA_API_SECRET="s",
        ALPACA_ENV="paper",
        ALPACA_FEED="iex",
        LIVE_TRADING_ENABLED=False,
        DRY_RUN=True,
        CONFIRM_LIVE_TRADING="",
        LOG_LEVEL="INFO",
        LOG_DIR=Path("/tmp/unused_logs"),
        STATE_DIR=Path("/tmp/unused_state"),
        HEARTBEAT_INTERVAL_SECONDS=60.0,
        ORCHESTRATOR_TICK_SECONDS=15.0,
        SYMBOLS="SPY",
        BAR_TIMEFRAME="5Min",
        RSI_LENGTH=14,
        RSI_OVERSOLD=30.0,
        RSI_EXIT=50.0,
        ATR_LENGTH=14,
        ATR_STOP_MULTIPLIER=2.0,
        ATR_PROFIT_MULTIPLIER=3.0,
        MAX_HOLD_BARS=24,
        ADX_LENGTH=14,
        ADX_RANGE_MAX=25.0,
        SMA_FILTER_LENGTH=200,
        SMA_SLOPE_LOOKBACK_BARS=5,
        TRAIL_TRIGGER_PCT=0.01,
        TRAIL_LOCKED_PROFIT_PCT=0.005,
        TRAIL_ATR_MULTIPLIER=1.5,
        HIGH_CONVICTION_RISK_MULTIPLIER=1.5,
        LOW_CONVICTION_RISK_MULTIPLIER=0.5,
        DYNAMIC_UNIVERSE_ENABLED=False,
        SCANNER_TOP_N=20,
        SCANNER_VOLUME_LOOKBACK_DAYS=30,
        SCANNER_MAX_CANDIDATES=500,
        SCANNER_MIN_HISTORY_DAYS=20,
        SCANNER_REFRESH_HOUR_ET=9,
        SCANNER_REFRESH_MINUTE_ET=31,
        CORRELATION_BREAKER_ENABLED=True,
        CORRELATION_LEADER_SYMBOL="SPY",
        CORRELATION_FOLLOWER_SYMBOLS="QQQ",
        CORRELATION_BREAKER_THRESHOLD=0.85,
        CORRELATION_LOOKBACK_CALENDAR_DAYS=30,
        BLACK_SWAN_ENABLED=True,
        BLACK_SWAN_SYMBOL="SPY",
        BLACK_SWAN_DROP_PCT=0.03,
        BLACK_SWAN_WINDOW_MINUTES=15,
        HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS=0.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        MAX_EQUITY_USAGE_USD=500.0,
        MAX_GROSS_EXPOSURE_PCT=0.5,
        MAX_OPEN_POSITIONS=1,
        KILL_SWITCH_DRAWDOWN_PCT=0.05,
        SPREAD_FILTER_PCT=0.0005,
        QUOTE_STALENESS_SECONDS=5.0,
        ORDER_TIMEOUT_SECONDS=30.0,
        EMERGENCY_AGGRESSIVENESS_PCT=0.0015,
        RETRY_MAX_ATTEMPTS=2,
        RETRY_BASE_DELAY_SECONDS=0.01,
        RETRY_MAX_DELAY_SECONDS=0.05,
        REGULATORY_MODE="auto",
        POST_RULE4210_SCALING_ENABLED=False,
        MIN_PRICE=5.0,
        MIN_AVG_DOLLAR_VOLUME=0.0,
        BOT_CAPITAL_BASE_USD=0.0,
        ENABLE_FRACTIONAL=False,
        RUN_LIVE_CANARY_ON_STARTUP=False,
        CANARY_SYMBOL="XLF",
        CANARY_NOTIONAL_USD=10.0,
        CANARY_TIMEOUT_SECONDS=60.0,
        CANARY_PERSIST_FILENAME="canary_state.json",
    )
    base.update(kwargs)
    return Settings(_env_file=None, **base)


@pytest.mark.asyncio
async def test_heartbeat_includes_tearsheet_fields(monkeypatch, tmp_path: Path, caplog):
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: date(2032, 2, 2))
    orders_log = tmp_path / "orders.log"
    orders_log.write_text(
        (
            "2032-02-02T14:05:01+00:00 | INFO | tradingbot.orders | mode=paper | reg=auto | "
            "symbol=SPY | strategy=s | coid=o1 | Trade update coid=o1 symbol=SPY side=buy "
            "status=filled filled=1.0000 avg=100.0000\n"
            "2032-02-02T14:05:03+00:00 | INFO | tradingbot.orders | mode=paper | reg=auto | "
            "symbol=SPY | strategy=s | coid=o2 | Trade update coid=o2 symbol=SPY side=sell "
            "status=filled filled=1.0000 avg=102.5000\n"
        ),
        encoding="utf-8",
    )

    settings = _minimal_settings()
    store = StateStore(tmp_path / "runtime")
    hb = HeartbeatService(
        settings,
        clock=_MarketClockStub(),
        quote_cache=_QuoteCacheStub(),
        stream_health=_HealthStub(),
        kill_switch=KillSwitch(store, drawdown_pct=0.05),
        compliance=ComplianceAdapter(settings),
        snapshot_provider=lambda: {
            "equity": 1.0,
            "buying_power": 2.0,
            "open_positions": 0,
            "open_orders": 0,
        },
        tearsheet_orders_path=orders_log,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_HEARTBEAT):
        await hb._emit_once()

    messages = "\n".join(r.getMessage() for r in caplog.records if LOGGER_HEARTBEAT in r.name)
    assert "tearsheet_closed=1" in messages
    assert "tearsheet_net=" in messages
    assert "tearsheet_mdd=" in messages
    assert "tearsheet_win_rate_pct=" in messages


@pytest.mark.asyncio
async def test_heartbeat_tearsheet_degrades_when_parse_fails(monkeypatch, tmp_path: Path, caplog):
    settings = _minimal_settings()
    store = StateStore(tmp_path / "runtime")
    bad_path = tmp_path / "bad.log"
    bad_path.write_text("not-a-valid-trade-line\n", encoding="utf-8")

    def _boom(_path: Path):
        raise RuntimeError("forced")

    monkeypatch.setattr("services.heartbeat.get_tearsheet_summary", _boom)

    hb = HeartbeatService(
        settings,
        clock=_MarketClockStub(),
        quote_cache=_QuoteCacheStub(),
        stream_health=_HealthStub(),
        kill_switch=KillSwitch(store, drawdown_pct=0.05),
        compliance=ComplianceAdapter(settings),
        snapshot_provider=lambda: {
            "equity": 1.0,
            "buying_power": 2.0,
            "open_positions": 0,
            "open_orders": 0,
        },
        tearsheet_orders_path=bad_path,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_HEARTBEAT):
        await hb._emit_once()

    messages = "\n".join(r.getMessage() for r in caplog.records if LOGGER_HEARTBEAT in r.name)
    assert "heartbeat session=" in messages
    assert "tearsheet_closed=n_a" in messages


@pytest.mark.asyncio
async def test_heartbeat_handles_missing_orders_log(tmp_path: Path, caplog):
    settings = _minimal_settings()
    store = StateStore(tmp_path / "runtime")
    missing = tmp_path / "nope_orders.log"

    hb = HeartbeatService(
        settings,
        clock=_MarketClockStub(),
        quote_cache=_QuoteCacheStub(),
        stream_health=_HealthStub(),
        kill_switch=KillSwitch(store, drawdown_pct=0.05),
        compliance=ComplianceAdapter(settings),
        snapshot_provider=lambda: {
            "equity": 1.0,
            "buying_power": 1.0,
            "open_positions": 0,
            "open_orders": 0,
        },
        tearsheet_orders_path=missing,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_HEARTBEAT):
        await hb._emit_once()

    messages = "\n".join(r.getMessage() for r in caplog.records if LOGGER_HEARTBEAT in r.name)
    assert "tearsheet_closed=n_a" in messages


@pytest.mark.asyncio
async def test_heartbeat_without_orders_path_has_no_tearsheet_suffix(caplog, tmp_path: Path):
    settings = _minimal_settings()
    store = StateStore(tmp_path / "hb_state")
    hb = HeartbeatService(
        settings,
        clock=_MarketClockStub(),
        quote_cache=_QuoteCacheStub(),
        stream_health=_HealthStub(),
        kill_switch=KillSwitch(store, drawdown_pct=0.05),
        compliance=ComplianceAdapter(settings),
        snapshot_provider=lambda: {"equity": 1.0, "buying_power": 1.0, "open_positions": 0, "open_orders": 0},
        tearsheet_orders_path=None,
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_HEARTBEAT):
        await hb._emit_once()

    messages = "\n".join(r.getMessage() for r in caplog.records if LOGGER_HEARTBEAT in r.name)
    assert "heartbeat session=" in messages
    assert "tearsheet_closed" not in messages


@pytest.mark.asyncio
async def test_heartbeat_emits_markdown_table_on_slow_cadence(monkeypatch, tmp_path: Path, caplog):
    monkeypatch.setattr("utils.tearsheet.today_eastern", lambda: date(2035, 1, 1))
    orders_log = tmp_path / "orders.log"
    orders_log.write_text(
        (
            "2035-01-01T14:05:01+00:00 | INFO | tradingbot.orders | mode=paper | reg=auto | "
            "symbol=SPY | strategy=s | coid=o1 | Trade update coid=o1 symbol=SPY side=buy "
            "status=filled filled=2.0000 avg=10.0000\n"
            "2035-01-01T14:05:03+00:00 | INFO | tradingbot.orders | mode=paper | reg=auto | "
            "symbol=SPY | strategy=s | coid=o2 | Trade update coid=o2 symbol=SPY side=sell "
            "status=filled filled=2.0000 avg=11.0000\n"
        ),
        encoding="utf-8",
    )

    settings = _minimal_settings(HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS=10.0)

    mono_vals = chain([500.0, 506.0, 518.0], repeat(518.0))

    def fake_mono() -> float:
        return float(next(mono_vals))

    store = StateStore(tmp_path / "hb_md")
    hb = HeartbeatService(
        settings,
        clock=_MarketClockStub(),
        quote_cache=_QuoteCacheStub(),
        stream_health=_HealthStub(),
        kill_switch=KillSwitch(store, drawdown_pct=0.05),
        compliance=ComplianceAdapter(settings),
        snapshot_provider=lambda: {
            "equity": 1.0,
            "buying_power": 1.0,
            "open_positions": 0,
            "open_orders": 0,
        },
        tearsheet_orders_path=orders_log,
    )

    monkeypatch.setattr("services.heartbeat.time.monotonic", fake_mono)

    with caplog.at_level(logging.INFO, logger=LOGGER_HEARTBEAT):
        await hb._emit_once()
        await hb._emit_once()
        await hb._emit_once()

    md_records = [
        r
        for r in caplog.records
        if LOGGER_HEARTBEAT in r.name and "| Metric | Value |" in r.getMessage()
    ]
    assert len(md_records) == 2
