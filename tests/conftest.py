"""Shared pytest fixtures and stub settings for offline tests.

Tests must NEVER hit the Alpaca API. Any test that needs `Settings` should
use `make_settings()` which provides validated defaults with bogus credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _reset_named_loggers_after_test():
    yield
    # Production `configure_logging()` sets propagate=False on bot loggers and
    # leaves `_configured=True`, which breaks unittest/pytest handlers on logger
    # names like LOGGER_STRATEGY for tests that follow an Orchestrator test.
    from config.logging_config import reset_logging_for_tests

    reset_logging_for_tests()


def make_settings(**overrides):
    """Construct a `Settings` with safe defaults for offline tests."""
    from config.settings import Settings

    defaults = {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_API_SECRET": "test_secret",
        "ALPACA_ENV": "paper",
        "ALPACA_FEED": "iex",
        "LIVE_TRADING_ENABLED": False,
        "DRY_RUN": True,
        "CONFIRM_LIVE_TRADING": "",
        "LOG_LEVEL": "WARNING",
        "LOG_DIR": str(ROOT / "logs"),
        "STATE_DIR": str(ROOT / "runtime"),
        "SYMBOLS": "AAPL,MSFT",
        "BAR_TIMEFRAME": "5Min",
        "RSI_LENGTH": 14,
        "RSI_OVERSOLD": 30.0,
        "RSI_EXIT": 50.0,
        "ATR_LENGTH": 14,
        "ATR_STOP_MULTIPLIER": 2.0,
        "ATR_PROFIT_MULTIPLIER": 3.0,
        "MAX_RISK_PER_TRADE_PCT": 0.01,
        "MAX_EQUITY_USAGE_USD": 500.0,
        "MAX_GROSS_EXPOSURE_PCT": 0.5,
        "MAX_OPEN_POSITIONS": 1,
        "SPREAD_FILTER_PCT": 0.0005,
        "QUOTE_STALENESS_SECONDS": 5.0,
        "ORDER_TIMEOUT_SECONDS": 30.0,
        "RETRY_MAX_ATTEMPTS": 2,
        "RETRY_BASE_DELAY_SECONDS": 0.01,
        "RETRY_MAX_DELAY_SECONDS": 0.05,
        "REGULATORY_MODE": "auto",
        "POST_RULE4210_SCALING_ENABLED": False,
        "MIN_PRICE": 5.0,
        "MIN_AVG_DOLLAR_VOLUME": 0.0,
        "ENABLE_FRACTIONAL": False,
        "KILL_SWITCH_DRAWDOWN_PCT": 0.05,
        "EMERGENCY_AGGRESSIVENESS_PCT": 0.0015,
        "HEARTBEAT_INTERVAL_SECONDS": 60.0,
        "ORCHESTRATOR_TICK_SECONDS": 15.0,
        "MAX_HOLD_BARS": 24,
        # Regime + trailing (Phase 2)
        "ADX_LENGTH": 14,
        "ADX_RANGE_MAX": 25.0,
        "SMA_FILTER_LENGTH": 200,
        "SMA_SLOPE_LOOKBACK_BARS": 5,
        "TRAIL_TRIGGER_PCT": 0.01,
        "TRAIL_LOCKED_PROFIT_PCT": 0.005,
        "TRAIL_ATR_MULTIPLIER": 1.5,
        "HIGH_CONVICTION_RISK_MULTIPLIER": 1.5,
        "LOW_CONVICTION_RISK_MULTIPLIER": 0.5,
        # Phase 3 — scanner + global risk
        "DYNAMIC_UNIVERSE_ENABLED": False,
        "SCANNER_TOP_N": 20,
        "SCANNER_VOLUME_LOOKBACK_DAYS": 30,
        "SCANNER_MAX_CANDIDATES": 500,
        "SCANNER_MIN_HISTORY_DAYS": 20,
        "SCANNER_REFRESH_HOUR_ET": 9,
        "SCANNER_REFRESH_MINUTE_ET": 31,
        "CORRELATION_BREAKER_ENABLED": True,
        "CORRELATION_LEADER_SYMBOL": "SPY",
        "CORRELATION_FOLLOWER_SYMBOLS": "QQQ",
        "CORRELATION_BREAKER_THRESHOLD": 0.85,
        "CORRELATION_LOOKBACK_CALENDAR_DAYS": 30,
        "BLACK_SWAN_ENABLED": True,
        "BLACK_SWAN_SYMBOL": "SPY",
        "BLACK_SWAN_DROP_PCT": 0.03,
        "BLACK_SWAN_WINDOW_MINUTES": 15,
        "HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS": 0.0,
        "BOT_CAPITAL_BASE_USD": 0.0,
        "RUN_LIVE_CANARY_ON_STARTUP": False,
        "CANARY_SYMBOL": "XLF",
        "CANARY_NOTIONAL_USD": 10.0,
        "CANARY_TIMEOUT_SECONDS": 60.0,
        "CANARY_PERSIST_FILENAME": "canary_state.json",
        "DATABASE_PATH": str(ROOT / "runtime" / "test_default.sqlite3"),
        "REPORTS_DIR": str(ROOT / "reports"),
        "ENABLE_AUTOTUNE": False,
        "AUTOTUNE_MAX_DRAWDOWN_ABS": 0.45,
        "ENABLE_ML_FILTER": False,
        "ENABLE_DISCORD_BOT": False,
        "REQUIRE_DISCORD_ON_STARTUP": False,
        "ENABLE_KELLY_SIZING": False,
        "DISCORD_BOT_TOKEN": "",
        "DISCORD_CHANNEL_ID": "",
        "DISCORD_ALLOWED_USER_IDS": "",
        "DYNAMIC_PARAMS_PATH": str(ROOT / "src" / "config" / "dynamic_params.json"),
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def make_settings_factory():
    return make_settings


@pytest.fixture
def state_store(tmp_path):
    from core.state_store import StateStore
    return StateStore(tmp_path)
