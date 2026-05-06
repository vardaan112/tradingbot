"""Shared pytest fixtures and stub settings for offline tests.

Tests must NEVER hit the Alpaca API. Any test that needs `Settings` should
use `make_settings()` which provides validated defaults with bogus credentials.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
        "BOT_CAPITAL_BASE_USD": 0.0,
        "RUN_LIVE_CANARY_ON_STARTUP": False,
        "CANARY_SYMBOL": "XLF",
        "CANARY_NOTIONAL_USD": 10.0,
        "CANARY_TIMEOUT_SECONDS": 60.0,
        "CANARY_PERSIST_FILENAME": "canary_state.json",
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
