"""Global compile-time constants shared across the trading bot.

Anything in this module is intentionally read-only and free of I/O. It is safe
to import from any layer.
"""

from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

# -----------------------------------------------------------------------------
# Timezones
# -----------------------------------------------------------------------------
NEW_YORK_TZ: ZoneInfo = ZoneInfo("America/New_York")
UTC_TZ: ZoneInfo = ZoneInfo("UTC")

# -----------------------------------------------------------------------------
# Regulatory transition (FINRA Rule 4210 intraday margin amendments)
# -----------------------------------------------------------------------------
# Effective date Alpaca and FINRA target for the new intraday margin logic and
# the deprecation of legacy PDT-oriented fields.
REG_RULE_4210_EFFECTIVE_DATE: date = date(2026, 6, 4)

REGULATORY_MODE_AUTO = "auto"
REGULATORY_MODE_PDT = "pdt"
REGULATORY_MODE_INTRADAY_MARGIN = "intraday_margin"

VALID_REGULATORY_MODES = frozenset(
    {
        REGULATORY_MODE_AUTO,
        REGULATORY_MODE_PDT,
        REGULATORY_MODE_INTRADAY_MARGIN,
    }
)

# -----------------------------------------------------------------------------
# Alpaca environment names
# -----------------------------------------------------------------------------
ALPACA_ENV_PAPER = "paper"
ALPACA_ENV_LIVE = "live"
VALID_ALPACA_ENVS = frozenset({ALPACA_ENV_PAPER, ALPACA_ENV_LIVE})

# -----------------------------------------------------------------------------
# Market data feeds
# -----------------------------------------------------------------------------
FEED_SIP = "sip"
FEED_IEX = "iex"
FEED_AUTO = "auto"
VALID_FEEDS = frozenset({FEED_SIP, FEED_IEX, FEED_AUTO})

# -----------------------------------------------------------------------------
# Confirmation phrase required to actually submit live orders
# -----------------------------------------------------------------------------
LIVE_TRADING_CONFIRMATION_PHRASE = "yes_i_understand"

# -----------------------------------------------------------------------------
# Session windows (regular session; configurable in Settings via env if needed)
# -----------------------------------------------------------------------------
DEFAULT_NO_NEW_ENTRY_OPEN_MINUTES = 5  # no entries first 5 minutes after open
DEFAULT_NO_NEW_ENTRY_CLOSE_MINUTES = 15  # no new positions in last 15 min

# -----------------------------------------------------------------------------
# Tick sizes (NMS Reg-NMS sub-penny rule; equities >= $1 trade in $0.01 ticks).
# -----------------------------------------------------------------------------
TICK_SIZE_ABOVE_ONE = 0.01
TICK_SIZE_BELOW_ONE = 0.0001

# -----------------------------------------------------------------------------
# Default reconnection / loop bounds
# -----------------------------------------------------------------------------
DEFAULT_WS_RECONNECT_MAX_DELAY = 60.0
DEFAULT_REST_TIMEOUT_SECONDS = 15.0

# Websocket staleness alerting (local / laptop supervision)
STREAM_STALE_SECONDS_DEFAULT: float = 30.0
STREAM_NOTIFICATION_COOLDOWN_SECONDS_DEFAULT: float = 300.0

# Laptop resource checks
LOW_BATTERY_THRESHOLD_PCT_DEFAULT: int = 20

# -----------------------------------------------------------------------------
# Logger names
# -----------------------------------------------------------------------------
LOGGER_APP = "tradingbot.app"
LOGGER_ORDERS = "tradingbot.orders"
LOGGER_RISK = "tradingbot.risk"
LOGGER_HEARTBEAT = "tradingbot.heartbeat"
LOGGER_ERRORS = "tradingbot.errors"
LOGGER_STRATEGY = "tradingbot.strategy"
LOGGER_DATA = "tradingbot.data"
LOGGER_STREAM = "tradingbot.stream"
