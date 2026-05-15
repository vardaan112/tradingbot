"""Strongly validated settings loaded from environment variables.

Every runtime configuration value flows through `Settings`. Direct use of
`os.environ` elsewhere in the codebase is intentionally avoided so that all
configuration validation happens in exactly one place.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    ALPACA_ENV_LIVE,
    ALPACA_ENV_PAPER,
    FEED_AUTO,
    FEED_IEX,
    FEED_SIP,
    LIVE_TRADING_CONFIRMATION_PHRASE,
    LOW_BATTERY_THRESHOLD_PCT_DEFAULT,
    REGULATORY_MODE_AUTO,
    REGULATORY_MODE_INTRADAY_MARGIN,
    REGULATORY_MODE_PDT,
    STREAM_NOTIFICATION_COOLDOWN_SECONDS_DEFAULT,
    STREAM_STALE_SECONDS_DEFAULT,
    VALID_ALPACA_ENVS,
    VALID_FEEDS,
    VALID_REGULATORY_MODES,
)


class Settings(BaseSettings):
    """Validated runtime settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- Credentials and Alpaca env ------------------------------------------------
    ALPACA_API_KEY: str = Field(..., min_length=1, repr=False)
    ALPACA_API_SECRET: str = Field(..., min_length=1, repr=False)
    ALPACA_ENV: Literal["paper", "live"] = ALPACA_ENV_PAPER
    ALPACA_FEED: Literal["sip", "iex", "auto"] = FEED_AUTO

    # ---- Master switches -----------------------------------------------------------
    LIVE_TRADING_ENABLED: bool = False
    DRY_RUN: bool = True
    CONFIRM_LIVE_TRADING: str = ""
    # Research/correctness guardrails. BLOCK_LIVE_DEPLOYMENT is advisory and
    # intentionally does not override the existing DRY_RUN/LIVE_TRADING gates.
    BLOCK_LIVE_DEPLOYMENT: bool = True
    ENABLE_RESEARCH_MODE: bool = True
    BACKTEST_PREVENT_SAME_BAR_FILLS: bool = True
    RESEARCH_REQUIRE_OOS_PASS: bool = True

    # ---- Logging and storage -------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")
    STATE_DIR: Path = Path("./runtime")

    # ---- Heartbeat and orchestration -----------------------------------------------
    HEARTBEAT_INTERVAL_SECONDS: float = Field(60.0, ge=5.0, le=600.0)
    ORCHESTRATOR_TICK_SECONDS: float = Field(15.0, ge=1.0, le=300.0)

    # ---- Universe and bars ---------------------------------------------------------
    # Default basket: small static set of liquid ETFs covering broad equities,
    # tech, small caps, financials, and emerging markets. Intentionally static
    # for the first live rollout - no dynamic screener.
    SYMBOLS: str = "SPY,QQQ,IWM,XLF,EEM"
    BAR_TIMEFRAME: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "5Min"

    # ---- Strategy parameters -------------------------------------------------------
    RSI_LENGTH: int = Field(14, ge=2, le=200)
    RSI_OVERSOLD: float = Field(30.0, ge=1.0, le=50.0)
    RSI_EXIT: float = Field(50.0, ge=1.0, le=99.0)
    DEFAULT_RSI_ENTRY: float = Field(30.0, ge=1.0, le=50.0)
    HIGH_VOL_RSI_ENTRY: float = Field(35.0, ge=1.0, le=50.0)
    HIGH_VOL_ATR_PCT_THRESHOLD: float = Field(0.05, gt=0.0, le=1.0)
    DYNAMIC_RSI_ENABLED: bool = False
    DYNAMIC_RSI_BASE: float = Field(30.0, ge=1.0, le=50.0)
    DYNAMIC_RSI_K: float = Field(2.0, ge=0.0, le=25.0)
    DYNAMIC_RSI_MIN: float = Field(20.0, ge=1.0, le=50.0)
    DYNAMIC_RSI_MAX: float = Field(35.0, ge=1.0, le=50.0)
    ATR_LOOKBACK: int = Field(50, ge=2, le=500)
    BOLLINGER_ENABLED: bool = False
    ENABLE_BOLLINGER_CONFLUENCE: bool = False
    BOLLINGER_LENGTH: int = Field(20, ge=2, le=500)
    BOLLINGER_STD: float = Field(2.0, gt=0.0, le=10.0)
    BOLLINGER_MIN_WIDTH_PCT: float = Field(0.0, ge=0.0, le=1.0)
    # Optional alias requested by the hybrid VWAP/Bollinger strategy.
    BOLLINGER_BW_MIN: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    BOLLINGER_REQUIRE_TOUCH: bool = False
    VWAP_STRATEGY_ENABLED: bool = False
    ENABLE_VWAP_CONFLUENCE: bool = False
    ENABLE_FILTER_SCORE_MODE: bool = False
    VWAP_LENGTH: int = Field(20, ge=2, le=500)
    VWAP_STD: float = Field(2.0, gt=0.0, le=10.0)
    VWAP_Z_THRESHOLD: float = Field(2.0, gt=0.0, le=10.0)
    DYNAMIC_RSI_SHORT_ATR: int = Field(5, ge=2, le=200)
    DYNAMIC_RSI_LONG_ATR: int = Field(20, ge=2, le=500)
    ADX_LOW: float = Field(20.0, gt=0.0, le=100.0)
    ADX_HIGH: float = Field(40.0, gt=0.0, le=100.0)
    TIME_OF_DAY_FILTER_ENABLED: bool = True
    TIME_OF_DAY_TRADE_START: str = "10:00"
    TIME_OF_DAY_TRADE_END: str = "15:00"
    AGGRESSIVE_MODE: bool = False
    AGGRESSIVE_RSI_BYPASS_THRESHOLD: float = Field(20.0, ge=1.0, le=50.0)
    SCALE_IN_ENABLED: bool = False
    SCALE_IN_UNDERWATER_PCT: float = Field(-0.03, gt=-1.0, lt=0.0)
    SCALE_IN_RSI_THRESHOLD: float = Field(25.0, ge=1.0, le=50.0)
    SCALE_IN_ADD_QTY: float = Field(1.0, gt=0.0, le=10_000.0)
    MAX_BULLETS_PER_SYMBOL: int = Field(2, ge=1, le=25)
    ATR_LENGTH: int = Field(14, ge=2, le=200)
    ATR_STOP_MULTIPLIER: float = Field(2.0, gt=0.0, le=20.0)
    ATR_PROFIT_MULTIPLIER: float = Field(3.0, gt=0.0, le=50.0)
    MAX_HOLD_BARS: int = Field(24, ge=1, le=10_000)

    # ---- Phase 2: multi-strategy registry / engine (evaluation only) ------------
    ACTIVE_STRATEGIES: str = "rsi_mean_reversion"
    STRATEGY_RUN_MODE: Literal["single", "independent", "ensemble", "both"] = "single"
    STRATEGY_WEIGHTS_JSON: str = '{"rsi_mean_reversion":1.0}'
    ENSEMBLE_ENABLED: bool = False
    ENSEMBLE_ENTER_THRESHOLD: float = Field(0.55, ge=0.0, le=1.0)
    ENSEMBLE_EXIT_THRESHOLD: float = Field(0.50, ge=0.0, le=1.0)
    ENSEMBLE_MIN_AGREEING_STRATEGIES: int = Field(1, ge=1, le=50)
    ENSEMBLE_EXIT_POLICY: Literal["weighted", "any", "risk_first"] = "risk_first"
    ENSEMBLE_WEIGHT_MODE: Literal["static", "performance"] = "static"
    ENSEMBLE_PERFORMANCE_SOURCE: Literal["replay", "shadow", "paper", "live"] = "shadow"
    ENSEMBLE_PERFORMANCE_LOOKBACK_DAYS: int = Field(30, ge=1, le=730)
    ENSEMBLE_MIN_TRADES_FOR_WEIGHT: int = Field(10, ge=1, le=10_000)
    ENSEMBLE_WEIGHT_SMOOTHING_ALPHA: float = Field(0.2, ge=0.0, le=1.0)
    ALLOW_LIVE_PERFORMANCE_WEIGHTS: bool = False
    ENSEMBLE_MAX_WEIGHT: float = Field(0.60, gt=0.0, le=1.0)
    ENSEMBLE_MIN_WEIGHT: float = Field(0.05, gt=0.0, le=1.0)

    # ---- Phase 7: live shadow portfolios (no broker orders) -----------------------
    SHADOW_TRADING_ENABLED: bool = False
    SHADOW_INITIAL_EQUITY: float = Field(10_000.0, gt=0.0)
    SHADOW_RECORD_INTERVAL_SECONDS: float = Field(60.0, ge=1.0, le=86_400.0)
    SHADOW_FILL_MODEL: Literal["midpoint", "conservative_quote"] = "conservative_quote"

    # ---- Phase 3: additional strategies (disabled by default) ----------------------
    MOMENTUM_ENABLED: bool = True
    MOMENTUM_FAST_SMA: int = Field(20, ge=2, le=500)
    MOMENTUM_SLOW_SMA: int = Field(50, ge=3, le=500)
    MOMENTUM_LOOKBACK_BARS: int = Field(20, ge=2, le=500)
    MOMENTUM_MIN_RETURN_PCT: float = Field(0.02, ge=0.0, le=1.0)
    MOMENTUM_ADX_MIN: float = Field(20.0, ge=0.0, le=100.0)
    MOMENTUM_REQUIRE_ADX: bool = True
    MOMENTUM_EXIT_SMA: int = Field(20, ge=2, le=500)
    MOMENTUM_ATR_STOP_MULT: float = Field(2.0, gt=0.0, le=20.0)
    MOMENTUM_MAX_HOLD_BARS: int = Field(24, ge=1, le=10_000)
    MOMENTUM_VOLUME_MA_BARS: int = Field(20, ge=2, le=500)
    MOMENTUM_VOLUME_FACTOR: float = Field(0.9, ge=0.0, le=5.0)

    BREAKOUT_ENABLED: bool = True
    BREAKOUT_LOOKBACK_BARS: int = Field(20, ge=2, le=500)
    BREAKOUT_VOLUME_MULTIPLIER: float = Field(1.5, ge=1.0, le=10.0)
    BREAKOUT_ATR_BUFFER_MULTIPLIER: float = Field(0.1, ge=0.0, le=5.0)
    BREAKOUT_MAX_HOLD_BARS: int = Field(24, ge=1, le=10_000)
    BREAKOUT_VOLUME_MA_BARS: int = Field(20, ge=2, le=500)
    BREAKOUT_ATR_STOP_MULT: float = Field(2.0, gt=0.0, le=20.0)
    BREAKOUT_ATR_TARGET_MULT: float = Field(3.0, gt=0.0, le=20.0)
    BREAKOUT_TRAIL_ATR_MULT: float = Field(2.0, gt=0.0, le=20.0)

    VWAP_PULLBACK_ENABLED: bool = True
    VWAP_PULLBACK_LENGTH: int = Field(20, ge=2, le=500)
    VWAP_PULLBACK_Z_THRESHOLD: float = Field(2.0, gt=0.0, le=10.0)
    VWAP_PULLBACK_MAX_DISTANCE_PCT: float = Field(0.003, ge=0.0, le=0.2)
    VWAP_PULLBACK_MIN_TREND_SLOPE: float = Field(0.0, ge=-10.0, le=10.0)
    VWAP_PULLBACK_RSI_MIN: float = Field(35.0, ge=1.0, le=99.0)
    VWAP_PULLBACK_RSI_MAX: float = Field(60.0, ge=1.0, le=99.0)
    VWAP_PULLBACK_ADX_MIN: float = Field(12.0, ge=0.0, le=100.0)
    VWAP_PULLBACK_ADX_MAX: float = Field(45.0, ge=0.0, le=100.0)
    VWAP_PULLBACK_TREND_FAST_SMA: int = Field(10, ge=2, le=200)
    VWAP_PULLBACK_TREND_SLOW_SMA: int = Field(30, ge=3, le=500)
    VWAP_PULLBACK_MAX_ZSCORE: float = Field(-0.2, ge=-5.0, le=5.0)
    VWAP_PULLBACK_EXIT_Z_MIN: float = Field(-0.25, ge=-5.0, le=5.0)
    VWAP_PULLBACK_ATR_STOP_MULT: float = Field(2.0, gt=0.0, le=20.0)
    VWAP_PULLBACK_ATR_TARGET_MULT: float = Field(2.5, gt=0.0, le=20.0)
    VWAP_PULLBACK_MAX_HOLD_BARS: int = Field(36, ge=1, le=10_000)

    ETF_ROTATION_ENABLED: bool = True
    ETF_ROTATION_SYMBOLS: str = "SPY,QQQ,IWM,XLK,XLF,XLE,TLT,GLD"
    ETF_ROTATION_LOOKBACK_BARS: int = Field(63, ge=5, le=500)
    ETF_ROTATION_TOP_N: int = Field(1, ge=1, le=20)
    ETF_ROTATION_MIN_SCORE: float = Field(0.0, ge=-10.0, le=10.0)
    ETF_ROTATION_TREND_SMA: int = Field(20, ge=2, le=200)
    ETF_ROTATION_REBALANCE_BARS: int = Field(12, ge=1, le=500)

    PAIRS_ENABLED: bool = True
    PAIRS_CONFIG_JSON: str = '{"QQQ":"SPY"}'
    PAIRS_LOOKBACK_BARS: int = Field(100, ge=10, le=2000)
    PAIRS_ENTRY_Z: float = Field(-2.0, ge=-10.0, le=0.0)
    PAIRS_EXIT_Z: float = Field(-0.5, ge=-10.0, le=10.0)

    # ---- Regime & synthetic trailing-profit (Phase Two) ----------------------------
    ADX_LENGTH: int = Field(14, ge=2, le=500)
    ADX_RANGE_MAX: float = Field(25.0, gt=0.0, le=100.0)
    # Optional alias. When set, this overrides ADX_RANGE_MAX.
    ADX_THRESHOLD: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    SMA_FILTER_LENGTH: int = Field(200, ge=10, le=5000)
    SMA_SLOPE_LOOKBACK_BARS: int = Field(5, ge=1, le=500)

    TRAIL_TRIGGER_PCT: float = Field(0.01, gt=0.0, le=0.25)
    TRAIL_LOCKED_PROFIT_PCT: float = Field(0.005, gt=0.0, le=0.25)
    TRAIL_ATR_MULTIPLIER: float = Field(1.5, gt=0.0, le=50.0)

    HIGH_CONVICTION_RISK_MULTIPLIER: float = Field(1.5, gt=0.05, le=10.0)
    LOW_CONVICTION_RISK_MULTIPLIER: float = Field(0.5, gt=0.05, le=10.0)

    # ---- Risk ----------------------------------------------------------------------
    MAX_RISK_PER_TRADE_PCT: float = Field(0.01, gt=0.0, le=0.05)
    MAX_EQUITY_USAGE_USD: float = Field(50.0, gt=0.0)
    # Optional alias: when set (>0), this overrides MAX_EQUITY_USAGE_USD.
    MAX_DOLLARS_PER_TRADE: float = Field(0.0, ge=0.0)
    MAX_GROSS_EXPOSURE_PCT: float = Field(0.5, gt=0.0, le=2.0)
    MAX_OPEN_POSITIONS: int = Field(1, ge=1, le=100)
    MAX_OPEN_POSITIONS_PER_SECTOR: int = Field(2, ge=1, le=100)
    KILL_SWITCH_DRAWDOWN_PCT: float = Field(0.05, gt=0.0, le=0.5)
    # JSON object string: {"AAPL":"Technology","MSFT":"Technology",...}
    SECTOR_MAP_JSON: str = (
        '{"AAPL":"Technology","MSFT":"Technology","NVDA":"Technology",'
        '"SHOP":"Technology","ABNB":"Consumer Cyclical","MARA":"Crypto / Digital Assets"}'
    )

    # The dollar capital base the bot is allocated. When >0 this overrides
    # full account equity for risk-budget computation, so the bot only risks
    # a percentage of *its* slice rather than the whole brokerage account.
    # 0 means "not configured": the sizer falls back to
    # min(account.equity, MAX_EQUITY_USAGE_USD).
    BOT_CAPITAL_BASE_USD: float = Field(0.0, ge=0.0)

    # ---- Quote / execution filters -------------------------------------------------
    SPREAD_FILTER_PCT: float = Field(0.0005, gt=0.0, le=0.05)
    # Optional wider cap for quotes tagged ``feed=iex`` (IEX top-of-book is often
    # wider than SIP). When unset (default), all feeds use SPREAD_FILTER_PCT.
    SPREAD_FILTER_PCT_IEX: Optional[float] = Field(default=None)
    SPREAD_FILTER_ELASTIC_ENABLED: bool = True
    # Optional aliases; when unset, existing explicit settings keep behavior.
    ENABLE_ELASTIC_SPREAD: Optional[bool] = None
    ENABLE_QUOTE_FALLBACK: Optional[bool] = None
    ENABLE_REGIME_ADAPTIVE_RSI: Optional[bool] = None
    # Absolute hard cap after all elasticity multipliers are applied.
    SPREAD_FILTER_MAX_PCT: float = Field(0.02, gt=0.0, le=0.25)
    SPREAD_FILTER_IEX_ELASTIC_MULTIPLIER: float = Field(1.75, ge=1.0, le=10.0)
    SPREAD_FILTER_LOW_PRICE_THRESHOLD: float = Field(25.0, gt=0.0, le=5000.0)
    SPREAD_FILTER_LOW_PRICE_MULTIPLIER: float = Field(1.5, ge=1.0, le=10.0)
    SPREAD_FILTER_SPARSE_SIZE_THRESHOLD: float = Field(5.0, ge=0.0, le=100000.0)
    SPREAD_FILTER_SPARSE_QUOTE_MULTIPLIER: float = Field(1.25, ge=1.0, le=10.0)
    SPREAD_FILTER_FRESH_QUOTE_MULTIPLIER: float = Field(1.15, ge=1.0, le=10.0)
    SPREAD_FILTER_FRESH_AGE_FRACTION: float = Field(0.5, gt=0.0, le=1.0)
    # Aliases / additional caps for IEX degraded-mode spread elasticity.
    ELASTIC_SPREAD_ENABLED: Optional[bool] = None
    ELASTIC_SPREAD_HARD_MAX_PCT: Optional[float] = Field(default=None, gt=0.0, le=0.25)
    ELASTIC_SPREAD_TARGET_PROFIT_PCT: float = Field(0.0150, gt=0.0, le=1.0)
    ELASTIC_SPREAD_MAX_COST_FRACTION: float = Field(0.35, gt=0.0, le=1.0)
    ELASTIC_SPREAD_ATR_MULTIPLIER: float = Field(0.20, gt=0.0, le=10.0)
    ELASTIC_SPREAD_BAR_CONFIRM_ENABLED: bool = True
    ELASTIC_SPREAD_REQUIRE_BAR_HEALTH: bool = True
    MIN_SPREAD_THRESHOLD_PERCENT: float = Field(0.0008, ge=0.0, le=0.05)
    QUOTE_STALENESS_SECONDS: float = Field(10.0, gt=0.0, le=300.0)
    QUOTE_FALLBACK_ENABLED: bool = True
    QUOTE_MAX_AGE_SECONDS: Optional[float] = Field(default=None, gt=0.0, le=300.0)
    QUOTE_STRICT_MAX_AGE_SECONDS: float = Field(3.0, gt=0.0, le=300.0)
    QUOTE_STALE_SPREAD_MULTIPLIER: float = Field(1.5, ge=1.0, le=10.0)
    QUOTE_FALLBACK_BAR_TIMEFRAME: Literal["1Min", "5Min", "15Min"] = "1Min"
    QUOTE_FALLBACK_MAX_BAR_AGE_SECONDS: float = Field(90.0, gt=0.0, le=900.0)
    QUOTE_FALLBACK_USE_BAR_MIDPOINT: bool = True
    MAX_STRATEGY_BAR_AGE_SECONDS: float = Field(900.0, ge=30.0, le=86_400.0)
    ORDER_TIMEOUT_SECONDS: float = Field(30.0, gt=0.0, le=3600.0)
    EMERGENCY_AGGRESSIVENESS_PCT: float = Field(0.0015, gt=0.0, le=0.05)

    # ---- Midpoint limit peg (IEX / limit-at-mid + timeout chase) -----------------
    MIDPOINT_PEG_ENABLED: bool = False
    MIDPOINT_PEG_TIMEOUT_SECONDS: float = Field(30.0, gt=0.5, le=600.0)
    MIDPOINT_PEG_MAX_CYCLES: int = Field(5, ge=1, le=50)
    MIDPOINT_PEG_TAG: str = "midpoint_peg"

    # ---- Global QQQ / macro regime (hourly bars) --------------------------------
    QQQ_REGIME_ENABLED: bool = True
    REGIME_QQQ_SYMBOL: str = "QQQ"
    REGIME_ADAPTIVE_RSI_ENABLED: bool = True
    REGIME_ANCHOR_SYMBOL: str = "SPY"
    REGIME_ANCHOR_TIMEFRAME: Literal["1Hour", "1Day"] = "1Hour"
    REGIME_RSI_PERIOD: int = Field(14, ge=2, le=200)
    REGIME_SMA_PERIOD: int = Field(50, ge=5, le=500)
    RSI_OVERSOLD_DEFAULT: float = Field(30.0, ge=1.0, le=50.0)
    RSI_OVERSOLD_BULL: float = Field(40.0, ge=1.0, le=50.0)
    RSI_OVERSOLD_BEAR: float = Field(25.0, ge=1.0, le=50.0)
    RSI_OVERSOLD_NEUTRAL: float = Field(30.0, ge=1.0, le=50.0)
    REGIME_BULL_RSI_MIN: float = Field(60.0, ge=1.0, le=100.0)
    REGIME_PARABOLIC_RSI_MIN: float = Field(70.0, ge=1.0, le=100.0)
    REGIME_BEAR_RSI_MAX: float = Field(45.0, ge=1.0, le=100.0)
    REGIME_ATR_RATIO_THRESHOLD: float = Field(1.2, gt=0.0, le=10.0)
    REGIME_USE_SMA50: bool = True
    REGIME_MAX_EQUITY_REDUCTION: float = Field(0.5, ge=0.0, le=1.0)
    REGIME_BEAR_VOLATILE_BLOCK_ENTRIES: bool = False
    REGIME_FAIL_CLOSED_ON_STARTUP: bool = True
    REGIME_STALE_AFTER_SECONDS: float = Field(3900.0, gt=60.0, le=86_400.0)
    REGIME_UNKNOWN_ACTION: Literal["block_entries", "reduce_size"] = "block_entries"
    REGIME_ATR_LENGTH: int = Field(14, ge=2, le=200)
    REGIME_ATR_MA_LENGTH: int = Field(50, ge=5, le=500)

    # ---- Orchestrator: ATR trail + bar-count time exit (5m bars) -----------------
    EXEC_ATR_TRAIL_ENABLED: bool = True
    ATR_TRAIL_MULTIPLIER: float = Field(2.0, gt=0.0, le=20.0)
    MAX_POSITION_BARS: int = Field(12, ge=1, le=10_000)

    # ---- Strategy liquidity gate (5m volume vs 20-bar average) ------------------
    LIQUIDITY_GATE_ENABLED: bool = True
    LIQUIDITY_THRESHOLD: float = Field(0.5, gt=0.0, le=1.0)

    # ---- Retries -------------------------------------------------------------------
    RETRY_MAX_ATTEMPTS: int = Field(5, ge=1, le=20)
    RETRY_BASE_DELAY_SECONDS: float = Field(0.5, gt=0.0, le=60.0)
    RETRY_MAX_DELAY_SECONDS: float = Field(20.0, gt=0.0, le=600.0)

    # ---- Regulatory ----------------------------------------------------------------
    REGULATORY_MODE: Literal["auto", "pdt", "intraday_margin"] = REGULATORY_MODE_AUTO
    POST_RULE4210_SCALING_ENABLED: bool = False

    # ---- Universe filters ----------------------------------------------------------
    MIN_PRICE: float = Field(5.0, gt=0.0)
    MIN_AVG_DOLLAR_VOLUME: float = Field(20_000_000.0, ge=0.0)

    # ---- Optional features ---------------------------------------------------------
    ENABLE_FRACTIONAL: bool = False
    # Optional alias. When set, this overrides ENABLE_FRACTIONAL.
    ALLOW_FRACTIONAL: Optional[bool] = None
    # Optional alias. When set, this overrides ENABLE_FRACTIONAL.
    FRACTIONAL_TRADING_ENABLED: Optional[bool] = None
    FRACTIONAL_MIN_QTY: float = Field(0.001, gt=0.0, le=1.0)
    MIN_SHARES: float = Field(1.0, gt=0.0, le=100.0)
    MIN_ORDER_DOLLARS: float = Field(10.0, ge=0.0)

    # ---- Live canary check (one-time per day, before main loop) --------------------
    # The canary verifies credentials, order submission, fills, reconciliation,
    # and clean flatten end-to-end with a tiny live trade. It only runs on the
    # live endpoint with LIVE_TRADING_ENABLED=true and DRY_RUN=false.
    RUN_LIVE_CANARY_ON_STARTUP: bool = False
    CANARY_SYMBOL: str = "XLF"
    CANARY_NOTIONAL_USD: float = Field(10.0, gt=0.0)
    CANARY_TIMEOUT_SECONDS: float = Field(60.0, gt=0.0, le=600.0)
    CANARY_PERSIST_FILENAME: str = "canary_state.json"

    # ---- Phase 3: dynamic universe + global risk -----------------------------------
    # When enabled, periodic liquidity scan selects top SCANNER_TOP_N US equities by
    # average daily dollar volume instead of relying on SYMBOLS alone. SYMBOLS remains
    # the failover basket if the scanner errors.
    DYNAMIC_UNIVERSE_ENABLED: bool = False
    SCANNER_TOP_N: int = Field(20, ge=1, le=100)
    SCANNER_VOLUME_LOOKBACK_DAYS: int = Field(30, ge=5, le=252)
    # Cap how many US equities we pull daily bars for (API + CPU budget).
    SCANNER_MAX_CANDIDATES: int = Field(500, ge=50, le=8000)
    # Require at least this many completed daily observations before ranking.
    SCANNER_MIN_HISTORY_DAYS: int = Field(20, ge=5, le=252)
    # Eastern-time clock hour/minute — refresh once per session-day after regular open.
    SCANNER_REFRESH_HOUR_ET: int = Field(9, ge=7, le=12)
    SCANNER_REFRESH_MINUTE_ET: int = Field(31, ge=0, le=59)

    # Correlation gate: skip follower entries when leader is long and SPY/QQQ closes
    # are too correlated over CORRELATION_LOOKBACK_CALENDAR_DAYS.
    CORRELATION_BREAKER_ENABLED: bool = True
    CORRELATION_LEADER_SYMBOL: str = "SPY"
    CORRELATION_FOLLOWER_SYMBOLS: str = "QQQ"
    CORRELATION_BREAKER_THRESHOLD: float = Field(0.85, ge=0.0, le=0.9999)
    CORRELATION_LOOKBACK_CALENDAR_DAYS: int = Field(30, ge=5, le=365)

    # Black swan: SPY rolls down BLACK_SWAN_DROP_PCT within BLACK_SWAN_WINDOW_MINUTES.
    BLACK_SWAN_ENABLED: bool = True
    BLACK_SWAN_SYMBOL: str = "SPY"
    BLACK_SWAN_DROP_PCT: float = Field(0.03, gt=0.0, le=0.5)
    BLACK_SWAN_WINDOW_MINUTES: int = Field(15, ge=1, le=240)

    # Emit a Markdown tearsheet table in heartbeat logs every N seconds (0 disables).
    HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS: float = Field(3600.0, ge=0.0, le=86400.0)

    # ---- Phase 4: sentiment + anti-martingale + SQLite + chase + reporter ---------
    SENTIMENT_ENABLED: bool = False
    SENTIMENT_HEADLINE_LIMIT: int = Field(10, ge=1, le=50)
    SENTIMENT_STRONG_NEGATIVE_THRESHOLD: float = Field(-0.5, ge=-1.0, le=0.0)
    SENTIMENT_CACHE_TTL_SECONDS: float = Field(300.0, ge=30.0, le=86400.0)
    SENTIMENT_STALE_AFTER_SECONDS: float = Field(1800.0, ge=60.0, le=604800.0)
    SENTIMENT_FAIL_CLOSED: bool = False
    SENTIMENT_FAIL_CONSECUTIVE_THRESHOLD: int = Field(3, ge=1, le=20)

    PASSIVE_JOINER_ENABLED: bool = False
    PASSIVE_JOINER_TIMEOUT_SECONDS: float = Field(15.0, gt=0.0, le=120.0)
    PASSIVE_JOINER_MAX_ATTEMPTS: int = Field(3, ge=1, le=10)
    PASSIVE_JOINER_SIDE_BUY_PRICE: str = "best_bid"
    PASSIVE_JOINER_REQUIRE_FRESH_QUOTE: bool = True

    DATABASE_PATH: Path = Field(Path("./runtime/tradingbot.sqlite3"))

    # ---- Phase 8: adaptive brain & remote command center ---------------------
    ENABLE_AUTOTUNE: bool = False
    AUTOTUNE_SUNDAY_HOUR_ET: int = Field(21, ge=17, le=23)
    AUTOTUNE_LOOKBACK_DAYS: int = Field(30, ge=7, le=366)
    AUTOTUNE_MIN_TRADES_PER_CONFIG: int = Field(10, ge=1, le=500)
    AUTOTUNE_MAX_DRAWDOWN_ABS: float = Field(0.45, gt=0.0, le=1.0)
    DYNAMIC_PARAMS_PATH: Path = Field(default_factory=lambda: Path("src/config/dynamic_params.json"))

    ENABLE_ML_FILTER: bool = False
    ML_FILTER_THRESHOLD: float = Field(0.55, gt=0.0, lt=1.0)
    MIN_ML_TRAINING_TRADES: int = Field(50, ge=5, le=50_000)
    ML_MODEL_PATH: Path = Field(Path("./runtime/models/ml_signal_filter.pkl"))
    ML_MODEL_META_PATH: Path = Field(Path("./runtime/models/ml_signal_filter_meta.json"))
    ML_MAX_TRAINING_TRADES: int = Field(500, ge=50, le=50_000)
    ML_INFERENCE_RECENT_CONTEXT: int = Field(100, ge=10, le=5000)
    ML_ABORT_ON_TRAINING_FAILURE: bool = False
    ML_BLOCK_ENTRIES_ON_TRAINING_FAILURE: bool = True

    ENABLE_DISCORD_BOT: bool = False
    # When True, standalone Discord startup/first-contact must succeed or startup aborts.
    REQUIRE_DISCORD_ON_STARTUP: bool = False
    DISCORD_BOT_TOKEN: str = Field("", repr=False)
    DISCORD_CHANNEL_ID: str = ""
    DISCORD_ALLOWED_USER_IDS: str = ""
    DISCORD_COMMAND_RATE_LIMIT_SECONDS: float = Field(5.0, ge=1.0, le=3600.0)

    # Entry-skip diagnostics: cooldown for Discord + optional log throttling for repetitive reasons.
    SKIP_DIAGNOSTICS_DISCORD_COOLDOWN_SECONDS: float = Field(45.0, ge=0.0, le=86400.0)
    DISCORD_SKIP_ALERTS_ENABLED: bool = True
    DISCORD_SKIP_ALERT_COOLDOWN_SECONDS: Optional[float] = Field(
        default=None, ge=0.0, le=86400.0
    )
    LOG_ALL_PRETRADE_SKIPS: bool = False
    SKIP_DIAGNOSTICS_NOISY_LOG_THROTTLE_SECONDS: float = Field(
        120.0, ge=0.0, le=86400.0
    )
    SKIP_DIAGNOSTICS_UNIVERSE_LOG_THROTTLE_SECONDS: float = Field(
        45.0, ge=0.0, le=86400.0
    )

    ENABLE_KELLY_SIZING: bool = False
    KELLY_USE_RETURN_PCT: bool = True
    KELLY_LOOKBACK_TRADES: int = Field(100, ge=5, le=50_000)
    KELLY_FRACTION: float = Field(0.25, gt=0.0, le=1.0)
    KELLY_MIN_TRADES: int = Field(30, ge=5, le=50_000)
    KELLY_MAX_RISK_MULTIPLIER: float = Field(1.5, gt=0.0, le=5.0)
    KELLY_MIN_RISK_MULTIPLIER: float = Field(0.25, gt=0.0, le=5.0)
    # Optional aliases requested by the research guardrail spec.
    KELLY_MAX_MULTIPLIER: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    KELLY_MIN_MULTIPLIER: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    KELLY_EXCLUDE_CANARY: bool = True
    KELLY_EXCLUDE_DEGRADED_LABELS: bool = True

    ANTI_MARTINGALE_ENABLED: bool = False
    ANTI_MARTINGALE_LOSS_STREAK: int = Field(3, ge=1, le=50)
    ANTI_MARTINGALE_WIN_RECOVERY: int = Field(2, ge=1, le=50)
    ANTI_MARTINGALE_DEFENSIVE_MULTIPLIER: float = Field(0.5, gt=0.0, le=1.0)
    ANTI_MARTINGALE_NORMAL_MULTIPLIER: float = Field(1.0, gt=0.0, le=1.0)
    ENABLE_REALIZED_CORRELATION: bool = True

    REPORTS_DIR: Path = Path("./reports")
    DAILY_REPORT_ENABLED: bool = False

    TEARSHEET_PRIMARY: Literal["sqlite", "orders_log"] = "sqlite"

    # ---- Local chaos / resilience (laptop soak testing; optional on VPS)
    STREAM_STALE_SECONDS: float = Field(default=STREAM_STALE_SECONDS_DEFAULT, gt=5.0, le=900.0)
    STREAM_NOTIFICATION_COOLDOWN_SECONDS: float = Field(
        default=STREAM_NOTIFICATION_COOLDOWN_SECONDS_DEFAULT,
        ge=30.0,
        le=86400.0,
    )
    ENABLE_LOCAL_NOTIFICATIONS: bool = True

    WARN_ON_LOW_BATTERY: bool = True
    LOW_BATTERY_THRESHOLD_PCT: int = Field(default=LOW_BATTERY_THRESHOLD_PCT_DEFAULT, ge=0, le=100)
    REQUIRE_POWER_FOR_LOCAL_LIVE: bool = False

    # ---- Validators ----------------------------------------------------------------
    @field_validator("ALPACA_ENV")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        if v not in VALID_ALPACA_ENVS:
            raise ValueError(f"ALPACA_ENV must be one of {sorted(VALID_ALPACA_ENVS)}")
        return v

    @field_validator("ALPACA_FEED")
    @classmethod
    def _validate_feed(cls, v: str) -> str:
        if v not in VALID_FEEDS:
            raise ValueError(f"ALPACA_FEED must be one of {sorted(VALID_FEEDS)}")
        return v

    @field_validator("REGULATORY_MODE")
    @classmethod
    def _validate_reg_mode(cls, v: str) -> str:
        if v not in VALID_REGULATORY_MODES:
            raise ValueError(
                f"REGULATORY_MODE must be one of {sorted(VALID_REGULATORY_MODES)}"
            )
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return upper

    @field_validator("SYMBOLS")
    @classmethod
    def _validate_symbols(cls, v: str) -> str:
        cleaned = ",".join(s.strip().upper() for s in v.split(",") if s.strip())
        if not cleaned:
            raise ValueError("SYMBOLS must contain at least one ticker")
        for sym in cleaned.split(","):
            if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
                raise ValueError(f"SYMBOLS contains invalid ticker: {sym!r}")
        return cleaned

    @field_validator("CANARY_SYMBOL")
    @classmethod
    def _validate_canary_symbol(cls, v: str) -> str:
        sym = v.strip().upper()
        if not sym:
            raise ValueError("CANARY_SYMBOL must not be empty")
        if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
            raise ValueError(f"CANARY_SYMBOL invalid ticker: {sym!r}")
        return sym

    @field_validator("CANARY_PERSIST_FILENAME")
    @classmethod
    def _validate_canary_filename(cls, v: str) -> str:
        name = v.strip()
        if not name:
            raise ValueError("CANARY_PERSIST_FILENAME must not be empty")
        # Disallow path separators - this is just a filename under STATE_DIR.
        if any(sep in name for sep in ("/", "\\", "..")):
            raise ValueError(f"CANARY_PERSIST_FILENAME must be a bare filename: {name!r}")
        return name

    @field_validator(
        "SPREAD_FILTER_PCT_IEX",
        "ELASTIC_SPREAD_HARD_MAX_PCT",
        "QUOTE_MAX_AGE_SECONDS",
        "DISCORD_SKIP_ALERT_COOLDOWN_SECONDS",
        "KELLY_MAX_MULTIPLIER",
        "KELLY_MIN_MULTIPLIER",
        mode="before",
    )
    @classmethod
    def _coerce_empty_optional_float(cls, value: object) -> object:
        if value in {"", None}:
            return None
        return value

    @field_validator("ADX_THRESHOLD", mode="before")
    @classmethod
    def _coerce_empty_adx_threshold(cls, value: object) -> object:
        if value in {"", None}:
            return None
        return value

    @field_validator("BOLLINGER_BW_MIN", mode="before")
    @classmethod
    def _coerce_empty_bollinger_bw_min(cls, value: object) -> object:
        if value in {"", None}:
            return None
        return value

    @field_validator("TIME_OF_DAY_TRADE_START", "TIME_OF_DAY_TRADE_END")
    @classmethod
    def _validate_hhmm_time(cls, value: str) -> str:
        raw = (value or "").strip()
        parts = raw.split(":")
        if len(parts) != 2:
            raise ValueError("time-of-day values must use HH:MM format")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:
            raise ValueError("time-of-day values must use HH:MM format") from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("time-of-day HH:MM must be within 00:00..23:59")
        return f"{hour:02d}:{minute:02d}"

    @field_validator("ACTIVE_STRATEGIES")
    @classmethod
    def _validate_active_strategies(cls, value: str) -> str:
        from strategies.registry import normalize_strategy_name, supported_strategy_names

        raw = (value or "").strip()
        if not raw:
            raise ValueError("ACTIVE_STRATEGIES must list at least one strategy name")
        known = supported_strategy_names()
        out: list[str] = []
        for chunk in raw.split(","):
            token = chunk.strip()
            if not token:
                raise ValueError("ACTIVE_STRATEGIES must not contain empty names")
            canon = normalize_strategy_name(token)
            if canon not in known:
                raise ValueError(
                    f"ACTIVE_STRATEGIES unknown strategy {token!r} "
                    f"(canonical={canon!r}; supported={sorted(known)})",
                )
            out.append(canon)
        return ",".join(out)

    @field_validator("STRATEGY_WEIGHTS_JSON")
    @classmethod
    def _validate_strategy_weights_json(cls, value: str) -> str:
        from strategies.registry import normalize_strategy_name, supported_strategy_names

        raw = (value or "").strip()
        if not raw:
            raise ValueError("STRATEGY_WEIGHTS_JSON must not be empty")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"STRATEGY_WEIGHTS_JSON must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("STRATEGY_WEIGHTS_JSON must decode to a JSON object")
        known = supported_strategy_names()
        weights: dict[str, float] = {}
        for k, v in parsed.items():
            ks = str(k).strip()
            if not ks:
                raise ValueError("STRATEGY_WEIGHTS_JSON keys must be non-empty")
            canon = normalize_strategy_name(ks)
            if canon not in known:
                raise ValueError(
                    f"STRATEGY_WEIGHTS_JSON unknown strategy key {k!r} "
                    f"(canonical={canon!r}; supported={sorted(known)})",
                )
            try:
                fv = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"STRATEGY_WEIGHTS_JSON value for {k!r} must be numeric") from exc
            if fv < 0.0 or fv > 1_000_000.0:
                raise ValueError(f"STRATEGY_WEIGHTS_JSON weight for {k!r} out of range")
            weights[canon] = fv
        return json.dumps(weights, separators=(",", ":"))

    @field_validator("ETF_ROTATION_SYMBOLS")
    @classmethod
    def _validate_etf_rotation_symbols(cls, v: str) -> str:
        cleaned = ",".join(s.strip().upper() for s in v.split(",") if s.strip())
        if not cleaned:
            raise ValueError("ETF_ROTATION_SYMBOLS requires at least one symbol")
        for sym in cleaned.split(","):
            if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
                raise ValueError(f"ETF_ROTATION_SYMBOLS invalid ticker: {sym!r}")
        return cleaned

    @field_validator("PAIRS_CONFIG_JSON")
    @classmethod
    def _validate_pairs_config_json(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PAIRS_CONFIG_JSON must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("PAIRS_CONFIG_JSON must decode to a JSON object")
        out: dict[str, str] = {}
        for k, v in parsed.items():
            fk = str(k).strip().upper()
            lv = str(v).strip().upper()
            if not fk or not lv:
                raise ValueError("PAIRS_CONFIG_JSON keys and values must be non-empty tickers")
            for sym in (fk, lv):
                if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
                    raise ValueError(f"PAIRS_CONFIG_JSON invalid ticker: {sym!r}")
            if fk == lv:
                raise ValueError("PAIRS_CONFIG_JSON follower and leader must differ")
            out[fk] = lv
        return json.dumps(out, separators=(",", ":"))

    @field_validator("SECTOR_MAP_JSON")
    @classmethod
    def _validate_sector_map_json(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return "{}"
        try:
            parsed = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"SECTOR_MAP_JSON must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("SECTOR_MAP_JSON must decode to an object/dict")
        normalized: dict[str, str] = {}
        for k, v in parsed.items():
            ks = str(k).strip().upper()
            vs = str(v).strip()
            if not ks or not vs:
                continue
            normalized[ks] = vs
        return json.dumps(normalized, separators=(",", ":"))

    @field_validator("SPREAD_FILTER_PCT_IEX")
    @classmethod
    def _validate_spread_iex(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        x = float(value)
        if x <= 0.0 or x > 0.05:
            raise ValueError("SPREAD_FILTER_PCT_IEX must satisfy 0 < value <= 0.05 (or be empty / unset).")
        return x

    @field_validator("CORRELATION_LEADER_SYMBOL", "BLACK_SWAN_SYMBOL")
    @classmethod
    def _validate_phase3_leader_symbols(cls, v: str) -> str:
        sym = v.strip().upper()
        if not sym:
            raise ValueError("Ticker must not be empty")
        if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
            raise ValueError(f"invalid ticker: {sym!r}")
        return sym

    @field_validator("CORRELATION_FOLLOWER_SYMBOLS")
    @classmethod
    def _validate_correlation_followers(cls, v: str) -> str:
        cleaned = ",".join(s.strip().upper() for s in v.split(",") if s.strip())
        if not cleaned:
            raise ValueError("CORRELATION_FOLLOWER_SYMBOLS requires at least one symbol")
        for sym in cleaned.split(","):
            if not sym.isascii() or not all(c.isalnum() or c in {".", "-", "/"} for c in sym):
                raise ValueError(f"CORRELATION_FOLLOWER_SYMBOLS invalid ticker: {sym!r}")
        return cleaned

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
        if self.MAX_DOLLARS_PER_TRADE > 0:
            self.MAX_EQUITY_USAGE_USD = float(self.MAX_DOLLARS_PER_TRADE)
        if self.ENABLE_ELASTIC_SPREAD is not None:
            self.SPREAD_FILTER_ELASTIC_ENABLED = bool(self.ENABLE_ELASTIC_SPREAD)
        if self.ELASTIC_SPREAD_ENABLED is not None:
            self.SPREAD_FILTER_ELASTIC_ENABLED = bool(self.ELASTIC_SPREAD_ENABLED)
        if self.ELASTIC_SPREAD_HARD_MAX_PCT is not None:
            self.SPREAD_FILTER_MAX_PCT = float(self.ELASTIC_SPREAD_HARD_MAX_PCT)
        if self.ENABLE_QUOTE_FALLBACK is not None:
            self.QUOTE_FALLBACK_ENABLED = bool(self.ENABLE_QUOTE_FALLBACK)
        if self.ENABLE_REGIME_ADAPTIVE_RSI is not None:
            self.REGIME_ADAPTIVE_RSI_ENABLED = bool(self.ENABLE_REGIME_ADAPTIVE_RSI)
        if self.QUOTE_MAX_AGE_SECONDS is not None:
            self.QUOTE_STALENESS_SECONDS = float(self.QUOTE_MAX_AGE_SECONDS)
        if self.DISCORD_SKIP_ALERT_COOLDOWN_SECONDS is not None:
            self.SKIP_DIAGNOSTICS_DISCORD_COOLDOWN_SECONDS = float(
                self.DISCORD_SKIP_ALERT_COOLDOWN_SECONDS
            )
        if self.LOG_ALL_PRETRADE_SKIPS:
            self.SKIP_DIAGNOSTICS_NOISY_LOG_THROTTLE_SECONDS = 0.0
            self.SKIP_DIAGNOSTICS_UNIVERSE_LOG_THROTTLE_SECONDS = 0.0
        if self.ALLOW_FRACTIONAL is not None:
            self.ENABLE_FRACTIONAL = bool(self.ALLOW_FRACTIONAL)
        if self.FRACTIONAL_TRADING_ENABLED is not None:
            self.ENABLE_FRACTIONAL = bool(self.FRACTIONAL_TRADING_ENABLED)
        if self.BOLLINGER_BW_MIN is not None:
            self.BOLLINGER_MIN_WIDTH_PCT = float(self.BOLLINGER_BW_MIN)
        if self.ADX_THRESHOLD is not None:
            self.ADX_RANGE_MAX = float(self.ADX_THRESHOLD)
        if self.KELLY_MAX_MULTIPLIER is not None:
            self.KELLY_MAX_RISK_MULTIPLIER = float(self.KELLY_MAX_MULTIPLIER)
        if self.KELLY_MIN_MULTIPLIER is not None:
            self.KELLY_MIN_RISK_MULTIPLIER = float(self.KELLY_MIN_MULTIPLIER)
        if self.KELLY_MIN_RISK_MULTIPLIER > self.KELLY_MAX_RISK_MULTIPLIER:
            raise ValueError("KELLY_MIN_RISK_MULTIPLIER must be <= KELLY_MAX_RISK_MULTIPLIER.")
        # Live trading requires explicit confirmation phrase.
        if (
            self.ALPACA_ENV == ALPACA_ENV_LIVE
            and self.LIVE_TRADING_ENABLED
            and self.CONFIRM_LIVE_TRADING != LIVE_TRADING_CONFIRMATION_PHRASE
        ):
            raise ValueError(
                "Live trading requested but CONFIRM_LIVE_TRADING is not set to "
                f"the required phrase {LIVE_TRADING_CONFIRMATION_PHRASE!r}."
            )
        # Sane RSI thresholds
        if self.RSI_OVERSOLD >= self.RSI_EXIT:
            raise ValueError(
                "RSI_OVERSOLD must be strictly less than RSI_EXIT for "
                "a mean-reversion long strategy."
            )
        if self.DEFAULT_RSI_ENTRY >= self.RSI_EXIT:
            raise ValueError("DEFAULT_RSI_ENTRY must be strictly less than RSI_EXIT.")
        if self.HIGH_VOL_RSI_ENTRY >= self.RSI_EXIT:
            raise ValueError("HIGH_VOL_RSI_ENTRY must be strictly less than RSI_EXIT.")
        if self.AGGRESSIVE_RSI_BYPASS_THRESHOLD >= self.RSI_EXIT:
            raise ValueError("AGGRESSIVE_RSI_BYPASS_THRESHOLD must be < RSI_EXIT.")
        if self.DYNAMIC_RSI_MIN > self.DYNAMIC_RSI_MAX:
            raise ValueError("DYNAMIC_RSI_MIN must be <= DYNAMIC_RSI_MAX.")
        if not (self.DYNAMIC_RSI_MIN <= self.DYNAMIC_RSI_BASE <= self.DYNAMIC_RSI_MAX):
            raise ValueError("DYNAMIC_RSI_BASE must be inside [DYNAMIC_RSI_MIN, DYNAMIC_RSI_MAX].")
        if self.DYNAMIC_RSI_SHORT_ATR > self.DYNAMIC_RSI_LONG_ATR:
            raise ValueError("DYNAMIC_RSI_SHORT_ATR must be <= DYNAMIC_RSI_LONG_ATR.")
        if self.ADX_LOW >= self.ADX_HIGH:
            raise ValueError("ADX_LOW must be strictly less than ADX_HIGH.")
        if self.REGIME_BEAR_RSI_MAX >= self.REGIME_BULL_RSI_MIN:
            raise ValueError("REGIME_BEAR_RSI_MAX must be below REGIME_BULL_RSI_MIN.")
        if self.REGIME_BULL_RSI_MIN > self.REGIME_PARABOLIC_RSI_MIN:
            raise ValueError("REGIME_BULL_RSI_MIN must be <= REGIME_PARABOLIC_RSI_MIN.")
        if self.SCALE_IN_UNDERWATER_PCT >= 0:
            raise ValueError("SCALE_IN_UNDERWATER_PCT must be negative.")
        if not (1.0 <= self.SCALE_IN_RSI_THRESHOLD <= 50.0):
            raise ValueError("SCALE_IN_RSI_THRESHOLD must be between 1 and 50.")
        if self.SCALE_IN_ADD_QTY <= 0:
            raise ValueError("SCALE_IN_ADD_QTY must be > 0.")
        if self.VWAP_PULLBACK_TREND_FAST_SMA >= self.VWAP_PULLBACK_TREND_SLOW_SMA:
            raise ValueError("VWAP_PULLBACK_TREND_FAST_SMA must be < VWAP_PULLBACK_TREND_SLOW_SMA.")
        if self.MOMENTUM_FAST_SMA >= self.MOMENTUM_SLOW_SMA:
            raise ValueError("MOMENTUM_FAST_SMA must be < MOMENTUM_SLOW_SMA.")
        if self.VWAP_PULLBACK_RSI_MIN >= self.VWAP_PULLBACK_RSI_MAX:
            raise ValueError("VWAP_PULLBACK_RSI_MIN must be < VWAP_PULLBACK_RSI_MAX.")
        if self.VWAP_PULLBACK_ADX_MIN > self.VWAP_PULLBACK_ADX_MAX:
            raise ValueError("VWAP_PULLBACK_ADX_MIN must be <= VWAP_PULLBACK_ADX_MAX.")
        if self.PAIRS_EXIT_Z <= self.PAIRS_ENTRY_Z:
            raise ValueError("PAIRS_EXIT_Z must be > PAIRS_ENTRY_Z for convergence exits.")
        if self.STRATEGY_RUN_MODE == "single":
            names = self.active_strategies_list
            if len(names) != 1:
                raise ValueError(
                    "STRATEGY_RUN_MODE=single requires exactly one strategy in ACTIVE_STRATEGIES",
                )
        # Retry bounds
        if self.RETRY_BASE_DELAY_SECONDS > self.RETRY_MAX_DELAY_SECONDS:
            raise ValueError(
                "RETRY_BASE_DELAY_SECONDS must be <= RETRY_MAX_DELAY_SECONDS."
            )
        db_str = str(self.DATABASE_PATH).strip()
        if not db_str:
            raise ValueError("DATABASE_PATH must not be empty")
        if not str(self.REPORTS_DIR).strip():
            raise ValueError("REPORTS_DIR must not be empty")
        if self.ENSEMBLE_MIN_WEIGHT > self.ENSEMBLE_MAX_WEIGHT:
            raise ValueError("ENSEMBLE_MIN_WEIGHT must be <= ENSEMBLE_MAX_WEIGHT.")
        return self

    # ---- Convenience ---------------------------------------------------------------
    @property
    def symbols_list(self) -> list[str]:
        """Return SYMBOLS as a clean uppercase list."""
        return [s for s in self.SYMBOLS.split(",") if s]

    @property
    def active_strategies_list(self) -> list[str]:
        """Canonical strategy keys from ``ACTIVE_STRATEGIES`` (order preserved)."""
        return [s for s in self.ACTIVE_STRATEGIES.split(",") if s]

    @property
    def strategy_weights_dict(self) -> dict[str, float]:
        """Weights keyed by canonical strategy name (defaults missing actives to 1.0)."""
        try:
            raw = json.loads(self.STRATEGY_WEIGHTS_JSON or "{}")
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        for name in self.active_strategies_list:
            out.setdefault(name, 1.0)
        return out

    @property
    def discord_allowed_user_ids_set(self) -> set[int]:
        ids: set[int] = set()
        for chunk in self.DISCORD_ALLOWED_USER_IDS.split(","):
            c = chunk.strip()
            if not c:
                continue
            try:
                ids.add(int(c))
            except ValueError:
                continue
        return ids

    @property
    def correlation_follower_symbols_list(self) -> list[str]:
        cleaned = ",".join(s.strip().upper() for s in self.CORRELATION_FOLLOWER_SYMBOLS.split(",") if s.strip())
        return [s for s in cleaned.split(",") if s]

    @property
    def is_paper(self) -> bool:
        return self.ALPACA_ENV == ALPACA_ENV_PAPER

    @property
    def is_live_endpoint(self) -> bool:
        return self.ALPACA_ENV == ALPACA_ENV_LIVE

    @property
    def can_submit_real_orders(self) -> bool:
        """True only when both the live switch is on AND DRY_RUN is off.

        On the paper endpoint we still respect DRY_RUN to allow developers to
        do a fully-instrumented dry run on paper without ever pinging the
        order-placement API.
        """
        return self.LIVE_TRADING_ENABLED and not self.DRY_RUN

    @property
    def feed_preference(self) -> str:
        """Return the user-requested feed; resolution to sip/iex happens at runtime."""
        return self.ALPACA_FEED

    def feed_resolved(self, sip_supported: bool) -> str:
        """Resolve `auto` based on whether SIP entitlement is detected."""
        if self.ALPACA_FEED == FEED_AUTO:
            return FEED_SIP if sip_supported else FEED_IEX
        return self.ALPACA_FEED

    @property
    def regulatory_mode(self) -> str:
        return self.REGULATORY_MODE

    @property
    def is_regulatory_auto(self) -> bool:
        return self.REGULATORY_MODE == REGULATORY_MODE_AUTO

    @property
    def is_regulatory_pdt(self) -> bool:
        return self.REGULATORY_MODE == REGULATORY_MODE_PDT

    @property
    def is_regulatory_intraday(self) -> bool:
        return self.REGULATORY_MODE == REGULATORY_MODE_INTRADAY_MARGIN

    def resolved_capital_base(self, account_equity: float) -> float:
        """Return the capital base used for per-trade risk budgeting.

        - If BOT_CAPITAL_BASE_USD > 0 the operator has explicitly allocated a
          slice of the account; that wins.
        - Otherwise we fall back to the smaller of the account's equity and
          the bot-managed MAX_EQUITY_USAGE_USD cap, so an underfunded account
          never inflates risk and a large account never lets the bot risk
          beyond its hard USD ceiling.
        """
        if self.BOT_CAPITAL_BASE_USD > 0:
            return float(self.BOT_CAPITAL_BASE_USD)
        eq = max(0.0, float(account_equity))
        return min(eq, float(self.MAX_EQUITY_USAGE_USD))

    @property
    def max_dollars_per_trade(self) -> float:
        return float(self.MAX_EQUITY_USAGE_USD)

    @property
    def sector_map(self) -> dict[str, str]:
        try:
            raw = json.loads(self.SECTOR_MAP_JSON or "{}")
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            ks = str(k).strip().upper()
            vs = str(v).strip()
            if ks and vs:
                out[ks] = vs
        return out

    def sector_for_symbol(self, symbol: str) -> str:
        return self.sector_map.get(str(symbol).strip().upper(), "Unknown")

    def spread_filter_pct_for_feed(self, feed: Optional[str]) -> float:
        """Max allowed relative spread for a quote from the given Alpaca data feed.

        SIP (and unknown feeds) always use ``SPREAD_FILTER_PCT``. When
        ``SPREAD_FILTER_PCT_IEX`` is set, quotes with ``feed=iex`` use that
        threshold instead; otherwise IEX quotes use the same threshold as other
        feeds.
        """
        f = (feed or "").strip().lower()
        if f == FEED_IEX and self.SPREAD_FILTER_PCT_IEX is not None:
            return float(self.SPREAD_FILTER_PCT_IEX)
        return float(self.SPREAD_FILTER_PCT)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    pydantic-settings raises a ValidationError on first access if anything is
    missing or invalid; we let that bubble up so the process fails fast.
    """
    return Settings()  # type: ignore[call-arg]
