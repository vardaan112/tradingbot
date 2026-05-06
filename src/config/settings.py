"""Strongly validated settings loaded from environment variables.

Every runtime configuration value flows through `Settings`. Direct use of
`os.environ` elsewhere in the codebase is intentionally avoided so that all
configuration validation happens in exactly one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    ALPACA_ENV_LIVE,
    ALPACA_ENV_PAPER,
    FEED_AUTO,
    FEED_IEX,
    FEED_SIP,
    LIVE_TRADING_CONFIRMATION_PHRASE,
    REGULATORY_MODE_AUTO,
    REGULATORY_MODE_INTRADAY_MARGIN,
    REGULATORY_MODE_PDT,
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

    # ---- Logging and storage -------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")
    STATE_DIR: Path = Path("./runtime")

    # ---- Heartbeat and orchestration -----------------------------------------------
    HEARTBEAT_INTERVAL_SECONDS: float = Field(60.0, ge=5.0, le=600.0)
    ORCHESTRATOR_TICK_SECONDS: float = Field(15.0, ge=1.0, le=300.0)

    # ---- Universe and bars ---------------------------------------------------------
    SYMBOLS: str = "AAPL,MSFT,SPY"
    BAR_TIMEFRAME: Literal["1Min", "5Min", "15Min", "1Hour", "1Day"] = "5Min"

    # ---- Strategy parameters -------------------------------------------------------
    RSI_LENGTH: int = Field(14, ge=2, le=200)
    RSI_OVERSOLD: float = Field(30.0, ge=1.0, le=99.0)
    RSI_EXIT: float = Field(50.0, ge=1.0, le=99.0)
    ATR_LENGTH: int = Field(14, ge=2, le=200)
    ATR_STOP_MULTIPLIER: float = Field(2.0, gt=0.0, le=20.0)
    ATR_PROFIT_MULTIPLIER: float = Field(3.0, gt=0.0, le=50.0)
    MAX_HOLD_BARS: int = Field(24, ge=1, le=10_000)

    # ---- Risk ----------------------------------------------------------------------
    MAX_RISK_PER_TRADE_PCT: float = Field(0.01, gt=0.0, le=0.05)
    MAX_EQUITY_USAGE_USD: float = Field(50.0, gt=0.0)
    MAX_GROSS_EXPOSURE_PCT: float = Field(0.5, gt=0.0, le=2.0)
    MAX_OPEN_POSITIONS: int = Field(1, ge=1, le=100)
    KILL_SWITCH_DRAWDOWN_PCT: float = Field(0.05, gt=0.0, le=0.5)

    # ---- Quote / execution filters -------------------------------------------------
    SPREAD_FILTER_PCT: float = Field(0.0005, gt=0.0, le=0.05)
    QUOTE_STALENESS_SECONDS: float = Field(5.0, gt=0.0, le=300.0)
    ORDER_TIMEOUT_SECONDS: float = Field(30.0, gt=0.0, le=3600.0)
    EMERGENCY_AGGRESSIVENESS_PCT: float = Field(0.0015, gt=0.0, le=0.05)

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

    @model_validator(mode="after")
    def _validate_consistency(self) -> "Settings":
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
        # Retry bounds
        if self.RETRY_BASE_DELAY_SECONDS > self.RETRY_MAX_DELAY_SECONDS:
            raise ValueError(
                "RETRY_BASE_DELAY_SECONDS must be <= RETRY_MAX_DELAY_SECONDS."
            )
        return self

    # ---- Convenience ---------------------------------------------------------------
    @property
    def symbols_list(self) -> list[str]:
        """Return SYMBOLS as a clean uppercase list."""
        return [s for s in self.SYMBOLS.split(",") if s]

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings instance.

    pydantic-settings raises a ValidationError on first access if anything is
    missing or invalid; we let that bubble up so the process fails fast.
    """
    return Settings()  # type: ignore[call-arg]
