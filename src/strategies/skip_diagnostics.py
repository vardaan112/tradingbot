"""Structured diagnostics when a long entry is skipped (logging + optional Discord)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from config.settings import Settings
from core.market_data import Quote
from utils.price_utils import spread_pct as _spread_pct

from .filters import RegimeSnapshot


@dataclass(frozen=True)
class SkipReason:
    """Single skip decision with optional numeric / regime context."""

    code: str
    message: str
    symbol: str
    decision: str = "SKIP"
    rsi: float | None = None
    price: float | None = None
    sma_200: float | None = None
    sma_200_slope: float | None = None
    adx: float | None = None
    atr: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    spread_threshold_pct: float | None = None
    quote_age_seconds: float | None = None
    strategy_bar_ts: datetime | None = None
    dashboard_bar_ts: datetime | None = None
    position_qty: float | None = None
    open_order_exists: bool | None = None
    risk_qty: float | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Codes referenced in logs / tests (stable contract).
class SkipCodes:
    SPREAD_TOO_WIDE = "SPREAD_TOO_WIDE"
    QUOTE_INVALID = "QUOTE_INVALID"
    SIZE_ZERO = "SIZE_ZERO"
    STALE_BARS = "STALE_BARS"
    RSI_NOT_TRIGGERED = "RSI_NOT_TRIGGERED"
    ADX_FILTER_FAIL = "ADX_FILTER_FAIL"
    SMA_FILTER_FAIL = "SMA_FILTER_FAIL"
    AGGRESSIVE_SMA_BYPASS = "AGGRESSIVE_SMA_BYPASS"
    VOLATILITY_THRESHOLD_USED = "VOLATILITY_THRESHOLD_USED"
    SECTOR_LIMIT_FAIL = "SECTOR_LIMIT_FAIL"
    ALREADY_IN_POSITION = "ALREADY_IN_POSITION"
    MARKET_CLOSED = "MARKET_CLOSED"
    ORDER_REJECTED = "ORDER_REJECTED"
    RISK_LIMIT_FAIL = "RISK_LIMIT_FAIL"
    KILL_SWITCH_LATCHED = "KILL_SWITCH_LATCHED"
    COMPLIANCE_REJECTED = "COMPLIANCE_REJECTED"
    MISSING_BARS = "MISSING_BARS"
    INDICATOR_WARMUP = "INDICATOR_WARMUP"
    INVALID_INDICATORS = "INVALID_INDICATORS"
    OPEN_ORDER_EXISTS = "OPEN_ORDER_EXISTS"
    MISSING_QUOTE = "MISSING_QUOTE"
    SENTIMENT_BLOCK = "SENTIMENT_BLOCK"
    ML_FILTER_BLOCK = "ML_FILTER_BLOCK"
    CORRELATION_BLOCK = "CORRELATION_BLOCK"
    STREAM_UNHEALTHY = "STREAM_UNHEALTHY"
    UNIVERSE_INELIGIBLE = "UNIVERSE_INELIGIBLE"
    PRICE_BELOW_MIN = "PRICE_BELOW_MIN"
    ADV_BELOW_MIN = "ADV_BELOW_MIN"
    SPREAD_COMPUTE_FAILED = "SPREAD_COMPUTE_FAILED"
    REGIME_FILTER = "REGIME_FILTER"
    UNKNOWN_SKIP = "UNKNOWN_SKIP"

    # Legacy aliases (kept for compatibility with older references/tests).
    RSI_NOT_OVERSOLD = RSI_NOT_TRIGGERED
    NO_COMPLETED_BARS = MISSING_BARS
    OPEN_ORDER_BLOCKS_ENTRY = OPEN_ORDER_EXISTS
    NO_QUOTE_FOR_ENTRY = MISSING_QUOTE
    SENTIMENT_FILTER = SENTIMENT_BLOCK
    ML_FILTER = ML_FILTER_BLOCK
    SPREAD_ABOVE_THRESHOLD = SPREAD_TOO_WIDE
    QUOTE_STALE_OR_INVALID = QUOTE_INVALID
    OPEN_ORDER_PRESENT = OPEN_ORDER_EXISTS
    NO_QUOTE = MISSING_QUOTE
    KILL_SWITCH = KILL_SWITCH_LATCHED
    WINDOW_CLOSED = MARKET_CLOSED
    COMPLIANCE = COMPLIANCE_REJECTED
    SIZING_ZERO = SIZE_ZERO


NOISY_SKIP_CODES: frozenset[str] = frozenset({SkipCodes.RSI_NOT_TRIGGERED})


@dataclass
class SkipDiagnosticsThrottle:
    """Per-process throttle for repetitive skip lines (monotonic clock)."""

    _log_last: dict[tuple[str, str], float] = field(default_factory=dict)
    _discord_last: dict[tuple[str, str], float] = field(default_factory=dict)

    def allow_log(self, symbol: str, code: str, throttle_seconds: float) -> bool:
        if throttle_seconds <= 0:
            return True
        key = (symbol.upper(), code)
        now = time.monotonic()
        prev = self._log_last.get(key, 0.0)
        if now - prev < throttle_seconds:
            return False
        self._log_last[key] = now
        return True

    def allow_discord(self, symbol: str, code: str, cooldown_seconds: float) -> bool:
        if cooldown_seconds <= 0:
            return True
        key = (symbol.upper(), code)
        now = time.monotonic()
        prev = self._discord_last.get(key, 0.0)
        if now - prev < cooldown_seconds:
            return False
        self._discord_last[key] = now
        return True


def skip_reason_to_discord_spec(
    sr: SkipReason,
    *,
    title: str = "ENTRY_SKIP",
    color: int = 0x95A5A6,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    """Queue payload for ``enqueue_discord_alert``."""

    lines: list[str] = [
        f"code={sr.code}",
        f"symbol={sr.symbol}",
        sr.message,
    ]
    if strategy_name:
        lines.append(f"strategy={strategy_name}")
    if sr.rsi is not None:
        lines.append(f"rsi={sr.rsi:.4f}")
    if sr.price is not None:
        lines.append(f"price={sr.price:.4f}")
    if sr.sma_200 is not None:
        lines.append(f"sma200={sr.sma_200:.4f}")
    if sr.sma_200_slope is not None:
        lines.append(f"sma200_slope={sr.sma_200_slope:.6f}")
    if sr.adx is not None:
        lines.append(f"adx={sr.adx:.4f}")
    if sr.atr is not None:
        lines.append(f"atr={sr.atr:.6f}")
    if sr.bid is not None:
        lines.append(f"bid={sr.bid:.4f}")
    if sr.ask is not None:
        lines.append(f"ask={sr.ask:.4f}")
    if sr.spread_pct is not None:
        lines.append(f"spread_pct={sr.spread_pct:.6f}")
    if sr.spread_threshold_pct is not None:
        lines.append(f"spread_threshold_pct={sr.spread_threshold_pct:.6f}")
    if sr.quote_age_seconds is not None:
        lines.append(f"quote_age_s={sr.quote_age_seconds:.3f}")
    if sr.strategy_bar_ts is not None:
        lines.append(f"strategy_bar_ts={sr.strategy_bar_ts.isoformat()}")
    if sr.dashboard_bar_ts is not None:
        lines.append(f"dashboard_bar_ts={sr.dashboard_bar_ts.isoformat()}")
    if sr.position_qty is not None:
        lines.append(f"position_qty={sr.position_qty:.6f}")
    if sr.open_order_exists is not None:
        lines.append(f"open_order_exists={str(sr.open_order_exists).lower()}")
    if sr.risk_qty is not None:
        lines.append(f"risk_qty={sr.risk_qty:.6f}")
    for k, v in list(sr.metadata.items())[:12]:
        lines.append(f"{k}={v}")
    ts = sr.timestamp or datetime.now(UTC)
    lines.append(f"ts={ts.isoformat()}")
    return {"title": title[:256], "lines": lines[:40], "color": color}


def _fmt_opt(name: str, val: float | None) -> str:
    if val is None:
        return f"{name}=n/a"
    return f"{name}={val:.6g}"


def format_skip_log_line(
    event: str,
    sr: SkipReason,
    *,
    strategy_name: str | None = None,
    phase: str | None = None,
) -> str:
    """One structured line for ``logger.info("%s", line)``."""

    ts = sr.timestamp or datetime.now(UTC)
    parts: list[str] = [
        f"event={event}",
        f"decision={sr.decision}",
        f"code={sr.code}",
        f"symbol={sr.symbol}",
        f'msg="{sr.message}"',
        _fmt_opt("rsi", sr.rsi),
        _fmt_opt("price", sr.price),
        _fmt_opt("sma200", sr.sma_200),
        _fmt_opt("sma200_slope", sr.sma_200_slope),
        _fmt_opt("adx", sr.adx),
        _fmt_opt("atr", sr.atr),
        _fmt_opt("bid", sr.bid),
        _fmt_opt("ask", sr.ask),
        _fmt_opt("spread_pct", sr.spread_pct),
        _fmt_opt("spread_threshold_pct", sr.spread_threshold_pct),
        _fmt_opt("quote_age_seconds", sr.quote_age_seconds),
        f"strategy_bar_ts={sr.strategy_bar_ts.isoformat() if sr.strategy_bar_ts else 'n/a'}",
        f"dashboard_bar_ts={sr.dashboard_bar_ts.isoformat() if sr.dashboard_bar_ts else 'n/a'}",
        _fmt_opt("position_qty", sr.position_qty),
        f"open_order_exists={str(sr.open_order_exists).lower() if sr.open_order_exists is not None else 'n/a'}",
        _fmt_opt("risk_qty", sr.risk_qty),
        f"ts={ts.isoformat()}",
    ]
    if strategy_name:
        parts.append(f"strategy={strategy_name}")
    if phase:
        parts.append(f"phase={phase}")
    meta_items = list(sr.metadata.items())[:16]
    for k, v in meta_items:
        parts.append(f"meta_{k}={v}")
    return " ".join(parts)


def emit_skip_diagnostic(
    *,
    settings: Settings,
    logger: Any,
    log_event: str,
    sr: SkipReason,
    discord_enqueue: Callable[[dict[str, Any]], None] | None,
    throttle: SkipDiagnosticsThrottle,
    strategy_name: str | None = None,
    phase: str | None = None,
    discord_title: str = "ENTRY_SKIP",
    discord_color: int = 0x95A5A6,
    log_guard_seconds: float | None = None,
) -> None:
    """Log skip to console; mirror to Discord subject to cooldown and queue."""

    noisy = sr.code in NOISY_SKIP_CODES
    noisy_interval = (
        float(settings.SKIP_DIAGNOSTICS_NOISY_LOG_THROTTLE_SECONDS)
        if noisy
        else 0.0
    )
    if log_guard_seconds is not None and log_guard_seconds > 0:
        should_log = throttle.allow_log(sr.symbol, sr.code, float(log_guard_seconds))
    elif noisy and noisy_interval > 0:
        should_log = throttle.allow_log(sr.symbol, sr.code, noisy_interval)
    else:
        should_log = True
    if should_log:
        logger.info(
            format_skip_log_line(
                log_event,
                sr,
                strategy_name=strategy_name,
                phase=phase,
            ),
            extra={"symbol": sr.symbol, "skip_code": sr.code},
        )

    if discord_enqueue is None:
        return
    cd = float(settings.SKIP_DIAGNOSTICS_DISCORD_COOLDOWN_SECONDS)
    if not throttle.allow_discord(sr.symbol, sr.code, cd):
        return
    discord_enqueue(
        skip_reason_to_discord_spec(
            sr,
            title=discord_title,
            color=discord_color,
            strategy_name=strategy_name,
        ),
    )


def regime_skip_reason(
    *,
    symbol: str,
    regime_reason: str,
    rsi_value: float,
    last_close: float,
    regime: RegimeSnapshot,
    quote: Quote | None = None,
) -> SkipReason:
    """Build a SkipReason from ``RegimeSnapshot`` and live quote context."""

    sp: float | None = None
    qa: float | None = None
    if quote is not None:
        try:
            qa = float(quote.age_seconds())
        except (AttributeError, TypeError, ValueError):
            qa = None
        try:
            if quote.bid > 0 and quote.ask > quote.bid:
                sp = float(_spread_pct(quote.bid, quote.ask))
        except (ValueError, TypeError):
            sp = None

    return SkipReason(
        code=SkipCodes.REGIME_FILTER,
        message=f"regime_blocked detail={regime_reason}",
        symbol=symbol,
        decision="SKIP",
        rsi=rsi_value,
        price=last_close,
        sma_200=float(regime.sma200),
        sma_200_slope=float(regime.sma_slope),
        adx=float(regime.adx),
        spread_pct=sp,
        quote_age_seconds=qa,
        metadata={
            "regime_type": regime.regime_type,
            "regime_reason": regime_reason,
            "price_above_sma200": regime.price_above_sma200,
            "allow_rsi_long": regime.allow_rsi_long,
        },
    )


__all__ = [
    "NOISY_SKIP_CODES",
    "SkipCodes",
    "SkipDiagnosticsThrottle",
    "SkipReason",
    "emit_skip_diagnostic",
    "format_skip_log_line",
    "regime_skip_reason",
    "skip_reason_to_discord_spec",
]
