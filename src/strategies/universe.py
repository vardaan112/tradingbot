"""Universe selection / per-symbol gating filters.

The orchestrator uses `UniverseFilter.is_eligible` to decide whether a symbol
should be considered for new entries this tick. Existing positions are still
managed (exits, stops) regardless of universe eligibility.

Logging contract:
- A symbol skipped because of a wide spread emits a single structured
  `event=strategy_skip_spread` log line including bid/ask/mid, spread_pct,
  threshold, feed, quote_age_seconds, strategy, and timestamp.
- Other ineligibility reasons emit event=universe_entry_skip with a stable
  ``code`` plus Discord mirroring (subject to cooldown in `skip_diagnostics`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Optional

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings
from core.market_data import Quote
from utils.price_utils import is_valid_quote, mid_price, spread_pct

from .skip_diagnostics import (
    SkipCodes,
    SkipDiagnosticsThrottle,
    SkipReason,
    emit_skip_diagnostic,
)


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason: str
    code: str


def compute_elastic_spread_cap(
    settings: Settings,
    *,
    quote: Quote,
    ref_price: float,
    quote_age_seconds: float,
) -> tuple[float, dict[str, Any]]:
    """Return spread cap (fraction) with elasticity metadata."""

    base_cap = float(settings.spread_filter_pct_for_feed(quote.feed))
    max_cap = float(settings.SPREAD_FILTER_MAX_PCT)
    if not bool(settings.SPREAD_FILTER_ELASTIC_ENABLED):
        return min(base_cap, max_cap), {"base_cap": base_cap, "elastic": False}

    mult = 1.0
    components: list[str] = []
    feed = str(quote.feed or "").strip().lower()
    if feed == "iex":
        mult *= float(settings.SPREAD_FILTER_IEX_ELASTIC_MULTIPLIER)
        components.append("iex")
    if ref_price <= float(settings.SPREAD_FILTER_LOW_PRICE_THRESHOLD):
        mult *= float(settings.SPREAD_FILTER_LOW_PRICE_MULTIPLIER)
        components.append("low_price")
    sparse_threshold = float(settings.SPREAD_FILTER_SPARSE_SIZE_THRESHOLD)
    if float(quote.bid_size) <= sparse_threshold or float(quote.ask_size) <= sparse_threshold:
        mult *= float(settings.SPREAD_FILTER_SPARSE_QUOTE_MULTIPLIER)
        components.append("sparse_quote")
    fresh_cut = float(settings.QUOTE_STALENESS_SECONDS) * float(
        settings.SPREAD_FILTER_FRESH_AGE_FRACTION,
    )
    if quote_age_seconds <= fresh_cut:
        mult *= float(settings.SPREAD_FILTER_FRESH_QUOTE_MULTIPLIER)
        components.append("fresh_quote")

    elastic_cap = min(base_cap * mult, max_cap)
    return elastic_cap, {
        "base_cap": base_cap,
        "elastic": True,
        "elastic_mult": mult,
        "elastic_components": ",".join(components) if components else "none",
        "spread_cap_max_pct": max_cap,
    }


class UniverseFilter:
    """Apply price, liquidity, and quote-quality filters to a symbol."""

    def __init__(
        self,
        settings: Settings,
        *,
        strategy_name: str = "rsi_meanrev",
        discord_enqueue: Callable[[dict[str, Any]], None] | None = None,
        skip_throttle: SkipDiagnosticsThrottle | None = None,
    ) -> None:
        self._settings = settings
        self._strategy_name = strategy_name
        self._log = logging.getLogger(LOGGER_STRATEGY)
        self._discord_enqueue = discord_enqueue
        self._skip_throttle = skip_throttle or SkipDiagnosticsThrottle()

    def _reference_price(self, bars: pd.DataFrame, quote: Quote) -> float:
        if bars is None or bars.empty:
            return (float(quote.bid) + float(quote.ask)) / 2.0
        return float(bars["close"].iloc[-1])

    def _elastic_spread_cap(
        self,
        *,
        quote: Quote,
        ref_price: float,
        quote_age_seconds: float,
    ) -> tuple[float, dict[str, Any]]:
        return compute_elastic_spread_cap(
            self._settings,
            quote=quote,
            ref_price=ref_price,
            quote_age_seconds=quote_age_seconds,
        )

    def _log_spread_gate(
        self,
        *,
        symbol: str,
        bid: float,
        ask: float,
        mid: float,
        spread_abs: float,
        spread_pct_value: float,
        threshold: float,
        passed: bool,
        metadata: dict[str, Any],
    ) -> None:
        self._log.info(
            "event=spread_gate_eval symbol=%s bid=%.6f ask=%.6f mid=%.6f spread=%.6f "
            "spread_pct=%.6f threshold=%.6f result=%s feed=%s elastic=%s components=%s",
            symbol,
            bid,
            ask,
            mid,
            spread_abs,
            spread_pct_value,
            threshold,
            "PASS" if passed else "SKIP",
            metadata.get("feed", "n_a"),
            metadata.get("elastic", False),
            metadata.get("elastic_components", "none"),
            extra={"symbol": symbol},
        )

    def is_eligible(
        self,
        symbol: str,
        *,
        quote: Optional[Quote],
        bars: pd.DataFrame,
        has_position: bool,
        has_open_order: bool,
    ) -> EligibilityResult:
        if has_position:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.ALREADY_IN_POSITION,
                    message="already_in_position cannot_enter_new_long",
                    symbol=symbol,
                ),
            )
            return EligibilityResult(False, "already_in_position", SkipCodes.ALREADY_IN_POSITION)
        if has_open_order:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.OPEN_ORDER_EXISTS,
                    message="open_working_order_blocks_additional_entries",
                    symbol=symbol,
                    open_order_exists=True,
                ),
            )
            return EligibilityResult(False, "open_order_present", SkipCodes.OPEN_ORDER_EXISTS)

        if quote is None:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.MISSING_QUOTE,
                    message="no_quote_available_for_eligibility",
                    symbol=symbol,
                ),
            )
            return EligibilityResult(False, "no_quote", SkipCodes.MISSING_QUOTE)
        max_age = float(self._settings.QUOTE_STALENESS_SECONDS)
        age = float(quote.age_seconds())
        if quote.bid <= 0 or quote.ask <= quote.bid:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.QUOTE_INVALID,
                    message="quote_missing_or_crossed_bid_ask",
                    symbol=symbol,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    quote_age_seconds=age,
                    metadata={"max_quote_age_seconds": max_age},
                ),
                quote=quote,
            )
            return EligibilityResult(False, "quote_invalid", SkipCodes.QUOTE_INVALID)
        if age > max_age or not is_valid_quote(
            quote.bid,
            quote.ask,
            quote_age_seconds=age,
            max_age_seconds=max_age,
        ):
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.QUOTE_INVALID,
                    message="quote_stale_for_spread_validation",
                    symbol=symbol,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    quote_age_seconds=age,
                    metadata={"max_quote_age_seconds": max_age},
                ),
                quote=quote,
            )
            return EligibilityResult(False, "quote_invalid_or_stale", SkipCodes.QUOTE_INVALID)

        try:
            sp = spread_pct(quote.bid, quote.ask)
        except ValueError:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.SPREAD_COMPUTE_FAILED,
                    message="cannot_compute_spread_from_bid_ask",
                    symbol=symbol,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    metadata={"bid": quote.bid, "ask": quote.ask},
                ),
                quote=quote,
            )
            return EligibilityResult(False, "spread_compute_failed", SkipCodes.SPREAD_COMPUTE_FAILED)
        ref_price = self._reference_price(bars, quote)
        spread_cap, cap_meta = self._elastic_spread_cap(
            quote=quote,
            ref_price=ref_price,
            quote_age_seconds=age,
        )
        cap_meta["feed"] = str(quote.feed or "").lower()
        cap_meta["quote_age_seconds"] = age
        cap_meta["reference_price"] = ref_price
        mid_v = float(mid_price(quote.bid, quote.ask))
        spread_abs = float(quote.ask - quote.bid)
        self._log_spread_gate(
            symbol=symbol,
            bid=float(quote.bid),
            ask=float(quote.ask),
            mid=mid_v,
            spread_abs=spread_abs,
            spread_pct_value=float(sp),
            threshold=spread_cap,
            passed=sp <= spread_cap,
            metadata=cap_meta,
        )
        if sp > spread_cap:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.SPREAD_TOO_WIDE,
                    message=f"spread_pct {sp:.6f} exceeds cap {spread_cap:.6f} feed={quote.feed}",
                    symbol=symbol,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    spread_pct=sp,
                    spread_threshold_pct=spread_cap,
                    quote_age_seconds=age,
                    metadata={
                        "bid": quote.bid,
                        "ask": quote.ask,
                        "mid": mid_v,
                        "feed": quote.feed,
                        **cap_meta,
                    },
                ),
                quote=quote,
                log_event="strategy_skip_spread",
            )
            return EligibilityResult(
                False,
                f"spread_{sp:.5f}_above_{spread_cap:.5f}",
                SkipCodes.SPREAD_TOO_WIDE,
            )

        # Price filter on most recent close (or quote mid if no bars).
        if ref_price < self._settings.MIN_PRICE:
            self._emit_universe_skip(
                SkipReason(
                    code=SkipCodes.PRICE_BELOW_MIN,
                    message=(
                        f"ref_price={ref_price:.4f} below MIN_PRICE={self._settings.MIN_PRICE}"
                    ),
                    symbol=symbol,
                    bid=float(quote.bid),
                    ask=float(quote.ask),
                    price=ref_price,
                    quote_age_seconds=age,
                ),
                quote=quote,
            )
            return EligibilityResult(
                False,
                f"price_{ref_price:.4f}_below_{self._settings.MIN_PRICE}",
                SkipCodes.PRICE_BELOW_MIN,
            )

        # Average dollar volume over the past N bars (use up to 20).
        if bars is not None and not bars.empty and "volume" in bars.columns:
            tail = bars.tail(20)
            if not tail.empty:
                avg_dv = float((tail["close"] * tail["volume"]).mean())
                if avg_dv < self._settings.MIN_AVG_DOLLAR_VOLUME:
                    self._emit_universe_skip(
                        SkipReason(
                            code=SkipCodes.ADV_BELOW_MIN,
                            message=(
                                f"avg_dollar_volume={avg_dv:.0f} "
                                f"below MIN_AVG_DOLLAR_VOLUME={self._settings.MIN_AVG_DOLLAR_VOLUME:.0f}"
                            ),
                            symbol=symbol,
                            price=ref_price,
                            quote_age_seconds=age,
                            metadata={"avg_dollar_volume": avg_dv},
                        ),
                        quote=quote,
                    )
                    return EligibilityResult(
                        False,
                        f"adv_{avg_dv:.0f}_below_{self._settings.MIN_AVG_DOLLAR_VOLUME:.0f}",
                        SkipCodes.ADV_BELOW_MIN,
                    )

        return EligibilityResult(True, "ok", "ok")

    def _emit_universe_skip(
        self,
        sr: SkipReason,
        *,
        quote: Quote | None = None,
        log_event: str = "universe_entry_skip",
    ) -> None:
        if "decision_fn" not in sr.metadata:
            sr = replace(
                sr,
                metadata={"decision_fn": "UniverseFilter.is_eligible", **dict(sr.metadata)},
            )
        if quote is not None:
            patch: dict[str, Any] = {}
            if sr.quote_age_seconds is None:
                patch["quote_age_seconds"] = float(quote.age_seconds())
            if sr.spread_pct is None and quote.bid > 0 and quote.ask > quote.bid:
                try:
                    patch["spread_pct"] = float(spread_pct(quote.bid, quote.ask))
                except ValueError:
                    pass
            if patch:
                sr = replace(sr, **patch)
        ug = float(self._settings.SKIP_DIAGNOSTICS_UNIVERSE_LOG_THROTTLE_SECONDS)
        log_guard = None if ug <= 0 else ug
        actionable_for_discord = {
            SkipCodes.SPREAD_TOO_WIDE,
            SkipCodes.QUOTE_INVALID,
            SkipCodes.SIZE_ZERO,
            SkipCodes.STALE_BARS,
            SkipCodes.ORDER_REJECTED,
        }
        emit_skip_diagnostic(
            settings=self._settings,
            logger=self._log,
            log_event=log_event,
            sr=sr,
            discord_enqueue=self._discord_enqueue if sr.code in actionable_for_discord else None,
            throttle=self._skip_throttle,
            strategy_name=self._strategy_name,
            phase="universe",
            log_guard_seconds=log_guard,
        )
