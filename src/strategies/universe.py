"""Universe selection / per-symbol gating filters.

The orchestrator uses `UniverseFilter.is_eligible` to decide whether a symbol
should be considered for new entries this tick. Existing positions are still
managed (exits, stops) regardless of universe eligibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings
from core.market_data import Quote
from utils.price_utils import is_valid_quote, spread_pct


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason: str


class UniverseFilter:
    """Apply price, liquidity, and quote-quality filters to a symbol."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = logging.getLogger(LOGGER_STRATEGY)

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
            return EligibilityResult(False, "already_in_position")
        if has_open_order:
            return EligibilityResult(False, "open_order_present")

        if quote is None:
            return EligibilityResult(False, "no_quote")
        if not is_valid_quote(
            quote.bid,
            quote.ask,
            quote_age_seconds=quote.age_seconds(),
            max_age_seconds=self._settings.QUOTE_STALENESS_SECONDS,
        ):
            return EligibilityResult(False, "quote_invalid_or_stale")

        try:
            sp = spread_pct(quote.bid, quote.ask)
        except ValueError:
            return EligibilityResult(False, "spread_compute_failed")
        if sp > self._settings.SPREAD_FILTER_PCT:
            return EligibilityResult(False, f"spread_{sp:.5f}_above_{self._settings.SPREAD_FILTER_PCT:.5f}")

        # Price filter on most recent close (or quote mid if no bars).
        if bars is None or bars.empty:
            ref_price = (quote.bid + quote.ask) / 2.0
        else:
            ref_price = float(bars["close"].iloc[-1])
        if ref_price < self._settings.MIN_PRICE:
            return EligibilityResult(False, f"price_{ref_price:.4f}_below_{self._settings.MIN_PRICE}")

        # Average dollar volume over the past N bars (use up to 20).
        if bars is not None and not bars.empty and "volume" in bars.columns:
            tail = bars.tail(20)
            if not tail.empty:
                avg_dv = float((tail["close"] * tail["volume"]).mean())
                if avg_dv < self._settings.MIN_AVG_DOLLAR_VOLUME:
                    return EligibilityResult(
                        False,
                        f"adv_{avg_dv:.0f}_below_{self._settings.MIN_AVG_DOLLAR_VOLUME:.0f}",
                    )

        return EligibilityResult(True, "ok")
