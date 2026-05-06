"""ATR-based position sizing with hard clamps.

Sizing pipeline:
1. risk_budget = equity * MAX_RISK_PER_TRADE_PCT
2. stop_distance = ATR * ATR_STOP_MULTIPLIER
3. raw_shares = floor(risk_budget / stop_distance)
4. clamp by:
     - MAX_EQUITY_USAGE_USD / entry_price
     - available buying power (compliance-aware)
     - MAX_GROSS_EXPOSURE_PCT
     - MAX_OPEN_POSITIONS
5. integer shares unless ENABLE_FRACTIONAL=True (default false)
6. final < 1 -> skip trade
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config.constants import LOGGER_RISK
from config.settings import Settings
from core.account import AccountSnapshot, PositionSnapshot

from .compliance import ComplianceAdapter
from .exposure import ExposureChecker


@dataclass(frozen=True)
class PositionSize:
    """Result of a sizing computation."""

    symbol: str
    shares: float  # 0 means "skip"
    notional: float
    entry_price: float
    stop_distance: float
    risk_budget: float
    rationale: str
    skipped_reason: Optional[str] = None


class PositionSizer:
    """Compute trade sizes with strict risk caps."""

    def __init__(
        self,
        settings: Settings,
        compliance: ComplianceAdapter,
        exposure: ExposureChecker,
    ) -> None:
        self._settings = settings
        self._compliance = compliance
        self._exposure = exposure
        self._log = logging.getLogger(LOGGER_RISK)

    def size(
        self,
        *,
        symbol: str,
        entry_price: float,
        atr: float,
        account: AccountSnapshot,
        positions: list[PositionSnapshot],
        bot_managed_notional: float,
    ) -> PositionSize:
        if entry_price <= 0:
            return self._skip(symbol, "non_positive_entry_price", entry_price=entry_price)
        if atr <= 0:
            return self._skip(symbol, "non_positive_atr", entry_price=entry_price)
        if account.equity <= 0:
            return self._skip(symbol, "non_positive_equity", entry_price=entry_price)

        risk_budget = account.equity * self._settings.MAX_RISK_PER_TRADE_PCT
        stop_distance = atr * self._settings.ATR_STOP_MULTIPLIER
        if stop_distance <= 0:
            return self._skip(symbol, "non_positive_stop_distance", entry_price=entry_price)

        raw_shares = risk_budget / stop_distance

        # USD cap
        usd_cap_shares = self._settings.MAX_EQUITY_USAGE_USD / entry_price

        # Buying power cap (compliance-aware)
        bp = self._compliance.buying_power(account)
        bp_shares = bp / entry_price if bp > 0 else 0.0

        # Gross exposure cap given current portfolio
        gross_cap_dollars = max(
            0.0,
            self._settings.MAX_GROSS_EXPOSURE_PCT * account.equity
            - sum(abs(p.market_value) for p in positions),
        )
        gross_cap_shares = gross_cap_dollars / entry_price if entry_price > 0 else 0.0

        # Bot-managed remaining USD cap
        remaining_bot = max(0.0, self._settings.MAX_EQUITY_USAGE_USD - bot_managed_notional)
        remaining_bot_shares = remaining_bot / entry_price if entry_price > 0 else 0.0

        candidates = [
            ("raw_atr", raw_shares),
            ("usd_cap", usd_cap_shares),
            ("buying_power", bp_shares),
            ("gross_exposure", gross_cap_shares),
            ("bot_managed_remaining", remaining_bot_shares),
        ]
        clamping_reason, clamped = min(candidates, key=lambda x: x[1])

        if not self._settings.ENABLE_FRACTIONAL:
            shares: float = float(math.floor(clamped))
        else:
            shares = max(0.0, math.floor(clamped * 1000.0) / 1000.0)

        if shares < 1.0:
            return self._skip(
                symbol,
                f"clamped_to_<1_via_{clamping_reason}",
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
            )

        # Final exposure check (defense in depth)
        proposed_notional = shares * entry_price
        decision = self._exposure.check(
            account=account,
            positions=positions,
            proposed_notional=proposed_notional,
            bot_managed_notional=bot_managed_notional,
        )
        if not decision.allowed:
            return self._skip(
                symbol,
                f"exposure_check:{decision.reason}",
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
            )

        return PositionSize(
            symbol=symbol,
            shares=shares,
            notional=proposed_notional,
            entry_price=entry_price,
            stop_distance=stop_distance,
            risk_budget=risk_budget,
            rationale=f"clamped_by={clamping_reason}",
        )

    def _skip(
        self,
        symbol: str,
        reason: str,
        *,
        entry_price: float = 0.0,
        stop_distance: float = 0.0,
        risk_budget: float = 0.0,
    ) -> PositionSize:
        self._log.info(
            "Sizing skip for %s: %s",
            symbol,
            reason,
            extra={"symbol": symbol},
        )
        return PositionSize(
            symbol=symbol,
            shares=0.0,
            notional=0.0,
            entry_price=entry_price,
            stop_distance=stop_distance,
            risk_budget=risk_budget,
            rationale=reason,
            skipped_reason=reason,
        )
