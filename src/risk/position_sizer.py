"""ATR-based position sizing with hard clamps.

Sizing pipeline:
1. capital_base = settings.resolved_capital_base(account.equity)
   - settings.BOT_CAPITAL_BASE_USD if > 0
   - otherwise min(account.equity, MAX_EQUITY_USAGE_USD)
2. risk_budget = capital_base * MAX_RISK_PER_TRADE_PCT
3. stop_distance = ATR * ATR_STOP_MULTIPLIER
4. raw_shares = floor(risk_budget / stop_distance) (or fractional when enabled)
5. clamp by:
     - MAX_EQUITY_USAGE_USD / entry_price
     - available buying power (compliance-aware)
     - MAX_GROSS_EXPOSURE_PCT
     - MAX_OPEN_POSITIONS (via ExposureChecker)
     - bot-managed remaining notional
6. integer shares unless ENABLE_FRACTIONAL=True (default false)
7. final < 1 -> skip trade

The change from "risk a percent of total account equity" to "risk a percent
of the bot's allocated capital slice" matters when the brokerage account
holds capital outside the bot's mandate. The bot must risk only its slice.
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
    capital_base: float
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

        capital_base = self._settings.resolved_capital_base(account.equity)
        if capital_base <= 0:
            return self._skip(
                symbol,
                "non_positive_capital_base",
                entry_price=entry_price,
                capital_base=capital_base,
            )

        risk_budget = capital_base * self._settings.MAX_RISK_PER_TRADE_PCT
        stop_distance = atr * self._settings.ATR_STOP_MULTIPLIER
        if stop_distance <= 0:
            return self._skip(
                symbol,
                "non_positive_stop_distance",
                entry_price=entry_price,
                capital_base=capital_base,
            )

        raw_shares = risk_budget / stop_distance

        # USD cap (hard ceiling on bot-managed notional, independent of capital base)
        usd_cap_shares = self._settings.MAX_EQUITY_USAGE_USD / entry_price

        # Buying power cap (compliance-aware: never reads daytrading_buying_power
        # in intraday_margin mode).
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
            self._log_sizing(
                symbol=symbol,
                capital_base=capital_base,
                risk_budget=risk_budget,
                atr=atr,
                stop_distance=stop_distance,
                raw_shares=raw_shares,
                clamping_reason=clamping_reason,
                final_shares=shares,
                outcome="skip",
            )
            return self._skip(
                symbol,
                f"clamped_to_<1_via_{clamping_reason}",
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
                capital_base=capital_base,
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
            self._log_sizing(
                symbol=symbol,
                capital_base=capital_base,
                risk_budget=risk_budget,
                atr=atr,
                stop_distance=stop_distance,
                raw_shares=raw_shares,
                clamping_reason=f"exposure:{decision.reason}",
                final_shares=0.0,
                outcome="skip",
            )
            return self._skip(
                symbol,
                f"exposure_check:{decision.reason}",
                entry_price=entry_price,
                stop_distance=stop_distance,
                risk_budget=risk_budget,
                capital_base=capital_base,
            )

        self._log_sizing(
            symbol=symbol,
            capital_base=capital_base,
            risk_budget=risk_budget,
            atr=atr,
            stop_distance=stop_distance,
            raw_shares=raw_shares,
            clamping_reason=clamping_reason,
            final_shares=shares,
            outcome="ok",
        )
        return PositionSize(
            symbol=symbol,
            shares=shares,
            notional=proposed_notional,
            entry_price=entry_price,
            stop_distance=stop_distance,
            risk_budget=risk_budget,
            capital_base=capital_base,
            rationale=f"clamped_by={clamping_reason}",
        )

    def _log_sizing(
        self,
        *,
        symbol: str,
        capital_base: float,
        risk_budget: float,
        atr: float,
        stop_distance: float,
        raw_shares: float,
        clamping_reason: str,
        final_shares: float,
        outcome: str,
    ) -> None:
        self._log.info(
            "event=sizing outcome=%s symbol=%s capital_base=%.4f "
            "risk_budget=%.6f atr=%.6f stop_distance=%.6f raw_shares=%.4f "
            "clamping_reason=%s final_shares=%.4f",
            outcome,
            symbol,
            capital_base,
            risk_budget,
            atr,
            stop_distance,
            raw_shares,
            clamping_reason,
            final_shares,
            extra={"symbol": symbol},
        )

    def _skip(
        self,
        symbol: str,
        reason: str,
        *,
        entry_price: float = 0.0,
        stop_distance: float = 0.0,
        risk_budget: float = 0.0,
        capital_base: float = 0.0,
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
            capital_base=capital_base,
            rationale=reason,
            skipped_reason=reason,
        )
