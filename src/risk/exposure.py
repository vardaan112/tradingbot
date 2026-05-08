"""Gross/net exposure and position-count limits.

`ExposureChecker` is a thin pure-logic helper used by the position sizer and
the orchestrator to confirm a candidate trade does not push the bot past its
configured limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.constants import LOGGER_RISK
from config.settings import Settings
from core.account import AccountSnapshot, PositionSnapshot


@dataclass(frozen=True)
class ExposureDecision:
    allowed: bool
    reason: str
    current_open_positions: int
    current_gross_pct: float
    proposed_gross_pct: float
    bot_managed_notional: float


class ExposureChecker:
    """Checks proposed trades against MAX_OPEN_POSITIONS, MAX_GROSS_EXPOSURE_PCT,
    and MAX_EQUITY_USAGE_USD limits."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = logging.getLogger(LOGGER_RISK)

    def check(
        self,
        *,
        account: AccountSnapshot,
        positions: list[PositionSnapshot],
        symbol: str | None = None,
        proposed_notional: float,
        bot_managed_notional: float,
    ) -> ExposureDecision:
        if account.equity <= 0:
            return ExposureDecision(
                allowed=False,
                reason="zero_equity",
                current_open_positions=len(positions),
                current_gross_pct=0.0,
                proposed_gross_pct=0.0,
                bot_managed_notional=bot_managed_notional,
            )

        current_gross = sum(abs(p.market_value) for p in positions)
        current_gross_pct = current_gross / account.equity if account.equity > 0 else 0.0

        proposed_gross = current_gross + max(0.0, proposed_notional)
        proposed_gross_pct = proposed_gross / account.equity

        sym_u = str(symbol or "").upper()
        adding_to_existing = any(
            p.symbol.upper() == sym_u and abs(float(p.qty)) > 1e-9
            for p in positions
        ) if sym_u else False

        # Position count check: adding to an existing symbol does not increase
        # the open-position count.
        if len(positions) >= self._settings.MAX_OPEN_POSITIONS and not adding_to_existing:
            return ExposureDecision(
                allowed=False,
                reason="max_open_positions_reached",
                current_open_positions=len(positions),
                current_gross_pct=current_gross_pct,
                proposed_gross_pct=proposed_gross_pct,
                bot_managed_notional=bot_managed_notional,
            )

        # Gross exposure %
        if proposed_gross_pct > self._settings.MAX_GROSS_EXPOSURE_PCT:
            return ExposureDecision(
                allowed=False,
                reason="max_gross_exposure_pct_exceeded",
                current_open_positions=len(positions),
                current_gross_pct=current_gross_pct,
                proposed_gross_pct=proposed_gross_pct,
                bot_managed_notional=bot_managed_notional,
            )

        # Bot-managed cap (USD)
        proposed_bot_notional = bot_managed_notional + max(0.0, proposed_notional)
        if proposed_bot_notional > self._settings.MAX_EQUITY_USAGE_USD:
            return ExposureDecision(
                allowed=False,
                reason="max_equity_usage_usd_exceeded",
                current_open_positions=len(positions),
                current_gross_pct=current_gross_pct,
                proposed_gross_pct=proposed_gross_pct,
                bot_managed_notional=bot_managed_notional,
            )

        return ExposureDecision(
            allowed=True,
            reason="ok",
            current_open_positions=len(positions),
            current_gross_pct=current_gross_pct,
            proposed_gross_pct=proposed_gross_pct,
            bot_managed_notional=bot_managed_notional,
        )
