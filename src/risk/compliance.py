"""Regulatory mode adapter for the FINRA Rule 4210 transition (2026-06-04).

The adapter resolves which regulatory regime the bot should treat as authoritative
for the current trading day, and gates whether legacy day-trade throttles can
be relaxed via `POST_RULE4210_SCALING_ENABLED`.

Behavior summary:
- REGULATORY_MODE=pdt              : always pre-rule, conservative
- REGULATORY_MODE=intraday_margin  : always post-rule, buying-power centric
- REGULATORY_MODE=auto             : pdt before 2026-06-04, intraday_margin on/after
- POST_RULE4210_SCALING_ENABLED=false: never increase trading frequency or
                                       relax legacy day-trade throttles, even
                                       in intraday_margin mode.

The adapter also exposes a buying-power resolver that NEVER reads
`daytrading_buying_power`, `pattern_day_trader`, or `daytrade_count` after the
effective date, while still falling back gracefully on accounts where those
fields are absent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from config.constants import (
    LOGGER_RISK,
    REGULATORY_MODE_AUTO,
    REGULATORY_MODE_INTRADAY_MARGIN,
    REGULATORY_MODE_PDT,
)
from config.settings import Settings
from core.account import AccountSnapshot
from utils.time_utils import is_after_rule_4210_effective, today_eastern


class EffectiveRegulatoryMode:
    """String constants for the resolved (effective) regulatory mode."""

    PDT = REGULATORY_MODE_PDT
    INTRADAY_MARGIN = REGULATORY_MODE_INTRADAY_MARGIN


@dataclass(frozen=True)
class ComplianceDecision:
    """Outcome of the per-tick compliance check.

    `allow_new_entries` is the gate the orchestrator must respect.
    `effective_mode`   is the regulatory regime currently in force.
    """

    allow_new_entries: bool
    effective_mode: str
    reason: str
    scaling_relaxation_allowed: bool


class ComplianceAdapter:
    """Resolve regulatory mode and gate trading activity accordingly."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = logging.getLogger(LOGGER_RISK)

    # ------------------------------------------------------------------ mode

    def effective_mode(self, *, reference_date: Optional[date] = None) -> str:
        """Resolve user-selected mode against the rule 4210 effective date."""
        ref = reference_date or today_eastern()
        configured = self._settings.REGULATORY_MODE
        if configured == REGULATORY_MODE_AUTO:
            return (
                EffectiveRegulatoryMode.INTRADAY_MARGIN
                if is_after_rule_4210_effective(ref)
                else EffectiveRegulatoryMode.PDT
            )
        if configured == REGULATORY_MODE_PDT:
            return EffectiveRegulatoryMode.PDT
        if configured == REGULATORY_MODE_INTRADAY_MARGIN:
            return EffectiveRegulatoryMode.INTRADAY_MARGIN
        # Settings validation guarantees we never reach here, but be safe.
        self._log.warning("Unknown REGULATORY_MODE %r; defaulting to PDT", configured)
        return EffectiveRegulatoryMode.PDT

    # ---------------------------------------------------------- account read

    def buying_power(self, account: AccountSnapshot, *, reference_date: Optional[date] = None) -> float:
        """Return the buying-power figure to use for sizing decisions.

        Post-2026-06-04 (intraday_margin mode) we use plain `buying_power` only
        and explicitly avoid `daytrading_buying_power` since that field is
        being deprecated. In PDT mode we fall back to the more conservative
        of `regt_buying_power` and `buying_power` to avoid using day-trade BP.
        """
        mode = self.effective_mode(reference_date=reference_date)
        if mode == EffectiveRegulatoryMode.INTRADAY_MARGIN:
            return max(0.0, float(account.buying_power))
        candidates = [account.buying_power]
        if account.regt_buying_power > 0:
            candidates.append(account.regt_buying_power)
        return max(0.0, min(candidates))

    # -------------------------------------------------------------- decision

    def decide(
        self,
        account: AccountSnapshot,
        *,
        reference_date: Optional[date] = None,
    ) -> ComplianceDecision:
        """Return the per-tick compliance decision for new entries."""
        mode = self.effective_mode(reference_date=reference_date)

        # Always block if Alpaca explicitly flagged the account as blocked.
        if account.account_blocked or account.trading_blocked:
            return ComplianceDecision(
                allow_new_entries=False,
                effective_mode=mode,
                reason="account_blocked_by_broker",
                scaling_relaxation_allowed=False,
            )

        scaling_allowed = bool(self._settings.POST_RULE4210_SCALING_ENABLED)

        # PDT-era extra caution: if we still see a daytrade_count >= 3 and
        # POST_RULE4210_SCALING_ENABLED is false, gate further activity.
        if mode == EffectiveRegulatoryMode.PDT:
            if (
                account.daytrade_count is not None
                and account.daytrade_count >= 3
                and not scaling_allowed
            ):
                return ComplianceDecision(
                    allow_new_entries=False,
                    effective_mode=mode,
                    reason="pdt_daytrade_count_throttle",
                    scaling_relaxation_allowed=scaling_allowed,
                )
            return ComplianceDecision(
                allow_new_entries=True,
                effective_mode=mode,
                reason="pdt_ok",
                scaling_relaxation_allowed=scaling_allowed,
            )

        # Intraday margin mode: do NOT depend on PDT fields.
        if account.buying_power <= 0.0:
            return ComplianceDecision(
                allow_new_entries=False,
                effective_mode=mode,
                reason="zero_buying_power",
                scaling_relaxation_allowed=scaling_allowed,
            )

        return ComplianceDecision(
            allow_new_entries=True,
            effective_mode=mode,
            reason="intraday_margin_ok",
            scaling_relaxation_allowed=scaling_allowed,
        )
