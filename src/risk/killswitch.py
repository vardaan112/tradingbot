"""Account-level kill switch with persistent latch.

Triggers when current equity drops `KILL_SWITCH_DRAWDOWN_PCT` or more below
the daily start-of-day equity baseline. Once latched, the latch survives
process restarts until manually reset by deleting the latch state file or
calling `KillSwitch.reset(force=True, operator_token=...)`.

Persisted latch path (under ``STATE_DIR``): ``kill_switch_state.json``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.constants import LOGGER_RISK
from core.state_store import DailyEquityRecord, KillSwitchRecord, StateStore
from utils.time_utils import now_eastern, today_eastern


@dataclass(frozen=True)
class KillSwitchDecision:
    """Result of a per-tick evaluation."""

    latched: bool
    daily_baseline: float
    current_equity: float
    drawdown_pct: float
    reason: str


class KillSwitch:
    """Persistent latching kill switch."""

    def __init__(
        self,
        state: StateStore,
        *,
        drawdown_pct: float,
    ) -> None:
        if not (0.0 < drawdown_pct < 1.0):
            raise ValueError("drawdown_pct must be in (0, 1)")
        self._state = state
        self._drawdown_pct = drawdown_pct
        self._log = logging.getLogger(LOGGER_RISK)

    # -------------------------------------------------------------- baseline

    def ensure_daily_baseline(self, current_equity: float) -> DailyEquityRecord:
        """Capture or restore today's start-of-day equity baseline.

        If the persisted baseline is for a prior date, it is replaced by the
        current equity. The kill-switch latch is intentionally NOT cleared
        when the day rolls over - latching is sticky until manual reset.
        """
        today = today_eastern().isoformat()
        existing = self._state.load_daily_equity()
        if existing is not None and existing.date == today:
            return existing
        record = DailyEquityRecord(date=today, equity=float(current_equity))
        self._state.save_daily_equity(record)
        self._log.info(
            "Captured daily start equity baseline: %.2f for %s",
            record.equity,
            record.date,
        )
        return record

    # ------------------------------------------------------------------ status

    def is_latched(self) -> bool:
        return bool(self._state.load_kill_switch().latched)

    def latch_record(self) -> KillSwitchRecord:
        return self._state.load_kill_switch()

    # -------------------------------------------------------------- evaluate

    def evaluate(self, current_equity: float) -> KillSwitchDecision:
        """Update internal state and return the resulting decision."""
        if self.is_latched():
            existing = self.latch_record()
            return KillSwitchDecision(
                latched=True,
                daily_baseline=existing.daily_baseline,
                current_equity=current_equity,
                drawdown_pct=_safe_drawdown(existing.daily_baseline, current_equity),
                reason="already_latched",
            )

        baseline = self.ensure_daily_baseline(current_equity)
        drawdown = _safe_drawdown(baseline.equity, current_equity)

        if drawdown >= self._drawdown_pct:
            self._latch(
                baseline_equity=baseline.equity,
                triggered_equity=current_equity,
                reason=f"drawdown {drawdown:.4%} >= threshold {self._drawdown_pct:.4%}",
            )
            return KillSwitchDecision(
                latched=True,
                daily_baseline=baseline.equity,
                current_equity=current_equity,
                drawdown_pct=drawdown,
                reason="latched_now",
            )

        return KillSwitchDecision(
            latched=False,
            daily_baseline=baseline.equity,
            current_equity=current_equity,
            drawdown_pct=drawdown,
            reason="ok",
        )

    # ------------------------------------------------------------------ latch

    def force_latch(self, reason: str, *, current_equity: float = 0.0) -> None:
        """Latch immediately (e.g. on websocket failure + ambiguous state)."""
        baseline = self._state.load_daily_equity()
        baseline_equity = baseline.equity if baseline else current_equity
        self._latch(baseline_equity=baseline_equity, triggered_equity=current_equity, reason=reason)

    def _latch(self, *, baseline_equity: float, triggered_equity: float, reason: str) -> None:
        record = KillSwitchRecord(
            latched=True,
            reason=reason,
            ts=datetime.now(timezone.utc).isoformat(),
            daily_baseline=baseline_equity,
            triggered_equity=triggered_equity,
        )
        self._state.save_kill_switch(record)
        self._log.critical(
            "KILL SWITCH LATCHED at %s. baseline=%.2f equity=%.2f reason=%s",
            now_eastern().isoformat(),
            baseline_equity,
            triggered_equity,
            reason,
        )

    # ------------------------------------------------------------------ reset

    def reset(self, *, force: bool, operator_token: Optional[str]) -> bool:
        """Manually reset the latch.

        Requires `force=True` AND a non-empty `operator_token` to make accidental
        resets unlikely. The token is logged as a coarse audit trail; do not
        store sensitive material in it.
        """
        if not force:
            return False
        if not operator_token or len(operator_token) < 6:
            self._log.warning("Refusing kill switch reset: operator_token too short.")
            return False
        self._state.save_kill_switch(KillSwitchRecord(latched=False))
        self._log.critical(
            "KILL SWITCH MANUALLY RESET by operator_token=%s at %s",
            operator_token[:6] + "***",
            now_eastern().isoformat(),
        )
        return True


def _safe_drawdown(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    if current >= baseline:
        return 0.0
    return (baseline - current) / baseline
