"""Multi-strategy evaluation (Phase 2): no risk, sizing, or orders."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Sequence

from config.constants import LOGGER_STRATEGY
from strategies.base import Signal, SignalAction, Strategy, StrategyContext

if TYPE_CHECKING:
    from config.settings import Settings
    from core.database import Database

_LOG = logging.getLogger(LOGGER_STRATEGY)


class StrategyEngine:
    """Evaluate many ``Strategy`` instances for one ``StrategyContext``."""

    def __init__(
        self,
        strategies: Sequence[Strategy],
        *,
        settings: "Settings",
        database: Optional["Database"] = None,
        signal_source: Optional[str] = None,
        replay_run_id: Optional[str] = None,
    ) -> None:
        self._strategies = list(strategies)
        self._settings = settings
        self._database = database
        self._signal_source = signal_source
        self._replay_run_id = replay_run_id

    @property
    def strategies(self) -> list[Strategy]:
        return list(self._strategies)

    def _normalize_signal(self, signal: Signal, default_name: str) -> Signal:
        name = signal.strategy_name if signal.strategy_name else default_name
        conf = float(signal.confidence)
        if conf < 0.0:
            conf = 0.0
        elif conf > 1.0:
            conf = 1.0
        if name == signal.strategy_name and conf == float(signal.confidence):
            return signal
        return replace(signal, strategy_name=name, confidence=conf)

    def _maybe_record_signal(self, signal: Signal, *, timestamp_iso: str) -> None:
        if self._database is None or self._signal_source is None:
            return
        if signal.action == SignalAction.NONE:
            return
        try:
            self._database.record_strategy_signal(
                source=self._signal_source,
                timestamp=timestamp_iso,
                symbol=signal.symbol,
                strategy_name=str(signal.strategy_name or ""),
                action=str(signal.action.value),
                run_id=self._replay_run_id,
                confidence=float(signal.confidence),
                reference_price=float(signal.reference_price),
                reason=signal.reason,
                metadata=dict(signal.metadata) if signal.metadata else None,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "event=strategy_signal_db_write_failed err=%s symbol=%s",
                exc,
                signal.symbol,
                extra={"symbol": signal.symbol},
            )

    def evaluate(self, ctx: StrategyContext) -> list[Signal]:
        """Run all strategies; normalize signals; optionally persist to SQLite."""

        if _LOG.isEnabledFor(logging.DEBUG) and len(self._strategies) > 1:
            _LOG.debug(
                "event=strategy_engine_evaluate mode=%s symbols_ctx=%s",
                self._settings.STRATEGY_RUN_MODE,
                ctx.symbol,
                extra={"symbol": ctx.symbol},
            )
        ts = ctx.now_utc.isoformat()
        out: list[Signal] = []
        for strat in self._strategies:
            default_name = strat.name
            try:
                for raw in strat.evaluate(ctx):
                    sig = self._normalize_signal(raw, default_name)
                    out.append(sig)
                    self._maybe_record_signal(sig, timestamp_iso=ts)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception(
                    "event=strategy_eval_failed strategy=%s symbol=%s err=%s",
                    default_name,
                    ctx.symbol,
                    exc,
                    extra={"symbol": ctx.symbol, "strategy": default_name},
                )
        return out


__all__ = ["StrategyEngine"]
