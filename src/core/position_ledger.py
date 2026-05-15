"""Reconcile broker holdings with the persisted bot ledger ("adopt orphans")."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from core.account import PositionSnapshot
from core.state_store import BotManagedPosition, RecoverySummary, StateStore

if TYPE_CHECKING:
    from core.database import Database


def reconcile_bot_ledger(
    *,
    state: StateStore,
    positions: list[PositionSnapshot],
    log: logging.Logger,
    adopt_trail: Callable[[str, float], None],
    eps_qty: float = 0.05,
    eps_avg_rel: float = 0.001,
    emit_position_adopted: bool = True,
) -> dict[str, BotManagedPosition]:
    """Persist long positions and invoke ``adopt_trail`` when data is new."""

    _ledger, _ = _reconcile_impl(
        state=state,
        positions=positions,
        log=log,
        adopt_trail=adopt_trail,
        eps_qty=eps_qty,
        eps_avg_rel=eps_avg_rel,
        emit_state_recovery=False,
        db=None,
        strategy_name="",
        emit_position_adopted_log=emit_position_adopted,
        execution_event_source=None,
    )
    return _ledger


def reconcile_open_positions(
    alpaca_positions: list[PositionSnapshot],
    *,
    state: StateStore,
    adopt_trail: Callable[[str, float], None],
    log: logging.Logger,
    db: Optional["Database"] = None,
    strategy_name: str = "strategy",
    eps_qty: float = 0.05,
    eps_avg_rel: float = 0.001,
    execution_event_source: Optional[str] = None,
) -> RecoverySummary:
    """Rebuild ledger from Alpaca holdings; log ``state_recovery`` for true orphans."""

    _, summary = _reconcile_impl(
        state=state,
        positions=alpaca_positions,
        log=log,
        adopt_trail=adopt_trail,
        eps_qty=eps_qty,
        eps_avg_rel=eps_avg_rel,
        emit_state_recovery=True,
        db=db,
        strategy_name=strategy_name,
        emit_position_adopted_log=True,
        execution_event_source=execution_event_source,
    )
    return summary


def _reconcile_impl(
    *,
    state: StateStore,
    positions: list[PositionSnapshot],
    log: logging.Logger,
    adopt_trail: Callable[[str, float], None],
    eps_qty: float,
    eps_avg_rel: float,
    emit_state_recovery: bool,
    db: Optional["Database"],
    strategy_name: str,
    emit_position_adopted_log: bool,
    execution_event_source: Optional[str] = None,
) -> tuple[dict[str, BotManagedPosition], RecoverySummary]:
    prev = state.load_bot_ledger()
    ledger: dict[str, BotManagedPosition] = {}
    broker_longs: dict[str, PositionSnapshot] = {}

    for p in positions:
        if p.side.lower() != "long" or abs(p.qty) <= 1e-9:
            continue
        sym = p.symbol.upper()
        broker_longs[sym] = p

    orphaned_recovered: list[str] = []

    for sym, pos in broker_longs.items():
        qty = float(pos.qty)
        avg = float(pos.avg_entry_price)
        old: Optional[BotManagedPosition] = prev.get(sym)
        changed = False
        if old is None:
            changed = True
        elif abs(old.qty - qty) > eps_qty:
            changed = True
        elif abs(old.avg_entry_price - avg) > max(avg * eps_avg_rel, 1e-4):
            changed = True

        is_orphan = old is None and changed

        if changed:
            if emit_position_adopted_log:
                log.info(
                    "event=position_adopted symbol=%s qty=%.6f avg=%.6f",
                    sym,
                    qty,
                    avg,
                    extra={"symbol": sym},
                )
            adopt_trail(sym, avg)

        if emit_state_recovery and is_orphan:
            log.info(
                "event=state_recovery symbol=%s adopted=True "
                "stop_loss_attached=True trailing_stop_attached=True",
                sym,
                extra={"symbol": sym, "strategy": strategy_name},
            )
            orphaned_recovered.append(sym)
            if db is not None:
                try:
                    db.record_execution_event(
                        event_type="state_recovery",
                        symbol=sym,
                        side="long",
                        client_order_id=None,
                        order_id=None,
                        status="adopted",
                        price=float(avg),
                        quantity=float(qty),
                        metadata={
                            "strategy": strategy_name,
                            "reason": "orphan_broker_position",
                        },
                        source=execution_event_source or "live",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("event=db_write_error kind=state_recovery symbol=%s err=%s", sym, exc)

        adopted = (old.adopted if old else False) or changed

        ledger[sym] = BotManagedPosition(
            symbol=sym,
            qty=qty,
            avg_entry_price=avg,
            updated_at=datetime.now(timezone.utc).isoformat(),
            adopted=adopted,
        )

    state.save_bot_ledger(ledger)
    return ledger, RecoverySummary(symbols_recovered=tuple(orphaned_recovered))


__all__ = ["reconcile_bot_ledger", "reconcile_open_positions"]
