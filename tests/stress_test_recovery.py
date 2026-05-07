"""Stress: crash/restart recovery — adopt broker orphan into ledger (no live Alpaca)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.account import PositionSnapshot
from core.database import Database
from core.position_ledger import reconcile_open_positions as reconcile_from_ledger
from core.state_store import StateStore, reconcile_open_positions as reconcile_from_store


def _pos(sym: str, qty: float, avg: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=sym,
        qty=qty,
        avg_entry_price=avg,
        side="long",
        market_value=qty * avg,
        cost_basis=qty * avg,
        unrealized_pl=0.0,
        current_price=avg,
    )


@pytest.mark.parametrize("use_state_store_api", [False, True])
def test_reconcile_adopts_orphan_logs_state_recovery(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    use_state_store_api: bool,
) -> None:
    """Empty in-memory ledger + broker long -> adopt once, duplicate reconcile is idempotent."""

    state = StateStore(tmp_path / "runtime")
    db_path = tmp_path / "trades.sqlite3"
    db = Database(db_path)
    db.init_schema()
    db.record_execution_event(
        event_type="simulated_position_open",
        symbol="MSFT",
        side="long",
        client_order_id=None,
        order_id=None,
        status="open",
        price=300.12,
        quantity=10.0,
        metadata={"note": "seed_for_stress_recovery_test"},
    )
    adopt = MagicMock()
    tlog = logging.getLogger("stress_test.recovery")
    tlog.propagate = True

    broker = [_pos("MSFT", 10.0, 300.12)]
    impl = reconcile_from_store if use_state_store_api else reconcile_from_ledger

    with caplog.at_level(logging.INFO):
        summ1 = impl(
            broker,
            state=state,
            adopt_trail=adopt,
            log=tlog,
            db=db,
            strategy_name="stress_test_strategy",
        )
        assert adopt.call_count == 1
        adopt.reset_mock()

        impl(
            broker,
            state=state,
            adopt_trail=adopt,
            log=tlog,
            db=db,
            strategy_name="stress_test_strategy",
        )

    assert adopt.call_count == 0

    assert "MSFT" in summ1.symbols_recovered
    recap = "\n".join(r.getMessage() for r in caplog.records)
    assert "event=state_recovery symbol=MSFT adopted=True" in recap
    assert "stop_loss_attached=True trailing_stop_attached=True" in recap

    ledger = state.load_bot_ledger()
    assert "MSFT" in ledger and abs(float(ledger["MSFT"].qty) - 10.0) < 1e-6

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT event_type, symbol, status FROM execution_events
              WHERE event_type = 'state_recovery' AND symbol = 'MSFT'
            """
        ).fetchall()
        assert len(rows) >= 1
        assert rows[0][0] == "state_recovery"
