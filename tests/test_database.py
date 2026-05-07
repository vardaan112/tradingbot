"""SQLite persistence helpers."""

from __future__ import annotations

from pathlib import Path

from core.database import Database


def test_schema_init_inserts_runtime_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "t.sqlite"
    db = Database(db_path)
    db.init_schema()
    assert db_path.parent.is_dir()


def test_completed_trade_roundtrip_order(tmp_path: Path) -> None:
    db = Database(tmp_path / "x.sqlite3")
    db.init_schema()
    rid = db.record_completed_trade(
        trade_id=None,
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=101.0,
        realized_pnl=1.0,
        realized_return=0.01,
        opened_at="2026-01-01T10:00:00+00:00",
        closed_at="2026-01-01T11:00:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode="normal",
        regime_type="trend",
        sentiment_score=0.1,
        sentiment_label="neutral",
        is_canary=0,
        metadata={"x": 1},
    )
    assert rid is not None
    recent = db.get_recent_completed_trades(limit=5)
    assert len(recent) == 1
    assert recent[0].symbol == "SPY"
    assert recent[0].realized_pnl == 1.0


def test_canary_trade_excluded_from_recent(tmp_path: Path) -> None:
    db = Database(tmp_path / "y.sqlite3")
    db.init_schema()
    db.record_completed_trade(
        trade_id=None,
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=100.5,
        realized_pnl=0.5,
        realized_return=0.005,
        opened_at=None,
        closed_at="2026-01-02T15:00:00+00:00",
        strategy_name="canary",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=1,
        metadata=None,
    )
    db.record_completed_trade(
        trade_id=None,
        symbol="QQQ",
        side="long",
        quantity=1.0,
        entry_price=200.0,
        exit_price=199.0,
        realized_pnl=-1.0,
        realized_return=-0.005,
        opened_at=None,
        closed_at="2026-01-02T15:01:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode="defensive",
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        metadata=None,
    )
    recent = db.get_recent_completed_trades(limit=10)
    assert len(recent) == 1
    assert recent[0].symbol == "QQQ"


def test_execution_event_count(tmp_path: Path) -> None:
    from utils.time_utils import today_eastern

    db = Database(tmp_path / "z.sqlite3")
    db.init_schema()
    day = today_eastern().strftime("%Y-%m-%d")
    db.record_execution_event(
        event_type="order_chase_attempt",
        symbol="SPY",
        side="buy",
        client_order_id="c1",
        order_id=None,
        status="new",
        price=1.0,
        quantity=1.0,
        metadata=None,
        created_at=f"{day}T12:00:00",
    )
    n = db.count_execution_events(
        event_type="order_chase_attempt",
        trading_day_yyyy_mm_dd=day,
    )
    assert n == 1
