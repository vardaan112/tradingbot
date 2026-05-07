"""Tests for ``scripts/replay_simulator.py`` (SQLite replay; no Alpaca orders)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import replay_simulator as rs  # noqa: E402

from core.database import Database  # noqa: E402


def test_pacing_seconds_per_day_formula() -> None:
    assert abs(rs.pacing_delay_seconds_per_day(100) - 10.0) < 1e-9
    assert abs(rs.pacing_delay_seconds_per_day(50) - 20.0) < 1e-9
    assert abs(rs.pacing_delay_seconds_per_day(10) - 100.0) < 1e-9


def test_replay_dry_run_inserts_zero(tmp_path) -> None:
    csvp = tmp_path / "bt.csv"
    pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "qty": 1.0,
                "entry_price": 100.0,
                "exit_price": 101.0,
                "net_pnl": 1.0,
                "entry_time": "2024-06-03T14:30:00+00:00",
                "exit_time": "2024-06-03T15:00:00+00:00",
                "parameter_set_id": "abc",
                "trailing_stop_active": False,
                "exit_reason": "rsi_exit",
                "regime_type": "Range",
            }
        ],
    ).to_csv(csvp, index=False)

    dbp = tmp_path / "replay.sqlite"
    db = Database(dbp)
    db.init_schema()
    rows = pd.read_csv(csvp).to_dict(orient="records")
    _rid, n = rs.replay(rows=rows, db=db, speed=1000.0, dry_run=True)
    assert n == 0
    conn = db._connect()
    try:
        cnt = int(conn.execute("SELECT COUNT(*) FROM completed_trades").fetchone()[0])
    finally:
        conn.close()
    assert cnt == 0


def test_replay_confirmed_writes_simulation_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rs.time, "sleep", lambda *_a, **_k: None)
    csvp = tmp_path / "bt.csv"
    pd.DataFrame(
        [
            {
                "symbol": "QQQ",
                "qty": 2.0,
                "entry_price": 200.0,
                "exit_price": 202.0,
                "net_pnl": 3.5,
                "gross_pnl": 4.0,
                "fees": 0.25,
                "slippage": 0.25,
                "entry_time": "2024-06-05T14:30:00+00:00",
                "exit_time": "2024-06-05T15:00:00+00:00",
                "parameter_set_id": "deadbeef",
                "trailing_stop_active": True,
                "exit_reason": "trailing_profit_breach",
                "regime_type": "Trending",
            }
        ],
    ).to_csv(csvp, index=False)

    dbp = tmp_path / "db.sqlite"
    db = Database(dbp)
    db.init_schema()
    db.record_completed_trade(
        trade_id="live_1",
        symbol="XLF",
        side="long",
        quantity=1.0,
        entry_price=30.0,
        exit_price=31.0,
        realized_pnl=1.0,
        realized_return=0.03,
        opened_at="2024-06-05T09:30:00+00:00",
        closed_at="2024-06-05T10:30:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        metadata=None,
        source="live",
    )

    rows = pd.read_csv(csvp).to_dict(orient="records")
    _rid2, n_ins = rs.replay(
        rows=rows,
        db=db,
        speed=10_000.0,
        dry_run=False,
        replay_run_id_override="treplay",
    )
    assert n_ins == 1

    conn = db._connect()
    try:
        sim = int(
            conn.execute(
                "SELECT COUNT(*) FROM completed_trades WHERE COALESCE(source,'live')='simulation'",
            ).fetchone()[0],
        )
        live = int(
            conn.execute(
                "SELECT COUNT(*) FROM completed_trades WHERE COALESCE(source,'live') IN ('live','paper')",
            ).fetchone()[0],
        )
        ev = int(
            conn.execute(
                "SELECT COUNT(*) FROM execution_events WHERE COALESCE(source,'live')='simulation'",
            ).fetchone()[0],
        )
    finally:
        conn.close()

    assert sim == 1
    assert live == 1
    assert ev >= 2

    summ = tmp_path / "replay_summary.md"
    rs.write_replay_summary(
        summ,
        replay_run_id="treplay",
        n_trades=n_ins,
        pnl_total=3.5,
        start_date="2024-06-05",
        end_date="2024-06-05",
        speed=500.0,
        db_path=dbp,
        dry_run=False,
    )
    assert summ.is_file()
    txt = summ.read_text(encoding="utf-8")
    assert "treplay" in txt
    assert "simulation" in txt
