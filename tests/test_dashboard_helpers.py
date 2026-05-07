"""Unit tests for ``utils.dashboard`` pure helpers (no Streamlit / Alpaca I/O)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from utils.dashboard import (
    TodayTradeRow,
    classify_latency_ms,
    connect_sqlite_readonly,
    infer_dashboard_risk_mode,
    query_latest_canary,
    query_today_trades,
    tail_log_lines_matching,
)


def test_classify_latency_ms() -> None:
    assert classify_latency_ms(None) == "unknown"
    assert classify_latency_ms(50.0) == "ok"
    assert classify_latency_ms(150.0) == "warn"
    assert classify_latency_ms(300.0) == "fail"


def test_infer_dashboard_risk_mode() -> None:
    assert infer_dashboard_risk_mode([])[0] == "Unknown"
    sample = [(-1.0, "normal"), (-2.0, None), (-0.5, "defensive")]
    assert infer_dashboard_risk_mode(sample)[0] == "Defensive"
    wins = [(1.0, None), (2.0, "normal")]
    assert infer_dashboard_risk_mode(wins)[0] == "Normal"
    assert infer_dashboard_risk_mode([(0.0, "defensive")])[0] == "Defensive"


def test_tail_log_lines_matching(tmp_path: Path) -> None:
    p = tmp_path / "x.log"
    p.write_text(
        "noise\n"
        '2026-01-01 | INFO | event=strategy_signal symbol=SPY action=none\n'
        "other\n",
        encoding="utf-8",
    )
    got = tail_log_lines_matching(p, needle="event=strategy_signal", max_lines=5)
    assert len(got) == 1
    assert "SPY" in got[0]


def test_sqlite_helpers(tmp_path: Path) -> None:
    dbf = tmp_path / "t.sqlite3"
    conn = sqlite3.connect(dbf)
    conn.executescript(
        """
        CREATE TABLE completed_trades (
          id INTEGER PRIMARY KEY,
          symbol TEXT, side TEXT, quantity REAL,
          closed_at TEXT NOT NULL, realized_pnl REAL, is_canary INTEGER DEFAULT 0,
          source TEXT DEFAULT 'live'
        );
        CREATE TABLE canary_results (
          id INTEGER PRIMARY KEY, success INTEGER, error TEXT, created_at TEXT
        );
        INSERT INTO completed_trades(symbol, side, quantity, closed_at, realized_pnl, is_canary)
        VALUES ('A','long',1,'2030-06-01T10:00:00', 5.0, 0);
        INSERT INTO canary_results(success, error, created_at) VALUES (1, NULL, '2030-06-01');
        """
    )
    conn.commit()
    conn.close()

    ro = connect_sqlite_readonly(dbf)
    assert ro is not None
    rows = query_today_trades(ro, "2030-06-01")
    assert len(rows) == 1
    assert isinstance(rows[0], TodayTradeRow)
    ok, err = query_latest_canary(ro)
    assert ok is True
    ro.close()
