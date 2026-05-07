"""Tearsheet helpers reading aggregate P&L rows (e.g. SQLite)."""

from __future__ import annotations

import pytest

from utils.tearsheet import build_summary_from_db_rows


def test_build_summary_from_db_rows_empty() -> None:
    s = build_summary_from_db_rows([])
    assert s["closed_trades"] == 0
    assert s["ok"] is True


def test_build_summary_excludes_none_pnl() -> None:
    rows = [
        {"realized_pnl": 2.0},
        {"realized_pnl": None},
        {"realized_pnl": -1.0},
    ]
    s = build_summary_from_db_rows(rows)
    assert s["closed_trades"] == 2
    assert s["net_pnl"] == pytest.approx(1.0)
