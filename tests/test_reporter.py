"""Markdown daily reporter (SQLite-backed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from core.database import Database
from services.reporter import generate_daily_report, sentiment_accuracy_pct


def test_sentiment_accuracy_na_on_empty() -> None:
    txt, corr, elig = sentiment_accuracy_pct([])
    assert txt == "n/a"
    assert corr == 0 and elig == 0


def test_sentiment_accuracy_positive_win() -> None:
    class R:
        realized_pnl = 1.0
        sentiment_label = "positive"

    txt, corr, elig = sentiment_accuracy_pct([R()])
    assert elig == 1 and corr == 1
    assert txt != "n/a"


def test_daily_report_writes_markdown(tmp_path: Path, make_settings_factory) -> None:
    settings = make_settings_factory(
        REPORTS_DIR=tmp_path / "reports",
        DAILY_REPORT_ENABLED=True,
        DATABASE_PATH=tmp_path / "ledger.sqlite3",
        TEARSHEET_PRIMARY="sqlite",
    )
    db = Database(tmp_path / "ledger.sqlite3")
    db.init_schema()
    db.record_completed_trade(
        trade_id=None,
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=50.0,
        exit_price=51.0,
        realized_pnl=1.0,
        realized_return=0.02,
        opened_at=None,
        closed_at="2026-05-06T16:00:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode="normal",
        regime_type="r",
        sentiment_score=0.1,
        sentiment_label="neutral",
        is_canary=0,
        metadata=None,
    )
    path = generate_daily_report(
        settings,
        db,
        trading_day=date(2026, 5, 6),
    )
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "2026-05-06" in text
    assert "## operational / safety" in text.lower()
    path2 = generate_daily_report(
        settings,
        db,
        trading_day=date(2026, 5, 6),
    )
    assert path == path2
    assert path2.read_text(encoding="utf-8") == text


