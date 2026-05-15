"""Runtime trade ``source`` labels from Settings."""

from __future__ import annotations

from config.settings import Settings
from core.database import Database
from core.trade_source import runtime_trade_source


def _minimal_settings(**kwargs: object) -> Settings:
    """Build Settings without env; only overrides are validated fields we touch."""

    base = dict(
        ALPACA_API_KEY="x" * 16,
        ALPACA_API_SECRET="y" * 32,
        ALPACA_ENV="live",
        DRY_RUN=False,
    )
    base.update(kwargs)
    return Settings.model_construct(**base)  # type: ignore[arg-type]


def test_runtime_trade_source_dry_run() -> None:
    s = _minimal_settings(DRY_RUN=True, ALPACA_ENV="paper")
    assert runtime_trade_source(s) == "dry_run"


def test_runtime_trade_source_paper() -> None:
    s = _minimal_settings(DRY_RUN=False, ALPACA_ENV="paper")
    assert runtime_trade_source(s) == "paper"


def test_runtime_trade_source_live() -> None:
    s = _minimal_settings(DRY_RUN=False, ALPACA_ENV="live")
    assert runtime_trade_source(s) == "live"


def test_ml_and_kelly_exclude_non_broker_sources(tmp_path) -> None:
    db = Database(tmp_path / "src.sqlite3")
    db.init_schema()

    def _row(tid: str, src: str, pnl: float, closed: str) -> None:
        db.record_completed_trade(
            trade_id=tid,
            symbol="SPY",
            side="long",
            quantity=1.0,
            entry_price=100.0,
            exit_price=100.0 + pnl,
            realized_pnl=pnl,
            realized_return=pnl / 100.0,
            opened_at="2026-01-01T10:00:00+00:00",
            closed_at=closed,
            strategy_name="rsi_meanrev",
            risk_mode=None,
            regime_type=None,
            sentiment_score=None,
            sentiment_label=None,
            is_canary=0,
            source=src,
            entry_fill_source="broker_fill",
            exit_fill_source="broker_fill",
        )

    _row("live1", "live", 1.0, "2026-01-01T12:00:00+00:00")
    _row("paper1", "paper", 2.0, "2026-01-01T12:01:00+00:00")
    _row("dry1", "dry_run", 100.0, "2026-01-01T12:02:00+00:00")
    _row("sim1", "simulation", 200.0, "2026-01-01T12:03:00+00:00")
    _row("rep1", "replay", 300.0, "2026-01-01T12:04:00+00:00")
    _row("sh1", "shadow", 400.0, "2026-01-01T12:05:00+00:00")

    rows = db.get_ml_training_rows(limit=20, exclude_simulation=True)
    assert len(rows) == 2
    assert {float(r["realized_pnl"]) for r in rows} == {1.0, 2.0}

    assert db.count_completed_trades_ml_eligible(exclude_simulation=True) == 2

    pnls = db.get_recent_realized_pnls_for_kelly(limit=20, exclude_simulation=True)
    assert set(pnls) == {1.0, 2.0}

    rets = db.get_recent_realized_returns_for_kelly(limit=20, exclude_simulation=True)
    assert len(rets) == 2

    assert abs(db.sum_realized_pnl_all_live() - 3.0) < 1e-9
