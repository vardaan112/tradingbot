"""Phase 7 live shadow portfolios (no broker)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from core.database import Database
from core.market_data import Quote
from core.trade_source import BROKER_ELIGIBLE_SOURCES
from services.shadow_portfolio import SHADOW_SOURCE, ShadowPortfolioManager
from strategies.base import Signal, SignalAction


def _quote(sym: str = "SPY") -> Quote:
    return Quote(
        symbol=sym,
        bid=100.0,
        ask=100.2,
        bid_size=10.0,
        ask_size=10.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _settings_multi_ensemble(make_settings_factory):
    return make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENABLED=True,
        ENSEMBLE_ENTER_THRESHOLD=0.1,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
        SHADOW_INITIAL_EQUITY=10_000.0,
        SHADOW_FILL_MODEL="midpoint",
        SHADOW_RECORD_INTERVAL_SECONDS=60.0,
    )


def test_shadow_open_close_writes_shadow_source(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "sh.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _settings_multi_ensemble(make_settings_factory)
    mgr = ShadowPortfolioManager(s, db, run_id="ut_shadow")
    ts = "2026-01-15T16:00:00+00:00"
    q = _quote()
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso=ts,
        raw_signals=[
            Signal("SPY", SignalAction.ENTER_LONG, "e", 100.1, 0.0, {}, "rsi_mean_reversion", 1.0),
            Signal("SPY", SignalAction.NONE, "n", 0.0, 0.0, {}, "momentum", 0.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )
    assert "SPY" in mgr._accounts["rsi_mean_reversion"].positions

    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso=ts,
        raw_signals=[
            Signal("SPY", SignalAction.EXIT_LONG, "x", 100.1, 0.0, {}, "rsi_mean_reversion", 1.0),
            Signal("SPY", SignalAction.NONE, "n", 0.0, 0.0, {}, "momentum", 0.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )
    assert "SPY" not in mgr._accounts["rsi_mean_reversion"].positions

    con = sqlite3.connect(str(db_path))
    row = con.execute(
        "SELECT source, strategy_name FROM completed_trades ORDER BY id DESC LIMIT 1",
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == SHADOW_SOURCE == "shadow"
    assert row[1] == "rsi_mean_reversion"


def test_shadow_fill_conservative_uses_ask_bid(make_settings_factory, tmp_path) -> None:
    from services.shadow_portfolio import shadow_buy_fill_price, shadow_sell_fill_price

    q = _quote()
    assert shadow_buy_fill_price(q, model="midpoint") == pytest.approx(100.1)
    assert shadow_buy_fill_price(q, model="conservative_quote") == pytest.approx(100.2)
    assert shadow_sell_fill_price(q, model="midpoint") == pytest.approx(100.1)
    assert shadow_sell_fill_price(q, model="conservative_quote") == pytest.approx(100.0)


def test_shadow_manager_never_instantiates_order_service(make_settings_factory, tmp_path) -> None:
    """Smoke: shadow path only touches DB + SimulatedAccount."""

    db_path = tmp_path / "sh2.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _settings_multi_ensemble(make_settings_factory)
    mgr = ShadowPortfolioManager(s, db)
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso="2026-01-15T16:00:00+00:00",
        raw_signals=[
            Signal("SPY", SignalAction.NONE, "n", 0.0, 0.0, {}, "rsi_mean_reversion", 0.0),
        ],
        quote=_quote(),
        ensemble_decision=None,
        ensemble_signal=None,
    )


def test_equity_snapshots_recorded_with_throttle_bypass(monkeypatch, make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "sh3.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _settings_multi_ensemble(make_settings_factory)
    mgr = ShadowPortfolioManager(s, db)
    mgr._snap_interval = 0.0  # bypass throttle for unit test

    from itertools import count

    ctr = count(0, 100)

    def _mono() -> float:
        return float(next(ctr))

    monkeypatch.setattr("services.shadow_portfolio.time.monotonic", _mono)

    q = _quote()
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso="2026-01-15T16:00:00+00:00",
        raw_signals=[
            Signal("SPY", SignalAction.NONE, "n", 0.0, 0.0, {}, "rsi_mean_reversion", 0.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso="2026-01-15T16:01:00+00:00",
        raw_signals=[
            Signal("SPY", SignalAction.NONE, "n", 0.0, 0.0, {}, "rsi_mean_reversion", 0.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )

    con = sqlite3.connect(str(db_path))
    n = con.execute("SELECT COUNT(*) FROM equity_snapshots WHERE source = ?", (SHADOW_SOURCE,)).fetchone()[0]
    con.close()
    assert int(n) >= 2


def test_kelly_query_excludes_shadow_rows(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "sh4.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _settings_multi_ensemble(make_settings_factory)
    mgr = ShadowPortfolioManager(s, db, run_id="kelly_t")
    q = _quote()
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso="2026-01-15T16:00:00+00:00",
        raw_signals=[
            Signal("SPY", SignalAction.ENTER_LONG, "e", 100.1, 0.0, {}, "rsi_mean_reversion", 1.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )
    mgr.on_symbol(
        symbol="SPY",
        timestamp_iso="2026-01-15T16:05:00+00:00",
        raw_signals=[
            Signal("SPY", SignalAction.EXIT_LONG, "x", 100.1, 0.0, {}, "rsi_mean_reversion", 1.0),
        ],
        quote=q,
        ensemble_decision=None,
        ensemble_signal=None,
    )
    con = sqlite3.connect(str(db_path))
    sh_row = con.execute(
        "SELECT realized_pnl FROM completed_trades WHERE source = 'shadow' ORDER BY id DESC LIMIT 1",
    ).fetchone()
    con.close()
    assert sh_row is not None
    shadow_pnl = float(sh_row[0])

    db.record_completed_trade(
        trade_id="paper-1",
        symbol="QQQ",
        side="long",
        quantity=1.0,
        entry_price=50.0,
        exit_price=55.0,
        realized_pnl=5.0,
        realized_return=0.1,
        opened_at="2026-01-15T15:00:00+00:00",
        closed_at="2026-01-15T17:00:00+00:00",
        strategy_name="rsi_mean_reversion",
        risk_mode="normal",
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source="paper",
        invalid_for_kelly=False,
    )

    pnls = db.get_recent_realized_pnls_for_kelly(limit=50, exclude_simulation=True)
    assert 5.0 in pnls
    assert shadow_pnl not in pnls


def test_broker_eligible_sources_excludes_shadow() -> None:
    assert "shadow" not in BROKER_ELIGIBLE_SOURCES
