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


def test_kelly_returns_use_normalized_trade_return_and_exclude_bad_labels(tmp_path: Path) -> None:
    db = Database(tmp_path / "kelly.sqlite3")
    db.init_schema()
    db.record_completed_trade(
        trade_id="a",
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=110.0,
        realized_pnl=10.0,
        realized_return=0.10,
        opened_at=None,
        closed_at="2026-01-03T15:00:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        entry_fill_source="broker_fill",
        exit_fill_source="broker_fill",
    )
    db.record_completed_trade(
        trade_id="b",
        symbol="QQQ",
        side="long",
        quantity=10.0,
        entry_price=100.0,
        exit_price=101.0,
        realized_pnl=10.0,
        realized_return=0.01,
        opened_at=None,
        closed_at="2026-01-03T15:01:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        entry_fill_source="broker_fill",
        exit_fill_source="broker_fill",
    )
    db.record_completed_trade(
        trade_id="bad",
        symbol="IWM",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=99.0,
        realized_pnl=-1.0,
        realized_return=-0.01,
        opened_at=None,
        closed_at="2026-01-03T15:02:00+00:00",
        strategy_name="rsi_meanrev",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        metadata={"invalid_for_kelly": True},
        entry_fill_source="broker_fill",
        exit_fill_source="quote_mid_fallback",
        invalid_for_kelly=True,
    )

    returns = db.get_recent_realized_returns_for_kelly(limit=10)

    assert returns == [0.01, 0.10]
    assert db.get_recent_realized_pnls_for_kelly(limit=10) == [10.0, 10.0]


def test_phase1_research_schema_roundtrip_and_queries(tmp_path: Path) -> None:
    import json
    import sqlite3

    db = Database(tmp_path / "phase1.sqlite3")
    db.init_schema()

    conn = sqlite3.connect(db.path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='replay_runs'",
        ).fetchone()[0]
        assert int(n) == 1
    finally:
        conn.close()

    rid = db.create_replay_run(
        run_id="run-phase1-1",
        start_time="2026-01-01T14:30:00+00:00",
        end_time="2026-01-05T20:00:00+00:00",
        timeframe="5Min",
        symbols_json=json.dumps(["SPY", "QQQ"], sort_keys=True),
        strategies_json=json.dumps(["rsi_meanrev", "momentum"], sort_keys=True),
        mode="ensemble",
        initial_equity=100_000.0,
        lookback_days=5,
        data_feed="sip",
        settings_json=json.dumps({"DRY_RUN": True}, sort_keys=True),
        status="running",
    )
    assert rid is not None

    meta_sig = {"rsi": 28.5, "nested": {"a": 1, "b": 2}}
    sid = db.record_strategy_signal(
        source="replay",
        timestamp="2026-01-02T15:00:00+00:00",
        symbol="SPY",
        strategy_name="rsi_meanrev",
        action="enter_long",
        run_id="run-phase1-1",
        confidence=0.82,
        reference_price=450.25,
        reason="oversold",
        metadata=meta_sig,
    )
    assert sid is not None

    did = db.record_strategy_decision(
        source="replay",
        timestamp="2026-01-02T15:00:01+00:00",
        symbol="SPY",
        final_action="enter_long",
        run_id="run-phase1-1",
        decision_type="ensemble",
        weighted_score=0.71,
        threshold=0.55,
        contributing_signals_json=json.dumps([{"strategy": "rsi_meanrev", "action": "enter_long"}]),
        metadata={"votes": 2},
    )
    assert did is not None

    eid = db.record_equity_snapshot(
        source="replay",
        timestamp="2026-01-02T15:05:00+00:00",
        run_id="run-phase1-1",
        strategy_name="ensemble",
        cash=10_000.0,
        equity=100_050.0,
        realized_pnl=50.0,
        unrealized_pnl=25.0,
        gross_exposure=90_000.0,
        net_exposure=90_000.0,
        benchmark_equity=101.0,
        metadata={"note": "curve"},
    )
    assert eid is not None

    kid = db.record_skip_event(
        source="replay",
        timestamp="2026-01-02T16:00:00+00:00",
        run_id="run-phase1-1",
        symbol="QQQ",
        strategy_name="rsi_meanrev",
        phase="risk",
        skip_code="exposure_cap",
        message="blocked",
        metadata={"cap": 0.5},
    )
    assert kid is not None

    runs = db.query_replay_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-phase1-1"
    assert runs[0]["mode"] == "ensemble"

    sigs = db.query_strategy_signals(run_id="run-phase1-1", limit=50)
    assert len(sigs) == 1
    assert sigs[0]["action"] == "enter_long"
    parsed = json.loads(str(sigs[0]["metadata_json"] or "{}"))
    assert parsed == meta_sig

    decs = db.query_strategy_decisions(run_id="run-phase1-1", limit=50)
    assert len(decs) == 1
    assert decs[0]["final_action"] == "enter_long"

    snaps = db.query_equity_snapshots(run_id="run-phase1-1", limit=50)
    assert len(snaps) == 1
    assert abs(float(snaps[0]["equity"]) - 100_050.0) < 1e-6
    snap_meta = json.loads(str(snaps[0]["metadata_json"] or "{}"))
    assert snap_meta == {"note": "curve"}

    skips = db.query_skip_events(run_id="run-phase1-1", phase="risk", limit=50)
    assert len(skips) == 1
    skip_meta = json.loads(str(skips[0]["metadata_json"] or "{}"))
    assert skip_meta == {"cap": 0.5}

    assert db.finish_replay_run(run_id="run-phase1-1", status="completed", error=None) is True
    done = db.query_replay_runs(status="completed", limit=5)
    assert len(done) == 1
    assert done[0]["status"] == "completed"

    # completed_trades path unchanged
    tid = db.record_completed_trade(
        trade_id="p1t",
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
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source="live",
        replay_run_id="run-phase1-1",
    )
    assert tid is not None
    recent = db.get_recent_completed_trades(limit=5)
    assert any(r.symbol == "SPY" and r.trade_id == "p1t" for r in recent)


def test_phase1_migrations_add_research_tables_on_legacy_completed_trades_only(
    tmp_path: Path,
) -> None:
    import sqlite3

    p = tmp_path / "legacy_min.sqlite3"
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE completed_trades (id INTEGER PRIMARY KEY, closed_at TEXT NOT NULL, symbol TEXT)",
    )
    conn.commit()
    conn.close()

    Database(p).apply_migrations()

    conn = sqlite3.connect(p)
    try:
        names = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        }
        assert "replay_runs" in names
        assert "strategy_signals" in names
        assert "completed_trades" in names
    finally:
        conn.close()
