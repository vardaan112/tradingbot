"""Tests for ``utils.dashboard`` replay / research SQL loaders (read-only helpers)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from core.database import Database
from utils.dashboard_helpers import (
    benchmark_return_from_equity_frame,
    build_replay_runs_summary_table,
    build_strategy_comparison_table,
    load_completed_trades_by_run,
    load_equity_snapshots,
    load_replay_runs,
    load_skip_events,
    load_strategy_decisions,
    load_strategy_signals,
    max_drawdown_from_equity,
)


def test_load_replay_runs_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.sqlite3"
    assert load_replay_runs(missing).empty


def test_load_replay_runs_no_table(tmp_path: Path) -> None:
    p = tmp_path / "bare.sqlite3"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE x (a INTEGER)")
    conn.commit()
    conn.close()
    assert load_replay_runs(p).empty


def test_load_replay_runs_and_equity_filters(tmp_path: Path) -> None:
    p = tmp_path / "r.sqlite3"
    db = Database(p)
    db.init_schema()
    assert db.create_replay_run(
        run_id="run_a",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-10T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["AAPL"]',
        strategies_json='["momentum"]',
        mode="independent",
        initial_equity=100_000.0,
        data_feed="iex",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    db.record_equity_snapshot(
        source="replay",
        timestamp="2024-01-02T00:00:00+00:00",
        run_id="run_a",
        strategy_name="ind::momentum",
        cash=50_000.0,
        equity=100_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        gross_exposure=50_000.0,
        net_exposure=50_000.0,
        benchmark_equity=100.0,
        metadata=None,
    )
    db.record_equity_snapshot(
        source="replay",
        timestamp="2024-01-03T00:00:00+00:00",
        run_id="run_a",
        strategy_name="ind::momentum",
        cash=49_000.0,
        equity=101_000.0,
        realized_pnl=0.0,
        unrealized_pnl=2000.0,
        gross_exposure=52_000.0,
        net_exposure=52_000.0,
        benchmark_equity=101.0,
        metadata=None,
    )
    db.record_equity_snapshot(
        source="live",
        timestamp="2024-01-03T00:00:00+00:00",
        run_id=None,
        strategy_name="live",
        cash=1.0,
        equity=1.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        gross_exposure=0.0,
        net_exposure=0.0,
        benchmark_equity=None,
        metadata=None,
    )

    runs = load_replay_runs(p, limit=50)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == "run_a"

    eq_all = load_equity_snapshots(p, run_id="run_a", source_scope="all")
    assert len(eq_all) == 2

    eq_rep = load_equity_snapshots(p, run_id="run_a", source_scope="replay")
    assert len(eq_rep) == 2
    assert set(eq_rep["source"].unique()) == {"replay"}

    eq_live = load_equity_snapshots(p, run_id=None, source_scope="live")
    assert len(eq_live) >= 1
    assert "live" in set(eq_live["source"].unique())


def test_strategy_comparison_groups_by_strategy(tmp_path: Path) -> None:
    p = tmp_path / "cmp.sqlite3"
    db = Database(p)
    db.init_schema()
    db.create_replay_run(
        run_id="run_b",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-10T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["AAPL"]',
        strategies_json='["momentum","rsi_mean_reversion"]',
        mode="independent",
        initial_equity=100_000.0,
        data_feed="iex",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    for ts, strat, eq, bench in [
        ("2024-01-02T00:00:00+00:00", "ind::momentum", 100_000.0, 100.0),
        ("2024-01-03T00:00:00+00:00", "ind::momentum", 110_000.0, 110.0),
        ("2024-01-02T00:00:00+00:00", "ind::rsi_mean_reversion", 100_000.0, 100.0),
        ("2024-01-03T00:00:00+00:00", "ind::rsi_mean_reversion", 105_000.0, 110.0),
    ]:
        db.record_equity_snapshot(
            source="replay",
            timestamp=ts,
            run_id="run_b",
            strategy_name=strat,
            cash=1.0,
            equity=eq,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            gross_exposure=1.0,
            net_exposure=1.0,
            benchmark_equity=bench,
            metadata=None,
        )
    db.record_completed_trade(
        trade_id="t1",
        symbol="AAPL",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=110.0,
        realized_pnl=10.0,
        realized_return=0.1,
        opened_at="2024-01-02T00:00:00+00:00",
        closed_at="2024-01-03T00:00:00+00:00",
        strategy_name="ind::momentum",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source="replay",
        replay_run_id="run_b",
    )
    db.record_completed_trade(
        trade_id="t2",
        symbol="AAPL",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=90.0,
        realized_pnl=-10.0,
        realized_return=-0.1,
        opened_at="2024-01-02T00:00:00+00:00",
        closed_at="2024-01-03T01:00:00+00:00",
        strategy_name="ind::rsi_mean_reversion",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source="replay",
        replay_run_id="run_b",
    )

    eq_df = load_equity_snapshots(p, run_id="run_b", source_scope="replay")
    tr_df = load_completed_trades_by_run(p, run_id="run_b", source_scope="replay")
    cmp = build_strategy_comparison_table(eq_df, tr_df, initial_equity=100_000.0)
    assert len(cmp) == 2
    mom = cmp.set_index("strategy_name").loc["ind::momentum"]
    rsi = cmp.set_index("strategy_name").loc["ind::rsi_mean_reversion"]
    assert float(mom["final_equity"]) == pytest.approx(110_000.0)
    assert float(rsi["final_equity"]) == pytest.approx(105_000.0)
    assert float(mom["realized_pnl_sum"]) == pytest.approx(10.0)
    assert float(rsi["realized_pnl_sum"]) == pytest.approx(-10.0)
    br = benchmark_return_from_equity_frame(eq_df)
    assert br is not None
    assert br == pytest.approx(0.1)


def test_max_drawdown_from_equity() -> None:
    s = pd.Series([100.0, 120.0, 90.0, 100.0])
    assert max_drawdown_from_equity(s) < 0


def test_completed_trades_source_and_run_filter(tmp_path: Path) -> None:
    p = tmp_path / "tr.sqlite3"
    db = Database(p)
    db.init_schema()
    db.create_replay_run(
        run_id="run_c",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-10T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["AAPL"]',
        strategies_json='["momentum"]',
        mode="independent",
        initial_equity=100_000.0,
        data_feed="iex",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    for tid, rid, src in (
        ("live1", None, "live"),
        ("rep1", "run_c", "replay"),
    ):
        db.record_completed_trade(
            trade_id=tid,
            symbol="AAPL",
            side="long",
            quantity=1.0,
            entry_price=10.0,
            exit_price=11.0,
            realized_pnl=1.0,
            realized_return=0.1,
            opened_at="2024-01-02T00:00:00+00:00",
            closed_at="2024-01-03T00:00:00+00:00",
            strategy_name="x",
            risk_mode=None,
            regime_type=None,
            sentiment_score=None,
            sentiment_label=None,
            is_canary=0,
            source=src,
            replay_run_id=rid,
        )
    only_rep = load_completed_trades_by_run(p, run_id="run_c", source_scope="replay")
    assert len(only_rep) == 1
    assert str(only_rep.iloc[0]["trade_id"]) == "rep1"
    live_only = load_completed_trades_by_run(p, run_id=None, source_scope="live")
    assert any(str(r.get("trade_id", "")) == "live1" for _, r in live_only.iterrows())


def test_signals_decisions_skips(tmp_path: Path) -> None:
    p = tmp_path / "sig.sqlite3"
    db = Database(p)
    db.init_schema()
    db.create_replay_run(
        run_id="run_d",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-10T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["AAPL"]',
        strategies_json='["momentum"]',
        mode="ensemble",
        initial_equity=100_000.0,
        data_feed="iex",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    db.record_strategy_signal(
        source="replay",
        timestamp="2024-01-02T00:00:00+00:00",
        symbol="AAPL",
        strategy_name="momentum",
        action="enter_long",
        run_id="run_d",
        confidence=0.9,
        reference_price=100.0,
        reason="test",
        metadata=None,
    )
    db.record_strategy_decision(
        source="replay",
        timestamp="2024-01-02T00:00:00+00:00",
        symbol="AAPL",
        final_action="enter_long,exit_long",
        run_id="run_d",
        decision_type="ensemble_equal_weight_stub",
        metadata={"k": 1},
    )
    db.record_skip_event(
        source="replay",
        timestamp="2024-01-02T00:00:00+00:00",
        run_id="run_d",
        symbol="AAPL",
        strategy_name="momentum",
        phase="replay",
        skip_code="zero_qty",
        message="sized zero",
        metadata=None,
    )
    sig = load_strategy_signals(p, run_id="run_d", strategy_name="momentum", symbol="AAPL")
    assert len(sig) == 1
    dec = load_strategy_decisions(p, run_id="run_d")
    assert len(dec) == 1
    sk = load_skip_events(p, run_id="run_d")
    assert len(sk) == 1


def test_build_replay_runs_summary_table(tmp_path: Path) -> None:
    p = tmp_path / "sum.sqlite3"
    db = Database(p)
    db.init_schema()
    db.create_replay_run(
        run_id="run_sum",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-05T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["SPY"]',
        strategies_json='["momentum"]',
        mode="independent",
        initial_equity=100_000.0,
        data_feed="sip",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    for ts, eq, bench in [
        ("2024-01-02T00:00:00+00:00", 100_000.0, 400.0),
        ("2024-01-04T00:00:00+00:00", 102_000.0, 408.0),
    ]:
        db.record_equity_snapshot(
            source="replay",
            timestamp=ts,
            run_id="run_sum",
            strategy_name="ind::momentum",
            cash=1.0,
            equity=eq,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            gross_exposure=1.0,
            net_exposure=1.0,
            benchmark_equity=bench,
            metadata=None,
        )
    runs = load_replay_runs(p)
    summ = build_replay_runs_summary_table(p, runs)
    assert len(summ) == 1
    assert summ.iloc[0]["Run"] == "run_sum"
    assert float(summ.iloc[0]["Final equity"]) == pytest.approx(102_000.0)
    assert float(summ.iloc[0]["SPY return"]) == pytest.approx(408.0 / 400.0 - 1.0)


def test_equity_snapshots_shadow_scope(tmp_path: Path) -> None:
    p = tmp_path / "shad.sqlite3"
    db = Database(p)
    db.init_schema()
    db.record_equity_snapshot(
        source="shadow",
        timestamp="2024-01-02T00:00:00+00:00",
        run_id=None,
        strategy_name="ind::a",
        cash=1.0,
        equity=50_000.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        gross_exposure=1.0,
        net_exposure=1.0,
        benchmark_equity=None,
        metadata=None,
    )
    sh = load_equity_snapshots(p, source_scope="shadow")
    assert len(sh) == 1
    assert sh.iloc[0]["source"] == "shadow"


def test_completed_trades_shadow_scope(tmp_path: Path) -> None:
    p = tmp_path / "sh2.sqlite3"
    db = Database(p)
    db.init_schema()
    db.record_completed_trade(
        trade_id="sh1",
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=400.0,
        exit_price=401.0,
        realized_pnl=1.0,
        realized_return=0.0,
        opened_at="2024-01-02T00:00:00+00:00",
        closed_at="2024-01-03T00:00:00+00:00",
        strategy_name="x",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source="shadow",
        replay_run_id=None,
    )
    df = load_completed_trades_by_run(p, run_id=None, source_scope="shadow")
    assert len(df) == 1


def test_strategy_comparison_has_sharpe_column(tmp_path: Path) -> None:
    p = tmp_path / "shp.sqlite3"
    db = Database(p)
    db.init_schema()
    db.create_replay_run(
        run_id="run_sh",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-10T00:00:00+00:00",
        timeframe="1Day",
        symbols_json='["AAPL"]',
        strategies_json='["momentum"]',
        mode="independent",
        initial_equity=100_000.0,
        data_feed="iex",
        benchmark_symbol="SPY",
        settings_json="{}",
        status="completed",
    )
    base = datetime(2024, 1, 2, tzinfo=UTC)
    for i in range(30):
        ts = (base + timedelta(hours=i)).isoformat()
        db.record_equity_snapshot(
            source="replay",
            timestamp=ts,
            run_id="run_sh",
            strategy_name="ind::momentum",
            cash=1.0,
            equity=100_000.0 + float(i) * 50.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            gross_exposure=1.0,
            net_exposure=1.0,
            benchmark_equity=100.0 + float(i) * 0.1,
            metadata=None,
        )
    eq_df = load_equity_snapshots(p, run_id="run_sh", source_scope="replay")
    cmp = build_strategy_comparison_table(eq_df, pd.DataFrame(), initial_equity=100_000.0)
    assert "sharpe_simple" in cmp.columns
    assert cmp.iloc[0]["sharpe_simple"] is not None
