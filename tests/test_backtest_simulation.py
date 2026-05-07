"""Unit tests for ``backtest_simulation`` (offline, no Alpaca/Discord API)."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_simulation as bts  # noqa: E402


def test_simulated_book_buy_sell_cash() -> None:
    book = bts.SimulatedAccountBook(initial_equity=10_000.0, commission_bps=10.0)
    book.mark_prices({"SPY": 100.0})
    book.apply_buy(symbol="SPY", qty=10, px=100.0, ts=datetime.now(UTC), reason="t", trade_id=1)
    assert book._cash < 10_000.0
    assert "SPY" in book.positions
    book.mark_prices({"SPY": 105.0})
    book.apply_sell(symbol="SPY", qty=10, px=105.0)
    assert "SPY" not in book.positions
    assert book.realized_pnl != 0.0


def test_limit_buy_fills_only_when_low_touches() -> None:
    book = bts.SimulatedAccountBook(initial_equity=50_000.0, commission_bps=0.0)
    pend: list[bts.PendingLimit] = [
        bts.PendingLimit(
            order_id="o1",
            client_order_id="c1",
            symbol="SPY",
            side="buy",
            qty=1,
            limit_price=100.0,
            submitted_ts=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
            bars_alive=0,
            reason="test",
            kind="entry",
        ),
    ]
    rows: list[bts.SimulationOrderRecord] = []
    trades: list[dict] = []
    tc = [1]
    d = bts.MockDiscordSimulator()
    spy_row = pd.Series({"high": 101.0, "low": 99.5, "close": 100.5})

    process = bts.process_pending_orders
    process(
        pending=pend,
        ts=datetime(2024, 1, 1, 14, 5, tzinfo=UTC),
        rowslice={"SPY": spy_row},
        slippage_bps=0.0,
        order_timeout_bars=5,
        book=book,
        order_rows=rows,
        closed_trades=trades,
        discord=d,
        trade_counter=tc,
    )
    assert not pend


def test_limit_buy_no_fill_when_above_low() -> None:
    book = bts.SimulatedAccountBook(initial_equity=50_000.0, commission_bps=0.0)
    pend = [
        bts.PendingLimit(
            order_id="o1",
            client_order_id="c1",
            symbol="SPY",
            side="buy",
            qty=1,
            limit_price=100.0,
            submitted_ts=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
            bars_alive=0,
            reason="test",
            kind="entry",
        ),
    ]
    rows: list[bts.SimulationOrderRecord] = []
    trades: list[dict] = []
    process = bts.process_pending_orders
    process(
        pending=pend,
        ts=datetime(2024, 1, 1, 14, 5, tzinfo=UTC),
        rowslice={"SPY": pd.Series({"high": 104.0, "low": 100.5, "close": 102.0})},
        slippage_bps=0.0,
        order_timeout_bars=5,
        book=book,
        order_rows=rows,
        closed_trades=trades,
        discord=bts.MockDiscordSimulator(),
        trade_counter=[1],
    )
    assert pend and book.positions == {}


def test_limit_sell_fills_when_high_touches() -> None:
    book = bts.SimulatedAccountBook(initial_equity=50_000.0, commission_bps=0.0)
    book.apply_buy(symbol="SPY", qty=2, px=90.0, ts=datetime(2024, 1, 1, 13, 0, tzinfo=UTC), reason="in", trade_id=7)
    pend = [
        bts.PendingLimit(
            order_id="o2",
            client_order_id="c2",
            symbol="SPY",
            side="sell",
            qty=2,
            limit_price=100.0,
            submitted_ts=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
            bars_alive=0,
            reason="out",
            kind="exit",
        ),
    ]
    rows: list[bts.SimulationOrderRecord] = []
    closed: list[dict] = []
    bts.process_pending_orders(
        pending=pend,
        ts=datetime(2024, 1, 1, 14, 5, tzinfo=UTC),
        rowslice={"SPY": pd.Series({"high": 100.0, "low": 99.0, "close": 99.5})},
        slippage_bps=0.0,
        order_timeout_bars=5,
        book=book,
        order_rows=rows,
        closed_trades=closed,
        discord=bts.MockDiscordSimulator(),
        trade_counter=[10],
    )
    assert not book.positions
    assert closed


def test_summarize_max_drawdown() -> None:
    eq_df = pd.DataFrame(
        {
            "equity": [100.0, 110.0, 95.0, 120.0],
        },
    )
    meta = bts.summarize_performance(eq_df, [], 100.0, [datetime.now(UTC)] * 4)
    assert meta["max_drawdown_pct"] < 0


def test_sharpe_zero_variance_flat_equity() -> None:
    n = 80
    eq_df = pd.DataFrame({"equity": [100.0] * n})
    meta = bts.summarize_performance(eq_df, [], 100.0, [datetime.now(UTC)] * n)
    assert meta["sharpe_ratio"] == 0.0


def test_mock_discord_prints_block(capsys) -> None:
    d = bts.MockDiscordSimulator()
    d.print_embed("Heartbeat", {"equity": 1001.2})
    out = capsys.readouterr().out
    assert "SIMULATION EMBED" in out
    assert "Heartbeat" in out


def test_bars_slice_no_lookahead() -> None:
    idx = pd.date_range("2024-06-03", periods=5, freq="5min", tz=UTC)
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 100.0},
        index=idx,
    )
    ts = idx[2].to_pydatetime()
    w = df.loc[:ts]
    assert w.index.max() == idx[2]


def test_insufficient_warmup_raises(tmp_path: Path) -> None:
    p = tmp_path / "SPY.csv"
    idx = pd.date_range("2024-06-03", periods=20, freq="5min", tz=UTC)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 1e6,
        },
    ).to_csv(p, index=False)

    ns = argparse.Namespace(
        data_dir=tmp_path,
        symbols=["SPY"],
        start="2024-06-03",
        end="2024-06-10",
        initial_equity=10_000.0,
        output_dir=tmp_path / "out",
        commission_bps=0.0,
        slippage_bps=0.0,
        spread_bps=1.0,
        warmup_bars=250,
        save_trades=False,
        save_equity_curve=False,
    )
    with pytest.raises(RuntimeError, match="Insufficient warmup"):
        bts.run_backtest(ns)


def test_data_dir_must_exist(tmp_path: Path) -> None:
    ns = argparse.Namespace(
        data_dir=tmp_path / "nope",
        symbols=["SPY"],
        start="2024-06-03",
        end="2024-06-10",
        initial_equity=10_000.0,
        output_dir=tmp_path / "out",
        commission_bps=0.0,
        slippage_bps=0.0,
        spread_bps=1.0,
        warmup_bars=10,
        save_trades=False,
        save_equity_curve=False,
    )
    with pytest.raises(NotADirectoryError):
        bts.run_backtest(ns)
