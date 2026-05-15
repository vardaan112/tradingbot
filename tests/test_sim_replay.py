"""Phase 4 sim / replay: synthetic bars, no Alpaca ``TradingClient``."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.base import Signal, SignalAction, Strategy, StrategyContext
from sim.benchmark import buy_hold_equity_curve
from sim.fill_model import FillModelParams, entry_long_fill_price, same_bar_stop_vs_target_long
from sim.replay_engine import HistoricalReplayEngine, resolve_replay_window
from sim.simulated_account import SimulatedAccount
from sim.simulated_broker import PendingFill, SimulatedBroker


def _ohlc(n: int, *, base: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=n, freq="1D", tz=timezone.utc)
    close = base + np.arange(n, dtype=float) * step
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )


class LenSignal(Strategy):
    """Emit enter/exit when ``len(ctx.bars)`` hits fixed thresholds (tests only)."""

    def __init__(self, enter_len: int, exit_len: int) -> None:
        self._e = int(enter_len)
        self._x = int(exit_len)

    def warmup_lookback(self) -> int:
        return 1

    def evaluate(self, ctx: StrategyContext):
        n = len(ctx.bars)
        sym = ctx.symbol
        px = float(ctx.bars["close"].iloc[-1])
        if n == self._e:
            yield Signal(
                symbol=sym,
                action=SignalAction.ENTER_LONG,
                reason="enter",
                reference_price=px,
                strategy_name=self.name,
            )
        if n == self._x:
            yield Signal(
                symbol=sym,
                action=SignalAction.EXIT_LONG,
                reason="exit",
                reference_price=px,
                strategy_name=self.name,
            )


def test_resolve_replay_window_lookback_days() -> None:
    end = datetime(2024, 6, 15, 16, 0, tzinfo=timezone.utc)
    start, end2 = resolve_replay_window(end=end, lookback_days=14)
    assert end2 == end
    assert (end - start).days == 14


def test_resolve_replay_window_rejects_start_and_lookback() -> None:
    end = datetime(2024, 6, 15, tzinfo=timezone.utc)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_replay_window(end=end, start=start, lookback_days=7)


def test_simulated_account_open_close_pnl() -> None:
    acct = SimulatedAccount(10_000.0)
    acct.open_long(symbol="AAA", quantity=10.0, price=100.0, fees_usd=0.0, ts_iso="t0")
    unreal, eq = acct.mark_to_market({"AAA": 110.0}, ts_iso="t1")
    assert unreal == pytest.approx(100.0)
    _, pnl = acct.close_long(symbol="AAA", quantity=10.0, price=110.0, fees_usd=0.0, ts_iso="t2")
    assert pnl == pytest.approx(100.0)
    assert acct.positions == {}


def test_entry_long_fill_price_next_bar_model() -> None:
    fp = FillModelParams(spread_pct=0.01, slippage_bps=10.0, fee_bps_per_side=0.0)
    o = 100.0
    px = entry_long_fill_price(o, params=fp)
    assert px == pytest.approx(o * (1.0 + 0.01 / 2 + 0.001))


def test_same_bar_stop_before_target() -> None:
    assert same_bar_stop_vs_target_long(
        bar_open=100,
        bar_high=120,
        bar_low=80,
        bar_close=100,
        stop_price=85,
        target_price=115,
    ) == "stop"


def test_next_bar_broker_fill_at_scheduled_open() -> None:
    acct = SimulatedAccount(50_000.0)
    br = SimulatedBroker(acct, fill_params=FillModelParams(0.0, 0.0, 0.0), prevent_same_bar_fills=True)
    br.schedule(
        PendingFill(
            execute_at_bar_index=3,
            symbol="AAA",
            action=SignalAction.ENTER_LONG,
            quantity=1.0,
            strategy_name="t",
        ),
    )
    ev = br.process_bar_open(bar_index=3, open_by_symbol={"AAA": 222.0}, ts_iso="t")
    assert ev[0]["kind"] == "fill"
    assert ev[0]["price"] == pytest.approx(222.0)
    assert "AAA" in acct.positions


def test_benchmark_buy_hold_curve() -> None:
    s = pd.Series([100.0, 110.0, 121.0])
    eq = buy_hold_equity_curve(s, initial_equity=10_000.0)
    assert float(eq.iloc[-1]) == pytest.approx(12_100.0)


def test_sim_sources_do_not_name_trading_client() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "sim"
    for p in sorted(root.glob("*.py")):
        text = p.read_text(encoding="utf-8")
        assert "TradingClient" not in text, p.name


def test_independent_replay_separate_equity_curves(make_settings_factory, monkeypatch, tmp_path) -> None:
    from sim import replay_engine as re_mod

    def fake_build(names, settings, **kwargs):
        key = names[0]
        if key == "rsi_mean_reversion":
            s = LenSignal(8, 18)
            s.name = "rsi_mean_reversion"
            return [s]
        if key == "momentum":
            s = LenSignal(10, 22)
            s.name = "momentum"
            return [s]
        raise AssertionError(key)

    monkeypatch.setattr(re_mod, "build_strategies", fake_build)

    n = 35
    bars = {"AAPL": _ohlc(n, base=50.0), "SPY": _ohlc(n, base=400.0)}
    settings = make_settings_factory(
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
    )
    eng = HistoricalReplayEngine(
        settings,
        symbols=["AAPL"],
        strategy_names=["rsi_mean_reversion", "momentum"],
        start=bars["AAPL"].index[0].to_pydatetime(),
        end=bars["AAPL"].index[-1].to_pydatetime(),
        timeframe="1Day",
        initial_equity=100_000.0,
        mode="independent",
        run_id="t_indep",
        output_dir=tmp_path,
        database=None,
        fill_params=FillModelParams(0.0, 0.0, 0.0),
        bars_by_symbol=bars,
    )
    res = eng.run()
    a = res.portfolios["ind::rsi_mean_reversion"].equity_curve
    b = res.portfolios["ind::momentum"].equity_curve
    assert not a.equals(b)
    merged = pd.read_csv(tmp_path / "equity_curve.csv")
    assert "ind__rsi_mean_reversion" in merged.columns
    assert "ind__momentum" in merged.columns


def test_replay_completed_trade_source_replay(make_settings_factory, monkeypatch, tmp_path) -> None:
    from sim import replay_engine as re_mod

    def fake_build(names, settings, **kwargs):
        s = LenSignal(6, 14)
        s.name = names[0]
        return [s]

    monkeypatch.setattr(re_mod, "build_strategies", fake_build)

    n = 30
    bars = {"AAPL": _ohlc(n, base=100.0), "SPY": _ohlc(n, base=400.0)}
    settings = make_settings_factory(
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
    )
    db_path = tmp_path / "r.sqlite3"
    from core.database import Database

    db = Database(db_path)
    db.init_schema()
    eng = HistoricalReplayEngine(
        settings,
        symbols=["AAPL"],
        strategy_names=["rsi_mean_reversion"],
        start=bars["AAPL"].index[0].to_pydatetime(),
        end=bars["AAPL"].index[-1].to_pydatetime(),
        timeframe="1Day",
        initial_equity=100_000.0,
        mode="independent",
        run_id="t_db",
        output_dir=tmp_path / "out",
        database=db,
        fill_params=FillModelParams(0.0, 0.0, 0.0),
        bars_by_symbol=bars,
    )
    eng.run()
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT source FROM completed_trades WHERE replay_run_id = ?",
        ("t_db",),
    ).fetchall()
    con.close()
    assert rows
    assert all(r[0] == "replay" for r in rows)


def test_replay_cli_end_now_resolves_window(tmp_path, monkeypatch) -> None:
    from sim import replay as replay_cli

    fixed = datetime(2025, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(replay_cli, "datetime", _FixedDT)

    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, settings, **kwargs):
            captured.update(kwargs)

        def run(self):
            from types import SimpleNamespace

            return SimpleNamespace(run_id="x", output_dir=tmp_path, portfolios={})

    monkeypatch.setattr(replay_cli, "HistoricalReplayEngine", FakeEngine)

    from tests.conftest import make_settings

    monkeypatch.setattr(replay_cli, "Settings", lambda: make_settings(LOG_DIR=str(tmp_path)))

    replay_cli.main(
        [
            "--symbols",
            "AAPL",
            "--strategies",
            "rsi_mean_reversion",
            "--lookback-days",
            "7",
            "--end",
            "now",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert captured["end"] == fixed
    assert captured["start"] == fixed - timedelta(days=7)


def test_replay_cli_rejects_start_with_lookback(tmp_path) -> None:
    from sim import replay as replay_cli

    code = replay_cli.main(
        [
            "--symbols",
            "AAPL",
            "--strategies",
            "rsi_mean_reversion",
            "--start",
            "2024-01-01T00:00:00+00:00",
            "--lookback-days",
            "5",
            "--end",
            "2024-06-01T00:00:00+00:00",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert code == 2


def test_replay_cli_initializes_sqlite_when_database_passed(tmp_path, monkeypatch) -> None:
    """``--database`` must create replay tables on a new file (regression: no such table)."""

    from sim import replay as replay_cli

    db_path = tmp_path / "fresh.sqlite3"

    class FakeEngine:
        def __init__(self, settings, **kwargs):
            self.kwargs = kwargs

        def run(self):
            from types import SimpleNamespace

            return SimpleNamespace(run_id="x", output_dir=tmp_path, portfolios={})

    monkeypatch.setattr(replay_cli, "HistoricalReplayEngine", FakeEngine)
    from tests.conftest import make_settings

    monkeypatch.setattr(replay_cli, "Settings", lambda: make_settings(LOG_DIR=str(tmp_path)))

    replay_cli.main(
        [
            "--symbols",
            "SPY",
            "--strategies",
            "rsi_mean_reversion",
            "--lookback-days",
            "1",
            "--end",
            "2025-03-01T12:00:00+00:00",
            "--output-dir",
            str(tmp_path),
            "--database",
            str(db_path),
        ],
    )
    assert db_path.is_file()
    con = sqlite3.connect(str(db_path))
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='replay_runs'",
    ).fetchone()
    con.close()
    assert row is not None


def test_fill_executes_next_bar_open_not_signal_bar(make_settings_factory, monkeypatch, tmp_path) -> None:
    """Signal at bar index i (len==8) fills at open of bar i+1."""

    from sim import replay_engine as re_mod

    def fake_build(names, settings, **kwargs):
        s = LenSignal(8, 16)
        s.name = "rsi_mean_reversion"
        return [s]

    monkeypatch.setattr(re_mod, "build_strategies", fake_build)

    n = 24
    o = np.arange(n, dtype=float) + 100.0
    idx = pd.date_range("2024-06-01", periods=n, freq="1D", tz=timezone.utc)
    bars = {
        "AAPL": pd.DataFrame({"open": o, "high": o + 1, "low": o - 1, "close": o, "volume": 1e6}, index=idx),
        "SPY": _ohlc(n, base=400.0),
    }
    settings = make_settings_factory(
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
    )
    eng = HistoricalReplayEngine(
        settings,
        symbols=["AAPL"],
        strategy_names=["rsi_mean_reversion"],
        start=idx[0].to_pydatetime(),
        end=idx[-1].to_pydatetime(),
        timeframe="1Day",
        initial_equity=100_000.0,
        mode="independent",
        run_id="t_next",
        output_dir=tmp_path,
        database=None,
        fill_params=FillModelParams(0.0, 0.0, 0.0),
        bars_by_symbol=bars,
    )
    eng.run()
    orders = pd.read_csv(tmp_path / "orders__ind__rsi_mean_reversion.csv")
    buys = orders[orders["side"] == "buy"]
    assert len(buys) == 1
    fill_ts = buys.iloc[0]["timestamp"]
    # bar index 7 has len 8 -> fill scheduled for index 8 -> timestamp idx[8]
    assert fill_ts == idx[8].isoformat()


def _minute_bars(idx: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(idx)
    close = np.linspace(100.0, 100.5, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )


def test_describe_bar_alignment_reports_overlap() -> None:
    from sim.replay_engine import align_symbol_frames, describe_bar_alignment

    idx_ok = pd.date_range("2024-06-01", periods=10, freq="1min", tz=timezone.utc)
    idx_off = pd.date_range("2024-12-01", periods=10, freq="1min", tz=timezone.utc)
    sym_frames = {
        "SPY": _minute_bars(idx_ok),
        "QQQ": _minute_bars(idx_ok),
        "MGRT": _minute_bars(idx_off),
    }
    common, _ = align_symbol_frames(sym_frames)
    assert len(common) == 0
    txt = describe_bar_alignment(sym_frames)
    assert "MGRT" in txt
    assert "ALL_SYMBOLS inner intersection: n=0" in txt


def test_greedy_drop_symbols_for_alignment() -> None:
    from sim.replay_engine import align_symbol_frames, greedy_drop_symbols_for_alignment

    idx_ok = pd.date_range("2024-06-01", periods=20, freq="1min", tz=timezone.utc)
    idx_off = pd.date_range("2024-12-01", periods=20, freq="1min", tz=timezone.utc)
    sym_frames = {
        "SPY": _minute_bars(idx_ok),
        "QQQ": _minute_bars(idx_ok),
        "MGRT": _minute_bars(idx_off),
    }
    assert len(align_symbol_frames(sym_frames)[0]) == 0
    aligned, dropped, common = greedy_drop_symbols_for_alignment(sym_frames, min_common=3)
    assert "MGRT" in dropped
    assert len(common) >= 3
    assert set(aligned.keys()) == {"SPY", "QQQ"}


def test_align_symbol_frames_master_clock_ffill_and_volume_zero() -> None:
    from sim.replay_engine import align_symbol_frames_master_clock

    idx = pd.date_range("2024-06-03 14:00", periods=10, freq="1min", tz=timezone.utc)
    spy = _minute_bars(idx)
    thin = _minute_bars(idx[[0, 2, 4, 6, 8]])  # sparse vs SPY master
    master, common, aligned = align_symbol_frames_master_clock(
        {"SPY": spy, "THIN": thin},
        benchmark_symbol="SPY",
        min_bars=5,
    )
    assert master == "SPY"
    assert len(common) == 10
    assert aligned["THIN"]["volume"].iloc[1] == 0.0
    assert aligned["THIN"]["close"].iloc[1] == pytest.approx(float(thin.loc[idx[0]]["close"]))
    assert not aligned["SPY"]["close"].isna().any()
    assert not aligned["THIN"]["close"].isna().any()


def test_build_default_replay_run_dirname_shape() -> None:
    from sim.replay import build_default_replay_run_dirname

    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 12, 17, 30, tzinfo=timezone.utc)
    name = build_default_replay_run_dirname(
        start=start,
        end=end,
        timeframe="5Min",
        mode="both",
        strategies=["rsi_mean_reversion", "momentum"],
        run_id="abcd1234-ef56-7890-abcd-ef1234567890",
    )
    assert name.startswith("replay__")
    assert "2026-04-01_to_2026-05-12" in name
    assert "5Min" in name
    assert "both" in name
    assert "rsi_mean_reversion-momentum" in name
    assert "abcd1234ef" in name  # first 10 alnum of run_id


def test_build_default_replay_run_dirname_same_calendar_day() -> None:
    from sim.replay import build_default_replay_run_dirname

    start = datetime(2026, 5, 14, 9, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 14, 16, 0, tzinfo=timezone.utc)
    name = build_default_replay_run_dirname(
        start=start,
        end=end,
        timeframe="1Min",
        mode="ensemble",
        strategies=["breakout"],
        run_id="run1",
    )
    assert "2026-05-14_0930to1600Z" in name
    acct = SimulatedAccount(50_000.0)
    br = SimulatedBroker(acct, fill_params=FillModelParams(0.0, 0.0, 0.0), prevent_same_bar_fills=True)
    br.schedule(
        PendingFill(
            execute_at_bar_index=2,
            symbol="AAA",
            action=SignalAction.ENTER_LONG,
            quantity=1.0,
            strategy_name="t",
        ),
    )
    skips: list[tuple[str, str, str]] = []

    def on_skip(sym: str, code: str, msg: str) -> None:
        skips.append((sym, code, msg))

    ev = br.process_bar_open(
        bar_index=2,
        open_by_symbol={"AAA": 100.0},
        ts_iso="t",
        on_skip=on_skip,
        volume_by_symbol={"AAA": 0.0},
    )
    assert ev[0]["kind"] == "skip"
    assert ev[0]["reason"] == "ghost_bar_zero_volume"
    assert "AAA" not in acct.positions
    assert any(c == "ghost_bar_zero_volume" for _, c, _ in skips)
