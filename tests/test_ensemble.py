"""Phase 6 weighted ensemble engine (unit tests, no broker)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from core.account import AccountSnapshot
from core.database import Database
from core.market_data import Quote
from services.ensemble import (
    WeightedEnsembleEngine,
    votes_to_contributing_json,
)
from services.orchestrator import Orchestrator
from sim.replay_engine import HistoricalReplayEngine
from strategies.base import Signal, SignalAction, Strategy, StrategyContext
from strategies.universe import EligibilityResult


def test_settings_allow_performance_weight_mode(make_settings_factory) -> None:
    s = make_settings_factory(ENSEMBLE_WEIGHT_MODE="performance")
    assert s.ENSEMBLE_WEIGHT_MODE == "performance"


def test_weighted_enter_threshold_not_met(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.55,
        ENSEMBLE_EXIT_THRESHOLD=0.5,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.ENTER_LONG, "a", 100.0, 0.0, {}, "rsi_mean_reversion", 0.5),
        Signal("SPY", SignalAction.NONE, "b", 0.0, 0.0, {}, "momentum", 0.0),
    ]
    d = eng.decide("SPY", sigs, has_position=False)
    assert d.final_action == SignalAction.NONE
    assert d.weighted_enter_score < 0.55


def test_weighted_enter_threshold_met(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.45,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.ENTER_LONG, "a", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
        Signal("SPY", SignalAction.NONE, "b", 0.0, 0.0, {}, "momentum", 0.0),
    ]
    d = eng.decide("SPY", sigs, has_position=False)
    assert d.final_action == SignalAction.ENTER_LONG


def test_weighted_exit_threshold(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_EXIT_THRESHOLD=0.6,
        ENSEMBLE_EXIT_POLICY="weighted",
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.EXIT_LONG, "x", 100.0, 0.0, {}, "rsi_mean_reversion", 0.5),
        Signal("SPY", SignalAction.NONE, "b", 0.0, 0.0, {}, "momentum", 0.0),
    ]
    d = eng.decide("SPY", sigs, has_position=True)
    assert d.final_action == SignalAction.NONE

    sigs2 = [
        Signal("SPY", SignalAction.EXIT_LONG, "x", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
        Signal("SPY", SignalAction.EXIT_LONG, "y", 100.0, 0.0, {}, "momentum", 1.0),
    ]
    d2 = eng.decide("SPY", sigs2, has_position=True)
    assert d2.final_action == SignalAction.EXIT_LONG


def test_min_agreeing_strategies_enter(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.1,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=2,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.ENTER_LONG, "a", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
        Signal("SPY", SignalAction.NONE, "b", 0.0, 0.0, {}, "momentum", 0.0),
    ]
    d = eng.decide("SPY", sigs, has_position=False)
    assert d.final_action == SignalAction.NONE


def test_emergency_exit_overrides(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.0,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.ENTER_LONG, "a", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
        Signal("SPY", SignalAction.EMERGENCY_EXIT_LONG, "e", 100.0, 0.0, {}, "momentum", 1.0),
    ]
    d = eng.decide("SPY", sigs, has_position=False)
    assert d.final_action == SignalAction.EMERGENCY_EXIT_LONG


def test_exit_policy_any(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_EXIT_POLICY="any",
        ENSEMBLE_EXIT_THRESHOLD=0.99,
    )
    eng = WeightedEnsembleEngine(s)
    sigs = [
        Signal("SPY", SignalAction.EXIT_LONG, "x", 100.0, 0.0, {}, "rsi_mean_reversion", 0.01),
        Signal("SPY", SignalAction.NONE, "b", 0.0, 0.0, {}, "momentum", 0.0),
    ]
    d = eng.decide("SPY", sigs, has_position=True)
    assert d.final_action == SignalAction.EXIT_LONG


def test_unknown_weight_uses_default_from_settings_dict(make_settings_factory) -> None:
    """``strategy_weights_dict`` sets missing actives to 1.0 before clamp/normalize."""

    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.8}',
        ENSEMBLE_MIN_WEIGHT=0.1,
        ENSEMBLE_MAX_WEIGHT=0.9,
    )
    eng = WeightedEnsembleEngine(s)
    w = eng._weights  # noqa: SLF001
    assert "momentum" in w
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_contributing_json_roundtrip(make_settings_factory) -> None:
    s = make_settings_factory(ACTIVE_STRATEGIES="rsi_mean_reversion")
    eng = WeightedEnsembleEngine(s)
    d = eng.decide(
        "QQQ",
        [Signal("QQQ", SignalAction.ENTER_LONG, "r", 50.0, 0.0, {}, "rsi_mean_reversion", 0.7)],
        has_position=False,
    )
    js = votes_to_contributing_json(d.contributing_votes)
    parsed = json.loads(js)
    assert len(parsed) == 1
    assert parsed[0]["action"] == "enter_long"


def test_to_signal_is_single_ensemble_signal(make_settings_factory) -> None:
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.2,
    )
    eng = WeightedEnsembleEngine(s)
    d = eng.decide(
        "SPY",
        [
            Signal("SPY", SignalAction.ENTER_LONG, "a", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
            Signal("SPY", SignalAction.ENTER_LONG, "b", 101.0, 0.0, {}, "momentum", 1.0),
        ],
        has_position=False,
    )
    sig = eng.to_signal(d)
    assert sig.strategy_name == "ensemble"
    assert sig.action == SignalAction.ENTER_LONG


class _RsiEnter(Strategy):
    name = "rsi_mean_reversion"

    def warmup_lookback(self) -> int:
        return 1

    def evaluate(self, ctx: StrategyContext):
        yield Signal(
            ctx.symbol,
            SignalAction.ENTER_LONG,
            "t",
            float(ctx.bars["close"].iloc[-1]),
            0.0,
            {},
            self.name,
            0.9,
        )


class _MomEnter(Strategy):
    name = "momentum"

    def warmup_lookback(self) -> int:
        return 1

    def evaluate(self, ctx: StrategyContext):
        yield Signal(
            ctx.symbol,
            SignalAction.ENTER_LONG,
            "t",
            float(ctx.bars["close"].iloc[-1]),
            0.0,
            {},
            self.name,
            0.9,
        )


class _RsiEnterLo(Strategy):
    name = "rsi_mean_reversion"

    def warmup_lookback(self) -> int:
        return 1

    def evaluate(self, ctx: StrategyContext):
        yield Signal(
            ctx.symbol,
            SignalAction.ENTER_LONG,
            "t",
            float(ctx.bars["close"].iloc[-1]),
            0.0,
            {},
            self.name,
            0.8,
        )


class _MomNone(Strategy):
    name = "momentum"

    def warmup_lookback(self) -> int:
        return 1

    def evaluate(self, ctx: StrategyContext):
        yield Signal(
            ctx.symbol,
            SignalAction.NONE,
            "n",
            float(ctx.bars["close"].iloc[-1]),
            0.0,
            {},
            self.name,
            0.0,
        )


def test_replay_ensemble_single_portfolio_equity(monkeypatch, make_settings_factory, tmp_path) -> None:
    from sim import replay_engine as re_mod

    def fake_build(names, settings, **kwargs):
        if len(names) == 2:
            return [_RsiEnter(), _MomEnter()]
        raise AssertionError(names)

    monkeypatch.setattr(re_mod, "build_strategies", fake_build)

    n = 25
    idx = pd.date_range("2024-06-01", periods=n, freq="1D", tz=timezone.utc)
    close = pd.Series(range(100, 100 + n), dtype=float, index=idx)
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=idx,
    )
    spy_b = bars.copy()
    for c in ("open", "high", "low", "close"):
        spy_b[c] = spy_b[c] * 4.0
    data = {"AAPL": bars.copy(), "SPY": spy_b}
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.3,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
    )
    eng = HistoricalReplayEngine(
        s,
        symbols=["AAPL"],
        strategy_names=["rsi_mean_reversion", "momentum"],
        start=idx[0].to_pydatetime(),
        end=idx[-1].to_pydatetime(),
        timeframe="1Day",
        initial_equity=100_000.0,
        mode="ensemble",
        run_id="ens_eq",
        output_dir=tmp_path,
        database=None,
        fill_params=__import__("sim.fill_model", fromlist=["FillModelParams"]).FillModelParams(0.0, 0.0, 0.0),
        bars_by_symbol=data,
    )
    res = eng.run()
    assert "ensemble" in res.portfolios
    assert len(res.portfolios) == 1


def test_strategy_decisions_contributing_json_persisted(monkeypatch, make_settings_factory, tmp_path) -> None:
    from sim import replay_engine as re_mod

    def fake_build(names, settings, **kwargs):
        if len(names) == 2:
            return [_RsiEnterLo(), _MomNone()]
        raise AssertionError(names)

    monkeypatch.setattr(re_mod, "build_strategies", fake_build)

    n = 12
    idx = pd.date_range("2024-06-01", periods=n, freq="1D", tz=timezone.utc)
    close = pd.Series(range(50, 50 + n), dtype=float, index=idx)
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6},
        index=idx,
    )
    spy_b = bars.copy()
    for c in ("open", "high", "low", "close"):
        spy_b[c] = spy_b[c] * 3.0
    data = {"AAPL": bars.copy(), "SPY": spy_b}
    db_path = tmp_path / "e.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.3,
        MAX_RISK_PER_TRADE_PCT=0.05,
        MAX_EQUITY_USAGE_USD=50_000.0,
        ENABLE_FRACTIONAL=True,
        DATABASE_PATH=str(db_path),
    )
    eng = HistoricalReplayEngine(
        s,
        symbols=["AAPL"],
        strategy_names=["rsi_mean_reversion", "momentum"],
        start=idx[0].to_pydatetime(),
        end=idx[-1].to_pydatetime(),
        timeframe="1Day",
        initial_equity=100_000.0,
        mode="ensemble",
        run_id="ens_db",
        output_dir=tmp_path / "out",
        database=db,
        fill_params=__import__("sim.fill_model", fromlist=["FillModelParams"]).FillModelParams(0.0, 0.0, 0.0),
        bars_by_symbol=data,
    )
    eng.run()
    con = sqlite3.connect(str(db_path))
    row = con.execute(
        "SELECT decision_type, contributing_signals_json FROM strategy_decisions WHERE run_id = ? LIMIT 1",
        ("ens_db",),
    ).fetchone()
    con.close()
    assert row is not None
    assert row[0] == "weighted_ensemble"
    assert row[1] is not None
    assert "rsi_mean_reversion" in row[1]


def _orch_account(equity: float = 50_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity * 2.0,
        regt_buying_power=equity * 2.0,
        portfolio_value=equity,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=2.0,
        status="ACTIVE",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
    )


def _orch_fresh_five_minute_bars(rows: int = 220) -> pd.DataFrame:
    end = datetime.now(timezone.utc) - timedelta(minutes=5)
    idx = pd.date_range(end=end, periods=rows, freq="5min")
    close = pd.Series([100.0] * len(idx), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": pd.Series([100_000.0] * len(idx), index=idx),
        },
    )


def test_orchestrator_ensemble_single_handle_signal_for_two_raw_enters(
    monkeypatch,
    make_settings_factory,
    tmp_path,
) -> None:
    """Two strategy ENTER_LONG votes must yield one `_handle_signal` (ensemble path)."""

    settings = make_settings_factory(
        SYMBOLS="SPY",
        STATE_DIR=str(tmp_path / "st"),
        LOG_DIR=str(tmp_path / "logs"),
        DATABASE_PATH=str(tmp_path / "db.sqlite"),
        ENSEMBLE_ENABLED=True,
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENTER_THRESHOLD=0.4,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=2,
        BLACK_SWAN_ENABLED=False,
    )
    orch = Orchestrator(settings, skip_startup_discord_embed=True)

    quote = Quote(
        symbol="SPY",
        bid=100.0,
        ask=100.02,
        bid_size=10.0,
        ask_size=10.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

    eval_calls = 0

    def _two_enters(_ctx: StrategyContext) -> list[Signal]:
        nonlocal eval_calls
        eval_calls += 1
        return [
            Signal("SPY", SignalAction.ENTER_LONG, "r", 100.0, 0.0, {}, "rsi_mean_reversion", 1.0),
            Signal("SPY", SignalAction.ENTER_LONG, "m", 100.0, 0.0, {}, "momentum", 1.0),
        ]

    monkeypatch.setattr(orch._strategy_engine, "evaluate", _two_enters)
    handle = AsyncMock()
    monkeypatch.setattr(orch, "_handle_signal", handle)

    class _OrderSvc:
        def cancel_stale(self, _seconds: float) -> None:
            return None

        def working_orders_snapshot(self) -> list:
            return []

    class _Clock:
        def get_session(self):
            return SimpleNamespace(is_open=True)

        def can_open_new_position(self, _session) -> bool:
            return True

        def can_exit_position(self, _session) -> bool:
            return True

    class _QuoteCache:
        feed = "iex"

        def get(self, _symbol: str) -> Quote:
            return quote

    async def _noop_refresh() -> None:
        return None

    async def _noop_phase8(_session) -> None:
        return None

    async def _no_forced_exit(_sym: str, _bars, _quote) -> bool:
        return False

    orch._latest_account = _orch_account()
    orch._latest_positions = []
    orch._refresh_account_state = _noop_refresh  # type: ignore[method-assign]
    orch._market_clock = _Clock()  # type: ignore[assignment]
    orch._order_service = _OrderSvc()  # type: ignore[assignment]
    orch._quote_cache = _QuoteCache()  # type: ignore[assignment]
    orch._stream_health = SimpleNamespace(all_ok=True)
    orch._stream_runner = None
    orch._rest_quote = lambda _symbol: quote  # type: ignore[method-assign]
    orch._fetch_symbol_bars = lambda _symbol: _orch_fresh_five_minute_bars()  # type: ignore[method-assign]
    orch._phase8_scheduled_jobs = _noop_phase8  # type: ignore[method-assign]
    orch._maybe_forced_execution_exit = _no_forced_exit  # type: ignore[method-assign]
    monkeypatch.setattr(orch._database, "get_recent_completed_trades", lambda limit=50: [])
    monkeypatch.setattr(orch._database, "record_strategy_decision", lambda **kwargs: None)
    orch._kill_switch.evaluate = lambda _eq: SimpleNamespace(latched=False)  # type: ignore[method-assign]
    orch._compliance.decide = lambda _acc: SimpleNamespace(  # type: ignore[method-assign]
        allow_new_entries=True,
        reason="ok",
        effective_mode="auto",
    )
    orch._universe.is_eligible = lambda _sym, **kwargs: EligibilityResult(True, "ok", "ok")  # type: ignore[method-assign]

    async def _run_tick() -> None:
        await orch._tick()

    asyncio.run(_run_tick())

    assert eval_calls == 1
    assert handle.call_count == 1
    sig = handle.call_args[0][0]
    assert sig.strategy_name == "ensemble"
    assert sig.action == SignalAction.ENTER_LONG
