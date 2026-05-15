"""Phase 2: strategy registry and StrategyEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import pytest

from conftest import make_settings
from core.account import AccountSnapshot
from pydantic import ValidationError
from services.strategy_engine import StrategyEngine
from strategies.base import Signal, SignalAction, Strategy, StrategyContext
from strategies.registry import build_strategy, build_strategies, normalize_strategy_name
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy


def _minimal_settings():
    return make_settings(
        ACTIVE_STRATEGIES="rsi_mean_reversion",
        STRATEGY_RUN_MODE="single",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":1.0}',
    )


def _ctx(symbol: str = "SPY") -> StrategyContext:
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz=timezone.utc)
    bars = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1e6] * 5,
        },
        index=idx,
    )
    acct = AccountSnapshot(
        equity=100_000.0,
        last_equity=100_000.0,
        cash=100_000.0,
        buying_power=200_000.0,
        regt_buying_power=200_000.0,
        portfolio_value=100_000.0,
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
    return StrategyContext(
        symbol=symbol,
        bars=bars,
        quote=None,
        account=acct,
        positions_by_symbol={},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )


def test_registry_builds_rsi_by_name_and_alias() -> None:
    s = _minimal_settings()
    a = build_strategy("rsi_mean_reversion", s)
    b = build_strategy("rsi", s)
    c = build_strategy("rsi_strategy", s)
    assert isinstance(a, RSIMeanReversionStrategy)
    assert isinstance(b, RSIMeanReversionStrategy)
    assert isinstance(c, RSIMeanReversionStrategy)


def test_registry_unknown_raises() -> None:
    s = _minimal_settings()
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("momentum_breakout", s)


def test_normalize_strategy_name_aliases() -> None:
    assert normalize_strategy_name("RSI") == "rsi_mean_reversion"
    assert normalize_strategy_name("rsi_mean_reversion") == "rsi_mean_reversion"


def test_strategy_engine_attaches_strategy_name() -> None:
    class _NoName(Strategy):
        name = "attach_me"

        def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
            yield Signal(
                symbol=ctx.symbol,
                action=SignalAction.NONE,
                reason="test",
                strategy_name=None,
            )

    eng = StrategyEngine([_NoName()], settings=_minimal_settings())
    out = eng.evaluate(_ctx())
    assert len(out) == 1
    assert out[0].strategy_name == "attach_me"


def test_strategy_engine_two_dummy_strategies() -> None:
    class _A(Strategy):
        name = "dummy_a"

        def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
            yield Signal(symbol=ctx.symbol, action=SignalAction.NONE, reason="a")

    class _B(Strategy):
        name = "dummy_b"

        def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
            yield Signal(symbol=ctx.symbol, action=SignalAction.NONE, reason="b")

    eng = StrategyEngine([_A(), _B()], settings=_minimal_settings())
    out = eng.evaluate(_ctx("QQQ"))
    assert len(out) == 2
    names = {sig.strategy_name for sig in out}
    assert names == {"dummy_a", "dummy_b"}


def test_build_strategies_order_preserved() -> None:
    s = make_settings(
        ACTIVE_STRATEGIES="rsi,rsi_mean_reversion",
        STRATEGY_RUN_MODE="independent",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5}',
    )
    xs = build_strategies(["rsi_strategy", "rsi_mean_reversion"], s)
    assert len(xs) == 2
    assert all(isinstance(x, RSIMeanReversionStrategy) for x in xs)


def test_settings_single_mode_rejects_multiple_active() -> None:
    with pytest.raises(ValidationError):
        make_settings(
            ACTIVE_STRATEGIES="rsi_mean_reversion,rsi",
            STRATEGY_RUN_MODE="single",
        )
