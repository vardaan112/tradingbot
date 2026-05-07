"""Orchestrator guardrail tests for stale strategy bars."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from core.account import AccountSnapshot
from core.market_data import Quote
from services.orchestrator import Orchestrator


def _account(equity: float = 50_000.0) -> AccountSnapshot:
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


def _stale_bars() -> pd.DataFrame:
    end = datetime.now(timezone.utc) - timedelta(minutes=15)
    idx = pd.date_range(end=end, periods=220, freq="5min")
    close = pd.Series([100.0] * len(idx), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": pd.Series([1_000_000.0] * len(idx), index=idx),
        },
    )


@pytest.mark.asyncio
async def test_tick_emits_stale_bars_skip_before_strategy_eval(monkeypatch, make_settings_factory, tmp_path):
    settings = make_settings_factory(
        SYMBOLS="SPY",
        MAX_STRATEGY_BAR_AGE_SECONDS=60.0,
        STATE_DIR=str(tmp_path / "st"),
        LOG_DIR=str(tmp_path / "logs"),
        DATABASE_PATH=str(tmp_path / "db.sqlite"),
    )
    orch = Orchestrator(settings)

    quote = Quote(
        symbol="SPY",
        bid=100.0,
        ask=100.02,
        bid_size=10.0,
        ask_size=10.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

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

    emitted_codes: list[str] = []
    strategy_eval_calls: list[str] = []

    orch._latest_account = _account()
    orch._latest_positions = []
    orch._refresh_account_state = _noop_refresh  # type: ignore[method-assign]
    orch._market_clock = _Clock()  # type: ignore[assignment]
    orch._order_service = _OrderSvc()  # type: ignore[assignment]
    orch._quote_cache = _QuoteCache()  # type: ignore[assignment]
    orch._stream_health = SimpleNamespace(all_ok=True)
    orch._stream_runner = None
    orch._rest_quote = lambda _symbol: quote  # type: ignore[method-assign]
    orch._fetch_symbol_bars = lambda _symbol: _stale_bars()  # type: ignore[method-assign]
    orch._phase8_scheduled_jobs = _noop_phase8  # type: ignore[method-assign]
    orch._database.get_recent_completed_trades = lambda limit=50: []  # type: ignore[method-assign]
    orch._kill_switch.evaluate = lambda _eq: SimpleNamespace(latched=False)  # type: ignore[method-assign]
    orch._compliance.decide = lambda _acc: SimpleNamespace(  # type: ignore[method-assign]
        allow_new_entries=True,
        reason="ok",
        effective_mode="auto",
    )

    def _capture_skip(sr):
        emitted_codes.append(sr.code)

    monkeypatch.setattr(orch, "_emit_orchestrator_enter_skip", _capture_skip)
    monkeypatch.setattr(
        orch._strategy,
        "evaluate",
        lambda _ctx: strategy_eval_calls.append("called") or [],
    )

    await orch._tick()

    assert "STALE_BARS" in emitted_codes
    assert not strategy_eval_calls
