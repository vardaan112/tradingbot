"""Sector exposure gate in orchestrator entry handling."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote
from services.orchestrator import Orchestrator
from strategies.base import Signal, SignalAction
from strategies.skip_diagnostics import SkipCodes


def _account(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity * 2,
        regt_buying_power=equity * 2,
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


def _pos(symbol: str, px: float = 100.0) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        qty=1.0,
        avg_entry_price=px,
        side="long",
        market_value=px,
        cost_basis=px,
        unrealized_pl=0.0,
        current_price=px,
    )


@pytest.mark.asyncio
async def test_sector_limit_skip_blocks_third_position(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(
        MAX_OPEN_POSITIONS_PER_SECTOR=2,
        SECTOR_MAP_JSON='{"AAPL":"Technology","NVDA":"Technology","SHOP":"Technology"}',
        ENABLE_DISCORD_BOT=False,
    )
    orch = Orchestrator(settings)
    orch._latest_account = _account()
    orch._latest_positions = [_pos("AAPL"), _pos("NVDA")]
    orch._stream_health = SimpleNamespace(all_ok=True)  # type: ignore[assignment]

    signal = Signal(
        symbol="SHOP",
        action=SignalAction.ENTER_LONG,
        reason="test",
        reference_price=100.0,
        atr=2.0,
        metadata={"rsi": 25.0, "adx": 18.0, "sma200": 101.0},
    )
    quote = Quote(
        symbol="SHOP",
        bid=99.95,
        ask=100.00,
        bid_size=10.0,
        ask_size=10.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

    captured: list[str] = []

    def _capture(sr):
        captured.append(sr.code)

    monkeypatch.setattr(orch, "_emit_orchestrator_enter_skip", _capture)

    def _should_not_size(**_kwargs):
        raise AssertionError("sizer should not be called when sector limit blocks")

    monkeypatch.setattr(orch._sizer, "size", _should_not_size)

    await orch._handle_signal(
        signal,
        quote=quote,
        can_open=True,
        can_exit=True,
        compliance_allow=True,
        eligible=True,
        eligibility_reason="ok",
        eligibility_code="ok",
        bot_managed_notional=0.0,
    )

    assert captured == [SkipCodes.SECTOR_LIMIT_FAIL]
