"""Stress: black-swan detection, kill latch, emergency flatten path (mocked broker)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from core.account import PositionSnapshot
from core.market_data import Quote
from risk.emergency import PricePoint, SpyFlashCrashMonitor, detect_black_swan_drop
from risk.killswitch import KillSwitch
from services.orchestrator import Orchestrator
from strategies.base import Signal, SignalAction


def _pts_drop(*, start: float, end: float, minutes_step: int = 5) -> list[PricePoint]:
    base = datetime(2026, 3, 2, 14, 30, tzinfo=timezone.utc)
    out: list[PricePoint] = []
    n = int(15 // minutes_step) + 1
    for i in range(n):
        frac = i / max(1, n - 1)
        mid = start + (end - start) * frac
        out.append(PricePoint(t=base + timedelta(minutes=i * minutes_step), mid=mid))
    return out


def test_detect_black_swan_2_9_pct_no_trigger() -> None:
    pts = _pts_drop(start=100.0, end=97.15, minutes_step=3)  # ~2.85% off peak 100
    assert (
        detect_black_swan_drop("SPY", pts, threshold_pct=0.03, window_minutes=15) is False
    )


def test_detect_black_swan_3_1_pct_triggers() -> None:
    pts = _pts_drop(start=100.0, end=96.85, minutes_step=3)  # last well below 97.0
    assert (
        detect_black_swan_drop("SPY", pts, threshold_pct=0.03, window_minutes=15) is True
    )


def test_detect_black_swan_10_pct_triggers() -> None:
    pts = _pts_drop(start=450.0, end=405.0, minutes_step=1)
    assert (
        detect_black_swan_drop("SPY", pts, threshold_pct=0.03, window_minutes=15) is True
    )


def test_spy_monitor_10pct_in_15m_triggers() -> None:
    mon = SpyFlashCrashMonitor(symbol="SPY", drop_pct=0.03, window_minutes=15)
    t0 = datetime(2026, 4, 1, 16, 0, tzinfo=timezone.utc)
    mon.observe(t0, 450.0)
    for i in range(1, 16):
        mon.observe(t0 + timedelta(minutes=i), 450.0 - 3.0 * i)
    assert mon.triggered() is True


@pytest.mark.asyncio
async def test_enter_killed_mode_cancel_and_emergency_flatten(make_settings_factory) -> None:
    settings = make_settings_factory()
    orch = Orchestrator(settings)
    orch._order_service = MagicMock()
    orch._settings = settings
    orch._quote_cache = MagicMock()
    orch._bar_fetcher = MagicMock()
    orch._latest_positions = [
        PositionSnapshot(
            symbol="XLF",
            qty=3.0,
            avg_entry_price=40.0,
            side="long",
            market_value=120.0,
            cost_basis=120.0,
            unrealized_pl=0.0,
            current_price=40.0,
        )
    ]
    q = Quote(
        symbol="XLF",
        bid=39.9,
        ask=40.1,
        bid_size=100.0,
        ask_size=100.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )
    orch._quote_cache.get.return_value = q

    await Orchestrator._enter_killed_mode(orch)

    orch._order_service.cancel_all_open.assert_called_once()
    orch._order_service.submit_emergency_flatten.assert_called_once()


def test_kill_switch_latch_flash_crash_and_manual_reset(state_store) -> None:
    ks = KillSwitch(state_store, drawdown_pct=0.05)
    ks.ensure_daily_baseline(100_000.0)
    ks.force_latch("black_swan_simulation", current_equity=90_000.0)
    assert ks.is_latched() is True
    assert ks.reset(force=True, operator_token="short") is False
    assert ks.is_latched() is True
    assert ks.reset(force=True, operator_token="operator_reset_1") is True
    assert ks.is_latched() is False


@pytest.mark.asyncio
async def test_enter_long_rejected_when_kill_switch_latched(
    make_settings_factory,
) -> None:
    settings = make_settings_factory()
    orch = Orchestrator(settings)
    orch._kill_switch.force_latch("test_latch", current_equity=100.0)
    orch._log_strategy = MagicMock()
    sig = Signal(
        symbol="AAPL",
        action=SignalAction.ENTER_LONG,
        reason="test",
        reference_price=100.0,
        atr=1.0,
    )
    quote = Quote(
        symbol="AAPL",
        bid=99.5,
        ask=100.5,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )

    await orch._handle_signal(
        sig,
        quote=quote,
        can_open=True,
        can_exit=True,
        compliance_allow=True,
        eligible=True,
        eligibility_reason="ok",
        bot_managed_notional=0.0,
    )

    orch._log_strategy.info.assert_called()
    fmt, args = orch._log_strategy.info.call_args[0][0], orch._log_strategy.info.call_args[0][1:]
    line = fmt % args if args else fmt
    assert "kill switch latched" in line
