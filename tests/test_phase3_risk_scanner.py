"""Phase 3: correlation gate, SPY flash monitor, scanner scheduling, ledger adopt."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from core.account import PositionSnapshot
from core.position_ledger import reconcile_bot_ledger
from core.state_store import StateStore
from risk.correlation import correlation_block_reason, pearson_from_closes
from risk.emergency import SpyFlashCrashMonitor
from strategies import scanner as scanner_mod
from strategies.scanner import UniverseScanRecord, persist_scan_record

_NY = ZoneInfo("America/New_York")


def _pos(sym: str, qty: float, avg: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=sym,
        qty=qty,
        avg_entry_price=avg,
        side="long",
        market_value=qty * avg,
        cost_basis=qty * avg,
        unrealized_pl=0.0,
        current_price=avg,
    )


def test_pearson_from_closes_insufficient_overlap_is_nan():
    a = pd.Series([1.0, 2.0], index=pd.date_range("2025-01-01", periods=2, freq="B"))
    b = pd.Series([1.1, 2.1], index=pd.date_range("2025-03-01", periods=2, freq="B"))
    assert pearson_from_closes(a, b) != pearson_from_closes(a, b)  # NaN


def test_correlation_breaker_blocks_highly_correlated_pair(make_settings_factory):
    idx = pd.date_range("2025-01-01", periods=40, freq="B")
    spy = pd.DataFrame(
        {"close": [100.0 + 0.1 * i + (0.03 * ((-1) ** i)) for i in range(len(idx))]},
        index=idx,
    )
    qqq = pd.DataFrame(
        {"close": [99.8 + 0.1 * i + (0.02 * ((-1) ** i)) for i in range(len(idx))]},
        index=idx,
    )

    class _BF:
        def fetch_bars(self, symbol: str, timeframe: str, *, lookback_bars: int):
            sym = symbol.upper()
            df = spy if sym == "SPY" else qqq
            return df.tail(min(lookback_bars, len(df))).copy()

    settings = make_settings_factory(
        CORRELATION_BREAKER_ENABLED=True,
        CORRELATION_LEADER_SYMBOL="SPY",
        CORRELATION_FOLLOWER_SYMBOLS="QQQ,IWM",
        CORRELATION_BREAKER_THRESHOLD=0.85,
        CORRELATION_LOOKBACK_CALENDAR_DAYS=30,
    )
    positions = [_pos("SPY", 10.0, 400.0)]
    reason = correlation_block_reason(
        settings,
        follower_symbol="QQQ",
        positions=positions,
        bar_fetcher=_BF(),  # type: ignore[arg-type]
    )
    assert reason == "correlation_breaker_leader_SPY_follower_QQQ"


def test_correlation_breaker_skips_without_leader_long(make_settings_factory):
    idx = pd.date_range("2025-01-01", periods=40, freq="B")

    class _BF:
        def fetch_bars(self, symbol: str, timeframe: str, *, lookback_bars: int):  # noqa: ARG002
            return pd.DataFrame({"close": list(range(len(idx)))}, index=idx)

    settings = make_settings_factory(CORRELATION_BREAKER_ENABLED=True)
    assert (
        correlation_block_reason(
            settings,
            follower_symbol="QQQ",
            positions=[_pos("IWM", 1.0, 50.0)],
            bar_fetcher=_BF(),  # type: ignore[arg-type]
        )
        is None
    )


def test_spy_flash_monitor_triggers_on_peak_to_last_drawdown():
    mon = SpyFlashCrashMonitor(symbol="SPY", drop_pct=0.03, window_minutes=15)
    base = datetime(2026, 1, 15, 19, 0, 0, tzinfo=timezone.utc)

    mon.observe(base, 100.0)
    mon.observe(base + timedelta(minutes=5), 102.0)
    mon.observe(base + timedelta(minutes=10), 98.44)  # 98.44/102 - 1 ~= -3.49%
    assert mon.triggered() is True

    mon.reset()
    mon.observe(base, 100.0)
    mon.observe(base + timedelta(minutes=5), 101.0)
    assert mon.triggered() is False


def test_maybe_refresh_before_gate_returns_none(monkeypatch, make_settings_factory, tmp_path):
    settings = make_settings_factory(DYNAMIC_UNIVERSE_ENABLED=True)

    def _now_et():
        return datetime(2026, 6, 1, 9, 30, 0, tzinfo=_NY)

    monkeypatch.setattr("utils.time_utils.now_eastern", _now_et)
    assert (
        scanner_mod.maybe_refresh_after_open(
            settings,
            trading=MagicMock(),
            bar_fetcher=MagicMock(),
            state=StateStore(tmp_path),
            session_is_open=True,
        )
        is None
    )


def test_maybe_refresh_reuses_cached_same_day(monkeypatch, make_settings_factory, tmp_path):
    settings = make_settings_factory(DYNAMIC_UNIVERSE_ENABLED=True)
    gate_ok = datetime(2026, 6, 1, 10, 0, 0, tzinfo=_NY)
    monkeypatch.setattr("utils.time_utils.now_eastern", lambda: gate_ok)

    state = StateStore(tmp_path)
    persist_scan_record(
        state,
        UniverseScanRecord(
            as_of_et_date=gate_ok.strftime("%Y-%m-%d"),
            symbols=["AAPL"],
            ts_utc="stub",
            reason="ok",
        ),
    )
    spy_refresh = MagicMock()
    monkeypatch.setattr(scanner_mod, "refresh_universe_now", spy_refresh)

    got = scanner_mod.maybe_refresh_after_open(
        settings,
        trading=MagicMock(),
        bar_fetcher=MagicMock(),
        state=state,
        session_is_open=True,
    )
    assert got is not None
    spy_refresh.assert_not_called()


def test_merge_tradeable_universe_includes_critical_symbols(make_settings_factory):
    s = make_settings_factory(
        DYNAMIC_UNIVERSE_ENABLED=True,
        SYMBOLS="SPY,DIA",
        CANARY_SYMBOL="XLF",
        BLACK_SWAN_SYMBOL="SPY",
        CORRELATION_LEADER_SYMBOL="SPY",
        CORRELATION_FOLLOWER_SYMBOLS="QQQ",
    )
    merged = scanner_mod.merge_tradeable_universe(s, scanned=["AMD", "INTC"])
    assert merged[0:2] == ["SPY", "DIA"]
    assert "XLF" in merged and "AMD" in merged and "QQQ" in merged


def test_symbols_for_strategy_ticks_prefers_scan_list_and_keeps_positions(make_settings_factory):
    s = make_settings_factory(DYNAMIC_UNIVERSE_ENABLED=True, SYMBOLS="SPY")
    out = scanner_mod.symbols_for_strategy_ticks(
        s,
        scanned=["NVDA"],
        broker_position_symbols={"META"},
    )
    assert out[0] == "NVDA" and "META" in out


def test_reconcile_adopts_once_then_stable(tmp_path):
    store = StateStore(tmp_path / "ledger")
    log = logging.getLogger("pytest.position_ledger")
    adopted: list[tuple[str, float]] = []

    def adopt(sym: str, avg: float) -> None:
        adopted.append((sym, avg))

    positions = [_pos("TSLA", 2.0, 205.25)]
    reconcile_bot_ledger(state=store, positions=positions, log=log, adopt_trail=adopt)
    assert adopted == [("TSLA", 205.25)]
    reconcile_bot_ledger(state=store, positions=positions, log=log, adopt_trail=adopt)
    assert adopted == [("TSLA", 205.25)]
