"""Scale-in decision tests for RSI mean-reversion strategy and orchestrator handoff."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote
from risk.position_sizer import PositionSize
from services.orchestrator import Orchestrator
from strategies.base import Signal, SignalAction, StrategyContext
from strategies.filters import RegimeSnapshot
from strategies.rsi_strategy import RSIMeanReversionStrategy
from strategies.skip_diagnostics import SkipCodes


def _account(equity: float = 100_000.0) -> AccountSnapshot:
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


def _position(symbol: str = "TSLA", qty: float = 1.0, avg_entry: float = 100.0) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg_entry,
        side="long",
        market_value=qty * avg_entry,
        cost_basis=qty * avg_entry,
        unrealized_pl=0.0,
        current_price=avg_entry,
    )


def _quote(
    symbol: str = "TSLA",
    bid: float = 96.49,
    ask: float = 96.50,
    *,
    age_seconds: float = 0.0,
) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=25.0,
        ask_size=25.0,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        feed="iex",
    )


def _bars(n: int = 260, close_px: float = 96.5) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=10),
        periods=n,
        freq="5min",
    )
    close = pd.Series(np.full(n, close_px), index=idx)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": pd.Series(np.full(n, 1_000_000.0), index=idx),
        },
    )


def _regime() -> RegimeSnapshot:
    return RegimeSnapshot(
        adx=12.0,
        adx_length=14,
        sma200=101.0,
        sma_length=200,
        sma_slope=0.5,
        sma_slope_lookback=5,
        price_above_sma200=False,
        regime_type="Range",
        high_conviction=False,
        allow_rsi_long=True,
        reason="ok",
    )


def _patch_compute(monkeypatch: pytest.MonkeyPatch, strat: RSIMeanReversionStrategy, *, rsi_v: float, atr_v: float):
    def _fake_compute(bars: pd.DataFrame):
        return (
            pd.Series(np.full(len(bars), rsi_v), index=bars.index, dtype=float),
            pd.Series(np.full(len(bars), atr_v), index=bars.index, dtype=float),
        )

    monkeypatch.setattr(strat, "_compute", _fake_compute)


def _ctx(
    *,
    symbol: str = "TSLA",
    bars: pd.DataFrame,
    quote: Quote,
    position: PositionSnapshot | None = None,
    open_orders: set[str] | None = None,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        bars=bars,
        quote=quote,
        account=_account(),
        positions_by_symbol={symbol: position} if position is not None else {},
        open_order_symbols=open_orders or set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
    )


def test_no_position_entry_path_unchanged(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, ADX_RANGE_MAX=100.0)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    signals = list(strat.evaluate(_ctx(bars=_bars(close_px=96.5), quote=_quote(), position=None)))
    enters = [s for s in signals if s.action == SignalAction.ENTER_LONG]
    assert enters
    assert enters[0].metadata.get("signal_type") != "scale_in"


def test_scale_in_emitted_when_underwater_and_secondary_rsi(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(
        SCALE_IN_ENABLED=True,
        SCALE_IN_UNDERWATER_PCT=-0.03,
        SCALE_IN_RSI_THRESHOLD=25.0,
        SCALE_IN_ADD_QTY=1.0,
        MAX_BULLETS_PER_SYMBOL=2,
    )
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=24.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    pos = _position(qty=1.0, avg_entry=100.0)
    signals = list(strat.evaluate(_ctx(bars=_bars(close_px=96.5), quote=_quote(), position=pos)))
    enters = [s for s in signals if s.action == SignalAction.ENTER_LONG]
    assert enters
    assert enters[0].metadata.get("signal_type") == "scale_in"
    assert enters[0].metadata.get("bullet_number") == 2


def test_scale_in_skips_when_rsi_not_low_enough(
    monkeypatch: pytest.MonkeyPatch,
    make_settings_factory,
    caplog: pytest.LogCaptureFixture,
):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, SCALE_IN_RSI_THRESHOLD=25.0)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=26.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    with caplog.at_level("INFO", logger="tradingbot.strategy"):
        signals = list(strat.evaluate(_ctx(bars=_bars(close_px=96.5), quote=_quote(), position=_position())))
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)
    assert any("skip_code=scale_in_rsi_not_low_enough" in r.getMessage() for r in caplog.records)


def test_scale_in_skips_when_not_underwater(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, SCALE_IN_UNDERWATER_PCT=-0.03)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    signals = list(strat.evaluate(_ctx(bars=_bars(close_px=98.5), quote=_quote(bid=98.45, ask=98.55), position=_position())))
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)


def test_scale_in_skips_max_bullets_reached(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, SCALE_IN_ADD_QTY=1.0, MAX_BULLETS_PER_SYMBOL=2)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    signals = list(
        strat.evaluate(
            _ctx(
                bars=_bars(close_px=96.5),
                quote=_quote(),
                position=_position(qty=2.0, avg_entry=100.0),
            ),
        ),
    )
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)


def test_scale_in_bullet_count_unknown_fails_closed(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, ENABLE_FRACTIONAL=False, SCALE_IN_ADD_QTY=1.0)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    signals = list(
        strat.evaluate(
            _ctx(
                bars=_bars(close_px=96.5),
                quote=_quote(),
                position=_position(qty=1.5, avg_entry=100.0),
            ),
        ),
    )
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)


def test_scale_in_spread_filter_still_blocks(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, SPREAD_FILTER_MAX_PCT=0.02)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    wide = _quote(bid=95.0, ask=105.0)
    signals = list(strat.evaluate(_ctx(bars=_bars(close_px=96.5), quote=wide, position=_position())))
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)


def test_scale_in_disabled_no_signal(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=False)
    strat = RSIMeanReversionStrategy(settings)
    _patch_compute(monkeypatch, strat, rsi_v=20.0, atr_v=2.0)
    monkeypatch.setattr("strategies.rsi_strategy.compute_regime_snapshot", lambda **_kw: _regime())

    signals = list(strat.evaluate(_ctx(bars=_bars(close_px=96.5), quote=_quote(), position=_position())))
    assert not any(s.action == SignalAction.ENTER_LONG for s in signals)


@pytest.mark.asyncio
async def test_orchestrator_scale_in_risk_guard_blocks(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, ENABLE_DISCORD_BOT=False)
    orch = Orchestrator(settings)
    orch._latest_account = _account()
    orch._latest_positions = [_position("TSLA", qty=1.0, avg_entry=100.0)]
    orch._stream_health = SimpleNamespace(all_ok=True)  # type: ignore[assignment]

    signal = Signal(
        symbol="TSLA",
        action=SignalAction.ENTER_LONG,
        reason="scale-in test",
        reference_price=96.5,
        atr=2.0,
        metadata={
            "signal_type": "scale_in",
            "proposed_add_qty": 1.0,
            "scale_in_stop_distance": 5.0,
            "position_qty": 1.0,
            "bullet_number": 2,
            "max_bullets": 2,
            "rsi": 24.0,
            "adx": 12.0,
            "atr": 2.0,
            "last_close": 96.5,
            "spread_pct": 0.0002,
        },
    )
    quote = _quote()

    captured_codes: list[str] = []

    def _capture_skip(sr):
        captured_codes.append(sr.code)

    monkeypatch.setattr(orch, "_emit_orchestrator_enter_skip", _capture_skip)

    def _fake_size(**_kwargs):
        return PositionSize(
            symbol="TSLA",
            shares=1.0,
            notional=96.5,
            entry_price=96.5,
            stop_distance=5.0,
            risk_budget=5.0,
            capital_base=10_000.0,
            conviction_risk_multiplier=1.0,
            effective_risk_pct=0.01,
            rationale="test",
        )

    monkeypatch.setattr(orch._sizer, "size", _fake_size)
    monkeypatch.setattr(orch, "_maybe_correlation_block", lambda _sym: None)

    await orch._handle_signal(
        signal,
        quote=quote,
        can_open=True,
        can_exit=True,
        compliance_allow=True,
        eligible=False,
        eligibility_reason="already_in_position",
        eligibility_code=SkipCodes.ALREADY_IN_POSITION,
        bot_managed_notional=100.0,
    )

    assert SkipCodes.RISK_LIMIT_FAIL in captured_codes


@pytest.mark.asyncio
async def test_orchestrator_killswitch_blocks_scale_in(monkeypatch: pytest.MonkeyPatch, make_settings_factory):
    settings = make_settings_factory(SCALE_IN_ENABLED=True, ENABLE_DISCORD_BOT=False)
    orch = Orchestrator(settings)
    orch._latest_account = _account()
    orch._latest_positions = [_position("TSLA", qty=1.0, avg_entry=100.0)]
    orch._stream_health = SimpleNamespace(all_ok=True)  # type: ignore[assignment]

    signal = Signal(
        symbol="TSLA",
        action=SignalAction.ENTER_LONG,
        reason="scale-in test",
        reference_price=96.5,
        atr=2.0,
        metadata={"signal_type": "scale_in", "proposed_add_qty": 1.0},
    )
    quote = _quote()

    monkeypatch.setattr(orch._kill_switch, "is_latched", lambda: True)
    captured: list[str] = []
    monkeypatch.setattr(orch, "_emit_orchestrator_enter_skip", lambda sr: captured.append(sr.code))

    await orch._handle_signal(
        signal,
        quote=quote,
        can_open=True,
        can_exit=True,
        compliance_allow=True,
        eligible=False,
        eligibility_reason="already_in_position",
        eligibility_code=SkipCodes.ALREADY_IN_POSITION,
        bot_managed_notional=100.0,
    )
    assert SkipCodes.KILL_SWITCH_LATCHED in captured
