"""Phase 3 strategies: synthetic bar tests."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import make_settings
from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote
from strategies.base import SignalAction, StrategyContext
from strategies.breakout_strategy import BreakoutStrategy
from strategies.etf_rotation_strategy import ETFRotationStrategy
from strategies.momentum_strategy import MomentumTrendStrategy
from strategies.pairs_mean_reversion_strategy import PairsMeanReversionStrategy
from strategies.registry import build_strategy, supported_strategy_names
from strategies.regime_overlay import compute_risk_regime_overlay
from strategies.vwap_pullback_strategy import VWAPPullbackStrategy


def _account() -> AccountSnapshot:
    return AccountSnapshot(
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


def _quote(sym: str = "TEST") -> Quote:
    return Quote(
        symbol=sym,
        bid=99.9,
        ask=100.1,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _ohlc(n: int, *, close_slope: float = 0.0, vol: float = 1e6, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-03", periods=n, freq="5min", tz=timezone.utc)
    t = np.arange(n, dtype=float)
    close = start + close_slope * t
    c = pd.Series(close, index=idx)
    h = c + 0.2
    lo = c - 0.2
    o = c.shift(1).fillna(c.iloc[0])
    v = pd.Series(np.full(n, vol), index=idx)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v})


def _ctx(
    sym: str,
    bars: pd.DataFrame,
    *,
    pos: dict[str, PositionSnapshot] | None = None,
    all_bars: dict[str, pd.DataFrame] | None = None,
) -> StrategyContext:
    return StrategyContext(
        symbol=sym,
        bars=bars,
        quote=_quote(sym),
        account=_account(),
        positions_by_symbol=pos or {},
        open_order_symbols=set(),
        now_utc=datetime.now(timezone.utc),
        feed="iex",
        all_bars_by_symbol=all_bars or {},
    )


def _long(sym: str, *, entry: float = 100.0, qty: float = 1.0) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=sym.upper(),
        qty=qty,
        avg_entry_price=entry,
        side="long",
        market_value=entry * qty,
        cost_basis=entry * qty,
        unrealized_pl=0.0,
        current_price=entry,
    )


# --- momentum ---


def test_momentum_insufficient_bars():
    s = make_settings(MOMENTUM_ENABLED=True)
    st = MomentumTrendStrategy(s)
    bars = _ohlc(10)
    assert list(st.evaluate(_ctx("AAA", bars))) == []


def test_momentum_entry_and_exit():
    s = make_settings(
        MOMENTUM_ENABLED=True,
        MOMENTUM_MIN_RETURN_PCT=0.001,
        MOMENTUM_REQUIRE_ADX=False,
        MOMENTUM_VOLUME_FACTOR=0.5,
        MOMENTUM_MAX_HOLD_BARS=500,
    )
    st = MomentumTrendStrategy(s)
    n = 120
    bars = _ohlc(n, close_slope=0.15, vol=2e6)
    sig = list(st.evaluate(_ctx("MOM", bars)))
    assert any(x.action == SignalAction.ENTER_LONG for x in sig)

    # Exit: drop below exit SMA
    c = bars["close"].astype(float).copy()
    c.iloc[-3:] = c.iloc[-4] * 0.85
    bars2 = bars.copy()
    bars2["close"] = c
    bars2["low"] = c - 0.3
    bars2["high"] = c + 0.3
    sig2 = list(
        st.evaluate(
            _ctx("MOM", bars2, pos={"MOM": _long("MOM", entry=float(c.iloc[-5]))}),
        ),
    )
    assert any(x.action == SignalAction.EXIT_LONG for x in sig2)


# --- breakout ---


def test_breakout_insufficient_bars():
    s = make_settings(BREAKOUT_ENABLED=True)
    st = BreakoutStrategy(s)
    assert list(st.evaluate(_ctx("B", _ohlc(15)))) == []


def test_breakout_entry_and_failed_exit():
    s = make_settings(BREAKOUT_ENABLED=True, BREAKOUT_VOLUME_MULTIPLIER=1.0)
    st = BreakoutStrategy(s)
    n = 80
    base = _ohlc(n, close_slope=0.01, vol=1e6)
    c = base["close"].astype(float).copy()
    # Flat then spike through prior high
    c.iloc[:-5] = 100.0
    c.iloc[-5:] = np.linspace(100.0, 106.0, 5)
    v = base["volume"].astype(float).copy()
    v.iloc[-1] = 5e6
    bars = base.copy()
    bars["close"] = c
    bars["high"] = c + 0.1
    bars["low"] = c - 0.1
    bars["volume"] = v
    sig = list(st.evaluate(_ctx("BRK", bars)))
    assert any(x.action == SignalAction.ENTER_LONG for x in sig)

    # Failed breakout: collapse back under level
    c2 = c.copy()
    c2.iloc[-1] = 99.0
    bars2 = bars.copy()
    bars2["close"] = c2
    bars2["low"] = c2 - 0.2
    bars2["high"] = c2 + 0.2
    out = list(st.evaluate(_ctx("BRK", bars2, pos={"BRK": _long("BRK", entry=100.0)})))
    assert any("failed_breakout" in x.reason for x in out if x.action == SignalAction.EXIT_LONG)


# --- VWAP pullback ---


def test_vwap_pullback_insufficient():
    s = make_settings(VWAP_PULLBACK_ENABLED=True)
    st = VWAPPullbackStrategy(s)
    assert list(st.evaluate(_ctx("V", _ohlc(30)))) == []


def test_vwap_pullback_entry_and_mean_revert_exit():
    s = make_settings(
        VWAP_PULLBACK_ENABLED=True,
        VWAP_PULLBACK_RSI_MIN=1.0,
        VWAP_PULLBACK_RSI_MAX=99.0,
        VWAP_PULLBACK_ADX_MIN=0.0,
        VWAP_PULLBACK_ADX_MAX=100.0,
        VWAP_PULLBACK_MIN_TREND_SLOPE=-10.0,
        VWAP_PULLBACK_MAX_ZSCORE=5.0,
        VWAP_PULLBACK_MAX_DISTANCE_PCT=0.08,
        VWAP_PULLBACK_LENGTH=15,
        VWAP_PULLBACK_TREND_FAST_SMA=5,
        VWAP_PULLBACK_TREND_SLOW_SMA=15,
    )
    st = VWAPPullbackStrategy(s)
    n = 120
    idx = pd.date_range("2026-01-03", periods=n, freq="5min", tz=timezone.utc)
    t = np.arange(n, dtype=float)
    close = 100.0 + 0.12 * t
    c = pd.Series(close, index=idx)
    bars = pd.DataFrame(
        {
            "open": c.shift(1).fillna(c.iloc[0]),
            "high": c + 0.12,
            "low": c - 0.12,
            "close": c,
            "volume": pd.Series(np.full(n, 3e6), index=idx),
        }
    )
    bars2 = bars.copy()
    c2 = bars2["close"].astype(float)
    # Mild pullback: still above slow SMA, close near VWAP band
    c2.iloc[-3:] = c2.iloc[-4] * 0.9995
    bars2["close"] = c2
    bars2["low"] = c2 - 0.15
    bars2["high"] = c2 + 0.15
    ent = list(st.evaluate(_ctx("VW", bars2)))
    assert any(x.action == SignalAction.ENTER_LONG for x in ent)

    c3 = c2.copy()
    c3.iloc[-1] = float(c.iloc[-1]) + 1.5
    bars3 = bars2.copy()
    bars3["close"] = c3
    bars3["high"] = c3 + 0.2
    bars3["low"] = c3 - 0.1
    ex = list(st.evaluate(_ctx("VW", bars3, pos={"VW": _long("VW", entry=float(c2.iloc[-1]))})))
    assert any(x.action == SignalAction.EXIT_LONG for x in ex)


# --- ETF rotation ---


def test_etf_rotation_requires_all_bars():
    s = make_settings(
        ETF_ROTATION_ENABLED=True,
        ETF_ROTATION_SYMBOLS="SPY,QQQ",
        ETF_ROTATION_LOOKBACK_BARS=25,
        ETF_ROTATION_TOP_N=1,
        ETF_ROTATION_MIN_SCORE=-9.0,
    )
    st = ETFRotationStrategy(s)
    bars_spy = _ohlc(40, close_slope=0.001)
    assert list(st.evaluate(_ctx("SPY", bars_spy))) == []


def test_etf_rotation_entry_exit_uses_cross_symbol():
    s = make_settings(
        ETF_ROTATION_ENABLED=True,
        ETF_ROTATION_SYMBOLS="SPY,QQQ",
        ETF_ROTATION_LOOKBACK_BARS=30,
        ETF_ROTATION_TOP_N=1,
        ETF_ROTATION_MIN_SCORE=-9.0,
        ETF_ROTATION_TREND_SMA=5,
    )
    st = ETFRotationStrategy(s)
    spy = _ohlc(50, close_slope=0.0, vol=1e6, start=400.0)
    qqq = _ohlc(50, close_slope=0.4, vol=1e6, start=100.0)
    all_b = {"SPY": spy, "QQQ": qqq}
    ent = list(st.evaluate(_ctx("QQQ", qqq, all_bars=all_b)))
    assert any(x.action == SignalAction.ENTER_LONG for x in ent)

    spy2 = spy.copy()
    c = spy2["close"].astype(float)
    c.iloc[-30:] = np.linspace(float(c.iloc[-31]), float(c.iloc[-31]) * 1.6, 30)
    spy2["close"] = c
    spy2["high"] = c + 0.05
    spy2["low"] = c - 0.05
    qqq2 = qqq.copy()
    cq = qqq2["close"].astype(float)
    cq.iloc[-30:] = np.linspace(float(cq.iloc[-31]), float(cq.iloc[-31]) * 0.85, 30)
    qqq2["close"] = cq
    qqq2["high"] = cq + 0.05
    qqq2["low"] = cq - 0.05
    ex = list(
        st.evaluate(
            _ctx("QQQ", qqq2, pos={"QQQ": _long("QQQ")}, all_bars={"SPY": spy2, "QQQ": qqq2}),
        ),
    )
    assert any(x.action == SignalAction.EXIT_LONG for x in ex)


# --- pairs ---


def test_pairs_insufficient_and_entry_exit():
    s = make_settings(
        PAIRS_ENABLED=True,
        PAIRS_CONFIG_JSON='{"QQQ":"SPY"}',
        PAIRS_LOOKBACK_BARS=40,
        PAIRS_ENTRY_Z=-0.5,
        PAIRS_EXIT_Z=0.5,
    )
    st = PairsMeanReversionStrategy(s)
    spy = _ohlc(50, close_slope=0.05, start=100.0)
    qqq = _ohlc(50, close_slope=0.05, start=100.0)
    short = _ctx("QQQ", qqq.iloc[:20].copy(), all_bars={"SPY": spy.iloc[:20], "QQQ": qqq.iloc[:20]})
    assert list(st.evaluate(short)) == []

    # Diverge follower down vs leader at end
    q2 = qqq.copy()
    cq = q2["close"].astype(float)
    cq.iloc[-15:] = cq.iloc[-16] * 0.55
    q2["close"] = cq
    q2["low"] = cq - 0.2
    q2["high"] = cq + 0.2
    s2 = make_settings(
        PAIRS_ENABLED=True,
        PAIRS_CONFIG_JSON='{"QQQ":"SPY"}',
        PAIRS_LOOKBACK_BARS=35,
        PAIRS_ENTRY_Z=-1.0,
        PAIRS_EXIT_Z=-0.3,
    )
    st2 = PairsMeanReversionStrategy(s2)
    ent = list(st2.evaluate(_ctx("QQQ", q2, all_bars={"SPY": spy, "QQQ": q2})))
    assert any(x.action == SignalAction.ENTER_LONG for x in ent)

    # Convergence: align follower with leader (scaled) over final window
    q3 = q2.copy()
    cq3 = q3["close"].astype(float)
    cs = spy["close"].astype(float)
    scale = float(cq3.iloc[-25]) / float(cs.iloc[-25])
    cq3.iloc[-25:] = cs.iloc[-25:].values * scale
    q3["close"] = cq3
    q3["high"] = cq3 + 0.2
    q3["low"] = cq3 - 0.2
    ex = list(
        st2.evaluate(
            _ctx(
                "QQQ",
                q3,
                pos={"QQQ": _long("QQQ", entry=float(cq3.iloc[-10]))},
                all_bars={"SPY": spy, "QQQ": q3},
            ),
        ),
    )
    assert any(x.action == SignalAction.EXIT_LONG for x in ex)


# --- registry ---


def test_registry_builds_all_phase3_strategies():
    s = make_settings()
    for name in sorted(supported_strategy_names()):
        obj = build_strategy(name, s)
        assert obj is not None


def test_regime_overlay_spy_qqq():
    spy = _ohlc(80, close_slope=0.05, start=400.0)
    qqq = _ohlc(80, close_slope=0.05, start=100.0)
    ov = compute_risk_regime_overlay(spy, qqq)
    assert ov is not None
    assert 0.0 <= ov.confidence <= 1.0
