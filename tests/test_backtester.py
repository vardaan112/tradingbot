"""Backtester (historical data only — no Alpaca execution)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.market_data import Quote
from strategies.base import SignalAction, StrategyContext
from strategies.filters import compute_regime_snapshot
from strategies.indicators import rsi
from strategies.rsi_strategy import RSIMeanReversionStrategy
from strategies.sentiment import sentiment_overlay_neutral
from utils import backtester as bt


def test_load_or_fetch_bars_cache_hit(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}
    stamps = pd.date_range("2024-08-01", periods=10, freq="15min", tz=timezone.utc)
    midx = pd.MultiIndex.from_arrays([["SPY"] * 10, stamps], names=["symbol", "timestamp"])
    raw = pd.DataFrame(
        {
            "Open": range(100, 110),
            "High": range(101, 111),
            "Low": range(99, 109),
            "Close": range(100, 110),
            "Volume": [1e6] * 10,
        },
        index=midx,
    )
    client = MagicMock()

    def _get(_req) -> MagicMock:
        calls["n"] += 1
        out = MagicMock()
        out.df = raw.copy()
        return out

    client.get_stock_bars.side_effect = _get
    start = stamps[0].to_pydatetime()
    end = stamps[-1].to_pydatetime()
    kwargs = dict(
        client=client,
        symbol="SPY",
        start=start,
        end=end,
        timeframe="15Min",
        feed=bt.DataFeed.IEX,
        adjustment=bt.Adjustment.ALL,
        cache_dir=tmp_path,
        use_cache=True,
        refresh_cache=False,
    )
    bt.load_or_fetch_bars(**kwargs)
    bt.load_or_fetch_bars(**kwargs)
    assert calls["n"] == 1


def test_normalize_bars_multiindex_slices_symbol() -> None:
    tz = timezone.utc
    stamps = pd.date_range("2024-06-01", periods=3, freq="15min", tz=tz)
    midx = pd.MultiIndex.from_arrays(
        [["SPY", "SPY", "SPY"], stamps],
        names=["symbol", "timestamp"],
    )
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.5, 2.5, 3.5],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.2, 2.2, 3.2],
            "Volume": [100.0, 200.0, 300.0],
        },
        index=midx,
    )
    out = bt.normalize_bars_dataframe(raw, "spy")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert len(out) == 3
    assert out.index.tz is not None


def test_rsi_on_hist_matches_full_series_slice(make_settings_factory) -> None:
    settings = make_settings_factory(BAR_TIMEFRAME="15Min", RSI_LENGTH=14)
    n = 400
    rng = pd.date_range("2024-01-01", periods=n, freq="15min", tz=timezone.utc)
    closes = pd.Series(range(500, 500 + n), index=rng, dtype=float)
    bars = pd.DataFrame(
        {
            "open": closes.shift(1).fillna(closes.iloc[0]),
            "high": closes + 0.25,
            "low": closes - 0.25,
            "close": closes,
            "volume": [1e6] * n,
        }
    )
    strat = RSIMeanReversionStrategy(settings, state_store=None, database=None)
    full = rsi(bars["close"], length=settings.RSI_LENGTH)
    warmup = strat.warmup_lookback()
    for i in range(warmup, n):
        hist = bars.iloc[: i + 1]
        r_hist = rsi(hist["close"], length=settings.RSI_LENGTH)
        assert pd.notna(full.iloc[i])
        assert abs(float(r_hist.iloc[-1]) - float(full.iloc[i])) < 1e-6


@pytest.mark.parametrize(
    ("adx_cap", "expect_allow_sample"),
    [(1.5, False), (100.0, True)],
)
def test_adx_filter_blocks_or_allows(make_settings_factory, adx_cap, expect_allow_sample) -> None:
    settings = make_settings_factory(ADX_RANGE_MAX=adx_cap)
    n = 320
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(minutes=30),
        periods=n,
        freq="5min",
    )
    t = __import__("numpy").linspace(240.0, 75.0, n)
    close = pd.Series(t, index=idx)
    bars = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.25,
            "low": close - 1.25,
            "close": close,
            "volume": pd.Series([1_000_000.0] * n, index=idx),
        }
    )
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    if expect_allow_sample:
        assert snap.allow_rsi_long
    else:
        assert not snap.allow_rsi_long


def test_sma_filter_price_above_gate(make_settings_factory) -> None:
    settings = make_settings_factory()
    n = 250
    idx = pd.date_range("2024-01-02", periods=n, freq="1D", tz=timezone.utc)
    base = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    bars = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base,
            "volume": [5e6] * n,
        }
    )
    snap = compute_regime_snapshot(bars=bars, settings=settings)
    assert snap is not None
    assert snap.price_above_sma200


def test_atr_stop_multiplier_changes_theoretical_share_count(make_settings_factory) -> None:
    """Sizing core: fewer shares when stop distance widens."""

    px, atr_, eq = 100.0, 1.0, 100_000.0

    def sized(mult: float) -> float:
        s = make_settings_factory(
            ATR_STOP_MULTIPLIER=mult,
            MAX_RISK_PER_TRADE_PCT=0.01,
            MAX_EQUITY_USAGE_USD=100_000,
            BOT_CAPITAL_BASE_USD=100_000,
        )
        cap = s.resolved_capital_base(eq)
        sd = atr_ * float(s.ATR_STOP_MULTIPLIER)
        return math.floor(min(cap * float(s.MAX_RISK_PER_TRADE_PCT) / sd, float(s.MAX_EQUITY_USAGE_USD) / px))

    assert sized(1.0) > sized(2.0)


def test_trailing_stop_activates_via_live_strategy_fixture(make_settings_factory) -> None:
    settings = make_settings_factory(
        ADX_RANGE_MAX=100.0,
        TRAIL_TRIGGER_PCT=0.005,
        TRAIL_LOCKED_PROFIT_PCT=0.001,
        TRAIL_ATR_MULTIPLIER=0.1,
        ATR_LENGTH=5,
        RSI_LENGTH=14,
        RSI_EXIT=99.0,
        ATR_PROFIT_MULTIPLIER=50.0,
        MAX_HOLD_BARS=500,
    )
    strat = RSIMeanReversionStrategy(settings, state_store=None, database=None)
    n = 120
    idx = pd.date_range("2024-06-01", periods=n, freq="5min", tz=timezone.utc)
    c = pd.Series(100.0, index=idx)
    c.iloc[-5] = 99.92
    c.iloc[-4] = 99.90
    c.iloc[-3] = 100.55
    c.iloc[-2] = 100.60
    c.iloc[-1] = 100.45

    strat.adopt_long_position("SPY", 100.0)
    strat._entry_bar_index["SPY"] = 10  # noqa: SLF001

    found = False
    for k in range(n - 4, n + 1):
        ck = c.iloc[:k]
        bars_k = pd.DataFrame(
            {"open": ck, "high": ck + 0.2, "low": ck - 0.2, "close": ck, "volume": [2e6] * len(ck)},
        )
        cl = float(ck.iloc[-1])
        quote = Quote(
            symbol="SPY",
            bid=cl - 0.05,
            ask=cl + 0.05,
            bid_size=1.0,
            ask_size=1.0,
            timestamp=datetime.now(timezone.utc),
            feed="test",
        )
        from core.account import AccountSnapshot, PositionSnapshot

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
            multiplier=1.0,
            status="ACTIVE",
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
        )
        pos = PositionSnapshot(
            symbol="SPY",
            qty=100.0,
            avg_entry_price=100.0,
            side="long",
            market_value=100.0 * cl,
            cost_basis=10_000.0,
            unrealized_pl=(cl - 100.0) * 100.0,
            current_price=cl,
        )
        ctx = StrategyContext(
            symbol="SPY",
            bars=bars_k,
            quote=quote,
            account=acct,
            positions_by_symbol={"SPY": pos},
            open_order_symbols=set(),
            now_utc=datetime.now(timezone.utc),
            feed="backtest",
            sentiment_overlay=sentiment_overlay_neutral("SPY"),
            anti_martingale_risk_mode="normal",
            anti_martingale_multiplier=1.0,
            recent_trade_outcomes_hint="",
        )
        for sig in strat.evaluate(ctx):
            if sig.action == SignalAction.EXIT_LONG and "trailing_profit_breach" in sig.reason:
                found = True
        if found:
            break
    assert found


def test_default_param_grid_count() -> None:
    assert len(list(bt.default_param_grid())) == 27


def test_run_grid_returns_rows_and_trades(monkeypatch, make_settings_factory) -> None:
    settings = make_settings_factory(BAR_TIMEFRAME="5Min")
    n = 350
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz=timezone.utc)
    c = pd.Series([100 + 0.01 * ((-1) ** i) + 0.0001 * i for i in range(n)], index=idx)
    df = pd.DataFrame(
        {"open": c.shift(1).bfill(), "high": c + 0.3, "low": c - 0.3, "close": c, "volume": [1e6] * n}
    )
    monkeypatch.setattr(bt, "load_or_fetch_bars", lambda **_k: df)

    cfg = bt.BacktestConfig(
        symbols=("SPY",),
        start=idx[0].to_pydatetime(),
        end=idx[-1].to_pydatetime(),
        timeframe="5Min",
        initial_equity=100_000.0,
        risk_pct=float(settings.MAX_RISK_PER_TRADE_PCT),
        spread_pct=0.0002,
        slippage_bps=1.0,
        fee_bps_per_side=0.0,
        data_feed="iex",
        use_cache=False,
        refresh_cache=False,
        cache_dir=Path("/tmp/bt_cache_unused"),
        reports_dir=Path("/tmp/reports_unused"),
        output_results=Path("/tmp/r.csv"),
        output_trades=Path("/tmp/t.csv"),
        output_summary=Path("/tmp/s.md"),
    )
    params = [
        bt.StrategyParams(30.0, 25.0, 1.5, 1.5),
        bt.StrategyParams(31.0, 26.0, 2.0, 2.0),
    ]
    rows, trs = bt.run_grid(
        run_id="unit-test",
        base_settings=settings,
        cfg=cfg,
        client=MagicMock(),
        param_grid=params,
    )
    assert len(rows) == len(params)
    assert all(r.symbol == "PORTFOLIO_AVG" for r in rows)
    assert isinstance(trs, list)


def test_write_results_and_trades_csv(tmp_path) -> None:
    row = bt.GridRow(
        run_id="r1",
        parameter_set_id="abc",
        params=bt.StrategyParams(30.0, 25.0, 1.5, 1.5),
        symbol="PORTFOLIO_AVG",
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.02,
        win_rate=0.4,
        profit_factor=1.2,
        n_trades=10,
        avg_trade_return_pct=0.1,
        avg_holding_bars=4.5,
        worst_trade_usd=-3.0,
        best_trade_usd=5.0,
        avg_r_multiple=0.2,
        score=0.0,
    )
    p = tmp_path / "out.csv"
    bt.write_results_csv(p, [row])
    reread = pd.read_csv(p)
    assert "parameter_set_id" in reread.columns
    assert len(reread) == 1

    tp = tmp_path / "tr.csv"
    bt.write_trades_csv(tp, [{"source": "simulation", "run_id": "r", "parameter_set_id": "p"}])
    assert tp.is_file()


def test_import_surface_excludes_live_trading_clients() -> None:
    lines = Path(__file__).resolve().parents[1].joinpath("src/utils/backtester.py").read_text().splitlines()
    bad = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if "alpaca.trading" in stripped or "TradingClient" in stripped:
                bad.append(stripped)
            if "core.orders" in stripped or "services.orchestrator" in stripped:
                bad.append(stripped)
            if "submit_order" in stripped:
                bad.append(stripped)
    assert not bad


def test_runtime_module_exposes_no_order_service() -> None:
    assert getattr(bt, "OrderService", None) is None


def test_vector_simulate_runs(monkeypatch, make_settings_factory) -> None:
    settings = make_settings_factory(ADX_RANGE_MAX=100.0, BAR_TIMEFRAME="5Min")
    n = 400
    idx = pd.date_range("2024-03-01", periods=n, freq="5min", tz=timezone.utc)
    rng = pd.Series(range(100, 100 + n), dtype=float)
    import numpy as np

    jitter = rng + 3 * np.sin(np.linspace(0, 20, n))
    bars = pd.DataFrame(
        {
            "open": jitter.shift(1).bfill(),
            "high": jitter + 0.5,
            "low": jitter - 0.5,
            "close": jitter,
            "volume": [1e6] * n,
        },
        index=idx,
    )
    res = bt.simulate_symbol(
        bars,
        "spy",
        settings,
        initial_equity=50_000.0,
        spread_pct=0.0005,
        slippage_bps=2.0,
        fee_bps_per_side=0.0,
    )
    assert res.symbol == "SPY"
    assert isinstance(res.trades, list)
