"""Vector-prep + fast path-dependent loop backtester (historical Alpaca data only).

Never imports ``TradingClient`` or live order APIs.

Run::

    python -m utils.backtester
    python -m src.utils.backtester
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.settings import Settings, get_settings
from strategies.filters import adx as adx_series
from strategies.filters import sma as sma_series
from strategies.indicators import atr, rsi

_LOG = logging.getLogger("tradingbot.backtest")


@dataclass(frozen=True)
class BacktestConfig:
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    timeframe: str
    initial_equity: float
    risk_pct: float
    spread_pct: float
    slippage_bps: float
    fee_bps_per_side: float
    data_feed: str
    use_cache: bool
    refresh_cache: bool
    cache_dir: Path
    reports_dir: Path
    output_results: Path
    output_trades: Path
    output_summary: Path


@dataclass(frozen=True)
class StrategyParams:
    rsi_oversold: float
    adx_range_max: float
    atr_stop_multiplier: float
    trail_atr_multiplier: float

    def label(self) -> str:
        return (
            f"rsi={self.rsi_oversold}_adxmax={self.adx_range_max}_"
            f"atrstop={self.atr_stop_multiplier}_trailatr={self.trail_atr_multiplier}"
        )

    def parameter_set_id(self) -> str:
        return hashlib.sha256(self.label().encode()).hexdigest()[:12]


@dataclass
class TradeResult:
    symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    shares: float
    pnl_usd: float
    gross_pnl: float
    fees_usd: float
    slippage_usd: float
    return_pct: float
    bars_held: int
    exit_reason: str
    rsi_entry: float
    adx_entry: float
    sma_200_entry: float
    atr_entry: float
    regime_type: str
    trailing_stop_active: bool
    mfpe: float
    mae: float
    r_multiple: float


@dataclass
class BacktestResult:
    params: StrategyParams
    symbol: str
    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return_pct: float = 0.0
    avg_holding_bars: float = 0.0
    worst_trade_usd: float = 0.0
    best_trade_usd: float = 0.0
    avg_r_multiple: float = 0.0


@dataclass
class GridRow:
    run_id: str
    parameter_set_id: str
    params: StrategyParams
    symbol: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_return_pct: float
    avg_holding_bars: float
    worst_trade_usd: float
    best_trade_usd: float
    avg_r_multiple: float
    score: float


def parse_timeframe(spec: str) -> TimeFrame:
    spec = spec.strip()
    mapping: dict[str, tuple[int, TimeFrameUnit]] = {
        "1Min": (1, TimeFrameUnit.Minute),
        "5Min": (5, TimeFrameUnit.Minute),
        "15Min": (15, TimeFrameUnit.Minute),
        "1Hour": (1, TimeFrameUnit.Hour),
        "1Day": (1, TimeFrameUnit.Day),
    }
    if spec not in mapping:
        raise ValueError(f"unsupported timeframe: {spec!r}")
    a, u = mapping[spec]
    return TimeFrame(a, u)


def resolve_data_feed(name: str) -> DataFeed:
    n = name.strip().lower()
    if n == "sip":
        return DataFeed.SIP
    if n == "iex":
        return DataFeed.IEX
    raise ValueError(f"unsupported data feed for backtest: {name!r}")


def normalize_bars_dataframe(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        try:
            df = df.xs(symbol.upper(), level="symbol")
        except KeyError:
            return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)

    df = df.sort_index()
    colmap = {c.lower(): c for c in df.columns}
    for need in ("open", "high", "low", "close", "volume"):
        if need not in [c.lower() for c in df.columns]:
            return pd.DataFrame()

    out = pd.DataFrame(
        {
            "open": df[[colmap["open"]]].iloc[:, 0].astype(float),
            "high": df[[colmap["high"]]].iloc[:, 0].astype(float),
            "low": df[[colmap["low"]]].iloc[:, 0].astype(float),
            "close": df[[colmap["close"]]].iloc[:, 0].astype(float),
            "volume": df[[colmap["volume"]]].iloc[:, 0].astype(float),
        },
        index=df.index,
    )
    return out[~out.index.duplicated(keep="last")]


def cache_stem(symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
    sd = start.astimezone(timezone.utc).date().isoformat()
    ed = end.astimezone(timezone.utc).date().isoformat()
    return f"{symbol.upper()}_{timeframe}_{sd}_{ed}"


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401,WPS433
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401,WPS433
            return True
        except Exception:
            return False


def _read_disk_cache(pq: Path, csvp: Path) -> Optional[pd.DataFrame]:
    if pq.is_file():
        try:
            df = pd.read_parquet(pq)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize(timezone.utc)
                else:
                    df.index = df.index.tz_convert(timezone.utc)
            return df.sort_index()
    if csvp.is_file():
        raw = pd.read_csv(csvp)
        if "timestamp" not in raw.columns:
            return None
        raw = raw.set_index(pd.to_datetime(raw["timestamp"], utc=True))
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize(timezone.utc)
        else:
            raw.index = raw.index.tz_convert(timezone.utc)
        need = {"open", "high", "low", "close", "volume"}
        lc = set(x.lower() for x in raw.columns)
        if not need <= lc:
            return None
        cm = {c.lower(): c for c in raw.columns}
        df = pd.DataFrame(
            {k: raw[cm[k]].astype(float) for k in need},
            index=raw.index,
        )
        return df.sort_index()
    return None


def _write_disk_cache(df: pd.DataFrame, pq: Path, csvp: Path) -> None:
    if df.empty:
        return
    if _parquet_ok():
        try:
            df.to_parquet(pq)
            csvp.unlink(missing_ok=True)
            return
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("event=backtest_cache_parquet_fail err=%s", exc)
    tmp = df.copy()
    tmp.insert(0, "timestamp", tmp.index.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp.to_csv(csvp, index=False)
    pq.unlink(missing_ok=True)


def load_or_fetch_bars(
    *,
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    feed: DataFeed,
    adjustment: Adjustment,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_stem(symbol, timeframe, start, end)
    pq = cache_dir / f"{stem}.parquet"
    csvp = cache_dir / f"{stem}.csv"

    if refresh_cache:
        pq.unlink(missing_ok=True)
        csvp.unlink(missing_ok=True)

    if use_cache and not refresh_cache:
        hit = _read_disk_cache(pq, csvp)
        if hit is not None and len(hit) > 5:
            norm = normalize_bars_dataframe(hit, symbol)
            if not norm.empty:
                _LOG.info("event=backtest_cache_hit symbol=%s rows=%d", symbol, len(norm))
                return norm

    _LOG.info("event=backtest_cache_miss symbol=%s", symbol)
    tf = parse_timeframe(timeframe)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol.upper()],
        timeframe=tf,
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
        limit=100_000,
    )
    resp = client.get_stock_bars(req)
    df = normalize_bars_dataframe(resp.df, symbol)
    if use_cache and not df.empty:
        _write_disk_cache(df, pq, csvp)
    return df


def _warmup(settings: Settings) -> int:
    return max(
        settings.RSI_LENGTH * 3,
        settings.ATR_LENGTH * 3,
        settings.ADX_LENGTH * 3,
        settings.SMA_FILTER_LENGTH + settings.SMA_SLOPE_LOOKBACK_BARS + 10,
        80,
    )


def build_strategy_settings(
    base: Settings,
    sp: StrategyParams,
    *,
    timeframe: str,
    initial_equity: float,
    risk_pct: float,
) -> Settings:
    cap = max(float(base.MAX_EQUITY_USAGE_USD), float(initial_equity))
    return base.model_copy(
        update={
            "RSI_OVERSOLD": float(sp.rsi_oversold),
            "ADX_RANGE_MAX": float(sp.adx_range_max),
            "ATR_STOP_MULTIPLIER": float(sp.atr_stop_multiplier),
            "TRAIL_ATR_MULTIPLIER": float(sp.trail_atr_multiplier),
            "BAR_TIMEFRAME": timeframe,  # type: ignore[arg-type]
            "MAX_RISK_PER_TRADE_PCT": float(risk_pct),
            "BOT_CAPITAL_BASE_USD": float(initial_equity),
            "MAX_EQUITY_USAGE_USD": cap,
        },
    )


def simulate_symbol(
    bars: pd.DataFrame,
    symbol: str,
    settings: Settings,
    *,
    initial_equity: float,
    spread_pct: float,
    slippage_bps: float,
    fee_bps_per_side: float,
) -> BacktestResult:
    sp = StrategyParams(
        rsi_oversold=float(settings.RSI_OVERSOLD),
        adx_range_max=float(settings.ADX_RANGE_MAX),
        atr_stop_multiplier=float(settings.ATR_STOP_MULTIPLIER),
        trail_atr_multiplier=float(settings.TRAIL_ATR_MULTIPLIER),
    )
    slip = float(slippage_bps) / 10_000.0
    half_sp = float(spread_pct) / 2.0
    fee = fee_bps_per_side / 10_000.0

    c = bars["close"].astype(float)
    h = bars["high"].astype(float)
    l = bars["low"].astype(float)
    o = bars["open"].astype(float)

    rsi_v = rsi(c, length=int(settings.RSI_LENGTH))
    atr_v = atr(h, l, c, length=int(settings.ATR_LENGTH))
    adx_v = adx_series(h, l, c, length=int(settings.ADX_LENGTH))
    sma_v = sma_series(c, int(settings.SMA_FILTER_LENGTH))
    lb = int(settings.SMA_SLOPE_LOOKBACK_BARS)
    sma_slope = sma_v - sma_v.shift(lb)
    adx_max = float(settings.ADX_RANGE_MAX)
    adx_ok = adx_v < adx_max
    allow = (sma_slope > 0.0) | adx_ok
    allow = allow & sma_slope.notna() & adx_v.notna()

    hi_c = float(settings.HIGH_CONVICTION_RISK_MULTIPLIER)
    lo_c = float(settings.LOW_CONVICTION_RISK_MULTIPLIER)
    conv = pd.Series(np.where(c > sma_v, hi_c, lo_c), index=bars.index)

    n = len(bars)
    w = _warmup(settings)
    if n <= w + 3:
        return BacktestResult(params=sp, symbol=symbol.upper(), equity_curve=pd.Series(dtype=float))

    cash = float(initial_equity)
    sh = 0.0
    entry_px = 0.0
    entry_i = -1
    mf_hi = 0.0
    mf_lo = 0.0
    trail_on = False
    t_lock = 0.0
    t_peak = 0.0
    t_stop = 0.0
    pending: Optional[dict[str, Any]] = None
    entry_meta: Optional[dict[str, Any]] = None

    rsi_e = rsi_v.to_numpy(dtype=float)
    atr_e = atr_v.to_numpy(dtype=float)
    adx_e = adx_v.to_numpy(dtype=float)
    sma_e = sma_v.to_numpy(dtype=float)
    allow_a = allow.to_numpy()
    conv_a = conv.to_numpy(dtype=float)
    idx = list(bars.index)

    trades: list[TradeResult] = []
    eq_pts: list[tuple[pd.Timestamp, float]] = []
    sym = symbol.upper()
    oversold = float(settings.RSI_OVERSOLD)
    rsi_x = float(settings.RSI_EXIT)

    def mark(i: int) -> None:
        eq_pts.append((idx[i], cash + sh * float(c.iloc[i])))

    for i in range(w, n):
        o_i, c_i, hi_i, lo_i = float(o.iloc[i]), float(c.iloc[i]), float(h.iloc[i]), float(l.iloc[i])

        if pending is not None and int(pending["ex"]) == i and sh < 1e-9:
            px = o_i * (1.0 + half_sp + slip)
            atr0 = float(pending["atr"])
            cm = float(pending["cm"])
            base = settings.resolved_capital_base(max(cash, 1.0))
            sd = max(atr0 * float(settings.ATR_STOP_MULTIPLIER), 1e-9)
            rb = base * float(settings.MAX_RISK_PER_TRADE_PCT) * cm
            nsh = math.floor(min(rb / sd, float(settings.MAX_EQUITY_USAGE_USD) / max(px, 1e-9)))
            snap = pending
            pending = None
            if nsh >= 1:
                cost = nsh * px * (1.0 + fee)
                if cost <= cash:
                    cash -= cost
                    sh = float(nsh)
                    entry_px = px
                    entry_i = i
                    mf_hi, mf_lo = hi_i, lo_i
                    trail_on, t_lock, t_peak, t_stop = False, 0.0, 0.0, 0.0
                    entry_meta = dict(snap)

        if sh > 1e-9 and entry_px > 0:
            atr_i = float(atr_e[i])
            if not math.isfinite(atr_i) or atr_i <= 0:
                mark(i)
                continue

            mf_hi = max(mf_hi, hi_i)
            mf_lo = min(mf_lo, lo_i)

            stop_px_lv = entry_px - float(settings.ATR_STOP_MULTIPLIER) * atr_i
            hit_cat = c_i <= stop_px_lv

            unreal = c_i / entry_px - 1.0 if entry_px > 0 else 0.0
            if not trail_on and unreal >= float(settings.TRAIL_TRIGGER_PCT):
                trail_on = True
                t_lock = entry_px * (1.0 + float(settings.TRAIL_LOCKED_PROFIT_PCT))
                t_peak = max(c_i, entry_px)
                t_stop = max(t_lock, t_peak - float(settings.TRAIL_ATR_MULTIPLIER) * atr_i)
            elif trail_on:
                t_peak = max(t_peak, c_i)
                t_stop = max(t_lock, t_peak - float(settings.TRAIL_ATR_MULTIPLIER) * atr_i)

            x_px = c_i * (1.0 - half_sp - slip) * (1.0 - fee)
            tp_lv = entry_px + float(settings.ATR_PROFIT_MULTIPLIER) * atr_i
            rsi_i = float(rsi_e[i]) if math.isfinite(rsi_e[i]) else 999.0
            t_bars = i - entry_i

            reason = ""
            if hit_cat:
                reason = "atr_stop_breach"
            elif trail_on and c_i <= t_stop + 1e-12:
                reason = "trailing_profit_breach"
            elif c_i >= tp_lv:
                reason = "tp_hit"
            elif rsi_i >= rsi_x:
                reason = "rsi_exit"
            elif t_bars >= int(settings.MAX_HOLD_BARS):
                reason = "time_exit"

            if reason:
                proceeds = sh * x_px
                entry_cost = sh * entry_px * (1.0 + fee)
                pnl = proceeds - entry_cost
                mid_in, mid_out = entry_px, x_px / ((1.0 - half_sp - slip) * (1.0 - fee))
                gross = sh * (mid_out - mid_in)
                fee_open = sh * entry_px * fee
                fee_close = sh * x_px * fee / max(1.0 - fee, 1e-9)
                fees_usd = float(fee_open + fee_close)
                slip_usd = sh * entry_px * (half_sp + slip) + sh * c_i * (half_sp + slip)

                snap = entry_meta or {}
                if not isinstance(snap, dict):
                    snap = {}
                rsi_ent = float(snap.get("rsi", rsi_e[entry_i]))
                adx_ent = float(snap.get("adx", adx_e[entry_i]))
                sma_ent = float(snap.get("sma", sma_e[entry_i]))
                reg = str(snap.get("regime", "Range"))
                atr_ent = float(snap.get("atr", atr_e[entry_i]))
                risk_1r = sh * atr_ent * float(settings.ATR_STOP_MULTIPLIER)
                r_mult = pnl / risk_1r if risk_1r > 1e-9 else float("nan")

                trades.append(
                    TradeResult(
                        symbol=sym,
                        entry_time=str(idx[entry_i]),
                        exit_time=str(idx[i]),
                        entry_price=entry_px,
                        exit_price=x_px,
                        shares=sh,
                        pnl_usd=float(pnl),
                        gross_pnl=float(gross),
                        fees_usd=float(fees_usd),
                        slippage_usd=float(slip_usd),
                        return_pct=float(x_px / entry_px - 1.0) if entry_px > 0 else 0.0,
                        bars_held=max(0, t_bars),
                        exit_reason=reason,
                        rsi_entry=rsi_ent,
                        adx_entry=adx_ent,
                        sma_200_entry=sma_ent,
                        atr_entry=atr_ent,
                        regime_type=reg,
                        trailing_stop_active=trail_on,
                        mfpe=float(mf_hi / entry_px - 1.0) if entry_px > 0 else 0.0,
                        mae=float(mf_lo / entry_px - 1.0) if entry_px > 0 else 0.0,
                        r_multiple=float(r_mult),
                    )
                )
                cash += proceeds
                sh = 0.0
                entry_px = 0.0
                trail_on = False
                entry_meta = None
                mark(i)
                continue

        if sh < 1e-9 and pending is None:
            r_i = float(rsi_e[i]) if math.isfinite(rsi_e[i]) else 999.0
            if bool(allow_a[i]) and r_i < oversold and i + 1 < n:
                reg_tp = "Range" if (math.isfinite(adx_e[i]) and float(adx_e[i]) < adx_max) else "Trending"
                pending = {
                    "ex": i + 1,
                    "atr": float(atr_e[i]),
                    "cm": float(conv_a[i]),
                    "rsi": r_i,
                    "adx": float(adx_e[i]) if math.isfinite(adx_e[i]) else 0.0,
                    "sma": float(sma_e[i]) if math.isfinite(sma_e[i]) else c_i,
                    "regime": reg_tp,
                }

        mark(i)

    eq = pd.Series([e for _, e in eq_pts], index=pd.DatetimeIndex([t for t, _ in eq_pts]))
    met = compute_performance_metrics(eq, trades, initial_equity)
    return BacktestResult(params=sp, symbol=sym, trades=trades, equity_curve=eq, **met)


def compute_performance_metrics(
    equity: pd.Series,
    trades: Sequence[TradeResult],
    initial_equity: float,
) -> dict[str, Any]:
    total_return = 0.0
    if len(equity) > 1 and initial_equity > 0:
        total_return = float(equity.iloc[-1] / initial_equity) - 1.0

    daily = equity.resample("1B").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = 0.0
    if len(rets) > 2 and float(rets.std()) > 1e-12:
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252.0))

    max_dd = 0.0
    if len(daily) > 1:
        peak = daily.cummax()
        dd = (daily - peak) / peak.replace(0.0, np.nan)
        max_dd = float(dd.min()) if dd.notna().any() else 0.0

    n = len(trades)
    wins = [t for t in trades if t.pnl_usd > 1e-9]
    losses = [t for t in trades if t.pnl_usd < -1e-9]
    win_rate = float(len(wins) / n) if n else 0.0
    gp = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    if gl > 1e-9:
        pf = float(gp / gl)
    elif gp > 0:
        pf = 9999.0
    else:
        pf = 0.0

    avg_tr = float(np.mean([t.return_pct for t in trades])) if trades else 0.0
    avg_hold = float(np.mean([t.bars_held for t in trades])) if trades else 0.0
    worst = float(min((t.pnl_usd for t in trades), default=0.0))
    best = float(max((t.pnl_usd for t in trades), default=0.0))
    r_vals = [t.r_multiple for t in trades if math.isfinite(t.r_multiple)]
    avg_r = float(np.mean(r_vals)) if r_vals else 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "n_trades": n,
        "avg_trade_return_pct": avg_tr * 100.0,
        "avg_holding_bars": avg_hold,
        "worst_trade_usd": worst,
        "best_trade_usd": best,
        "avg_r_multiple": avg_r,
    }


def default_param_grid() -> Iterable[StrategyParams]:
    rsi = [25.0, 30.0, 35.0]
    adx = [20.0, 25.0, 30.0]
    atrs = [1.5, 2.0, 3.0]
    for a, b, c in itertools.product(rsi, adx, atrs):
        yield StrategyParams(
            rsi_oversold=a,
            adx_range_max=b,
            atr_stop_multiplier=c,
            trail_atr_multiplier=c,
        )


def grid_score(sharpe: float, max_dd: float) -> float:
    return float(sharpe) - abs(float(max_dd)) * 2.0


def run_grid(
    *,
    run_id: str,
    base_settings: Settings,
    cfg: BacktestConfig,
    client: StockHistoricalDataClient,
    param_grid: Sequence[StrategyParams],
) -> tuple[list[GridRow], list[dict[str, Any]]]:
    feed = resolve_data_feed(cfg.data_feed)
    adjustment = Adjustment.ALL
    bar_cache: dict[str, pd.DataFrame] = {}
    for sym in cfg.symbols:
        bar_cache[sym] = load_or_fetch_bars(
            client=client,
            symbol=sym,
            start=cfg.start,
            end=cfg.end,
            timeframe=cfg.timeframe,
            feed=feed,
            adjustment=adjustment,
            cache_dir=cfg.cache_dir,
            use_cache=cfg.use_cache,
            refresh_cache=cfg.refresh_cache,
        )

    rows: list[GridRow] = []
    trade_rows: list[dict[str, Any]] = []

    for sp in param_grid:
        pid = sp.parameter_set_id()
        _LOG.info("event=backtest_param_run_start run_id=%s param_set=%s", run_id, pid)
        st = build_strategy_settings(
            base_settings,
            sp,
            timeframe=cfg.timeframe,
            initial_equity=cfg.initial_equity,
            risk_pct=cfg.risk_pct,
        )

        per: list[BacktestResult] = []
        for sym in cfg.symbols:
            df = bar_cache.get(sym, pd.DataFrame())
            if df.empty:
                continue
            res = simulate_symbol(
                df,
                sym,
                st,
                initial_equity=cfg.initial_equity,
                spread_pct=cfg.spread_pct,
                slippage_bps=cfg.slippage_bps,
                fee_bps_per_side=cfg.fee_bps_per_side,
            )
            per.append(res)
            for tr in res.trades:
                trade_rows.append(trade_row_dict(run_id, pid, sp, tr))

        if not per:
            _LOG.info("event=backtest_param_run_complete run_id=%s param_set=%s trades=0", run_id, pid)
            continue

        avg_ret = float(np.mean([r.total_return for r in per]))
        avg_sh = float(np.mean([r.sharpe_ratio for r in per]))
        avg_dd = float(np.mean([r.max_drawdown for r in per]))
        all_t = [t for r in per for t in r.trades]
        n_tr = len(all_t)
        wins = [t for t in all_t if t.pnl_usd > 1e-9]
        losses = [t for t in all_t if t.pnl_usd < -1e-9]
        wr = len(wins) / n_tr if n_tr else 0.0
        gp = sum(t.pnl_usd for t in wins)
        gl = abs(sum(t.pnl_usd for t in losses))
        if gl > 1e-9:
            pf = float(gp / gl)
        elif gp > 0:
            pf = 9999.0
        else:
            pf = 0.0
        avg_trade = float(np.mean([t.return_pct for t in all_t])) * 100.0 if all_t else 0.0
        avg_hold = float(np.mean([t.bars_held for t in all_t])) if all_t else 0.0
        worst = float(min((t.pnl_usd for t in all_t), default=0.0))
        best = float(max((t.pnl_usd for t in all_t), default=0.0))
        rvals = [t.r_multiple for t in all_t if math.isfinite(t.r_multiple)]
        avg_r = float(np.mean(rvals)) if rvals else 0.0
        sc = grid_score(avg_sh, avg_dd)

        rows.append(
            GridRow(
                run_id=run_id,
                parameter_set_id=pid,
                params=sp,
                symbol="PORTFOLIO_AVG",
                total_return=avg_ret,
                sharpe_ratio=avg_sh,
                max_drawdown=avg_dd,
                win_rate=wr,
                profit_factor=pf,
                n_trades=n_tr,
                avg_trade_return_pct=avg_trade,
                avg_holding_bars=avg_hold,
                worst_trade_usd=worst,
                best_trade_usd=best,
                avg_r_multiple=avg_r,
                score=sc,
            )
        )
        _LOG.info(
            "event=backtest_param_run_complete run_id=%s param_set=%s trades=%d sharpe=%.3f",
            run_id,
            pid,
            n_tr,
            avg_sh,
        )

    rows.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    return rows, trade_rows


def trade_row_dict(run_id: str, pid: str, sp: StrategyParams, t: TradeResult) -> dict[str, Any]:
    return {
        "source": "simulation",
        "run_id": run_id,
        "parameter_set_id": pid,
        "parameter_label": sp.label(),
        "symbol": t.symbol,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "side": "long",
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "qty": t.shares,
        "gross_pnl": t.gross_pnl,
        "net_pnl": t.pnl_usd,
        "fees": t.fees_usd,
        "slippage": t.slippage_usd,
        "return_pct": t.return_pct * 100.0,
        "exit_reason": t.exit_reason,
        "rsi_entry": t.rsi_entry,
        "adx_entry": t.adx_entry,
        "sma_200_entry": t.sma_200_entry,
        "atr_entry": t.atr_entry,
        "regime_type": t.regime_type,
        "trailing_stop_active": t.trailing_stop_active,
        "max_favorable_excursion": t.mfpe,
        "max_adverse_excursion": t.mae,
        "r_multiple": t.r_multiple,
        "bars_held": t.bars_held,
    }


def write_results_csv(path: Path, rows: Sequence[GridRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in rows:
        d = asdict(r.params)
        d.update(
            {
                "run_id": r.run_id,
                "parameter_set_id": r.parameter_set_id,
                "symbol_scope": r.symbol,
                "total_return": r.total_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "n_trades": r.n_trades,
                "avg_trade_return_pct": r.avg_trade_return_pct,
                "avg_holding_bars": r.avg_holding_bars,
                "worst_trade_usd": r.worst_trade_usd,
                "best_trade_usd": r.best_trade_usd,
                "avg_r_multiple": r.avg_r_multiple,
                "score": r.score,
                "params_label": r.params.label(),
            },
        )
        recs.append(d)
    pd.DataFrame(recs).to_csv(path, index=False)


def write_trades_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(
            columns=[
                "source",
                "run_id",
                "parameter_set_id",
                "symbol",
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "qty",
                "gross_pnl",
                "net_pnl",
                "fees",
                "slippage",
                "return_pct",
                "exit_reason",
                "rsi_entry",
                "adx_entry",
                "sma_200_entry",
                "atr_entry",
                "regime_type",
                "trailing_stop_active",
                "max_favorable_excursion",
                "max_adverse_excursion",
                "parameter_label",
                "bars_held",
                "r_multiple",
            ],
        ).to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_md(path: Path, rows: Sequence[GridRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Backtest grid summary",
        "",
        "> **Not live performance.** Slippage, halts, and fills differ in production.",
        "> **Overfitting:** grid champions are tuned in-sample.",
        "",
    ]
    top_sh = sorted(rows, key=lambda r: r.sharpe_ratio, reverse=True)[:10]
    lines += ["## Top 10 by Sharpe", ""]
    for r in top_sh:
        lines.append(
            f"- `{r.params.label()}` Sharpe={r.sharpe_ratio:.3f} MDD={r.max_drawdown:.4f} PF={r.profit_factor:.2f}",
        )
    lines.append("")
    top_dd = sorted(rows, key=lambda r: r.max_drawdown, reverse=True)[:10]
    lines += ["## Top 10 by lowest drawdown (closest to zero)", ""]
    for r in top_dd:
        lines.append(f"- `{r.params.label()}` MDD={r.max_drawdown:.4f} Sharpe={r.sharpe_ratio:.3f}")
    lines.append("")
    top_pf = sorted(rows, key=lambda r: r.profit_factor, reverse=True)[:10]
    lines += ["## Top 10 by profit factor", ""]
    for r in top_pf:
        lines.append(f"- `{r.params.label()}` PF={r.profit_factor:.2f} Sharpe={r.sharpe_ratio:.3f}")
    lines.append("")
    bal = sorted(rows, key=lambda r: r.score, reverse=True)
    lines += ["## Best balanced (Sharpe − 2×|MDD|)", ""]
    if bal:
        b = bal[0]
        lines.append(
            f"- `{b.params.label()}` score={b.score:.4f} Sharpe={b.sharpe_ratio:.3f} MDD={b.max_drawdown:.4f}",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_iso_utc(day: str, *, end_of_day: bool) -> datetime:
    d = datetime.fromisoformat(day)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone(timezone.utc)
    if end_of_day:
        return d.replace(hour=23, minute=59, second=59, microsecond=0)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = argparse.ArgumentParser(description="RSI vector backtester (historical data only).")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--start", type=str, required=False)
    p.add_argument("--end", type=str, required=False)
    p.add_argument("--timeframe", type=str, default="15Min")
    p.add_argument("--initial-equity", type=float, default=None)
    p.add_argument("--risk-pct", type=float, default=None)
    p.add_argument("--use-cache", action="store_true", default=True)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--output-results", type=str, default="reports/backtest_results.csv")
    p.add_argument("--output-trades", type=str, default="reports/backtest_trades.csv")
    p.add_argument("--summary", type=str, default="reports/backtest_summary.md")
    args = p.parse_args(list(argv) if argv is not None else None)

    use_cache = not bool(args.no_cache)
    settings = get_settings()
    root = Path.cwd()
    end = _parse_iso_utc(args.end, end_of_day=True) if args.end else datetime.now(timezone.utc)
    if args.start:
        start = _parse_iso_utc(args.start, end_of_day=False)
    else:
        start = end - timedelta(days=365)

    syms = tuple(s.strip().upper() for s in args.symbols) if args.symbols else tuple(settings.symbols_list)
    ieq = float(args.initial_equity) if args.initial_equity is not None else max(10_000.0, float(settings.MAX_EQUITY_USAGE_USD))
    rpct = float(args.risk_pct) if args.risk_pct is not None else float(settings.MAX_RISK_PER_TRADE_PCT)

    cache_dir = root / "runtime" / "cache"
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    out_res = Path(args.output_results)
    out_tr = Path(args.output_trades)
    out_sum = Path(args.summary)
    if not out_res.is_absolute():
        out_res = root / out_res
    if not out_tr.is_absolute():
        out_tr = root / out_tr
    if not out_sum.is_absolute():
        out_sum = root / out_sum

    cfg = BacktestConfig(
        symbols=syms,
        start=start,
        end=end,
        timeframe=args.timeframe,
        initial_equity=ieq,
        risk_pct=rpct,
        spread_pct=float(settings.SPREAD_FILTER_PCT),
        slippage_bps=1.5,
        fee_bps_per_side=0.0,
        data_feed=settings.feed_resolved(sip_supported=False),
        use_cache=use_cache,
        refresh_cache=bool(args.refresh_cache),
        cache_dir=cache_dir,
        reports_dir=rep,
        output_results=out_res,
        output_trades=out_tr,
        output_summary=out_sum,
    )

    client = StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
    )

    run_id = str(uuid.uuid4())
    grid = list(default_param_grid())
    _LOG.info(
        "event=backtest_start run_id=%s symbols=%s grid=%d %s..%s",
        run_id,
        ",".join(syms),
        len(grid),
        start.date(),
        end.date(),
    )

    rows, trs = run_grid(
        run_id=run_id,
        base_settings=settings,
        cfg=cfg,
        client=client,
        param_grid=grid,
    )

    write_results_csv(out_res, rows)
    write_trades_csv(out_tr, trs)
    write_summary_md(out_sum, rows)

    _LOG.info(
        "event=backtest_complete run_id=%s results=%s trades=%s summary=%s",
        run_id,
        out_res,
        out_tr,
        out_sum,
    )


if __name__ == "__main__":
    main()
