"""Historical replay: walk bars, run ``StrategyEngine``, simulate fills and research DB writes."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical.stock import StockHistoricalDataClient

from config.settings import Settings
from core.database import Database
from core.market_data import Quote
from services.ensemble import WeightedEnsembleEngine, votes_to_contributing_json
from services.strategy_engine import StrategyEngine
from strategies.base import Signal, SignalAction, Strategy, StrategyContext
from strategies.registry import build_strategies
from strategies.sentiment import sentiment_overlay_neutral
from utils import backtester as bt

from .benchmark import buy_hold_equity_curve
from .fill_model import FillModelParams
from .simulated_account import SimulatedAccount
from .simulated_broker import PendingFill, SimulatedBroker

_LOG = logging.getLogger("tradingbot.sim.replay")


def resolve_replay_window(
    *,
    end: datetime,
    start: Optional[datetime] = None,
    lookback_days: Optional[int] = None,
) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc). ``end`` must be timezone-aware UTC."""

    if start is not None and lookback_days is not None:
        raise ValueError("ambiguous window: pass only one of start or lookback_days (not both)")
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    if lookback_days is not None:
        start_utc = end_utc - timedelta(days=int(lookback_days))
        return start_utc, end_utc
    if start is None:
        raise ValueError("must pass start or lookback_days")
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    if start_utc >= end_utc:
        raise ValueError("start must be before end")
    return start_utc, end_utc


def align_symbol_frames(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Intersect indices and return reindexed OHLCV frames."""

    nonempty = {k: v for k, v in bars_by_symbol.items() if v is not None and not v.empty}
    if not nonempty:
        return pd.DatetimeIndex([], tz=timezone.utc), {}
    common: Optional[pd.DatetimeIndex] = None
    for df in nonempty.values():
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("bars index must be DatetimeIndex")
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        return pd.DatetimeIndex([], tz=timezone.utc), {}
    aligned = {s: nonempty[s].loc[common].copy() for s in nonempty}
    return common, aligned


def describe_bar_alignment(sym_frames: dict[str, pd.DataFrame], *, ref_symbol: str = "SPY") -> str:
    """Human-readable lines: per-symbol row counts, span, overlap vs a liquid reference index."""

    lines: list[str] = []
    ref = ref_symbol if ref_symbol in sym_frames else next(iter(sym_frames.keys()))
    ref_idx = sym_frames[ref].index
    for s in sorted(sym_frames.keys()):
        df = sym_frames[s]
        if df is None or df.empty:
            lines.append(f"  {s}: EMPTY")
            continue
        inter = ref_idx.intersection(df.index)
        lines.append(
            f"  {s}: n={len(df)}  {df.index.min()} .. {df.index.max()}  "
            f"overlap_vs_{ref}_timestamps={len(inter)}",
        )
    common, _ = align_symbol_frames(sym_frames)
    lines.append(f"  ALL_SYMBOLS inner intersection: n={len(common)}")
    return "\n".join(lines)


def greedy_drop_symbols_for_alignment(
    sym_frames: dict[str, pd.DataFrame],
    *,
    min_common: int = 3,
) -> tuple[dict[str, pd.DataFrame], list[str], pd.DatetimeIndex]:
    """Remove symbols (greedy) until inner timestamp intersection length >= ``min_common``.

    At each step, drop the symbol whose removal yields the largest intersection size.
    Use when a few illiquid or mis-listed names collapse the shared timeline to
    almost nothing on minute bars.
    """

    dropped: list[str] = []
    frames = {k: v for k, v in sym_frames.items() if v is not None and not v.empty}
    while len(frames) >= 1:
        common, aligned = align_symbol_frames(frames)
        if len(common) >= min_common:
            return aligned, dropped, common
        if len(frames) <= 1:
            return aligned, dropped, common
        best_sym: Optional[str] = None
        best_n = -1
        for sym in list(frames.keys()):
            sub = {k: v for k, v in frames.items() if k != sym}
            c_sub, _ = align_symbol_frames(sub)
            n_sub = len(c_sub)
            if n_sub > best_n:
                best_n = n_sub
                best_sym = sym
        if best_sym is None:
            break
        dropped.append(best_sym)
        del frames[best_sym]
    common, aligned = align_symbol_frames(frames)
    return aligned, dropped, common


_OHLC_COLS = ("open", "high", "low", "close")


def _bars_to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by index with UTC-aware DatetimeIndex."""

    d = df.sort_index().copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        raise TypeError("bars index must be DatetimeIndex")
    if d.index.tz is None:
        d.index = d.index.tz_localize(timezone.utc)
    else:
        d.index = d.index.tz_convert(timezone.utc)
    return d


def resolve_master_clock_symbol(
    sym_frames: dict[str, pd.DataFrame],
    *,
    benchmark_symbol: str = "SPY",
) -> str:
    """Pick SPY when present; else benchmark if present; else the symbol with the most rows."""

    if "SPY" in sym_frames:
        df = sym_frames["SPY"]
        if df is not None and not df.empty:
            return "SPY"
    bench = (benchmark_symbol or "SPY").upper()
    if bench in sym_frames:
        df = sym_frames[bench]
        if df is not None and not df.empty:
            return bench
    nonempty = [k for k, v in sym_frames.items() if v is not None and not v.empty]
    if not nonempty:
        raise ValueError("no non-empty symbol frames for master clock resolution")
    return max(nonempty, key=lambda k: len(sym_frames[k]))


def align_symbol_frames_master_clock(
    sym_frames: dict[str, pd.DataFrame],
    *,
    benchmark_symbol: str = "SPY",
    min_bars: int = 11,
) -> tuple[str, pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """Left-align all symbols to a master (SPY) timeline; ffill OHLC; volume 0 for gaps.

    Rows before every symbol has a real print (finite OHLC after ffill) are dropped
    from the left so the replay loop never sees all-NaN prices for thin names.
    """

    if min_bars < 1:
        raise ValueError("min_bars must be >= 1")
    master = resolve_master_clock_symbol(sym_frames, benchmark_symbol=benchmark_symbol)
    master_df = sym_frames[master]
    if master_df is None or master_df.empty:
        raise ValueError(f"master clock symbol {master!r} has no bars")

    m_sorted = _bars_to_utc_index(master_df)
    master_idx = m_sorted.index.unique().sort_values()
    if len(master_idx) == 0:
        raise ValueError("master clock index is empty")

    aligned: dict[str, pd.DataFrame] = {}
    for sym, raw in sym_frames.items():
        if raw is None or raw.empty:
            raise ValueError(f"empty bars for {sym} during master clock alignment")
        d = _bars_to_utc_index(raw)
        for c in ("open", "high", "low", "close", "volume"):
            if c not in d.columns:
                raise ValueError(f"{sym}: missing required column {c!r}")
        out = d.reindex(master_idx).copy()
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).astype(float)
        for c in _OHLC_COLS:
            out[c] = pd.to_numeric(out[c], errors="coerce").ffill()
        bad = out["high"] < out["low"]
        if bad.any():
            pack = out.loc[bad, list(_OHLC_COLS)]
            lo = pack.min(axis=1)
            hi = pack.max(axis=1)
            out.loc[bad, "low"] = lo
            out.loc[bad, "high"] = hi
        aligned[sym] = out

    n = len(master_idx)
    lead = n
    for i in range(n):
        all_ok = True
        for sym in aligned:
            row = aligned[sym].iloc[i]
            try:
                vals = [float(row[c]) for c in _OHLC_COLS]
            except (TypeError, ValueError):
                all_ok = False
                break
            if not all(math.isfinite(v) for v in vals):
                all_ok = False
                break
        if all_ok:
            lead = i
            break

    if lead >= n:
        raise ValueError(
            "master clock alignment: no timestamp has valid OHLC for every symbol after reindex/ffill. "
            "Check for symbols with no overlapping session data vs the master.",
        )

    common = master_idx[lead:]
    aligned_out = {sym: aligned[sym].loc[common].copy() for sym in aligned}

    if len(common) < min_bars:
        raise ValueError(
            f"master clock alignment produced only {len(common)} bars (need >= {min_bars}). "
            f"master={master!r}.",
        )

    return master, common, aligned_out


def _qty_for_entry(settings: Settings, equity: float, ref_px: float) -> float:
    if ref_px <= 0 or equity <= 0:
        return 0.0
    risk_usd = float(settings.MAX_RISK_PER_TRADE_PCT) * float(equity)
    cap = min(risk_usd, float(settings.MAX_EQUITY_USAGE_USD))
    q = cap / ref_px
    if bool(settings.ENABLE_FRACTIONAL):
        return max(1e-6, float(q))
    return float(max(1, int(q)))


def combine_ensemble_signals(signals: list[Signal]) -> list[Signal]:
    """Prefer exits over entries; otherwise pick highest-confidence enter."""

    from collections import defaultdict

    by_sym: dict[str, list[Signal]] = defaultdict(list)
    for s in signals:
        if s.action == SignalAction.NONE:
            continue
        by_sym[s.symbol.upper()].append(s)
    out: list[Signal] = []
    for _sym, lst in by_sym.items():
        exits = [x for x in lst if x.action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)]
        enters = [x for x in lst if x.action == SignalAction.ENTER_LONG]
        if exits:
            out.append(max(exits, key=lambda z: float(z.confidence)))
        elif enters:
            out.append(max(enters, key=lambda z: float(z.confidence)))
    return out


def pick_actionable_signals(signals: list[Signal]) -> list[Signal]:
    """At most one primary action per symbol for the sim layer."""

    from collections import defaultdict

    by_sym: dict[str, list[Signal]] = defaultdict(list)
    for s in signals:
        if s.action == SignalAction.NONE:
            continue
        by_sym[s.symbol.upper()].append(s)
    chosen: list[Signal] = []
    for _sym, lst in by_sym.items():
        ex = next(
            (x for x in lst if x.action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)),
            None,
        )
        if ex is not None:
            chosen.append(ex)
            continue
        ent = next((x for x in lst if x.action == SignalAction.ENTER_LONG), None)
        if ent is not None:
            chosen.append(ent)
    return chosen


@dataclass
class OpenReplayTrade:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: str
    strategy_name: str


@dataclass
class ReplayPortfolioResult:
    portfolio_key: str
    equity_curve: pd.Series
    trades_rows: list[dict[str, Any]] = field(default_factory=list)
    signals_rows: list[dict[str, Any]] = field(default_factory=list)
    orders_rows: list[dict[str, Any]] = field(default_factory=list)
    decisions_rows: list[dict[str, Any]] = field(default_factory=list)
    skips_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HistoricalReplayResult:
    run_id: str
    output_dir: Path
    start: datetime
    end: datetime
    portfolios: dict[str, ReplayPortfolioResult]
    benchmark: pd.Series


class HistoricalReplayEngine:
    """Bar-walking replay with optional SQLite + CSV outputs."""

    def __init__(
        self,
        settings: Settings,
        *,
        symbols: list[str],
        strategy_names: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        initial_equity: float,
        mode: str,
        run_id: str,
        output_dir: Path,
        database: Optional[Database] = None,
        data_feed: str = "iex",
        cache_dir: Optional[Path] = None,
        fill_params: Optional[FillModelParams] = None,
        bars_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
        benchmark_symbol: str = "SPY",
        drop_thin_align: bool = False,
    ) -> None:
        self._settings = settings
        self._symbols = [s.strip().upper() for s in symbols if s.strip()]
        self._strategy_names = [s.strip() for s in strategy_names if s.strip()]
        self._start = start.astimezone(timezone.utc)
        self._end = end.astimezone(timezone.utc)
        self._tf = timeframe
        self._initial = float(initial_equity)
        self._mode = mode.strip().lower()
        self._run_id = run_id.strip()
        self._out = Path(output_dir)
        self._db = database
        self._feed = data_feed
        self._cache = Path(cache_dir) if cache_dir is not None else Path(settings.LOG_DIR) / "replay_cache"
        self._fill = fill_params or FillModelParams()
        self._bars_injected = bars_by_symbol
        self._bench_sym = benchmark_symbol.upper()
        self._drop_thin_align = bool(drop_thin_align)

    def _load_frames(self) -> dict[str, pd.DataFrame]:
        if self._bars_injected is not None:
            return {k.upper(): v.copy() for k, v in self._bars_injected.items()}
        key = (self._settings.ALPACA_API_KEY or "").strip()
        secret = (self._settings.ALPACA_API_SECRET or "").strip()
        if not key or not secret:
            raise ValueError("ALPACA_API_KEY/ALPACA_API_SECRET required to fetch replay bars")
        client = StockHistoricalDataClient(key, secret)
        feed = bt.resolve_data_feed(self._feed)
        frames: dict[str, pd.DataFrame] = {}
        need_syms = set(self._symbols) | {self._bench_sym}
        for sym in sorted(need_syms):
            df = bt.load_or_fetch_bars(
                client=client,
                symbol=sym,
                start=self._start,
                end=self._end,
                timeframe=self._tf,
                feed=feed,
                adjustment=Adjustment.RAW,
                cache_dir=self._cache,
                use_cache=True,
                refresh_cache=False,
            )
            frames[sym] = df
        return frames

    def _write_csv(self, name: str, rows: list[dict[str, Any]]) -> Path:
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_summary(self, portfolios: dict[str, ReplayPortfolioResult], bench: pd.Series) -> Path:
        self._out.mkdir(parents=True, exist_ok=True)
        p = self._out / "replay_summary.md"
        lines = [
            f"# Replay {self._run_id}",
            "",
            f"- window: `{self._start.isoformat()}` .. `{self._end.isoformat()}`",
            f"- timeframe: `{self._tf}`",
            f"- symbols: `{','.join(self._symbols)}`",
            f"- strategies: `{','.join(self._strategy_names)}`",
            f"- mode: `{self._mode}`",
            "",
            "## Portfolios",
            "",
        ]
        for k, pr in portfolios.items():
            final = float(pr.equity_curve.iloc[-1]) if len(pr.equity_curve) else self._initial
            ret = (final / self._initial - 1.0) if self._initial > 0 else 0.0
            lines.append(f"- **{k}**: final_equity={final:.2f} total_return={ret:.4f} n_trades={len(pr.trades_rows)}")
        if len(bench):
            lines.extend(
                [
                    "",
                    "## Benchmark",
                    "",
                    f"- {self._bench_sym} buy-hold final: {float(bench.iloc[-1]):.2f}",
                ],
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def _run_single_portfolio(
        self,
        *,
        portfolio_key: str,
        strategies: list[Strategy],
        aligned: dict[str, pd.DataFrame],
        full_frames: dict[str, pd.DataFrame],
        common: pd.DatetimeIndex,
        bench: pd.Series,
        ensemble_combine: bool,
    ) -> ReplayPortfolioResult:
        warmup = max((s.warmup_lookback() for s in strategies), default=50)
        acct = SimulatedAccount(self._initial)
        broker = SimulatedBroker(acct, fill_params=self._fill, prevent_same_bar_fills=True)
        engine = StrategyEngine(
            strategies,
            settings=self._settings,
            database=self._db,
            signal_source="replay",
            replay_run_id=self._run_id,
        )
        ensemble_engine = WeightedEnsembleEngine(self._settings, database=self._db) if ensemble_combine else None
        open_trade: dict[str, OpenReplayTrade] = {}
        equity_rows: list[dict[str, Any]] = []
        trades_rows: list[dict[str, Any]] = []
        signals_rows: list[dict[str, Any]] = []
        decisions_rows: list[dict[str, Any]] = []
        skips_rows: list[dict[str, Any]] = []
        orders_rows: list[dict[str, Any]] = []

        def skip(sym: str, code: str, msg: str, phase: str = "replay") -> None:
            row = {
                "timestamp": ts_iso,
                "symbol": sym,
                "skip_code": code,
                "message": msg,
                "phase": phase,
                "portfolio": portfolio_key,
            }
            skips_rows.append(row)
            if self._db:
                self._db.record_skip_event(
                    source="replay",
                    timestamp=ts_iso,
                    run_id=self._run_id,
                    symbol=sym,
                    strategy_name=portfolio_key,
                    phase=phase,
                    skip_code=code,
                    message=msg,
                    metadata={"portfolio": portfolio_key},
                )

        n = len(common)
        for i in range(n):
            ts = common[i]
            ts_iso = ts.isoformat() if ts.tzinfo else ts.tz_localize(timezone.utc).isoformat()
            opens = {s: float(aligned[s]["open"].iloc[i]) for s in self._symbols if s in aligned}
            closes = {s: float(aligned[s]["close"].iloc[i]) for s in aligned}
            volumes = {s: float(aligned[s]["volume"].iloc[i]) for s in aligned}

            if i > 0:
                ev = broker.process_bar_open(
                    bar_index=i,
                    open_by_symbol=opens,
                    ts_iso=ts_iso,
                    on_skip=lambda sym, code, msg: skip(sym, code, msg, phase="fill"),
                    volume_by_symbol=volumes,
                )
                for e in ev:
                    if e.get("kind") != "fill":
                        continue
                    sym_u = str(e["symbol"]).upper()
                    if e.get("action") == "enter_long":
                        open_trade[sym_u] = OpenReplayTrade(
                            symbol=sym_u,
                            quantity=float(e["qty"]),
                            entry_price=float(e["price"]),
                            entry_time=ts_iso,
                            strategy_name=portfolio_key,
                        )
                        orders_rows.append(
                            {
                                "timestamp": ts_iso,
                                "portfolio": portfolio_key,
                                "symbol": sym_u,
                                "side": "buy",
                                "qty": e["qty"],
                                "price": e["price"],
                                "reason": "enter",
                            },
                        )
                    elif e.get("action") == "exit_long":
                        ot = open_trade.pop(sym_u, None)
                        if ot is not None:
                            if self._db:
                                qty = float(e["qty"])
                                ep = float(ot.entry_price)
                                xp = float(e["price"])
                                pnl = float(e.get("pnl", 0.0))
                                ret = (xp - ep) / ep if ep else 0.0
                                tid = f"replay_{self._run_id}_{sym_u}_{uuid.uuid4().hex[:10]}"
                                self._db.record_completed_trade(
                                    trade_id=tid,
                                    symbol=sym_u,
                                    side="long",
                                    quantity=qty,
                                    entry_price=ep,
                                    exit_price=xp,
                                    realized_pnl=pnl,
                                    realized_return=ret,
                                    opened_at=ot.entry_time,
                                    closed_at=ts_iso,
                                    strategy_name=ot.strategy_name,
                                    risk_mode=None,
                                    regime_type=None,
                                    sentiment_score=None,
                                    sentiment_label=None,
                                    is_canary=0,
                                    metadata={"portfolio": portfolio_key, "replay_run_id": self._run_id},
                                    source="replay",
                                    replay_run_id=self._run_id,
                                )
                            trades_rows.append(
                                {
                                    "portfolio": portfolio_key,
                                    "symbol": sym_u,
                                    "entry_time": ot.entry_time,
                                    "exit_time": ts_iso,
                                    "qty": e.get("qty"),
                                    "entry_price": ot.entry_price,
                                    "exit_price": e.get("price"),
                                    "pnl": e.get("pnl"),
                                },
                            )
                            orders_rows.append(
                                {
                                    "timestamp": ts_iso,
                                    "portfolio": portfolio_key,
                                    "symbol": sym_u,
                                    "side": "sell",
                                    "qty": e["qty"],
                                    "price": e["price"],
                                    "reason": "exit",
                                },
                            )

            unreal, eq = acct.mark_to_market(closes, ts_iso=ts_iso)
            bench_px = float(bench.loc[ts]) if ts in bench.index else None
            equity_rows.append(
                {
                    "timestamp": ts_iso,
                    "portfolio": portfolio_key,
                    "equity": eq,
                    "cash": acct.cash,
                    "unrealized": unreal,
                    "benchmark": bench_px,
                },
            )
            if self._db:
                gross = 0.0
                for s, pos in acct.positions.items():
                    gross += float(closes.get(s.upper(), pos.avg_entry_price)) * float(pos.quantity)
                self._db.record_equity_snapshot(
                    source="replay",
                    timestamp=ts_iso,
                    run_id=self._run_id,
                    strategy_name=portfolio_key,
                    cash=acct.cash,
                    equity=eq,
                    realized_pnl=acct.realized_pnl,
                    unrealized_pnl=unreal,
                    gross_exposure=gross,
                    net_exposure=gross,
                    benchmark_equity=bench_px,
                    metadata={"portfolio": portfolio_key},
                )

            if i + 1 < warmup:
                continue

            all_slice = {s: aligned[s].iloc[: i + 1] for s in aligned}
            bsym = self._bench_sym
            if bsym in full_frames and bsym not in all_slice:
                all_slice[bsym] = full_frames[bsym].iloc[: i + 1]
            pos_snaps = acct.positions_snapshot(closes)
            acct_snap = acct.account_snapshot(prices=closes, ts=ts)

            if (
                ensemble_combine
                and ensemble_engine is not None
                and self._settings.ENSEMBLE_WEIGHT_MODE == "performance"
            ):
                ensemble_engine.refresh_weights(record_decision=False)

            for sym in self._symbols:
                if sym not in aligned:
                    continue
                df_sym = all_slice[sym]
                if df_sym.empty:
                    continue
                last_c = float(df_sym["close"].iloc[-1])
                spr = max(1e-6, self._fill.spread_pct * last_c)
                q = Quote(
                    symbol=sym,
                    bid=max(1e-9, last_c - spr / 2),
                    ask=last_c + spr / 2,
                    bid_size=1.0,
                    ask_size=1.0,
                    timestamp=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                    feed="replay",
                )
                all_quotes = {
                    s: Quote(
                        symbol=s,
                        bid=max(1e-9, closes[s] * (1 - self._fill.spread_pct / 2)),
                        ask=closes[s] * (1 + self._fill.spread_pct / 2),
                        bid_size=1.0,
                        ask_size=1.0,
                        timestamp=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                        feed="replay",
                    )
                    for s in closes
                }
                ctx = StrategyContext(
                    symbol=sym,
                    bars=df_sym,
                    quote=q,
                    account=acct_snap,
                    positions_by_symbol=pos_snaps,
                    open_order_symbols=set(),
                    now_utc=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                    feed="replay",
                    sentiment_overlay=sentiment_overlay_neutral(sym),
                    all_bars_by_symbol=all_slice,
                    all_quotes_by_symbol=all_quotes,
                )
                raw = engine.evaluate(ctx)
                for s in raw:
                    signals_rows.append(
                        {
                            "timestamp": ts_iso,
                            "portfolio": portfolio_key,
                            "symbol": s.symbol,
                            "strategy_name": s.strategy_name,
                            "action": s.action.value,
                            "confidence": s.confidence,
                            "reason": s.reason,
                        },
                    )
                if ensemble_combine:
                    assert ensemble_engine is not None
                    dec = ensemble_engine.decide(sym, raw, has_position=sym in acct.positions)
                    ens_sig = ensemble_engine.to_signal(dec)
                    sigs = [] if ens_sig.action == SignalAction.NONE else [ens_sig]
                    if self._db:
                        wscore = (
                            float(dec.weighted_exit_score)
                            if dec.final_action
                            in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                            else float(dec.weighted_enter_score)
                        )
                        th = (
                            float(dec.exit_threshold)
                            if dec.final_action
                            in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                            else float(dec.enter_threshold)
                        )
                        self._db.record_strategy_decision(
                            source="replay",
                            timestamp=ts_iso,
                            symbol=sym,
                            final_action=dec.final_action.value,
                            run_id=self._run_id,
                            decision_type="weighted_ensemble",
                            weighted_score=wscore,
                            threshold=th,
                            contributing_signals_json=votes_to_contributing_json(dec.contributing_votes),
                            metadata={
                                "portfolio": portfolio_key,
                                "weighted_enter_score": dec.weighted_enter_score,
                                "weighted_exit_score": dec.weighted_exit_score,
                                "enter_threshold": dec.enter_threshold,
                                "exit_threshold": dec.exit_threshold,
                                "reason": dec.reason,
                            },
                        )
                    decisions_rows.append(
                        {
                            "timestamp": ts_iso,
                            "portfolio": portfolio_key,
                            "symbol": sym,
                            "final": dec.final_action.value,
                            "enter_score": dec.weighted_enter_score,
                            "exit_score": dec.weighted_exit_score,
                        },
                    )
                else:
                    sigs = pick_actionable_signals(raw)

                for sig in sigs:
                    if sig.symbol.upper() != sym:
                        continue
                    execute_at = i + 1
                    if execute_at >= n:
                        skip(sym, "no_next_bar", "signal has no next bar for fill")
                        continue
                    if sig.action == SignalAction.ENTER_LONG:
                        if sym in acct.positions:
                            skip(sym, "already_long", "skip duplicate enter")
                            continue
                        ref = float(sig.reference_price or last_c)
                        qty = _qty_for_entry(self._settings, eq, ref)
                        if qty <= 0:
                            skip(sym, "zero_qty", "sized qty zero")
                            continue
                        broker.schedule(
                            PendingFill(
                                execute_at_bar_index=execute_at,
                                symbol=sym,
                                action=SignalAction.ENTER_LONG,
                                quantity=qty,
                                strategy_name=str(sig.strategy_name or portfolio_key),
                                reason=sig.reason,
                            ),
                        )
                    elif sig.action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG):
                        if sym not in acct.positions:
                            skip(sym, "no_position", "exit without position")
                            continue
                        broker.cancel_symbol_pending(sym)
                        qtyp = float(acct.positions[sym].quantity)
                        broker.schedule(
                            PendingFill(
                                execute_at_bar_index=execute_at,
                                symbol=sym,
                                action=SignalAction.EXIT_LONG,
                                quantity=qtyp,
                                strategy_name=str(sig.strategy_name or portfolio_key),
                                reason=sig.reason,
                            ),
                        )

        eq_series = pd.Series(
            [float(r["equity"]) for r in equity_rows],
            index=pd.to_datetime([r["timestamp"] for r in equity_rows], utc=True),
            name=portfolio_key,
        )
        return ReplayPortfolioResult(
            portfolio_key=portfolio_key,
            equity_curve=eq_series,
            trades_rows=trades_rows,
            signals_rows=signals_rows,
            orders_rows=orders_rows,
            decisions_rows=decisions_rows,
            skips_rows=skips_rows,
        )

    def run(self) -> HistoricalReplayResult:
        frames = self._load_frames()
        for s in self._symbols:
            if s not in frames or frames[s].empty:
                raise ValueError(f"missing bars for symbol {s}")
        bench_sym = self._bench_sym
        if bench_sym not in frames or frames[bench_sym].empty:
            raise ValueError(f"missing benchmark bars for {bench_sym}")

        sym_frames = {s: frames[s] for s in self._symbols}
        if self._drop_thin_align:
            _LOG.warning(
                "event=replay_drop_thin_align_deprecated "
                "msg=master_clock_left_join_supersedes_inner_join_greedy_drop_flag_is_ignored",
            )
        try:
            master_used, common, aligned = align_symbol_frames_master_clock(
                sym_frames,
                benchmark_symbol=bench_sym,
                min_bars=11,
            )
        except ValueError as exc:
            try:
                ref = resolve_master_clock_symbol(sym_frames, benchmark_symbol=bench_sym)
            except ValueError:
                ref = "SPY"
            detail = describe_bar_alignment(sym_frames, ref_symbol=ref)
            raise ValueError(f"{exc}\n{detail}") from exc

        _LOG.info(
            "event=replay_master_clock master=%s common_bars=%d symbols=%s",
            master_used,
            len(common),
            ",".join(self._symbols),
        )

        bench_close = frames[bench_sym].reindex(common)["close"].ffill()
        bench = buy_hold_equity_curve(bench_close, initial_equity=self._initial)

        self._out.mkdir(parents=True, exist_ok=True)
        cfg = {
            "run_id": self._run_id,
            "symbols": self._symbols,
            "strategies": self._strategy_names,
            "start": self._start.isoformat(),
            "end": self._end.isoformat(),
            "timeframe": self._tf,
            "initial_equity": self._initial,
            "mode": self._mode,
            "feed": self._feed,
        }
        (self._out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        if self._db:
            self._db.create_replay_run(
                run_id=self._run_id,
                start_time=self._start.isoformat(),
                end_time=self._end.isoformat(),
                timeframe=self._tf,
                symbols_json=json.dumps(self._symbols),
                strategies_json=json.dumps(self._strategy_names),
                mode=self._mode,
                initial_equity=self._initial,
                data_feed=self._feed,
                benchmark_symbol=bench_sym,
                settings_json=json.dumps(cfg),
                status="running",
            )

        portfolios: dict[str, ReplayPortfolioResult] = {}
        try:
            if self._mode in ("independent", "both"):
                for name in self._strategy_names:
                    sts = build_strategies([name], self._settings)
                    pk = f"ind::{name}"
                    portfolios[pk] = self._run_single_portfolio(
                        portfolio_key=pk,
                        strategies=sts,
                        aligned=aligned,
                        full_frames=frames,
                        common=common,
                        bench=bench,
                        ensemble_combine=False,
                    )
            if self._mode in ("ensemble", "both"):
                sts = build_strategies(self._strategy_names, self._settings)
                pk = "ensemble"
                portfolios[pk] = self._run_single_portfolio(
                    portfolio_key=pk,
                    strategies=sts,
                    aligned=aligned,
                    full_frames=frames,
                    common=common,
                    bench=bench,
                    ensemble_combine=True,
                )
            if self._mode not in ("independent", "ensemble", "both"):
                raise ValueError(f"unknown replay mode: {self._mode}")

            for pk, pr in portfolios.items():
                pk_safe = pk.replace(":", "_").replace("/", "_")
                eq_df = pr.equity_curve.reset_index()
                eq_df.columns = ["timestamp", "equity"]
                self._write_csv(f"equity__{pk_safe}.csv", eq_df.to_dict("records"))
                self._write_csv(f"trades__{pk_safe}.csv", pr.trades_rows)
                self._write_csv(f"signals__{pk_safe}.csv", pr.signals_rows)
                self._write_csv(f"orders__{pk_safe}.csv", pr.orders_rows)
                self._write_csv(f"decisions__{pk_safe}.csv", pr.decisions_rows)
                self._write_csv(f"skips__{pk_safe}.csv", pr.skips_rows)

            merged: Optional[pd.DataFrame] = None
            for pk, pr in portfolios.items():
                pk_safe = pk.replace(":", "_").replace("/", "_")
                col = pr.equity_curve.rename(pk_safe).reset_index()
                col.columns = ["timestamp", pk_safe]
                merged = col if merged is None else merged.merge(col, on="timestamp", how="outer")
            if merged is not None and not merged.empty:
                merged = merged.sort_values("timestamp")
                self._write_csv("equity_curve.csv", merged.to_dict("records"))

            bench_rows = [{"timestamp": ix.isoformat(), "benchmark_equity": float(v)} for ix, v in bench.items()]
            self._write_csv("benchmark.csv", bench_rows)
            self._write_summary(portfolios, bench)

            if self._db:
                self._db.finish_replay_run(run_id=self._run_id, status="completed", error=None)
        except Exception as exc:  # noqa: BLE001
            if self._db:
                self._db.finish_replay_run(run_id=self._run_id, status="failed", error=str(exc))
            raise

        return HistoricalReplayResult(
            run_id=self._run_id,
            output_dir=self._out,
            start=self._start,
            end=self._end,
            portfolios=portfolios,
            benchmark=bench,
        )


__all__ = [
    "HistoricalReplayEngine",
    "HistoricalReplayResult",
    "ReplayPortfolioResult",
    "align_symbol_frames",
    "align_symbol_frames_master_clock",
    "describe_bar_alignment",
    "greedy_drop_symbols_for_alignment",
    "resolve_master_clock_symbol",
    "combine_ensemble_signals",
    "pick_actionable_signals",
    "resolve_replay_window",
]
