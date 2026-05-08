#!/usr/bin/env python3
"""Standalone RSI backtest harness: reuses live strategy/risk/universe code without Alpaca or Discord."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import uuid
from dataclasses import dataclass, asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.settings import Settings
from config.strategy_runtime import merge_strategy_thresholds  # noqa: E402
from core.account import AccountSnapshot, PositionSnapshot  # noqa: E402
from core.market_data import Quote  # noqa: E402
from risk.compliance import ComplianceAdapter  # noqa: E402
from risk.exposure import ExposureChecker  # noqa: E402
from risk.position_sizer import PositionSizer  # noqa: E402
from risk.anti_martingale import resolve_anti_martingale  # noqa: E402
from strategies.base import SignalAction, StrategyContext  # noqa: E402
from strategies.rsi_strategy import RSIMeanReversionStrategy  # noqa: E402
from strategies.sentiment import sentiment_overlay_neutral  # noqa: E402
from strategies.scanner import symbols_for_strategy_ticks  # noqa: E402
from strategies.universe import UniverseFilter  # noqa: E402
from utils.ids import generate_client_order_id  # noqa: E402
from utils.price_utils import round_to_tick  # noqa: E402

_LOG = logging.getLogger("backtest.sim")
_BARS_PER_TRADING_YEAR = 252 * 78  # aligned with spec (5‑minute bars, regular session heuristic)


class MockDiscordSimulator:
    """Console-only Discord stand-in."""

    def print_embed(self, title: str, body: dict[str, Any]) -> None:
        lines = "\n".join(f"{k}: {v}" for k, v in body.items())
        print("\n================ SIMULATION EMBED =================")
        print(f"Title: {title}")
        print(lines)
        print("==================================================\n")


@dataclass
class SimulationOrderRecord:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: int
    limit_price: float
    status: str
    submitted_at: str
    filled_at: str
    filled_avg_price: str
    reason: str


@dataclass
class PendingLimit:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    qty: int
    limit_price: float
    submitted_ts: datetime
    bars_alive: int
    reason: str
    kind: Literal["entry", "exit", "emergency"]

    def __post_init__(self) -> None:
        if self.submitted_ts.tzinfo is None:
            raise ValueError(
                f"PendingLimit.submitted_ts must be tz-aware; got naive datetime for {self.symbol}"
            )


@dataclass
class SimLot:
    qty: float
    avg_px: float
    entry_time: datetime
    entry_reason: str
    trade_id: int = 0


class SimulatedAccountBook:
    """Cash + positions + rounding fees (long-only simulator)."""

    def __init__(self, *, initial_equity: float, commission_bps: float) -> None:
        self.initial_equity = float(initial_equity)
        self._cash = float(initial_equity)
        self._commission_bps = float(commission_bps)
        self.positions: dict[str, SimLot] = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self._mark: dict[str, float] = {}

    def mark_prices(self, prices: dict[str, float]) -> None:
        self._mark = {k.upper(): float(v) for k, v in prices.items()}

    def unrealized(self) -> float:
        px = self._mark
        u = 0.0
        for sym, lot in self.positions.items():
            m = px.get(sym, lot.avg_px)
            u += (m - lot.avg_px) * lot.qty
        return u

    def equity(self) -> float:
        mkt = sum(self._mark.get(sym, lot.avg_px) * lot.qty for sym, lot in self.positions.items())
        return self._cash + mkt

    def gross_exposure(self) -> float:
        return sum(self._mark.get(sym, lot.avg_px) * lot.qty for sym, lot in self.positions.items())

    def _fee(self, notional: float) -> float:
        return abs(notional) * self._commission_bps / 10_000.0

    def apply_buy(
        self, *, symbol: str, qty: float, px: float, ts: datetime, reason: str, trade_id: int,
    ) -> None:
        sym = symbol.upper()
        gross = qty * px
        fee = self._fee(gross)
        cost = gross + fee
        if cost > self._cash + 1e-9:
            raise RuntimeError(f"Insufficient cash buy {sym} cost={cost} cash={self._cash}")
        self._cash -= cost
        prev = self.positions.get(sym)
        if prev is None:
            self.positions[sym] = SimLot(
                qty=float(qty), avg_px=float(px), entry_time=ts, entry_reason=reason,
                trade_id=int(trade_id),
            )
            return
        new_qty = prev.qty + float(qty)
        new_avg = (prev.avg_px * prev.qty + float(px) * float(qty)) / new_qty
        self.positions[sym] = SimLot(
            qty=float(new_qty), avg_px=float(new_avg), entry_time=prev.entry_time,
            entry_reason=prev.entry_reason, trade_id=int(prev.trade_id),
        )

    def apply_sell(
        self, *, symbol: str, qty: float, px: float,
    ) -> tuple[float, str, datetime, float, float, int]:
        sym = symbol.upper()
        lot = self.positions.get(sym)
        if lot is None or qty - lot.qty > 1e-9:
            raise RuntimeError(f"Cannot sell qty={qty} on {sym} lot={lot}")
        gross = qty * px
        fee = self._fee(gross)
        proceeds = gross - fee
        self._cash += proceeds
        pnl_units = qty * (px - lot.avg_px) - fee
        self.realized_pnl += pnl_units
        self.total_fees += fee
        entry_reason = lot.entry_reason
        entry_time = lot.entry_time
        entry_px = lot.avg_px
        trade_id = int(lot.trade_id)
        if qty >= lot.qty - 1e-9:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = SimLot(
                qty=float(lot.qty - qty), avg_px=float(lot.avg_px),
                entry_time=lot.entry_time, entry_reason=lot.entry_reason,
                trade_id=int(lot.trade_id),
            )
        return pnl_units, entry_reason, entry_time, float(entry_px), float(qty), trade_id


def synthetic_quote_from_bar(
    *,
    symbol: str,
    close_px: float,
    ts: datetime,
    spread_bps: float,
    feed: str = "sim",
) -> Quote:
    half = spread_bps / 20_000.0
    bid = max(1e-6, close_px * (1 - half))
    ask = close_px * (1 + half)
    return Quote(
        symbol=symbol.upper(),
        bid=float(bid),
        ask=float(ask),
        bid_size=1.0,
        ask_size=1.0,
        timestamp=ts,
        feed=feed,
    )


def limit_buy_px(q: Quote) -> float:
    sp = q.ask - q.bid
    return float(round_to_tick(q.bid + 0.25 * sp, mode="down"))


def limit_sell_px(q: Quote) -> float:
    sp = q.ask - q.bid
    return float(round_to_tick(q.ask - 0.25 * sp, mode="up"))


def emergency_sell_px(q: Quote, aggressiveness_pct: float) -> float:
    mid = (q.bid + q.ask) / 2.0
    off = abs(aggressiveness_pct) * mid
    raw = max(0.01, q.bid - off)
    return float(round_to_tick(raw, mode="down"))


def snapshot_from_book(book: SimulatedAccountBook) -> tuple[AccountSnapshot, list[PositionSnapshot]]:
    eq = book.equity()
    cash = book._cash
    rows: list[PositionSnapshot] = []
    px = book._mark
    for sym, lot in book.positions.items():
        mx = px.get(sym, lot.avg_px)
        mv = lot.qty * mx
        u = (mx - lot.avg_px) * lot.qty
        rows.append(
            PositionSnapshot(
                symbol=sym,
                qty=lot.qty,
                avg_entry_price=lot.avg_px,
                side="long",
                market_value=mv,
                cost_basis=lot.qty * lot.avg_px,
                unrealized_pl=u,
                current_price=float(mx),
            ),
        )
    long_mv = sum(p.market_value for p in rows)
    acct = AccountSnapshot(
        equity=float(eq),
        last_equity=float(eq),
        cash=float(cash),
        buying_power=float(cash),
        regt_buying_power=float(max(cash, 0)),
        portfolio_value=float(eq),
        long_market_value=float(long_mv),
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=1.0,
        status="SIM",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
        fetched_at=datetime.now(UTC),
    )
    return acct, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-fidelity RSI backtest (no live broker / Discord).")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/backtests/5m")
    p.add_argument("--symbols", type=str, nargs="+", default=["SPY"])
    p.add_argument("--start", type=str, default="2024-06-03")
    p.add_argument("--end", type=str, default="2024-08-01")
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports/backtests")
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--warmup-bars", type=int, default=250)
    p.add_argument("--save-trades", action="store_true", default=True)
    p.add_argument("--no-save-trades", action="store_true")
    p.add_argument("--save-equity-curve", action="store_true", default=True)
    p.add_argument("--no-save-equity-curve", action="store_true")
    args_ns = p.parse_args()
    if args_ns.no_save_trades:
        args_ns.save_trades = False
    if args_ns.no_save_equity_curve:
        args_ns.save_equity_curve = False
    return args_ns


def load_symbol_csv(sym: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file for {sym}: expected {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for need in ("timestamp", "open", "high", "low", "close", "volume"):
        if need not in df.columns:
            raise ValueError(f"{path}: missing column {need!r}")
    if "symbol" not in df.columns:
        df["symbol"] = sym.upper()
    ts_raw = pd.to_datetime(df["timestamp"].iloc[0])
    utc_flag = getattr(ts_raw, "tzinfo", None) is not None
    if utc_flag:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False).dt.tz_localize(
            UTC, ambiguous="infer", nonexistent="shift_forward",
        )
    numeric = df[["open", "high", "low", "close"]].astype(float)
    if (numeric.values < -1e-12).any():
        raise ValueError(f"{sym}: invalid negative OHLC")
    df["volume"] = df["volume"].fillna(0).astype(float)
    if df["volume"].lt(0).any():
        raise ValueError(f"{sym}: negative volume")
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{sym}: high < low rows present")
    for c in ["open", "close"]:
        if ((df[c] > df["high"]) | (df[c] < df["low"])).any():
            raise ValueError(f"{sym}: {c} violates high/low bounds")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df


def build_settings_for_sim(
    *,
    symbols: list[str],
    initial_equity: float,
    state_dir: Path,
    database_path: Path,
) -> Settings:
    return Settings(
        _env_file=None,
        ALPACA_API_KEY="simulation",
        ALPACA_API_SECRET="simulation",
        ALPACA_ENV="paper",
        ALPACA_FEED="iex",
        LIVE_TRADING_ENABLED=False,
        DRY_RUN=True,
        SYMBOLS=",".join(s.upper() for s in symbols),
        STATE_DIR=state_dir,
        DATABASE_PATH=database_path,
        LOG_DIR=state_dir / "logs",
        REPORTS_DIR=state_dir / "reports",
        ENABLE_DISCORD_BOT=False,
        ENABLE_ML_FILTER=False,
        ENABLE_AUTOTUNE=False,
        ENABLE_KELLY_SIZING=False,
        SENTIMENT_ENABLED=False,
        PASSIVE_JOINER_ENABLED=False,
        DYNAMIC_UNIVERSE_ENABLED=False,
        CORRELATION_BREAKER_ENABLED=False,
        BLACK_SWAN_ENABLED=False,
        QUOTE_STALENESS_SECONDS=300.0,
        MIN_AVG_DOLLAR_VOLUME=0.0,
        MAX_EQUITY_USAGE_USD=max(float(initial_equity), 50.0),
        RUN_LIVE_CANARY_ON_STARTUP=False,
        REGULATORY_MODE="intraday_margin",
    )





def union_timestamps(dfs_map: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    u: pd.DatetimeIndex | None = None
    for df in dfs_map.values():
        u = df.index if u is None else u.union(df.index)
    assert u is not None
    return u.sort_values()


def process_pending_orders(
    *,
    pending: list[PendingLimit],
    ts: datetime,
    rowslice: dict[str, pd.Series],
    slippage_bps: float,
    order_timeout_bars: int,
    book: SimulatedAccountBook,
    order_rows: list[SimulationOrderRecord],
    closed_trades: list[dict[str, Any]],
    discord: MockDiscordSimulator,
    trade_counter: list[int],
    bar_delta_s: float = 5 * 60,
) -> None:
    slip = abs(float(slippage_bps))

    def max_wait(po: PendingLimit) -> int:
        return 1 if po.kind == "emergency" else int(order_timeout_bars)

    def record_cancel(po: PendingLimit, suffix: str) -> None:
        order_rows.append(
            SimulationOrderRecord(
                order_id=po.order_id,
                client_order_id=po.client_order_id,
                symbol=po.symbol,
                side=po.side,
                qty=po.qty,
                limit_price=po.limit_price,
                status="expired_" + suffix,
                submitted_at=po.submitted_ts.isoformat(),
                filled_at="",
                filled_avg_price="",
                reason=po.reason,
            ),
        )

    next_pending: list[PendingLimit] = []
    for po in pending:
        if po.submitted_ts >= ts:
            # Same-bar fill would peek at OHLC of the bar that produced the
            # signal — only fill from the next bar onward.
            next_pending.append(po)
            continue
        row = rowslice.get(po.symbol.upper())
        if row is None:
            po.bars_alive += 1
            if po.bars_alive >= max_wait(po):
                record_cancel(po, "missing_bar_timeout")
            else:
                next_pending.append(po)
            continue

        high_px = float(row["high"])
        low_px = float(row["low"])
        sy = po.side.lower()

        if sy == "buy" and low_px <= po.limit_price + 1e-12:
            fill_px = float(po.limit_price) + slip * abs(float(po.limit_price)) / 10_000.0
            tid = int(trade_counter[0])
            trade_counter[0] += 1
            book.apply_buy(
                symbol=po.symbol,
                qty=float(po.qty),
                px=float(fill_px),
                ts=ts,
                reason=po.reason,
                trade_id=tid,
            )
            order_rows.append(
                SimulationOrderRecord(
                    order_id=po.order_id,
                    client_order_id=po.client_order_id,
                    symbol=po.symbol,
                    side="buy",
                    qty=po.qty,
                    limit_price=float(po.limit_price),
                    status="filled",
                    submitted_at=po.submitted_ts.isoformat(),
                    filled_at=ts.isoformat(),
                    filled_avg_price=f"{fill_px:.6f}",
                    reason=po.reason,
                ),
            )
            discord.print_embed(
                "Simulated BUY fill",
                {"symbol": po.symbol, "qty": po.qty, "px": fill_px},
            )
            continue

        if sy == "sell" and high_px >= po.limit_price - 1e-12:
            fill_px = float(po.limit_price) - slip * abs(float(po.limit_price)) / 10_000.0
            rp, er, et, ep, qtys, tid = book.apply_sell(
                symbol=po.symbol,
                qty=float(po.qty),
                px=float(fill_px),
            )
            hb = max(1, int((ts - et).total_seconds() / max(bar_delta_s, 1.0))) if isinstance(et, datetime) else 1
            rret = ((fill_px - ep) / ep) if ep > 1e-12 else math.nan
            closed_trades.append(
                {
                    "trade_id": tid,
                    "symbol": po.symbol.upper(),
                    "side": "long_roundtrip_close",
                    "qty": qtys,
                    "entry_time": et.isoformat() if isinstance(et, datetime) else "",
                    "entry_price": ep,
                    "exit_time": ts.isoformat(),
                    "exit_price": fill_px,
                    "realized_pnl": rp,
                    "realized_return": float(rret) if math.isfinite(rret) else "",
                    "holding_bars": hb,
                    "entry_reason": er,
                    "exit_reason": po.reason,
                },
            )
            order_rows.append(
                SimulationOrderRecord(
                    order_id=po.order_id,
                    client_order_id=po.client_order_id,
                    symbol=po.symbol,
                    side="sell",
                    qty=po.qty,
                    limit_price=float(po.limit_price),
                    status="filled",
                    submitted_at=po.submitted_ts.isoformat(),
                    filled_at=ts.isoformat(),
                    filled_avg_price=f"{fill_px:.6f}",
                    reason=po.reason,
                ),
            )
            discord.print_embed(
                "Simulated SELL fill",
                {"symbol": po.symbol, "qty": po.qty, "px": fill_px},
            )
            continue

        po.bars_alive += 1
        if po.bars_alive >= max_wait(po):
            record_cancel(po, "limit_timeout")
        else:
            next_pending.append(po)

    pending.clear()
    pending.extend(next_pending)


def run_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[SimulationOrderRecord], list[dict[str, Any]]]:
    print("\nWARNING: SIMULATION MODE — no live orders or Discord API calls.\n")

    discord = MockDiscordSimulator()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir is not a directory: {data_dir}")
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / f"_scratch_{uuid.uuid4().hex[:12]}"
    scratch.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols]

    dfs_raw: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        dfs_raw[sym] = load_symbol_csv(sym, data_dir / f"{sym}.csv")

    start = pd.Timestamp(args.start, tz=UTC).to_pydatetime()
    end = pd.Timestamp(args.end, tz=UTC).to_pydatetime()

    start_data = min(df.index.min() for df in dfs_raw.values()).to_pydatetime()
    _warm_anchor = pd.Timestamp(min(start, start_data)) - timedelta(minutes=5 * int(args.warmup_bars + 120))
    warmup_start = _warm_anchor.to_pydatetime()
    if warmup_start.tzinfo is None:
        warmup_start = warmup_start.replace(tzinfo=UTC)

    dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        clipped = dfs_raw[sym][(dfs_raw[sym].index >= warmup_start) & (dfs_raw[sym].index <= end)].copy()
        if clipped.empty:
            raise RuntimeError(f"{sym}: empty after clip — check start/end versus file coverage.")
        dfs[sym] = clipped

    u_idx = union_timestamps(dfs)
    filtered_ts: list[datetime] = []
    for raw in u_idx:
        tt = pd.Timestamp(raw).to_pydatetime()
        if tt.tzinfo is None:
            tt = tt.replace(tzinfo=UTC)
        if warmup_start <= tt <= end:
            filtered_ts.append(tt)

    settings = build_settings_for_sim(
        symbols=symbols,
        initial_equity=float(args.initial_equity),
        state_dir=scratch,
        database_path=scratch / "sim.sqlite",
    )

    strat_rt = merge_strategy_thresholds(settings)
    strategy = RSIMeanReversionStrategy(
        settings,
        state_store=None,
        database=None,
        runtime_thresholds=strat_rt,
        ml_filter=None,
    )
    need_bars_strategy = strategy.warmup_lookback()
    need_eff = max(need_bars_strategy, int(args.warmup_bars))
    for sym in symbols:
        n_through_start = len(dfs[sym].loc[dfs[sym].index <= start])
        if n_through_start < need_eff:
            raise RuntimeError(
                f"Insufficient warmup data for {sym}: {n_through_start} bars through start={start.isoformat()} "
                f"(need {need_eff}; strategy_lookback={need_bars_strategy}, --warmup-bars={args.warmup_bars})",
            )
    discord.print_embed(
        "Simulation startup",
        {
            "mode": "simulation_only",
            "symbols": ",".join(symbols),
            "start": args.start,
            "end": args.end,
            "warmup_eff": need_eff,
            "initial_equity": args.initial_equity,
            "spread_bps": args.spread_bps,
            "slippage_bps": args.slippage_bps,
            "commission_bps": args.commission_bps,
        },
    )

    compliance = ComplianceAdapter(settings)
    exposure_x = ExposureChecker(settings)
    sizer = PositionSizer(settings, compliance, exposure_x, database=None)
    univ = UniverseFilter(settings)

    book = SimulatedAccountBook(initial_equity=float(args.initial_equity), commission_bps=float(args.commission_bps))

    pending: list[PendingLimit] = []
    order_rows: list[SimulationOrderRecord] = []
    trade_counter = [1000]
    closed_trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, float | str]] = []
    warnings: list[str] = []

    bar_duration = timedelta(minutes=5)
    order_timeout_bars = max(1, int(math.ceil(settings.ORDER_TIMEOUT_SECONDS / bar_duration.total_seconds())))

    peak_eq = float(args.initial_equity)
    bars_in_position = 0

    warmup_started_logged = False
    strategy_ready_ts: datetime | None = None

    for ts in filtered_ts:
        rowslice: dict[str, pd.Series] = {}
        for sym in symbols:
            if ts in dfs[sym].index:
                rowslice[sym] = dfs[sym].loc[ts]

        if pending:
            process_pending_orders(
                pending=pending,
                ts=ts,
                rowslice=rowslice,
                slippage_bps=float(args.slippage_bps),
                order_timeout_bars=order_timeout_bars,
                book=book,
                order_rows=order_rows,
                closed_trades=closed_trades,
                discord=discord,
                trade_counter=trade_counter,
            )

        book.mark_prices({s: float(r.close) for s, r in rowslice.items()})

        equity_pre = book.equity()
        peak_eq = max(peak_eq, equity_pre)

        dd_pct = equity_pre / peak_eq - 1.0 if peak_eq > 0 else 0.0

        unreal = book.unrealized()
        equity_rows.append(
            {
                "timestamp": ts.isoformat(),
                "equity": equity_pre,
                "cash": book._cash,
                "unrealized_pnl": unreal,
                "realized_pnl": book.realized_pnl,
                "gross_exposure": book.gross_exposure(),
                "drawdown_pct": dd_pct * 100.0,
            },
        )

        if book.positions:
            bars_in_position += 1

        merged_open = {po.symbol.upper() for po in pending}

        broker_syms = {sym for sym, lp in book.positions.items()}
        tick_syms = symbols_for_strategy_ticks(settings, scanned=None, broker_position_symbols=broker_syms)

        bars_cache = {sym: dfs[sym].loc[:ts] for sym in symbols}

        warmup_ok_every = True
        for sym in symbols:
            if ts < start:
                warmup_ok_every = False
                break
            n_b = len(dfs[sym].loc[dfs[sym].index <= ts])
            if n_b < need_eff:
                warmup_ok_every = False
                if not warmup_started_logged:
                    _LOG.info("event=warmup_started need_bars=%s per_symbol=%s", need_eff, sym)
                    warmup_started_logged = True
                warnings.append(f"symbol_warmup_insufficient pending {sym} have={n_b} need={need_eff}")

        if warmup_ok_every and strategy_ready_ts is None and ts >= start:
            strategy_ready_ts = ts
            _LOG.info("event=warmup_completed first_tradable_ts=%s", ts.isoformat())

        acct, pos_list = snapshot_from_book(book)
        comp = compliance.decide(acct, reference_date=ts.date())

        positions_by_symbol = {p.symbol.upper(): p for p in pos_list}
        open_order_symbols = set(merged_open)

        tick_set_upper = {s.upper() for s in tick_syms}
        bot_managed_notional = sum(
            abs(p.market_value) for p in pos_list if p.symbol.upper() in tick_set_upper
        )

        tick_recent: list[Any] = []
        t_mode, t_mult, _t_r = resolve_anti_martingale(settings, tick_recent)
        t_preview = ""

        for sym in tick_syms:
            if sym not in rowslice:
                continue
            bars = bars_cache[sym]
            last_row = rowslice[sym]
            close_px = float(last_row.close)
            q = synthetic_quote_from_bar(
                symbol=sym, close_px=close_px, ts=ts, spread_bps=float(args.spread_bps),
            )

            elig = univ.is_eligible(
                sym,
                quote=q,
                bars=bars,
                has_position=sym in positions_by_symbol,
                has_open_order=sym in open_order_symbols,
            )

            ctx = StrategyContext(
                symbol=sym,
                bars=bars,
                quote=q,
                account=acct,
                positions_by_symbol=positions_by_symbol,
                open_order_symbols=open_order_symbols,
                now_utc=ts,
                feed="sim",
                sentiment_overlay=sentiment_overlay_neutral(sym),
                anti_martingale_risk_mode=t_mode.value,
                anti_martingale_multiplier=t_mult,
                recent_trade_outcomes_hint=t_preview,
            )

            if not warmup_ok_every or ts < start:
                continue

            can_open = True
            can_exit = True

            for signal in strategy.evaluate(ctx):
                if signal.action == SignalAction.NONE:
                    continue
                if signal.action == SignalAction.ENTER_LONG:
                    if not can_open or not comp.allow_new_entries or not elig.eligible:
                        continue
                    if q is None:
                        continue
                    conv_mult = float(signal.metadata.get("conviction_risk_multiplier", 1.0) or 1.0)
                    sizing_block = None
                    sizing = sizer.size(
                        symbol=sym,
                        entry_price=float(signal.reference_price or q.bid),
                        atr=float(signal.atr),
                        account=acct,
                        positions=list(pos_list),
                        bot_managed_notional=bot_managed_notional,
                        conviction_risk_multiplier=conv_mult,
                        sizing_block_reason=sizing_block,
                        anti_martingale_multiplier=float(t_mult),
                        risk_mode=t_mode.value,
                        recent_trade_hint=t_preview,
                    )
                    if sizing.shares < 1:
                        continue
                    if any(po.symbol.upper() == sym.upper() for po in pending):
                        continue
                    lim_px = limit_buy_px(q)
                    cid = generate_client_order_id(strategy.name, sym, "buy")
                    oid = f"sim-{uuid.uuid4().hex[:10]}"
                    pending.append(
                        PendingLimit(
                            order_id=oid,
                            client_order_id=cid,
                            symbol=sym,
                            side="buy",
                            qty=int(sizing.shares),
                            limit_price=float(lim_px),
                            submitted_ts=ts,
                            bars_alive=0,
                            reason=str(signal.reason)[:380],
                            kind="entry",
                        ),
                    )
                    order_rows.append(
                        SimulationOrderRecord(
                            order_id=oid,
                            client_order_id=cid,
                            symbol=sym,
                            side="buy",
                            qty=int(sizing.shares),
                            limit_price=float(lim_px),
                            status="accepted_sim",
                            submitted_at=ts.isoformat(),
                            filled_at="pending",
                            filled_avg_price="",
                            reason=str(signal.reason)[:380],
                        ),
                    )
                    discord.print_embed(
                        "ENTER LONG (sim queued)",
                        {
                            "symbol": sym,
                            "qty": int(sizing.shares),
                            "limit": lim_px,
                            "reason": signal.reason[:200],
                        },
                    )
                elif signal.action == SignalAction.EXIT_LONG:
                    if not can_exit:
                        continue
                    pos = positions_by_symbol.get(sym.upper())
                    if pos is None or q is None:
                        continue
                    qty = int(abs(pos.qty))
                    if qty < 1:
                        continue
                    if any(po.symbol.upper() == sym.upper() for po in pending):
                        continue
                    lim_px = limit_sell_px(q)
                    cid = generate_client_order_id(strategy.name, sym, "sell")
                    oid = f"sim-{uuid.uuid4().hex[:10]}"
                    pending.append(
                        PendingLimit(
                            order_id=oid,
                            client_order_id=cid,
                            symbol=sym,
                            side="sell",
                            qty=qty,
                            limit_price=float(lim_px),
                            submitted_ts=ts,
                            bars_alive=0,
                            reason=str(signal.reason)[:380],
                            kind="exit",
                        ),
                    )
                    discord.print_embed("EXIT LONG (sim queued)", {"symbol": sym, "qty": qty})
                elif signal.action == SignalAction.EMERGENCY_EXIT_LONG:
                    pos = positions_by_symbol.get(sym.upper())
                    if pos is None or q is None:
                        continue
                    qty = int(abs(pos.qty))
                    if qty < 1:
                        continue
                    lim_px = emergency_sell_px(q, settings.EMERGENCY_AGGRESSIVENESS_PCT)
                    cid = generate_client_order_id(strategy.name + "EMG", sym, "sell")
                    oid = f"sim-{uuid.uuid4().hex[:10]}"
                    pending.append(
                        PendingLimit(
                            order_id=oid,
                            client_order_id=cid,
                            symbol=sym,
                            side="sell",
                            qty=qty,
                            limit_price=float(lim_px),
                            submitted_ts=ts,
                            bars_alive=0,
                            reason=str(signal.reason)[:380],
                            kind="emergency",
                        ),
                    )
                    discord.print_embed("EMERGENCY EXIT (IOC sim)", {"symbol": sym, "qty": qty})

    eq_df = pd.DataFrame(equity_rows)
    meta = summarize_performance(eq_df, closed_trades, float(args.initial_equity), filtered_ts)
    meta["warnings"] = warnings[:200]
    ticks = max(len(filtered_ts), 1)
    meta["exposure_time_pct"] = 100.0 * float(bars_in_position) / ticks
    return eq_df, meta, warnings, order_rows, closed_trades


def summarize_performance(eq_df: pd.DataFrame, trades: list[dict], initial: float, ts_list: list[datetime]) -> dict[str, Any]:
    eq = eq_df["equity"].astype(float)
    total_ret = (eq.iloc[-1] / float(initial)) - 1 if len(eq.index) > 1 else 0.0

    yrs = len(ts_list) / max(1e-12, float(_BARS_PER_TRADING_YEAR))
    if yrs <= 1e-18 or (1 + total_ret) <= 0:
        ann = total_ret
    else:
        log_ann = math.log1p(total_ret) / yrs
        ann = math.exp(min(700.0, max(-700.0, log_ann))) - 1
    if not math.isfinite(ann):
        ann = total_ret

    rets = eq.pct_change().dropna().to_numpy(dtype=float)
    std = float(np.std(rets)) if len(rets) > 1 else 0.0
    mu = float(np.mean(rets)) if len(rets) else 0.0

    sharpe = 0.0
    if std > 1e-15:
        sharpe = math.sqrt(_BARS_PER_TRADING_YEAR) * (mu / std)

    downside = rets[rets < 0]
    dstd = float(np.std(downside)) if len(downside) > 1 else 0.0
    sortino = 0.0
    if dstd > 1e-15:
        sortino = math.sqrt(_BARS_PER_TRADING_YEAR) * (mu / dstd)

    peak = eq.cummax()
    dd = eq / peak - 1
    max_dd_pct = float(dd.min())

    finalized = [
        t for t in trades
        if "entry_price" in t
        and "exit_price" in t
        and t.get("realized_pnl") is not None
    ]
    wins = [t for t in finalized if float(t.get("realized_pnl") or 0) > 0]
    losses = [t for t in finalized if float(t.get("realized_pnl") or 0) < 0]
    nw, nl = len(wins), len(losses)
    win_rate = nw / max(1, (nw + nl))
    gw = sum(float(t["realized_pnl"]) for t in wins)
    gl_abs = sum(abs(float(t["realized_pnl"])) for t in losses)

    pf = float("inf") if gl_abs < 1e-12 else gw / gl_abs if gl_abs > 0 else float("nan")
    consec_loss = 0
    max_cons = 0
    streak = sorted(finalized, key=lambda x: str(x.get("exit_time", "")))
    for tt in streak:
        pnl = float(tt.get("realized_pnl") or 0)
        if pnl < 0:
            consec_loss += 1
            max_cons = max(max_cons, consec_loss)
        else:
            consec_loss = 0

    hbs = [float(t["holding_bars"]) for t in finalized if isinstance(t.get("holding_bars"), (int, float))]
    avg_hold_bars = float(np.mean(hbs)) if hbs else 0.0

    return {
        "total_return": total_ret,
        "annualized_return": ann,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct * 100.0,
        "win_rate": win_rate,
        "profit_factor": pf,
        "num_trades": len(finalized),
        "final_equity": float(eq.iloc[-1]) if len(eq) else initial,
        "peak_equity": float(peak.iloc[-1]) if len(peak) else initial,
        "avg_win": float(np.mean([float(t["realized_pnl"]) for t in wins])) if wins else 0.0,
        "avg_loss": float(np.mean([float(t["realized_pnl"]) for t in losses])) if losses else 0.0,
        "largest_win": float(max((float(t["realized_pnl"]) for t in wins), default=0.0)),
        "largest_loss": float(min((float(t["realized_pnl"]) for t in losses), default=0.0)),
        "max_consecutive_losses": max_cons,
        "exposure_time_pct": None,
        "average_holding_period_bars": avg_hold_bars,
    }


def write_reports(
    *,
    out_dir: Path,
    eq_df: pd.DataFrame,
    meta: dict[str, Any],
    trades: list[dict],
    orders: list[SimulationOrderRecord],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_equity_curve:
        eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    if args.save_trades:
        pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    by_oid: dict[str, SimulationOrderRecord] = {}
    for o in orders:
        by_oid[o.order_id] = o
    pd.DataFrame([asdict(o) for o in by_oid.values()]).to_csv(out_dir / "orders.csv", index=False)

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    md = [
        "# Backtest summary (simulation)",
        "",
        f"- generated_utc: {datetime.now(UTC).isoformat()}",
        f"- symbols: {args.symbols}",
        f"- range: {args.start} → {args.end}",
        f"- initial_equity: {args.initial_equity}",
        f"- final_equity: {meta.get('final_equity')}",
        f"- total_return: {meta.get('total_return'):.4%}",
        f"- annualized_return: {meta.get('annualized_return'):.4%}",
        f"- sharpe: {meta.get('sharpe_ratio'):.4f}",
        f"- sortino: {meta.get('sortino_ratio'):.4f}",
        f"- max_drawdown_pct: {meta.get('max_drawdown_pct'):.4f}",
        f"- win_rate: {meta.get('win_rate'):.4%}",
        f"- profit_factor: {meta.get('profit_factor')}",
        f"- num_trades: {meta.get('num_trades')}",
        f"- exposure_time_pct: {meta.get('exposure_time_pct')}",
        f"- average_holding_period_bars: {meta.get('average_holding_period_bars')}",
        "",
        "## Assumptions",
        "- Fill model: signal on bar T close; limit active next bar; buy if low<=limit; sell if high>=limit",
        f"- slippage_bps: {args.slippage_bps}",
        f"- spread_bps: {args.spread_bps}",
        f"- commission_bps: {args.commission_bps}",
        f"- warmup_bars (cli floor): {args.warmup_bars}",
        "",
        "## Warnings",
        *[f"- {w}" for w in meta.get("warnings", [])[:50]],
    ]
    (out_dir / "backtest_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    eq_df, meta, _warns, order_rows, closed_trades = run_backtest(args)
    out_dir = args.output_dir.resolve()
    _LOG.info("Backtest complete; writing reports to %s", out_dir)
    write_reports(
        out_dir=out_dir,
        eq_df=eq_df,
        meta=meta,
        trades=closed_trades,
        orders=order_rows,
        args=args,
    )


if __name__ == "__main__":
    main()
