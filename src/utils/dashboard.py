"""Read-only Streamlit **Trading Bot Command Center** (dark theme).

Run from the repository root (same ``.env`` as the bot)::

    streamlit run src/utils/dashboard.py

Uses ``get_settings()`` from ``config.settings``, SQLite (read-only URI),
``logs/app.log``, ``runtime/kill_switch_state.json``, Alpaca **market data**
(bars + latest quotes for the Live Watchlist), and ``TradingClient``
**get_account** / **get_all_positions** only.
Never submits or cancels orders, never writes SQLite, never mutates kill-switch
files.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import streamlit as st

# --- Ensure ``src`` is on path when launched as ``streamlit run src/utils/dashboard.py`` ---
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional until pip install
    go = None  # type: ignore[assignment]

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None  # type: ignore[misc, assignment]

from config.settings import Settings, get_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers (safe to import from tests without running Streamlit UI)
# ---------------------------------------------------------------------------

LatencyStatus = Literal["ok", "warn", "fail", "unknown"]

_TRADE_TABLE_CANDIDATES = ("completed_trades", "trades", "executions")
_PNL_COLUMN_CANDIDATES = ("net_pnl", "realized_pnl", "pnl", "profit_loss")
_TIME_COLUMN_CANDIDATES = ("closed_at", "exit_time", "closed_time", "timestamp", "created_at")


@dataclass(frozen=True)
class PowerStatus:
    """Laptop power probe for status light."""

    plugged: Optional[bool]
    label: str  # "Plugged" | "On battery" | "Unknown"


@dataclass(frozen=True)
class TodayTradeRow:
    """Minimal row for P&L charts (non-canary)."""

    symbol: str
    closed_at: str
    realized_pnl: Optional[float]


def classify_latency_ms(latency_ms: Optional[float]) -> LatencyStatus:
    """Map round-trip time to green / yellow / red bands."""

    if latency_ms is None or latency_ms < 0:
        return "unknown"
    if latency_ms < 100.0:
        return "ok"
    if latency_ms <= 250.0:
        return "warn"
    return "fail"


def infer_dashboard_risk_mode(
    recent_newest_first: list[tuple[Optional[float], Optional[str]]],
) -> tuple[str, str]:
    """Heuristic risk label for the dashboard (not the runtime sizer)."""

    if len(recent_newest_first) >= 3:
        tri = recent_newest_first[:3]
        if all((p or 0.0) < -1e-9 for p, _ in tri):
            return "Defensive", "last_3_losses"
    if len(recent_newest_first) >= 2:
        duo = recent_newest_first[:2]
        if all((p or 0.0) > 1e-9 for p, _ in duo):
            return "Normal", "last_2_wins"
    newest = recent_newest_first[0] if recent_newest_first else None
    if newest:
        _, db_mode = newest
        if db_mode:
            rm = str(db_mode).strip().lower()
            if rm == "defensive":
                return "Defensive", "db_risk_mode"
            if rm == "normal":
                return "Normal", "db_risk_mode"
    return "Unknown", "insufficient_signal"


def tail_log_lines_matching(
    log_path: Path,
    *,
    needle: str,
    max_lines: int = 10,
    max_scan_bytes: int = 2_000_000,
) -> list[str]:
    """Return up to ``max_lines`` recent lines containing ``needle`` (best-effort)."""

    if max_lines < 1:
        return []
    try:
        if not log_path.is_file():
            return []
        size = log_path.stat().st_size
        read_size = min(size, max_scan_bytes)
        with log_path.open("rb") as f:
            if read_size > 0:
                f.seek(-read_size, 2)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in chunk.splitlines() if needle in ln]
        return lines[-max_lines:]
    except OSError:
        return []


def tail_last_lines(
    log_path: Path,
    *,
    max_lines: int = 50,
    max_read_bytes: int = 512_000,
) -> list[str]:
    """Return the last ``max_lines`` lines of a text file (tail -n)."""

    if max_lines < 1:
        return []
    try:
        if not log_path.is_file():
            return []
        size = log_path.stat().st_size
        read_size = min(size, max_read_bytes)
        with log_path.open("rb") as f:
            if read_size > 0:
                f.seek(-read_size, 2)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


def connect_sqlite_readonly(db_path: Path) -> Optional[sqlite3.Connection]:
    """Open SQLite in read-only mode (no WAL side effects)."""

    try:
        path = db_path.expanduser().resolve()
        if not path.exists():
            return None
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)
    except sqlite3.Error:
        return None


def discover_trade_table(conn: sqlite3.Connection) -> Optional[str]:
    """Pick ``completed_trades`` if present, else ``trades`` / ``executions``."""

    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        ).fetchall()
    except sqlite3.Error:
        return None
    names = {str(r[0]).lower(): str(r[0]) for r in rows}
    for cand in _TRADE_TABLE_CANDIDATES:
        if cand in names:
            return names[cand]
    return None


def query_today_trades(
    conn: sqlite3.Connection,
    trading_day_yyyy_mm_dd: str,
    *,
    source_scope: str = "live",
) -> list[TodayTradeRow]:
    start = f"{trading_day_yyyy_mm_dd}T00:00:00"
    end = f"{trading_day_yyyy_mm_dd}T23:59:59"
    src_clause = ""
    if source_scope == "simulation":
        src_clause = " AND COALESCE(source, 'live') = 'simulation' "
    elif source_scope == "live":
        src_clause = " AND COALESCE(source, 'live') IN ('live', 'paper') "
    else:
        src_clause = ""
    sql = f"""
    SELECT symbol, closed_at, realized_pnl
      FROM completed_trades
     WHERE closed_at >= ? AND closed_at <= ?
       AND COALESCE(is_canary, 0) = 0
       {src_clause}
     ORDER BY datetime(closed_at) ASC
    """
    try:
        rows = conn.execute(sql, (start, end)).fetchall()
    except sqlite3.Error:
        return []
    out: list[TodayTradeRow] = []
    for r in rows:
        try:
            out.append(
                TodayTradeRow(
                    symbol=str(r[0]),
                    closed_at=str(r[1]),
                    realized_pnl=float(r[2]) if r[2] is not None else None,
                ),
            )
        except (TypeError, ValueError, IndexError):
            continue
    return out


def query_recent_pnl_and_risk(
    conn: sqlite3.Connection,
    *,
    limit: int = 24,
    source_scope: str = "live",
) -> list[tuple[Optional[float], Optional[str]]]:
    src_clause = ""
    if source_scope == "simulation":
        src_clause = " AND COALESCE(source, 'live') = 'simulation' "
    elif source_scope == "live":
        src_clause = " AND COALESCE(source, 'live') IN ('live', 'paper') "
    else:
        src_clause = ""
    sql = f"""
    SELECT realized_pnl, risk_mode
      FROM completed_trades
     WHERE COALESCE(is_canary, 0) = 0
       {src_clause}
     ORDER BY datetime(closed_at) DESC
     LIMIT ?
    """
    try:
        raw = conn.execute(sql, (int(limit),)).fetchall()
    except sqlite3.Error:
        return []
    out: list[tuple[Optional[float], Optional[str]]] = []
    for pnl_raw, rm in raw:
        pnl: Optional[float]
        try:
            pnl = float(pnl_raw) if pnl_raw is not None else None
        except (TypeError, ValueError):
            pnl = None
        rm_s = str(rm) if rm is not None else None
        out.append((pnl, rm_s))
    return out


def query_latest_canary(conn: sqlite3.Connection) -> tuple[Optional[bool], Optional[str]]:
    """Return ``(success, error_or_none)`` for latest ``canary_results`` row."""

    sql = """
    SELECT success, error FROM canary_results ORDER BY id DESC LIMIT 1
    """
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.Error:
        return None, None
    if not row:
        return None, None
    ok = bool(int(row[0])) if row[0] is not None else False
    err = str(row[1]) if row[1] else None
    return ok, err


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [str(r[1]) for r in rows]
    except sqlite3.Error:
        return []


def _first_matching_column(cols_lower: dict[str, str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def load_trades_dataframe_schema_tolerant(
    conn: sqlite3.Connection,
    *,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, Optional[str], str, str]:
    """Load trade rows; return ``(df, table, pnl_col, time_col)`` for metrics."""

    table = discover_trade_table(conn)
    if not table:
        return pd.DataFrame(), None, "", ""

    cols = _table_columns(conn, table)
    if not cols:
        return pd.DataFrame(), table, "", ""

    cols_lower = {c.lower(): c for c in cols}
    pnl_col = _first_matching_column(cols_lower, _PNL_COLUMN_CANDIDATES)
    time_col = _first_matching_column(cols_lower, _TIME_COLUMN_CANDIDATES)
    if not pnl_col or not time_col:
        return pd.DataFrame(), table, pnl_col or "", time_col or ""

    select_cols: list[str] = [pnl_col, time_col]
    sym_col = _first_matching_column(cols_lower, ("symbol",))
    if sym_col:
        select_cols.append(sym_col)
    for extra in ("side", "quantity", "entry_price", "exit_price", "strategy_name", "risk_mode"):
        ec = _first_matching_column(cols_lower, (extra,))
        if ec and ec not in select_cols:
            select_cols.append(ec)

    if "is_canary" in cols_lower:
        where = " WHERE COALESCE(is_canary, 0) = 0 "
    else:
        where = " WHERE 1=1 "

    order_col = time_col
    lim = f" LIMIT {int(limit)} " if limit is not None else ""
    sql = f'SELECT {", ".join(f'"{c}"' for c in select_cols)} FROM "{table}" {where} ORDER BY datetime("{order_col}") ASC{lim}'

    try:
        df = pd.read_sql_query(sql, conn)
    except (ValueError, sqlite3.Error):
        return pd.DataFrame(), table, pnl_col, time_col

    df = df.rename(columns={pnl_col: "_pnl", time_col: "_ts"})
    df["_pnl"] = pd.to_numeric(df["_pnl"], errors="coerce").fillna(0.0)
    return df, table, "_pnl", "_ts"


def compute_trade_performance(df: pd.DataFrame, *, pnl_key: str = "_pnl") -> dict[str, Any]:
    """Aggregate metrics from a trade frame with canonical ``_pnl`` column."""

    if df.empty or pnl_key not in df.columns:
        return {
            "total_realized": 0.0,
            "win_rate": 0.0,
            "profit_factor": None,
            "profit_factor_label": "N/A",
            "avg_trade": 0.0,
            "n_trades": 0,
            "best": 0.0,
            "worst": 0.0,
        }

    pnl = df[pnl_key].astype(float)
    wins = pnl[pnl > 1e-9]
    losses = pnl[pnl < -1e-9]
    n = len(pnl)
    nw, nl = len(wins), len(losses)
    win_rate = nw / max(1, nw + nl)
    gp = float(wins.sum()) if nw else 0.0
    gl_abs = float(losses.abs().sum()) if nl else 0.0

    if gl_abs < 1e-12:
        pf_label = "∞" if gp > 1e-12 else "N/A"
        pf_val: Optional[float] = None if nl == 0 else float("inf")
    else:
        pf_val = gp / gl_abs
        pf_label = f"{pf_val:.4f}"

    return {
        "total_realized": float(pnl.sum()),
        "win_rate": win_rate,
        "profit_factor": pf_val,
        "profit_factor_label": pf_label,
        "avg_trade": float(pnl.mean()) if n else 0.0,
        "n_trades": n,
        "best": float(pnl.max()) if n else 0.0,
        "worst": float(pnl.min()) if n else 0.0,
    }


# Default ETF basket when ``SYMBOLS`` is empty.
WATCHLIST_DEFAULT_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLF", "EEM")

# Live watchlist RSI uses explicit 5-minute bars (dashboard contract).
WATCHLIST_BAR_TIMEFRAME = "5Min"

_WATCHLIST_FRESH_SEC = 10 * 60

_TIMEFRAME_DELTAS_WATCHLIST: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "5Min": timedelta(minutes=5),
    "15Min": timedelta(minutes=15),
    "1Hour": timedelta(hours=1),
    "1Day": timedelta(days=1),
}


def watchlist_symbols(settings: Settings) -> list[str]:
    """Prefer ``Settings.symbols_list``; otherwise static five-symbol basket."""

    raw = [s.strip().upper() for s in settings.symbols_list if s.strip()]
    return raw if raw else list(WATCHLIST_DEFAULT_SYMBOLS)


def dashboard_drop_inprogress_bar(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Exclude the last bar if it is still forming (aligned with RSI strategy)."""

    if bars is None or bars.empty:
        return bars
    delta = _TIMEFRAME_DELTAS_WATCHLIST.get(timeframe)
    if delta is None:
        return bars
    try:
        last_ts = bars.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            last_dt = last_ts.to_pydatetime()
        elif isinstance(last_ts, datetime):
            last_dt = last_ts
        else:
            return bars
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=UTC)
        if datetime.now(UTC) < last_dt + delta:
            return bars.iloc[:-1]
    except (IndexError, AttributeError, TypeError):
        return bars
    return bars


def watchlist_rsi_signal_label(
    rsi_val: float | None,
    *,
    rsi_ready: bool,
    oversold: float,
    overbought: float,
) -> str:
    if not rsi_ready or rsi_val is None or not math.isfinite(rsi_val):
        return "Warming Up"
    if rsi_val < oversold:
        return "Oversold"
    if rsi_val > overbought:
        return "Overbought"
    return "Neutral"


def watchlist_bar_freshness_label(last_bar_ts: datetime | None) -> str:
    if last_bar_ts is None:
        return "Unknown"
    try:
        ts = last_bar_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_s = (datetime.now(UTC) - ts).total_seconds()
    except (TypeError, ValueError, OSError):
        return "Unknown"
    if age_s <= _WATCHLIST_FRESH_SEC:
        return "🟢 Fresh"
    return "🟡 Stale"


def watchlist_spread_pct_cell(bid: float, ask: float) -> str:
    from utils.price_utils import spread_pct as _sp_pct  # noqa: PLC0415

    try:
        if bid <= 0 or ask <= bid:
            return "—"
        return f"{_sp_pct(bid, ask) * 100:.4f}%"
    except ValueError:
        return "—"


def read_kill_switch_latched(state_dir: Path) -> Optional[bool]:
    """Read latch from ``kill_switch_state.json`` (multiple keys, read-only)."""

    path = state_dir.expanduser().resolve() / "kill_switch_state.json"
    try:
        if not path.is_file():
            return None
        raw = json.load(path.open(encoding="utf-8"))
        if isinstance(raw, dict):
            for key in ("latched", "is_latched", "kill_switch_latched"):
                if key in raw:
                    return bool(raw[key])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def read_trailing_stop_price(state_dir: Path, symbol: str) -> Optional[float]:
    """Return persisted trailing stop price for symbol, if any."""

    path = state_dir.expanduser().resolve() / "trail_trailing_state.json"
    try:
        if not path.is_file():
            return None
        data = json.load(path.open(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        row = data.get(symbol.upper())
        if not isinstance(row, dict):
            return None
        tsp = row.get("trailing_stop_price")
        if tsp is None:
            return None
        return float(tsp)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def probe_power_plugged() -> PowerStatus:
    try:
        import psutil  # type: ignore[import-untyped]

        bat = psutil.sensors_battery()
        if bat is None:
            return PowerStatus(plugged=None, label="Unknown")
        plugged = bool(bat.power_plugged)
        return PowerStatus(
            plugged=plugged,
            label="Plugged" if plugged else "On battery",
        )
    except Exception:
        return PowerStatus(plugged=None, label="Unknown")


def measure_alpaca_clock_latency_ms(
    *,
    api_key: str,
    secret_key: str,
    paper: bool,
    timeout_s: float = 5.0,
) -> Optional[float]:
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    url = f"{base}/v2/clock"
    req = urllib.request.Request(
        url,
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        },
        method="GET",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read(4096)
        return (time.perf_counter() - t0) * 1000.0
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None


def _position_to_row(
    p: Any,
    state_dir: Path,
) -> dict[str, Any]:
    """Convert alpaca Position + optional trail file into a dashboard row."""

    try:
        sym = str(p.symbol).upper()
        qty = float(p.qty)
        side = str(p.side).lower()
        mv = float(p.market_value) if p.market_value is not None else 0.0
        avg = float(p.avg_entry_price) if p.avg_entry_price is not None else 0.0
        cur = float(p.current_price) if p.current_price is not None else 0.0
        u_pnl = float(p.unrealized_pl) if p.unrealized_pl is not None else 0.0
    except (TypeError, ValueError, AttributeError):
        return {}

    cost = abs(qty) * avg if avg > 0 else 0.0
    u_pct = (100.0 * u_pnl / cost) if cost > 1e-9 else None

    trail_px = read_trailing_stop_price(state_dir, sym)
    trail_dist: Optional[float]
    if trail_px is not None and cur > 0 and side == "long":
        trail_dist = cur - trail_px
    else:
        trail_dist = None

    return {
        "Symbol": sym,
        "Quantity": qty,
        "Side": side,
        "Market Value": mv,
        "Avg Entry": avg,
        "Current Price": cur,
        "Unrealized P&L": u_pnl,
        "Unrealized P&L %": u_pct if u_pct is not None else "",
        "Trailing Stop Distance": trail_dist if trail_dist is not None else "",
    }


# ---------------------------------------------------------------------------
# Streamlit caches (Alpaca read-only)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def fetch_watchlist_rows(
    symbols_tuple: tuple[str, ...],
    rsi_length: int,
    rsi_oversold_threshold: float,
    rsi_overbought_threshold: float,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Load latest 5-minute bars + quotes; compute RSI using completed bars only."""

    from core.alpaca_clients import build_alpaca_clients  # noqa: PLC0415
    from core.market_data import BarFetcher  # noqa: PLC0415
    from strategies.indicators import rsi as rsi_indicator  # noqa: PLC0415

    rows: list[dict[str, Any]] = []
    try:
        settings = get_settings()
    except Exception as exc:
        return [], str(exc)

    try:
        clients = build_alpaca_clients(settings)
    except Exception as exc:
        return [], f"Alpaca client build failed: {exc}"

    fetcher = BarFetcher(
        clients.historical_data,
        feed=clients.resolved_feed,
        max_attempts=settings.RETRY_MAX_ATTEMPTS,
        base_delay=settings.RETRY_BASE_DELAY_SECONDS,
        max_delay=settings.RETRY_MAX_DELAY_SECONDS,
    )

    min_bars = max(50, int(rsi_length) + 25)
    rsi_min_samples = int(rsi_length) + 5

    for sym in symbols_tuple:
        sym_u = sym.strip().upper()
        try:
            df_raw = fetcher.fetch_bars(sym_u, WATCHLIST_BAR_TIMEFRAME, lookback_bars=min_bars)
            df_done = dashboard_drop_inprogress_bar(df_raw, WATCHLIST_BAR_TIMEFRAME)
            if df_done is None or df_done.empty:
                raise ValueError("empty bars")

            close = df_done["close"].astype(float)
            rsi_series = rsi_indicator(close, length=int(rsi_length))
            rsi_last = float(rsi_series.iloc[-1])
            rsi_ready = len(df_done) >= rsi_min_samples and not pd.isna(rsi_last)

            last_ix = df_done.index[-1]
            ts = pd.Timestamp(last_ix)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            ts_parsed = ts.to_pydatetime()
            last_bar_iso = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            freshness = watchlist_bar_freshness_label(ts_parsed)

            quote = fetcher.fetch_latest_quote(sym_u)
            spr = watchlist_spread_pct_cell(quote.bid, quote.ask)
            px = float(close.iloc[-1])

            rsi_cell = "Warming Up" if not rsi_ready else f"{rsi_last:.2f}"
            signal = watchlist_rsi_signal_label(
                rsi_last,
                rsi_ready=rsi_ready,
                oversold=float(rsi_oversold_threshold),
                overbought=float(rsi_overbought_threshold),
            )
            rows.append(
                {
                    "Symbol": sym_u,
                    "Price": f"{px:.4f}",
                    "RSI": rsi_cell,
                    "Signal Status": signal,
                    "Latest Bar Time": last_bar_iso,
                    "Freshness": freshness,
                    "Spread %": spr,
                },
            )
        except Exception as exc:  # noqa: BLE001 - row-level degrade
            rows.append(
                {
                    "Symbol": sym_u,
                    "Price": "—",
                    "RSI": "—",
                    "Signal Status": "🔴 Error",
                    "Latest Bar Time": "—",
                    "Freshness": "🔴 Error",
                    "Spread %": "—",
                },
            )

    return rows, None


@st.cache_resource(show_spinner=False)
def _cached_trading_client(api_key: str, secret_key: str, paper: bool) -> Any:
    from alpaca.trading.client import TradingClient  # noqa: PLC0415

    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


@st.cache_data(ttl=10, show_spinner=False)
def load_account_snapshot(api_key: str, secret_key: str, paper: bool) -> dict[str, Any]:
    try:
        client = _cached_trading_client(api_key, secret_key, paper)
        account = client.get_account()
        data: dict[str, Any]
        md = getattr(account, "model_dump", None)
        if callable(md):
            raw = md()
            data = raw if isinstance(raw, dict) else {}
        else:
            data = {}
        if not data:
            data = {
                k: getattr(account, k, None)
                for k in (
                    "equity",
                    "cash",
                    "buying_power",
                    "portfolio_value",
                    "long_market_value",
                    "short_market_value",
                    "multiplier",
                )
            }
        return {"ok": True, "data": data, "error": None}
    except Exception as exc:  # pragma: no cover - network / broker deps
        return {"ok": False, "data": {}, "error": str(exc)}


@st.cache_data(ttl=10, show_spinner=False)
def load_open_positions_rows(
    api_key: str,
    secret_key: str,
    paper: bool,
    state_dir_posix: str,
) -> dict[str, Any]:
    state_dir = Path(state_dir_posix)
    try:
        client = _cached_trading_client(api_key, secret_key, paper)
        positions = client.get_all_positions()
        rows = []
        for p in positions:
            row = _position_to_row(p, state_dir)
            if row:
                rows.append(row)
        unreal = sum(float(r.get("Unrealized P&L", 0) or 0) for r in rows)
        return {"ok": True, "positions": rows, "open_unrealized": unreal, "error": None}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "positions": [], "open_unrealized": None, "error": str(exc)}


@st.cache_data(ttl=3, show_spinner=False)
def load_db_trade_payload(db_path_posix: str) -> dict[str, Any]:
    """Load full history + recent 25 for UI (short TTL)."""

    path = Path(db_path_posix)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return {
            "ok": False,
            "error": "Database file missing or unreadable",
            "hist": pd.DataFrame(),
            "recent": pd.DataFrame(),
            "table": None,
        }
    try:
        hist, table, _, _ = load_trades_dataframe_schema_tolerant(conn, limit=None)
        if hist.empty:
            recent = hist
        else:
            recent = hist.sort_values("_ts", ascending=False).head(25)
        metrics = compute_trade_performance(hist)
        daily = pd.DataFrame()
        if not hist.empty and "_ts" in hist.columns:
            tss = pd.to_datetime(hist["_ts"], utc=True, errors="coerce")
            hist2 = hist.assign(_day=tss.dt.date)
            daily = hist2.groupby("_day", dropna=True)["_pnl"].sum().reset_index()
            daily.columns = ["day", "realized_pnl"]
        return {
            "ok": True,
            "error": None,
            "hist": hist,
            "recent": recent,
            "table": table,
            "metrics": metrics,
            "daily": daily,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "hist": pd.DataFrame(),
            "recent": pd.DataFrame(),
            "table": None,
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _fmt_path(p: Optional[Path]) -> str:
    if p is None:
        return "—"
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def _inject_theme_css() -> None:
    st.markdown(
        """
<style>
  .stApp { background: linear-gradient(160deg, #0b0e11 0%, #12161c 55%, #0b0e11 100%); }
  div[data-testid="stToolbar"] { background: transparent; }
  .cc-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  .cc-badge-live {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 700;
    background: rgba(239,68,68,0.22);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.55);
    margin-right: 8px;
  }
  .cc-badge-safe {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
    background: rgba(34,211,238,0.14);
    color: #93c5fd;
    border: 1px solid rgba(96,165,250,0.35);
    margin-right: 8px;
  }
  .cc-sub { color: #9fb0c7; font-size: 13px; }
  .profit { color: #4ade80 !important; }
  .loss { color: #f87171 !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Trading Bot Command Center",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme_css()

    try:
        settings = get_settings()
    except Exception as exc:
        st.error("Failed to load settings (``.env`` missing or invalid).")
        st.caption(str(type(exc).__name__))
        return

    sidebar = st.sidebar
    sidebar.header("Session")
    sidebar.caption("`get_settings()` → same `.env` as the bot.")

    refresh_seconds = sidebar.slider(
        "Watchlist auto-refresh interval (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        step=5,
        help="Drives `streamlit-autorefresh` rerun cadence.",
    )
    if st_autorefresh is not None:
        st_autorefresh(interval=int(refresh_seconds) * 1000, key="watchlist_refresh")
        sidebar.caption(f"Auto-refresh: **{refresh_seconds}s** (`watchlist_refresh`).")
    else:
        sidebar.info(
            "Install **`streamlit-autorefresh`** for automatic watchlist refresh: "
            "`pip install streamlit-autorefresh`",
        )

    refresh_clicked = sidebar.button("Refresh now", use_container_width=True)
    if refresh_clicked:
        load_account_snapshot.clear()
        load_open_positions_rows.clear()
        load_db_trade_payload.clear()
        fetch_watchlist_rows.clear()
        try:
            _cached_trading_client.clear()
        except Exception:
            pass
        st.rerun()

    trade_source_scope = sidebar.selectbox(
        "SQLite daily / risk scope",
        options=["live", "simulation", "all"],
        index=0,
    )

    is_paper = settings.is_paper
    raw_db = Path(settings.DATABASE_PATH)
    resolved_db = (
        raw_db.expanduser().resolve()
        if raw_db.is_absolute()
        else (Path.cwd() / raw_db).expanduser().resolve()
    )
    raw_state = Path(settings.STATE_DIR)
    state_dir = (
        raw_state.expanduser().resolve()
        if raw_state.is_absolute()
        else (Path.cwd() / raw_state).expanduser().resolve()
    )
    raw_log = Path(settings.LOG_DIR)
    log_dir = (
        raw_log.expanduser().resolve()
        if raw_log.is_absolute()
        else (Path.cwd() / raw_log).expanduser().resolve()
    )
    app_log_path = log_dir / "app.log"

    env_label = "PAPER" if is_paper else "LIVE"
    dry_txt = "true" if settings.DRY_RUN else "false"

    latch = read_kill_switch_latched(state_dir)
    if latch is None:
        ks_display = "Unknown / not initialized"
    elif latch:
        ks_display = "🔴 Latched"
    else:
        ks_display = "🟢 Clear"

    db_exists = resolved_db.is_file()
    sidebar.markdown("**Environment**")
    sidebar.write(f"- Alpaca: **{env_label}**")
    sidebar.write(f"- `DRY_RUN`: **{dry_txt}**")
    sidebar.write(f"- Kill switch: {ks_display}")
    sidebar.write(f"- DB file: {'OK' if db_exists else 'missing'}")
    sidebar.markdown("**Spread filter**")
    sidebar.caption(
        f"- Default `SPREAD_FILTER_PCT`: **{settings.SPREAD_FILTER_PCT:.6f}** "
        f"(~{settings.SPREAD_FILTER_PCT * 10000:.2f} bps)",
    )
    if settings.SPREAD_FILTER_PCT_IEX is not None:
        iex_v = float(settings.SPREAD_FILTER_PCT_IEX)
        sidebar.caption(
            f"- IEX quotes use `SPREAD_FILTER_PCT_IEX`: **{iex_v:.6f}** (~{iex_v * 10000:.2f} bps)",
        )
    else:
        sidebar.caption(
            "- IEX override unset — same max spread for all feeds (set `SPREAD_FILTER_PCT_IEX` if IEX quotes skip too often).",
        )
    sidebar.markdown("**Paths**")
    sidebar.caption(f"Database: `{_fmt_path(resolved_db)}`")
    sidebar.caption(f"Logs: `{_fmt_path(log_dir)}`")
    sidebar.caption(f"State: `{_fmt_path(state_dir)}`")

    try:
        log_mtime = datetime.fromtimestamp(app_log_path.stat().st_mtime, UTC).isoformat()
    except OSError:
        log_mtime = "N/A"
    sidebar.caption(f"Last `app.log` write (mtime): **{log_mtime}**")

    now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # --- Alpaca (cached) ---
    acct_payload = load_account_snapshot(
        settings.ALPACA_API_KEY,
        settings.ALPACA_API_SECRET,
        settings.is_paper,
    )
    pos_payload = load_open_positions_rows(
        settings.ALPACA_API_KEY,
        settings.ALPACA_API_SECRET,
        settings.is_paper,
        str(state_dir),
    )

    alpaca_ok = bool(acct_payload.get("ok")) and bool(pos_payload.get("ok"))
    if not acct_payload.get("ok"):
        st.warning(f"Alpaca account: {acct_payload.get('error') or 'unavailable'}")
    if not pos_payload.get("ok"):
        st.warning(f"Alpaca positions: {pos_payload.get('error') or 'unavailable'}")

    ad = acct_payload.get("data") or {}
    try:
        equity = float(ad.get("equity")) if ad.get("equity") is not None else None
    except (TypeError, ValueError):
        equity = None
    try:
        cash = float(ad.get("cash")) if ad.get("cash") is not None else None
    except (TypeError, ValueError):
        cash = None
    try:
        bp = float(ad.get("buying_power")) if ad.get("buying_power") is not None else None
    except (TypeError, ValueError):
        bp = None

    open_unreal = pos_payload.get("open_unrealized")
    open_unreal_f = float(open_unreal) if open_unreal is not None else None

    # --- SQLite ---
    db_pack = load_db_trade_payload(str(resolved_db))
    hist_df = db_pack.get("hist") if isinstance(db_pack.get("hist"), pd.DataFrame) else pd.DataFrame()
    metrics = db_pack.get("metrics") if isinstance(db_pack.get("metrics"), dict) else compute_trade_performance(hist_df)

    db_conn_daily = connect_sqlite_readonly(resolved_db)
    daily_realized: Optional[float] = None
    try:
        if db_conn_daily is not None:
            try:
                from utils.time_utils import today_eastern  # noqa: PLC0415

                today_et = today_eastern().strftime("%Y-%m-%d")
            except Exception:
                today_et = datetime.now(UTC).strftime("%Y-%m-%d")
            chart_rows = query_today_trades(
                db_conn_daily,
                today_et,
                source_scope=trade_source_scope,
            )
            daily_realized = sum(r.realized_pnl or 0.0 for r in chart_rows)
    except sqlite3.Error:
        daily_realized = None
    finally:
        if db_conn_daily is not None:
            try:
                db_conn_daily.close()
            except Exception:
                pass

    if not db_pack.get("ok"):
        sidebar.caption(f"SQLite trades: ⚠️ {db_pack.get('error')}")
    else:
        sidebar.success(f"SQLite trades: **`{db_pack.get('table') or 'unknown'}`**")

    sidebar.caption(f"Alpaca API: **`{'reachable' if alpaca_ok else 'degraded'}`**")

    # --- Header ---
    st.markdown('<div style="margin-bottom: 8px;">', unsafe_allow_html=True)
    h1_cols = st.columns([4, 1])
    with h1_cols[0]:
        st.markdown("## Trading Bot Command Center")
        st.markdown(
            f'<p class="cc-sub">Alpaca · <b>{env_label}</b> · DRY_RUN={dry_txt} · '
            f"Last refresh: <b>{now_iso}</b><br/>"
            f"Database: <code style='color:#a5d6ff;'>{_fmt_path(resolved_db)}</code></p>",
            unsafe_allow_html=True,
        )

    with h1_cols[1]:
        if not is_paper and not settings.DRY_RUN:
            st.markdown(
                '<span class="cc-badge-live">⚠️ LIVE MODE — REAL ACCOUNT DATA</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="cc-badge-safe">Read-only monitor</span>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Metric cards ---
    mcols = st.columns(8)
    mc = metrics

    def _metric(idx: int, label: str, value: str, *, profit: bool | None = None) -> None:
        with mcols[idx]:
            if profit is True:
                st.markdown(f'<div class="cc-card profit"><div style="font-size:12px;color:#94a3b8;">{label}</div>'
                            f'<div style="font-size:22px;font-weight:700;">{value}</div></div>',
                            unsafe_allow_html=True)
            elif profit is False:
                st.markdown(f'<div class="cc-card loss"><div style="font-size:12px;color:#94a3b8;">{label}</div>'
                            f'<div style="font-size:22px;font-weight:700;">{value}</div></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="cc-card"><div style="font-size:12px;color:#94a3b8;">{label}</div>'
                            f'<div style="font-size:22px;font-weight:700;">{value}</div></div>',
                            unsafe_allow_html=True)

    eq_s = f"{equity:,.2f}" if equity is not None else "—"
    cash_s = f"{cash:,.2f}" if cash is not None else "—"
    bp_s = f"{bp:,.2f}" if bp is not None else "—"
    ou = open_unreal_f
    ou_s = f"{ou:,.2f}" if ou is not None else "—"
    dr = daily_realized
    dr_s = f"{dr:,.2f}" if dr is not None else "—"
    wr = f"{float(mc.get('win_rate', 0)) * 100:.1f}%" if mc.get("n_trades", 0) else "—"
    pf = str(mc.get("profit_factor_label", "N/A"))
    at = f"{float(mc.get('avg_trade', 0)):,.2f}" if mc.get("n_trades", 0) else "—"

    pr_open = None if ou is None else (True if ou > 1e-6 else (False if ou < -1e-6 else None))
    pr_day = None if dr is None else (True if dr > 1e-6 else (False if dr < -1e-6 else None))

    _metric(0, "Total equity", eq_s)
    _metric(1, "Buying power", bp_s)
    _metric(2, "Cash", cash_s)
    _metric(3, "Open P/L", ou_s, profit=pr_open)
    _metric(4, "Realized P/L (today)", dr_s, profit=pr_day)
    _metric(5, "Win rate (DB)", wr)
    _metric(6, "Profit factor", pf)
    _metric(7, "Avg trade (DB)", at)

    st.markdown("---")

    # --- Live Watchlist (market data + RSI; no SQLite trades required) ---
    st.subheader("Live Watchlist")
    syms_wl = tuple(watchlist_symbols(settings))
    wl_rows, wl_err = fetch_watchlist_rows(
        syms_wl,
        int(settings.RSI_LENGTH),
        float(settings.RSI_OVERSOLD),
        70.0,
    )
    st.caption(
        f"5-minute bars · RSI length **{settings.RSI_LENGTH}** · "
        f"signals: **below {settings.RSI_OVERSOLD:.0f}** = Oversold, **above 70** = Overbought · "
        f"{', '.join(syms_wl)}",
    )
    if wl_err:
        st.warning(f"Live Watchlist unavailable: {wl_err}")
    elif wl_rows:
        st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No watchlist rows.")

    st.caption(
        "Watchlist data updates automatically. This dashboard is read-only and does not place trades.",
    )

    st.markdown("---")

    # --- Charts ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Cumulative Realized P/L")
        if go is not None and not hist_df.empty and "_ts" in hist_df.columns and "_pnl" in hist_df.columns:
            h2 = hist_df.sort_values("_ts")
            cum = h2["_pnl"].cumsum()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=h2["_ts"],
                    y=cum,
                    mode="lines",
                    line=dict(color="#4ade80", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(74,222,128,0.12)",
                ),
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,24,30,0.9)",
                margin=dict(l=40, r=20, t=40, b=40),
                height=360,
                xaxis_title="Time",
                yaxis_title="Cumulative P/L",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif hist_df.empty:
            st.info("No trade history in SQLite for cumulative P/L.")
        else:
            st.warning("Install **plotly** for the equity curve chart: `pip install plotly`.")

    with c2:
        st.subheader("Current Asset Allocation")
        pos_rows = pos_payload.get("positions") or []
        if go is not None and pos_rows:
            pdf = pd.DataFrame(pos_rows)
            if "Symbol" in pdf.columns and "Market Value" in pdf.columns:
                pdf["Market Value"] = pd.to_numeric(pdf["Market Value"], errors="coerce").fillna(0.0)
                pdf = pdf[pdf["Market Value"] > 0]
                if not pdf.empty:
                    fig2 = go.Figure(
                        data=[
                            go.Pie(
                                labels=pdf["Symbol"],
                                values=pdf["Market Value"],
                                hole=0.42,
                                marker=dict(line=dict(color="#0b0e11", width=1)),
                            ),
                        ],
                    )
                    fig2.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=360,
                        showlegend=True,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No positive market-value positions.")
            else:
                st.info("Unexpected position payload shape.")
        elif not pos_rows:
            st.info("No open positions from Alpaca.")
        else:
            st.warning("Install **plotly** for allocation chart: `pip install plotly`.")

    daily_df = db_pack.get("daily")
    if isinstance(daily_df, pd.DataFrame) and not daily_df.empty and go is not None:
        st.subheader("Daily Realized P/L (SQLite)")
        figd = go.Figure(
            data=[go.Bar(x=daily_df["day"].astype(str), y=daily_df["realized_pnl"], marker_color="#60a5fa")],
        )
        figd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,24,30,0.9)",
            height=300,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(figd, use_container_width=True)

    st.subheader("Open positions")
    if pos_rows:
        st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No rows (or Alpaca unavailable).")

    st.subheader("Recent trades (SQLite, latest 25)")
    recent_df = db_pack.get("recent")
    if isinstance(recent_df, pd.DataFrame) and not recent_df.empty:
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No completed trades loaded.")

    st.subheader("Log monitor (`app.log`, last 50 lines)")
    log_lines = tail_last_lines(app_log_path, max_lines=50)
    if log_lines:
        st.code("\n".join(log_lines), language="")
    else:
        st.caption("Log file missing or empty.")

    st.caption(
        "Read-only Command Center · no orders · no DB writes · no kill-switch mutations. "
        f"Account/positions cache TTL ~10s · SQLite cache ~3s · watchlist/bar cache TTL ~30s · "
        f"page auto-rerun ~**{refresh_seconds}s** when `streamlit-autorefresh` is installed.",
    )


# Legacy thread-local cache (kept for any external callers; not used by Streamlit path)
_alpaca_cache_lock = threading.Lock()
_alpaca_cache_ts: float = 0.0
_alpaca_cache_payload: dict[str, Any] = {
    "ok": False,
    "error": None,
    "equity": None,
    "buying_power": None,
    "positions": [],
}


def fetch_alpaca_snapshot(
    *,
    api_key: str,
    secret_key: str,
    paper: bool,
    state_dir: Path,
    ttl_seconds: float,
    force_refresh: bool,
) -> dict[str, Any]:
    """Read-only Alpaca snapshot with a simple process-local TTL (non-Streamlit)."""

    global _alpaca_cache_ts, _alpaca_cache_payload

    now = time.monotonic()
    with _alpaca_cache_lock:
        if (
            not force_refresh
            and _alpaca_cache_payload.get("ok")
            and (now - _alpaca_cache_ts) < max(1.0, ttl_seconds)
        ):
            return dict(_alpaca_cache_payload)

    out: dict[str, Any] = {
        "ok": False,
        "error": None,
        "equity": None,
        "buying_power": None,
        "positions": [],
    }
    try:
        from alpaca.trading.client import TradingClient  # noqa: PLC0415

        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        account = client.get_account()
        raw_a = getattr(account, "model_dump", None)
        data_a = raw_a() if callable(raw_a) else None
        if not isinstance(data_a, dict):
            data_a = {
                "equity": getattr(account, "equity", None),
                "buying_power": getattr(account, "buying_power", None),
            }

        equity = data_a.get("equity") if isinstance(data_a, dict) else None
        bp = data_a.get("buying_power") if isinstance(data_a, dict) else None

        pos_list = []
        positions = client.get_all_positions()
        for p in positions:
            row = _position_to_row(p, state_dir)
            if row:
                pos_list.append(row)

        out["ok"] = True
        out["equity"] = float(equity) if equity is not None else None
        out["buying_power"] = float(bp) if bp is not None else None
        out["positions"] = pos_list
    except Exception as exc:
        out["error"] = str(exc)
        out["ok"] = False

    with _alpaca_cache_lock:
        _alpaca_cache_payload = dict(out)
        _alpaca_cache_ts = now

    return out


def status_emoji(kind: Literal["green", "yellow", "red"]) -> str:
    return {"green": "🟢", "yellow": "🟡", "red": "🔴"}[kind]


if __name__ == "__main__":
    main()
