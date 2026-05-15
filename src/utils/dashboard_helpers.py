"""Read-only SQLite / pandas helpers for the Streamlit dashboard (Phase 5).

Imported by ``utils.dashboard`` and tests. No Streamlit dependency.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

_TRADE_TABLE_CANDIDATES = ("completed_trades", "trades", "executions")
_PNL_COLUMN_CANDIDATES = ("net_pnl", "realized_pnl", "pnl", "profit_loss")
_TIME_COLUMN_CANDIDATES = ("closed_at", "exit_time", "closed_time", "timestamp", "created_at")


def connect_sqlite_readonly(db_path: Path) -> sqlite3.Connection | None:
    """Open SQLite in read-only mode (no WAL side effects)."""

    try:
        path = db_path.expanduser().resolve()
        if not path.exists():
            return None
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)
    except sqlite3.Error:
        return None


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
            (table,),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _source_scope_sql_clause(source_scope: str) -> str:
    """Filter ``equity_snapshots.source`` (and similar) for dashboard scopes."""

    col = "COALESCE(source, 'live')"
    sc = source_scope.strip().lower()
    if sc == "live":
        return f" AND {col} IN ('live', 'paper') "
    if sc == "simulation":
        return f" AND {col} = 'simulation' "
    if sc == "replay":
        return f" AND {col} = 'replay' "
    if sc == "shadow":
        return f" AND {col} = 'shadow' "
    if sc == "dry_run":
        return f" AND {col} = 'dry_run' "
    return ""


def load_replay_runs(db_path: str | Path, *, limit: int = 200) -> pd.DataFrame:
    """Load ``replay_runs`` rows newest-first; empty frame if missing/unreadable."""

    lim = max(1, min(int(limit), 2000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        if not _sqlite_table_exists(conn, "replay_runs"):
            return pd.DataFrame()
        return pd.read_sql_query(
            f"SELECT * FROM replay_runs ORDER BY datetime(created_at) DESC LIMIT {lim}",
            conn,
        )
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def load_equity_snapshots(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    source_scope: str = "all",
    strategy_name: str | None = None,
    limit: int = 50_000,
) -> pd.DataFrame:
    """Load ``equity_snapshots`` ascending by time (for curves / drawdown)."""

    lim = max(1, min(int(limit), 200_000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        if not _sqlite_table_exists(conn, "equity_snapshots"):
            return pd.DataFrame()
        clauses: list[str] = ["1=1"]
        if run_id:
            clauses.append("run_id = ?")
        if strategy_name is not None:
            clauses.append("COALESCE(strategy_name, '') = ?")
        scope_sql = _source_scope_sql_clause(source_scope)
        where = " AND ".join(clauses)
        sql = f"""
        SELECT * FROM equity_snapshots
         WHERE {where}
         {scope_sql}
         ORDER BY datetime(timestamp) ASC, id ASC
         LIMIT {lim}
        """
        params: list[Any] = []
        if run_id:
            params.append(run_id)
        if strategy_name is not None:
            params.append(strategy_name)
        return pd.read_sql_query(sql, conn, params=params if params else None)
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def load_strategy_signals(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    strategy_name: str | None = None,
    symbol: str | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    lim = max(1, min(int(limit), 50_000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        if not _sqlite_table_exists(conn, "strategy_signals"):
            return pd.DataFrame()
        clauses: list[str] = ["1=1"]
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if strategy_name:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.strip().upper())
        where = " AND ".join(clauses)
        sql = f"""
        SELECT * FROM strategy_signals
         WHERE {where}
         ORDER BY datetime(timestamp) DESC, id DESC
         LIMIT {lim}
        """
        return pd.read_sql_query(sql, conn, params=params if params else None)
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def load_strategy_decisions(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    lim = max(1, min(int(limit), 50_000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        if not _sqlite_table_exists(conn, "strategy_decisions"):
            return pd.DataFrame()
        if run_id:
            sql = f"""
            SELECT * FROM strategy_decisions
             WHERE run_id = ?
             ORDER BY datetime(timestamp) DESC, id DESC
             LIMIT {lim}
            """
            return pd.read_sql_query(sql, conn, params=(run_id,))
        sql = f"""
        SELECT * FROM strategy_decisions
         ORDER BY datetime(timestamp) DESC, id DESC
         LIMIT {lim}
        """
        return pd.read_sql_query(sql, conn)
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def load_skip_events(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    lim = max(1, min(int(limit), 50_000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        if not _sqlite_table_exists(conn, "skip_events"):
            return pd.DataFrame()
        if run_id:
            sql = f"""
            SELECT * FROM skip_events
             WHERE run_id = ?
             ORDER BY datetime(timestamp) DESC, id DESC
             LIMIT {lim}
            """
            return pd.read_sql_query(sql, conn, params=(run_id,))
        sql = f"""
        SELECT * FROM skip_events
         ORDER BY datetime(timestamp) DESC, id DESC
         LIMIT {lim}
        """
        return pd.read_sql_query(sql, conn)
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def discover_trade_table(conn: sqlite3.Connection) -> str | None:
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


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [str(r[1]) for r in rows]
    except sqlite3.Error:
        return []


def _first_matching_column(cols_lower: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def load_completed_trades_by_run(
    db_path: str | Path,
    *,
    run_id: str | None = None,
    source_scope: str = "all",
    limit: int = 20_000,
) -> pd.DataFrame:
    """Load ``completed_trades`` filtered by ``replay_run_id`` and optional ``source`` scope."""

    lim = max(1, min(int(limit), 100_000))
    path = Path(db_path)
    conn = connect_sqlite_readonly(path)
    if conn is None:
        return pd.DataFrame()
    table = discover_trade_table(conn)
    if not table:
        with contextlib.suppress(Exception):
            conn.close()
        return pd.DataFrame()
    cols = _table_columns(conn, table)
    if not cols:
        with contextlib.suppress(Exception):
            conn.close()
        return pd.DataFrame()
    cols_lower = {c.lower(): c for c in cols}
    rid_col = cols_lower.get("replay_run_id")
    src_col = cols_lower.get("source")
    clauses = ["COALESCE(is_canary, 0) = 0"]
    params: list[Any] = []
    if run_id and rid_col:
        clauses.append(f'"{rid_col}" = ?')
        params.append(run_id)
    scope = source_scope.strip().lower()
    if src_col and scope == "live":
        clauses.append(f' COALESCE("{src_col}", \'live\') IN (\'live\', \'paper\') ')
    elif src_col and scope == "simulation":
        clauses.append(f' COALESCE("{src_col}", \'live\') = \'simulation\' ')
    elif src_col and scope == "replay":
        clauses.append(f' COALESCE("{src_col}", \'live\') = \'replay\' ')
    elif src_col and scope == "shadow":
        clauses.append(f' COALESCE("{src_col}", \'live\') = \'shadow\' ')
    elif src_col and scope == "dry_run":
        clauses.append(f' COALESCE("{src_col}", \'live\') = \'dry_run\' ')
    where = " AND ".join(clauses)
    quoted = ", ".join(f'"{c}"' for c in cols)
    order_col = _first_matching_column(cols_lower, _TIME_COLUMN_CANDIDATES) or "closed_at"
    sql = f'SELECT {quoted} FROM "{table}" WHERE {where} ORDER BY datetime("{order_col}") DESC LIMIT {lim}'
    try:
        return pd.read_sql_query(sql, conn, params=params if params else None)
    except (ValueError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Return max drawdown as a negative fraction (e.g. -0.12 for -12%)."""

    s = pd.to_numeric(equity, errors="coerce").dropna()
    if s.empty:
        return 0.0
    peak = s.cummax()
    dd = (s / peak) - 1.0
    return float(dd.min())


def benchmark_total_return(benchmark: pd.Series) -> float | None:
    """Return (last/first - 1) for non-null positive benchmark series."""

    b = pd.to_numeric(benchmark, errors="coerce").dropna()
    b = b[b > 0]
    if len(b) < 2:
        return None
    first, last = float(b.iloc[0]), float(b.iloc[-1])
    if first <= 0:
        return None
    return last / first - 1.0


def benchmark_return_from_equity_frame(equity_df: pd.DataFrame) -> float | None:
    """Benchmark total return using one value per timestamp (deduped)."""

    if equity_df.empty or "benchmark_equity" not in equity_df.columns:
        return None
    if "timestamp" in equity_df.columns:
        sub = equity_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    else:
        sub = equity_df
    return benchmark_total_return(sub["benchmark_equity"])


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
        pf_val: float | None = None if nl == 0 else float("inf")
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


def _sharpe_simple_daily_proxy(eq: pd.Series) -> float | None:
    """Heuristic Sharpe from equity step returns (treats steps as ~daily; intraday replay: indicative only)."""

    r = eq.pct_change().dropna()
    if len(r) < 5:
        return None
    std = float(r.std())
    if std < 1e-12:
        return None
    return float(r.mean() / std * math.sqrt(252.0))


def build_strategy_comparison_table(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    initial_equity: float,
) -> pd.DataFrame:
    """Aggregate per-``strategy_name`` (portfolio) metrics for a replay run."""

    if equity_df.empty or "strategy_name" not in equity_df.columns or "equity" not in equity_df.columns:
        return pd.DataFrame()

    ie = float(initial_equity) if initial_equity and initial_equity > 0 else 1.0
    bench_ret = benchmark_return_from_equity_frame(equity_df)

    rows: list[dict[str, Any]] = []
    for strat in sorted(equity_df["strategy_name"].dropna().unique(), key=str):
        sub = equity_df[equity_df["strategy_name"] == strat]
        ts_col = "timestamp" if "timestamp" in sub.columns else None
        if ts_col:
            sub = sub.sort_values(ts_col)
        if sub.empty:
            continue
        eq = pd.to_numeric(sub["equity"], errors="coerce")
        final = float(eq.iloc[-1]) if not eq.empty else ie
        total_ret = final / ie - 1.0 if ie > 0 else 0.0
        mdd = max_drawdown_from_equity(eq)
        excess = (total_ret - bench_ret) if bench_ret is not None else None
        sharpe_simple = _sharpe_simple_daily_proxy(eq)

        tsub = pd.DataFrame()
        if not trades_df.empty and "strategy_name" in trades_df.columns:
            tsub = trades_df[trades_df["strategy_name"].astype(str) == str(strat)]
        pnl_col = None
        for cand in ("realized_pnl", "net_pnl", "pnl"):
            if cand in tsub.columns:
                pnl_col = cand
                break
        realized = 0.0
        n_tr = 0
        win_rate = 0.0
        pf_label = "N/A"
        if pnl_col and not tsub.empty:
            pnl = pd.to_numeric(tsub[pnl_col], errors="coerce").fillna(0.0)
            realized = float(pnl.sum())
            n_tr = int(len(pnl))
            m = compute_trade_performance(tsub.assign(_pnl=pnl), pnl_key="_pnl")
            win_rate = float(m.get("win_rate", 0.0))
            pf_label = str(m.get("profit_factor_label", "N/A"))

        rows.append(
            {
                "strategy_name": strat,
                "final_equity": final,
                "total_return": total_ret,
                "benchmark_return": bench_ret,
                "excess_return": excess,
                "max_drawdown": mdd,
                "sharpe_simple": sharpe_simple,
                "realized_pnl_sum": realized,
                "n_trades": n_tr,
                "win_rate": win_rate,
                "profit_factor": pf_label,
            },
        )
    return pd.DataFrame(rows)


def _pick_final_equity_for_replay_run(eq_run: pd.DataFrame) -> float | None:
    """Prefer ``ensemble`` book; else mean of ``ind::*`` finals; else last row."""

    if eq_run.empty or "strategy_name" not in eq_run.columns or "equity" not in eq_run.columns:
        return None
    s2 = eq_run.sort_values("timestamp")
    last_by = s2.groupby("strategy_name", as_index=False).last()
    if last_by.empty:
        return None
    ens = last_by[last_by["strategy_name"].astype(str) == "ensemble"]
    if not ens.empty:
        return float(pd.to_numeric(ens.iloc[-1]["equity"], errors="coerce"))
    ind = last_by[last_by["strategy_name"].astype(str).str.startswith("ind::")]
    if not ind.empty:
        return float(pd.to_numeric(ind["equity"], errors="coerce").mean())
    return float(pd.to_numeric(last_by.iloc[-1]["equity"], errors="coerce"))


def build_replay_runs_summary_table(db_path: str | Path, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable replay run table: date range, final equity, benchmark return."""

    if runs_df.empty or "run_id" not in runs_df.columns:
        return pd.DataFrame()
    path = Path(db_path)
    eq_all = load_equity_snapshots(path, run_id=None, source_scope="replay", limit=200_000)
    out_rows: list[dict[str, Any]] = []
    for _, rr in runs_df.iterrows():
        rid = str(rr.get("run_id", "") or "")
        sub = (
            eq_all[eq_all["run_id"].astype(str) == rid]
            if not eq_all.empty and "run_id" in eq_all.columns
            else pd.DataFrame()
        )
        spy_ret = benchmark_return_from_equity_frame(sub) if not sub.empty else None
        final_eq = _pick_final_equity_for_replay_run(sub)
        raw_sj = rr.get("strategies_json", "")
        try:
            sj = json.loads(str(raw_sj)) if raw_sj else []
            strat_disp = ", ".join(str(x) for x in sj) if isinstance(sj, list) else str(raw_sj)
        except (json.JSONDecodeError, TypeError):
            strat_disp = str(raw_sj)
        st = str(rr.get("start_time", "") or "")
        en = str(rr.get("end_time", "") or "")
        dr = f"{st[:10]} → {en[:10]}" if len(st) >= 10 and len(en) >= 10 else f"{st} → {en}"
        out_rows.append(
            {
                "Run": rid,
                "Date range": dr,
                "Strategies": strat_disp,
                "Mode": rr.get("mode"),
                "Initial equity": rr.get("initial_equity"),
                "Final equity": final_eq,
                "SPY return": spy_ret,
                "Status": rr.get("status"),
            },
        )
    return pd.DataFrame(out_rows)


__all__ = [
    "benchmark_return_from_equity_frame",
    "benchmark_total_return",
    "build_replay_runs_summary_table",
    "build_strategy_comparison_table",
    "compute_trade_performance",
    "connect_sqlite_readonly",
    "discover_trade_table",
    "load_completed_trades_by_run",
    "load_equity_snapshots",
    "load_replay_runs",
    "load_skip_events",
    "load_strategy_decisions",
    "load_strategy_signals",
    "max_drawdown_from_equity",
]
