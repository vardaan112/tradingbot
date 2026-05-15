"""SQLite persistence for trades, sentiment, canary audits, and execution events.

Uses WAL mode and parameterized queries only. Writes are best-effort: callers
must catch failures for non-critical paths.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from config.constants import LOGGER_APP
from core.trade_source import sql_broker_eligible_sources_clause

_LOG = logging.getLogger(LOGGER_APP)


def _json_metadata(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    return json.dumps(metadata, sort_keys=True)


def _norm_run_id(run_id: Optional[str]) -> Optional[str]:
    if run_id is None:
        return None
    s = str(run_id).strip()
    return s or None


@dataclass(frozen=True)
class CompletedTradeRow:
    id: int
    trade_id: Optional[str]
    symbol: str
    side: str
    quantity: float
    entry_price: Optional[float]
    exit_price: Optional[float]
    realized_pnl: Optional[float]
    realized_return: Optional[float]
    opened_at: Optional[str]
    closed_at: str
    strategy_name: Optional[str]
    risk_mode: Optional[str]
    regime_type: Optional[str]
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    is_canary: int
    source: Optional[str] = None
    realized_return_pct: Optional[float] = None
    entry_notional: Optional[float] = None
    exit_notional: Optional[float] = None
    entry_fill_source: Optional[str] = None
    exit_fill_source: Optional[str] = None
    invalid_for_ml: int = 0
    invalid_for_kelly: int = 0


class Database:
    """Lightweight SQLite access for Phase 4 analytics."""

    SCHEMA_VERSION = 3

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._path,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except sqlite3.Error as exc:
            _LOG.warning("event=db_pragma_issue error=%s", exc)
        return conn

    def init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_meta (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS completed_trades (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      trade_id TEXT UNIQUE,
                      symbol TEXT NOT NULL,
                      side TEXT NOT NULL,
                      quantity REAL NOT NULL,
                      entry_price REAL,
                      exit_price REAL,
                      realized_pnl REAL,
                      realized_return REAL,
                      opened_at TEXT,
                      closed_at TEXT NOT NULL,
                      strategy_name TEXT,
                      risk_mode TEXT,
                      regime_type TEXT,
                      sentiment_score REAL,
                      sentiment_label TEXT,
                      is_canary INTEGER DEFAULT 0,
                      realized_return_pct REAL,
                      entry_notional REAL,
                      exit_notional REAL,
                      entry_fill_source TEXT,
                      exit_fill_source TEXT,
                      data_quality_flags_json TEXT,
                      invalid_for_ml INTEGER DEFAULT 0,
                      invalid_for_kelly INTEGER DEFAULT 0,
                      metadata_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS sentiment_scores (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      symbol TEXT NOT NULL,
                      score REAL NOT NULL,
                      label TEXT NOT NULL,
                      headline_count INTEGER NOT NULL,
                      latest_headline_timestamp TEXT,
                      stale_news INTEGER DEFAULT 0,
                      created_at TEXT NOT NULL,
                      metadata_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS canary_results (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      success INTEGER NOT NULL,
                      symbol TEXT,
                      quantity REAL,
                      notional REAL,
                      error TEXT,
                      created_at TEXT NOT NULL,
                      metadata_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS execution_events (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      event_type TEXT NOT NULL,
                      symbol TEXT,
                      side TEXT,
                      client_order_id TEXT,
                      order_id TEXT,
                      status TEXT,
                      price REAL,
                      quantity REAL,
                      created_at TEXT NOT NULL,
                      metadata_json TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_trades_closed
                      ON completed_trades(closed_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_trades_closed_day
                      ON completed_trades(closed_at);
                    CREATE INDEX IF NOT EXISTS idx_exec_created
                      ON execution_events(created_at);
                    CREATE INDEX IF NOT EXISTS idx_sentiment_created
                      ON sentiment_scores(created_at);
                    """
                )
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?, ?)",
                    ("version", str(self.SCHEMA_VERSION)),
                )
                conn.commit()
            finally:
                conn.close()
        self.apply_migrations()
        _LOG.info("event=db_schema_initialized path=%s", self._path)

    def _ensure_research_event_tables(self, conn: sqlite3.Connection) -> None:
        """Phase 1: research / replay / signal event tables (CREATE IF NOT EXISTS)."""

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT UNIQUE NOT NULL,
              created_at TEXT NOT NULL,
              start_time TEXT NOT NULL,
              end_time TEXT NOT NULL,
              lookback_days INTEGER,
              timeframe TEXT NOT NULL,
              symbols_json TEXT NOT NULL,
              strategies_json TEXT NOT NULL,
              mode TEXT NOT NULL,
              initial_equity REAL NOT NULL,
              data_feed TEXT,
              benchmark_symbol TEXT NOT NULL DEFAULT 'SPY',
              settings_json TEXT,
              status TEXT NOT NULL,
              error TEXT
            );

            CREATE TABLE IF NOT EXISTS strategy_signals (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              source TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              symbol TEXT NOT NULL,
              strategy_name TEXT NOT NULL,
              action TEXT NOT NULL,
              confidence REAL,
              reference_price REAL,
              reason TEXT,
              metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS strategy_decisions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              source TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              symbol TEXT NOT NULL,
              decision_type TEXT,
              final_action TEXT NOT NULL,
              weighted_score REAL,
              threshold REAL,
              contributing_signals_json TEXT,
              metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS equity_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              source TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              strategy_name TEXT,
              cash REAL,
              equity REAL,
              realized_pnl REAL,
              unrealized_pnl REAL,
              gross_exposure REAL,
              net_exposure REAL,
              benchmark_equity REAL,
              metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS skip_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT,
              source TEXT NOT NULL,
              timestamp TEXT NOT NULL,
              symbol TEXT,
              strategy_name TEXT,
              phase TEXT,
              skip_code TEXT,
              message TEXT,
              metadata_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_replay_runs_created
              ON replay_runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_replay_runs_status
              ON replay_runs(status);
            CREATE INDEX IF NOT EXISTS idx_strategy_signals_run_ts
              ON strategy_signals(run_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_strategy_signals_src_sym
              ON strategy_signals(source, symbol);
            CREATE INDEX IF NOT EXISTS idx_strategy_decisions_run_ts
              ON strategy_decisions(run_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_equity_snapshots_run_ts
              ON equity_snapshots(run_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_equity_snapshots_src_ts
              ON equity_snapshots(source, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_skip_events_run_ts
              ON skip_events(run_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_skip_events_src_phase
              ON skip_events(source, phase);
            """
        )

    def apply_migrations(self) -> None:
        """Additive ALTERs for simulations / replay."""

        specs = (
            ("completed_trades", "source", "TEXT DEFAULT 'live'"),
            ("completed_trades", "replay_run_id", "TEXT"),
            ("completed_trades", "inserted_at", "TEXT"),
            ("completed_trades", "original_entry_time", "TEXT"),
            ("completed_trades", "original_exit_time", "TEXT"),
            ("completed_trades", "realized_return_pct", "REAL"),
            ("completed_trades", "entry_notional", "REAL"),
            ("completed_trades", "exit_notional", "REAL"),
            ("completed_trades", "entry_fill_source", "TEXT"),
            ("completed_trades", "exit_fill_source", "TEXT"),
            ("completed_trades", "data_quality_flags_json", "TEXT"),
            ("completed_trades", "invalid_for_ml", "INTEGER DEFAULT 0"),
            ("completed_trades", "invalid_for_kelly", "INTEGER DEFAULT 0"),
            ("execution_events", "source", "TEXT DEFAULT 'live'"),
            ("execution_events", "replay_run_id", "TEXT"),
            ("execution_events", "simulated_timestamp", "TEXT"),
        )
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_meta (
                      key TEXT PRIMARY KEY,
                      value TEXT NOT NULL
                    );
                    """
                )

                def _table_exists(table: str) -> bool:
                    row = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    ).fetchone()
                    return row is not None

                def _has_col(table: str, name: str) -> bool:
                    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                    return any(str(r[1]) == name for r in rows)

                for table, col, decl in specs:
                    if not _table_exists(table):
                        continue
                    if not _has_col(table, col):
                        try:
                            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                        except sqlite3.Error as exc:
                            _LOG.warning(
                                "event=db_migrate_issue table=%s col=%s error=%s",
                                table,
                                col,
                                exc,
                            )
                self._ensure_research_event_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key,value) VALUES(?, ?)",
                    ("version", str(self.SCHEMA_VERSION)),
                )
                conn.commit()
            finally:
                conn.close()

    # --------------------------------------------------------------------- trades

    def record_completed_trade(
        self,
        *,
        trade_id: Optional[str],
        symbol: str,
        side: str,
        quantity: float,
        entry_price: Optional[float],
        exit_price: Optional[float],
        realized_pnl: Optional[float],
        realized_return: Optional[float],
        opened_at: Optional[str],
        closed_at: str,
        strategy_name: Optional[str],
        risk_mode: Optional[str],
        regime_type: Optional[str],
        sentiment_score: Optional[float],
        sentiment_label: Optional[str],
        is_canary: int,
        metadata: Optional[dict[str, Any]] = None,
        source: str = "live",
        replay_run_id: Optional[str] = None,
        inserted_at: Optional[str] = None,
        original_entry_time: Optional[str] = None,
        original_exit_time: Optional[str] = None,
        realized_return_pct: Optional[float] = None,
        entry_notional: Optional[float] = None,
        exit_notional: Optional[float] = None,
        entry_fill_source: Optional[str] = None,
        exit_fill_source: Optional[str] = None,
        data_quality_flags: Optional[dict[str, Any]] = None,
        invalid_for_ml: bool = False,
        invalid_for_kelly: bool = False,
    ) -> Optional[int]:
        md = dict(metadata or {})
        flags = dict(data_quality_flags or {})
        if isinstance(md.get("data_quality_flags"), dict):
            flags.update(md.get("data_quality_flags") or {})
        invalid_for_ml = bool(invalid_for_ml or md.get("invalid_for_ml") is True)
        invalid_for_kelly = bool(invalid_for_kelly or md.get("invalid_for_kelly") is True)
        realized_return_pct = realized_return if realized_return_pct is None else realized_return_pct
        if entry_notional is None and entry_price is not None:
            entry_notional = abs(float(quantity) * float(entry_price))
        if exit_notional is None and exit_price is not None:
            exit_notional = abs(float(quantity) * float(exit_price))
        flags_json = json.dumps(flags, sort_keys=True) if flags else None
        meta = json.dumps(md, sort_keys=True) if md else None
        ins_ts = inserted_at or datetime.now(timezone.utc).isoformat()
        oet = original_entry_time or opened_at
        oxt = original_exit_time or closed_at
        sql = """
        INSERT INTO completed_trades (
          trade_id, symbol, side, quantity, entry_price, exit_price,
          realized_pnl, realized_return, opened_at, closed_at,
          strategy_name, risk_mode, regime_type, sentiment_score,
          sentiment_label, is_canary, metadata_json,
          source, replay_run_id, inserted_at, original_entry_time, original_exit_time,
          realized_return_pct, entry_notional, exit_notional,
          entry_fill_source, exit_fill_source, data_quality_flags_json,
          invalid_for_ml, invalid_for_kelly
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            trade_id,
            symbol.upper(),
            side,
            float(quantity),
            entry_price,
            exit_price,
            realized_pnl,
            realized_return,
            opened_at,
            closed_at,
            strategy_name,
            risk_mode,
            regime_type,
            sentiment_score,
            sentiment_label,
            int(is_canary),
            meta,
            source,
            replay_run_id,
            ins_ts,
            oet,
            oxt,
            realized_return_pct,
            entry_notional,
            exit_notional,
            entry_fill_source,
            exit_fill_source,
            flags_json,
            1 if invalid_for_ml else 0,
            1 if invalid_for_kelly else 0,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error(
                    "event=db_write_error kind=completed_trade error=%s", exc,
                )
                return None

    def get_recent_completed_trades(self, *, limit: int) -> list[CompletedTradeRow]:
        if limit < 1:
            return []
        sql = """
        SELECT id, trade_id, symbol, side, quantity, entry_price, exit_price,
               realized_pnl, realized_return, opened_at, closed_at,
               strategy_name, risk_mode, regime_type, sentiment_score,
               sentiment_label, is_canary, COALESCE(source, 'live') AS source
        FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
        ORDER BY datetime(closed_at) DESC
        LIMIT ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, (limit,)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error(
                    "event=db_write_error kind=query_recent_trades error=%s", exc,
                )
                return []
        return [
            CompletedTradeRow(
                id=int(r["id"]),
                trade_id=r["trade_id"],
                symbol=str(r["symbol"]),
                side=str(r["side"]),
                quantity=float(r["quantity"]),
                entry_price=r["entry_price"],
                exit_price=r["exit_price"],
                realized_pnl=r["realized_pnl"],
                realized_return=r["realized_return"],
                opened_at=r["opened_at"],
                closed_at=str(r["closed_at"]),
                strategy_name=r["strategy_name"],
                risk_mode=r["risk_mode"],
                regime_type=r["regime_type"],
                sentiment_score=r["sentiment_score"],
                sentiment_label=r["sentiment_label"],
                is_canary=int(r["is_canary"] or 0),
                source=r["source"],
            )
            for r in rows
        ]

    def query_completed_trades_for_performance(
        self,
        *,
        source: str,
        closed_after_iso: str,
        limit: int = 50_000,
    ) -> list[sqlite3.Row]:
        """Completed trades for ensemble performance weights (single ``source`` filter)."""

        lim = max(1, min(int(limit), 100_000))
        src = str(source).strip().lower()
        sql = """
        SELECT strategy_name, realized_pnl, realized_return, entry_notional,
               quantity, entry_price, closed_at, COALESCE(source, 'live') AS source
        FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND COALESCE(source, 'live') = ?
          AND datetime(closed_at) >= datetime(?)
          AND strategy_name IS NOT NULL
          AND TRIM(strategy_name) != ''
        ORDER BY datetime(closed_at) ASC
        LIMIT ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, (src, closed_after_iso, lim)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_perf_trades error=%s", exc)
                return []

    def _day_bounds_et(self, trading_day: str) -> tuple[str, str]:
        """closed_at compares as ISO strings; bound full ET calendar day."""
        start = f"{trading_day}T00:00:00"
        end = f"{trading_day}T23:59:59"
        return start, end

    def get_completed_trades_for_calendar_day_et(
        self,
        *,
        trading_day_yyyy_mm_dd: str,
        exclude_canary: bool = True,
    ) -> list[sqlite3.Row]:
        start, end = self._day_bounds_et(trading_day_yyyy_mm_dd)
        clause = ""
        params: Sequence[Any]
        if exclude_canary:
            clause = " AND COALESCE(is_canary,0)=0 "
        sql = f"""
        SELECT realized_pnl, sentiment_score, sentiment_label, regime_type,
               symbol, closed_at
        FROM completed_trades
        WHERE closed_at >= ? AND closed_at <= ? {clause}
        ORDER BY datetime(closed_at) ASC
        """
        params = (start, end)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, params).fetchall()
                finally:
                    conn.close()
                return list(rows)
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=day_trades error=%s", exc)
                return []

    def get_recent_realized_pnls_for_kelly(
        self,
        *,
        limit: int,
        exclude_simulation: bool = True,
    ) -> list[float]:
        if limit < 1:
            return []
        # When True, restrict to broker path (live/paper); excludes simulation,
        # replay, shadow, dry_run, and any future non-broker labels.
        src_clause = sql_broker_eligible_sources_clause(enabled=exclude_simulation)
        sql = f"""
        SELECT realized_pnl FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND realized_pnl IS NOT NULL
          AND COALESCE(invalid_for_kelly, 0) = 0
          {src_clause}
        ORDER BY datetime(closed_at) DESC
        LIMIT ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, (limit,)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=kelly_pnls error=%s", exc)
                return []
        out: list[float] = []
        for r in rows:
            try:
                v = float(r["realized_pnl"])
                if math.isfinite(v):
                    out.append(v)
            except (TypeError, ValueError):
                continue
        return out

    def get_recent_realized_returns_for_kelly(
        self,
        *,
        limit: int,
        exclude_simulation: bool = True,
        exclude_canary: bool = True,
        exclude_degraded: bool = True,
    ) -> list[float]:
        """Return normalized per-trade returns for Kelly.

        Raw dollar P&L is not comparable across varying position notionals. This
        method uses stored return fields first, then falls back to
        realized_pnl / entry_notional when that label is complete.
        """

        if limit < 1:
            return []
        src_clause = sql_broker_eligible_sources_clause(enabled=exclude_simulation)
        canary_clause = " AND COALESCE(is_canary, 0) = 0 " if exclude_canary else ""
        degraded_clause = " AND COALESCE(invalid_for_kelly, 0) = 0 " if exclude_degraded else ""
        sql = f"""
        SELECT realized_return_pct, realized_return, realized_pnl, entry_notional,
               COALESCE(metadata_json,'') AS metadata_json,
               COALESCE(data_quality_flags_json,'') AS data_quality_flags_json,
               COALESCE(entry_fill_source,'') AS entry_fill_source,
               COALESCE(exit_fill_source,'') AS exit_fill_source
        FROM completed_trades
        WHERE (
            realized_return_pct IS NOT NULL
            OR realized_return IS NOT NULL
            OR (realized_pnl IS NOT NULL AND entry_notional IS NOT NULL AND entry_notional > 0)
        )
        {canary_clause}
        {src_clause}
        {degraded_clause}
        ORDER BY datetime(closed_at) DESC
        LIMIT ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, (limit,)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=kelly_returns error=%s", exc)
                return []

        out: list[float] = []
        for r in rows:
            if exclude_degraded:
                try:
                    md = json.loads(str(r["metadata_json"] or "{}"))
                    flags = json.loads(str(r["data_quality_flags_json"] or "{}"))
                except json.JSONDecodeError:
                    md = {}
                    flags = {}
                if isinstance(md, dict) and (md.get("invalid_for_kelly") or md.get("invalid_for_ml")):
                    continue
                if isinstance(flags, dict) and flags:
                    continue
                if str(r["exit_fill_source"] or "").lower() in {"quote_mid_fallback", "degraded_fallback"}:
                    continue
            try:
                raw = r["realized_return_pct"]
                if raw is None:
                    raw = r["realized_return"]
                if raw is None and r["entry_notional"] not in {None, 0}:
                    raw = float(r["realized_pnl"]) / float(r["entry_notional"])
                v = float(raw)
                if math.isfinite(v):
                    out.append(v)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        return out

    def get_ml_training_rows(
        self,
        *,
        limit: int,
        exclude_simulation: bool = True,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        src_clause = sql_broker_eligible_sources_clause(enabled=exclude_simulation)
        sql = f"""
        SELECT symbol, realized_pnl, opened_at, closed_at,
               sentiment_score, regime_type,
               COALESCE(metadata_json,'') AS metadata_json
        FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND realized_pnl IS NOT NULL
          AND COALESCE(invalid_for_ml, 0) = 0
          {src_clause}
        ORDER BY datetime(closed_at) DESC
        LIMIT ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    raw = conn.execute(sql, (limit,)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=ml_training_rows error=%s", exc)
                return []

        parsed: list[dict[str, Any]] = []
        for r in raw:
            mj: dict[str, Any] = {}
            if r["metadata_json"]:
                try:
                    obj = json.loads(str(r["metadata_json"]))
                    if isinstance(obj, dict):
                        mj = obj
                except json.JSONDecodeError:
                    mj = {}
            parsed.append(
                {
                    "symbol": str(r["symbol"] or "").upper(),
                    "realized_pnl": float(r["realized_pnl"])
                    if r["realized_pnl"] is not None
                    else 0.0,
                    "opened_at": r["opened_at"],
                    "closed_at": r["closed_at"],
                    "sentiment_score": r["sentiment_score"],
                    "regime_type": r["regime_type"],
                    "metadata": mj,
                },
            )
        return parsed

    def count_completed_trades_ml_eligible(
        self,
        *,
        exclude_canary: bool = True,
        exclude_simulation: bool = True,
    ) -> int:
        src_clause = sql_broker_eligible_sources_clause(enabled=exclude_simulation)
        canary_clause = " AND COALESCE(is_canary, 0) = 0 " if exclude_canary else ""
        sql = f"""
        SELECT COUNT(*) AS n FROM completed_trades
        WHERE realized_pnl IS NOT NULL
        {canary_clause}
        {src_clause}
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(sql).fetchone()
                    return int(row["n"]) if row and row["n"] is not None else 0
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=ml_trade_count error=%s", exc)
                return 0

    def sum_realized_pnl_all_live(self) -> float:
        sql = """
        SELECT COALESCE(SUM(realized_pnl), 0) AS s FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND COALESCE(source, 'live') IN ('live', 'paper')
          AND realized_pnl IS NOT NULL
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(sql).fetchone()
                    return float(row["s"]) if row and row["s"] is not None else 0.0
                finally:
                    conn.close()
            except sqlite3.Error:
                return 0.0

    # ------------------------------------------------------------------ sentiment

    def record_sentiment_score(
        self,
        *,
        symbol: str,
        score: float,
        label: str,
        headline_count: int,
        latest_headline_timestamp: Optional[str],
        stale_news: int,
        created_at: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ts = created_at or datetime.now(timezone.utc).isoformat()
        meta = json.dumps(metadata) if metadata else None
        sql = """
        INSERT INTO sentiment_scores (
          symbol, score, label, headline_count, latest_headline_timestamp,
          stale_news, created_at, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?)
        """
        args = (
            symbol.upper(),
            float(score),
            label,
            int(headline_count),
            latest_headline_timestamp,
            int(stale_news),
            ts,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(sql, args)
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error(
                    "event=db_write_error kind=sentiment_score error=%s", exc,
                )

    def get_today_sentiment_scores(
        self,
        *,
        trading_day_yyyy_mm_dd: str,
    ) -> list[sqlite3.Row]:
        start, end = self._day_bounds_et(trading_day_yyyy_mm_dd)
        sql = """
        SELECT * FROM sentiment_scores
        WHERE created_at >= ? AND created_at <= ?
        ORDER BY id ASC
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, (start, end)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=sentiment_list error=%s", exc)
                return []

    # -------------------------------------------------------------------- canary

    def record_canary_result(
        self,
        *,
        success: bool,
        symbol: Optional[str],
        quantity: Optional[float],
        notional: Optional[float],
        error: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        meta = json.dumps(metadata) if metadata else None
        sql = """
        INSERT INTO canary_results (
          success, symbol, quantity, notional, error, created_at, metadata_json
        ) VALUES (?,?,?,?,?,?,?)
        """
        args = (
            1 if success else 0,
            symbol,
            quantity,
            notional,
            error,
            ts,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(sql, args)
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=canary_result error=%s", exc)

    # ------------------------------------------------------------- execution log

    def record_execution_event(
        self,
        *,
        event_type: str,
        symbol: Optional[str],
        side: Optional[str],
        client_order_id: Optional[str],
        order_id: Optional[str],
        status: Optional[str],
        price: Optional[float],
        quantity: Optional[float],
        metadata: Optional[dict[str, Any]] = None,
        source: str = "live",
        replay_run_id: Optional[str] = None,
        created_at: Optional[str] = None,
        simulated_timestamp: Optional[str] = None,
    ) -> None:
        ts = created_at or datetime.now(timezone.utc).isoformat()
        meta_obj = dict(metadata or {})
        if simulated_timestamp:
            meta_obj.setdefault("simulated_timestamp", simulated_timestamp)
        meta = json.dumps(meta_obj) if meta_obj else None
        sql = """
        INSERT INTO execution_events (
          event_type, symbol, side, client_order_id, order_id, status,
          price, quantity, created_at, metadata_json, source, replay_run_id,
          simulated_timestamp
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            event_type,
            symbol.upper() if symbol else None,
            side,
            client_order_id,
            order_id,
            status,
            price,
            quantity,
            ts,
            meta,
            source,
            replay_run_id,
            simulated_timestamp,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    conn.execute(sql, args)
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error(
                    "event=db_write_error kind=execution_event error=%s", exc,
                )

    def count_execution_events(
        self,
        *,
        event_type: str,
        trading_day_yyyy_mm_dd: str,
    ) -> int:
        start, end = self._day_bounds_et(trading_day_yyyy_mm_dd)
        sql = """
        SELECT COUNT(*) FROM execution_events
        WHERE event_type = ? AND created_at >= ? AND created_at <= ?
        """
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(sql, (event_type, start, end)).fetchone()
                    return int(row[0]) if row else 0
                finally:
                    conn.close()
            except sqlite3.Error:
                return 0

    # ------------------------------------------------------------------ Phase 1 research / events

    def create_replay_run(
        self,
        *,
        run_id: str,
        start_time: str,
        end_time: str,
        timeframe: str,
        symbols_json: str,
        strategies_json: str,
        mode: str,
        initial_equity: float,
        lookback_days: Optional[int] = None,
        data_feed: Optional[str] = None,
        benchmark_symbol: str = "SPY",
        settings_json: Optional[str] = None,
        status: str = "running",
        error: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Optional[int]:
        ts = created_at or datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO replay_runs (
          run_id, created_at, start_time, end_time, lookback_days, timeframe,
          symbols_json, strategies_json, mode, initial_equity, data_feed,
          benchmark_symbol, settings_json, status, error
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            run_id.strip(),
            ts,
            start_time,
            end_time,
            lookback_days,
            timeframe,
            symbols_json,
            strategies_json,
            mode,
            float(initial_equity),
            data_feed,
            benchmark_symbol,
            settings_json,
            status,
            error,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=replay_run_create error=%s", exc)
                return None

    def finish_replay_run(
        self,
        *,
        run_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        sql = "UPDATE replay_runs SET status = ?, error = ? WHERE run_id = ?"
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, (status, error, run_id.strip()))
                    conn.commit()
                    return int(cur.rowcount or 0) > 0
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=replay_run_finish error=%s", exc)
                return False

    def record_strategy_signal(
        self,
        *,
        source: str,
        timestamp: str,
        symbol: str,
        strategy_name: str,
        action: str,
        run_id: Optional[str] = None,
        confidence: Optional[float] = None,
        reference_price: Optional[float] = None,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        meta = _json_metadata(metadata)
        sql = """
        INSERT INTO strategy_signals (
          run_id, source, timestamp, symbol, strategy_name, action,
          confidence, reference_price, reason, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            _norm_run_id(run_id),
            source,
            timestamp,
            symbol.upper(),
            strategy_name,
            action,
            confidence,
            reference_price,
            reason,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=strategy_signal error=%s", exc)
                return None

    def record_strategy_decision(
        self,
        *,
        source: str,
        timestamp: str,
        symbol: str,
        final_action: str,
        run_id: Optional[str] = None,
        decision_type: Optional[str] = None,
        weighted_score: Optional[float] = None,
        threshold: Optional[float] = None,
        contributing_signals_json: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        meta = _json_metadata(metadata)
        sql = """
        INSERT INTO strategy_decisions (
          run_id, source, timestamp, symbol, decision_type, final_action,
          weighted_score, threshold, contributing_signals_json, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            _norm_run_id(run_id),
            source,
            timestamp,
            symbol.upper(),
            decision_type,
            final_action,
            weighted_score,
            threshold,
            contributing_signals_json,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=strategy_decision error=%s", exc)
                return None

    def record_equity_snapshot(
        self,
        *,
        source: str,
        timestamp: str,
        run_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        cash: Optional[float] = None,
        equity: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        gross_exposure: Optional[float] = None,
        net_exposure: Optional[float] = None,
        benchmark_equity: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        meta = _json_metadata(metadata)
        sql = """
        INSERT INTO equity_snapshots (
          run_id, source, timestamp, strategy_name, cash, equity,
          realized_pnl, unrealized_pnl, gross_exposure, net_exposure,
          benchmark_equity, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        args = (
            _norm_run_id(run_id),
            source,
            timestamp,
            strategy_name,
            cash,
            equity,
            realized_pnl,
            unrealized_pnl,
            gross_exposure,
            net_exposure,
            benchmark_equity,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=equity_snapshot error=%s", exc)
                return None

    def record_skip_event(
        self,
        *,
        source: str,
        timestamp: str,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        phase: Optional[str] = None,
        skip_code: Optional[str] = None,
        message: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[int]:
        meta = _json_metadata(metadata)
        sql = """
        INSERT INTO skip_events (
          run_id, source, timestamp, symbol, strategy_name, phase,
          skip_code, message, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """
        args = (
            _norm_run_id(run_id),
            source,
            timestamp,
            symbol.upper() if symbol else None,
            strategy_name,
            phase,
            skip_code,
            message,
            meta,
        )
        with self._lock:
            try:
                conn = self._connect()
                try:
                    cur = conn.execute(sql, args)
                    rid = int(cur.lastrowid)
                    conn.commit()
                    return rid
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_write_error kind=skip_event error=%s", exc)
                return None

    def query_replay_runs(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"""
        SELECT * FROM replay_runs
        {where}
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, tuple(params)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_replay_runs error=%s", exc)
                return []

    def query_strategy_signals(
        self,
        *,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if strategy_name is not None:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"""
        SELECT * FROM strategy_signals
        {where}
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, tuple(params)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_strategy_signals error=%s", exc)
                return []

    def query_strategy_decisions(
        self,
        *,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        symbol: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if decision_type is not None:
            clauses.append("COALESCE(decision_type, '') = ?")
            params.append(decision_type)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"""
        SELECT * FROM strategy_decisions
        {where}
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, tuple(params)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_strategy_decisions error=%s", exc)
                return []

    def query_equity_snapshots(
        self,
        *,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        strategy_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if strategy_name is not None:
            clauses.append("COALESCE(strategy_name, '') = ?")
            params.append(strategy_name)
        if start_time is not None:
            clauses.append("datetime(timestamp) >= datetime(?)")
            params.append(start_time)
        if end_time is not None:
            clauses.append("datetime(timestamp) <= datetime(?)")
            params.append(end_time)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"""
        SELECT * FROM equity_snapshots
        {where}
        ORDER BY datetime(timestamp) ASC, id ASC
        LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, tuple(params)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_equity_snapshots error=%s", exc)
                return []

    def query_skip_events(
        self,
        *,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
        symbol: Optional[str] = None,
        phase: Optional[str] = None,
        limit: int = 500,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if phase is not None:
            clauses.append("phase = ?")
            params.append(phase)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        sql = f"""
        SELECT * FROM skip_events
        {where}
        ORDER BY datetime(timestamp) DESC, id DESC
        LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    return list(conn.execute(sql, tuple(params)).fetchall())
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                _LOG.error("event=db_read_error kind=query_skip_events error=%s", exc)
                return []

    def count_canary_results_for_calendar_day_et(
        self,
        *,
        trading_day_yyyy_mm_dd: str,
        successes_only: bool = False,
    ) -> int:
        """Count canary audits whose ``created_at`` falls in this ET calendar day."""

        start, end = self._day_bounds_et(trading_day_yyyy_mm_dd)
        if successes_only:
            sql = """
            SELECT COUNT(*) FROM canary_results
            WHERE success = 1 AND created_at >= ? AND created_at <= ?
            """
        else:
            sql = """
            SELECT COUNT(*) FROM canary_results
            WHERE created_at >= ? AND created_at <= ?
            """
        args: Sequence[Any] = (start, end)
        with self._lock:
            try:
                conn = self._connect()
                try:
                    row = conn.execute(sql, args).fetchone()
                    return int(row[0]) if row else 0
                finally:
                    conn.close()
            except sqlite3.Error:
                return 0


__all__ = ["Database", "CompletedTradeRow"]
