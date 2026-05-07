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

_LOG = logging.getLogger(LOGGER_APP)


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


class Database:
    """Lightweight SQLite access for Phase 4 analytics."""

    SCHEMA_VERSION = 2

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

    def apply_migrations(self) -> None:
        """Additive ALTERs for simulations / replay."""

        specs = (
            ("completed_trades", "source", "TEXT DEFAULT 'live'"),
            ("completed_trades", "replay_run_id", "TEXT"),
            ("completed_trades", "inserted_at", "TEXT"),
            ("completed_trades", "original_entry_time", "TEXT"),
            ("completed_trades", "original_exit_time", "TEXT"),
            ("execution_events", "source", "TEXT DEFAULT 'live'"),
            ("execution_events", "replay_run_id", "TEXT"),
            ("execution_events", "simulated_timestamp", "TEXT"),
        )
        with self._lock:
            conn = self._connect()
            try:

                def _has_col(table: str, name: str) -> bool:
                    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                    return any(str(r[1]) == name for r in rows)

                for table, col, decl in specs:
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
    ) -> Optional[int]:
        meta = json.dumps(metadata) if metadata else None
        ins_ts = inserted_at or datetime.now(timezone.utc).isoformat()
        oet = original_entry_time or opened_at
        oxt = original_exit_time or closed_at
        sql = """
        INSERT INTO completed_trades (
          trade_id, symbol, side, quantity, entry_price, exit_price,
          realized_pnl, realized_return, opened_at, closed_at,
          strategy_name, risk_mode, regime_type, sentiment_score,
          sentiment_label, is_canary, metadata_json,
          source, replay_run_id, inserted_at, original_entry_time, original_exit_time
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        sim_clause = " AND COALESCE(source,'live') NOT IN ('simulation') " if exclude_simulation else ""
        sql = f"""
        SELECT realized_pnl FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND realized_pnl IS NOT NULL
          {sim_clause}
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

    def get_ml_training_rows(
        self,
        *,
        limit: int,
        exclude_simulation: bool = True,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        sim_clause = " AND COALESCE(source,'live') NOT IN ('simulation') " if exclude_simulation else ""
        sql = f"""
        SELECT symbol, realized_pnl, opened_at, closed_at,
               sentiment_score, regime_type,
               COALESCE(metadata_json,'') AS metadata_json
        FROM completed_trades
        WHERE COALESCE(is_canary, 0) = 0
          AND realized_pnl IS NOT NULL
          {sim_clause}
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
        sim_clause = " AND COALESCE(source,'live') NOT IN ('simulation') " if exclude_simulation else ""
        canary_clause = " AND COALESCE(is_canary, 0) = 0 " if exclude_canary else ""
        sql = f"""
        SELECT COUNT(*) AS n FROM completed_trades
        WHERE realized_pnl IS NOT NULL
        {canary_clause}
        {sim_clause}
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
          AND COALESCE(source,'live') NOT IN ('simulation')
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
