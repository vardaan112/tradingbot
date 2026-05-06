"""Durable runtime state stored as small JSON files under STATE_DIR.

The state store deliberately uses simple files (no DB) because:
- the bot's working set is tiny (a handful of small JSONs)
- atomic writes via os.replace are sufficient
- it is trivial to inspect/edit on a VPS

Files used:
- daily_start_equity.json     - {"date": "YYYY-MM-DD", "equity": <float>}
- kill_switch_state.json      - {"latched": bool, "reason": str, "ts": iso8601}
- open_order_index.json       - {symbol: {client_order_id, qty, side, ts}}
- last_session_snapshot.json  - free-form snapshot for diagnostics
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from config.constants import LOGGER_APP


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path_str, path)
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger(LOGGER_APP).error("Failed to read %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class DailyEquityRecord:
    date: str
    equity: float


@dataclass(frozen=True)
class KillSwitchRecord:
    latched: bool
    reason: str = ""
    ts: str = ""
    daily_baseline: float = 0.0
    triggered_equity: float = 0.0


@dataclass
class OpenOrderEntry:
    symbol: str
    client_order_id: str
    qty: float
    side: str
    ts: str
    broker_order_id: Optional[str] = None
    strategy: str = ""


@dataclass
class SessionSnapshot:
    timestamp: str
    equity: float = 0.0
    buying_power: float = 0.0
    open_positions: int = 0
    open_orders: int = 0
    feed: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class StateStore:
    """File-backed state store. Thread-safe via a single coarse lock."""

    DAILY_EQUITY_FILE = "daily_start_equity.json"
    KILL_SWITCH_FILE = "kill_switch_state.json"
    OPEN_ORDER_INDEX_FILE = "open_order_index.json"
    SESSION_SNAPSHOT_FILE = "last_session_snapshot.json"

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---- Daily equity ------------------------------------------------------

    def load_daily_equity(self) -> Optional[DailyEquityRecord]:
        with self._lock:
            data = _read_json(self.state_dir / self.DAILY_EQUITY_FILE)
            if not data:
                return None
            try:
                return DailyEquityRecord(date=str(data["date"]), equity=float(data["equity"]))
            except (KeyError, TypeError, ValueError):
                return None

    def save_daily_equity(self, record: DailyEquityRecord) -> None:
        with self._lock:
            _atomic_write_json(self.state_dir / self.DAILY_EQUITY_FILE, asdict(record))

    # ---- Kill switch -------------------------------------------------------

    def load_kill_switch(self) -> KillSwitchRecord:
        with self._lock:
            data = _read_json(self.state_dir / self.KILL_SWITCH_FILE)
            if not data:
                return KillSwitchRecord(latched=False)
            try:
                return KillSwitchRecord(
                    latched=bool(data.get("latched", False)),
                    reason=str(data.get("reason", "")),
                    ts=str(data.get("ts", "")),
                    daily_baseline=float(data.get("daily_baseline", 0.0)),
                    triggered_equity=float(data.get("triggered_equity", 0.0)),
                )
            except (TypeError, ValueError):
                return KillSwitchRecord(latched=False)

    def save_kill_switch(self, record: KillSwitchRecord) -> None:
        with self._lock:
            _atomic_write_json(self.state_dir / self.KILL_SWITCH_FILE, asdict(record))

    # ---- Open order index --------------------------------------------------

    def load_open_orders(self) -> dict[str, OpenOrderEntry]:
        with self._lock:
            data = _read_json(self.state_dir / self.OPEN_ORDER_INDEX_FILE)
            if not data:
                return {}
            out: dict[str, OpenOrderEntry] = {}
            for sym, raw in data.items():
                try:
                    out[sym] = OpenOrderEntry(
                        symbol=str(raw["symbol"]),
                        client_order_id=str(raw["client_order_id"]),
                        qty=float(raw["qty"]),
                        side=str(raw["side"]),
                        ts=str(raw["ts"]),
                        broker_order_id=raw.get("broker_order_id"),
                        strategy=str(raw.get("strategy", "")),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            return out

    def save_open_orders(self, entries: dict[str, OpenOrderEntry]) -> None:
        with self._lock:
            payload = {sym: asdict(entry) for sym, entry in entries.items()}
            _atomic_write_json(self.state_dir / self.OPEN_ORDER_INDEX_FILE, payload)

    # ---- Session snapshot --------------------------------------------------

    def save_session_snapshot(self, snapshot: SessionSnapshot) -> None:
        with self._lock:
            _atomic_write_json(self.state_dir / self.SESSION_SNAPSHOT_FILE, asdict(snapshot))

    def load_session_snapshot(self) -> Optional[SessionSnapshot]:
        with self._lock:
            data = _read_json(self.state_dir / self.SESSION_SNAPSHOT_FILE)
            if not data:
                return None
            try:
                return SessionSnapshot(
                    timestamp=str(data["timestamp"]),
                    equity=float(data.get("equity", 0.0)),
                    buying_power=float(data.get("buying_power", 0.0)),
                    open_positions=int(data.get("open_positions", 0)),
                    open_orders=int(data.get("open_orders", 0)),
                    feed=str(data.get("feed", "")),
                    extra=dict(data.get("extra", {})),
                )
            except (KeyError, TypeError, ValueError):
                return None
