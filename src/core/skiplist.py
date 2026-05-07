"""Per-session-day symbol skips (Discord /skip).

Skips expire when the ET calendar ``session_day`` advances (typically after
Friday close → Monday is a fresh day).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from config.constants import LOGGER_APP

_LOG = logging.getLogger(LOGGER_APP)


class SymbolSkiplist:
    """Stores skipped symbols keyed by ET ``YYYY-MM-DD``."""

    FILENAME = "symbol_skiplist_by_day.json"

    def __init__(self, state_dir: Path) -> None:
        self._dir = Path(state_dir)
        self._path = self._dir / self.FILENAME
        self._lock = threading.Lock()

    def load(self) -> dict[str, list[str]]:
        if not self._path.is_file():
            return {}
        try:
            with self._path.open(encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            out: dict[str, list[str]] = {}
            for k, v in raw.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [str(x).upper() for x in v if x]
            return out
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("symbol skiplist read failed: %s", exc)
            return {}

    def _save(self, data: dict[str, list[str]]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)

    def skip_for_session_day(self, *, session_day_et: str, symbol: str) -> None:
        sym = symbol.strip().upper()
        if not sym:
            return
        with self._lock:
            data = self.load()
            day = session_day_et[:10]
            cur = list(dict.fromkeys(data.get(day, [])))
            if sym not in cur:
                cur.append(sym)
            data[day] = cur
            # prune old keys (keep last 5)
            for k in sorted(data.keys())[:-5]:
                if k != day:
                    data.pop(k, None)
            self._save(data)
        _LOG.info(
            "event=discord_skip_symbol symbol=%s session_day_et=%s",
            sym,
            day,
            extra={"symbol": sym},
        )

    def is_skipped(self, *, session_day_et: str, symbol: str) -> bool:
        day = session_day_et[:10]
        with self._lock:
            data = self.load()
        syms = {s.upper() for s in data.get(day, [])}
        return symbol.strip().upper() in syms


__all__ = ["SymbolSkiplist"]
