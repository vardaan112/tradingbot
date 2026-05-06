"""Market hours guard and session windows.

The bot defaults to:
- regular session only (no extended hours)
- no entries in the first DEFAULT_NO_NEW_ENTRY_OPEN_MINUTES of the session
- no new entries in the last DEFAULT_NO_NEW_ENTRY_CLOSE_MINUTES of the session
- exits are allowed during a wider exit window than entries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from alpaca.trading.client import TradingClient

from config.constants import (
    DEFAULT_NO_NEW_ENTRY_CLOSE_MINUTES,
    DEFAULT_NO_NEW_ENTRY_OPEN_MINUTES,
    LOGGER_APP,
    NEW_YORK_TZ,
)
from utils.time_utils import now_eastern

from .exceptions import BrokerConnectionError
from .retries import retry_call


@dataclass(frozen=True)
class MarketSession:
    is_open: bool
    next_open_utc: Optional[datetime]
    next_close_utc: Optional[datetime]
    fetched_at_utc: datetime


def _ensure_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MarketClock:
    """Wraps the broker clock with safe entry/exit window logic."""

    def __init__(
        self,
        trading: TradingClient,
        *,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
        no_entry_open_minutes: int = DEFAULT_NO_NEW_ENTRY_OPEN_MINUTES,
        no_entry_close_minutes: int = DEFAULT_NO_NEW_ENTRY_CLOSE_MINUTES,
    ) -> None:
        self._client = trading
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._no_entry_open = timedelta(minutes=no_entry_open_minutes)
        self._no_entry_close = timedelta(minutes=no_entry_close_minutes)
        self._log = logging.getLogger(LOGGER_APP)
        self._cached: Optional[MarketSession] = None
        self._cache_ttl = timedelta(seconds=15)

    def get_session(self, *, force_refresh: bool = False) -> MarketSession:
        if (
            not force_refresh
            and self._cached is not None
            and datetime.now(timezone.utc) - self._cached.fetched_at_utc < self._cache_ttl
        ):
            return self._cached

        def _do() -> MarketSession:
            try:
                clock = self._client.get_clock()
            except Exception as exc:  # noqa: BLE001
                raise BrokerConnectionError(f"get_clock failed: {exc}") from exc

            return MarketSession(
                is_open=bool(getattr(clock, "is_open", False)),
                next_open_utc=_ensure_aware_utc(getattr(clock, "next_open", None)),
                next_close_utc=_ensure_aware_utc(getattr(clock, "next_close", None)),
                fetched_at_utc=datetime.now(timezone.utc),
            )

        session = retry_call(
            _do,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            op_name="get_clock",
            logger=self._log,
        )
        self._cached = session
        return session

    # ---- Window predicates ------------------------------------------------

    def can_open_new_position(self, session: Optional[MarketSession] = None) -> bool:
        s = session or self.get_session()
        if not s.is_open or s.next_close_utc is None:
            return False
        # Next open is in the future when market is open => use last open window.
        # Approximate "session open time" by current time minus elapsed; we
        # rely on next_close - 6.5h as a stand-in for open. Simpler: treat
        # the window as "more than N minutes from open" by using next_open
        # cycle: if next_close is roughly < 30 min away, block; if minutes
        # since open is < no_entry_open, block.
        now = datetime.now(timezone.utc)
        # Heuristic open: regular session is 6h30m, so deduct from next_close.
        approx_open = s.next_close_utc - timedelta(hours=6, minutes=30)
        if now < approx_open + self._no_entry_open:
            return False
        if now > s.next_close_utc - self._no_entry_close:
            return False
        return True

    def can_exit_position(self, session: Optional[MarketSession] = None) -> bool:
        s = session or self.get_session()
        return bool(s.is_open)

    @staticmethod
    def now_eastern() -> datetime:
        return now_eastern().astimezone(NEW_YORK_TZ)
