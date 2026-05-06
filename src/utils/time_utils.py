"""Timezone-aware date/time helpers.

All datetimes used by the trading bot must be timezone-aware. America/New_York
is the canonical exchange time; UTC is used for log timestamps and order IDs.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from config.constants import NEW_YORK_TZ, REG_RULE_4210_EFFECTIVE_DATE


def now_utc() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


def now_eastern() -> datetime:
    """Return current America/New_York time as a timezone-aware datetime."""
    return datetime.now(tz=NEW_YORK_TZ)


def today_eastern() -> date:
    """Return today's date in America/New_York."""
    return now_eastern().date()


def utc_compact_timestamp(dt: Optional[datetime] = None) -> str:
    """Return a compact UTC timestamp string suitable for client_order_id.

    Format: YYYYMMDDTHHMMSSffffff (microsecond precision).
    """
    moment = dt.astimezone(timezone.utc) if dt else now_utc()
    return moment.strftime("%Y%m%dT%H%M%S%f")


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime.

    Accepts trailing 'Z' as UTC. Naive timestamps are interpreted as UTC.
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def seconds_since(ts: datetime, *, reference: Optional[datetime] = None) -> float:
    """Return non-negative seconds elapsed between `ts` and `reference` (now)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ref = reference or now_utc()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = (ref - ts).total_seconds()
    return max(0.0, delta)


def is_after_rule_4210_effective(reference: Optional[date] = None) -> bool:
    """Return True if `reference` is on/after the FINRA Rule 4210 effective date."""
    ref = reference or today_eastern()
    return ref >= REG_RULE_4210_EFFECTIVE_DATE
