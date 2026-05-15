"""Runtime labels for SQLite ``source`` on completed trades and execution events.

Phase 0 separates broker-attributed rows (``live``, ``paper``) from simulated
and research rows (``dry_run``, ``simulation``, ``replay``, ``shadow``).
"""

from __future__ import annotations

from typing import Final

from config.settings import Settings

# Rows eligible for default ML / Kelly training (broker path only).
BROKER_ELIGIBLE_SOURCES: Final[frozenset[str]] = frozenset({"live", "paper"})


def runtime_trade_source(settings: Settings) -> str:
    """Resolve ``source`` for runtime persistence from current Settings.

    - DRY_RUN=true -> ``dry_run`` (no broker order POST; simulated fills)
    - else ALPACA_ENV=paper -> ``paper``
    - else -> ``live``

    ``simulation`` is used by ``scripts/replay_simulator.py``; ``replay`` and
    ``shadow`` are written by the historical replay engine and shadow portfolios.
    This function never returns those — only the broker-path labels above.
    """

    if settings.DRY_RUN:
        return "dry_run"
    if settings.ALPACA_ENV == "paper":
        return "paper"
    return "live"


def sql_broker_eligible_sources_clause(*, enabled: bool = True) -> str:
    """SQL fragment restricting ``completed_trades.source`` to live/paper."""

    if not enabled:
        return ""
    return " AND COALESCE(source, 'live') IN ('live', 'paper') "


__all__ = [
    "BROKER_ELIGIBLE_SOURCES",
    "runtime_trade_source",
    "sql_broker_eligible_sources_clause",
]
