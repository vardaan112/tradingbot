"""Periodic heartbeat task.

Logs a single structured INFO line every HEARTBEAT_INTERVAL_SECONDS with the
operationally interesting state of the bot. Designed to be greppable on a
VPS via `tail -f logs/heartbeat.log`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from config.constants import LOGGER_HEARTBEAT
from config.settings import Settings
from core.market_clock import MarketClock
from core.market_data import QuoteCache
from core.trading_stream import StreamHealth
from risk.compliance import ComplianceAdapter
from risk.killswitch import KillSwitch


class HeartbeatService:
    """Async task emitting a heartbeat line on a fixed interval."""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: MarketClock,
        quote_cache: QuoteCache,
        stream_health: StreamHealth,
        kill_switch: KillSwitch,
        compliance: ComplianceAdapter,
        snapshot_provider: Callable[[], dict],
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._quotes = quote_cache
        self._health = stream_health
        self._kill = kill_switch
        self._compliance = compliance
        self._snapshot_provider = snapshot_provider
        self._log = logging.getLogger(LOGGER_HEARTBEAT)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._emit_once()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("heartbeat tick failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._settings.HEARTBEAT_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass

    async def _emit_once(self) -> None:
        snapshot = self._snapshot_provider()
        try:
            session = self._clock.get_session()
            session_state = "open" if session.is_open else "closed"
        except Exception as exc:  # noqa: BLE001
            session_state = f"unknown({exc})"

        ks_record = self._kill.latch_record()
        latest_age = self._quotes.latest_age_seconds()
        latest_age_str = f"{latest_age:.1f}s" if latest_age is not None else "n/a"

        self._log.info(
            "heartbeat session=%s ws_trading=%s ws_market=%s "
            "equity=%.2f buying_power=%.2f open_positions=%d open_orders=%d "
            "latest_quote_age=%s killswitch=%s reg_mode=%s feed=%s",
            session_state,
            self._health.trading_ok,
            self._health.market_ok,
            snapshot.get("equity", 0.0),
            snapshot.get("buying_power", 0.0),
            snapshot.get("open_positions", 0),
            snapshot.get("open_orders", 0),
            latest_age_str,
            "LATCHED" if ks_record.latched else "ok",
            self._compliance.effective_mode(),
            self._quotes.feed,
        )
