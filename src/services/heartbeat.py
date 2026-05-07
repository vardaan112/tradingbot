"""Periodic heartbeat task.

Logs a single structured INFO line every HEARTBEAT_INTERVAL_SECONDS with the
operationally interesting state of the bot. Designed to be greppable on a
VPS via `tail -f logs/heartbeat.log`.

When ``HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS`` is > 0, also emits a
Markdown tearsheet snapshot on that slower cadence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from config.constants import LOGGER_APP, LOGGER_HEARTBEAT
from config.settings import Settings
from core.market_clock import MarketClock
from core.market_data import QuoteCache
from core.trading_stream import StreamHealth
from risk.compliance import ComplianceAdapter
from risk.killswitch import KillSwitch
from utils.local_health import log_local_resource_check
from utils.stream_alerts import evaluate_stream_websocket_notifications
from utils.tearsheet import format_tearsheet_markdown_table, get_tearsheet_summary


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
        tearsheet_orders_path: Optional[Path] = None,
        tearsheet_summary_fn: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._quotes = quote_cache
        self._health = stream_health
        self._kill = kill_switch
        self._compliance = compliance
        self._snapshot_provider = snapshot_provider
        self._orders_log = tearsheet_orders_path
        self._tearsheet_summary_fn = tearsheet_summary_fn
        self._log = logging.getLogger(LOGGER_HEARTBEAT)
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_md_emit_mono = float("-inf")

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

        ts_summary: dict[str, Any] | None = None
        tear_fragment = ""
        try:
            if self._tearsheet_summary_fn is not None:
                ts_summary = self._tearsheet_summary_fn()
            elif self._orders_log is not None:
                ts_summary = get_tearsheet_summary(self._orders_log)

            if isinstance(ts_summary, dict) and ts_summary.get("ok"):
                pf = ts_summary.get("profit_factor")
                if pf is None:
                    pf_txt = "n_a"
                elif isinstance(pf, float) and pf == float("inf"):
                    pf_txt = "inf"
                else:
                    pf_txt = f"{float(pf):.4f}"
                shr = ts_summary.get("sharpe_ratio")
                shr_txt = "n_a" if shr is None else f"{float(shr):.4f}"
                mdd = ts_summary.get("max_drawdown")
                mdd_txt = "n_a" if mdd is None else f"{float(mdd):.4f}"
                wr = ts_summary.get("win_rate_pct")
                wr_txt = "n_a" if wr is None else f"{float(wr):.2f}"
                tear_fragment = (
                    f"tearsheet_closed={int(ts_summary.get('closed_trades', 0))} "
                    f"tearsheet_net={float(ts_summary.get('net_pnl', 0.0)):.4f} "
                    f"tearsheet_pf={pf_txt} tearsheet_sharpe={shr_txt} "
                    f"tearsheet_mdd={mdd_txt} tearsheet_win_rate_pct={wr_txt}"
                )

                hb_md_s = float(self._settings.HEARTBEAT_TEARSHEET_MARKDOWN_INTERVAL_SECONDS)
                if hb_md_s > 0:
                    nowm = time.monotonic()
                    if nowm - self._last_md_emit_mono >= hb_md_s:
                        table = format_tearsheet_markdown_table(ts_summary)
                        if table.strip():
                            self._last_md_emit_mono = nowm
                            self._log.info(
                                "event=heartbeat_tearsheet_md session=%s", session_state,
                            )
                            self._log.info("%s", table)
            elif ts_summary is not None:
                tear_fragment = (
                    "tearsheet_closed=n_a tearsheet_net=n_a tearsheet_pf=n_a "
                    "tearsheet_sharpe=n_a tearsheet_mdd=n_a tearsheet_win_rate_pct=n_a"
                )
            # else: leave tear_fragment="" — no telemetry path configured
        except Exception:
            tear_fragment = (
                "tearsheet_closed=n_a tearsheet_net=n_a tearsheet_pf=n_a "
                "tearsheet_sharpe=n_a tearsheet_mdd=n_a tearsheet_win_rate_pct=n_a"
            )

        tear_suffix = f" {tear_fragment.strip()}" if tear_fragment.strip() else ""

        self._log.info(
            "heartbeat session=%s ws_trading=%s ws_market=%s "
            "equity=%.2f buying_power=%.2f open_positions=%d open_orders=%d "
            "latest_quote_age=%s killswitch=%s reg_mode=%s feed=%s%s",
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
            tear_suffix,
        )
        try:
            chaos_log = logging.getLogger(LOGGER_APP)
            evaluate_stream_websocket_notifications(
                settings=self._settings,
                stream_health=self._health,
                log=chaos_log,
            )
            log_local_resource_check(self._settings, log=chaos_log)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("chaos/health heartbeat hooks failed: %s", exc)
