"""Wrapper for the Alpaca TradingStream that handles reconnects and routing.

When the websocket disconnects, the wrapper:
- pauses signal generation via a shared `connection_healthy` flag
- attempts reconnection with bounded exponential backoff
- triggers the orchestrator's reconciliation routine on reconnect

The wrapper also maintains an in-memory map of recently observed broker
order ids keyed by client_order_id, used by the order service for
post-failure reconciliation.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Coroutine, Optional

from alpaca.data.live.stock import StockDataStream
from alpaca.trading.stream import TradingStream

from config.constants import DEFAULT_WS_RECONNECT_MAX_DELAY, LOGGER_STREAM


OrderEventHandler = Callable[[Any], Awaitable[None]]
QuoteEventHandler = Callable[[Any], Awaitable[None]]
FatalStreamHandler = Callable[[BaseException], None]


def is_connection_limit_exceeded(exc: BaseException) -> bool:
    """Detect Alpaca websocket auth/session-limit failures across wrappers."""

    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        cid = id(cur)
        if cid in seen:
            continue
        seen.add(cid)
        text = f"{type(cur).__name__}: {cur}".lower()
        if "connection limit exceeded" in text:
            return True
        if "429" in text and "connection" in text and "limit" in text:
            return True
        cause = getattr(cur, "__cause__", None)
        ctx = getattr(cur, "__context__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        if isinstance(ctx, BaseException):
            stack.append(ctx)
    return False


class StreamHealth:
    """Mutable health flag shared with the orchestrator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._trading_ok: bool = False
        self._market_ok: bool = False
        self._last_quote_ts: Optional[datetime] = None
        self._last_order_event_ts: Optional[datetime] = None
        self._reconnect_attempts: int = 0
        self._trading_connected_at: Optional[datetime] = None
        self._trading_disconnected_at: Optional[datetime] = None
        self._market_connected_at: Optional[datetime] = None
        self._market_disconnected_at: Optional[datetime] = None

    def increment_ws_reconnect_attempts(self) -> None:
        with self._lock:
            self._reconnect_attempts += 1

    def websocket_health_snapshot(
        self, *, stale_seconds_threshold: float
    ) -> tuple[str, float, int]:
        """Return `(status, seconds_since_last_msg, reconnect_count)`."""

        with self._lock:
            secs = self._secs_since_latest_message_locked()
            reconn = int(self._reconnect_attempts)
            ok_both = self._trading_ok and self._market_ok
        status = "connected"
        if not ok_both:
            status = "disconnected"
        elif secs > float(stale_seconds_threshold):
            status = "stale"
        return status, secs, reconn

    def _secs_since_latest_message_locked(self) -> float:
        now = datetime.now(timezone.utc)
        candidates: list[datetime] = []
        if self._last_quote_ts is not None:
            candidates.append(self._last_quote_ts)
        if self._last_order_event_ts is not None:
            candidates.append(self._last_order_event_ts)
        if not candidates:
            return float("inf")
        latest = max(candidates)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        else:
            latest = latest.astimezone(timezone.utc)
        return max(0.0, (now - latest).total_seconds())

    @property
    def reconnect_attempt_count(self) -> int:
        with self._lock:
            return int(self._reconnect_attempts)

    def set_trading_ok(self, ok: bool) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            prev = self._trading_ok
            if ok and not prev:
                self._trading_connected_at = now
            if not ok and prev:
                self._trading_disconnected_at = now
            self._trading_ok = ok

    def set_market_ok(self, ok: bool) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            prev = self._market_ok
            if ok and not prev:
                self._market_connected_at = now
            if not ok and prev:
                self._market_disconnected_at = now
            self._market_ok = ok

    def mark_quote_event(self) -> None:
        with self._lock:
            self._last_quote_ts = datetime.now(timezone.utc)

    def mark_order_event(self) -> None:
        with self._lock:
            self._last_order_event_ts = datetime.now(timezone.utc)

    @property
    def trading_ok(self) -> bool:
        with self._lock:
            return self._trading_ok

    @property
    def market_ok(self) -> bool:
        with self._lock:
            return self._market_ok

    @property
    def all_ok(self) -> bool:
        with self._lock:
            return self._trading_ok and self._market_ok

    @property
    def last_quote_event(self) -> Optional[datetime]:
        with self._lock:
            return self._last_quote_ts


class TradingStreamRunner:
    """Run the TradingStream + StockDataStream in supervised background tasks."""

    def __init__(
        self,
        trading_stream: TradingStream,
        market_stream: StockDataStream,
        symbols: list[str],
        *,
        on_trade_update: OrderEventHandler,
        on_quote: QuoteEventHandler,
        health: StreamHealth,
        on_fatal_error: FatalStreamHandler | None = None,
    ) -> None:
        self._trading_stream = trading_stream
        self._market_stream = market_stream
        self._symbols = [s.upper() for s in symbols]
        self._subscribed_syms: set[str] = set(self._symbols)
        self._on_trade_update = on_trade_update
        self._on_quote = on_quote
        self._health = health
        self._on_fatal_error = on_fatal_error
        self._fatal_error: BaseException | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._stop_event = asyncio.Event()
        self._log = logging.getLogger(LOGGER_STREAM)

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    async def start(self) -> None:
        """Subscribe to streams and run them under supervision."""
        try:
            self._trading_stream.subscribe_trade_updates(self._wrap_trade_handler)
        except Exception as exc:  # noqa: BLE001
            self._log.error("Failed to subscribe trade updates: %s", exc)
            raise

        try:
            self._market_stream.subscribe_quotes(self._wrap_quote_handler, *self._symbols)
        except Exception as exc:  # noqa: BLE001
            self._log.error("Failed to subscribe quotes for %s: %s", self._symbols, exc)
            raise

        self._tasks.append(asyncio.create_task(self._supervise(self._run_trading, "trading")))
        self._tasks.append(asyncio.create_task(self._supervise(self._run_market, "market")))
        self._subscribed_syms = set(self._symbols)
        self._log.info("Trading + market streams started for %s", self._symbols)

    def subscribe_quote_symbols(self, symbols: list[str]) -> list[str]:
        """Subscribe to additional NBBO feeds (additive with alpaca-py StockDataStream).

        Returns symbols successfully targeted in this call that were not previously
        subscribed on this runner instance.
        """
        new_syms = [s.upper() for s in symbols if s.strip() and s.upper() not in self._subscribed_syms]
        if not new_syms:
            return []
        try:
            self._market_stream.subscribe_quotes(self._wrap_quote_handler, *new_syms)
        except Exception as exc:  # noqa: BLE001
            self._log.error("Failed extra quote subscriptions for %s: %s", new_syms, exc)
            raise
        self._subscribed_syms.update(new_syms)
        self._log.info("Extra NBBO subscriptions online: %s", new_syms)
        return new_syms

    async def stop(self) -> None:
        self._stop_event.set()
        for stream, label in (
            (self._trading_stream, "trading"),
            (self._market_stream, "market"),
        ):
            await self._close_stream(stream, label)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        self._health.set_trading_ok(False)
        self._health.set_market_ok(False)

    async def _close_stream(self, stream: Any, label: str) -> None:
        self._log.warning("event=websocket_shutdown_initiated stream=%s", label)
        for method_name in ("stop_ws", "stop", "close"):
            close_fn = getattr(stream, method_name, None)
            if close_fn is None:
                continue
            try:
                res = close_fn()
                if asyncio.iscoroutine(res):
                    await res
                self._log.info(
                    "event=websocket_shutdown_completed stream=%s method=%s",
                    label,
                    method_name,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "event=websocket_shutdown_error stream=%s method=%s err=%s",
                    label,
                    method_name,
                    exc,
                )
        self._log.warning("event=websocket_shutdown_noop stream=%s reason=no_close_method", label)

    # ---- handlers ---------------------------------------------------------

    async def _wrap_trade_handler(self, event: Any) -> None:
        self._health.mark_order_event()
        try:
            await self._on_trade_update(event)
        except Exception as exc:  # noqa: BLE001
            self._log.exception("Error handling trade update: %s", exc)

    async def _wrap_quote_handler(self, event: Any) -> None:
        self._health.mark_quote_event()
        try:
            await self._on_quote(event)
        except Exception as exc:  # noqa: BLE001
            self._log.exception("Error handling quote: %s", exc)

    # ---- supervision ------------------------------------------------------

    async def _run_trading(self) -> None:
        self._health.set_trading_ok(True)
        try:
            await self._trading_stream._run_forever()
        finally:
            self._health.set_trading_ok(False)

    async def _run_market(self) -> None:
        self._health.set_market_ok(True)
        try:
            await self._market_stream._run_forever()
        finally:
            self._health.set_market_ok(False)

    async def _supervise(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, None]],
        label: str,
    ) -> None:
        """Restart `coro_factory()` with bounded exponential backoff on failure."""
        backoff = 1.0
        first_iteration = True
        while not self._stop_event.is_set():
            if not first_iteration:
                self._health.increment_ws_reconnect_attempts()
            first_iteration = False
            try:
                await coro_factory()
                if self._stop_event.is_set():
                    return
                self._log.warning("%s stream coroutine returned; reconnecting.", label)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if is_connection_limit_exceeded(exc):
                    self._fatal_error = exc
                    self._stop_event.set()
                    self._health.set_trading_ok(False)
                    self._health.set_market_ok(False)
                    self._log.critical(
                        "event=alpaca_connection_limit_exceeded stream=%s err=%s",
                        label,
                        exc,
                    )
                    if self._on_fatal_error is not None:
                        self._on_fatal_error(exc)
                    return
                self._log.error(
                    "%s stream crashed: %s. Reconnecting in %.2fs.", label, exc, backoff
                )
            await asyncio.sleep(backoff)
            backoff = min(DEFAULT_WS_RECONNECT_MAX_DELAY, backoff * 2)
