"""Order service: limit-only entries, idempotent COIDs, reconciliation, and
emergency-flatten with marketable limit IOC orders.

Production rules enforced here:
- No market orders for entries.
- Every order placement uses an explicit, idempotent client_order_id.
- Order placement POSTs are NOT auto-retried; on ambiguous failure the
  service reconciles by client_order_id before any retry.
- Cancels are bounded and rate-aware.
- Emergency flatten uses marketable limit IOC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

from config.constants import LOGGER_ORDERS
from config.settings import Settings
from utils.ids import generate_client_order_id, short_uuid
from utils.price_utils import round_to_tick, spread_pct

from .database import Database

from .exceptions import (
    BrokerConnectionError,
    NonRetryableBrokerError,
    OrderPlacementError,
    OrderRejectedError,
)
from .market_data import Quote, QuoteCache
from .retries import retry_call
from .state_store import OpenOrderEntry, StateStore


@dataclass
class WorkingOrder:
    """Bot-side representation of a live or recently submitted order."""

    client_order_id: str
    symbol: str
    side: str
    qty: float
    submitted_qty: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    limit_price: Optional[float] = None
    status: str = "new"
    broker_order_id: Optional[str] = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_at: Optional[datetime] = None
    strategy: str = ""

    def is_terminal(self) -> bool:
        terminal = {"filled", "canceled", "expired", "rejected", "done_for_day"}
        return self.status.lower() in terminal


class OrderService:
    """Submits, cancels, and reconciles orders. Owns per-symbol working state."""

    def __init__(
        self,
        trading: TradingClient,
        settings: Settings,
        state: StateStore,
        quote_cache: QuoteCache,
        *,
        strategy_name: str = "rsi_meanrev",
        database: Optional[Database] = None,
        simulated_fill_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._client = trading
        self._settings = settings
        self._state = state
        self._quotes = quote_cache
        self._strategy_name = strategy_name
        self._database = database
        self._simulated_fill_sink = simulated_fill_sink
        self._log = logging.getLogger(LOGGER_ORDERS)

        self._lock = threading.RLock()
        self._working: dict[str, WorkingOrder] = {}  # by client_order_id
        self._by_symbol: dict[str, str] = {}  # symbol -> client_order_id

    # ------------------------------------------------------------------ utils

    def _retry(self, op: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return retry_call(
            fn,
            *args,
            max_attempts=self._settings.RETRY_MAX_ATTEMPTS,
            base_delay=self._settings.RETRY_BASE_DELAY_SECONDS,
            max_delay=self._settings.RETRY_MAX_DELAY_SECONDS,
            op_name=op,
            logger=self._log,
            **kwargs,
        )

    def _coid(self, symbol: str, side: str) -> str:
        return generate_client_order_id(self._strategy_name, symbol, side)

    def has_open_for_symbol(self, symbol: str) -> bool:
        with self._lock:
            coid = self._by_symbol.get(symbol.upper())
            if coid is None:
                return False
            wo = self._working.get(coid)
            if wo is None:
                self._by_symbol.pop(symbol.upper(), None)
                return False
            return not wo.is_terminal()

    def get_working(self, client_order_id: str) -> Optional[WorkingOrder]:
        with self._lock:
            return self._working.get(client_order_id)

    def working_orders_snapshot(self) -> list[WorkingOrder]:
        with self._lock:
            return [w for w in self._working.values() if not w.is_terminal()]

    def _notify_simulated_fill(
        self,
        *,
        wo: WorkingOrder,
        symbol: str,
        side: str,
        qty: float | int,
        limit_price: float,
        quote: Optional[Quote],
        intent_reason: str,
    ) -> None:
        mid_px: Optional[float] = None
        if quote is not None and quote.bid > 0 and quote.ask > quote.bid:
            mid_px = (quote.bid + quote.ask) / 2.0
        sim_fill = float(mid_px if mid_px is not None else limit_price)
        ts = wo.submitted_at.isoformat()
        discord_notified = False
        pl: dict[str, Any] = {
            "dry_run": True,
            "symbol": symbol.upper(),
            "side": side.lower(),
            "qty": int(qty),
            "limit_price": float(limit_price),
            "simulated_fill_price": sim_fill,
            "strategy": wo.strategy or self._strategy_name,
            "reason": intent_reason,
            "timestamp": ts,
        }
        if self._simulated_fill_sink is not None:
            try:
                self._simulated_fill_sink(pl)
                discord_notified = True
            except Exception as exc:  # noqa: BLE001
                self._log.warning("event=simulated_fill_sink_failed err=%s", exc)
        self._log.info(
            "event=simulated_fill dry_run=true symbol=%s side=%s qty=%s limit_price=%.4f "
            "simulated_fill_price=%.4f strategy=%s reason=%s discord_notified=%s timestamp=%s",
            symbol,
            side.lower(),
            int(qty),
            limit_price,
            sim_fill,
            wo.strategy or self._strategy_name,
            intent_reason,
            str(discord_notified).lower(),
            ts,
            extra={
                "symbol": symbol,
                "client_order_id": wo.client_order_id,
                "strategy": wo.strategy or self._strategy_name,
            },
        )

    # ------------------------------------------------------------------ entries

    def submit_limit_entry(
        self,
        symbol: str,
        qty: int,
        side: str,
        *,
        quote: Quote,
        spread_fraction: float = 0.25,
        intent_reason: str = "limit_entry",
    ) -> Optional[WorkingOrder]:
        """Submit a conservative limit entry.

        Buy: limit at bid + spread_fraction * (ask - bid).
        Sell: limit at ask - spread_fraction * (ask - bid).
        """
        symbol = symbol.upper()
        side_l = side.lower()

        if qty < 1:
            self._log.warning("Refusing to submit qty<1 for %s", symbol)
            return None

        if self.has_open_for_symbol(symbol):
            self._log.warning("Open order already exists for %s; not submitting another.", symbol)
            return None

        spread = quote.ask - quote.bid
        if spread <= 0:
            self._log.warning("Cannot construct limit for %s: invalid spread.", symbol)
            return None

        if side_l == "buy":
            raw_price = quote.bid + spread_fraction * spread
            limit_price = round_to_tick(raw_price, mode="down")
            order_side = OrderSide.BUY
        elif side_l == "sell":
            raw_price = quote.ask - spread_fraction * spread
            limit_price = round_to_tick(raw_price, mode="up")
            order_side = OrderSide.SELL
        else:
            raise ValueError(f"unknown side {side!r}")

        coid = self._coid(symbol, side_l)
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            client_order_id=coid,
        )

        wo = WorkingOrder(
            client_order_id=coid,
            symbol=symbol,
            side=side_l,
            qty=float(qty),
            submitted_qty=float(qty),
            limit_price=limit_price,
            status="pending_new",
            strategy=self._strategy_name,
        )
        with self._lock:
            self._working[coid] = wo
            self._by_symbol[symbol] = coid

        if self._settings.DRY_RUN or not self._settings.LIVE_TRADING_ENABLED:
            self._log.info(
                "event=dry_run_order_blocked symbol=%s side=%s qty=%d limit_price=%.4f coid=%s strategy=%s",
                symbol,
                side_l,
                qty,
                limit_price,
                coid,
                self._strategy_name,
                extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
            )
            self._notify_simulated_fill(
                wo=wo,
                symbol=symbol,
                side=side_l,
                qty=qty,
                limit_price=limit_price,
                quote=quote,
                intent_reason=intent_reason,
            )
            self._log.info(
                "[DRY_RUN] would submit %s %s qty=%d limit=%.4f coid=%s",
                side_l, symbol, qty, limit_price, coid,
                extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
            )
            wo.status = "dry_run"
            self._persist_index()
            return wo

        try:
            placed = self._client.submit_order(request)
        except Exception as exc:  # noqa: BLE001
            return self._reconcile_after_placement_failure(wo, exc)

        broker_id = getattr(placed, "id", None) or getattr(placed, "order_id", None)
        wo.broker_order_id = str(broker_id) if broker_id is not None else None
        wo.status = str(getattr(placed, "status", "new"))
        wo.last_event_at = datetime.now(timezone.utc)
        self._log.info(
            "Submitted LIMIT %s %s qty=%d limit=%.4f coid=%s broker_id=%s",
            side_l, symbol, qty, limit_price, coid, wo.broker_order_id,
            extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
        )
        self._persist_index()
        return wo

    def _chase_spread_bps(self, bid: float, ask: float) -> float:
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 9999.0
        return spread_pct(bid, ask) * 10000.0

    def _persist_chase_db(
        self,
        event_type: str,
        *,
        symbol: str,
        side: str,
        attempt: int,
        limit_price: float,
        bid: float,
        ask: float,
        coid: str,
        reason: str,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        if self._database is None:
            return
        sp = self._chase_spread_bps(bid, ask)
        self._database.record_execution_event(
            event_type=event_type,
            symbol=symbol,
            side=side,
            client_order_id=None if coid == "n_a" else coid,
            order_id=order_id,
            status=status,
            price=limit_price,
            quantity=None,
            metadata={
                "attempt": attempt,
                "best_bid": bid,
                "best_ask": ask,
                "spread_bps": sp,
                "reason": reason,
            },
        )

    def _log_chase(
        self,
        event_type: str,
        *,
        symbol: str,
        side: str,
        attempt: int,
        limit_price: float,
        bid: float,
        ask: float,
        coid: str,
        reason: str,
        order_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        sp = self._chase_spread_bps(bid, ask)
        self._log.info(
            "event=%s symbol=%s side=%s attempt=%d limit_price=%.4f best_bid=%.4f "
            "best_ask=%.4f spread_bps=%.4f client_order_id=%s reason=%s timestamp=%s",
            event_type,
            symbol,
            side,
            attempt,
            limit_price,
            bid,
            ask,
            sp,
            coid,
            reason,
            ts,
            extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
        )
        self._persist_chase_db(
            event_type,
            symbol=symbol,
            side=side,
            attempt=attempt,
            limit_price=limit_price,
            bid=bid,
            ask=ask,
            coid=coid,
            reason=reason,
            order_id=order_id,
            status=status,
        )

    def _release_symbol_lock(self, wo: WorkingOrder) -> None:
        with self._lock:
            self._working.pop(wo.client_order_id, None)
            if self._by_symbol.get(wo.symbol) == wo.client_order_id:
                self._by_symbol.pop(wo.symbol, None)
        self._persist_index()

    async def submit_buy_passive_joiner_async(
        self,
        symbol: str,
        qty: int,
        *,
        quote_refresher: Callable[[], Optional[Quote]],
    ) -> Optional[WorkingOrder]:
        """Chase best bid with cancel-replace; limit BUY only."""
        symbol = symbol.upper()
        if qty < 1:
            return None
        if not self._settings.PASSIVE_JOINER_ENABLED:
            q0 = quote_refresher()
            if q0 is None:
                return None
            return self.submit_limit_entry(symbol, qty, "buy", quote=q0)

        max_a = int(self._settings.PASSIVE_JOINER_MAX_ATTEMPTS)
        timeout_s = float(self._settings.PASSIVE_JOINER_TIMEOUT_SECONDS)
        stale_lim = float(self._settings.QUOTE_STALENESS_SECONDS)
        require_fresh = bool(self._settings.PASSIVE_JOINER_REQUIRE_FRESH_QUOTE)

        for attempt in range(1, max_a + 1):
            quote = quote_refresher()
            if quote is None:
                self._log_chase(
                    "order_chase_error",
                    symbol=symbol,
                    side="buy",
                    attempt=attempt,
                    limit_price=0.0,
                    bid=0.0,
                    ask=0.0,
                    coid="n_a",
                    reason="no_quote",
                )
                return None
            if require_fresh and not quote.is_fresh(stale_lim):
                self._log_chase(
                    "order_chase_giveup",
                    symbol=symbol,
                    side="buy",
                    attempt=attempt,
                    limit_price=0.0,
                    bid=quote.bid,
                    ask=quote.ask,
                    coid="n_a",
                    reason="stale_quote",
                )
                return None
            if quote.bid <= 0 or quote.ask <= quote.bid:
                self._log_chase(
                    "order_chase_error",
                    symbol=symbol,
                    side="buy",
                    attempt=attempt,
                    limit_price=0.0,
                    bid=quote.bid,
                    ask=quote.ask,
                    coid="n_a",
                    reason="invalid_quote",
                )
                return None
            if spread_pct(quote.bid, quote.ask) > float(self._settings.SPREAD_FILTER_PCT):
                self._log_chase(
                    "order_chase_giveup",
                    symbol=symbol,
                    side="buy",
                    attempt=attempt,
                    limit_price=0.0,
                    bid=quote.bid,
                    ask=quote.ask,
                    coid="n_a",
                    reason="spread_filter",
                )
                return None

            limit_price = round_to_tick(quote.bid, mode="down")
            coid = generate_client_order_id(
                self._strategy_name,
                symbol,
                "buy",
                short_id=f"pj{attempt}{short_uuid(6)}",
            )

            if self.has_open_for_symbol(symbol):
                self._log.warning("Chase blocked: existing open for %s", symbol)
                return None

            wo = WorkingOrder(
                client_order_id=coid,
                symbol=symbol,
                side="buy",
                qty=float(qty),
                submitted_qty=float(qty),
                limit_price=limit_price,
                status="pending_new",
                strategy=self._strategy_name,
            )
            with self._lock:
                self._working[coid] = wo
                self._by_symbol[symbol] = coid

            self._log_chase(
                "order_chase_attempt",
                symbol=symbol,
                side="buy",
                attempt=attempt,
                limit_price=limit_price,
                bid=quote.bid,
                ask=quote.ask,
                coid=coid,
                reason="submit",
            )

            if self._settings.DRY_RUN or not self._settings.LIVE_TRADING_ENABLED:
                self._log.info(
                    "event=dry_run_order_blocked symbol=%s side=buy qty=%s limit_price=%.4f coid=%s strategy=%s",
                    symbol,
                    qty,
                    limit_price,
                    coid,
                    self._strategy_name,
                    extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
                )
                self._notify_simulated_fill(
                    wo=wo,
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    limit_price=limit_price,
                    quote=quote,
                    intent_reason=f"passive_joiner_buy_attempt_{attempt}",
                )
                wo.status = "dry_run"
                self._persist_index()
                return wo

            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                client_order_id=coid,
            )
            try:
                placed = await asyncio.to_thread(self._client.submit_order, request)
            except Exception as exc:  # noqa: BLE001
                self._log_chase(
                    "order_chase_error",
                    symbol=symbol,
                    side="buy",
                    attempt=attempt,
                    limit_price=limit_price,
                    bid=quote.bid,
                    ask=quote.ask,
                    coid=coid,
                    reason=f"submit_exc:{exc}",
                )
                self._release_symbol_lock(wo)
                continue

            broker_id = getattr(placed, "id", None) or getattr(placed, "order_id", None)
            wo.broker_order_id = str(broker_id) if broker_id is not None else None
            wo.status = str(getattr(placed, "status", "new"))
            wo.last_event_at = datetime.now(timezone.utc)
            self._persist_index()

            deadline = time.monotonic() + timeout_s
            filled = False
            while time.monotonic() < deadline:
                await asyncio.sleep(0.35)
                try:
                    ord_obj = await asyncio.to_thread(
                        self._client.get_order_by_client_id,
                        coid,
                    )
                except Exception:  # noqa: BLE001
                    continue
                fq = float(getattr(ord_obj, "filled_qty", 0) or 0)
                st = str(getattr(ord_obj, "status", "")).lower()
                if fq >= float(qty) - 1e-6 or st == "filled":
                    wo.filled_qty = fq
                    wo.status = st or "filled"
                    filled = True
                    self._log_chase(
                        "order_chase_filled",
                        symbol=symbol,
                        side="buy",
                        attempt=attempt,
                        limit_price=limit_price,
                        bid=quote.bid,
                        ask=quote.ask,
                        coid=coid,
                        reason="filled",
                        order_id=wo.broker_order_id,
                        status=wo.status,
                    )
                    return wo
                if st in {"canceled", "expired", "rejected"}:
                    break

            if filled:
                return wo

            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.cancel, coid)

            with contextlib.suppress(Exception):
                ord_obj = await asyncio.to_thread(self._client.get_order_by_client_id, coid)
                fq = float(getattr(ord_obj, "filled_qty", 0) or 0)
                st = str(getattr(ord_obj, "status", "")).lower()
                if fq >= float(qty) - 1e-6 or st == "filled":
                    wo.filled_qty = fq
                    wo.status = st or "filled"
                    self._log_chase(
                        "order_chase_filled",
                        symbol=symbol,
                        side="buy",
                        attempt=attempt,
                        limit_price=limit_price,
                        bid=quote.bid,
                        ask=quote.ask,
                        coid=coid,
                        reason="filled_during_cancel",
                        order_id=wo.broker_order_id,
                        status=wo.status,
                    )
                    return wo

            self._log_chase(
                "order_chase_replace",
                symbol=symbol,
                side="buy",
                attempt=attempt,
                limit_price=limit_price,
                bid=quote.bid,
                ask=quote.ask,
                coid=coid,
                reason="timeout_cancel",
            )
            self._release_symbol_lock(wo)

        self._log_chase(
            "order_chase_giveup",
            symbol=symbol,
            side="buy",
            attempt=max_a,
            limit_price=0.0,
            bid=0.0,
            ask=0.0,
            coid="n_a",
            reason="max_attempts",
        )
        return None

    def _reconcile_after_placement_failure(
        self, wo: WorkingOrder, exc: BaseException
    ) -> Optional[WorkingOrder]:
        """Check whether the order was actually accepted before retrying."""
        self._log.error(
            "Placement failure for coid=%s symbol=%s: %s. Reconciling...",
            wo.client_order_id, wo.symbol, exc,
            extra={"symbol": wo.symbol, "client_order_id": wo.client_order_id},
        )
        try:
            existing = self._client.get_order_by_client_id(wo.client_order_id)
        except Exception as get_exc:  # noqa: BLE001
            # If lookup itself fails, do NOT retry placement; surface as ambiguous.
            self._log.error(
                "Reconcile lookup failed for coid=%s: %s. Marking ambiguous.",
                wo.client_order_id, get_exc,
                extra={"symbol": wo.symbol, "client_order_id": wo.client_order_id},
            )
            with self._lock:
                wo.status = "ambiguous"
            raise OrderPlacementError(
                f"order placement ambiguous for {wo.client_order_id}: {exc}"
            ) from exc

        broker_id = getattr(existing, "id", None) or getattr(existing, "order_id", None)
        if broker_id is not None:
            wo.broker_order_id = str(broker_id)
            wo.status = str(getattr(existing, "status", "accepted"))
            wo.last_event_at = datetime.now(timezone.utc)
            self._log.info(
                "Reconciled coid=%s already accepted as broker_id=%s status=%s",
                wo.client_order_id, wo.broker_order_id, wo.status,
                extra={"symbol": wo.symbol, "client_order_id": wo.client_order_id},
            )
            self._persist_index()
            return wo

        # No order found server-side -> placement truly failed.
        with self._lock:
            self._working.pop(wo.client_order_id, None)
            if self._by_symbol.get(wo.symbol) == wo.client_order_id:
                self._by_symbol.pop(wo.symbol, None)
        self._persist_index()
        raise OrderPlacementError(
            f"order placement failed for {wo.client_order_id}: {exc}"
        ) from exc

    # ------------------------------------------------------------------ exits

    def submit_limit_exit(
        self,
        symbol: str,
        qty: int,
        side: str,
        *,
        quote: Quote,
        spread_fraction: float = 0.25,
    ) -> Optional[WorkingOrder]:
        """Mirror of submit_limit_entry intended for normal (non-emergency) exits."""
        return self.submit_limit_entry(
            symbol, qty, side, quote=quote, spread_fraction=spread_fraction
        )

    def submit_emergency_flatten(
        self,
        symbol: str,
        qty: int,
        side: str,
        *,
        quote: Quote,
        aggressiveness_pct: float,
    ) -> Optional[WorkingOrder]:
        """Marketable limit IOC for emergency flatten / stop breach."""
        symbol = symbol.upper()
        side_l = side.lower()
        if qty < 1:
            self._log.warning("Emergency flatten qty<1 for %s; skipping.", symbol)
            return None

        mid = (quote.bid + quote.ask) / 2.0
        offset = abs(aggressiveness_pct) * mid

        if side_l == "sell":
            raw = max(0.01, quote.bid - offset)
            limit_price = round_to_tick(raw, mode="down")
            order_side = OrderSide.SELL
        elif side_l == "buy":
            raw = quote.ask + offset
            limit_price = round_to_tick(raw, mode="up")
            order_side = OrderSide.BUY
        else:
            raise ValueError(f"unknown side {side!r}")

        coid = self._coid(symbol + "EMG", side_l)
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.IOC,
            limit_price=limit_price,
            client_order_id=coid,
        )

        wo = WorkingOrder(
            client_order_id=coid,
            symbol=symbol,
            side=side_l,
            qty=float(qty),
            submitted_qty=float(qty),
            limit_price=limit_price,
            status="pending_emergency",
            strategy=self._strategy_name + ":emg",
        )
        with self._lock:
            self._working[coid] = wo

        self._log.critical(
            "EMERGENCY FLATTEN %s %s qty=%d limit=%.4f coid=%s",
            side_l, symbol, qty, limit_price, coid,
            extra={"symbol": symbol, "client_order_id": coid, "strategy": wo.strategy},
        )

        if self._settings.DRY_RUN or not self._settings.LIVE_TRADING_ENABLED:
            self._log.info(
                "event=dry_run_order_blocked symbol=%s side=%s qty=%d limit_price=%.4f coid=%s strategy=%s_emg",
                symbol,
                side_l,
                qty,
                limit_price,
                coid,
                wo.strategy,
                extra={"symbol": symbol, "client_order_id": coid, "strategy": wo.strategy},
            )
            self._notify_simulated_fill(
                wo=wo,
                symbol=symbol,
                side=side_l,
                qty=qty,
                limit_price=limit_price,
                quote=quote,
                intent_reason="emergency_flatten",
            )
            wo.status = "dry_run"
            return wo

        try:
            placed = self._client.submit_order(request)
        except Exception as exc:  # noqa: BLE001
            return self._reconcile_after_placement_failure(wo, exc)

        broker_id = getattr(placed, "id", None) or getattr(placed, "order_id", None)
        wo.broker_order_id = str(broker_id) if broker_id is not None else None
        wo.status = str(getattr(placed, "status", "new"))
        wo.last_event_at = datetime.now(timezone.utc)
        return wo

    # ------------------------------------------------------------------ cancel

    def cancel(self, client_order_id: str) -> bool:
        with self._lock:
            wo = self._working.get(client_order_id)
        if wo is None:
            return False
        if wo.broker_order_id is None:
            self._log.info(
                "Cancel requested for coid=%s but no broker id; clearing local entry.",
                client_order_id,
                extra={"symbol": wo.symbol, "client_order_id": client_order_id},
            )
            with self._lock:
                self._working.pop(client_order_id, None)
                if self._by_symbol.get(wo.symbol) == client_order_id:
                    self._by_symbol.pop(wo.symbol, None)
            return True
        try:
            self._retry(
                f"cancel_order[{wo.symbol}]",
                self._client.cancel_order_by_id,
                wo.broker_order_id,
            )
            self._log.info(
                "Cancel submitted for coid=%s broker_id=%s",
                client_order_id, wo.broker_order_id,
                extra={"symbol": wo.symbol, "client_order_id": client_order_id},
            )
            return True
        except (NonRetryableBrokerError, BrokerConnectionError, OrderRejectedError) as exc:
            self._log.warning("Cancel failed for coid=%s: %s", client_order_id, exc)
            return False

    def cancel_all_open(self) -> None:
        with self._lock:
            ids = [c for c, w in self._working.items() if not w.is_terminal()]
        for coid in ids:
            self.cancel(coid)
        try:
            self._retry("cancel_orders_all", self._client.cancel_orders)
        except Exception as exc:  # noqa: BLE001
            self._log.error("Bulk cancel call failed: %s", exc)

    def cancel_stale(self, timeout_seconds: float) -> None:
        now = datetime.now(timezone.utc)
        stale_ids: list[str] = []
        with self._lock:
            for coid, wo in self._working.items():
                if wo.is_terminal():
                    continue
                if wo.status.lower() == "filled":
                    continue
                age = (now - wo.submitted_at).total_seconds()
                if age > timeout_seconds:
                    stale_ids.append(coid)
        for coid in stale_ids:
            self._log.info("Cancelling stale order coid=%s", coid)
            self.cancel(coid)

    # ------------------------------------------------------------------ events

    async def handle_trade_update(self, event: Any) -> None:
        """Process a TradingStream trade update event."""
        order = getattr(event, "order", None)
        if order is None:
            return

        coid = getattr(order, "client_order_id", None)
        if not coid:
            return

        status = str(getattr(order, "status", ""))
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        filled_avg = float(getattr(order, "filled_avg_price", 0) or 0)
        broker_id = getattr(order, "id", None) or getattr(order, "order_id", None)

        with self._lock:
            wo = self._working.get(coid)
            if wo is None:
                wo = WorkingOrder(
                    client_order_id=coid,
                    symbol=str(getattr(order, "symbol", "")).upper(),
                    side=str(getattr(order, "side", "")).lower(),
                    qty=float(getattr(order, "qty", 0) or 0),
                    submitted_qty=float(getattr(order, "qty", 0) or 0),
                    strategy=self._strategy_name,
                )
                self._working[coid] = wo
                if wo.symbol:
                    self._by_symbol.setdefault(wo.symbol, coid)
            wo.status = status or wo.status
            wo.filled_qty = max(wo.filled_qty, filled_qty)
            if filled_avg > 0:
                wo.avg_fill_price = filled_avg
            if broker_id is not None:
                wo.broker_order_id = str(broker_id)
            wo.last_event_at = datetime.now(timezone.utc)
            symbol = wo.symbol
            wo_side_raw = wo.side or str(getattr(order, "side", "") or "").lower()
            wo_side = str(wo_side_raw).lower()
            terminal = wo.is_terminal()

        self._log.info(
            "Trade update coid=%s symbol=%s side=%s status=%s filled=%.4f avg=%.4f",
            coid, symbol, wo_side, status, filled_qty, filled_avg,
            extra={"symbol": symbol, "client_order_id": coid, "strategy": self._strategy_name},
        )

        if terminal:
            with self._lock:
                if self._by_symbol.get(symbol) == coid:
                    self._by_symbol.pop(symbol, None)

        self._persist_index()

    # ------------------------------------------------------------------ recon

    def reconcile_open_orders_from_broker(self) -> None:
        """Pull open orders from REST and merge into the working set."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
            open_orders = self._retry(
                "list_open_orders", self._client.get_orders, filter=request
            )
        except (BrokerConnectionError, NonRetryableBrokerError) as exc:
            self._log.error("Failed to reconcile open orders: %s", exc)
            return

        with self._lock:
            current_by_coid = dict(self._working)
            self._by_symbol.clear()

            for o in open_orders or []:
                coid = getattr(o, "client_order_id", None)
                if not coid:
                    continue
                symbol = str(getattr(o, "symbol", "")).upper()
                broker_id = getattr(o, "id", None) or getattr(o, "order_id", None)
                wo = current_by_coid.get(coid)
                if wo is None:
                    wo = WorkingOrder(
                        client_order_id=coid,
                        symbol=symbol,
                        side=str(getattr(o, "side", "")).lower(),
                        qty=float(getattr(o, "qty", 0) or 0),
                        submitted_qty=float(getattr(o, "qty", 0) or 0),
                        broker_order_id=str(broker_id) if broker_id else None,
                        strategy=self._strategy_name,
                    )
                    self._working[coid] = wo
                else:
                    if broker_id is not None:
                        wo.broker_order_id = str(broker_id)
                wo.status = str(getattr(o, "status", "open"))
                if symbol:
                    self._by_symbol[symbol] = coid

            for coid, wo in list(self._working.items()):
                if wo.is_terminal():
                    self._working.pop(coid, None)

        self._persist_index()

    def _persist_index(self) -> None:
        with self._lock:
            entries: dict[str, OpenOrderEntry] = {}
            for symbol, coid in self._by_symbol.items():
                wo = self._working.get(coid)
                if wo is None or wo.is_terminal():
                    continue
                entries[symbol] = OpenOrderEntry(
                    symbol=symbol,
                    client_order_id=coid,
                    qty=wo.qty,
                    side=wo.side,
                    ts=wo.submitted_at.isoformat(),
                    broker_order_id=wo.broker_order_id,
                    strategy=wo.strategy,
                )
        self._state.save_open_orders(entries)
