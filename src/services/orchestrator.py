"""Top-level orchestrator: wires Alpaca clients, streams, risk, and strategy
into a supervised event loop.

Lifecycle:
    boot -> initialize logging -> build clients ->
    snapshot account/positions/open orders -> capture daily start equity ->
    start streams + heartbeat -> run strategy tick loop until shutdown.

Per-tick algorithm:
    1. Snapshot account + positions (REST, retried).
    2. Evaluate kill switch on current equity.
    3. If latched -> cancel all + emergency flatten -> sleep.
    4. Validate session window via MarketClock.
    5. Validate stream health and quote freshness.
    6. Validate compliance/regulatory mode.
    7. For each symbol in universe:
         - build StrategyContext from cached bars + quote.
         - run strategy.evaluate.
         - handle each Signal: enter/exit/emergency-exit using OrderService.
    8. Cancel stale entry orders.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config.constants import LOGGER_APP, LOGGER_RISK, LOGGER_STRATEGY
from config.logging_config import configure_logging, get_context_filter
from config.settings import Settings
from core.account import AccountAdapter, AccountSnapshot, PositionSnapshot
from core.alpaca_clients import AlpacaClients, build_alpaca_clients, shutdown_clients
from core.exceptions import (
    BrokerConnectionError,
    KillSwitchLatchedError,
    OrderPlacementError,
)
from core.market_clock import MarketClock
from core.market_data import BarFetcher, Quote, QuoteCache
from core.orders import OrderService, WorkingOrder
from core.state_store import SessionSnapshot, StateStore
from core.trading_stream import StreamHealth, TradingStreamRunner
from risk.compliance import ComplianceAdapter
from risk.exposure import ExposureChecker
from risk.killswitch import KillSwitch
from risk.position_sizer import PositionSizer
from strategies.base import Signal, SignalAction, StrategyContext
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.universe import UniverseFilter
from utils.time_utils import now_utc

from .heartbeat import HeartbeatService


class Orchestrator:
    """Application root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        configure_logging(settings.LOG_DIR, settings.LOG_LEVEL)
        ctx_filter = get_context_filter()
        ctx_filter.update(
            bot_mode=f"{settings.ALPACA_ENV}/{'live' if settings.can_submit_real_orders else 'dry'}",
            reg_mode=settings.REGULATORY_MODE,
        )
        self._log = logging.getLogger(LOGGER_APP)
        self._log_strategy = logging.getLogger(LOGGER_STRATEGY)
        self._log_risk = logging.getLogger(LOGGER_RISK)

        self._state = StateStore(settings.STATE_DIR)
        self._stop = asyncio.Event()

        self._clients: Optional[AlpacaClients] = None
        self._account_adapter: Optional[AccountAdapter] = None
        self._market_clock: Optional[MarketClock] = None
        self._quote_cache: Optional[QuoteCache] = None
        self._bar_fetcher: Optional[BarFetcher] = None
        self._stream_runner: Optional[TradingStreamRunner] = None
        self._stream_health = StreamHealth()
        self._compliance = ComplianceAdapter(settings)
        self._kill_switch = KillSwitch(self._state, drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
        self._exposure = ExposureChecker(settings)
        self._sizer = PositionSizer(settings, self._compliance, self._exposure)
        self._strategy = RSIMeanReversionStrategy(settings)
        self._universe = UniverseFilter(settings)
        self._order_service: Optional[OrderService] = None
        self._heartbeat: Optional[HeartbeatService] = None

        self._bars_cache: dict[str, pd.DataFrame] = {}
        self._latest_account: Optional[AccountSnapshot] = None
        self._latest_positions: list[PositionSnapshot] = []
        self._latest_open_orders: int = 0

    # ----------------------------------------------------------------- boot

    async def boot(self) -> None:
        self._log.info(
            "Boot: env=%s live_enabled=%s dry_run=%s reg_mode=%s symbols=%s",
            self._settings.ALPACA_ENV,
            self._settings.LIVE_TRADING_ENABLED,
            self._settings.DRY_RUN,
            self._settings.REGULATORY_MODE,
            self._settings.symbols_list,
        )

        self._clients = build_alpaca_clients(self._settings)
        feed = self._clients.resolved_feed

        self._account_adapter = AccountAdapter(
            self._clients.trading,
            max_attempts=self._settings.RETRY_MAX_ATTEMPTS,
            base_delay=self._settings.RETRY_BASE_DELAY_SECONDS,
            max_delay=self._settings.RETRY_MAX_DELAY_SECONDS,
        )
        self._market_clock = MarketClock(
            self._clients.trading,
            max_attempts=self._settings.RETRY_MAX_ATTEMPTS,
            base_delay=self._settings.RETRY_BASE_DELAY_SECONDS,
            max_delay=self._settings.RETRY_MAX_DELAY_SECONDS,
        )
        self._quote_cache = QuoteCache(
            max_age_seconds=self._settings.QUOTE_STALENESS_SECONDS,
            feed=feed,
        )
        self._bar_fetcher = BarFetcher(
            self._clients.historical_data,
            feed=feed,
            max_attempts=self._settings.RETRY_MAX_ATTEMPTS,
            base_delay=self._settings.RETRY_BASE_DELAY_SECONDS,
            max_delay=self._settings.RETRY_MAX_DELAY_SECONDS,
        )
        self._order_service = OrderService(
            self._clients.trading,
            self._settings,
            self._state,
            self._quote_cache,
            strategy_name=self._strategy.name,
        )

        # Pull initial snapshots
        await self._refresh_account_state()
        if self._latest_account is None:
            raise RuntimeError("Failed to fetch initial account snapshot")

        # Capture (or restore) daily start equity
        self._kill_switch.ensure_daily_baseline(self._latest_account.equity)

        # Reconcile any open orders
        self._order_service.reconcile_open_orders_from_broker()

        # Warm up bar caches
        self._warmup_bars()

        # Seed quote cache with latest REST quotes (may be replaced by ws later)
        self._seed_quotes()

        # Build streams
        self._stream_runner = TradingStreamRunner(
            self._clients.trading_stream,
            self._clients.market_stream,
            symbols=self._settings.symbols_list,
            on_trade_update=self._handle_trade_update,
            on_quote=self._handle_quote_event,
            health=self._stream_health,
        )
        await self._stream_runner.start()

        self._heartbeat = HeartbeatService(
            self._settings,
            clock=self._market_clock,
            quote_cache=self._quote_cache,
            stream_health=self._stream_health,
            kill_switch=self._kill_switch,
            compliance=self._compliance,
            snapshot_provider=self._snapshot_for_heartbeat,
        )
        await self._heartbeat.start()

    # ----------------------------------------------------------------- run

    async def run_forever(self) -> None:
        try:
            await self.boot()
        except Exception as exc:  # noqa: BLE001
            self._log.exception("Boot failed: %s", exc)
            raise

        self._log.info("Orchestrator entering main tick loop")
        try:
            while not self._stop.is_set():
                try:
                    await self._tick()
                except KillSwitchLatchedError as exc:
                    self._log.error("Kill switch latched: %s", exc)
                except Exception as exc:  # noqa: BLE001
                    self._log.exception("Unhandled tick error: %s", exc)
                try:
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self._settings.ORCHESTRATOR_TICK_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.shutdown()

    def request_shutdown(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        self._log.info("Shutting down orchestrator...")
        if self._heartbeat is not None:
            with contextlib.suppress(Exception):
                await self._heartbeat.stop()
        if self._stream_runner is not None:
            with contextlib.suppress(Exception):
                await self._stream_runner.stop()
        if self._clients is not None:
            with contextlib.suppress(Exception):
                shutdown_clients(self._clients)
        self._save_session_snapshot()
        self._log.info("Shutdown complete.")

    # ----------------------------------------------------------------- tick

    async def _tick(self) -> None:
        await self._refresh_account_state()
        if self._latest_account is None:
            self._log.warning("Skipping tick: no account snapshot")
            return

        # Kill switch evaluation
        decision = self._kill_switch.evaluate(self._latest_account.equity)
        if decision.latched:
            await self._enter_killed_mode()
            return

        # Session window
        session = self._market_clock.get_session()
        can_open = self._market_clock.can_open_new_position(session)
        can_exit = self._market_clock.can_exit_position(session)

        # Stream health
        if not self._stream_health.all_ok:
            self._log.warning("Stream not fully healthy; trading paused this tick")

        # Compliance
        compliance_decision = self._compliance.decide(self._latest_account)
        if not compliance_decision.allow_new_entries:
            self._log_risk.info(
                "Compliance blocks new entries: %s (mode=%s)",
                compliance_decision.reason,
                compliance_decision.effective_mode,
            )

        # Cancel stale unfilled limits regardless of session
        self._order_service.cancel_stale(self._settings.ORDER_TIMEOUT_SECONDS)

        # Evaluate per-symbol strategy
        positions_by_symbol = {p.symbol.upper(): p for p in self._latest_positions}
        open_order_symbols = {
            wo.symbol.upper()
            for wo in self._order_service.working_orders_snapshot()
            if wo.symbol
        }

        bot_managed_notional = sum(
            abs(p.market_value)
            for p in self._latest_positions
            if p.symbol.upper() in {s.upper() for s in self._settings.symbols_list}
        )

        for symbol in self._settings.symbols_list:
            sym = symbol.upper()
            quote = self._quote_cache.get(sym)
            bars = self._bars_cache.get(sym, pd.DataFrame())

            elig = self._universe.is_eligible(
                sym,
                quote=quote,
                bars=bars,
                has_position=sym in positions_by_symbol,
                has_open_order=sym in open_order_symbols,
            )

            ctx = StrategyContext(
                symbol=sym,
                bars=bars,
                quote=quote,
                account=self._latest_account,
                positions_by_symbol=positions_by_symbol,
                open_order_symbols=open_order_symbols,
                now_utc=now_utc(),
                feed=self._quote_cache.feed,
            )

            for signal in self._strategy.evaluate(ctx):
                await self._handle_signal(
                    signal,
                    quote=quote,
                    can_open=can_open,
                    can_exit=can_exit,
                    compliance_allow=compliance_decision.allow_new_entries,
                    eligible=elig.eligible,
                    eligibility_reason=elig.reason,
                    bot_managed_notional=bot_managed_notional,
                )

    # ------------------------------------------------------------ handlers

    async def _handle_trade_update(self, event) -> None:  # noqa: ANN001 - alpaca model
        if self._order_service is None:
            return
        await self._order_service.handle_trade_update(event)

    async def _handle_quote_event(self, event) -> None:  # noqa: ANN001 - alpaca model
        if self._quote_cache is None:
            return
        self._quote_cache.update_from_event(event)

    # ------------------------------------------------------------ signals

    async def _handle_signal(
        self,
        signal: Signal,
        *,
        quote: Optional[Quote],
        can_open: bool,
        can_exit: bool,
        compliance_allow: bool,
        eligible: bool,
        eligibility_reason: str,
        bot_managed_notional: float,
    ) -> None:
        sym = signal.symbol
        coid_extra = {"symbol": sym, "strategy": self._strategy.name}

        if signal.action == SignalAction.NONE:
            return

        if signal.action == SignalAction.ENTER_LONG:
            if not can_open:
                self._log_strategy.info("ENTER skipped (window closed) %s", sym, extra=coid_extra)
                return
            if not compliance_allow:
                self._log_strategy.info("ENTER skipped (compliance) %s", sym, extra=coid_extra)
                return
            if not eligible:
                self._log_strategy.info(
                    "ENTER skipped (universe: %s) %s",
                    eligibility_reason, sym, extra=coid_extra,
                )
                return
            if quote is None:
                self._log_strategy.info("ENTER skipped (no quote) %s", sym, extra=coid_extra)
                return
            if not self._stream_health.all_ok:
                self._log_strategy.info("ENTER skipped (stream unhealthy) %s", sym, extra=coid_extra)
                return

            sizing = self._sizer.size(
                symbol=sym,
                entry_price=signal.reference_price or quote.bid,
                atr=signal.atr,
                account=self._latest_account,
                positions=self._latest_positions,
                bot_managed_notional=bot_managed_notional,
            )
            if sizing.shares < 1:
                self._log_strategy.info(
                    "ENTER size=0 reason=%s %s", sizing.skipped_reason or sizing.rationale, sym,
                    extra=coid_extra,
                )
                return

            try:
                self._order_service.submit_limit_entry(
                    sym,
                    int(sizing.shares),
                    side="buy",
                    quote=quote,
                )
            except OrderPlacementError as exc:
                self._log.error("Entry placement error %s: %s", sym, exc, extra=coid_extra)
            return

        if signal.action == SignalAction.EXIT_LONG:
            if not can_exit:
                self._log_strategy.info("EXIT skipped (window closed) %s", sym, extra=coid_extra)
                return
            position = self._position_for(sym)
            if position is None or quote is None:
                return
            qty = int(abs(position.qty))
            if qty < 1:
                return
            try:
                self._order_service.submit_limit_exit(sym, qty, side="sell", quote=quote)
            except OrderPlacementError as exc:
                self._log.error("Exit placement error %s: %s", sym, exc, extra=coid_extra)
            return

        if signal.action == SignalAction.EMERGENCY_EXIT_LONG:
            position = self._position_for(sym)
            if position is None or quote is None:
                return
            qty = int(abs(position.qty))
            if qty < 1:
                return
            try:
                self._order_service.submit_emergency_flatten(
                    sym,
                    qty,
                    side="sell",
                    quote=quote,
                    aggressiveness_pct=self._settings.EMERGENCY_AGGRESSIVENESS_PCT,
                )
            except OrderPlacementError as exc:
                self._log.error("Emergency flatten error %s: %s", sym, exc, extra=coid_extra)
            return

    # ------------------------------------------------------------ helpers

    def _position_for(self, symbol: str) -> Optional[PositionSnapshot]:
        sym = symbol.upper()
        for p in self._latest_positions:
            if p.symbol.upper() == sym:
                return p
        return None

    async def _refresh_account_state(self) -> None:
        try:
            self._latest_account = self._account_adapter.fetch_account()
            self._latest_positions = self._account_adapter.fetch_positions()
            self._latest_open_orders = len(self._account_adapter.fetch_open_orders() or [])
        except BrokerConnectionError as exc:
            self._log.warning("Account refresh failed: %s", exc)

    def _warmup_bars(self) -> None:
        lookback = self._strategy.warmup_lookback()
        for sym in self._settings.symbols_list:
            try:
                df = self._bar_fetcher.fetch_bars(
                    sym,
                    self._settings.BAR_TIMEFRAME,
                    lookback_bars=lookback,
                )
                self._bars_cache[sym] = df
                self._log.info("Warmup bars %s rows=%d", sym, len(df))
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Warmup bars failed %s: %s", sym, exc)
                self._bars_cache[sym] = pd.DataFrame()

    def _seed_quotes(self) -> None:
        for sym in self._settings.symbols_list:
            try:
                q = self._bar_fetcher.fetch_latest_quote(sym)
                self._quote_cache.set_quote(q)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Latest quote seed failed %s: %s", sym, exc)

    async def _enter_killed_mode(self) -> None:
        self._log.critical("Kill switch latched - cancelling all + emergency flatten.")
        try:
            self._order_service.cancel_all_open()
        except Exception as exc:  # noqa: BLE001
            self._log.error("cancel_all during kill failed: %s", exc)

        for position in list(self._latest_positions):
            sym = position.symbol.upper()
            quote = self._quote_cache.get(sym) if self._quote_cache else None
            if quote is None or not quote.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
                # Pull a fresh quote via REST as a fallback.
                try:
                    quote = self._bar_fetcher.fetch_latest_quote(sym)
                except Exception as exc:  # noqa: BLE001
                    self._log.error("Cannot get quote for emergency exit %s: %s", sym, exc)
                    continue
            qty = int(abs(position.qty))
            if qty < 1:
                continue
            side = "sell" if position.side.lower() == "long" else "buy"
            try:
                self._order_service.submit_emergency_flatten(
                    sym, qty, side, quote=quote,
                    aggressiveness_pct=self._settings.EMERGENCY_AGGRESSIVENESS_PCT,
                )
            except OrderPlacementError as exc:
                self._log.error("Emergency flatten error %s: %s", sym, exc)

    def _snapshot_for_heartbeat(self) -> dict:
        if self._latest_account is None:
            return {"equity": 0.0, "buying_power": 0.0, "open_positions": 0, "open_orders": 0}
        return {
            "equity": self._latest_account.equity,
            "buying_power": self._latest_account.buying_power,
            "open_positions": len(self._latest_positions),
            "open_orders": self._latest_open_orders,
        }

    def _save_session_snapshot(self) -> None:
        if self._latest_account is None or self._quote_cache is None:
            return
        snapshot = SessionSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            equity=self._latest_account.equity,
            buying_power=self._latest_account.buying_power,
            open_positions=len(self._latest_positions),
            open_orders=self._latest_open_orders,
            feed=self._quote_cache.feed,
            extra={
                "regulatory_mode": self._compliance.effective_mode(),
                "live_trading_enabled": self._settings.LIVE_TRADING_ENABLED,
                "dry_run": self._settings.DRY_RUN,
            },
        )
        try:
            self._state.save_session_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("Failed to persist session snapshot: %s", exc)


# Convenience helper for `WorkingOrder` reference (re-export for tests).
__all__ = ["Orchestrator", "WorkingOrder"]
