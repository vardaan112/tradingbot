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
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.data.historical.news import NewsClient

from communication.discord_client import (
    DiscordCallbacks,
    DiscordCommandCenter,
    enqueue_discord_alert,
    format_report,
    format_status,
    simulated_fill_discord_spec,
    startup_initialization_notification,
)
from config.constants import LOGGER_APP, LOGGER_RISK, LOGGER_STRATEGY
from config.logging_config import configure_logging, get_context_filter
from config.settings import Settings
from config.strategy_runtime import merge_strategy_thresholds, resolve_dynamic_params_path
from core.account import AccountAdapter, AccountSnapshot, PositionSnapshot
from core.alpaca_clients import AlpacaClients, build_alpaca_clients, shutdown_clients
from core.database import Database
from core.exceptions import (
    BrokerConnectionError,
    KillSwitchLatchedError,
    OrderPlacementError,
)
from core.market_clock import MarketClock
from core.market_data import BarFetcher, Quote, QuoteCache
from core.orders import OrderService, WorkingOrder
from core.position_ledger import reconcile_open_positions
from core.skiplist import SymbolSkiplist
from core.state_store import SessionSnapshot, StateStore
from core.trading_stream import StreamHealth, TradingStreamRunner
from risk.anti_martingale import (
    RiskMode,
    recent_trade_pnls_preview,
    resolve_anti_martingale,
)
from risk.compliance import ComplianceAdapter
from risk.correlation import correlation_block_reason
from risk.emergency import SpyFlashCrashMonitor
from risk.exposure import ExposureChecker
from risk.killswitch import KillSwitch
from risk.position_sizer import PositionSizer
from strategies.base import Signal, SignalAction, StrategyContext
from strategies.ml_filter import MLSignalFilter
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.skip_diagnostics import (
    SkipCodes,
    SkipDiagnosticsThrottle,
    SkipReason,
    emit_skip_diagnostic,
)
from strategies.scanner import (
    load_scan_record,
    maybe_refresh_after_open,
    merge_tradeable_universe,
    refresh_universe_now,
    symbols_for_strategy_ticks,
)
from strategies.sentiment import (
    AlpacaNewsSentimentFetcher,
    CachedSentimentProvider,
    SentimentSnapshot,
    VaderSentimentBackend,
    sentiment_overlay_neutral,
)
from strategies.universe import UniverseFilter
from utils.tearsheet import tearsheet_primary
from utils.time_utils import now_eastern, now_utc, today_eastern

from .autotune import iso_week_token, run_autotune_job
from .heartbeat import HeartbeatService
from .reporter import generate_daily_report

_TIMEFRAME_SECONDS: dict[str, float] = {
    "1Min": 60.0,
    "5Min": 300.0,
    "15Min": 900.0,
    "1Hour": 3600.0,
    "1Day": 86400.0,
}


class Orchestrator:
    """Application root."""

    def __init__(self, settings: Settings, *, skip_startup_discord_embed: bool = False) -> None:
        self._settings = settings
        self._skip_startup_discord_embed = bool(skip_startup_discord_embed)

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
        dbp = Path(settings.DATABASE_PATH)
        self._database = Database(dbp if dbp.is_absolute() else Path.cwd() / dbp)
        self._sentiment_provider: CachedSentimentProvider | None = None
        self._skiplist = SymbolSkiplist(Path(settings.STATE_DIR))
        self._strategy_runtime_thr = merge_strategy_thresholds(
            settings,
            dyn_path=resolve_dynamic_params_path(settings),
        )
        self._ml_filter: MLSignalFilter | None = (
            MLSignalFilter(settings) if settings.ENABLE_ML_FILTER else None
        )
        self._last_positions_snapshot: list[PositionSnapshot] = []
        self._entry_audit: dict[str, dict[str, Any]] = {}
        self._last_report_date_et: str | None = None
        self._tick_anti_mart: tuple[RiskMode, float, str] = (RiskMode.NORMAL, 1.0, "")
        self._stop = asyncio.Event()

        self._clients: AlpacaClients | None = None
        self._account_adapter: AccountAdapter | None = None
        self._market_clock: MarketClock | None = None
        self._quote_cache: QuoteCache | None = None
        self._bar_fetcher: BarFetcher | None = None
        self._stream_runner: TradingStreamRunner | None = None
        self._stream_health = StreamHealth()
        self._compliance = ComplianceAdapter(settings)
        self._kill_switch = KillSwitch(self._state, drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
        self._exposure = ExposureChecker(settings)
        self._sizer = PositionSizer(settings, self._compliance, self._exposure, database=self._database)
        self._discord_out: asyncio.Queue[dict[str, Any]] | None = (
            asyncio.Queue(maxsize=64) if settings.ENABLE_DISCORD_BOT else None
        )
        self._discord_task: asyncio.Task[Any] | None = None
        self._orchestrator_skip_throttle = SkipDiagnosticsThrottle()
        self._universe_skip_throttle = SkipDiagnosticsThrottle()
        self._strategy = RSIMeanReversionStrategy(
            settings,
            state_store=self._state,
            database=self._database,
            runtime_thresholds=self._strategy_runtime_thr,
            ml_filter=self._ml_filter,
            discord_embed_fn=(
                (lambda spec, orch=self: enqueue_discord_alert(orch._discord_out, spec))
                if settings.ENABLE_DISCORD_BOT
                else None
            ),
        )
        self._universe = UniverseFilter(
            settings,
            strategy_name=self._strategy.name,
            discord_enqueue=(
                (lambda spec, orch=self: enqueue_discord_alert(orch._discord_out, spec))
                if settings.ENABLE_DISCORD_BOT
                else None
            ),
            skip_throttle=self._universe_skip_throttle,
        )
        self._order_service: OrderService | None = None
        self._heartbeat: HeartbeatService | None = None

        self._bars_cache: dict[str, pd.DataFrame] = {}
        self._latest_account: AccountSnapshot | None = None
        self._latest_positions: list[PositionSnapshot] = []
        self._latest_open_orders: int = 0

        self._scanned_symbols: list[str] = []
        self._stream_symbol_list: list[str] = []
        self._corr_cache: dict[str, tuple[float, str | None]] = {}
        self._phase8_ml_last_et_day: str | None = None
        self._phase8_autotune_last_et_week: str | None = None
        self._stream_bad_ticks_consec: int = 0
        self._black_swan = SpyFlashCrashMonitor(
            symbol=str(settings.BLACK_SWAN_SYMBOL),
            drop_pct=settings.BLACK_SWAN_DROP_PCT,
            window_minutes=settings.BLACK_SWAN_WINDOW_MINUTES,
        )
        self._shutdown_completed = False
        self._shutdown_flatten_live_positions = False
        self._canary_gate_label: str = "n/a"

    def set_canary_gate_label(self, label: str) -> None:
        self._canary_gate_label = label[:120]

    def request_shutdown_flatten_live_positions(self) -> None:
        """User interrupt: request live flatten during shutdown (non-dry-run)."""

        self._shutdown_flatten_live_positions = True
        self._stop.set()

    async def run_ml_startup_gate(self) -> bool:
        """Pre-boot ML training using SQLite only. Returns False to abort process."""

        filt = self._ml_filter
        mp = Path(self._settings.ML_MODEL_PATH)
        if not mp.is_absolute():
            mp = Path.cwd() / mp
        model_found = mp.is_file()
        model_loaded_initial = bool(filt and filt.is_trained)

        gate_ok = True
        gate_reason = "n_a"
        model_trained = model_loaded_initial

        if not bool(self._settings.ENABLE_ML_FILTER) or filt is None:
            self._log.info("event=ml_startup_training_skipped reason=disabled")
            gate_reason = "ml_disabled"
            model_trained = False
        else:
            n = int(self._database.count_completed_trades_ml_eligible())
            thresh = int(self._settings.MIN_ML_TRAINING_TRADES)
            if n <= thresh:
                self._log.info(
                    "event=ml_startup_training_skipped reason=insufficient_completed_trades "
                    "have=%s required_gt=%s",
                    n,
                    thresh,
                )
                gate_reason = "insufficient_completed_trades"
                model_trained = bool(filt.is_trained)
            else:
                self._log.info(
                    "event=ml_startup_training_started trade_count=%s min_ml_training_trades_gt=%s",
                    n,
                    thresh,
                )
                try:
                    await asyncio.to_thread(filt.train_from_database, self._database)
                except Exception as exc:  # noqa: BLE001
                    self._log.exception("event=ml_startup_training_failed err=%s", exc)
                    if bool(self._settings.ML_BLOCK_ENTRIES_ON_TRAINING_FAILURE):
                        filt.mark_startup_training_failure()
                    enqueue_discord_alert(
                        self._discord_out,
                        {
                            "title": "ML_STARTUP_TRAINING_FAILED",
                            "lines": [
                                "Startup model training raised an exception.",
                                "See logs: event=ml_startup_training_failed",
                            ],
                            "color": 0xE67E22,
                        },
                    )
                    gate_ok = not bool(self._settings.ML_ABORT_ON_TRAINING_FAILURE)
                    gate_reason = "train_exception"
                    model_trained = bool(filt.is_trained)
                else:
                    if not bool(filt.is_trained):
                        self._log.error(
                            "event=ml_startup_training_failed "
                            "reason=train_returned_without_model trade_count=%s",
                            n,
                        )
                        if bool(self._settings.ML_BLOCK_ENTRIES_ON_TRAINING_FAILURE):
                            filt.mark_startup_training_failure()
                        enqueue_discord_alert(
                            self._discord_out,
                            {
                                "title": "ML_STARTUP_TRAINING_FAILED",
                                "lines": [
                                    "Training finished but no model was persisted "
                                    "(missing sklearn/joblib or no rows).",
                                    "See logs: event=ml_startup_training_failed",
                                ],
                                "color": 0xE67E22,
                            },
                        )
                        gate_ok = not bool(self._settings.ML_ABORT_ON_TRAINING_FAILURE)
                        gate_reason = "train_returned_without_model"
                        model_trained = False
                    else:
                        self._log.info("event=ml_startup_training_completed trade_count=%s", n)
                        enqueue_discord_alert(
                            self._discord_out,
                            {
                                "title": "ML_STARTUP_TRAINING_COMPLETED",
                                "lines": [f"eligible_completed_trades={n}", f"required_gt={thresh}"],
                                "color": 0x1ABC9C,
                            },
                        )
                        gate_ok = True
                        gate_reason = "train_completed"
                        model_trained = True

        block_active = bool(
            filt is not None and getattr(filt, "_block_entries_due_to_startup_failure", False),
        )
        infer_fail_open = not block_active

        self._log.info(
            "event=ml_startup_gate model_found=%s model_loaded_initial=%s model_trained=%s "
            "gate_ok=%s fail_open=%s reason=%s",
            str(model_found).lower(),
            str(model_loaded_initial).lower(),
            str(model_trained).lower(),
            str(gate_ok).lower(),
            str(infer_fail_open).lower(),
            gate_reason,
        )
        return gate_ok

    async def boot(self) -> None:
        self._log.info(
            "Boot: env=%s live_enabled=%s dry_run=%s reg_mode=%s symbols=%s",
            self._settings.ALPACA_ENV,
            self._settings.LIVE_TRADING_ENABLED,
            self._settings.DRY_RUN,
            self._settings.REGULATORY_MODE,
            self._settings.symbols_list,
        )

        with contextlib.suppress(Exception):
            self._database.init_schema()

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
            database=self._database,
            simulated_fill_sink=(
                (
                    lambda evt, orch=self: enqueue_discord_alert(
                        orch._discord_out,
                        simulated_fill_discord_spec(evt),
                    )
                )
                if self._settings.ENABLE_DISCORD_BOT and self._discord_out is not None
                else None
            ),
        )

        if self._settings.SENTIMENT_ENABLED:
            try:
                news_client = NewsClient(
                    api_key=self._settings.ALPACA_API_KEY,
                    secret_key=self._settings.ALPACA_API_SECRET,
                )
                fetcher = AlpacaNewsSentimentFetcher(news_client)
                backend = VaderSentimentBackend()

                def _persist_sentiment_snap(snap: SentimentSnapshot) -> None:
                    self._database.record_sentiment_score(
                        symbol=snap.symbol,
                        score=snap.sentiment_score,
                        label=snap.sentiment_label,
                        headline_count=snap.headline_count,
                        latest_headline_timestamp=snap.latest_headline_timestamp,
                        stale_news=1 if snap.stale_news else 0,
                        metadata={
                            "reason": snap.reason,
                            "source_count": snap.source_count,
                        },
                    )

                self._sentiment_provider = CachedSentimentProvider(
                    self._settings,
                    fetcher,
                    backend,
                    record_fn=_persist_sentiment_snap,
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Sentiment disabled (init error): %s", exc)
                self._sentiment_provider = None

        # Pull initial snapshots
        await self._refresh_account_state()
        if self._latest_account is None:
            raise RuntimeError("Failed to fetch initial account snapshot")

        acc0 = self._latest_account
        self._log.info(
            "event=startup_initialization_notification account_snapshot_ok=true "
            "equity_usd=%.2f buying_power_usd=%.2f",
            float(acc0.equity),
            float(acc0.buying_power),
        )

        # Capture (or restore) daily start equity
        self._kill_switch.ensure_daily_baseline(self._latest_account.equity)

        # Reconcile any open orders
        self._order_service.reconcile_open_orders_from_broker()

        await self._refresh_account_state()
        reconcile_open_positions(
            self._latest_positions,
            state=self._state,
            adopt_trail=self._strategy.adopt_long_position,
            log=self._log,
            db=self._database,
            strategy_name=self._strategy.name,
        )

        if self._settings.DYNAMIC_UNIVERSE_ENABLED:
            snap = load_scan_record(self._state)
            self._scanned_symbols = list(snap.symbols) if snap and snap.symbols else []
            try:
                sess0 = self._market_clock.get_session()
                if sess0.is_open:
                    rec = await asyncio.to_thread(
                        refresh_universe_now,
                        self._settings,
                        trading=self._clients.trading,
                        bar_fetcher=self._bar_fetcher,
                        state=self._state,
                        force=False,
                    )
                    self._scanned_symbols = list(rec.symbols)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Initial universe scan deferred: %s", exc)
        else:
            self._scanned_symbols = []

        self._stream_symbol_list = merge_tradeable_universe(
            self._settings,
            self._scanned_symbols if self._settings.DYNAMIC_UNIVERSE_ENABLED else None,
        )

        # Warm up bar caches
        self._warmup_bars(self._stream_symbol_list)

        # Seed quote cache with latest REST quotes (may be replaced by ws later)
        self._seed_quotes(self._stream_symbol_list)
        self._warn_unbuyable_symbols_under_cap()

        # Build streams
        self._stream_runner = TradingStreamRunner(
            self._clients.trading_stream,
            self._clients.market_stream,
            symbols=self._stream_symbol_list,
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
            tearsheet_orders_path=self._settings.LOG_DIR / "orders.log",
            tearsheet_summary_fn=lambda: tearsheet_primary(
                self._settings,
                db=self._database,
                orders_log_path=self._settings.LOG_DIR / "orders.log",
            ),
        )
        await self._heartbeat.start()

        if self._settings.ENABLE_DISCORD_BOT and self._discord_out is not None:

            async def _remote_kill() -> None:
                if self._latest_account is None:
                    return
                self._kill_switch.force_latch(
                    "discord_remote_kill",
                    current_equity=float(self._latest_account.equity),
                )
                await self._enter_killed_mode()

            cb = DiscordCallbacks(
                status_text_fn=lambda: format_status(self),
                report_text_fn=lambda: format_report(self),
                kill_fn=_remote_kill,
                skip_fn=lambda sym: self._skiplist.skip_for_session_day(
                    session_day_et=today_eastern().isoformat(),
                    symbol=sym,
                ),
            )
            self._discord_task = asyncio.create_task(
                DiscordCommandCenter(self._settings, cb).run(self._discord_out),
            )
            if not self._skip_startup_discord_embed:
                ml_ok = bool(self._ml_filter.is_trained) if self._ml_filter is not None else False
                sym_preview = ",".join(self._stream_symbol_list[:24])[:900]
                acc = self._latest_account
                init_spec = startup_initialization_notification(
                    settings=self._settings,
                    equity=float(acc.equity) if acc is not None else None,
                    buying_power=float(acc.buying_power) if acc is not None else None,
                    symbols_preview=sym_preview or ",".join(self._settings.symbols_list),
                    kill_switch_latched=self._kill_switch.is_latched(),
                    heartbeat_active=True,
                    canary_status=self._canary_gate_label,
                    ml_ready=ml_ok,
                    risk_mode_label=RiskMode.NORMAL.value,
                )
                enqueue_discord_alert(self._discord_out, init_spec)
                self._log.info(
                    "event=discord_startup_notification_sent title=%s",
                    init_spec.get("title", ""),
                )

    async def discord_remote_kill(self) -> None:
        """Expose remote kill latch for integrations/tests."""

        if self._latest_account is None:
            return
        self._kill_switch.force_latch(
            "discord_remote_kill",
            current_equity=float(self._latest_account.equity),
        )
        await self._enter_killed_mode()

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
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._stop.wait(),
                        timeout=self._settings.ORCHESTRATOR_TICK_SECONDS,
                    )
        except KeyboardInterrupt:
            self._log.warning("event=shutdown_requested source=keyboard_interrupt")
            self.request_shutdown_flatten_live_positions()
        finally:
            await self.shutdown()

    def request_shutdown(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        if self._shutdown_completed:
            return
        self._shutdown_completed = True
        flatten_exc: Exception | None = None

        want_flatten = (
            self._shutdown_flatten_live_positions
            and not self._settings.DRY_RUN
            and bool(self._settings.can_submit_real_orders)
        )

        if want_flatten and self._order_service is not None:
            with contextlib.suppress(Exception):
                await self._refresh_account_state()
            has_pos = any(
                abs(float(p.qty)) >= 0.99
                for p in (self._latest_positions or [])
                if str(p.side).lower() == "long"
            )
            if has_pos:
                self._log.critical("event=shutdown_flatten_started")
                try:
                    await self._emergency_flatten_all_positions_no_discord()
                    self._log.info("event=shutdown_flatten_completed")
                except Exception as exc:  # noqa: BLE001
                    flatten_exc = exc
                    self._log.exception("event=shutdown_flatten_failed err=%s", exc)
                    enqueue_discord_alert(
                        self._discord_out,
                        {
                            "title": "SHUTDOWN_FLATTEN_FAILED",
                            "lines": [
                                "Shutdown flatten raised an exception — verify open positions.",
                                "See logs: event=shutdown_flatten_failed",
                            ],
                            "color": 0xC0392B,
                        },
                    )

        if self._settings.DRY_RUN:
            enqueue_discord_alert(
                self._discord_out,
                {
                    "title": "Dry-run shutdown complete",
                    "lines": ["Services closing; no broker flatten in DRY_RUN."],
                    "color": 0x95A5A6,
                },
            )
        else:
            lines = ["Orchestrator shutting down."]
            if want_flatten and flatten_exc is None and self._order_service is not None:
                lines.insert(0, "Live shutdown flatten finished (emergency stack).")
            enqueue_discord_alert(
                self._discord_out,
                {
                    "title": "BOT_SHUTDOWN",
                    "lines": lines,
                    "color": 0x7F8C8D,
                },
            )
        self._log.info("event=discord_shutdown_notification_sent dry_run=%s", str(self._settings.DRY_RUN).lower())

        if self._discord_task is not None:
            self._discord_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._discord_task
            self._discord_task = None
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
        self._log.info("event=shutdown_services_closed")
        self._log.info("Shutdown complete.")

    # ----------------------------------------------------------------- tick

    async def _tick(self) -> None:
        await self._refresh_account_state()
        if self._latest_account is None:
            self._log.warning("Skipping tick: no account snapshot")
            return

        reconcile_open_positions(
            self._latest_positions,
            state=self._state,
            adopt_trail=self._strategy.adopt_long_position,
            log=self._log,
            db=self._database,
            strategy_name=self._strategy.name,
        )

        self._record_position_closures()

        self._strategy_runtime_thr = merge_strategy_thresholds(
            self._settings,
            dyn_path=resolve_dynamic_params_path(self._settings),
        )
        self._strategy.set_runtime_thresholds(self._strategy_runtime_thr)

        nu = datetime.now(UTC)
        if (
            self._settings.BLACK_SWAN_ENABLED
            and self._quote_cache is not None
            and self._bar_fetcher is not None
            and self._clients is not None
        ):
            bss = self._settings.BLACK_SWAN_SYMBOL.upper()
            qq = self._quote_cache.get(bss)
            if qq is None or not qq.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
                qq = self._rest_quote(bss)
            if qq is not None and qq.bid > 0 and qq.ask > qq.bid:
                mid = (qq.bid + qq.ask) / 2.0
                self._black_swan.observe(nu, mid)
                if self._black_swan.triggered():
                    enqueue_discord_alert(
                        self._discord_out,
                        {
                            "title": "BLACK_SWAN_TRIGGER",
                            "lines": [
                                f"symbol={bss}",
                                f"mid_px={mid:.4f}",
                                f"thr={self._settings.BLACK_SWAN_DROP_PCT:.4f}",
                            ],
                            "color": 0x8E44AD,
                        },
                    )
                    self._log_risk.critical(
                        "event=black_swan_trigger symbol=%s drop_pct=%.4f window_m=%s",
                        bss,
                        self._settings.BLACK_SWAN_DROP_PCT,
                        self._settings.BLACK_SWAN_WINDOW_MINUTES,
                        extra={"symbol": bss},
                    )
                    reason = (
                        "black_swan_flash_drop "
                        f"symbol={bss} thr={self._settings.BLACK_SWAN_DROP_PCT:.4f} "
                        f"window_m={self._settings.BLACK_SWAN_WINDOW_MINUTES}"
                    )
                    self._kill_switch.force_latch(
                        reason,
                        current_equity=float(self._latest_account.equity),
                    )
                    await self._enter_killed_mode()
                    return

        decision = self._kill_switch.evaluate(self._latest_account.equity)
        if decision.latched:
            await self._enter_killed_mode()
            return

        session = self._market_clock.get_session()
        can_open = self._market_clock.can_open_new_position(session)
        can_exit = self._market_clock.can_exit_position(session)

        if self._settings.DYNAMIC_UNIVERSE_ENABLED and self._clients is not None and session.is_open:
            maybe_rec = maybe_refresh_after_open(
                self._settings,
                trading=self._clients.trading,
                bar_fetcher=self._bar_fetcher,
                state=self._state,
                session_is_open=True,
            )
            if maybe_rec and maybe_rec.symbols:
                self._scanned_symbols = list(maybe_rec.symbols)

        merged_stream = merge_tradeable_universe(
            self._settings,
            self._scanned_symbols if self._settings.DYNAMIC_UNIVERSE_ENABLED else None,
        )
        self._stream_symbol_list = merged_stream
        if self._stream_runner is not None:
            try:
                added = self._stream_runner.subscribe_quote_symbols(merged_stream)
                if added:
                    self._warmup_bars(added)
                    self._seed_quotes(added)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Incremental quote subscription skipped: %s", exc)

        if not self._stream_health.all_ok:
            self._log.warning("Stream not fully healthy; trading paused this tick")
            self._stream_bad_ticks_consec += 1
            if self._stream_bad_ticks_consec >= 3:
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": "WEBSOCKET_STALE",
                        "lines": ["Quote stream degraded for >=3 orchestrator ticks."],
                        "color": 0xF39C12,
                    },
                )
                self._stream_bad_ticks_consec = 0
        else:
            self._stream_bad_ticks_consec = 0
        compliance_decision = self._compliance.decide(self._latest_account)
        if not compliance_decision.allow_new_entries:
            self._log_risk.info(
                "Compliance blocks new entries: %s (mode=%s)",
                compliance_decision.reason,
                compliance_decision.effective_mode,
            )

        self._order_service.cancel_stale(self._settings.ORDER_TIMEOUT_SECONDS)

        positions_by_symbol = {p.symbol.upper(): p for p in self._latest_positions}
        open_order_symbols = {
            wo.symbol.upper()
            for wo in self._order_service.working_orders_snapshot()
            if wo.symbol
        }

        held = {p.symbol.upper() for p in self._latest_positions if p.side.lower() == "long"}
        tick_syms = symbols_for_strategy_ticks(
            self._settings,
            self._scanned_symbols if self._settings.DYNAMIC_UNIVERSE_ENABLED else None,
            broker_position_symbols=held,
        )

        tick_set_upper = {s.upper() for s in tick_syms}
        bot_managed_notional = sum(
            abs(p.market_value)
            for p in self._latest_positions
            if p.symbol.upper() in tick_set_upper
        )

        tick_recent: list[Any] = []
        with contextlib.suppress(Exception):
            tick_recent = list(self._database.get_recent_completed_trades(limit=50))
        t_mode, t_mult, _t_r = resolve_anti_martingale(self._settings, tick_recent)
        t_preview = recent_trade_pnls_preview(tick_recent, 12)
        self._tick_anti_mart = (t_mode, t_mult, t_preview)

        for symbol in tick_syms:
            sym = symbol.upper()
            if self._skiplist.is_skipped(session_day_et=today_eastern().isoformat(), symbol=sym):
                self._log.info(
                    "event=symbol_skip_list symbol=%s day_et=%s",
                    sym,
                    today_eastern().isoformat(),
                    extra={"symbol": sym},
                )
                continue
            self._warmup_symbol_if_needed(sym)
            bars = self._refresh_symbol_bars(sym)
            quote = self._quote_cache.get(sym)
            if quote is None or not quote.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
                quote = self._rest_quote(sym)

            now_eval = now_utc()
            latest_bar_ts = self._latest_bar_timestamp(bars)
            bar_age_s = self._bar_age_seconds(bars, now_eval)
            max_bar_age_s = self._max_allowed_bar_age_seconds()
            live_ts = quote.timestamp if quote is not None else None
            if bars is None or bars.empty:
                sector = self._settings.sector_for_symbol(sym)
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.MISSING_BARS,
                        message="strategy_bars_missing_for_symbol",
                        symbol=sym,
                        strategy_bar_ts=None,
                        dashboard_bar_ts=live_ts,
                        metadata={
                            "decision_fn": "Orchestrator._tick",
                            "sector": sector,
                            "bar_rows": 0,
                            "max_bar_age_seconds": max_bar_age_s,
                        },
                    ),
                )
                if sym not in positions_by_symbol:
                    continue
            if bar_age_s is None or bar_age_s > max_bar_age_s:
                sector = self._settings.sector_for_symbol(sym)
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.STALE_BARS,
                        message="STALE_BARS: latest strategy bar is older than allowed threshold",
                        symbol=sym,
                        strategy_bar_ts=latest_bar_ts,
                        dashboard_bar_ts=live_ts,
                        metadata={
                            "decision_fn": "Orchestrator._tick",
                            "sector": sector,
                            "bar_rows": len(bars),
                            "bar_age_seconds": bar_age_s,
                            "max_bar_age_seconds": max_bar_age_s,
                        },
                    ),
                )
                if sym not in positions_by_symbol:
                    continue
            self._log_strategy.info(
                "event=strategy_bar_freshness symbol=%s latest_bar_ts=%s strategy_ts=%s "
                "live_quote_ts=%s bar_age_seconds=%.3f bar_rows=%s",
                sym,
                latest_bar_ts.isoformat() if latest_bar_ts is not None else "n_a",
                now_eval.isoformat(),
                live_ts.isoformat() if live_ts is not None else "n_a",
                float(bar_age_s),
                len(bars),
                extra={"symbol": sym},
            )

            elig = self._universe.is_eligible(
                sym,
                quote=quote,
                bars=bars,
                has_position=sym in positions_by_symbol,
                has_open_order=sym in open_order_symbols,
            )

            overlay = sentiment_overlay_neutral(sym)
            if self._settings.SENTIMENT_ENABLED and self._sentiment_provider is not None:
                overlay = self._sentiment_provider.snapshot_for_symbol(sym).to_overlay_dict()

            ctx = StrategyContext(
                symbol=sym,
                bars=bars,
                quote=quote,
                account=self._latest_account,
                positions_by_symbol=positions_by_symbol,
                open_order_symbols=open_order_symbols,
                now_utc=now_eval,
                feed=self._quote_cache.feed,
                sentiment_overlay=overlay,
                anti_martingale_risk_mode=t_mode.value,
                anti_martingale_multiplier=t_mult,
                recent_trade_outcomes_hint=t_preview,
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
                    eligibility_code=elig.code,
                    bot_managed_notional=bot_managed_notional,
                )

        if self._market_clock is not None and not session.is_open and self._settings.DAILY_REPORT_ENABLED:
            det = today_eastern().strftime("%Y-%m-%d")
            if self._last_report_date_et != det:
                with contextlib.suppress(Exception):
                    generate_daily_report(self._settings, self._database)
                    enqueue_discord_alert(
                        self._discord_out,
                        {
                            "title": "DAILY_RECAP_READY",
                            "lines": [f"Report generated for {det}", f"dir={self._settings.REPORTS_DIR}"],
                            "color": 0x1ABC9C,
                        },
                    )
                self._last_report_date_et = det

        await self._phase8_scheduled_jobs(session)

    async def _phase8_scheduled_jobs(self, session: Any) -> None:
        et = now_eastern()
        day_key = et.date().isoformat()
        week_key = iso_week_token(et.astimezone(UTC))

        async def tune() -> None:
            outcome: dict[str, Any] = {"ok": False, "reason": "pending"}
            try:
                outcome = await asyncio.to_thread(run_autotune_job, self._settings)
            finally:
                thr = merge_strategy_thresholds(
                    self._settings,
                    dyn_path=resolve_dynamic_params_path(self._settings),
                )
                self._strategy_runtime_thr = thr
                self._strategy.set_runtime_thresholds(thr)
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": "WEEKLY_AUTOTUNE_COMPLETE",
                        "lines": [
                            f"ok={outcome.get('ok')}",
                            f"applied={outcome.get('applied', 'n/a')}",
                            f"reason={outcome.get('apply_reason', outcome.get('reason', 'n/a'))}",
                        ],
                        "color": 0x16A085,
                    },
                )

        if self._settings.ENABLE_AUTOTUNE and et.weekday() == 6 and et.hour >= int(
            self._settings.AUTOTUNE_SUNDAY_HOUR_ET,
        ):
            if self._phase8_autotune_last_et_week != week_key:
                self._phase8_autotune_last_et_week = week_key
                asyncio.create_task(tune())

        if self._settings.ENABLE_ML_FILTER and self._ml_filter is not None:
            if self._phase8_ml_last_et_day != day_key and et.hour >= 16 and not getattr(
                session, "is_open", False,
            ):
                self._phase8_ml_last_et_day = day_key

                async def retr() -> None:
                    await asyncio.to_thread(self._ml_filter.train_from_database, self._database)

                asyncio.create_task(retr())

    def _finalize_closed_trade(self, sym: str, prev_pos: PositionSnapshot) -> None:
        try:
            exit_px: float | None = None
            qq = self._quote_cache.get(sym) if self._quote_cache else None
            if qq is not None and qq.bid > 0 and qq.ask > qq.bid:
                exit_px = (qq.bid + qq.ask) / 2.0
            audit = self._entry_audit.pop(sym, {})
            entry_px = audit.get("entry_price")
            if entry_px is None:
                entry_px = float(prev_pos.avg_entry_price)
            else:
                entry_px = float(entry_px)
            qty = abs(float(prev_pos.qty))
            pnl = (exit_px - entry_px) * qty if exit_px is not None else None
            ret = ((exit_px / entry_px) - 1.0) if exit_px and entry_px > 0 else None
            now_iso = datetime.now(UTC).isoformat()
            meta_reserve = {
                "opened_at",
                "entry_price",
                "risk_mode",
                "regime_type",
                "sentiment_score",
                "sentiment_label",
            }
            extra_meta = {k: v for k, v in audit.items() if k not in meta_reserve}
            self._database.record_completed_trade(
                trade_id=None,
                symbol=sym,
                side="long",
                quantity=qty,
                entry_price=entry_px,
                exit_price=exit_px,
                realized_pnl=pnl,
                realized_return=ret,
                opened_at=audit.get("opened_at"),
                closed_at=now_iso,
                strategy_name=self._strategy.name,
                risk_mode=audit.get("risk_mode"),
                regime_type=audit.get("regime_type"),
                sentiment_score=audit.get("sentiment_score"),
                sentiment_label=audit.get("sentiment_label"),
                is_canary=0,
                metadata=extra_meta or None,
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "event=db_write_error kind=finalize_closed_trade symbol=%s err=%s",
                sym,
                exc,
            )

    def _record_position_closures(self) -> None:
        prev = list(self._last_positions_snapshot)
        prev_map = {p.symbol.upper(): p for p in prev}
        curr = list(self._latest_positions or [])
        curr_map = {p.symbol.upper(): p for p in curr}

        for sym, p_prev in prev_map.items():
            if str(p_prev.side).lower() != "long":
                continue
            if abs(float(p_prev.qty)) < 0.99:
                continue
            p_now = curr_map.get(sym)
            q_now = abs(float(p_now.qty)) if p_now is not None else 0.0
            if q_now < 0.99:
                self._finalize_closed_trade(sym, p_prev)

        self._last_positions_snapshot = curr

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

    def _emit_orchestrator_enter_skip(self, sr: SkipReason) -> None:
        if "decision_fn" not in sr.metadata:
            sr = replace(
                sr,
                metadata={"decision_fn": "Orchestrator._handle_signal", **dict(sr.metadata)},
            )
        actionable_for_discord = {
            SkipCodes.SIZE_ZERO,
            SkipCodes.STALE_BARS,
            SkipCodes.SPREAD_TOO_WIDE,
            SkipCodes.ORDER_REJECTED,
            SkipCodes.RISK_LIMIT_FAIL,
            SkipCodes.SECTOR_LIMIT_FAIL,
        }
        discord_sink = (
            (lambda spec, o=self: enqueue_discord_alert(o._discord_out, spec))
            if sr.code in actionable_for_discord
            else None
        )
        emit_skip_diagnostic(
            settings=self._settings,
            logger=self._log_strategy,
            log_event="orchestrator_entry_skip",
            sr=sr,
            discord_enqueue=discord_sink,
            throttle=self._orchestrator_skip_throttle,
            phase="orchestrator",
            discord_title="ENTER_BLOCKED",
        )

    async def _handle_signal(
        self,
        signal: Signal,
        *,
        quote: Quote | None,
        can_open: bool,
        can_exit: bool,
        compliance_allow: bool,
        eligible: bool,
        eligibility_reason: str,
        bot_managed_notional: float,
        eligibility_code: str = "ok",
    ) -> None:
        sym = signal.symbol
        coid_extra = {"symbol": sym, "strategy": self._strategy.name}

        if signal.action == SignalAction.NONE:
            return

        if signal.action == SignalAction.ENTER_LONG:
            sector = self._settings.sector_for_symbol(sym)
            if self._kill_switch.is_latched():
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.KILL_SWITCH_LATCHED,
                        message="kill_switch_latched blocks_new_long_entries",
                        symbol=sym,
                        metadata={"sector": sector},
                    ),
                )
                return
            if not can_open:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.MARKET_CLOSED,
                        message="market_or_bot_window_closed_for_new_entries",
                        symbol=sym,
                        metadata={"sector": sector},
                    ),
                )
                return
            if not compliance_allow:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.RISK_LIMIT_FAIL,
                        message="compliance_adapter_disallows_new_entries_this_session",
                        symbol=sym,
                        metadata={"risk_gate": "compliance", "sector": sector},
                    ),
                )
                return
            if not eligible:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.UNIVERSE_INELIGIBLE,
                        message=f"universe_gate detail={eligibility_reason}",
                        symbol=sym,
                        metadata={
                            "eligibility_reason": eligibility_reason,
                            "eligibility_code": eligibility_code,
                            "sector": sector,
                        },
                    ),
                )
                return
            if quote is None:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.QUOTE_INVALID,
                        message="no_quote_at_execution_layer after_strategy_emitted_enter",
                        symbol=sym,
                        metadata={"sector": sector},
                    ),
                )
                return
            if not self._stream_health.all_ok:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.STREAM_UNHEALTHY,
                        message="quote_or_stream_health_gate_failed",
                        symbol=sym,
                        metadata={"stream_all_ok": False, "sector": sector},
                    ),
                )
                return

            max_per_sector = int(self._settings.MAX_OPEN_POSITIONS_PER_SECTOR)
            open_in_sector = self._open_sector_symbols(sector)
            if len(open_in_sector) >= max_per_sector:
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.SECTOR_LIMIT_FAIL,
                        message=(
                            "sector_limit_reached blocks_new_entry "
                            f"sector={sector} open={len(open_in_sector)} "
                            f"max={max_per_sector}"
                        ),
                        symbol=sym,
                        metadata={
                            "sector": sector,
                            "open_in_sector": len(open_in_sector),
                            "max_open_positions_per_sector": max_per_sector,
                            "current_positions_in_sector": ",".join(open_in_sector),
                        },
                    ),
                )
                return

            corr_block = self._maybe_correlation_block(sym)

            try:
                conv_mult = float(signal.metadata.get("conviction_risk_multiplier", 1.0))
            except (TypeError, ValueError):
                conv_mult = 1.0
            am_mode, am_mult, am_preview = getattr(
                self, "_tick_anti_mart", (RiskMode.NORMAL, 1.0, ""),
            )
            sizing = self._sizer.size(
                symbol=sym,
                entry_price=signal.reference_price or quote.bid,
                atr=signal.atr,
                account=self._latest_account,
                positions=self._latest_positions,
                bot_managed_notional=bot_managed_notional,
                conviction_risk_multiplier=conv_mult,
                sizing_block_reason=corr_block,
                anti_martingale_multiplier=am_mult,
                risk_mode=am_mode.value,
                recent_trade_hint=am_preview,
            )
            min_shares = float(self._settings.MIN_SHARES)
            if sizing.shares < min_shares:
                sm = getattr(signal, "metadata", {}) or {}

                def _fmeta(v: object) -> float | None:
                    try:
                        if v is None:
                            return None
                        x = float(v)
                        return x if x == x else None
                    except (TypeError, ValueError):
                        return None

                lc = _fmeta(sm.get("last_close")) or _fmeta(signal.reference_price)
                is_corr = bool(corr_block) or (
                    sizing.skipped_reason is not None
                    and "correlation" in str(sizing.skipped_reason).lower()
                )
                sk_code = SkipCodes.RISK_LIMIT_FAIL if is_corr else SkipCodes.SIZE_ZERO
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=sk_code,
                        message=(
                            "cannot_allocate_positive_share_count "
                            f"skipped_reason={str(sizing.skipped_reason)} "
                            f"rationale={str(sizing.rationale)}"[:480]
                        ),
                        symbol=sym,
                        rsi=_fmeta(sm.get("rsi")),
                        price=lc,
                        sma_200=_fmeta(sm.get("sma200")),
                        sma_200_slope=_fmeta(sm.get("sma_slope")),
                        adx=_fmeta(sm.get("adx")),
                        atr=_fmeta(sm.get("atr")),
                        bid=float(quote.bid) if quote is not None else None,
                        ask=float(quote.ask) if quote is not None else None,
                        spread_pct=_fmeta(sm.get("spread_pct")),
                        quote_age_seconds=float(quote.age_seconds()) if quote is not None else None,
                        risk_qty=float(sizing.shares),
                        metadata={
                            "correlation_block": corr_block or "",
                            "sizer_skipped_reason": sizing.skipped_reason or "",
                            "sizer_rationale": sizing.rationale or "",
                            "risk_mode": sizing.risk_mode,
                            "conviction_risk_multiplier": conv_mult,
                            "max_dollars_per_trade": float(self._settings.max_dollars_per_trade),
                            "raw_shares": sizing.risk_budget / sizing.stop_distance
                            if sizing.stop_distance > 0
                            else 0.0,
                            "final_shares": float(sizing.shares),
                            "fractional_enabled": str(self._settings.ENABLE_FRACTIONAL).lower(),
                            "sector": sector,
                            "open_in_sector": len(open_in_sector),
                            "max_open_positions_per_sector": max_per_sector,
                        },
                    ),
                )
                return

            sym_u = sym.upper()
            meta = getattr(signal, "metadata", {}) or {}
            def _meta_float(v: object) -> float | None:
                try:
                    if v is None:
                        return None
                    x = float(v)
                    return x if x == x else None
                except (TypeError, ValueError):
                    return None

            audit = {
                "opened_at": datetime.now(UTC).isoformat(),
                "entry_price": float(signal.reference_price or quote.bid),
                "risk_mode": sizing.risk_mode,
                "regime_type": meta.get("regime_type"),
                "sentiment_score": meta.get("sentiment_score"),
                "sentiment_label": meta.get("sentiment_label"),
                "rsi": meta.get("rsi"),
                "adx": meta.get("adx"),
                "atr": meta.get("atr"),
                "atr_pct": meta.get("atr_pct"),
                "spread_pct": meta.get("spread_pct"),
                "price_above_sma200": meta.get("price_above_sma200"),
                "last_close": meta.get("last_close"),
                "volatility_tier": meta.get("volatility_tier"),
                "rsi_threshold_used": meta.get("rsi_threshold_used"),
                "sma_filter_passed": meta.get("sma_filter_passed"),
                "aggressive_sma_bypassed": meta.get("aggressive_sma_bypassed"),
                "sector": meta.get("sector") or sector,
                "open_in_sector": len(open_in_sector),
            }
            audit = {k: v for k, v in audit.items() if v is not None}
            self._entry_audit[sym_u] = audit
            try:
                wo: WorkingOrder | None
                if self._settings.PASSIVE_JOINER_ENABLED:
                    wo = await self._order_service.submit_buy_passive_joiner_async(
                        sym_u,
                        int(sizing.shares),
                        quote_refresher=lambda s=sym_u: (
                            self._quote_cache.get(s) if self._quote_cache else None
                        )
                        or self._rest_quote(s),
                    )
                else:
                    wo = self._order_service.submit_limit_entry(
                        sym_u,
                        int(sizing.shares),
                        side="buy",
                        quote=quote,
                        intent_reason=str(signal.reason or "enter_long")[:480],
                    )
            except OrderPlacementError as exc:
                self._entry_audit.pop(sym_u, None)
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.ORDER_REJECTED,
                        message=f"order_service_rejected_entry: {exc}",
                        symbol=sym_u,
                        rsi=_meta_float(meta.get("rsi")),
                        adx=_meta_float(meta.get("adx")),
                        atr=_meta_float(meta.get("atr")),
                        price=float(signal.reference_price or quote.bid),
                        sma_200=_meta_float(meta.get("sma200")),
                        bid=float(quote.bid) if quote is not None else None,
                        ask=float(quote.ask) if quote is not None else None,
                        spread_pct=_meta_float(meta.get("spread_pct")),
                        risk_qty=float(sizing.shares),
                        metadata={
                            "max_dollars_per_trade": float(self._settings.max_dollars_per_trade),
                            "fractional_enabled": str(self._settings.ENABLE_FRACTIONAL).lower(),
                            "sector": sector,
                            "open_in_sector": len(open_in_sector),
                            "max_open_positions_per_sector": max_per_sector,
                        },
                    ),
                )
                self._log.error("Entry placement error %s: %s", sym, exc, extra=coid_extra)
                return
            if wo is None:
                self._entry_audit.pop(sym_u, None)
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.UNKNOWN_SKIP,
                        message="order_service_returned_none_without_exception",
                        symbol=sym_u,
                    ),
                )
            elif str(wo.status).lower() != "dry_run":
                risk_pct_hint = sizing.effective_risk_pct
                base_lines = [
                    f"symbol={sym_u}",
                    f"sector={audit.get('sector','Unknown')}",
                    f"qty={int(sizing.shares)} price={audit.get('entry_price')}",
                    f"risk_pct_eff={risk_pct_hint:.6f}",
                    f"regime={audit.get('regime_type','')}",
                    f"sentiment={audit.get('sentiment_score','')}",
                    f"rsi={audit.get('rsi','')}",
                    f"adx={audit.get('adx','')}",
                    f"atr_pct={audit.get('atr_pct','')}",
                    f"vol_tier={audit.get('volatility_tier','')}",
                    f"rsi_thr={audit.get('rsi_threshold_used','')}",
                    f"sma_bypass={audit.get('aggressive_sma_bypassed','')}",
                    f"open_in_sector={audit.get('open_in_sector','')}",
                    f"spread_pct={audit.get('spread_pct','')}",
                    f"source={'paper' if self._settings.ALPACA_ENV=='paper' else 'live'}",
                ]
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": "ENTER_LONG",
                        "lines": base_lines,
                        "color": 0x2ECC71,
                    },
                )
            self._log_strategy.info(
                "event=trade_decision decision=%s symbol=%s reason=entry_submitted sector=%s qty=%s "
                "rsi=%s adx=%s atr_pct=%s volatility_tier=%s rsi_threshold=%s "
                "price=%s sma200=%s sma_filter_passed=%s aggressive_sma_bypassed=%s "
                "open_in_sector=%s bid=%s ask=%s spread_pct=%s latest_bar_ts=%s",
                "ORDER_SUBMITTED" if wo is not None else "SKIP",
                sym_u,
                str(audit.get("sector", sector)),
                int(sizing.shares),
                str(audit.get("rsi", "n_a")),
                str(audit.get("adx", "n_a")),
                str(audit.get("atr_pct", "n_a")),
                str(audit.get("volatility_tier", "n_a")),
                str(audit.get("rsi_threshold_used", "n_a")),
                str(audit.get("entry_price", "n_a")),
                str(meta.get("sma200", "n_a")),
                str(audit.get("sma_filter_passed", "n_a")),
                str(audit.get("aggressive_sma_bypassed", "n_a")),
                str(audit.get("open_in_sector", "n_a")),
                f"{quote.bid:.6f}" if quote is not None else "n_a",
                f"{quote.ask:.6f}" if quote is not None else "n_a",
                str(audit.get("spread_pct", "n_a")),
                str(meta.get("bar_timestamp", "n_a")),
                extra={"symbol": sym_u},
            )
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
                bid = quote.bid if quote is not None else 0
                ask = quote.ask if quote is not None else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > bid else bid
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": "EXIT_LONG",
                        "lines": [
                            f"symbol={sym}",
                            f"qty={qty}",
                            f"px~{mid:.4f}",
                            str(signal.reason),
                        ],
                    },
                )
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

    def _open_sector_symbols(self, sector: str) -> list[str]:
        target = str(sector or "Unknown")
        out: list[str] = []
        for p in list(self._latest_positions or []):
            if str(p.side).lower() != "long":
                continue
            sym = str(p.symbol).upper()
            if self._settings.sector_for_symbol(sym) == target:
                out.append(sym)
        return out

    def _position_for(self, symbol: str) -> PositionSnapshot | None:
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

    def _warmup_bars(self, symbols: list[str] | None = None) -> None:
        syms = symbols if symbols is not None else (self._stream_symbol_list or self._settings.symbols_list)
        for sym in syms:
            try:
                df = self._fetch_symbol_bars(sym)
                self._bars_cache[sym] = df
                self._log.info("Warmup bars %s rows=%d", sym, len(df))
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Warmup bars failed %s: %s", sym, exc)
                self._bars_cache[sym] = pd.DataFrame()

    def _seed_quotes(self, symbols: list[str] | None = None) -> None:
        syms = symbols if symbols is not None else (self._stream_symbol_list or self._settings.symbols_list)
        for sym in syms:
            try:
                q = self._bar_fetcher.fetch_latest_quote(sym)
                self._quote_cache.set_quote(q)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("Latest quote seed failed %s: %s", sym, exc)

    def _warmup_symbol_if_needed(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym in self._bars_cache and not self._bars_cache[sym].empty:
            return
        self._refresh_symbol_bars(sym, force=True)

    def _fetch_symbol_bars(self, symbol: str) -> pd.DataFrame:
        lookback = self._strategy.warmup_lookback()
        return self._bar_fetcher.fetch_bars(
            symbol.upper(),
            self._settings.BAR_TIMEFRAME,
            lookback_bars=lookback,
        )

    def _refresh_symbol_bars(self, symbol: str, *, force: bool = False) -> pd.DataFrame:
        sym = symbol.upper()
        current = self._bars_cache.get(sym, pd.DataFrame())
        try:
            df = self._fetch_symbol_bars(sym)
            self._bars_cache[sym] = df
            return df
        except Exception as exc:  # noqa: BLE001
            if force:
                self._log.warning("Incremental warmup failed %s: %s", sym, exc)
            else:
                self._log.warning("Bar refresh failed %s: %s", sym, exc)
            if current is not None:
                self._bars_cache[sym] = current
                return current
            self._bars_cache[sym] = pd.DataFrame()
            return self._bars_cache[sym]

    def _latest_bar_timestamp(self, bars: pd.DataFrame) -> datetime | None:
        if bars is None or bars.empty:
            return None
        try:
            ts = bars.index[-1]
            if hasattr(ts, "to_pydatetime"):
                dt = ts.to_pydatetime()
            elif isinstance(ts, datetime):
                dt = ts
            else:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (IndexError, AttributeError, TypeError, ValueError):
            return None

    def _bar_age_seconds(self, bars: pd.DataFrame, now_ts: datetime) -> float | None:
        latest_ts = self._latest_bar_timestamp(bars)
        if latest_ts is None:
            return None
        return max(0.0, float((now_ts - latest_ts).total_seconds()))

    def _max_allowed_bar_age_seconds(self) -> float:
        tf = float(_TIMEFRAME_SECONDS.get(self._settings.BAR_TIMEFRAME, 300.0))
        cfg = float(self._settings.MAX_STRATEGY_BAR_AGE_SECONDS)
        return max(cfg, tf * 1.5)

    def _warn_unbuyable_symbols_under_cap(self) -> None:
        if bool(self._settings.ENABLE_FRACTIONAL):
            return
        cap = float(self._settings.max_dollars_per_trade)
        offenders: list[str] = []
        for sym in list(self._stream_symbol_list or self._settings.symbols_list):
            sym_u = sym.upper()
            px: float | None = None
            qq = self._quote_cache.get(sym_u) if self._quote_cache is not None else None
            if qq is not None and qq.bid > 0 and qq.ask > qq.bid:
                px = (float(qq.bid) + float(qq.ask)) / 2.0
            if px is None:
                bars = self._bars_cache.get(sym_u, pd.DataFrame())
                if bars is not None and not bars.empty:
                    with contextlib.suppress(Exception):
                        px = float(bars["close"].iloc[-1])
            if px is None or px <= 0:
                continue
            if px > cap:
                offenders.append(f"{sym_u}@{px:.2f}")
        if not offenders:
            return
        msg = (
            "event=sizing_preflight_warning code=SIZE_ZERO "
            "fractional_enabled=false "
            f"max_dollars_per_trade={cap:.2f} "
            f"symbols_above_cap={','.join(offenders[:20])}"
        )
        self._log_risk.warning(msg)
        enqueue_discord_alert(
            self._discord_out,
            {
                "title": "SIZING_PRECHECK_WARNING",
                "lines": [
                    "Fractional trading is disabled and some symbols are above allocation cap.",
                    f"max_dollars_per_trade={cap:.2f}",
                    f"symbols={','.join(offenders[:12])}",
                ],
                "color": 0xF39C12,
            },
        )

    def _rest_quote(self, symbol: str) -> Quote | None:
        try:
            q = self._bar_fetcher.fetch_latest_quote(symbol.upper())
            self._quote_cache.set_quote(q)
            return q
        except Exception:  # noqa: BLE001
            return None

    def _maybe_correlation_block(self, symbol: str) -> str | None:
        if not self._settings.CORRELATION_BREAKER_ENABLED or self._bar_fetcher is None:
            return None
        now = time.monotonic()
        cached = self._corr_cache.get(symbol.upper())
        if cached and (now - cached[0]) < 900.0:
            return cached[1]
        reason = correlation_block_reason(
            self._settings,
            follower_symbol=symbol,
            positions=list(self._latest_positions or []),
            bar_fetcher=self._bar_fetcher,
        )
        self._corr_cache[symbol.upper()] = (now, reason)
        return reason

    async def _emergency_flatten_all_positions_no_discord(self) -> None:
        """Cancel opens then limit IOC emergency flatten per position (kill + shutdown)."""

        if self._order_service is None:
            return
        try:
            self._order_service.cancel_all_open()
        except Exception as exc:  # noqa: BLE001
            self._log.error("cancel_all during flatten failed: %s", exc)

        for position in list(self._latest_positions or []):
            sym = position.symbol.upper()
            quote = self._quote_cache.get(sym) if self._quote_cache else None
            if quote is None or not quote.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
                try:
                    if self._bar_fetcher is not None:
                        quote = self._bar_fetcher.fetch_latest_quote(sym)
                except Exception as exc:  # noqa: BLE001
                    self._log.error("Cannot get quote for emergency exit %s: %s", sym, exc)
                    continue
            if quote is None:
                continue
            qty = int(abs(position.qty))
            if qty < 1:
                continue
            side = "sell" if str(position.side).lower() == "long" else "buy"
            try:
                self._order_service.submit_emergency_flatten(
                    sym,
                    qty,
                    side,
                    quote=quote,
                    aggressiveness_pct=self._settings.EMERGENCY_AGGRESSIVENESS_PCT,
                )
            except OrderPlacementError as exc:
                self._log.error("Emergency flatten error %s: %s", sym, exc)

    async def _enter_killed_mode(self) -> None:
        self._log.critical("Kill switch latched - cancelling all + emergency flatten.")
        try:
            latch_reason = str(self._kill_switch.latch_record().reason or "unknown")[:420]
        except Exception:  # noqa: BLE001
            latch_reason = "unknown"
        enqueue_discord_alert(
            self._discord_out,
            {
                "title": "KILL_SWITCH_LATCHED",
                "lines": [latch_reason],
                "color": 0xE74C3C,
            },
        )
        await self._emergency_flatten_all_positions_no_discord()

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
            timestamp=datetime.now(UTC).isoformat(),
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
