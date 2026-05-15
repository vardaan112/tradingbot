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
from core.trade_source import runtime_trade_source
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
from strategies.registry import build_strategies
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.indicators import atr as atr_indicator
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
from services.regime_detector import QqqRegimeDetector
from utils.tearsheet import tearsheet_primary
from utils.time_utils import now_eastern, now_utc, today_eastern
from utils.price_utils import round_to_tick, tick_size_for

from .autotune import iso_week_token, run_autotune_job
from .heartbeat import HeartbeatService
from .reporter import generate_daily_report
from .ensemble import WeightedEnsembleEngine, votes_to_contributing_json
from .shadow_portfolio import ShadowPortfolioManager
from .strategy_engine import StrategyEngine

_TIMEFRAME_SECONDS: dict[str, float] = {
    "1Min": 60.0,
    "5Min": 300.0,
    "15Min": 900.0,
    "1Hour": 3600.0,
    "1Day": 86400.0,
}


def _fmt_qty(qty: float | int) -> str:
    q = float(qty)
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return f"{q:.6f}".rstrip("0").rstrip(".")


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
        self._finalized_symbols: set[str] = set()
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
        discord_fn = (
            (lambda spec, orch=self: enqueue_discord_alert(orch._discord_out, spec))
            if settings.ENABLE_DISCORD_BOT
            else None
        )
        self._strategies = build_strategies(
            settings.active_strategies_list,
            settings,
            state_store=self._state,
            database=self._database,
            runtime_thresholds=self._strategy_runtime_thr,
            ml_filter=self._ml_filter,
            discord_embed_fn=discord_fn,
        )
        rsi_instances = [s for s in self._strategies if isinstance(s, RSIMeanReversionStrategy)]
        if not rsi_instances:
            raise ValueError(
                "Orchestrator requires RSIMeanReversionStrategy in ACTIVE_STRATEGIES "
                "(Phase 2: order path and reconcile still RSI-backed).",
            )
        self._strategy = rsi_instances[0]
        self._strategy_engine = StrategyEngine(
            self._strategies,
            settings=settings,
            database=self._database,
            signal_source=None,
            replay_run_id=None,
        )
        self._weighted_ensemble: WeightedEnsembleEngine | None = None
        if settings.ENSEMBLE_ENABLED and settings.STRATEGY_RUN_MODE in ("ensemble", "both"):
            self._weighted_ensemble = WeightedEnsembleEngine(settings, database=self._database)
        self._perf_weight_refresh_mono: float | None = None
        self._shadow_portfolios: ShadowPortfolioManager | None = None
        if settings.SHADOW_TRADING_ENABLED and len(self._strategies) > 1:
            self._shadow_portfolios = ShadowPortfolioManager(settings, self._database)
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
        self._regime_detector: QqqRegimeDetector | None = None
        self._exec_risk_entry_ix: dict[str, int] = {}
        self._exec_risk_trail: dict[str, float] = {}

    def set_canary_gate_label(self, label: str) -> None:
        self._canary_gate_label = label[:120]

    def request_shutdown_flatten_live_positions(self) -> None:
        """User interrupt: request live flatten during shutdown (non-dry-run)."""

        self._shutdown_flatten_live_positions = True
        self._stop.set()

    @staticmethod
    def _is_auth_error(exc: BaseException) -> bool:
        """Best-effort 401 / auth detection across wrapped exception chains."""

        seen: set[int] = set()
        stack: list[BaseException] = [exc]
        while stack:
            cur = stack.pop()
            cid = id(cur)
            if cid in seen:
                continue
            seen.add(cid)
            txt = f"{type(cur).__name__}: {cur}".lower()
            if any(
                token in txt
                for token in (
                    "401",
                    "unauthorized",
                    "forbidden",
                    "invalid api key",
                    "auth",
                    "permission denied",
                )
            ):
                return True
            cause = getattr(cur, "__cause__", None)
            ctx = getattr(cur, "__context__", None)
            if isinstance(cause, BaseException):
                stack.append(cause)
            if isinstance(ctx, BaseException):
                stack.append(ctx)
        return False

    def _log_auth_hint_if_needed(self, exc: BaseException, *, phase: str) -> None:
        if not self._is_auth_error(exc):
            return
        self._log.critical(
            "AUTH ERROR: Check if your .env Alpaca Keys match your PAPER/LIVE mode. "
            "phase=%s err=%s",
            phase,
            exc,
        )

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
        symbols_csv = ",".join(self._settings.symbols_list)
        self._log.info(
            "Boot: env=%s live_enabled=%s dry_run=%s reg_mode=%s symbols=%s",
            self._settings.ALPACA_ENV,
            self._settings.LIVE_TRADING_ENABLED,
            self._settings.DRY_RUN,
            self._settings.REGULATORY_MODE,
            self._settings.symbols_list,
        )
        self._log.info(
            "event=startup_symbols_loaded symbols_count=%s symbols_csv=%s",
            len(self._settings.symbols_list),
            symbols_csv,
        )

        try:
            self._database.init_schema()
        except Exception as exc:  # noqa: BLE001
            self._log.exception("event=schema_migration_failed component=database err=%s", exc)
            enqueue_discord_alert(
                self._discord_out,
                {
                    "title": "DATABASE_INIT_FAILED",
                    "lines": ["Database schema initialization failed.", "See logs: event=schema_migration_failed"],
                    "color": 0xE74C3C,
                },
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
        self._regime_detector = (
            QqqRegimeDetector(self._settings, self._bar_fetcher)
            if self._settings.QQQ_REGIME_ENABLED
            else None
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
            execution_event_source=runtime_trade_source(self._settings),
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
            self._log_auth_hint_if_needed(exc, phase="boot")
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
            execution_event_source=runtime_trade_source(self._settings),
        )

        self._record_position_closures()

        self._strategy_runtime_thr = merge_strategy_thresholds(
            self._settings,
            dyn_path=resolve_dynamic_params_path(self._settings),
        )
        for strat in self._strategies:
            setter = getattr(strat, "set_runtime_thresholds", None)
            if callable(setter):
                setter(self._strategy_runtime_thr)

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

        for dead in list(self._exec_risk_entry_ix.keys()):
            if dead not in positions_by_symbol:
                self._exec_risk_entry_ix.pop(dead, None)
                self._exec_risk_trail.pop(dead, None)

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

        all_quotes_by_symbol: dict[str, Quote] = {}
        if self._quote_cache is not None:
            for s in tick_syms:
                qq = self._quote_cache.get(s)
                if qq is not None:
                    all_quotes_by_symbol[s.upper()] = qq

        tick_now = now_utc()
        if self._regime_detector is not None:
            self._regime_detector.refresh_if_stale(now_utc=tick_now)

        if (
            self._weighted_ensemble is not None
            and self._settings.ENSEMBLE_WEIGHT_MODE == "performance"
        ):
            now_m = time.monotonic()
            interval = max(120.0, float(self._settings.ORCHESTRATOR_TICK_SECONDS) * 4.0)
            if self._perf_weight_refresh_mono is None or (now_m - self._perf_weight_refresh_mono) >= interval:
                self._weighted_ensemble.refresh_weights(record_decision=True)
                self._perf_weight_refresh_mono = now_m

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
            if bars is None or bars.empty:
                bars = await self._recover_missing_bars(sym)
            quote = self._quote_cache.get(sym)
            if quote is None or not quote.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
                quote = self._rest_quote(sym)
            quote = self._quote_with_bar_fallback(sym, quote, bars, now_eval=now_utc())

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
                self._log_strategy.info(
                    "event=strategy_bar_stale symbol=%s last_bar_time=%s current_time=%s "
                    "bar_age_seconds=%s max_bar_age_seconds=%.3f bar_rows=%s",
                    sym,
                    latest_bar_ts.isoformat() if latest_bar_ts is not None else "n_a",
                    now_eval.isoformat(),
                    f"{bar_age_s:.3f}" if bar_age_s is not None else "n_a",
                    max_bar_age_s,
                    len(bars),
                    extra={"symbol": sym, "skip_code": SkipCodes.STALE_BARS},
                )
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

            pos_chk = positions_by_symbol.get(sym)
            if (
                pos_chk is not None
                and str(pos_chk.side).lower() == "long"
                and bars is not None
                and not bars.empty
            ):
                min_q = (
                    float(self._settings.FRACTIONAL_MIN_QTY)
                    if self._settings.ENABLE_FRACTIONAL
                    else float(self._settings.MIN_SHARES)
                )
                if abs(float(pos_chk.qty)) + 1e-9 >= min_q:
                    if await self._maybe_forced_execution_exit(sym, bars, quote):
                        continue

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
                qqq_regime_bear_volatile=bool(
                    self._regime_detector.snapshot.bear_volatile if self._regime_detector else False
                ),
                regime_anchor_state=(
                    self._regime_detector.snapshot.anchor_state if self._regime_detector else "Unknown"
                ),
                regime_anchor_rsi=(
                    self._regime_detector.snapshot.anchor_rsi if self._regime_detector else None
                ),
                regime_anchor_close=(
                    self._regime_detector.snapshot.anchor_close if self._regime_detector else None
                ),
                regime_anchor_sma=(
                    self._regime_detector.snapshot.anchor_sma if self._regime_detector else None
                ),
                all_bars_by_symbol=dict(self._bars_cache),
                all_quotes_by_symbol=all_quotes_by_symbol,
            )

            primary = self._strategy.name
            raw_all = list(self._strategy_engine.evaluate(ctx))

            ens_dec = None
            ens_sig = None
            if self._weighted_ensemble is not None:
                ens_dec = self._weighted_ensemble.decide(
                    sym,
                    raw_all,
                    has_position=sym in positions_by_symbol,
                )
                ens_sig = self._weighted_ensemble.to_signal(ens_dec)

            if self._shadow_portfolios is not None:
                self._shadow_portfolios.on_symbol(
                    symbol=sym,
                    timestamp_iso=ctx.now_utc.isoformat(),
                    raw_signals=raw_all,
                    quote=quote,
                    ensemble_decision=ens_dec,
                    ensemble_signal=ens_sig,
                )

            if self._weighted_ensemble is not None:
                dec = ens_dec
                assert dec is not None and ens_sig is not None
                if self._database is not None:
                    wscore = (
                        float(dec.weighted_exit_score)
                        if dec.final_action
                        in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                        else float(dec.weighted_enter_score)
                    )
                    th = (
                        float(dec.exit_threshold)
                        if dec.final_action
                        in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                        else float(dec.enter_threshold)
                    )
                    self._database.record_strategy_decision(
                        source=runtime_trade_source(self._settings),
                        timestamp=ctx.now_utc.isoformat(),
                        symbol=sym,
                        final_action=dec.final_action.value,
                        run_id=None,
                        decision_type="weighted_ensemble",
                        weighted_score=wscore,
                        threshold=th,
                        contributing_signals_json=votes_to_contributing_json(dec.contributing_votes),
                        metadata={
                            "weighted_enter_score": dec.weighted_enter_score,
                            "weighted_exit_score": dec.weighted_exit_score,
                            "ensemble_reason": dec.reason,
                        },
                    )
                sig = ens_sig
                if (
                    self._settings.STRATEGY_RUN_MODE != "independent"
                    and sig.action != SignalAction.NONE
                ):
                    await self._handle_signal(
                        sig,
                        quote=quote,
                        can_open=can_open,
                        can_exit=can_exit,
                        compliance_allow=compliance_decision.allow_new_entries,
                        eligible=elig.eligible,
                        eligibility_reason=elig.reason,
                        eligibility_code=elig.code,
                        bot_managed_notional=bot_managed_notional,
                    )
            elif self._settings.STRATEGY_RUN_MODE != "independent":
                for signal in raw_all:
                    if signal.strategy_name != primary:
                        continue
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
                for strat in self._strategies:
                    setter = getattr(strat, "set_runtime_thresholds", None)
                    if callable(setter):
                        setter(thr)
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
                self._track_background_task(tune(), name="weekly_autotune")

        if self._settings.ENABLE_ML_FILTER and self._ml_filter is not None:
            if self._phase8_ml_last_et_day != day_key and et.hour >= 16 and not getattr(
                session, "is_open", False,
            ):
                self._phase8_ml_last_et_day = day_key

                async def retr() -> None:
                    await asyncio.to_thread(self._ml_filter.train_from_database, self._database)

                self._track_background_task(retr(), name="daily_ml_retrain")

    def _track_background_task(self, coro: Any, *, name: str) -> asyncio.Task:
        """Create a task and log any exception instead of losing it silently."""

        task = asyncio.create_task(coro, name=name)

        def _done(t: asyncio.Task) -> None:
            try:
                t.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                self._log.exception("event=async_task_failed task=%s err=%s", name, exc)
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": "ASYNC_TASK_FAILED",
                        "lines": [f"task={name}", "See logs: event=async_task_failed"],
                        "color": 0xE74C3C,
                    },
                )

        task.add_done_callback(_done)
        return task

    def _finalize_closed_trade(self, sym: str, prev_pos: PositionSnapshot) -> None:
        try:
            exit_px: float | None = None
            exit_px_source = "unknown"
            ef = (
                self._order_service.consume_last_exit_fill(sym)
                if self._order_service is not None
                else None
            )
            if ef is not None and ef.avg_fill_price > 0:
                exit_px = float(ef.avg_fill_price)
                exit_px_source = "broker_fill"
                expected_qty = abs(float(prev_pos.qty))
                mismatch = abs(ef.filled_qty - expected_qty) / max(expected_qty, 1e-9)
                if mismatch > 0.05:
                    self._log.warning(
                        "event=exit_fill_qty_mismatch symbol=%s ef_qty=%.4f pos_qty=%.4f "
                        "mismatch_pct=%.1f exit_price_source=broker_fill",
                        sym, ef.filled_qty, expected_qty, mismatch * 100,
                    )
            else:
                qq = self._quote_cache.get(sym) if self._quote_cache else None
                if qq is not None and qq.bid > 0 and qq.ask > qq.bid:
                    exit_px = (qq.bid + qq.ask) / 2.0
                    exit_px_source = "quote_mid_fallback"
            audit = self._entry_audit.pop(sym, {})
            # Closed-trade labels must be anchored to broker/ledger fill data,
            # not the quote used when the entry signal was created.
            entry_px_source = str(audit.get("entry_fill_source") or "position_snapshot_avg")
            entry_px = audit.get("entry_filled_avg_price")
            if entry_px is None or float(entry_px) <= 0:
                entry_px = float(prev_pos.avg_entry_price)
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
            extra_meta["exit_price_source"] = exit_px_source
            extra_meta["entry_price_source"] = entry_px_source
            # ML / analytics: persist how this row was produced so training can exclude
            # dry-run or paper rows later (see README + .env ML_TRAINING_* comments).
            extra_meta["execution_alpaca_env"] = str(self._settings.ALPACA_ENV)
            extra_meta["execution_dry_run"] = 1 if self._settings.DRY_RUN else 0
            extra_meta["execution_live_order_api"] = (
                1 if self._settings.can_submit_real_orders else 0
            )
            invalid_label = bool(exit_px_source != "broker_fill" or exit_px is None)
            data_quality_flags = {}
            if invalid_label:
                data_quality_flags["degraded_exit_label"] = exit_px_source
                extra_meta["invalid_for_ml"] = True
                extra_meta["invalid_for_kelly"] = True
            trade_source = runtime_trade_source(self._settings)
            row_id = self._database.record_completed_trade(
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
                source=trade_source,
                realized_return_pct=ret,
                entry_notional=entry_px * qty if entry_px is not None else None,
                exit_notional=exit_px * qty if exit_px is not None else None,
                entry_fill_source=entry_px_source,
                exit_fill_source=exit_px_source,
                data_quality_flags=data_quality_flags or None,
                invalid_for_ml=invalid_label,
                invalid_for_kelly=invalid_label,
            )
            if row_id is not None:
                self._log.info(
                    "event=completed_trade_persisted id=%s symbol=%s source=%s "
                    "execution_alpaca_env=%s execution_dry_run=%s execution_live_order_api=%s",
                    row_id,
                    sym,
                    trade_source,
                    self._settings.ALPACA_ENV,
                    str(self._settings.DRY_RUN).lower(),
                    str(self._settings.can_submit_real_orders).lower(),
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
                if sym in self._finalized_symbols:
                    self._log.debug(
                        "event=closure_dedup_skipped symbol=%s reason=already_finalized_this_session",
                        sym,
                    )
                    continue
                self._finalized_symbols.add(sym)
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
            SkipCodes.COMPLIANCE_REJECTED,
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
        sig_meta = getattr(signal, "metadata", {}) or {}
        is_scale_in = str(sig_meta.get("signal_type", "")).strip().lower() == "scale_in"

        def _meta_float(v: object) -> float | None:
            try:
                if v is None:
                    return None
                x = float(v)
                return x if x == x else None
            except (TypeError, ValueError):
                return None

        def _scale_in_skip(
            *,
            skip_code: str,
            reason: str,
            public_code: str = SkipCodes.RISK_LIMIT_FAIL,
            quote_age_seconds: float | None = None,
            risk_qty: float | None = None,
            extra_meta: dict[str, Any] | None = None,
        ) -> None:
            if not is_scale_in:
                return
            self._log_strategy.info(
                "event=scale_in_skip symbol=%s skip_code=%s reason=%s rsi=%s adx=%s "
                "price=%s current_qty=%s proposed_add_qty=%s bullet_number=%s max_bullets=%s "
                "spread_pct=%s quote_age_seconds=%s strategy_bar_ts=%s",
                sym,
                skip_code,
                reason,
                str(sig_meta.get("rsi", "n_a")),
                str(sig_meta.get("adx", "n_a")),
                str(sig_meta.get("last_close", signal.reference_price)),
                str(sig_meta.get("position_qty", "n_a")),
                str(sig_meta.get("proposed_add_qty", "n_a")),
                str(sig_meta.get("bullet_number", "n_a")),
                str(sig_meta.get("max_bullets", "n_a")),
                str(sig_meta.get("spread_pct", "n_a")),
                f"{quote_age_seconds:.3f}" if quote_age_seconds is not None else "n_a",
                str(sig_meta.get("bar_timestamp", "n_a")),
                extra={"symbol": sym, "strategy": self._strategy.name, "skip_code": skip_code},
            )
            self._emit_orchestrator_enter_skip(
                SkipReason(
                    code=public_code,
                    message=f"{skip_code}: {reason}",
                    symbol=sym,
                    rsi=_meta_float(sig_meta.get("rsi")),
                    adx=_meta_float(sig_meta.get("adx")),
                    atr=_meta_float(sig_meta.get("atr")),
                    price=_meta_float(sig_meta.get("last_close")) or _meta_float(signal.reference_price),
                    sma_200=_meta_float(sig_meta.get("sma200")),
                    sma_200_slope=_meta_float(sig_meta.get("sma_slope")),
                    bid=float(quote.bid) if quote is not None else None,
                    ask=float(quote.ask) if quote is not None else None,
                    spread_pct=_meta_float(sig_meta.get("spread_pct")),
                    quote_age_seconds=quote_age_seconds,
                    risk_qty=risk_qty,
                    metadata={
                        "scale_in_skip_code": skip_code,
                        **(extra_meta or {}),
                    },
                ),
            )

        if signal.action == SignalAction.NONE:
            return

        if signal.action == SignalAction.ENTER_LONG:
            sector = self._settings.sector_for_symbol(sym)
            if self._kill_switch.is_latched():
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_kill_switch_latched",
                        reason="kill_switch_latched blocks_new_long_entries",
                        public_code=SkipCodes.KILL_SWITCH_LATCHED,
                    )
                    return
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
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_market_closed",
                        reason="market_or_bot_window_closed_for_new_entries",
                        public_code=SkipCodes.MARKET_CLOSED,
                    )
                    return
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
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_compliance_rejected",
                        reason="compliance_adapter_disallows_new_entries_this_session",
                        public_code=SkipCodes.COMPLIANCE_REJECTED,
                    )
                    return
                self._emit_orchestrator_enter_skip(
                    SkipReason(
                        code=SkipCodes.RISK_LIMIT_FAIL,
                        message="compliance_adapter_disallows_new_entries_this_session",
                        symbol=sym,
                        metadata={"risk_gate": "compliance", "sector": sector},
                    ),
                )
                return
            if (not is_scale_in) and (not eligible):
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
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_stale_quote",
                        reason="no_quote_at_execution_layer after_strategy_emitted_enter",
                        public_code=SkipCodes.QUOTE_INVALID,
                    )
                    return
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
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_stream_unhealthy",
                        reason="quote_or_stream_health_gate_failed",
                        public_code=SkipCodes.STREAM_UNHEALTHY,
                    )
                    return
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
            if (not is_scale_in) and len(open_in_sector) >= max_per_sector:
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

            regime_mult = 1.0
            if self._regime_detector is not None:
                rs = self._regime_detector.snapshot
                regime_unknown = bool(
                    rs.error in {"init_pending", "insufficient_bars", "stale_regime_data"}
                    or (not rs.anchor_symbol and str(rs.anchor_state).lower() == "unknown")
                )
                block_for_regime = bool(
                    (regime_unknown and self._settings.REGIME_UNKNOWN_ACTION == "block_entries")
                    or (rs.bear_volatile and self._settings.REGIME_BEAR_VOLATILE_BLOCK_ENTRIES)
                )
                if rs.bear_volatile or regime_unknown:
                    if block_for_regime:
                        msg = (
                            "Regime_unknown_blocks_entry"
                            if regime_unknown
                            else "QQQ_BearVolatile_macro_gate_blocks_entry"
                        )
                        if is_scale_in:
                            _scale_in_skip(
                                skip_code="scale_in_macro_regime",
                                reason=msg,
                                public_code=SkipCodes.SKIP_MARKET_REGIME,
                                extra_meta={
                                    "QQQ_ATR_ratio": rs.atr_ratio,
                                    "QQQ_close": rs.close,
                                    "regime_error": rs.error,
                                },
                            )
                        else:
                            self._emit_orchestrator_enter_skip(
                                SkipReason(
                                    code=SkipCodes.SKIP_MARKET_REGIME,
                                    message=msg,
                                    symbol=sym,
                                    metadata={
                                        "QQQ_ATR_ratio": rs.atr_ratio,
                                        "QQQ_close": rs.close,
                                        "QQQ_sma50": rs.sma50,
                                        "regime_error": rs.error,
                                        "regime_unknown_action": self._settings.REGIME_UNKNOWN_ACTION,
                                        "sector": sector,
                                    },
                                ),
                            )
                        return
                    regime_mult = max(
                        0.0,
                        1.0 - float(self._settings.REGIME_MAX_EQUITY_REDUCTION),
                    )

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
                regime_equity_multiplier=regime_mult,
            )
            min_shares = float(self._settings.MIN_SHARES)
            if self._settings.ENABLE_FRACTIONAL:
                min_shares = min(min_shares, float(self._settings.FRACTIONAL_MIN_QTY))
            requested_scale_qty = float(
                _meta_float(sig_meta.get("proposed_add_qty")) or float(self._settings.SCALE_IN_ADD_QTY),
            )
            if requested_scale_qty <= 0:
                requested_scale_qty = float(self._settings.FRACTIONAL_MIN_QTY if self._settings.ENABLE_FRACTIONAL else 1.0)
            if not self._settings.ENABLE_FRACTIONAL:
                requested_scale_qty = float(int(round(requested_scale_qty)))
                if requested_scale_qty < 1.0:
                    requested_scale_qty = 1.0

            is_corr = bool(corr_block) or (
                sizing.skipped_reason is not None
                and "correlation" in str(sizing.skipped_reason).lower()
            )
            scale_in_current_qty: float | None = None
            scale_in_existing_risk_usd: float | None = None
            scale_in_added_risk_usd: float | None = None
            scale_in_total_risk_usd: float | None = None
            scale_in_max_allowed_risk_usd: float | None = None
            scale_in_stop_distance: float | None = None

            if is_scale_in:
                current_pos = self._position_for(sym)
                current_qty = abs(float(current_pos.qty)) if current_pos is not None else 0.0
                scale_in_current_qty = current_qty
                quote_age = float(quote.age_seconds()) if quote is not None else None
                if current_pos is None or current_qty <= 0:
                    _scale_in_skip(
                        skip_code="scale_in_bullet_count_unknown",
                        reason="position_snapshot_missing_or_zero_qty",
                        public_code=SkipCodes.UNKNOWN_SKIP,
                        quote_age_seconds=quote_age,
                    )
                    return
                if is_corr:
                    _scale_in_skip(
                        skip_code="scale_in_risk_exceeded",
                        reason=f"correlation_blocked: {corr_block}",
                        public_code=SkipCodes.RISK_LIMIT_FAIL,
                        quote_age_seconds=quote_age,
                        risk_qty=float(sizing.shares),
                        extra_meta={
                            "correlation_block": corr_block or "",
                            "sizer_skipped_reason": sizing.skipped_reason or "",
                        },
                    )
                    return
                if sizing.shares + 1e-9 < requested_scale_qty:
                    _scale_in_skip(
                        skip_code="scale_in_risk_exceeded",
                        reason=(
                            "position_sizer_allocation_below_requested_scale_qty "
                            f"alloc={sizing.shares:.6f} requested={requested_scale_qty:.6f}"
                        ),
                        public_code=SkipCodes.RISK_LIMIT_FAIL,
                        quote_age_seconds=quote_age,
                        risk_qty=float(sizing.shares),
                        extra_meta={
                            "sizer_skipped_reason": sizing.skipped_reason or "",
                            "sizer_rationale": sizing.rationale or "",
                        },
                    )
                    return
                stop_distance = (
                    _meta_float(sig_meta.get("scale_in_stop_distance"))
                    or (float(sizing.stop_distance) if sizing.stop_distance > 0 else 0.0)
                    or float(signal.atr) * float(self._settings.ATR_STOP_MULTIPLIER)
                )
                scale_in_stop_distance = stop_distance
                if stop_distance <= 0:
                    _scale_in_skip(
                        skip_code="scale_in_risk_exceeded",
                        reason="non_positive_stop_distance_for_risk_guard",
                        public_code=SkipCodes.RISK_LIMIT_FAIL,
                        quote_age_seconds=quote_age,
                        risk_qty=float(sizing.shares),
                    )
                    return
                existing_risk_usd = current_qty * stop_distance
                added_risk_usd = requested_scale_qty * stop_distance
                total_risk_usd = existing_risk_usd + added_risk_usd
                max_allowed_risk_usd = float(sizing.risk_budget)
                scale_in_existing_risk_usd = existing_risk_usd
                scale_in_added_risk_usd = added_risk_usd
                scale_in_total_risk_usd = total_risk_usd
                scale_in_max_allowed_risk_usd = max_allowed_risk_usd
                if total_risk_usd > max_allowed_risk_usd + 1e-9:
                    _scale_in_skip(
                        skip_code="scale_in_risk_exceeded",
                        reason=(
                            "combined_position_risk_exceeds_limit "
                            f"existing={existing_risk_usd:.6f} add={added_risk_usd:.6f} "
                            f"total={total_risk_usd:.6f} max={max_allowed_risk_usd:.6f}"
                        ),
                        public_code=SkipCodes.RISK_LIMIT_FAIL,
                        quote_age_seconds=quote_age,
                        risk_qty=float(requested_scale_qty),
                        extra_meta={
                            "current_qty": current_qty,
                            "proposed_add_qty": requested_scale_qty,
                            "avg_entry_price": float(current_pos.avg_entry_price),
                            "current_price": float(signal.reference_price or quote.bid),
                            "stop_price": float(current_pos.avg_entry_price - stop_distance),
                            "existing_risk_usd": existing_risk_usd,
                            "added_risk_usd": added_risk_usd,
                            "total_risk_usd": total_risk_usd,
                            "max_allowed_risk_usd": max_allowed_risk_usd,
                        },
                    )
                    return
                order_qty = float(requested_scale_qty)
            else:
                if sizing.shares < min_shares:
                    lc = _meta_float(sig_meta.get("last_close")) or _meta_float(signal.reference_price)
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
                            rsi=_meta_float(sig_meta.get("rsi")),
                            price=lc,
                            sma_200=_meta_float(sig_meta.get("sma200")),
                            sma_200_slope=_meta_float(sig_meta.get("sma_slope")),
                            adx=_meta_float(sig_meta.get("adx")),
                            atr=_meta_float(sig_meta.get("atr")),
                            bid=float(quote.bid) if quote is not None else None,
                            ask=float(quote.ask) if quote is not None else None,
                            spread_pct=_meta_float(sig_meta.get("spread_pct")),
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
                order_qty = float(sizing.shares)

            sym_u = sym.upper()
            meta = sig_meta

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
                "bollinger_width_pct": meta.get("bollinger_width_pct"),
                "bollinger_lower": meta.get("bollinger_lower"),
                "vwap_distance_pct": meta.get("vwap_distance_pct"),
                "vwap_lower": meta.get("vwap_lower"),
                "signal_type": meta.get("signal_type", "entry"),
                "bullet_number": meta.get("bullet_number"),
                "max_bullets": meta.get("max_bullets"),
                "underwater_pct": meta.get("underwater_pct"),
                "position_qty": meta.get("position_qty"),
                "proposed_add_qty": meta.get("proposed_add_qty"),
                "sector": meta.get("sector") or sector,
                "open_in_sector": len(open_in_sector),
            }
            audit = {k: v for k, v in audit.items() if v is not None}
            self._entry_audit[sym_u] = audit
            self._finalized_symbols.discard(sym_u)
            try:
                wo: WorkingOrder | None
                if self._settings.MIDPOINT_PEG_ENABLED:
                    wo = await self._order_service.submit_midpoint_peg_async(
                        sym_u,
                        order_qty,
                        "buy",
                        quote_refresher=lambda s=sym_u: (
                            self._quote_cache.get(s) if self._quote_cache else None
                        )
                        or self._rest_quote(s),
                        intent_reason=str(signal.reason or "enter_long")[:480],
                    )
                elif self._settings.PASSIVE_JOINER_ENABLED:
                    wo = await self._order_service.submit_buy_passive_joiner_async(
                        sym_u,
                        order_qty,
                        quote_refresher=lambda s=sym_u: (
                            self._quote_cache.get(s) if self._quote_cache else None
                        )
                        or self._rest_quote(s),
                    )
                else:
                    wo = self._order_service.submit_limit_entry(
                        sym_u,
                        order_qty,
                        side="buy",
                        quote=quote,
                        intent_reason=str(signal.reason or "enter_long")[:480],
                    )
            except OrderPlacementError as exc:
                self._entry_audit.pop(sym_u, None)
                if is_scale_in:
                    _scale_in_skip(
                        skip_code="scale_in_order_rejected",
                        reason=f"order_service_rejected_entry: {exc}",
                        public_code=SkipCodes.ORDER_REJECTED,
                        quote_age_seconds=float(quote.age_seconds()) if quote is not None else None,
                        risk_qty=float(order_qty),
                    )
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
                        risk_qty=float(order_qty),
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
            if wo is not None:
                audit["entry_client_order_id"] = wo.client_order_id
                audit["entry_order_id"] = wo.broker_order_id
                audit["entry_limit_price"] = wo.limit_price
                audit["entry_fill_source"] = (
                    "dry_run_fill" if str(wo.status).lower() == "dry_run" else "position_snapshot_avg"
                )
                if float(wo.avg_fill_price or 0.0) > 0:
                    audit["entry_filled_avg_price"] = float(wo.avg_fill_price)
                    audit["entry_fill_source"] = "broker_fill"
                self._entry_audit[sym_u] = audit
            if wo is None:
                self._entry_audit.pop(sym_u, None)
                if self._settings.MIDPOINT_PEG_ENABLED:
                    self._emit_orchestrator_enter_skip(
                        SkipReason(
                            code=SkipCodes.SKIP_MIDPOINT_TIMEOUT,
                            message="midpoint_peg_give_up_no_fill_after_chase_cycles",
                            symbol=sym_u,
                            metadata={"skip_code": "skip_midpoint_timeout"},
                        ),
                    )
                else:
                    self._emit_orchestrator_enter_skip(
                        SkipReason(
                            code=SkipCodes.UNKNOWN_SKIP,
                            message="order_service_returned_none_without_exception",
                            symbol=sym_u,
                        ),
                    )
            elif str(wo.status).lower() != "dry_run":
                risk_pct_hint = sizing.effective_risk_pct
                if is_scale_in:
                    base_lines = [
                        f"symbol={sym_u}",
                        f"current_qty={scale_in_current_qty}",
                        f"add_qty={_fmt_qty(order_qty)}",
                        f"bullet={audit.get('bullet_number','n_a')} / {audit.get('max_bullets','n_a')}",
                        f"avg_entry={meta.get('avg_entry_price','n_a')}",
                        f"current_price={audit.get('entry_price','n_a')}",
                        f"underwater_pct={audit.get('underwater_pct','n_a')}",
                        f"rsi={audit.get('rsi','n_a')}",
                        f"risk_after={scale_in_total_risk_usd} / {scale_in_max_allowed_risk_usd}",
                        "reason=existing_position_underwater_and_secondary_rsi_triggered",
                    ]
                    title = "SCALE_IN_LONG"
                    color = 0xF1C40F
                else:
                    base_lines = [
                        f"symbol={sym_u}",
                        f"sector={audit.get('sector','Unknown')}",
                        f"qty={_fmt_qty(order_qty)} price={audit.get('entry_price')}",
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
                    title = "ENTER_LONG"
                    color = 0x2ECC71
                enqueue_discord_alert(
                    self._discord_out,
                    {
                        "title": title,
                        "lines": base_lines,
                        "color": color,
                    },
                )
            if is_scale_in and wo is not None:
                self._log_strategy.info(
                    "event=scale_in_signal symbol=%s action=BUY qty=%s rsi=%s underwater_pct=%s "
                    "bullet_number=%s max_bullets=%s existing_risk_usd=%s added_risk_usd=%s "
                    "total_risk_usd=%s max_allowed_risk_usd=%s stop_distance=%s",
                    sym_u,
                    _fmt_qty(order_qty),
                    str(audit.get("rsi", "n_a")),
                    str(audit.get("underwater_pct", "n_a")),
                    str(audit.get("bullet_number", "n_a")),
                    str(audit.get("max_bullets", "n_a")),
                    str(scale_in_existing_risk_usd),
                    str(scale_in_added_risk_usd),
                    str(scale_in_total_risk_usd),
                    str(scale_in_max_allowed_risk_usd),
                    str(scale_in_stop_distance),
                    extra={"symbol": sym_u},
                )
            self._log_strategy.info(
                "event=trade_decision decision=%s symbol=%s reason=%s sector=%s qty=%s "
                "rsi=%s adx=%s atr_pct=%s volatility_tier=%s rsi_threshold=%s "
                "price=%s sma200=%s sma_filter_passed=%s aggressive_sma_bypassed=%s "
                "bollinger_width_pct=%s vwap_distance_pct=%s open_in_sector=%s "
                "bid=%s ask=%s spread_pct=%s latest_bar_ts=%s",
                "ORDER_SUBMITTED" if wo is not None else "SKIP",
                sym_u,
                "scale_in_submitted" if is_scale_in else "entry_submitted",
                str(audit.get("sector", sector)),
                _fmt_qty(order_qty),
                str(audit.get("rsi", "n_a")),
                str(audit.get("adx", "n_a")),
                str(audit.get("atr_pct", "n_a")),
                str(audit.get("volatility_tier", "n_a")),
                str(audit.get("rsi_threshold_used", "n_a")),
                str(audit.get("entry_price", "n_a")),
                str(meta.get("sma200", "n_a")),
                str(audit.get("sma_filter_passed", "n_a")),
                str(audit.get("aggressive_sma_bypassed", "n_a")),
                str(audit.get("bollinger_width_pct", "n_a")),
                str(audit.get("vwap_distance_pct", "n_a")),
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
            qty = abs(float(position.qty))
            min_qty = float(self._settings.FRACTIONAL_MIN_QTY) if self._settings.ENABLE_FRACTIONAL else 1.0
            if qty + 1e-12 < min_qty:
                return
            try:
                if self._settings.MIDPOINT_PEG_ENABLED:
                    await self._order_service.submit_midpoint_peg_async(
                        sym,
                        qty,
                        "sell",
                        quote_refresher=lambda s=sym: (
                            self._quote_cache.get(s) if self._quote_cache else None
                        )
                        or self._rest_quote(s),
                        intent_reason=str(signal.reason or "exit_long")[:480],
                    )
                else:
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
                            f"qty={_fmt_qty(qty)}",
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
            qty = abs(float(position.qty))
            min_qty = float(self._settings.FRACTIONAL_MIN_QTY) if self._settings.ENABLE_FRACTIONAL else 1.0
            if qty + 1e-12 < min_qty:
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

    async def _maybe_forced_execution_exit(
        self,
        sym: str,
        bars: pd.DataFrame,
        quote: Quote | None,
    ) -> bool:
        """ATR trailing + bar-count time stop at execution layer (long-only)."""

        if not self._settings.EXEC_ATR_TRAIL_ENABLED and not (
            self._settings.MAX_POSITION_BARS > 0
        ):
            return False
        pos = self._position_for(sym)
        if pos is None or str(pos.side).lower() != "long":
            return False
        qty = abs(float(pos.qty))
        min_q = (
            float(self._settings.FRACTIONAL_MIN_QTY)
            if self._settings.ENABLE_FRACTIONAL
            else float(self._settings.MIN_SHARES)
        )
        if qty + 1e-9 < min_q:
            return False
        if bars.empty or len(bars) < max(25, self._settings.ATR_LENGTH + 3):
            return False

        sym_u = sym.upper()
        bar_idx = len(bars) - 1
        if sym_u not in self._exec_risk_entry_ix:
            self._exec_risk_entry_ix[sym_u] = bar_idx
        entry_ix = self._exec_risk_entry_ix[sym_u]
        bars_held = max(0, bar_idx - entry_ix)

        close = float(bars["close"].iloc[-1])
        atr_s = atr_indicator(
            bars["high"].astype(float),
            bars["low"].astype(float),
            bars["close"].astype(float),
            length=int(self._settings.ATR_LENGTH),
        )
        atr5 = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
        entry_px = float(pos.avg_entry_price)
        k = float(self._settings.ATR_TRAIL_MULTIPLIER)

        reason = ""
        trail = self._exec_risk_trail.get(sym_u)
        if self._settings.EXEC_ATR_TRAIL_ENABLED and atr5 > 0:
            if trail is None:
                trail = entry_px - k * atr5
            else:
                trail = max(float(trail), close - k * atr5)
            self._exec_risk_trail[sym_u] = float(trail)
            if close <= float(trail) + 1e-12:
                reason = "trail"

        if not reason and self._settings.MAX_POSITION_BARS > 0:
            if bars_held >= int(self._settings.MAX_POSITION_BARS):
                reason = "time"

        if not reason:
            return False

        q = quote
        if q is None or not q.is_fresh(self._settings.QUOTE_STALENESS_SECONDS):
            q = self._rest_quote(sym_u)
        if q is None:
            self._log_strategy.info(
                "event=strategy_skip symbol=%s skip_code=skip_time_exit_cancelled reason=no_quote_for_exit",
                sym_u,
                extra={"symbol": sym_u},
            )
            return False

        ev = "exit_trail_stop" if reason == "trail" else "exit_time_stop"
        self._log_strategy.info(
            "event=%s symbol=%s stop=%.6f close=%.6f bars_held=%d atr5=%.6f k=%.4f reason=%s",
            ev,
            sym_u,
            float(self._exec_risk_trail.get(sym_u, close)),
            close,
            bars_held,
            atr5,
            k,
            reason,
            extra={"symbol": sym_u, "skip_code": ev},
        )

        try:
            if self._settings.MIDPOINT_PEG_ENABLED:
                await self._order_service.submit_midpoint_peg_async(
                    sym_u,
                    qty,
                    "sell",
                    quote_refresher=lambda s=sym_u: (
                        self._quote_cache.get(s) if self._quote_cache else None
                    )
                    or self._rest_quote(s),
                    intent_reason=ev,
                )
            else:
                self._order_service.submit_limit_exit(sym_u, qty, side="sell", quote=q)
        except OrderPlacementError as exc:
            self._log_strategy.info(
                "event=strategy_skip symbol=%s skip_code=skip_trail_exit_failed err=%s",
                sym_u,
                exc,
                extra={"symbol": sym_u},
            )
            return False
        return True

    async def _refresh_account_state(self) -> None:
        try:
            self._latest_account = self._account_adapter.fetch_account()
            self._latest_positions = self._account_adapter.fetch_positions()
            self._latest_open_orders = len(self._account_adapter.fetch_open_orders() or [])
        except BrokerConnectionError as exc:
            self._log_auth_hint_if_needed(exc, phase="account_refresh")
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
        lookback = max(int(s.warmup_lookback()) for s in self._strategies)
        return self._bar_fetcher.fetch_bars(
            symbol.upper(),
            self._settings.BAR_TIMEFRAME,
            lookback_bars=lookback,
        )

    async def _recover_missing_bars(self, symbol: str) -> pd.DataFrame:
        """Force a one-symbol backfill when the normal cache refresh returns no bars."""

        sym = symbol.upper()
        lookback = max(200, max(int(s.warmup_lookback()) for s in self._strategies))
        self._log_strategy.warning(
            "event=data_recovery_start symbol=%s reason=bar_rows_0 timeframe=%s lookback_bars=%s",
            sym,
            self._settings.BAR_TIMEFRAME,
            lookback,
            extra={"symbol": sym, "skip_code": SkipCodes.MISSING_BARS},
        )
        if self._bar_fetcher is None:
            self._log_strategy.warning(
                "event=data_recovery_failed symbol=%s reason=no_bar_fetcher",
                sym,
                extra={"symbol": sym, "skip_code": SkipCodes.MISSING_BARS},
            )
            return pd.DataFrame()
        try:
            df = await asyncio.to_thread(
                self._bar_fetcher.fetch_bars,
                sym,
                self._settings.BAR_TIMEFRAME,
                lookback_bars=lookback,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_strategy.warning(
                "event=data_recovery_failed symbol=%s reason=fetch_error err=%s",
                sym,
                exc,
                extra={"symbol": sym, "skip_code": SkipCodes.MISSING_BARS},
            )
            return pd.DataFrame()
        if df is None or df.empty:
            self._log_strategy.warning(
                "event=data_recovery_failed symbol=%s reason=no_bars_returned",
                sym,
                extra={"symbol": sym, "skip_code": SkipCodes.MISSING_BARS},
            )
            self._bars_cache[sym] = pd.DataFrame()
            return self._bars_cache[sym]
        self._bars_cache[sym] = df
        self._log_strategy.info(
            "event=data_recovery_success symbol=%s bar_rows=%s latest_bar_ts=%s",
            sym,
            len(df),
            self._latest_bar_timestamp(df).isoformat()
            if self._latest_bar_timestamp(df) is not None
            else "n_a",
            extra={"symbol": sym},
        )
        return df

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

    def _quote_with_bar_fallback(
        self,
        symbol: str,
        quote: Quote | None,
        bars: pd.DataFrame,
        *,
        now_eval: datetime,
    ) -> Quote | None:
        """Use recent healthy bars as a controlled fallback for bad IEX quotes."""

        if not bool(self._settings.QUOTE_FALLBACK_ENABLED):
            return quote

        sym = symbol.upper()
        quote_max_age = float(self._settings.QUOTE_STALENESS_SECONDS)
        bad_reason = ""
        if quote is None:
            bad_reason = "missing_quote"
        else:
            try:
                q_age = float(quote.age_seconds(reference=now_eval))
            except (TypeError, ValueError):
                q_age = float("inf")
            if quote.bid <= 0 or quote.ask <= quote.bid:
                bad_reason = "invalid_quote"
            elif q_age > quote_max_age:
                bad_reason = f"stale_quote:{q_age:.3f}>{quote_max_age:.3f}"

        if not bad_reason:
            return quote

        fallback_bars = bars
        tf = str(self._settings.QUOTE_FALLBACK_BAR_TIMEFRAME)
        if self._bar_fetcher is not None:
            with contextlib.suppress(Exception):
                fallback_bars = self._bar_fetcher.fetch_bars(sym, tf, lookback_bars=3)
        if fallback_bars is None or fallback_bars.empty:
            self._log_strategy.info(
                "event=quote_fallback_skip symbol=%s skip_code=skip_stale_quote reason=%s fallback_reason=no_bars",
                sym,
                bad_reason,
                extra={"symbol": sym, "skip_code": SkipCodes.SKIP_STALE_QUOTE},
            )
            return quote

        latest_ts = self._latest_bar_timestamp(fallback_bars)
        bar_age = (
            max(0.0, float((now_eval - latest_ts).total_seconds()))
            if latest_ts is not None
            else None
        )
        max_bar_age = float(self._settings.QUOTE_FALLBACK_MAX_BAR_AGE_SECONDS)
        if bar_age is None or bar_age > max_bar_age:
            self._log_strategy.info(
                "event=quote_fallback_skip symbol=%s skip_code=skip_stale_quote reason=%s "
                "fallback_reason=bar_stale bar_age_seconds=%s max_bar_age_seconds=%.3f",
                sym,
                bad_reason,
                f"{bar_age:.3f}" if bar_age is not None else "n_a",
                max_bar_age,
                extra={"symbol": sym, "skip_code": SkipCodes.SKIP_STALE_QUOTE},
            )
            return quote

        try:
            last = fallback_bars.iloc[-1]
            high = float(last["high"])
            low = float(last["low"])
            close = float(last["close"])
            volume = float(last.get("volume", 0.0))
        except (KeyError, TypeError, ValueError, IndexError):
            return quote
        healthy = close > 0 and high >= low > 0 and low <= close <= high and volume > 0
        if not healthy and bool(self._settings.ELASTIC_SPREAD_REQUIRE_BAR_HEALTH):
            self._log_strategy.info(
                "event=quote_fallback_skip symbol=%s skip_code=skip_stale_quote reason=%s "
                "fallback_reason=bar_unhealthy high=%.6f low=%.6f close=%.6f volume=%.0f",
                sym,
                bad_reason,
                high,
                low,
                close,
                volume,
                extra={"symbol": sym, "skip_code": SkipCodes.SKIP_STALE_QUOTE},
            )
            return quote

        midpoint = ((high + low) / 2.0) if self._settings.QUOTE_FALLBACK_USE_BAR_MIDPOINT else close
        if midpoint <= 0:
            return quote
        tick = tick_size_for(midpoint)
        bid = round_to_tick(max(tick, midpoint - tick), mode="down")
        ask = round_to_tick(midpoint + tick, mode="up")
        if ask <= bid:
            ask = bid + tick
        fallback = Quote(
            symbol=sym,
            bid=float(bid),
            ask=float(ask),
            bid_size=0.0,
            ask_size=0.0,
            timestamp=now_eval,
            feed="iex",
        )
        if self._quote_cache is not None:
            self._quote_cache.set_quote(fallback)
        self._log_strategy.info(
            "event=quote_fallback_used symbol=%s reason=%s source_timeframe=%s "
            "bar_ts=%s bar_age_seconds=%.3f bid=%.6f ask=%.6f mid=%.6f close=%.6f volume=%.0f",
            sym,
            bad_reason,
            tf,
            latest_ts.isoformat() if latest_ts is not None else "n_a",
            float(bar_age),
            fallback.bid,
            fallback.ask,
            midpoint,
            close,
            volume,
            extra={"symbol": sym, "skip_code": "quote_fallback_used"},
        )
        return fallback

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
            qty = abs(float(position.qty))
            min_qty = float(self._settings.FRACTIONAL_MIN_QTY) if self._settings.ENABLE_FRACTIONAL else 1.0
            if qty + 1e-12 < min_qty:
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
