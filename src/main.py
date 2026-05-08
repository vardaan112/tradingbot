"""Entry point for the trading bot.

Startup order (``python main.py`` / ``python -m main``):

1. Load ``.env`` into the process environment (``python-dotenv``).
2. Build validated ``Settings`` (pydantic-settings also reads ``.env``).
3. Configure rotating logging.
4. Preflight: create/verify writable runtime paths.
5. Discord first-contact embed when enabled (accurate persisted kill-switch state; no broker calls).
6. ``REQUIRE_DISCORD_ON_STARTUP`` enforcement when configured.
7. Early kill-switch latch check (persisted state); abort if latched.
8. SQLite schema best-effort init for canary / bookkeeping.
9. Local health probes.
10. Canary gate (when configured for live).
11. Orchestrator construction (ML filter loads from disk here; fail-open if missing).
12. ML startup gate / optional training (abort only if ``ML_ABORT_ON_TRAINING_FAILURE``).
13. Install signal handlers and enter ``run_forever`` (streams, heartbeat, tick loop).

Hard safety:
- Canary failure aborts startup; the main loop is not entered.
- Kill switch latched at startup aborts before any broker work (after optional Discord banner).
- ``REQUIRE_DISCORD_ON_STARTUP`` fails closed if Discord is enabled but unreachable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path
from typing import NoReturn

from communication.discord_client import (
    discord_first_contact_standalone,
    post_discord_standalone_embed,
)
from config.constants import LOGGER_APP
from config.logging_config import configure_logging
from config.settings import Settings, get_settings
from core.database import Database
from core.exceptions import KillSwitchLatchedError
from core.state_store import StateStore
from risk.killswitch import KillSwitch
from services.canary import maybe_run_canary
from services.orchestrator import Orchestrator
from utils.local_health import log_startup_local_health
from utils.preflight import ensure_runtime_paths, path_writable_quick


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, orchestrator: Orchestrator) -> None:
    def _shutdown() -> None:
        logging.getLogger(LOGGER_APP).warning("event=shutdown_requested source=unix_signal")
        orchestrator.request_shutdown_flatten_live_positions()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _shutdown)


def _log_env_mode(settings: Settings, log: logging.Logger) -> None:
    log.info(
        "event=startup_env_loaded dry_run=%s alpaca_env=%s live_trading_enabled=%s "
        "can_submit_real_orders=%s",
        str(settings.DRY_RUN).lower(),
        settings.ALPACA_ENV,
        str(settings.LIVE_TRADING_ENABLED).lower(),
        str(settings.can_submit_real_orders).lower(),
    )


def _log_startup_safety_summary(
    *,
    settings: Settings,
    log: logging.Logger,
    canary_label: str,
    discord_first_ok: bool,
    ml_trained: bool,
    preflight_ok: bool,
) -> None:
    ks = KillSwitch(StateStore(settings.STATE_DIR), drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
    log_dir_ok = path_writable_quick(
        settings.LOG_DIR if settings.LOG_DIR.is_absolute() else Path.cwd() / settings.LOG_DIR,
    )
    db_parent = (
        settings.DATABASE_PATH.parent
        if settings.DATABASE_PATH.is_absolute()
        else (Path.cwd() / settings.DATABASE_PATH).parent
    )
    log.info(
        "event=startup_safety_summary dry_run=%s canary=%s kill_switch_latched=%s "
        "discord_enabled=%s discord_first_contact_ok=%s ml_filter_enabled=%s ml_trained=%s "
        "kelly_enabled=%s autotune_enabled=%s runtime_paths_ok=%s log_dir_writable=%s "
        "database_parent=%s black_swan=%s",
        str(settings.DRY_RUN).lower(),
        canary_label,
        str(ks.is_latched()).lower(),
        str(settings.ENABLE_DISCORD_BOT).lower(),
        str(discord_first_ok).lower(),
        str(settings.ENABLE_ML_FILTER).lower(),
        str(ml_trained).lower(),
        str(settings.ENABLE_KELLY_SIZING).lower(),
        str(settings.ENABLE_AUTOTUNE).lower(),
        str(preflight_ok).lower(),
        str(log_dir_ok).lower(),
        str(db_parent.resolve()),
        str(settings.BLACK_SWAN_ENABLED).lower(),
    )


async def canary_check(settings: Settings, *, database: Database | None = None) -> bool:
    """Top-level pre-flight gate (delegates to ``maybe_run_canary``)."""

    log = logging.getLogger(LOGGER_APP)
    try:
        ok = await maybe_run_canary(settings, database=database)
    except KillSwitchLatchedError:
        log.critical("canary_check: refusing to start - kill switch latched")
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("canary_check: unexpected error: %s", exc)
        return False

    if ok:
        log.info("canary_check: OK - main loop may proceed")
    else:
        log.error("canary_check: failed - main loop will NOT start")
    return ok


async def _amain() -> int:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)

    settings = get_settings()

    configure_logging(settings.LOG_DIR, settings.LOG_LEVEL)
    log = logging.getLogger(LOGGER_APP)
    _log_env_mode(settings, log)

    try:
        ensure_runtime_paths(settings)
    except RuntimeError as exc:
        log.critical("Preflight paths failed: %s", exc)
        return 10

    preflight_ok = True

    ks_early = KillSwitch(StateStore(settings.STATE_DIR), drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
    latch_now = ks_early.is_latched()

    discord_first_ok = await discord_first_contact_standalone(
        settings,
        kill_switch_latched=latch_now,
    )
    if (
        settings.ENABLE_DISCORD_BOT
        and settings.REQUIRE_DISCORD_ON_STARTUP
        and not discord_first_ok
    ):
        log.critical("event=startup_aborted_discord reason=require_discord_on_startup_failed")
        return 7

    if latch_now:
        log.critical("event=startup_aborted_kill_switch reason=already_latched")
        await post_discord_standalone_embed(
            settings,
            title="STARTUP_ABORT",
            lines=["Kill switch already latched — refusing to boot."],
            color=0xC0392B,
        )
        return 8

    dbp = Path(settings.DATABASE_PATH)
    resolved_db = dbp if dbp.is_absolute() else Path.cwd() / dbp
    startup_db = Database(resolved_db)
    with contextlib.suppress(Exception):
        startup_db.init_schema()

    log_startup_local_health(settings, log)

    try:
        ok = await canary_check(settings, database=startup_db)
    except KillSwitchLatchedError:
        return 2
    if not ok:
        await post_discord_standalone_embed(
            settings,
            title="CANARY_FAILED",
            lines=[
                "Startup aborted: live canary check returned False.",
                "See app.log / canary logs for diagnostics.",
            ],
            color=0xC0392B,
        )
        log.critical("Aborting startup: canary_check returned False")
        return 3

    skip_embed = settings.ENABLE_DISCORD_BOT and discord_first_ok
    orchestrator = Orchestrator(settings, skip_startup_discord_embed=skip_embed)

    orch_syms = sorted(orchestrator._settings.symbols_list)
    log.info(
        "event=startup_config_loaded dry_run=%s active_symbols_count=%s kelly_enabled=%s "
        "ml_filter_enabled=%s discord_enabled=%s",
        str(settings.DRY_RUN).lower(),
        len(orch_syms),
        str(settings.ENABLE_KELLY_SIZING).lower(),
        str(settings.ENABLE_ML_FILTER).lower(),
        str(settings.ENABLE_DISCORD_BOT).lower(),
        extra={
            "dry_run": str(settings.DRY_RUN).lower(),
            "symbols_count": str(len(orch_syms)),
        },
    )
    log.info(
        "event=startup_strategy_params symbols=%s vwap_enabled=%s vwap_z_threshold=%.4f "
        "bollinger_enabled=%s bollinger_bw_min=%.6f bollinger_require_touch=%s "
        "dynamic_rsi_enabled=%s dynamic_rsi_base=%.4f dynamic_rsi_short_atr=%s "
        "dynamic_rsi_long_atr=%s adx_low=%.4f adx_high=%.4f time_filter_enabled=%s "
        "trade_start_et=%s trade_end_et=%s enable_fractional=%s min_order_dollars=%.2f",
        ",".join(orch_syms),
        str(settings.VWAP_STRATEGY_ENABLED).lower(),
        float(settings.VWAP_Z_THRESHOLD),
        str(settings.BOLLINGER_ENABLED).lower(),
        float(settings.BOLLINGER_MIN_WIDTH_PCT),
        str(settings.BOLLINGER_REQUIRE_TOUCH).lower(),
        str(settings.DYNAMIC_RSI_ENABLED).lower(),
        float(settings.DYNAMIC_RSI_BASE),
        int(settings.DYNAMIC_RSI_SHORT_ATR),
        int(settings.DYNAMIC_RSI_LONG_ATR),
        float(settings.ADX_LOW),
        float(settings.ADX_HIGH),
        str(settings.TIME_OF_DAY_FILTER_ENABLED).lower(),
        settings.TIME_OF_DAY_TRADE_START,
        settings.TIME_OF_DAY_TRADE_END,
        str(settings.ENABLE_FRACTIONAL).lower(),
        float(settings.MIN_ORDER_DOLLARS),
        extra={"symbols": ",".join(orch_syms)},
    )

    orchestrator.set_canary_gate_label("passed")

    ml_loaded = bool(orchestrator._ml_filter.is_trained) if orchestrator._ml_filter else False
    if orchestrator._ml_filter is None:
        log.info(
            "event=ml_filter_state enabled=false note=skipped",
        )
    else:
        log.info(
            "event=ml_filter_state enabled=%s model_on_disk_loaded=%s",
            str(settings.ENABLE_ML_FILTER).lower(),
            str(ml_loaded).lower(),
        )

    _log_startup_safety_summary(
        settings=settings,
        log=log,
        canary_label="passed",
        discord_first_ok=discord_first_ok,
        ml_trained=ml_loaded,
        preflight_ok=preflight_ok,
    )

    if orchestrator._kill_switch.is_latched():
        log.critical("event=startup_aborted_kill_switch reason=reconcile_after_canary_implausible")
        return 8

    if not await orchestrator.run_ml_startup_gate():
        log.critical("Aborting startup: ML startup gate demanded exit")
        await post_discord_standalone_embed(
            settings,
            title="STARTUP_ABORT",
            lines=[
                "ML startup gate returned False.",
                "If ML_ABORT_ON_TRAINING_FAILURE is enabled, inspect logs.",
            ],
            color=0xC0392B,
        )
        return 4

    ml_final = (
        bool(orchestrator._ml_filter.is_trained) if orchestrator._ml_filter is not None else False
    )

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, orchestrator)

    log.info(
        "event=pref_boot_ready ml_trained=%s proceeding_to_run_forever=true",
        str(ml_final).lower(),
    )

    await orchestrator.run_forever()
    return 0


def main() -> NoReturn:
    try:
        rc = asyncio.run(_amain())
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
