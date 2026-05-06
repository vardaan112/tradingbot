"""Entry point for the trading bot.

Usage:
    python -m main                   # run under the default settings
    python src/main.py               # equivalent

Behavior:
- Loads settings via `Settings()` (which reads the .env file at the repo root).
- Configures rotating-file logging.
- Runs `canary_check(settings)` ONCE per startup before the main loop:
    - On paper / DRY_RUN / disabled live trading, the canary is a no-op.
    - On true live mode (live endpoint + LIVE_TRADING_ENABLED=true +
      DRY_RUN=false + RUN_LIVE_CANARY_ON_STARTUP=true), it performs a single
      tiny round-trip trade and only then proceeds.
- Boots the Orchestrator and runs until SIGINT/SIGTERM.

Hard safety:
- Live endpoint + live trading requires CONFIRM_LIVE_TRADING=yes_i_understand.
- Canary failure aborts startup; the main loop is not entered.
- A latched kill switch aborts startup before any live action is taken.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import NoReturn

from config.constants import LOGGER_APP
from config.logging_config import configure_logging
from config.settings import Settings, get_settings
from core.exceptions import KillSwitchLatchedError
from services.canary import maybe_run_canary
from services.orchestrator import Orchestrator


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, orchestrator: Orchestrator) -> None:
    def _shutdown() -> None:
        orchestrator.request_shutdown()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # add_signal_handler is unavailable on Windows but we target Linux.
            pass


async def canary_check(settings: Settings) -> bool:
    """Top-level pre-flight gate.

    Returns True iff the main trading loop is allowed to start. The actual
    canary trade (when applicable) is implemented in `services.canary`.
    All gating, logging, and persistence rules described in the project spec
    are enforced inside `maybe_run_canary`.

    This thin wrapper exists in `main.py` so the operational surface is
    obvious from the entry point and so tests can stub it independently
    of the orchestrator.

    Raises:
        KillSwitchLatchedError: when the kill switch is latched at startup
            and live trading was requested. Startup must abort in that case.
    """
    log = logging.getLogger(LOGGER_APP)
    try:
        ok = await maybe_run_canary(settings)
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
    settings = get_settings()
    configure_logging(settings.LOG_DIR, settings.LOG_LEVEL)
    log = logging.getLogger(LOGGER_APP)

    try:
        ok = await canary_check(settings)
    except KillSwitchLatchedError:
        return 2
    if not ok:
        log.critical("Aborting startup: canary_check returned False")
        return 3

    orchestrator = Orchestrator(settings)
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, orchestrator)
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
