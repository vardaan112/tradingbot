"""Entry point for the trading bot.

Usage:
    python -m main                   # run under the default settings
    python src/main.py               # equivalent

Behavior:
- Loads settings via `Settings()` (which reads the .env file at the repo root).
- Configures rotating-file logging.
- Boots the Orchestrator and runs until SIGINT/SIGTERM.

Hard safety: even when LIVE_TRADING_ENABLED=true and DRY_RUN=false, the
Settings layer requires CONFIRM_LIVE_TRADING=yes_i_understand for the live
endpoint. Until that confirmation is set, the bot refuses to load.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import NoReturn

from config.logging_config import configure_logging
from config.settings import get_settings
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


async def _amain() -> int:
    settings = get_settings()
    configure_logging(settings.LOG_DIR, settings.LOG_LEVEL)
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
