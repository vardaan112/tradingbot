"""Production logging configuration with rotating file handlers.

Each named logger gets a dedicated rotating file (or a shared file when
explicitly mapped). All loggers also stream to stdout. A `BotContextFilter`
injects per-event metadata (mode, regulatory mode, symbol, strategy,
client_order_id) so every line carries operational context.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Any

from .constants import (
    LOGGER_APP,
    LOGGER_DATA,
    LOGGER_ERRORS,
    LOGGER_HEARTBEAT,
    LOGGER_ORDERS,
    LOGGER_RISK,
    LOGGER_STRATEGY,
    LOGGER_STREAM,
)

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "mode=%(bot_mode)s | reg=%(reg_mode)s | "
    "symbol=%(symbol)s | strategy=%(strategy)s | coid=%(client_order_id)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

# Mapping of each named logger -> the file name it writes to.
# Loggers sharing a file simply share the same handler instance.
_LOG_FILES: dict[str, str] = {
    LOGGER_APP: "app.log",
    LOGGER_ORDERS: "orders.log",
    LOGGER_RISK: "risk.log",
    LOGGER_HEARTBEAT: "heartbeat.log",
    LOGGER_ERRORS: "errors.log",
    LOGGER_STRATEGY: "app.log",
    LOGGER_DATA: "app.log",
    LOGGER_STREAM: "app.log",
}

_DEFAULTS: dict[str, str] = {
    "bot_mode": "boot",
    "reg_mode": "unknown",
    "symbol": "-",
    "strategy": "-",
    "client_order_id": "-",
}

_configured_lock = threading.Lock()
_configured = False


class BotContextFilter(logging.Filter):
    """Inject bot-wide mutable context fields into every log record."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._context: dict[str, str] = dict(_DEFAULTS)

    def update(self, **fields: Any) -> None:
        """Set one or more context fields. Values of None map to '-'."""
        with self._lock:
            for k, v in fields.items():
                self._context[k] = str(v) if v is not None else "-"

    def get(self, key: str) -> str:
        with self._lock:
            return self._context.get(key, "-")

    def filter(self, record: logging.LogRecord) -> bool:
        with self._lock:
            for key, default in _DEFAULTS.items():
                if not hasattr(record, key) or getattr(record, key, None) in (None, ""):
                    setattr(record, key, self._context.get(key, default))
        return True


_context_filter = BotContextFilter()


def get_context_filter() -> BotContextFilter:
    """Return the shared BotContextFilter singleton."""
    return _context_filter


def _build_file_handler(path: Path, level: int) -> logging.Handler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
        delay=True,
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.addFilter(_context_filter)
    return handler


def _build_stream_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.addFilter(_context_filter)
    return handler


def configure_logging(
    log_dir: Path,
    level_name: str = "INFO",
) -> None:
    """Configure global rotating-file + stdout logging.

    Idempotent across calls.
    """
    global _configured
    with _configured_lock:
        if _configured:
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        level = getattr(logging, level_name.upper(), logging.INFO)

        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):
            root.removeHandler(h)

        # Single stream handler at root so unknown loggers still surface.
        root.addHandler(_build_stream_handler(level))

        # Build one file handler per unique destination file and reuse it.
        unique_handlers: dict[str, logging.Handler] = {}
        for file_name in set(_LOG_FILES.values()):
            unique_handlers[file_name] = _build_file_handler(log_dir / file_name, level)

        # Errors log: a second WARNING-only handler attached to root so any
        # warning from anywhere ends up in errors.log too.
        errors_warn_handler = _build_file_handler(log_dir / "errors.log", logging.WARNING)
        errors_warn_handler.setLevel(logging.WARNING)
        # Reuse the rotating file already created for errors.log to avoid
        # double rotation contention.
        root.addHandler(errors_warn_handler)

        # Wire each named logger to its dedicated file handler. Disable
        # propagation so we do not get duplicates on root's stream/errors.
        for logger_name, file_name in _LOG_FILES.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = False
            logger.addHandler(unique_handlers[file_name])
            # Each named logger should still hit stdout and the errors file.
            logger.addHandler(_build_stream_handler(level))
            logger.addHandler(errors_warn_handler)

        # Tame noisy third-party loggers.
        for noisy in ("httpx", "httpcore", "websockets", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True
