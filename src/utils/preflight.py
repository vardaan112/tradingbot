"""Filesystem preflight checks before touching brokers or Discord long-lived clients."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterable

from config.constants import LOGGER_APP
from config.settings import Settings

_LOG = logging.getLogger(LOGGER_APP)


def _to_abs(p: Path) -> Path:
    return p if p.is_absolute() else (Path.cwd() / p)


def _iter_preflight_roots(settings: Settings) -> Iterable[Path]:
    state = _to_abs(settings.STATE_DIR)
    yield state
    yield state / "models"
    yield state / "cache"
    yield state / "param_backups"
    yield _to_abs(settings.LOG_DIR)
    yield _to_abs(settings.REPORTS_DIR)
    yield _to_abs(settings.DATABASE_PATH).parent


def ensure_runtime_paths(settings: Settings) -> None:
    """Ensure standard runtime dirs exist and are writable (fail fast).

    Logs ``event=preflight_path_check`` per path checked.
    """

    seen: set[str] = set()
    for raw in _iter_preflight_roots(settings):
        path = raw.resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)

        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)

        writable = False
        err: str | None = None
        try:
            probe = tempfile.NamedTemporaryFile(delete=True, prefix=".preflight_write_", dir=path)
            probe.close()
            writable = True
        except OSError as exc:
            err = str(exc)

        _LOG.info(
            'event=preflight_path_check path="%s" exists=%s writable=%s',
            key,
            str(existed or path.exists()).lower(),
            str(writable).lower(),
            extra={"path": key},
        )

        if not writable:
            raise RuntimeError(
                f"Preflight failed: directory not writable: {path} ({err})",
            )


def path_writable_quick(path: Path) -> bool:
    """Best-effort writability probe (does not mkdir)."""

    try:
        p = path.resolve()
        probe = tempfile.NamedTemporaryFile(delete=True, prefix=".preflight_write_", dir=p)
        probe.close()
        return True
    except OSError:
        return False


__all__ = ["ensure_runtime_paths", "path_writable_quick"]
