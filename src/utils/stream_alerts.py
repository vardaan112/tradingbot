"""Periodic websocket health checks + rate-limited desktop notifications."""

from __future__ import annotations

import logging
import time

from config.settings import Settings
from core.trading_stream import StreamHealth
from utils.notifications import notify_user

_COOLDOWN_BUCKET: dict[str, float] = {}


def evaluate_stream_websocket_notifications(
    *,
    settings: Settings,
    stream_health: StreamHealth,
    log: logging.Logger,
    bucket_key: str = "websocket_stale",
) -> None:
    """Log ``event=websocket_health`` and optionally notify on disconnect/stale."""

    if not settings.ENABLE_LOCAL_NOTIFICATIONS:
        only_log_status(settings, stream_health, log)
        return

    status, secs, reconn = stream_health.websocket_health_snapshot(
        stale_seconds_threshold=float(settings.STREAM_STALE_SECONDS),
    )
    should_notify = status in {"disconnected", "stale"}

    nowm = time.monotonic()
    last = _COOLDOWN_BUCKET.get(bucket_key, 0.0)
    cooldown_s = float(settings.STREAM_NOTIFICATION_COOLDOWN_SECONDS)
    can_notify = should_notify and (nowm - last >= cooldown_s)

    notification_sent = False
    if can_notify:
        try:
            notify_user(
                title="TradingBot: websocket unhealthy",
                message=f"status={status} seconds_since_last_message={secs:.1f}",
                severity="warning" if status == "stale" else "critical",
            )
            notification_sent = True
            _COOLDOWN_BUCKET[bucket_key] = nowm
        except Exception as exc:  # noqa: BLE001
            log.warning("notification pipeline error: %s", exc)

    log.info(
        "event=websocket_health status=%s seconds_since_last_message=%.3f "
        "reconnect_attempt_count=%d notification_sent=%s",
        status,
        secs,
        reconn,
        str(notification_sent).lower(),
    )


def only_log_status(settings: Settings, stream_health: StreamHealth, log: logging.Logger) -> None:
    """Structured log without OS notification."""

    status, secs, reconn = stream_health.websocket_health_snapshot(
        stale_seconds_threshold=float(settings.STREAM_STALE_SECONDS),
    )
    log.info(
        "event=websocket_health status=%s seconds_since_last_message=%.3f "
        "reconnect_attempt_count=%d notification_sent=false",
        status,
        secs,
        reconn,
    )


def reset_notification_cooldown(bucket_key: str | None = None) -> None:
    """Test hook to clear rate-limit state."""

    if bucket_key is None:
        _COOLDOWN_BUCKET.clear()
    else:
        _COOLDOWN_BUCKET.pop(bucket_key, None)


__all__ = ["evaluate_stream_websocket_notifications", "only_log_status", "reset_notification_cooldown"]
