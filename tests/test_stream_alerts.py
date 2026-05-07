"""Websocket health logging + notification cooldown (no live broker)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from unittest.mock import patch

import pytest

from config.constants import LOGGER_APP
from core.trading_stream import StreamHealth
from utils import stream_alerts


def _stale_stream_health() -> StreamHealth:
    h = StreamHealth()
    h.set_trading_ok(True)
    h.set_market_ok(True)
    with h._lock:
        h._last_quote_ts = datetime.now(timezone.utc) - timedelta(seconds=60)
        h._last_order_event_ts = None
    return h


def test_websocket_stale_triggers_notify_once_then_rate_limited(
    make_settings_factory,
) -> None:
    stream_alerts.reset_notification_cooldown()
    settings = make_settings_factory(
        ENABLE_LOCAL_NOTIFICATIONS=True,
        STREAM_STALE_SECONDS=30.0,
        STREAM_NOTIFICATION_COOLDOWN_SECONDS=300.0,
    )
    log = logging.getLogger(LOGGER_APP)
    health = _stale_stream_health()

    with patch("utils.stream_alerts.notify_user") as nn:
        with patch(
            "utils.stream_alerts.time.monotonic",
            side_effect=[1000.0, 1000.1, 1310.0],
        ):
            stream_alerts.evaluate_stream_websocket_notifications(
                settings=settings, stream_health=health, log=log,
            )
            assert nn.call_count == 1

            stream_alerts.evaluate_stream_websocket_notifications(
                settings=settings, stream_health=health, log=log,
            )
            assert nn.call_count == 1

            stream_alerts.evaluate_stream_websocket_notifications(
                settings=settings, stream_health=health, log=log,
            )
            assert nn.call_count == 2

    stream_alerts.reset_notification_cooldown()
