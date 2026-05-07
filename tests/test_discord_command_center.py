"""Discord command center — short-circuit paths and /report text (no live socket)."""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

import pytest

from communication.discord_client import (
    DiscordCallbacks,
    DiscordCommandCenter,
    enqueue_discord_alert,
    format_report,
)


@pytest.mark.asyncio
async def test_discord_drive_short_circuits_without_token(monkeypatch) -> None:
    from communication import discord_client as dc

    s = MagicMock()
    s.ENABLE_DISCORD_BOT = True
    s.DISCORD_BOT_TOKEN = ""
    s.DISCORD_CHANNEL_ID = "999"
    s.discord_allowed_user_ids_set = set()
    s.DISCORD_COMMAND_RATE_LIMIT_SECONDS = 5.0
    warns: list[str] = []

    monkeypatch.setattr(dc._LOG, "warning", lambda m, *a, **k: warns.append(str(m)))

    cb = DiscordCallbacks(lambda: "a", lambda: "b", MagicMock(), lambda _x: None)
    q: asyncio.Queue = asyncio.Queue()
    await DiscordCommandCenter(s, cb).run(q)
    assert warns
    assert any("discord_init_failed" in str(w) for w in warns)


@pytest.mark.asyncio
async def test_remote_kill_callback_contract() -> None:
    called = asyncio.Event()

    async def latch() -> None:
        called.set()

    cb = DiscordCallbacks(lambda: "", lambda: "", latch, lambda _x: None)
    await cb.kill_fn()
    assert called.is_set()


def test_enqueue_noop_when_queue_none() -> None:
    enqueue_discord_alert(None, {"title": "X", "lines": ["a"]})


def test_format_report_contains_day_and_counts(make_settings_factory, monkeypatch) -> None:
    class FakeDb:
        def get_completed_trades_for_calendar_day_et(self, **_k):
            return [{"realized_pnl": 10.0}, {"realized_pnl": -4.0}]

    orch = MagicMock()
    orch._settings = make_settings_factory(DYNAMIC_PARAMS_PATH="src/config/dynamic_params.json")
    orch._database = FakeDb()
    orch._latest_positions = []
    monkeypatch.setattr("communication.discord_client.today_eastern", lambda: date(2025, 5, 6))

    txt = format_report(orch)
    assert "2025-05-06" in txt
    assert "closed_trades=2" in txt
