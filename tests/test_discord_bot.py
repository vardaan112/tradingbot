"""Discord helpers — mocked (no real Discord connection)."""

from __future__ import annotations

from unittest.mock import MagicMock

from core.skiplist import SymbolSkiplist
from services.discord_bot import enqueue_discord_alert, format_status


def test_enqueue_noop_when_queue_none() -> None:
    enqueue_discord_alert(None, {"title": "X", "lines": ["a"]})


def test_format_status_excludes_credentials(make_settings_factory) -> None:
    class O:
        _settings = make_settings_factory(ALPACA_API_KEY="xk", ALPACA_API_SECRET="ys")
        _kill_switch = MagicMock()
        _kill_switch.is_latched = lambda: False
        _black_swan = MagicMock()
        _black_swan.triggered = lambda: False
        _stream_health = MagicMock()
        _stream_health.all_ok = True
        _database = MagicMock()
        _database.sum_realized_pnl_all_live = lambda: 0.0
        _latest_positions = []
        _ml_filter = None

    blob = format_status(O())
    assert "xk" not in blob and "ys" not in blob


def test_skiplist_day_expiry_boundary(tmp_path) -> None:
    sk = SymbolSkiplist(tmp_path)
    sk.skip_for_session_day(session_day_et="2024-06-03", symbol="spy")
    assert sk.is_skipped(session_day_et="2024-06-03", symbol="SPY")
    assert not sk.is_skipped(session_day_et="2024-06-04", symbol="spy")
