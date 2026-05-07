"""Passive joiner / chase helpers (broker-free stubs)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.market_data import Quote, QuoteCache
from core.orders import OrderService
from core.state_store import StateStore


class _StubBroker:
    def submit_order(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("broker should not be called in offline tests")

    def get_order_by_client_id(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("broker lookup should not be called in offline tests")


def _fresh_quote(sym: str = "SPY") -> Quote:
    return Quote(
        symbol=sym,
        bid=100.0,
        ask=100.08,
        bid_size=1.0,
        ask_size=1.0,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


@pytest.mark.asyncio
async def test_chase_giveup_on_stale_quote(make_settings_factory, tmp_path: Path) -> None:
    settings = make_settings_factory(
        PASSIVE_JOINER_ENABLED=True,
        PASSIVE_JOINER_REQUIRE_FRESH_QUOTE=True,
        QUOTE_STALENESS_SECONDS=5.0,
        SPREAD_FILTER_PCT=0.01,
        DRY_RUN=True,
        LIVE_TRADING_ENABLED=False,
        STATE_DIR=str(tmp_path / "st"),
        LOG_DIR=str(tmp_path / "logs"),
        DATABASE_PATH=tmp_path / "db.sqlite",
    )
    qc = QuoteCache(max_age_seconds=5.0, feed="iex")
    state = StateStore(tmp_path / "state")
    svc = OrderService(_StubBroker(), settings, state, qc, strategy_name="test")

    old = datetime.now(timezone.utc) - timedelta(seconds=3600)

    stale = Quote("SPY", 100.0, 100.08, 1.0, 1.0, old, "iex")

    wo = await svc.submit_buy_passive_joiner_async(
        "spy",
        1,
        quote_refresher=lambda: stale,
    )
    assert wo is None


@pytest.mark.asyncio
async def test_chase_dry_run_first_attempt(make_settings_factory, tmp_path: Path) -> None:
    settings = make_settings_factory(
        PASSIVE_JOINER_ENABLED=True,
        PASSIVE_JOINER_REQUIRE_FRESH_QUOTE=True,
        PASSIVE_JOINER_TIMEOUT_SECONDS=0.1,
        QUOTE_STALENESS_SECONDS=120.0,
        SPREAD_FILTER_PCT=0.01,
        DRY_RUN=True,
        LIVE_TRADING_ENABLED=False,
        STATE_DIR=str(tmp_path / "st2"),
        LOG_DIR=str(tmp_path / "logs2"),
        DATABASE_PATH=tmp_path / "db2.sqlite",
    )
    qc = QuoteCache(max_age_seconds=120.0, feed="iex")
    state = StateStore(tmp_path / "state2")
    svc = OrderService(_StubBroker(), settings, state, qc, strategy_name="test")
    q = _fresh_quote()
    wo = await svc.submit_buy_passive_joiner_async(
        "SPY",
        1,
        quote_refresher=lambda: q,
    )
    assert wo is not None
    assert wo.status == "dry_run"


@pytest.mark.asyncio
async def test_passive_disabled_delegates_to_limit_path(make_settings_factory, tmp_path: Path) -> None:
    settings = make_settings_factory(
        PASSIVE_JOINER_ENABLED=False,
        QUOTE_STALENESS_SECONDS=120.0,
        SPREAD_FILTER_PCT=0.01,
        DRY_RUN=True,
        LIVE_TRADING_ENABLED=False,
        STATE_DIR=str(tmp_path / "st3"),
        LOG_DIR=str(tmp_path / "logs3"),
        DATABASE_PATH=tmp_path / "db3.sqlite",
    )
    qc = QuoteCache(max_age_seconds=120.0, feed="iex")
    state = StateStore(tmp_path / "state3")
    svc = OrderService(_StubBroker(), settings, state, qc, strategy_name="test")
    q = _fresh_quote()
    wo = await svc.submit_buy_passive_joiner_async(
        "SPY",
        1,
        quote_refresher=lambda: q,
    )
    assert wo is not None
    assert wo.status == "dry_run"
