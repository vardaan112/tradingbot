"""Tests for the live canary check.

All tests run fully offline. Real Alpaca clients are never constructed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from core.account import AccountSnapshot, PositionSnapshot
from core.exceptions import KillSwitchLatchedError
from core.market_data import Quote
from core.state_store import KillSwitchRecord, StateStore
from risk.compliance import ComplianceAdapter, ComplianceDecision
from risk.killswitch import KillSwitch
from services.canary import (
    CANARY_STRATEGY_NAME,
    CanaryPersistedRecord,
    CanaryService,
    canary_already_succeeded_today,
    canary_state_path,
    maybe_run_canary,
    persist_canary_record,
)
from utils.time_utils import today_eastern


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeOrder:
    def __init__(self, *, status: str, filled_qty: float = 0.0, filled_avg_price: float = 0.0,
                 broker_id: str = "broker-id-1") -> None:
        self.id = broker_id
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class _FakeTradingClient:
    """Minimal stub for the canary's expected TradingClient surface."""

    def __init__(self, *, fractionable: bool = True, tradable: bool = True,
                 reject_fractional_limit: bool = False) -> None:
        self._fractionable = fractionable
        self._tradable = tradable
        self._reject_fractional = reject_fractional_limit
        self.submitted: list[dict] = []
        self._next_buy_status: list[_FakeOrder] = []
        self._next_sell_status: list[_FakeOrder] = []

    def queue_buy_states(self, states: list[_FakeOrder]) -> None:
        self._next_buy_status = list(states)

    def queue_sell_states(self, states: list[_FakeOrder]) -> None:
        self._next_sell_status = list(states)

    def get_asset(self, symbol: str):
        return SimpleNamespace(tradable=self._tradable, fractionable=self._fractionable)

    def submit_order(self, request):
        side_raw = str(request.side).lower()
        # alpaca-py OrderSide stringifies as "OrderSide.BUY"; normalize to bare side.
        side = side_raw.split(".")[-1] if "." in side_raw else side_raw
        is_fractional = float(request.qty) != int(request.qty)
        if is_fractional and self._reject_fractional:
            raise RuntimeError("fractional orders are not allowed for limit orders")
        record = {
            "side": side,
            "qty": float(request.qty),
            "limit_price": float(request.limit_price),
            "tif": str(request.time_in_force),
            "client_order_id": request.client_order_id,
        }
        self.submitted.append(record)
        return SimpleNamespace(id=f"broker-{record['client_order_id']}", status="accepted")

    def get_order_by_id(self, broker_id: str):
        # Return the next queued buy/sell state in FIFO; we route by 'broker-' prefix.
        # Simpler: pop from buy queue first while it has items, then sell queue.
        if self._next_buy_status:
            return self._next_buy_status.pop(0)
        if self._next_sell_status:
            return self._next_sell_status.pop(0)
        return _FakeOrder(status="filled", filled_qty=0.0, filled_avg_price=0.0)

    def get_order_by_client_id(self, coid: str):
        return self.get_order_by_id(coid)

    def cancel_order_by_id(self, broker_id: str) -> None:
        return None


class _FakeAccountAdapter:
    def __init__(self, account: AccountSnapshot, positions: list[PositionSnapshot] | None = None,
                 open_orders: list | None = None) -> None:
        self._account = account
        self._positions = list(positions or [])
        self._open_orders = list(open_orders or [])

    def fetch_account(self) -> AccountSnapshot:
        return self._account

    def fetch_positions(self) -> list[PositionSnapshot]:
        return list(self._positions)

    def fetch_open_orders(self):
        return list(self._open_orders)


class _FakeBarFetcher:
    def __init__(self, quote: Quote) -> None:
        self._quote = quote

    def fetch_latest_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=self._quote.symbol,
            bid=self._quote.bid,
            ask=self._quote.ask,
            bid_size=self._quote.bid_size,
            ask_size=self._quote.ask_size,
            timestamp=datetime.now(timezone.utc),
            feed=self._quote.feed,
        )


class _FakeMarketClock:
    def __init__(self, *, is_open: bool = True, can_open: bool = True) -> None:
        self._is_open = is_open
        self._can_open = can_open

    def get_session(self, *, force_refresh: bool = False):
        from core.market_clock import MarketSession
        return MarketSession(
            is_open=self._is_open,
            next_open_utc=None,
            next_close_utc=None,
            fetched_at_utc=datetime.now(timezone.utc),
        )

    def can_open_new_position(self, session) -> bool:
        return self._can_open

    def can_exit_position(self, session) -> bool:
        return True


class _FakeCompliance(ComplianceAdapter):
    def __init__(self, settings, *, allow: bool = True) -> None:
        super().__init__(settings)
        self._allow = allow

    def decide(self, account, *, reference_date=None) -> ComplianceDecision:
        return ComplianceDecision(
            allow_new_entries=self._allow,
            effective_mode="intraday_margin",
            reason="ok" if self._allow else "blocked",
            scaling_relaxation_allowed=False,
        )


def _account_snapshot(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=equity,
        regt_buying_power=equity,
        portfolio_value=equity,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=2.0,
        status="ACTIVE",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
    )


def _quote(symbol: str = "XLF", bid: float = 39.99, ask: float = 40.00) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=100,
        ask_size=100,
        timestamp=datetime.now(timezone.utc),
        feed="iex",
    )


def _build_settings(make_settings_factory, tmp_path: Path, **overrides):
    base = {
        "STATE_DIR": str(tmp_path),
        "ALPACA_ENV": "live",
        "LIVE_TRADING_ENABLED": True,
        "DRY_RUN": False,
        "CONFIRM_LIVE_TRADING": "yes_i_understand",
        "RUN_LIVE_CANARY_ON_STARTUP": True,
        "CANARY_SYMBOL": "XLF",
        "CANARY_NOTIONAL_USD": 10.0,
        "CANARY_TIMEOUT_SECONDS": 2.0,
        "SPREAD_FILTER_PCT": 0.01,
    }
    base.update(overrides)
    return make_settings_factory(**base)


# ---------------------------------------------------------------------------
# Gating: maybe_run_canary
# ---------------------------------------------------------------------------


def test_maybe_run_canary_noop_in_paper(make_settings_factory, tmp_path):
    settings = make_settings_factory(
        STATE_DIR=str(tmp_path),
        ALPACA_ENV="paper",
        LIVE_TRADING_ENABLED=True,
        DRY_RUN=False,
        RUN_LIVE_CANARY_ON_STARTUP=True,
    )
    ok = asyncio.run(maybe_run_canary(settings))
    assert ok is True


def test_maybe_run_canary_noop_in_dry_run(make_settings_factory, tmp_path):
    settings = make_settings_factory(
        STATE_DIR=str(tmp_path),
        ALPACA_ENV="live",
        LIVE_TRADING_ENABLED=True,
        DRY_RUN=True,
        CONFIRM_LIVE_TRADING="yes_i_understand",
        RUN_LIVE_CANARY_ON_STARTUP=True,
    )
    ok = asyncio.run(maybe_run_canary(settings))
    assert ok is True


def test_maybe_run_canary_noop_when_disabled(make_settings_factory, tmp_path):
    settings = make_settings_factory(
        STATE_DIR=str(tmp_path),
        ALPACA_ENV="live",
        LIVE_TRADING_ENABLED=True,
        DRY_RUN=False,
        CONFIRM_LIVE_TRADING="yes_i_understand",
        RUN_LIVE_CANARY_ON_STARTUP=False,
    )
    ok = asyncio.run(maybe_run_canary(settings))
    assert ok is True


def test_maybe_run_canary_noop_when_live_disabled(make_settings_factory, tmp_path):
    settings = make_settings_factory(
        STATE_DIR=str(tmp_path),
        ALPACA_ENV="live",
        LIVE_TRADING_ENABLED=False,
        DRY_RUN=False,
        RUN_LIVE_CANARY_ON_STARTUP=True,
    )
    ok = asyncio.run(maybe_run_canary(settings))
    assert ok is True


def test_maybe_run_canary_aborts_when_kill_switch_latched(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path)
    state = StateStore(Path(settings.STATE_DIR))
    state.save_kill_switch(KillSwitchRecord(latched=True, reason="prior_drawdown"))
    with pytest.raises(KillSwitchLatchedError):
        asyncio.run(maybe_run_canary(settings))


def test_maybe_run_canary_skipped_after_today_success(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path)
    persist_canary_record(
        settings,
        CanaryPersistedRecord(
            date=today_eastern().isoformat(),
            success=True,
            completed_at=datetime.now(timezone.utc).isoformat(),
            symbol="XLF",
            notional_attempted=10.0,
            fractional=False,
        ),
    )
    assert canary_already_succeeded_today(settings) is True
    # Even with all live gates enabled, the canary should skip.
    ok = asyncio.run(maybe_run_canary(settings))
    assert ok is True


# ---------------------------------------------------------------------------
# CanaryService.run direct tests (no client construction)
# ---------------------------------------------------------------------------


def _make_service(
    *,
    settings,
    state_store: StateStore,
    quote: Quote,
    is_open: bool = True,
    can_open: bool = True,
    fractionable: bool = True,
    tradable: bool = True,
    reject_fractional_limit: bool = False,
    positions: Optional[list[PositionSnapshot]] = None,
    open_orders: Optional[list] = None,
    compliance_allow: bool = True,
) -> tuple[CanaryService, _FakeTradingClient]:
    client = _FakeTradingClient(
        fractionable=fractionable,
        tradable=tradable,
        reject_fractional_limit=reject_fractional_limit,
    )
    account_adapter = _FakeAccountAdapter(
        account=_account_snapshot(),
        positions=positions or [],
        open_orders=open_orders or [],
    )
    bar_fetcher = _FakeBarFetcher(quote=quote)
    market_clock = _FakeMarketClock(is_open=is_open, can_open=can_open)
    compliance = _FakeCompliance(settings, allow=compliance_allow)
    kill = KillSwitch(state_store, drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
    service = CanaryService(
        settings,
        trading=client,  # type: ignore[arg-type]
        account_adapter=account_adapter,  # type: ignore[arg-type]
        bar_fetcher=bar_fetcher,  # type: ignore[arg-type]
        market_clock=market_clock,  # type: ignore[arg-type]
        kill_switch=kill,
        compliance=compliance,
    )
    return service, client


def test_canary_aborts_when_fractionable_unavailable(make_settings_factory, tmp_path):
    """If $10 < share price AND asset is not fractionable, the canary aborts."""
    settings = _build_settings(make_settings_factory, tmp_path, CANARY_NOTIONAL_USD=10.0)
    state = StateStore(Path(settings.STATE_DIR))
    # Share price $40 -> $10 < $40 forces the fractional path.
    quote = _quote(bid=39.99, ask=40.00)
    service, _client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
        fractionable=False,
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert result.reason == "asset_not_fractionable"


def test_canary_whole_share_round_trip_success(make_settings_factory, tmp_path):
    """Whole-share path: $50 notional > $40 share -> 1 share buy/sell, both fill."""
    settings = _build_settings(make_settings_factory, tmp_path, CANARY_NOTIONAL_USD=50.0)
    state = StateStore(Path(settings.STATE_DIR))
    quote = _quote(bid=39.99, ask=40.00)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
        fractionable=False,  # whole-share path doesn't need fractionable
    )
    client.queue_buy_states([
        _FakeOrder(status="filled", filled_qty=1.0, filled_avg_price=40.00),
    ])
    client.queue_sell_states([
        _FakeOrder(status="filled", filled_qty=1.0, filled_avg_price=39.99),
    ])
    result = asyncio.run(service.run())
    assert result.success is True
    assert result.buy_filled_qty == 1.0
    assert result.sell_filled_qty == 1.0
    assert result.fractional is False
    assert len(client.submitted) == 2  # one buy, one sell
    assert client.submitted[0]["side"] == "buy"
    assert client.submitted[1]["side"] == "sell"


def test_canary_aborts_when_kill_switch_latched(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path)
    state = StateStore(Path(settings.STATE_DIR))
    state.save_kill_switch(KillSwitchRecord(latched=True, reason="prior_drawdown"))
    quote = _quote(bid=39.99, ask=40.00)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert result.reason == "kill_switch_latched"
    assert client.submitted == []


def test_canary_aborts_when_market_closed(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path)
    state = StateStore(Path(settings.STATE_DIR))
    quote = _quote(bid=39.99, ask=40.00)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
        is_open=False,
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert result.reason == "market_closed"
    assert client.submitted == []


def test_canary_aborts_when_existing_position(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path, CANARY_NOTIONAL_USD=50.0)
    state = StateStore(Path(settings.STATE_DIR))
    existing = PositionSnapshot(
        symbol="XLF",
        qty=5,
        avg_entry_price=40.0,
        side="long",
        market_value=200.0,
        cost_basis=200.0,
        unrealized_pl=0.0,
        current_price=40.0,
    )
    quote = _quote(bid=39.99, ask=40.00)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
        positions=[existing],
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert result.reason == "existing_position_in_canary_symbol"
    assert client.submitted == []


def test_canary_aborts_when_spread_too_wide(make_settings_factory, tmp_path):
    settings = _build_settings(
        make_settings_factory, tmp_path,
        CANARY_NOTIONAL_USD=50.0,
        SPREAD_FILTER_PCT=0.0001,  # 1 bp threshold
    )
    state = StateStore(Path(settings.STATE_DIR))
    # 200 bps spread -> should be rejected
    quote = _quote(bid=39.0, ask=40.0)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert "spread_too_wide" in result.reason
    assert client.submitted == []


def test_canary_persistence_round_trip(make_settings_factory, tmp_path):
    settings = _build_settings(make_settings_factory, tmp_path)
    record = CanaryPersistedRecord(
        date=today_eastern().isoformat(),
        success=True,
        completed_at=datetime.now(timezone.utc).isoformat(),
        symbol="XLF",
        notional_attempted=10.0,
        fractional=False,
    )
    persist_canary_record(settings, record)
    p = canary_state_path(settings)
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["success"] is True
    assert data["symbol"] == "XLF"
    assert canary_already_succeeded_today(settings) is True


def test_canary_fractional_limit_rejection_aborts(make_settings_factory, tmp_path):
    """When a fractional path is required but the broker rejects fractional limit,
    the canary must abort with a clear reason rather than silently fall back."""
    settings = _build_settings(
        make_settings_factory, tmp_path,
        CANARY_NOTIONAL_USD=10.0,  # < $40 share -> fractional path forced
    )
    state = StateStore(Path(settings.STATE_DIR))
    quote = _quote(bid=39.99, ask=40.00)
    service, client = _make_service(
        settings=settings,
        state_store=state,
        quote=quote,
        fractionable=True,
        reject_fractional_limit=True,
    )
    result = asyncio.run(service.run())
    assert result.success is False
    assert result.reason == "fractional_limit_rejected_by_broker"
    # The buy was attempted but never accepted by the broker.
    assert len(client.submitted) == 0
