"""Tests for the ATR position sizer."""

from __future__ import annotations

from core.account import AccountSnapshot
from risk.compliance import ComplianceAdapter
from risk.exposure import ExposureChecker
from risk.position_sizer import PositionSizer


def make_account(equity=100_000.0, buying_power=400_000.0, regt=200_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=buying_power,
        regt_buying_power=regt,
        portfolio_value=equity,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=4.0,
        status="ACTIVE",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
    )


def _build(settings):
    compliance = ComplianceAdapter(settings)
    exposure = ExposureChecker(settings)
    return PositionSizer(settings, compliance, exposure)


def test_size_clamped_by_max_equity_usage_usd(settings):
    sizer = _build(settings)
    account = make_account()
    result = sizer.size(
        symbol="AAPL",
        entry_price=100.0,
        atr=2.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    # MAX_EQUITY_USAGE_USD=500 -> at $100 entry that's <= 5 shares.
    assert 0 < result.shares <= 5
    assert result.notional <= 500.0 + 1e-9


def test_size_zero_on_invalid_inputs(settings):
    sizer = _build(settings)
    account = make_account()
    result = sizer.size(
        symbol="AAPL",
        entry_price=0.0,
        atr=2.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert result.shares == 0
    assert result.skipped_reason is not None


def test_risk_budget_caps_shares(make_settings_factory):
    """At very high entry price the risk-budget cap should be hit before usd cap."""
    settings = make_settings_factory(MAX_EQUITY_USAGE_USD=1_000_000.0, MAX_RISK_PER_TRADE_PCT=0.001)
    sizer = _build(settings)
    account = make_account(equity=100_000.0)
    # risk_budget = 100k * 0.1% = $100; stop_distance = 2 * 5 = $10 -> 10 shares.
    result = sizer.size(
        symbol="AAPL",
        entry_price=200.0,
        atr=5.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert result.shares == 10


def test_size_skips_when_floor_below_one(settings):
    sizer = _build(settings)
    account = make_account()
    # Very small risk budget vs huge stop distance -> < 1 share.
    result = sizer.size(
        symbol="AAPL",
        entry_price=10000.0,
        atr=500.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert result.shares == 0
    assert result.skipped_reason is not None


def test_buying_power_cap_respected(make_settings_factory):
    """When buying power is tiny, sizer should clamp to it."""
    settings = make_settings_factory(MAX_EQUITY_USAGE_USD=1_000_000.0, MAX_RISK_PER_TRADE_PCT=0.05)
    sizer = _build(settings)
    account = make_account(equity=100_000.0, buying_power=300.0, regt=300.0)
    result = sizer.size(
        symbol="AAPL",
        entry_price=100.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    # buying_power 300 / entry 100 -> floor(3) = 3 shares cap.
    assert result.shares == 3
