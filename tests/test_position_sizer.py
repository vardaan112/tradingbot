"""Tests for the ATR position sizer."""

from __future__ import annotations

import pytest

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


def test_conviction_multiplier_scales_risk_budget(make_settings_factory):
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=10_000.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000_000.0,
        MAX_GROSS_EXPOSURE_PCT=2.0,
        MAX_OPEN_POSITIONS=5,
    )
    sizer = _build(settings)
    account = make_account(equity=100_000.0, buying_power=1_000_000.0, regt=1_000_000.0)

    base = sizer.size(
        symbol="SPY",
        entry_price=50.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.0,
    )
    hi = sizer.size(
        symbol="SPY",
        entry_price=50.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.5,
    )
    lo = sizer.size(
        symbol="SPY",
        entry_price=50.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=0.5,
    )
    assert base.shares > 0 and hi.shares > 0 and lo.shares > 0
    assert hi.shares > base.shares > lo.shares
    assert hi.effective_risk_pct == pytest.approx(0.015)
    assert lo.effective_risk_pct == pytest.approx(0.005)
    assert "conv_mult=" in hi.rationale


def test_tight_usd_cap_dominates_conviction_multipliers(make_settings_factory):
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=1_000_000.0,
        MAX_RISK_PER_TRADE_PCT=0.05,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=500.0,
        MAX_GROSS_EXPOSURE_PCT=2.0,
        MAX_OPEN_POSITIONS=5,
    )
    sizer = _build(settings)
    account = make_account(equity=100_000.0, buying_power=1_000_000.0)

    a = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.0,
    )
    b = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.5,
    )
    assert a.shares == b.shares == 5.0
    assert "usd_cap" in a.rationale or "usd_cap" in b.rationale


def test_sizing_block_reason_returns_zero_shares(settings):
    sizer = _build(settings)
    account = make_account()
    reason = "correlation_breaker_leader_SPY_follower_QQQ"
    out = sizer.size(
        symbol="QQQ",
        entry_price=100.0,
        atr=2.0,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
        sizing_block_reason=reason,
    )
    assert out.shares == 0
    assert out.skipped_reason == reason


def test_size_zero_reason_is_explicit_when_price_above_cap_and_no_fractionals(make_settings_factory):
    settings = make_settings_factory(
        MAX_EQUITY_USAGE_USD=50.0,
        MAX_DOLLARS_PER_TRADE=50.0,
        ENABLE_FRACTIONAL=False,
        MIN_SHARES=1.0,
    )
    sizer = _build(settings)
    account = make_account(equity=25_000.0, buying_power=25_000.0, regt=25_000.0)
    out = sizer.size(
        symbol="IWM",
        entry_price=237.41,
        atr=1.5,
        account=account,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert out.shares == 0
    assert out.skipped_reason is not None
    assert "SIZE_ZERO" in out.skipped_reason
    assert "fractional_enabled=false" in out.skipped_reason
