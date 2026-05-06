"""Tests for the BOT_CAPITAL_BASE_USD-aware position sizer."""

from __future__ import annotations

from core.account import AccountSnapshot, PositionSnapshot
from risk.compliance import ComplianceAdapter
from risk.exposure import ExposureChecker
from risk.position_sizer import PositionSizer


def _account(equity: float = 100_000.0, buying_power: float = 1_000_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        last_equity=equity,
        cash=equity,
        buying_power=buying_power,
        regt_buying_power=buying_power,
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


def _build_sizer(settings) -> PositionSizer:
    return PositionSizer(settings, ComplianceAdapter(settings), ExposureChecker(settings))


def test_capital_base_overrides_full_account_equity(make_settings_factory):
    """When BOT_CAPITAL_BASE_USD is set, risk_budget uses it - not equity.

    risk_budget = capital_base * MAX_RISK_PER_TRADE_PCT
    With capital_base=$1000 and MAX_RISK_PER_TRADE_PCT=0.01 -> risk_budget=$10
    With ATR_STOP_MULTIPLIER=1.0 and atr=$1 -> stop_distance=$1
    raw_shares = 10 / 1 = 10 shares
    """
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=1000.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000_000.0,  # USD cap not the binding constraint
        MAX_GROSS_EXPOSURE_PCT=2.0,
    )
    sizer = _build_sizer(settings)
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=1_000_000.0),  # huge equity that should NOT inflate sizing
        positions=[],
        bot_managed_notional=0.0,
    )
    assert result.shares == 10
    assert abs(result.risk_budget - 10.0) < 1e-9
    assert abs(result.capital_base - 1000.0) < 1e-9


def test_capital_base_falls_back_to_min_equity_max_usage(make_settings_factory):
    """When BOT_CAPITAL_BASE_USD is 0, capital_base = min(equity, MAX_EQUITY_USAGE_USD)."""
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=0.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        MAX_EQUITY_USAGE_USD=500.0,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_GROSS_EXPOSURE_PCT=2.0,
    )
    sizer = _build_sizer(settings)
    # Equity 100k, MAX_EQUITY_USAGE_USD 500 -> capital_base = 500.
    # risk_budget = 500 * 0.01 = 5; stop_distance = 1*1 = 1; raw_shares = 5.
    # But MAX_EQUITY_USAGE_USD also caps directly: usd_cap_shares=500/100=5 -> still 5.
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=100_000.0),
        positions=[],
        bot_managed_notional=0.0,
    )
    assert abs(result.capital_base - 500.0) < 1e-9
    assert result.shares == 5


def test_capital_base_falls_back_to_equity_when_smaller(make_settings_factory):
    """If equity is below MAX_EQUITY_USAGE_USD, capital_base equals equity."""
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=0.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        MAX_EQUITY_USAGE_USD=10_000.0,
        ATR_STOP_MULTIPLIER=1.0,
    )
    sizer = _build_sizer(settings)
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=2_000.0, buying_power=2_000.0),
        positions=[],
        bot_managed_notional=0.0,
    )
    assert abs(result.capital_base - 2_000.0) < 1e-9


def test_buying_power_clamp_still_respected_with_capital_base(make_settings_factory):
    """Tight buying power must still clamp shares regardless of capital base."""
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=1_000_000.0,  # huge -> would otherwise allow many shares
        MAX_RISK_PER_TRADE_PCT=0.05,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000_000.0,
        MAX_GROSS_EXPOSURE_PCT=2.0,
    )
    sizer = _build_sizer(settings)
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=100_000.0, buying_power=300.0),
        positions=[],
        bot_managed_notional=0.0,
    )
    # buying_power 300 / 100 -> floor 3 shares.
    assert result.shares == 3
    assert "buying_power" in result.rationale


def test_gross_exposure_clamp_still_respected_with_capital_base(make_settings_factory):
    """Existing gross exposure must still cap incremental shares."""
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=1_000_000.0,
        MAX_RISK_PER_TRADE_PCT=0.05,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000_000.0,
        MAX_GROSS_EXPOSURE_PCT=0.6,  # 60% of 100k = $60k cap
        MAX_OPEN_POSITIONS=5,  # allow the additional position
    )
    sizer = _build_sizer(settings)
    existing_pos = PositionSnapshot(
        symbol="OTHER",
        qty=0,
        avg_entry_price=0.0,
        side="long",
        market_value=58_000.0,  # already 58k of 60k cap used
        cost_basis=58_000.0,
        unrealized_pl=0.0,
        current_price=0.0,
    )
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=100_000.0, buying_power=1_000_000.0),
        positions=[existing_pos],
        bot_managed_notional=0.0,
    )
    # Remaining gross headroom: 60k - 58k = 2k -> 20 shares at $100. But the
    # raw_atr risk bound at $1M base * 5% / 1 = 50k shares; so the gross-exposure
    # clamp wins.
    assert result.shares == 20
    assert "gross_exposure" in result.rationale


def test_bot_managed_remaining_clamp_still_respected(make_settings_factory):
    """Bot-managed notional already in use must still clamp the next size."""
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=1_000_000.0,
        MAX_RISK_PER_TRADE_PCT=0.05,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000.0,  # tight
        MAX_GROSS_EXPOSURE_PCT=2.0,
    )
    sizer = _build_sizer(settings)
    result = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=1.0,
        account=_account(equity=100_000.0, buying_power=1_000_000.0),
        positions=[],
        bot_managed_notional=900.0,  # leaves $100 headroom -> 1 share at $100
    )
    assert result.shares == 1
    assert "bot_managed_remaining" in result.rationale or "usd_cap" in result.rationale
