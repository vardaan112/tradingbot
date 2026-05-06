"""Tests for the regulatory compliance adapter (FINRA Rule 4210 transition)."""

from __future__ import annotations

from datetime import date

from core.account import AccountSnapshot
from risk.compliance import ComplianceAdapter, EffectiveRegulatoryMode


def _account(**overrides) -> AccountSnapshot:
    base = dict(
        equity=100_000.0,
        last_equity=100_000.0,
        cash=100_000.0,
        buying_power=200_000.0,
        regt_buying_power=200_000.0,
        portfolio_value=100_000.0,
        long_market_value=0.0,
        short_market_value=0.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        multiplier=2.0,
        status="ACTIVE",
        trading_blocked=False,
        transfers_blocked=False,
        account_blocked=False,
        pattern_day_trader=False,
        daytrade_count=0,
        daytrading_buying_power=400_000.0,
    )
    base.update(overrides)
    return AccountSnapshot(**base)


def test_mode_resolution_pdt_before_effective_date(settings):
    adapter = ComplianceAdapter(settings)
    mode = adapter.effective_mode(reference_date=date(2026, 6, 3))
    assert mode == EffectiveRegulatoryMode.PDT


def test_mode_resolution_intraday_on_or_after_effective_date(settings):
    adapter = ComplianceAdapter(settings)
    mode_on = adapter.effective_mode(reference_date=date(2026, 6, 4))
    mode_after = adapter.effective_mode(reference_date=date(2026, 6, 5))
    assert mode_on == EffectiveRegulatoryMode.INTRADAY_MARGIN
    assert mode_after == EffectiveRegulatoryMode.INTRADAY_MARGIN


def test_explicit_pdt_mode_locks_pdt(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="pdt")
    adapter = ComplianceAdapter(settings)
    assert adapter.effective_mode(reference_date=date(2027, 1, 1)) == EffectiveRegulatoryMode.PDT


def test_explicit_intraday_mode_locks_intraday(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="intraday_margin")
    adapter = ComplianceAdapter(settings)
    assert (
        adapter.effective_mode(reference_date=date(2024, 1, 1))
        == EffectiveRegulatoryMode.INTRADAY_MARGIN
    )


def test_buying_power_intraday_uses_buying_power_field(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="intraday_margin")
    adapter = ComplianceAdapter(settings)
    account = _account(buying_power=150_000.0, daytrading_buying_power=999_999.0)
    bp = adapter.buying_power(account, reference_date=date(2026, 7, 1))
    # Must NOT use daytrading_buying_power post-rule.
    assert bp == 150_000.0


def test_buying_power_pdt_takes_min_of_buying_power_and_regt(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="pdt")
    adapter = ComplianceAdapter(settings)
    account = _account(buying_power=200_000.0, regt_buying_power=120_000.0)
    bp = adapter.buying_power(account, reference_date=date(2026, 1, 1))
    assert bp == 120_000.0


def test_decide_blocks_when_account_blocked(settings):
    adapter = ComplianceAdapter(settings)
    account = _account(account_blocked=True)
    decision = adapter.decide(account)
    assert not decision.allow_new_entries
    assert "block" in decision.reason


def test_decide_pdt_throttle_on_high_daytrade_count(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="pdt", POST_RULE4210_SCALING_ENABLED=False)
    adapter = ComplianceAdapter(settings)
    account = _account(daytrade_count=4)
    decision = adapter.decide(account, reference_date=date(2026, 1, 1))
    assert not decision.allow_new_entries
    assert "pdt_daytrade" in decision.reason


def test_decide_pdt_throttle_can_be_relaxed_only_via_scaling_flag(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="pdt", POST_RULE4210_SCALING_ENABLED=True)
    adapter = ComplianceAdapter(settings)
    account = _account(daytrade_count=4)
    decision = adapter.decide(account, reference_date=date(2026, 1, 1))
    assert decision.allow_new_entries
    assert decision.scaling_relaxation_allowed


def test_decide_intraday_blocks_zero_buying_power(make_settings_factory):
    settings = make_settings_factory(REGULATORY_MODE="intraday_margin")
    adapter = ComplianceAdapter(settings)
    account = _account(buying_power=0.0)
    decision = adapter.decide(account, reference_date=date(2026, 7, 1))
    assert not decision.allow_new_entries
    assert decision.reason == "zero_buying_power"
