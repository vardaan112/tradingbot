"""Modified Kelly capped sizing (SQLite pnls mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.account import AccountSnapshot
from risk.compliance import ComplianceAdapter
from risk.exposure import ExposureChecker
from risk.position_sizer import PositionSizer, compute_kelly_risk_scaling


def make_account(**kw) -> AccountSnapshot:
    return AccountSnapshot(
        equity=kw.get("equity", 100_000.0),
        last_equity=kw.get("last_equity", 100_000.0),
        cash=kw.get("cash", 100_000.0),
        buying_power=kw.get("buying_power", 400_000.0),
        regt_buying_power=kw.get("regt_buying_power", 200_000.0),
        portfolio_value=kw.get("portfolio_value", 100_000.0),
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


def test_kelly_falls_back_with_insufficient_trades(make_settings_factory) -> None:
    settings = make_settings_factory(
        ENABLE_KELLY_SIZING=True,
        KELLY_MIN_TRADES=30,
        KELLY_LOOKBACK_TRADES=50,
        KELLY_MAX_RISK_MULTIPLIER=1.5,
        KELLY_MIN_RISK_MULTIPLIER=0.25,
        KELLY_FRACTION=0.25,
    )
    mult, stats = compute_kelly_risk_scaling(settings, pnls_newest_first=[1.0, -1.0])
    assert mult == 1.0
    assert float(stats["sample_n"]) <= 30


def test_kelly_mult_cap(make_settings_factory) -> None:
    settings = make_settings_factory(
        ENABLE_KELLY_SIZING=True,
        KELLY_MIN_TRADES=5,
        KELLY_LOOKBACK_TRADES=80,
        KELLY_MAX_RISK_MULTIPLIER=1.5,
        KELLY_MIN_RISK_MULTIPLIER=0.25,
        KELLY_FRACTION=0.25,
    )
    wins = [2.0] * 35
    losses = [-1.0] * 35
    pnls = [x for pair in zip(wins, losses, strict=False) for x in pair]
    mult, _stats = compute_kelly_risk_scaling(settings, pnls_newest_first=pnls[-80:])
    assert mult <= settings.KELLY_MAX_RISK_MULTIPLIER


def test_sizer_fallback_without_database(make_settings_factory) -> None:
    settings = make_settings_factory(ENABLE_KELLY_SIZING=True)
    sizer = PositionSizer(settings, ComplianceAdapter(settings), ExposureChecker(settings), database=None)
    acc = make_account()
    r = sizer.size(
        symbol="SPY",
        entry_price=100.0,
        atr=2.0,
        account=acc,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert r.shares >= 1.0


def test_sizer_handles_zero_avg_loss_via_db_stub(make_settings_factory) -> None:
    settings = make_settings_factory(ENABLE_KELLY_SIZING=True, KELLY_MIN_TRADES=5, KELLY_LOOKBACK_TRADES=20)

    mock_db = MagicMock()
    mock_db.get_recent_realized_returns_for_kelly = lambda **_k: [0.01, 0.02, 0.015, 0.05, 0.044]
    assert len(mock_db.get_recent_realized_returns_for_kelly(limit=20)) >= 5

    sizer = PositionSizer(settings, ComplianceAdapter(settings), ExposureChecker(settings), database=mock_db)
    acc = make_account(equity=50_000.0)
    r = sizer.size(
        symbol="SPY",
        entry_price=400.0,
        atr=1.0,
        account=acc,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert r.shares >= 0


def test_default_position_sizing_unchanged_when_kelly_disabled(make_settings_factory) -> None:
    settings = make_settings_factory(ENABLE_KELLY_SIZING=False)
    sizer = PositionSizer(settings, ComplianceAdapter(settings), ExposureChecker(settings), database=None)
    acc = make_account(equity=100_000.0)
    r = sizer.size(
        symbol="AAPL",
        entry_price=100.0,
        atr=2.0,
        account=acc,
        positions=[],
        bot_managed_notional=0.0,
    )
    assert r.sizing_mode == "flat_atr"
