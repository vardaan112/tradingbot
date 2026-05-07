"""Anti-martingale policy (SQLite-trade-row driven)."""

from __future__ import annotations

import pytest

from core.database import CompletedTradeRow
from core.account import AccountSnapshot
from risk.anti_martingale import RiskMode, recent_trade_pnls_preview, resolve_anti_martingale
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


def _sizer(settings):
    return PositionSizer(settings, ComplianceAdapter(settings), ExposureChecker(settings))


def _row(rid: int, t: str, pnl: float) -> CompletedTradeRow:
    """Build a CompletedTradeRow; lists below are newest-first."""

    return CompletedTradeRow(
        id=rid,
        trade_id=None,
        symbol="SPY",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        exit_price=None,
        realized_pnl=pnl,
        realized_return=None,
        opened_at=None,
        closed_at=t,
        strategy_name="s",
        risk_mode=None,
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source=None,
    )


def test_defensive_on_three_losses(make_settings_factory) -> None:
    settings = make_settings_factory(
        ANTI_MARTINGALE_ENABLED=True,
        ANTI_MARTINGALE_LOSS_STREAK=3,
        ANTI_MARTINGALE_WIN_RECOVERY=2,
    )
    recent = [
        _row(1, "2026-05-06T18:00:03+00:00", -1.0),
        _row(2, "2026-05-06T18:00:02+00:00", -2.0),
        _row(3, "2026-05-06T18:00:01+00:00", -0.5),
    ]
    mode, mult, tok = resolve_anti_martingale(settings, recent)
    assert mode == RiskMode.DEFENSIVE and mult == 0.5
    assert "loss_streak" in tok


def test_normal_two_wins_recovery(make_settings_factory) -> None:
    settings = make_settings_factory(
        ANTI_MARTINGALE_ENABLED=True,
        ANTI_MARTINGALE_LOSS_STREAK=3,
        ANTI_MARTINGALE_WIN_RECOVERY=2,
    )
    recent = [
        _row(1, "2026-05-06T19:05:03+00:00", 1.0),
        _row(2, "2026-05-06T19:05:02+00:00", 2.0),
        _row(3, "2026-05-06T19:05:01+00:00", -5.0),
    ]
    mode, mult, tok = resolve_anti_martingale(settings, recent)
    assert mode == RiskMode.NORMAL and mult == 1.0
    assert "recovery" in tok


def test_disabled_is_neutral_multiplier(make_settings_factory) -> None:
    settings = make_settings_factory(
        ANTI_MARTINGALE_ENABLED=False,
        ANTI_MARTINGALE_LOSS_STREAK=3,
    )
    mode, mult, tok = resolve_anti_martingale(
        settings,
        [
            _row(1, "2026-05-06T20:01:02+00:00", -9.0),
            _row(2, "2026-05-06T20:01:01+00:00", -9.0),
        ],
    )
    assert mode == RiskMode.NORMAL and mult == 1.0
    assert "disabled" in tok


def test_recent_preview_tokens() -> None:
    txt = recent_trade_pnls_preview(
        [
            _row(7, "2026-05-06T21:00:01+00:00", -1),
            _row(8, "2026-05-06T21:00:02+00:00", 1),
            _row(9, "2026-05-06T21:00:03+00:00", 0.0),
        ],
        5,
    )
    assert "L" in txt and "W" in txt and "0" in txt


def test_defensive_multiplier_halves_effective_risk_pct(make_settings_factory) -> None:
    settings = make_settings_factory(
        BOT_CAPITAL_BASE_USD=10_000.0,
        MAX_RISK_PER_TRADE_PCT=0.01,
        ATR_STOP_MULTIPLIER=1.0,
        MAX_EQUITY_USAGE_USD=1_000_000.0,
        MAX_GROSS_EXPOSURE_PCT=2.0,
        MAX_OPEN_POSITIONS=5,
    )
    s = _sizer(settings)
    acct = make_account()
    norm = s.size(
        symbol="SPY",
        entry_price=50.0,
        atr=1.0,
        account=acct,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.0,
        anti_martingale_multiplier=1.0,
        risk_mode="normal",
    )
    defn = s.size(
        symbol="SPY",
        entry_price=50.0,
        atr=1.0,
        account=acct,
        positions=[],
        bot_managed_notional=0.0,
        conviction_risk_multiplier=1.0,
        anti_martingale_multiplier=0.5,
        risk_mode="defensive",
    )
    assert norm.shares > 0 and defn.shares > 0
    assert defn.effective_risk_pct == pytest.approx(norm.effective_risk_pct * 0.5)
