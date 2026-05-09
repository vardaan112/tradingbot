"""KellySizer class (delegates SQLite stats via mocked DB)."""

from __future__ import annotations

from unittest.mock import MagicMock

from risk.kelly_sizer import KellySizer


def test_kelly_sizer_disabled_returns_baseline(make_settings_factory) -> None:
    s = make_settings_factory(ENABLE_KELLY_SIZING=False)
    db = MagicMock()
    ks = KellySizer(s, db)
    d = ks.get_adjusted_risk_pct("QQQ", 0.015, {})
    assert d.risk_pct == 0.015
    db.get_recent_realized_pnls_for_kelly.assert_not_called()


def test_kelly_sizer_insufficient_samples(make_settings_factory) -> None:
    s = make_settings_factory(
        ENABLE_KELLY_SIZING=True,
        KELLY_MIN_TRADES=50,
        KELLY_LOOKBACK_TRADES=100,
    )
    db = MagicMock()
    db.get_recent_realized_returns_for_kelly.return_value = [0.01, -0.01, 0.03]

    ks = KellySizer(s, db)
    d = ks.get_adjusted_risk_pct("SPY", 0.01, {})
    assert d.risk_pct == 0.01
    assert d.sizing_mode == "baseline_insufficient_history"


def test_kelly_sizer_uses_return_percentages_not_raw_dollars(make_settings_factory) -> None:
    s = make_settings_factory(
        ENABLE_KELLY_SIZING=True,
        KELLY_USE_RETURN_PCT=True,
        KELLY_MIN_TRADES=5,
        KELLY_LOOKBACK_TRADES=10,
    )
    db = MagicMock()
    db.get_recent_realized_returns_for_kelly.return_value = [0.02, 0.01, -0.01, 0.03, -0.01]
    db.get_recent_realized_pnls_for_kelly.return_value = [10_000.0] * 5

    ks = KellySizer(s, db)
    d = ks.get_adjusted_risk_pct("SPY", 0.01, {})

    assert d.sample_n == 5
    db.get_recent_realized_returns_for_kelly.assert_called_once()
    db.get_recent_realized_pnls_for_kelly.assert_not_called()
