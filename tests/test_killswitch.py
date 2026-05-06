"""Tests for the latching kill switch."""

from __future__ import annotations

from core.state_store import StateStore
from risk.killswitch import KillSwitch


def test_no_latch_when_within_drawdown(tmp_path):
    state = StateStore(tmp_path)
    ks = KillSwitch(state, drawdown_pct=0.05)

    # Day starts at 100k.
    decision = ks.evaluate(current_equity=100_000.0)
    assert not decision.latched
    assert decision.daily_baseline == 100_000.0
    assert decision.drawdown_pct == 0.0

    # 2% drawdown - should not latch.
    decision = ks.evaluate(current_equity=98_000.0)
    assert not decision.latched
    assert abs(decision.drawdown_pct - 0.02) < 1e-9


def test_latches_at_threshold(tmp_path):
    state = StateStore(tmp_path)
    ks = KillSwitch(state, drawdown_pct=0.05)
    ks.evaluate(current_equity=100_000.0)
    decision = ks.evaluate(current_equity=95_000.0)
    assert decision.latched
    assert ks.is_latched()


def test_latch_persists_across_instances(tmp_path):
    state = StateStore(tmp_path)
    ks = KillSwitch(state, drawdown_pct=0.05)
    ks.evaluate(current_equity=100_000.0)
    ks.evaluate(current_equity=94_000.0)
    assert ks.is_latched()

    # Recreate the KillSwitch and StateStore from the same dir.
    state2 = StateStore(tmp_path)
    ks2 = KillSwitch(state2, drawdown_pct=0.05)
    assert ks2.is_latched(), "latch must survive process restart"


def test_manual_reset_requires_token(tmp_path):
    state = StateStore(tmp_path)
    ks = KillSwitch(state, drawdown_pct=0.05)
    ks.evaluate(current_equity=100_000.0)
    ks.evaluate(current_equity=80_000.0)
    assert ks.is_latched()

    # No token -> refused.
    assert not ks.reset(force=True, operator_token=None)
    assert ks.is_latched()

    # Short token -> refused.
    assert not ks.reset(force=True, operator_token="abc")
    assert ks.is_latched()

    # Force=False -> refused.
    assert not ks.reset(force=False, operator_token="operator123")
    assert ks.is_latched()

    # Proper reset.
    assert ks.reset(force=True, operator_token="operator123")
    assert not ks.is_latched()


def test_force_latch_writes_record(tmp_path):
    state = StateStore(tmp_path)
    ks = KillSwitch(state, drawdown_pct=0.05)
    ks.ensure_daily_baseline(100_000.0)
    ks.force_latch("manual_test", current_equity=99_000.0)
    record = ks.latch_record()
    assert record.latched
    assert record.reason == "manual_test"
    assert record.daily_baseline == 100_000.0
    assert record.triggered_equity == 99_000.0
