"""Phase 8 performance-weighted ensemble helpers."""

from __future__ import annotations

from datetime import timedelta

import pytest

from core.database import Database
from services.ensemble import WeightedEnsembleEngine
from services.performance_weights import PerformanceWeightCalculator, _clamp_normalize_weights
from utils.time_utils import now_utc


def _shadow_settings(make_settings_factory, **kwargs):
    base = dict(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENABLED=True,
        ENSEMBLE_ENTER_THRESHOLD=0.5,
        ENSEMBLE_MIN_AGREEING_STRATEGIES=1,
        ENSEMBLE_WEIGHT_MODE="performance",
        ENSEMBLE_PERFORMANCE_SOURCE="shadow",
        ENSEMBLE_PERFORMANCE_LOOKBACK_DAYS=30,
        ENSEMBLE_MIN_TRADES_FOR_WEIGHT=5,
        ENSEMBLE_WEIGHT_SMOOTHING_ALPHA=0.2,
        ENSEMBLE_MIN_WEIGHT=0.1,
        ENSEMBLE_MAX_WEIGHT=0.9,
        ALLOW_LIVE_PERFORMANCE_WEIGHTS=False,
    )
    base.update(kwargs)
    return make_settings_factory(**base)


def _insert_trade(
    db: Database,
    *,
    strategy: str,
    pnl: float,
    closed_at: str,
    source: str = "shadow",
) -> None:
    db.record_completed_trade(
        trade_id=None,
        symbol="SPY",
        side="long",
        quantity=10.0,
        entry_price=100.0,
        exit_price=100.0 + pnl / 10.0,
        realized_pnl=pnl,
        realized_return=pnl / 1000.0,
        opened_at="2026-01-01T10:00:00+00:00",
        closed_at=closed_at,
        strategy_name=strategy,
        risk_mode="shadow",
        regime_type=None,
        sentiment_score=None,
        sentiment_label=None,
        is_canary=0,
        source=source,
        entry_notional=1000.0,
        invalid_for_ml=True,
        invalid_for_kelly=True,
    )


def test_thin_data_falls_back_to_static(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _shadow_settings(make_settings_factory, ENSEMBLE_MIN_TRADES_FOR_WEIGHT=20)
    t0 = now_utc() - timedelta(days=2)
    for i in range(3):
        ts = (t0 + timedelta(hours=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=1.0, closed_at=ts)
        _insert_trade(db, strategy="momentum", pnl=-0.5, closed_at=ts)
    calc = PerformanceWeightCalculator(s, db)
    r = calc.compute(previous_smoothed_weights=None, record=False)
    assert not r.used_performance
    assert r.fallback_reason is not None
    assert "insufficient_trades" in r.fallback_reason


def test_weights_normalize_and_clamp(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw2.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _shadow_settings(make_settings_factory, ENSEMBLE_MIN_TRADES_FOR_WEIGHT=5)
    base = now_utc() - timedelta(days=10)
    for i in range(8):
        ts = (base + timedelta(days=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=5.0, closed_at=ts)
        _insert_trade(db, strategy="momentum", pnl=-1.0, closed_at=ts)
    calc = PerformanceWeightCalculator(s, db)
    r = calc.compute(record=False)
    assert r.used_performance
    ssum = sum(r.weights.values())
    assert abs(ssum - 1.0) < 1e-6
    for _k, w in r.weights.items():
        assert 0.1 - 1e-9 <= w <= 0.9 + 1e-9


def test_smoothing_blends_toward_target(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw3.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _shadow_settings(make_settings_factory, ENSEMBLE_MIN_TRADES_FOR_WEIGHT=5, ENSEMBLE_WEIGHT_SMOOTHING_ALPHA=0.5)
    base = now_utc() - timedelta(days=8)
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=10.0, closed_at=ts)
        _insert_trade(db, strategy="momentum", pnl=-5.0, closed_at=ts)
    calc = PerformanceWeightCalculator(s, db)
    prev = {"rsi_mean_reversion": 0.5, "momentum": 0.5}
    r1 = calc.compute(previous_smoothed_weights=prev, record=False)
    assert r1.used_performance
    # With α=0.5, blended should differ from pure target but stay normalized
    assert abs(sum(r1.weights.values()) - 1.0) < 1e-6


def test_live_performance_blocked_without_flag(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw4.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENABLED=True,
        ENSEMBLE_WEIGHT_MODE="performance",
        ENSEMBLE_PERFORMANCE_SOURCE="live",
        ENSEMBLE_MIN_TRADES_FOR_WEIGHT=3,
        ALLOW_LIVE_PERFORMANCE_WEIGHTS=False,
    )
    for i in range(5):
        ts = (now_utc() - timedelta(days=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=2.0, closed_at=ts, source="live")
        _insert_trade(db, strategy="momentum", pnl=1.0, closed_at=ts, source="live")
    calc = PerformanceWeightCalculator(s, db)
    r = calc.compute(record=False)
    assert not r.used_performance
    assert r.fallback_reason == "live_performance_weights_disabled"


def test_live_performance_allowed_with_flag(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw5.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = make_settings_factory(
        STRATEGY_RUN_MODE="ensemble",
        ACTIVE_STRATEGIES="rsi_mean_reversion,momentum",
        STRATEGY_WEIGHTS_JSON='{"rsi_mean_reversion":0.5,"momentum":0.5}',
        ENSEMBLE_ENABLED=True,
        ENSEMBLE_WEIGHT_MODE="performance",
        ENSEMBLE_PERFORMANCE_SOURCE="live",
        ENSEMBLE_MIN_TRADES_FOR_WEIGHT=5,
        ALLOW_LIVE_PERFORMANCE_WEIGHTS=True,
        ENSEMBLE_PERFORMANCE_LOOKBACK_DAYS=60,
    )
    base = now_utc() - timedelta(days=10)
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=3.0, closed_at=ts, source="live")
        _insert_trade(db, strategy="momentum", pnl=1.0, closed_at=ts, source="live")
    calc = PerformanceWeightCalculator(s, db)
    r = calc.compute(record=False)
    assert r.used_performance


def test_weighted_ensemble_engine_uses_performance_weights(make_settings_factory, tmp_path) -> None:
    db_path = tmp_path / "pw6.sqlite3"
    db = Database(db_path)
    db.init_schema()
    s = _shadow_settings(make_settings_factory, ENSEMBLE_MIN_TRADES_FOR_WEIGHT=5)
    base = now_utc() - timedelta(days=10)
    for i in range(6):
        ts = (base + timedelta(days=i)).isoformat()
        _insert_trade(db, strategy="rsi_mean_reversion", pnl=8.0, closed_at=ts)
        _insert_trade(db, strategy="momentum", pnl=-2.0, closed_at=ts)
    eng = WeightedEnsembleEngine(s, database=db)
    w = eng.effective_weights
    assert abs(sum(w.values()) - 1.0) < 1e-5


def test_clamp_normalize_helper() -> None:
    raw = {"a": 0.01, "b": 5.0}
    out = _clamp_normalize_weights(raw, lo=0.2, hi=0.8, active=["a", "b"])
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert 0.2 <= out["a"] <= 0.8 and 0.2 <= out["b"] <= 0.8
