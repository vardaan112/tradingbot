"""ML signal filter — fail-open behavior (no broker calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.database import Database
from strategies.ml_filter import MLSignalFilter, build_feature_vector


@pytest.fixture()
def ml_settings(make_settings_factory, tmp_path: Path):
    return make_settings_factory(
        ENABLE_ML_FILTER=True,
        ML_FILTER_THRESHOLD=0.55,
        MIN_ML_TRAINING_TRADES=5,
        ML_MAX_TRAINING_TRADES=500,
        ML_MODEL_PATH=str(tmp_path / "ml_test.pkl"),
        ML_MODEL_META_PATH=str(tmp_path / "ml_test_meta.json"),
        DATABASE_PATH=str(tmp_path / "ml.sqlite"),
    )


def test_predict_fail_open_no_model(ml_settings) -> None:
    filt = MLSignalFilter(ml_settings)
    filt._model = None  # noqa: SLF001
    md = {
        "rsi": 30.0,
        "adx": 20.0,
        "sma200": 100.0,
        "last_close": 99.0,
        "spread_pct": 0.001,
        "atr": 2.0,
    }
    r = filt.predict_gate(entry_metadata=md, symbol="SPY")
    assert r.allowed


def test_probability_and_allow_when_above_threshold(ml_settings, monkeypatch) -> None:
    class Dummy:
        def predict_proba(self, x):  # noqa: ANN001
            import numpy as np

            return np.array([[0.4, 0.72]], dtype=float)

    filt = MLSignalFilter(ml_settings)
    monkeypatch.setattr(filt, "_model", Dummy())
    md = {
        "rsi": 35.0,
        "adx": 18.0,
        "regime_type": "Range",
        "sma200": 420.0,
        "last_close": 418.0,
        "spread_pct": 0.002,
        "atr": 1.8,
        "bar_timestamp": "2024-06-03T14:35:00+00:00",
    }
    r = filt.predict_gate(entry_metadata=md, symbol="SPY")
    assert r.probability is not None and 0.0 <= float(r.probability) <= 1.0
    assert r.allowed


def test_skip_when_below_threshold(ml_settings, monkeypatch) -> None:
    class Poor:
        def predict_proba(self, x):  # noqa: ANN001
            import numpy as np

            return np.array([[0.65, 0.35]], dtype=float)

    filt = MLSignalFilter(ml_settings)
    monkeypatch.setattr(filt, "_model", Poor())
    md = {"rsi": 40.0, "adx": 22.0, "sma200": 100.0, "last_close": 99.0, "atr": 1.0, "spread_pct": 0.0}
    r = filt.predict_gate(entry_metadata=md, symbol="QQQ")
    assert not r.allowed


def test_train_skips_when_insufficient_trades(tmp_path, make_settings_factory) -> None:
    s = make_settings_factory(
        ENABLE_ML_FILTER=True,
        MIN_ML_TRAINING_TRADES=50,
        ML_MAX_TRAINING_TRADES=100,
        ML_MODEL_PATH=str(tmp_path / "m.pkl"),
        ML_MODEL_META_PATH=str(tmp_path / "m.json"),
        DATABASE_PATH=str(tmp_path / "d.sqlite"),
    )
    Database(tmp_path / "d.sqlite").init_schema()
    MLSignalFilter(s).train_from_database(Database(tmp_path / "d.sqlite"))
    assert not (tmp_path / "m.pkl").exists()


def test_feature_vector_shape() -> None:
    md = {
        "rsi": 10.0,
        "atr": 1.0,
        "last_close": 100.0,
        "sma200": 99.0,
        "bar_timestamp": "2024-01-02T15:00:00+00:00",
    }
    v = build_feature_vector(md, symbol="SPY")
    assert len(v) == 11
