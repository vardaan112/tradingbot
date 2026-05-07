"""Phase 8 orchestration gates: ML startup sizing, Discord init copy, Kelly log hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from communication.discord_client import startup_initialization_notification
from core.database import Database


@pytest.mark.asyncio
async def test_ml_startup_gate_skips_insufficient_trades(
    monkeypatch, make_settings_factory, tmp_path: Path,
) -> None:
    from ml.signal_filter import MLSignalFilter
    from services.orchestrator import Orchestrator

    db_path = tmp_path / "ml_gate.sqlite"
    Database(db_path).init_schema()

    settings = make_settings_factory(
        ENABLE_ML_FILTER=True,
        ENABLE_DISCORD_BOT=False,
        MIN_ML_TRAINING_TRADES=500,
        DATABASE_PATH=str(db_path),
        STATE_DIR=str(tmp_path / "st_ml"),
        LOG_DIR=str(tmp_path / "logs_ml"),
    )

    invoked: list[bool] = []

    def guard(self: MLSignalFilter, db: Database) -> None:  # noqa: ARG002
        invoked.append(True)

    monkeypatch.setattr(MLSignalFilter, "train_from_database", guard)

    orch = Orchestrator(settings)
    assert orch._ml_filter is not None

    assert await orch.run_ml_startup_gate() is True
    assert not invoked


@pytest.mark.asyncio
async def test_ml_startup_gate_trains_when_above_threshold(monkeypatch, make_settings_factory, tmp_path: Path) -> None:
    from ml.signal_filter import MLSignalFilter
    from services.orchestrator import Orchestrator

    db_path = tmp_path / "ml_gate2.sqlite"
    Database(db_path).init_schema()

    settings = make_settings_factory(
        ENABLE_ML_FILTER=True,
        ENABLE_DISCORD_BOT=False,
        MIN_ML_TRAINING_TRADES=5,
        DATABASE_PATH=str(db_path),
        STATE_DIR=str(tmp_path / "st_ml2"),
        LOG_DIR=str(tmp_path / "logs_ml2"),
    )

    called: list[bool] = []

    def fake_train(self: MLSignalFilter, db: Database) -> None:
        called.append(True)
        self._model = object()

    monkeypatch.setattr(MLSignalFilter, "train_from_database", fake_train)

    orch = Orchestrator(settings)
    monkeypatch.setattr(orch._database, "count_completed_trades_ml_eligible", lambda **_k: 999)

    assert await orch.run_ml_startup_gate()
    assert called


@pytest.mark.asyncio
async def test_ml_startup_gate_aborts_when_configured(monkeypatch, make_settings_factory, tmp_path: Path) -> None:
    from ml.signal_filter import MLSignalFilter
    from services.orchestrator import Orchestrator

    db_path = tmp_path / "ml_gate3.sqlite"
    Database(db_path).init_schema()

    settings = make_settings_factory(
        ENABLE_ML_FILTER=True,
        ENABLE_DISCORD_BOT=False,
        ML_ABORT_ON_TRAINING_FAILURE=True,
        ML_BLOCK_ENTRIES_ON_TRAINING_FAILURE=True,
        MIN_ML_TRAINING_TRADES=5,
        DATABASE_PATH=str(db_path),
        STATE_DIR=str(tmp_path / "st_ml3"),
        LOG_DIR=str(tmp_path / "logs_ml3"),
    )

    def boom(_self: MLSignalFilter, _db: Database) -> None:
        raise RuntimeError("train boom")

    monkeypatch.setattr(MLSignalFilter, "train_from_database", boom)

    orch = Orchestrator(settings)
    monkeypatch.setattr(orch._database, "count_completed_trades_ml_eligible", lambda **_k: 999)

    assert await orch.run_ml_startup_gate() is False
    assert orch._ml_filter is not None


def test_startup_initialization_embedding_has_mode_and_kelly_preview(make_settings_factory) -> None:
    s = make_settings_factory(DRY_RUN=True, ENABLE_KELLY_SIZING=True, ENABLE_ML_FILTER=True)
    spec = startup_initialization_notification(
        settings=s,
        equity=125_431.52,
        buying_power=200_000.0,
        symbols_preview="SPY,QQQ",
        kill_switch_latched=False,
        heartbeat_active=True,
        canary_status="passed",
        ml_ready=True,
        risk_mode_label="normal",
    )
    assert "Dry Run Initialization" in spec["title"]
    body = "\n".join(str(x) for x in spec["lines"])
    assert "risk_mode=normal" in body
    assert "Kelly_sizing=Enabled" in body

