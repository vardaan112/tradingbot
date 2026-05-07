"""Autotune logic tests (no Alpaca API)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config.strategy_runtime import merge_strategy_thresholds
from services import autotune as at
from utils.backtester import GridRow, StrategyParams


def _mk_row(
    *,
    n_trades: int,
    sharpe: float,
    mdd: float,
    pf: float,
    pid: str = "deadbeefaaa",
    spi: StrategyParams | None = None,
) -> GridRow:
    sp = spi or StrategyParams(30.0, 25.0, 2.0, 2.0)
    return GridRow(
        run_id="r1",
        parameter_set_id=pid,
        params=sp,
        symbol="PORTFOLIO_AVG",
        total_return=0.01,
        sharpe_ratio=sharpe,
        max_drawdown=mdd,
        win_rate=0.5,
        profit_factor=pf,
        n_trades=n_trades,
        avg_trade_return_pct=0.0,
        avg_holding_bars=1.0,
        worst_trade_usd=-10.0,
        best_trade_usd=10.0,
        avg_r_multiple=0.1,
        score=0.0,
    )


def test_score_grid_row_respects_min_trades() -> None:
    row = _mk_row(n_trades=5, sharpe=2.0, mdd=-0.02, pf=3.0)
    assert at.score_grid_row(row, min_trades=10) is None
    assert at.score_grid_row(replace(row, n_trades=12), min_trades=10) is not None


def test_pick_best_prefers_higher_composite() -> None:
    low = _mk_row(n_trades=20, sharpe=1.0, mdd=-0.15, pf=1.0, pid="aaa")
    high_pf = replace(
        _mk_row(n_trades=20, sharpe=1.0, mdd=-0.15, pf=2.0, pid="bbb"),
        params=StrategyParams(25.0, 22.0, 1.5, 2.5),
    )
    winner, sc = at.pick_best_grid_row([low, high_pf], min_trades=10)
    assert winner is not None and sc is not None
    sl, sh = at.score_grid_row(low, min_trades=10), at.score_grid_row(high_pf, min_trades=10)
    assert sh is not None and sl is not None
    assert sc == max(sl, sh)
    assert winner.parameter_set_id == high_pf.parameter_set_id


def test_persist_rejects_below_prior_score(tmp_path: Path) -> None:
    dyn = tmp_path / "dynamic_params.json"
    dyn.write_text(
        json.dumps(
            {
                "source": "autotune",
                "score": 10.0,
                "rsi_entry_threshold": 30.0,
                "rsi_exit_threshold": 50.0,
                "adx_threshold": 25.0,
                "atr_stop_multiplier": 2.0,
                "atr_trailing_multiplier": 2.0,
            },
        ),
        encoding="utf-8",
    )
    assert at.read_prior_autotune_score(dyn) == 10.0

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 1, 30, tzinfo=timezone.utc)
    payload = at.build_autotune_payload(
        winner=_mk_row(
            n_trades=15,
            sharpe=0.5,
            mdd=-0.1,
            pf=1.1,
            spi=StrategyParams(28.0, 24.0, 2.0, 2.0),
        ),
        composite_score=2.0,
        start=t0,
        end=t1,
        rsi_exit_static=50.0,
    )
    ok, reason = at.persist_dynamic_params_safe(
        dyn,
        payload,
        backup_dir=tmp_path / "bk",
        prior_best_score=10.0,
    )
    assert not ok and reason == "below_prior_score"


def test_persist_applies_and_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    bk = tmp_path / "bk"
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 1, 30, tzinfo=timezone.utc)
    payload = at.build_autotune_payload(
        winner=_mk_row(
            n_trades=15,
            sharpe=1.0,
            mdd=-0.05,
            pf=1.3,
            spi=StrategyParams(28.0, 24.0, 2.0, 2.0),
        ),
        composite_score=5.0,
        start=t0,
        end=t1,
        rsi_exit_static=50.0,
    )
    ok, reason = at.persist_dynamic_params_safe(out, payload, backup_dir=bk, prior_best_score=None)
    assert ok and reason == "applied"
    assert out.is_file()
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["source"] == "autotune" and float(raw["score"]) == 5.0


def test_run_autotune_job_handles_grid_exception(make_settings_factory) -> None:
    settings = make_settings_factory(
        ENABLE_AUTOTUNE=True,
        AUTOTUNE_MIN_TRADES_PER_CONFIG=10,
        DYNAMIC_PARAMS_PATH="runtime/dynamic_test.json",
        STATE_DIR="runtime",
    )

    def boom(**_kw: object) -> tuple[list[GridRow], list[dict]]:
        raise RuntimeError("no api")

    out = at.run_autotune_job(settings, run_grid_fn=boom)
    assert out["ok"] is False


def test_run_autotune_rejects_winner_when_drawdown_too_deep(make_settings_factory) -> None:
    settings = make_settings_factory(
        ENABLE_AUTOTUNE=True,
        AUTOTUNE_MIN_TRADES_PER_CONFIG=1,
        AUTOTUNE_MAX_DRAWDOWN_ABS=0.1,
        DYNAMIC_PARAMS_PATH="runtime/dd_test.json",
    )
    nasty = _mk_row(n_trades=20, sharpe=2.0, mdd=-0.6, pf=5.0)

    def shallow_grid(**_kw: object) -> tuple[list[GridRow], list[dict]]:
        return ([nasty], [])

    res = at.run_autotune_job(settings, run_grid_fn=shallow_grid)
    assert res["ok"] is False and res["reason"] == "max_drawdown_rejected"


def test_merge_runtime_ignores_json_when_autotune_disabled(make_settings_factory) -> None:
    s = make_settings_factory(ENABLE_AUTOTUNE=False, RSI_OVERSOLD=28.0)
    thr = merge_strategy_thresholds(s)
    assert thr.rsi_oversold == 28.0
