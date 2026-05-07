"""Weekly walk-forward parameter autotune using the existing backtest grid."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

from alpaca.data.historical.stock import StockHistoricalDataClient

from config.constants import LOGGER_APP
from config.settings import Settings
from utils.backtester import BacktestConfig, GridRow, default_param_grid, run_grid

_LOG = logging.getLogger(LOGGER_APP)


def iso_week_token(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def autotune_composite_score(
    *,
    sharpe: float,
    max_drawdown: float,
    profit_factor: float,
    n_trades: int,
    min_trades: int,
) -> Optional[float]:
    if n_trades < min_trades:
        return None
    bonus = 0.25 if profit_factor > 1.2 else 0.0
    return float(sharpe) - abs(float(max_drawdown)) * 2.0 + bonus


def score_grid_row(row: GridRow, *, min_trades: int) -> Optional[float]:
    return autotune_composite_score(
        sharpe=float(row.sharpe_ratio),
        max_drawdown=float(row.max_drawdown),
        profit_factor=float(row.profit_factor),
        n_trades=int(row.n_trades),
        min_trades=min_trades,
    )


def pick_best_grid_row(rows: Sequence[GridRow], *, min_trades: int) -> tuple[Optional[GridRow], Optional[float]]:
    best_row: Optional[GridRow] = None
    best_sc: Optional[float] = None
    for r in rows:
        sc = score_grid_row(r, min_trades=min_trades)
        if sc is None:
            continue
        if best_sc is None or sc > best_sc:
            best_sc = sc
            best_row = r
    return best_row, best_sc


def _resolve_path(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else (root / p).resolve()


def _validate_payload(d: dict[str, Any]) -> bool:
    try:
        if str(d.get("source", "")).lower() != "autotune":
            return False
        rsi_e = float(d["rsi_entry_threshold"])
        rsi_x = float(d["rsi_exit_threshold"])
        adx = float(d["adx_threshold"])
        ast = float(d["atr_stop_multiplier"])
        atl = float(d["atr_trailing_multiplier"])
        if rsi_e >= rsi_x or adx <= 0 or ast <= 0 or atl <= 0:
            return False
        if not (1.0 <= rsi_e <= 99.0 and 1.0 <= rsi_x <= 99.0):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def backup_params_file(src: Path, backup_dir: Path) -> Optional[Path]:
    if not src.is_file():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = backup_dir / f"dynamic_params_{ts}.json"
    shutil.copy2(src, dst)
    return dst


def read_prior_autotune_score(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        if str(raw.get("source", "")).lower() != "autotune":
            return None
        sc = raw.get("score")
        return float(sc) if sc is not None else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def build_autotune_payload(
    *,
    winner: GridRow,
    composite_score: float,
    start: datetime,
    end: datetime,
    rsi_exit_static: float,
) -> dict[str, Any]:
    params = winner.params
    return {
        "rsi_entry_threshold": float(params.rsi_oversold),
        "rsi_exit_threshold": float(rsi_exit_static),
        "adx_threshold": float(params.adx_range_max),
        "atr_stop_multiplier": float(params.atr_stop_multiplier),
        "atr_multiplier": float(params.atr_stop_multiplier),
        "atr_trailing_multiplier": float(params.trail_atr_multiplier),
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "lookback_start": start.astimezone(timezone.utc).date().isoformat(),
        "lookback_end": end.astimezone(timezone.utc).date().isoformat(),
        "backtest_start": start.astimezone(timezone.utc).date().isoformat(),
        "backtest_end": end.astimezone(timezone.utc).date().isoformat(),
        "score": float(composite_score),
        "sharpe_ratio": float(winner.sharpe_ratio),
        "max_drawdown": float(winner.max_drawdown),
        "profit_factor": float(winner.profit_factor),
        "win_rate": float(winner.win_rate),
        "trade_count": int(winner.n_trades),
        "parameter_set_id": winner.parameter_set_id,
        "source": "autotune",
    }


def persist_dynamic_params_safe(
    path: Path,
    payload: dict[str, Any],
    *,
    backup_dir: Path,
    prior_best_score: Optional[float],
) -> tuple[bool, str]:
    """Return (applied?, reason).

    Applies only if payload validates and beats prior autotune score (if any).
    """
    if not _validate_payload(payload):
        _LOG.warning("event=autotune_rejected reason=invalid_payload")
        return False, "invalid_payload"
    composite = float(payload["score"])
    if prior_best_score is not None and composite + 1e-12 < float(prior_best_score):
        _LOG.info(
            "event=autotune_rejected reason=below_prior prior=%.6f new=%.6f",
            prior_best_score,
            composite,
        )
        return False, "below_prior_score"

    backup_params_file(path, backup_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _LOG.info("event=autotune_applied score=%.6f path=%s", composite, path)
    return True, "applied"


def run_autotune_job(
    settings: Settings,
    *,
    refresh_cache: bool = False,
    run_grid_fn: Optional[
        Callable[
            ...,
            Tuple[list[GridRow], list[dict[str, Any]]],
        ]
    ] = None,
    client_factory: Optional[Callable[[Settings], StockHistoricalDataClient]] = None,
) -> dict[str, Any]:
    """Execute one autotune pass. Intended for Sundays / manual triggers."""

    root = Path.cwd()
    dyn_path = _resolve_path(root, Path(settings.DYNAMIC_PARAMS_PATH))
    backup_dir = _resolve_path(root, Path(settings.STATE_DIR)) / "param_backups"
    last_run_marker = _resolve_path(root, Path(settings.STATE_DIR)) / "autotune_last_week.json"

    if not settings.ENABLE_AUTOTUNE:
        _LOG.info("event=autotune_skipped reason=disabled")
        return {"ok": False, "reason": "disabled"}

    _LOG.info("event=autotune_start")

    prior_sc = read_prior_autotune_score(dyn_path)
    end = datetime.now(timezone.utc)
    lb = max(7, int(settings.AUTOTUNE_LOOKBACK_DAYS))
    start = end - timedelta(days=lb)

    syms = tuple(settings.symbols_list)
    ieq = max(10_000.0, float(settings.MAX_EQUITY_USAGE_USD))
    rpct = float(settings.MAX_RISK_PER_TRADE_PCT)
    cache_dir = root / "runtime" / "cache"
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    cfg = BacktestConfig(
        symbols=syms,
        start=start,
        end=end,
        timeframe="15Min",
        initial_equity=ieq,
        risk_pct=rpct,
        spread_pct=float(settings.SPREAD_FILTER_PCT),
        slippage_bps=1.5,
        fee_bps_per_side=0.0,
        data_feed=settings.feed_resolved(sip_supported=False),
        use_cache=True,
        refresh_cache=refresh_cache,
        cache_dir=cache_dir,
        reports_dir=rep,
        output_results=rep / "_autotune_results.csv",
        output_trades=rep / "_autotune_trades.csv",
        output_summary=rep / "_autotune_summary.md",
    )

    run_id = str(uuid.uuid4())
    grid_specs = list(default_param_grid())

    try:
        if run_grid_fn is not None:
            rows, _trs = run_grid_fn(
                run_id=run_id,
                base_settings=settings,
                cfg=cfg,
                param_grid=grid_specs,
            )
        else:
            cf = client_factory or (
                lambda s: StockHistoricalDataClient(
                    api_key=s.ALPACA_API_KEY,
                    secret_key=s.ALPACA_API_SECRET,
                )
            )
            client = cf(settings)
            rows, _trs = run_grid(
                run_id=run_id,
                base_settings=settings,
                cfg=cfg,
                client=client,
                param_grid=grid_specs,
            )
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("event=autotune_failed err=%s", exc)
        return {"ok": False, "reason": f"exception:{exc}"}

    winner, composite = pick_best_grid_row(rows, min_trades=int(settings.AUTOTUNE_MIN_TRADES_PER_CONFIG))
    if winner is None or composite is None:
        _LOG.warning(
            "event=autotune_failed reason=no_eligible_candidate min_trades=%s",
            settings.AUTOTUNE_MIN_TRADES_PER_CONFIG,
        )
        _LOG.warning("event=autotune_rejected reason=no_eligible_candidate")
        return {"ok": False, "reason": "no_eligible_candidate"}

    mdd_abs = abs(float(winner.max_drawdown))
    if mdd_abs > float(settings.AUTOTUNE_MAX_DRAWDOWN_ABS):
        _LOG.warning(
            "event=autotune_rejected reason=max_drawdown_abs mdd=%.6f cap=%.6f",
            mdd_abs,
            float(settings.AUTOTUNE_MAX_DRAWDOWN_ABS),
        )
        return {"ok": False, "reason": "max_drawdown_rejected", "max_drawdown": mdd_abs}

    payload = build_autotune_payload(
        winner=winner,
        composite_score=composite,
        start=start,
        end=end,
        rsi_exit_static=float(settings.RSI_EXIT),
    )

    _LOG.info(
        "event=autotune_selected_params score=%.6f rsi=%s adx_max=%s atr_stop=%s trail=%s trades=%s",
        composite,
        payload["rsi_entry_threshold"],
        payload["adx_threshold"],
        payload["atr_stop_multiplier"],
        payload["atr_trailing_multiplier"],
        payload["trade_count"],
    )

    applied, reason = persist_dynamic_params_safe(
        dyn_path,
        payload,
        backup_dir=backup_dir,
        prior_best_score=prior_sc,
    )
    marker = {"week": iso_week_token(datetime.now(timezone.utc)), "ts": datetime.now(timezone.utc).isoformat()}
    try:
        last_run_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    except OSError:
        pass

    _LOG.info(
        "event=autotune_complete ok=%s reason=%s",
        applied,
        reason,
    )
    return {
        "ok": True,
        "applied": applied,
        "apply_reason": reason,
        "composite_score": composite,
        "parameter_set_id": winner.parameter_set_id,
        "dyn_path": str(dyn_path),
    }


__all__ = [
    "build_autotune_payload",
    "pick_best_grid_row",
    "persist_dynamic_params_safe",
    "read_prior_autotune_score",
    "run_autotune_job",
    "score_grid_row",
    "iso_week_token",
]
