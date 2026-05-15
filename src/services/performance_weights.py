"""Phase 8: optional performance-based ensemble weights (conservative, smoothed, clamped)."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from strategies.registry import normalize_strategy_name
from utils.time_utils import now_utc

if TYPE_CHECKING:
    from config.settings import Settings
    from core.database import Database

_LOG = logging.getLogger(__name__)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _norm_cohort(raw: dict[str, float], *, invert: bool = False) -> dict[str, float]:
    """Map values to [0, 1] within cohort; ``invert`` for lower-is-better metrics."""

    if not raw:
        return {}
    vals = {k: float(v) for k, v in raw.items()}
    lo = min(vals.values())
    hi = max(vals.values())
    span = hi - lo
    out: dict[str, float] = {}
    for k, v in vals.items():
        if span <= 1e-18:
            x = 0.5
        else:
            x = (v - lo) / span
        out[k] = 1.0 - x if invert else x
    return out


def _clamp_normalize_weights(
    ws: dict[str, float],
    *,
    lo: float,
    hi: float,
    active: list[str],
) -> dict[str, float]:
    """Clamp each weight then renormalize to sum 1 over ``active`` keys."""

    out: dict[str, float] = {}
    for name in active:
        w = max(lo, min(hi, float(ws.get(name, 0.0))))
        out[name] = w
    s = sum(out.values())
    if s <= 1e-18:
        n = max(1, len(active))
        return {k: 1.0 / n for k in active}
    return {k: float(out[k]) / s for k in active}


def _static_weights_normalized(settings: "Settings") -> dict[str, float]:
    active = [normalize_strategy_name(n) for n in settings.active_strategies_list]
    raw = settings.strategy_weights_dict
    lo = float(settings.ENSEMBLE_MIN_WEIGHT)
    hi = float(settings.ENSEMBLE_MAX_WEIGHT)
    ws: dict[str, float] = {}
    for name in active:
        w = float(raw.get(name, 1.0))
        w = max(lo, min(hi, w))
        ws[name] = w
    s = sum(ws.values())
    if s <= 1e-18:
        n = max(1, len(active))
        return {k: 1.0 / n for k in active}
    return {k: float(ws[k]) / s for k in active}


@dataclass
class StrategyPerfSnapshot:
    """Per-strategy aggregates used for scoring."""

    strategy_name: str
    trade_count: int
    total_pnl: float
    total_entry_notional: float
    period_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_like: float


@dataclass
class PerformanceWeightResult:
    weights: dict[str, float]
    used_performance: bool
    fallback_reason: Optional[str]
    metrics: dict[str, StrategyPerfSnapshot] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


class PerformanceWeightCalculator:
    """Compute smoothed ensemble weights from recent ``completed_trades``."""

    def __init__(self, settings: "Settings", database: "Database") -> None:
        self._settings = settings
        self._database = database

    def compute(
        self,
        *,
        previous_smoothed_weights: Optional[dict[str, float]] = None,
        record: bool = True,
    ) -> PerformanceWeightResult:
        settings = self._settings
        active = [normalize_strategy_name(n) for n in settings.active_strategies_list]
        static_w = _static_weights_normalized(settings)
        src = str(settings.ENSEMBLE_PERFORMANCE_SOURCE).strip().lower()
        detail: dict[str, Any] = {"source": src, "active": list(active)}

        if src == "live" and not bool(settings.ALLOW_LIVE_PERFORMANCE_WEIGHTS):
            out = PerformanceWeightResult(
                weights=dict(static_w),
                used_performance=False,
                fallback_reason="live_performance_weights_disabled",
                detail={**detail, "blocked": True},
            )
            if record:
                self._persist(out, static_w)
            return out

        lookback = max(1, int(settings.ENSEMBLE_PERFORMANCE_LOOKBACK_DAYS))
        cutoff = (now_utc() - timedelta(days=lookback)).isoformat()
        rows = self._database.query_completed_trades_for_performance(
            source=src,
            closed_after_iso=cutoff,
            limit=50_000,
        )
        by_strat: dict[str, list[dict[str, Any]]] = {k: [] for k in active}
        for r in rows:
            sn = str(r["strategy_name"] or "").strip()
            if not sn:
                continue
            key = normalize_strategy_name(sn)
            if key not in by_strat:
                continue
            pnl = _safe_float(r["realized_pnl"])
            entry_notional = r["entry_notional"]
            if entry_notional is None:
                q = _safe_float(r["quantity"])
                ep = _safe_float(r["entry_price"])
                entry_notional = abs(q * ep) if q and ep else 0.0
            else:
                entry_notional = _safe_float(entry_notional)
            ret = r["realized_return"]
            if ret is None:
                ret = (pnl / entry_notional) if entry_notional > 1e-12 else 0.0
            else:
                ret = _safe_float(ret)
            by_strat[key].append(
                {
                    "pnl": pnl,
                    "ret": ret,
                    "entry_notional": entry_notional,
                    "closed_at": str(r["closed_at"] or ""),
                },
            )

        min_tr = max(1, int(settings.ENSEMBLE_MIN_TRADES_FOR_WEIGHT))
        thin: list[str] = [k for k in active if len(by_strat[k]) < min_tr]
        if thin:
            reason = f"insufficient_trades strategies={thin} need>={min_tr}"
            out = PerformanceWeightResult(
                weights=dict(static_w),
                used_performance=False,
                fallback_reason=reason,
                detail={**detail, "thin": thin},
            )
            if record:
                self._persist(out, static_w)
            return out

        metrics: dict[str, StrategyPerfSnapshot] = {}
        period_returns: dict[str, float] = {}
        win_rates: dict[str, float] = {}
        pfs: dict[str, float] = {}
        dds: dict[str, float] = {}
        sharpes: dict[str, float] = {}

        for k in active:
            trades = by_strat[k]
            n = len(trades)
            total_pnl = sum(t["pnl"] for t in trades)
            total_notional = sum(max(t["entry_notional"], 0.0) for t in trades)
            period_ret = total_pnl / max(total_notional, 1e-12)
            wins = sum(1 for t in trades if t["pnl"] > 1e-12)
            losses = sum(1 for t in trades if t["pnl"] < -1e-12)
            win_rate = wins / max(n, 1)
            gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gl = sum(t["pnl"] for t in trades if t["pnl"] < 0)
            pf = gp / max(abs(gl), 1e-12)
            pf = min(pf, 25.0)

            cum = 0.0
            peak = 0.0
            max_dd = 0.0
            for t in trades:
                cum += t["pnl"]
                peak = max(peak, cum)
                if peak > 1e-12:
                    max_dd = max(max_dd, (peak - cum) / peak)

            rets = [float(t["ret"]) for t in trades]
            m = sum(rets) / max(len(rets), 1)
            var = sum((x - m) ** 2 for x in rets) / max(len(rets), 1)
            std = math.sqrt(max(var, 0.0))
            sharpe_like = (m / (std + 1e-9)) * math.sqrt(max(len(rets), 1))

            metrics[k] = StrategyPerfSnapshot(
                strategy_name=k,
                trade_count=n,
                total_pnl=float(total_pnl),
                total_entry_notional=float(total_notional),
                period_return=float(period_ret),
                win_rate=float(win_rate),
                profit_factor=float(pf),
                max_drawdown=float(max_dd),
                sharpe_like=float(sharpe_like),
            )
            period_returns[k] = metrics[k].period_return
            win_rates[k] = metrics[k].win_rate
            pfs[k] = metrics[k].profit_factor
            dds[k] = metrics[k].max_drawdown
            sharpes[k] = metrics[k].sharpe_like

        n_ret = _norm_cohort(period_returns)
        n_wr = _norm_cohort(win_rates)
        n_pf = _norm_cohort(pfs)
        n_dd = _norm_cohort(dds, invert=False)
        n_sh = _norm_cohort(sharpes)

        scores: dict[str, float] = {}
        for k in active:
            scores[k] = (
                0.35 * n_ret.get(k, 0.5)
                + 0.25 * n_wr.get(k, 0.5)
                + 0.25 * n_pf.get(k, 0.5)
                + 0.15 * n_sh.get(k, 0.5)
                - 0.35 * n_dd.get(k, 0.0)
            )

        base = min(scores.values()) - 0.02
        raw_mass = {k: max(0.01, (scores[k] - base) ** 1.35) for k in active}
        sm = sum(raw_mass.values())
        if sm <= 1e-18:
            target = {k: 1.0 / max(len(active), 1) for k in active}
        else:
            target = {k: float(raw_mass[k]) / sm for k in active}

        alpha = float(settings.ENSEMBLE_WEIGHT_SMOOTHING_ALPHA)
        old = previous_smoothed_weights if previous_smoothed_weights is not None else static_w
        blended: dict[str, float] = {}
        for k in active:
            blended[k] = (1.0 - alpha) * float(old.get(k, 1.0 / len(active))) + alpha * float(target.get(k, 0.0))

        lo = float(settings.ENSEMBLE_MIN_WEIGHT)
        hi = float(settings.ENSEMBLE_MAX_WEIGHT)
        final_w = _clamp_normalize_weights(blended, lo=lo, hi=hi, active=active)

        out = PerformanceWeightResult(
            weights=final_w,
            used_performance=True,
            fallback_reason=None,
            metrics=metrics,
            scores=scores,
            target_weights=target,
            detail={
                **detail,
                "lookback_days": lookback,
                "cutoff_iso": cutoff,
                "alpha": alpha,
                "normalized": {"return": n_ret, "win_rate": n_wr, "profit_factor": n_pf, "drawdown": n_dd, "sharpe": n_sh},
            },
        )
        if record:
            self._persist(out, static_w)
        return out

    def _persist(self, result: PerformanceWeightResult, static_w: dict[str, float]) -> None:
        try:
            meta = {
                "ensemble_weight_mode": self._settings.ENSEMBLE_WEIGHT_MODE,
                "used_performance": result.used_performance,
                "fallback_reason": result.fallback_reason,
                "weights": result.weights,
                "static_weights": static_w,
                "scores": result.scores,
                "target_weights": result.target_weights,
                "metrics": {
                    k: {
                        "trade_count": v.trade_count,
                        "period_return": v.period_return,
                        "win_rate": v.win_rate,
                        "profit_factor": v.profit_factor,
                        "max_drawdown": v.max_drawdown,
                        "sharpe_like": v.sharpe_like,
                        "total_pnl": v.total_pnl,
                    }
                    for k, v in result.metrics.items()
                },
                "detail": result.detail,
            }
            self._database.record_strategy_decision(
                source=str(self._settings.ENSEMBLE_PERFORMANCE_SOURCE),
                timestamp=now_utc().isoformat(),
                symbol="PORTFOLIO",
                final_action="weights",
                run_id=None,
                decision_type="ensemble_performance_weights",
                weighted_score=None,
                threshold=None,
                contributing_signals_json=json.dumps(result.weights, separators=(",", ":")),
                metadata=meta,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("event=performance_weights_persist_failed err=%s", exc)


__all__ = [
    "PerformanceWeightCalculator",
    "PerformanceWeightResult",
    "StrategyPerfSnapshot",
    "_static_weights_normalized",
]
