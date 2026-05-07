"""Fractional Kelly adjustment for per-trade risk percentage (SQLite-backed)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from config.constants import LOGGER_RISK
from config.settings import Settings
from core.database import Database


def _kelly_stats_from_pnls(pnls: list[float]) -> tuple[float, float, float, float, int]:
    vals = [float(x) for x in pnls if math.isfinite(float(x))]
    n = len(vals)
    wins = [x for x in vals if x > 1e-9]
    losses = [x for x in vals if x < -1e-9]
    wn = len(wins)
    wp = len(losses)
    avg_win = float(sum(wins) / wn) if wn else 0.0
    avg_loss_mag = float(abs(sum(losses)) / wp) if wp else 0.0
    gp = sum(wins)
    gl = abs(sum(losses))
    if gl > 1e-9:
        pf = float(gp / gl)
    elif gp > 0:
        pf = 9999.0
    else:
        pf = 0.0
    return float(wn) / float(n) if n else 0.0, avg_win, avg_loss_mag, pf, n


def kelly_fraction_from_trade_stats(*, avg_win: float, avg_loss: float, win_rate: float) -> float | None:
    if win_rate <= 0 or win_rate >= 1:
        return 0.0
    q = 1.0 - float(win_rate)
    p = float(win_rate)
    if avg_loss < 1e-9:
        return None
    b = float(avg_win) / float(avg_loss)
    if b <= 1e-12:
        return 0.0
    return (b * p - q) / b


def compute_kelly_risk_scaling(
    settings: Settings,
    *,
    pnls_newest_first: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Return multiplier in ``[KELLY_MIN_RISK_MULTIPLIER, KELLY_MAX_RISK_MULTIPLIER]`` around 1.0."""

    lim = max(1, int(settings.KELLY_LOOKBACK_TRADES))
    min_n = max(1, int(settings.KELLY_MIN_TRADES))
    window = list(pnls_newest_first[:lim])

    fallback: dict[str, float] = {
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "sample_n": float(len(window)),
        "full_kelly": 0.0,
        "modified_kelly": 0.0,
        "risk_mult_uncapped": 1.0,
    }

    if len(window) < min_n:
        return 1.0, fallback

    wr, avg_w, avg_l, pf, n = _kelly_stats_from_pnls(list(window))
    avg_l = float(avg_l)
    if avg_l < 1e-9:
        fk = 0.0
    else:
        fk_opt = kelly_fraction_from_trade_stats(avg_win=avg_w, avg_loss=avg_l, win_rate=wr)
        fk = 0.0 if fk_opt is None else float(fk_opt)
    mod_k = float(settings.KELLY_FRACTION) * max(0.0, fk)
    if fk <= 0:
        uncapped_mult = float(settings.KELLY_MIN_RISK_MULTIPLIER)
    else:
        uncapped_mult = float(min(1.0 + mod_k, float(settings.KELLY_MAX_RISK_MULTIPLIER)))
    mult = max(
        float(settings.KELLY_MIN_RISK_MULTIPLIER),
        min(float(settings.KELLY_MAX_RISK_MULTIPLIER), uncapped_mult),
    )
    capped = bool(abs(mult - uncapped_mult) > 1e-12)
    stats = {
        "win_rate": float(wr),
        "avg_win": float(avg_w),
        "avg_loss": float(avg_l),
        "profit_factor": float(pf),
        "sample_n": float(n),
        "full_kelly": float(fk),
        "modified_kelly": float(mod_k),
        "risk_mult_uncapped": float(uncapped_mult),
        "capped_bucket": 1.0 if capped else 0.0,
    }
    return float(mult), stats


@dataclass(frozen=True)
class RiskSizingDecision:
    """Kelly-adjusted risk fraction (applied after conviction / anti-martingale stacking)."""

    risk_pct: float
    sizing_mode: str
    kelly_fraction: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    reason: str
    capped: bool
    sample_n: int = 0


class KellySizer:
    """SQLite-backed fractional Kelly scaler; fail-safe to ``base_risk_pct``."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._database = database
        self._log = logging.getLogger(LOGGER_RISK)

    def get_adjusted_risk_pct(self, symbol: str, base_risk_pct: float, context: dict[str, Any]) -> RiskSizingDecision:
        del context  # Reserved for future gated sizing (kill switch, breaker, etc.)

        if (
            not bool(self._settings.ENABLE_KELLY_SIZING)
            or not math.isfinite(float(base_risk_pct))
            or float(base_risk_pct) <= 0.0
        ):
            return RiskSizingDecision(
                risk_pct=float(base_risk_pct),
                sizing_mode="baseline",
                kelly_fraction=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                reason="kelly_disabled_or_invalid_base",
                capped=False,
                sample_n=0,
            )

        pnls = list(
            self._database.get_recent_realized_pnls_for_kelly(
                limit=int(self._settings.KELLY_LOOKBACK_TRADES),
                exclude_simulation=True,
            ),
        )
        min_n = int(self._settings.KELLY_MIN_TRADES)
        if len(pnls) < min_n:
            self._log.info(
                "event=kelly_sizing_fallback symbol=%s enabled=true multiplier=1.0 sample_size=%s "
                "win_rate=n_a payoff_ratio=n_a base_risk_pct=%.8f adjusted_risk_pct=%.8f reason=insufficient_history",
                symbol,
                len(pnls),
                float(base_risk_pct),
                float(base_risk_pct),
                extra={"symbol": symbol},
            )
            return RiskSizingDecision(
                risk_pct=float(base_risk_pct),
                sizing_mode="baseline_insufficient_history",
                kelly_fraction=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                reason="insufficient_history",
                capped=False,
                sample_n=len(pnls),
            )

        mult, stats = compute_kelly_risk_scaling(self._settings, pnls_newest_first=pnls)
        sample_n = int(stats.get("sample_n", 0))
        adj = float(base_risk_pct) * float(mult)
        uncapped = float(base_risk_pct) * float(stats.get("risk_mult_uncapped", mult))
        capped = bool(abs(adj - uncapped) > 1e-12 * max(1.0, abs(base_risk_pct)))
        fk = float(stats.get("full_kelly", 0.0))
        wr = float(stats.get("win_rate", 0.0))
        pf = float(stats.get("profit_factor", 0.0))

        if not math.isfinite(adj) or adj < 0.0:
            adj = float(base_risk_pct)
            capped = True
            reason = "non_finite_kelly_fallback"
        else:
            reason = "fractional_kelly_applied"

        payoff = (float(stats.get("avg_win", 0.0)) / float(stats.get("avg_loss", 1e-12))) if float(
            stats.get("avg_loss", 0.0),
        ) > 1e-12 else 0.0

        self._log.info(
            "event=kelly_sizing_applied symbol=%s enabled=true multiplier=%.6f sample_size=%s "
            "win_rate=%.6f payoff_ratio=%.6f base_risk_pct=%.8f adjusted_risk_pct=%.8f "
            "kelly_fraction=%.6f capped=%s",
            symbol,
            float(mult),
            sample_n,
            wr,
            payoff,
            float(base_risk_pct),
            float(adj),
            fk,
            str(capped).lower(),
            extra={"symbol": symbol},
        )
        if reason != "fractional_kelly_applied" or capped:
            self._log.info(
                "event=kelly_sizing_fallback symbol=%s reason=%s multiplier=%.6f sample_size=%s",
                symbol,
                reason,
                float(mult),
                sample_n,
                extra={"symbol": symbol},
            )

        return RiskSizingDecision(
            risk_pct=float(adj),
            sizing_mode="kelly_adjusted",
            kelly_fraction=fk,
            win_rate=wr,
            avg_win=float(stats.get("avg_win", 0.0)),
            avg_loss=float(stats.get("avg_loss", 0.0)),
            profit_factor=pf,
            reason=reason,
            capped=capped,
            sample_n=sample_n,
        )


__all__ = [
    "KellySizer",
    "RiskSizingDecision",
    "compute_kelly_risk_scaling",
    "kelly_fraction_from_trade_stats",
]
