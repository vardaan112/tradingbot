"""Pure helpers for risk-on / risk-off style overlays from benchmark bars (Phase 3).

No I/O. Not wired into order flow; strategies and orchestrator may consume later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .filters import sma
from .indicators import atr, rsi


@dataclass(frozen=True)
class RiskRegimeOverlay:
    """Lightweight overlay snapshot for logging or future gating."""

    risk_on: bool
    confidence: float
    spy_trend_up: bool
    spy_volatile: bool
    qqq_trend_up: bool
    reasons: tuple[str, ...]
    metadata: dict[str, Any]


def _need_cols(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty and {"close", "high", "low"}.issubset(df.columns)


def compute_risk_regime_overlay(
    spy_bars: pd.DataFrame,
    qqq_bars: pd.DataFrame,
    *,
    sma_length: int = 50,
    rsi_length: int = 14,
    atr_length: int = 14,
    vol_atr_pct_high: float = 0.03,
) -> RiskRegimeOverlay | None:
    """Derive a simple risk-on view from SPY/QQQ completed-bar history.

    - ``risk_on`` when both trends are non-down (close vs SMA) and SPY is not
      in a high short-term vol spike (ATR% vs close).
    """

    if not _need_cols(spy_bars) or not _need_cols(qqq_bars):
        return None
    if len(spy_bars) < sma_length + atr_length + 5 or len(qqq_bars) < sma_length + 5:
        return None

    s_close = spy_bars["close"].astype(float)
    s_high = spy_bars["high"].astype(float)
    s_low = spy_bars["low"].astype(float)
    q_close = qqq_bars["close"].astype(float)

    spy_sma = sma(s_close, sma_length)
    qqq_sma = sma(q_close, sma_length)
    spy_last = float(s_close.iloc[-1])
    qqq_last = float(q_close.iloc[-1])
    spy_s = float(spy_sma.iloc[-1])
    qqq_s = float(qqq_sma.iloc[-1])

    atr_s = atr(s_high, s_low, s_close, length=atr_length)
    atr_pct = float(atr_s.iloc[-1] / spy_last) if spy_last > 0 else 0.0
    spy_volatile = atr_pct >= vol_atr_pct_high

    spy_rsi = rsi(s_close, length=rsi_length)
    qqq_rsi = rsi(q_close, length=rsi_length)
    spy_trend_up = spy_last >= spy_s
    qqq_trend_up = qqq_last >= qqq_s

    reasons: list[str] = []
    if not spy_trend_up:
        reasons.append("spy_below_sma")
    if not qqq_trend_up:
        reasons.append("qqq_below_sma")
    if spy_volatile:
        reasons.append("spy_high_atr_pct")

    risk_on = spy_trend_up and qqq_trend_up and not spy_volatile
    n_ok = sum([spy_trend_up, qqq_trend_up, not spy_volatile])
    confidence = n_ok / 3.0

    return RiskRegimeOverlay(
        risk_on=risk_on,
        confidence=confidence,
        spy_trend_up=spy_trend_up,
        spy_volatile=spy_volatile,
        qqq_trend_up=qqq_trend_up,
        reasons=tuple(reasons),
        metadata={
            "spy_close": spy_last,
            "spy_sma": spy_s,
            "qqq_close": qqq_last,
            "qqq_sma": qqq_s,
            "spy_atr_pct": atr_pct,
            "spy_rsi": float(spy_rsi.iloc[-1]),
            "qqq_rsi": float(qqq_rsi.iloc[-1]),
        },
    )


__all__ = ["RiskRegimeOverlay", "compute_risk_regime_overlay"]
