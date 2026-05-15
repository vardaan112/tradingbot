"""Cross-symbol ETF rotation by risk-adjusted momentum (Phase 3)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import sma


class ETFRotationStrategy(Strategy):
    """Rank configured ETFs by return/vol; enter top-N, exit when falling out."""

    name = "etf_rotation"

    def __init__(self, settings: Settings, **_kwargs: Any) -> None:
        self._s = settings

    def _universe(self) -> list[str]:
        raw = (self._s.ETF_ROTATION_SYMBOLS or "").strip()
        return [x.strip().upper() for x in raw.split(",") if x.strip()]

    def warmup_lookback(self) -> int:
        return int(self._s.ETF_ROTATION_LOOKBACK_BARS) + int(self._s.ETF_ROTATION_TREND_SMA) + 10

    def _scores(
        self,
        all_bars: dict[str, pd.DataFrame],
        lookback: int,
    ) -> dict[str, float] | None:
        universe = self._universe()
        closes: dict[str, pd.Series] = {}
        for u in universe:
            df = all_bars.get(u)
            if df is None or df.empty or len(df) < lookback + 2:
                return None
            closes[u] = df["close"].astype(float)

        scores: dict[str, float] = {}
        for u, c in closes.items():
            window = c.iloc[-lookback:]
            rets = window.pct_change().dropna()
            if rets.empty or float(rets.std(ddof=0) or 0.0) < 1e-12:
                scores[u] = 0.0
            else:
                mu = float(rets.mean())
                sig = float(rets.std(ddof=0))
                scores[u] = mu / sig if sig > 0 else 0.0
        return scores

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        sym = ctx.symbol.upper()
        if not self._s.ETF_ROTATION_ENABLED:
            return []

        universe = self._universe()
        if sym not in universe:
            return []

        lookback = int(self._s.ETF_ROTATION_LOOKBACK_BARS)
        all_bars = ctx.all_bars_by_symbol or {}
        scores = self._scores(all_bars, lookback)
        if scores is None:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_n = int(self._s.ETF_ROTATION_TOP_N)
        top_set = {t[0] for t in ranked[:top_n]}
        my_score = float(scores.get(sym, 0.0))
        min_score = float(self._s.ETF_ROTATION_MIN_SCORE)

        df = all_bars.get(sym)
        if df is None or df.empty:
            return []
        close = df["close"].astype(float)
        sma_s = sma(close, int(self._s.ETF_ROTATION_TREND_SMA))
        last_close = float(close.iloc[-1])
        last_sma = float(sma_s.iloc[-1])
        above_sma = last_close >= last_sma

        meta = {
            "scores": scores,
            "top_set": sorted(top_set),
            "my_score": my_score,
            "min_score": min_score,
            "above_sma": above_sma,
            "lookback": lookback,
        }

        if ctx.has_position and ctx.position is not None and str(ctx.position.side).lower() == "long":
            if sym not in top_set or my_score < min_score:
                yield Signal(
                    symbol=sym,
                    action=SignalAction.EXIT_LONG,
                    reason="etf_rotation_exit:not_top_or_weak_score",
                    reference_price=last_close,
                    atr=0.0,
                    strategy_name=self.name,
                    confidence=0.75,
                    metadata=meta,
                )
            return []

        if ctx.has_open_order:
            return []

        if sym in top_set and my_score >= min_score and above_sma:
            yield Signal(
                symbol=sym,
                action=SignalAction.ENTER_LONG,
                reason=f"etf_rotation_entry score={my_score:.4f} top={sorted(top_set)}",
                reference_price=last_close,
                atr=0.0,
                strategy_name=self.name,
                confidence=min(1.0, max(0.0, 0.5 + 0.1 * (my_score - min_score))),
                metadata=meta,
            )
