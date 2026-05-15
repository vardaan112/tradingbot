"""Long-only momentum trend-following strategy (Phase 3)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import adx, sma
from .indicators import atr, rsi


class MomentumTrendStrategy(Strategy):
    """Buy strength: SMA stack, N-bar return, optional ADX + volume confirmation."""

    name = "momentum_trend"

    def __init__(self, settings: Settings, **_kwargs: Any) -> None:
        self._s = settings
        self._entry_bar_index: dict[str, int] = {}

    def warmup_lookback(self) -> int:
        return max(
            int(self._s.MOMENTUM_SLOW_SMA) + 10,
            int(self._s.MOMENTUM_LOOKBACK_BARS) + int(self._s.MOMENTUM_FAST_SMA) + 10,
            int(self._s.MOMENTUM_EXIT_SMA) + 10,
            int(self._s.MOMENTUM_VOLUME_MA_BARS) + 10,
            int(self._s.ADX_LENGTH) * 3,
            int(self._s.ATR_LENGTH) * 3,
            80,
        )

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        sym = ctx.symbol.upper()
        if not self._s.MOMENTUM_ENABLED:
            return []

        bars = ctx.bars
        min_bars = self.warmup_lookback()
        if bars is None or bars.empty or len(bars) < min_bars:
            return []

        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        vol = bars["volume"].astype(float)

        fast_n = int(self._s.MOMENTUM_FAST_SMA)
        slow_n = int(self._s.MOMENTUM_SLOW_SMA)
        exit_n = int(self._s.MOMENTUM_EXIT_SMA)
        lb = int(self._s.MOMENTUM_LOOKBACK_BARS)

        sma_f = sma(close, fast_n)
        sma_s = sma(close, slow_n)
        sma_exit = sma(close, exit_n)
        adx_s = adx(high, low, close, length=int(self._s.ADX_LENGTH))
        atr_s = atr(high, low, close, length=int(self._s.ATR_LENGTH))
        vol_ma = vol.rolling(int(self._s.MOMENTUM_VOLUME_MA_BARS), min_periods=int(self._s.MOMENTUM_VOLUME_MA_BARS)).mean()

        last_close = float(close.iloc[-1])
        last_atr = float(atr_s.iloc[-1])
        sf = float(sma_f.iloc[-1])
        ss = float(sma_s.iloc[-1])
        sx = float(sma_exit.iloc[-1])
        adx_v = float(adx_s.iloc[-1])
        if pd.isna(adx_v):
            adx_v = 0.0
        vol_ok = float(vol.iloc[-1]) >= float(self._s.MOMENTUM_VOLUME_FACTOR) * float(vol_ma.iloc[-1])

        past = close.shift(lb)
        ret_pct = float(last_close / float(past.iloc[-1]) - 1.0) if float(past.iloc[-1]) > 0 else 0.0
        mom_1 = float(close.iloc[-1] - close.iloc[-2])
        rsi_s = rsi(close, length=14)

        meta_base = {
            "sma_fast": sf,
            "sma_slow": ss,
            "sma_exit": sx,
            "adx": adx_v,
            "atr": last_atr,
            "return_pct": ret_pct,
            "volume_ok": vol_ok,
            "rsi": float(rsi_s.iloc[-1]),
            "thresholds": {
                "min_return_pct": float(self._s.MOMENTUM_MIN_RETURN_PCT),
                "adx_min": float(self._s.MOMENTUM_ADX_MIN),
                "require_adx": bool(self._s.MOMENTUM_REQUIRE_ADX),
            },
        }

        def _conf(components: dict[str, bool]) -> float:
            ok = sum(1 for v in components.values() if v)
            return max(0.0, min(1.0, ok / max(1, len(components))))

        if ctx.has_position and ctx.position is not None and str(ctx.position.side).lower() == "long":
            entry_ix = self._entry_bar_index.get(sym, len(bars) - 1)
            held = max(0, len(bars) - 1 - entry_ix)
            stop_px = float(ctx.position.avg_entry_price) - float(self._s.MOMENTUM_ATR_STOP_MULT) * last_atr
            exit_sma_break = last_close < sx
            neg_mom = mom_1 < 0 and float(close.iloc[-1]) < float(close.iloc[-2])
            atr_stop = last_close <= stop_px + 1e-12
            time_exit = held >= int(self._s.MOMENTUM_MAX_HOLD_BARS)

            if exit_sma_break or neg_mom or atr_stop or time_exit:
                tag = (
                    "exit_sma_break"
                    if exit_sma_break
                    else ("neg_momentum" if neg_mom else ("atr_stop" if atr_stop else "max_hold"))
                )
                self._entry_bar_index.pop(sym, None)
                yield Signal(
                    symbol=sym,
                    action=SignalAction.EXIT_LONG,
                    reason=f"momentum_exit:{tag} close={last_close:.4f}",
                    reference_price=last_close,
                    atr=last_atr,
                    strategy_name=self.name,
                    confidence=0.85,
                    metadata={**meta_base, "exit_reason": tag, "held_bars": held},
                )
            return []

        if ctx.has_open_order:
            return []

        need_adx = bool(self._s.MOMENTUM_REQUIRE_ADX) and adx_v >= float(self._s.MOMENTUM_ADX_MIN)
        adx_ok = (not bool(self._s.MOMENTUM_REQUIRE_ADX)) or need_adx
        stack = sf > ss and last_close > sf
        ret_ok = ret_pct >= float(self._s.MOMENTUM_MIN_RETURN_PCT)

        components = {"stack": stack, "return": ret_ok, "adx": adx_ok, "volume": vol_ok}
        if not (stack and ret_ok and adx_ok and vol_ok):
            return []

        self._entry_bar_index[sym] = len(bars) - 1
        yield Signal(
            symbol=sym,
            action=SignalAction.ENTER_LONG,
            reason=(
                f"momentum_entry ret={ret_pct:.4f} adx={adx_v:.2f} "
                f"sf={sf:.4f} ss={ss:.4f}"
            ),
            reference_price=last_close,
            atr=last_atr,
            strategy_name=self.name,
            confidence=_conf(components),
            metadata={**meta_base, "entry_checks": components},
        )
