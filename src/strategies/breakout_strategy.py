"""Long-only rolling-high breakout with volume confirmation (Phase 3)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext
from .indicators import atr


class BreakoutStrategy(Strategy):
    """Enter when close clears prior-window high plus ATR buffer; volume-confirmed."""

    name = "breakout"

    def __init__(self, settings: Settings, **_kwargs: Any) -> None:
        self._s = settings
        self._entry_bar_index: dict[str, int] = {}
        self._breakout_level: dict[str, float] = {}
        self._trail_high: dict[str, float] = {}

    def warmup_lookback(self) -> int:
        return max(
            int(self._s.BREAKOUT_LOOKBACK_BARS) + int(self._s.BREAKOUT_VOLUME_MA_BARS) + 15,
            int(self._s.ATR_LENGTH) * 3,
            60,
        )

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        sym = ctx.symbol.upper()
        if not self._s.BREAKOUT_ENABLED:
            return []

        bars = ctx.bars
        min_bars = self.warmup_lookback()
        if bars is None or bars.empty or len(bars) < min_bars:
            return []

        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        vol = bars["volume"].astype(float)

        lb = int(self._s.BREAKOUT_LOOKBACK_BARS)
        atr_s = atr(high, low, close, length=int(self._s.ATR_LENGTH))
        vma = vol.rolling(int(self._s.BREAKOUT_VOLUME_MA_BARS), min_periods=int(self._s.BREAKOUT_VOLUME_MA_BARS)).mean()

        last_close = float(close.iloc[-1])
        last_atr = float(atr_s.iloc[-1])
        prev_slice = high.iloc[-(lb + 1) : -1]
        if prev_slice.empty:
            return []
        resist = float(prev_slice.max())
        buffer = float(self._s.BREAKOUT_ATR_BUFFER_MULTIPLIER) * last_atr
        breakout_line = resist + buffer
        vol_ok = float(vol.iloc[-1]) >= float(self._s.BREAKOUT_VOLUME_MULTIPLIER) * float(vma.iloc[-1])

        meta_base = {
            "resistance": resist,
            "breakout_line": breakout_line,
            "atr": last_atr,
            "volume_ok": vol_ok,
            "thresholds": {
                "volume_mult": float(self._s.BREAKOUT_VOLUME_MULTIPLIER),
                "atr_buffer_mult": float(self._s.BREAKOUT_ATR_BUFFER_MULTIPLIER),
            },
        }

        if ctx.has_position and ctx.position is not None and str(ctx.position.side).lower() == "long":
            entry_px = float(ctx.position.avg_entry_price)
            entry_ix = self._entry_bar_index.get(sym, len(bars) - 1)
            held = max(0, len(bars) - 1 - entry_ix)
            level = self._breakout_level.get(sym, resist)
            stop_px = entry_px - float(self._s.BREAKOUT_ATR_STOP_MULT) * last_atr
            tp_px = entry_px + float(self._s.BREAKOUT_ATR_TARGET_MULT) * last_atr

            th = self._trail_high.get(sym, last_close)
            th = max(th, last_close)
            self._trail_high[sym] = th
            trail_stop = th - float(self._s.BREAKOUT_TRAIL_ATR_MULT) * last_atr

            failed = last_close < level - 1e-12
            atr_hit = last_close <= stop_px + 1e-12
            trail_hit = last_close <= trail_stop + 1e-12
            target_hit = last_close >= tp_px - 1e-12
            time_exit = held >= int(self._s.BREAKOUT_MAX_HOLD_BARS)

            if failed or atr_hit or trail_hit or target_hit or time_exit:
                tag = (
                    "failed_breakout"
                    if failed
                    else (
                        "atr_stop"
                        if atr_hit
                        else (
                            "trail_stop"
                            if trail_hit
                            else ("profit_target" if target_hit else "max_hold")
                        )
                    )
                )
                self._entry_bar_index.pop(sym, None)
                self._breakout_level.pop(sym, None)
                self._trail_high.pop(sym, None)
                yield Signal(
                    symbol=sym,
                    action=SignalAction.EXIT_LONG,
                    reason=f"breakout_exit:{tag} close={last_close:.4f}",
                    reference_price=last_close,
                    atr=last_atr,
                    strategy_name=self.name,
                    confidence=0.85,
                    metadata={
                        **meta_base,
                        "exit_reason": tag,
                        "breakout_level": level,
                        "held_bars": held,
                        "trail_stop": trail_stop,
                    },
                )
            return []

        if ctx.has_open_order:
            return []

        broke = last_close > breakout_line + 1e-12
        if not (broke and vol_ok):
            return []

        self._entry_bar_index[sym] = len(bars) - 1
        self._breakout_level[sym] = resist
        self._trail_high[sym] = last_close
        yield Signal(
            symbol=sym,
            action=SignalAction.ENTER_LONG,
            reason=f"breakout_entry close={last_close:.4f} above={breakout_line:.4f}",
            reference_price=last_close,
            atr=last_atr,
            strategy_name=self.name,
            confidence=0.75 if vol_ok else 0.5,
            metadata={**meta_base, "volume_confirmed": vol_ok},
        )
