"""Long-only pullback toward rolling VWAP in an uptrend (Phase 3)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import adx, sma
from .indicators import atr, rsi, rolling_vwap_zscore_bands


class VWAPPullbackStrategy(Strategy):
    """Uptrend + controlled dip toward VWAP; exits on mean reversion or ATR rules."""

    name = "vwap_pullback"

    def __init__(self, settings: Settings, **_kwargs: Any) -> None:
        self._s = settings
        self._entry_bar_index: dict[str, int] = {}
        self._entry_price: dict[str, float] = {}

    def warmup_lookback(self) -> int:
        return max(
            int(self._s.VWAP_PULLBACK_LENGTH) + 15,
            int(self._s.VWAP_PULLBACK_TREND_SLOW_SMA) + 10,
            int(self._s.VWAP_PULLBACK_TREND_FAST_SMA) + 10,
            int(self._s.ADX_LENGTH) * 3,
            int(self._s.ATR_LENGTH) * 3,
            80,
        )

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        sym = ctx.symbol.upper()
        if not self._s.VWAP_PULLBACK_ENABLED:
            return []

        bars = ctx.bars
        min_bars = self.warmup_lookback()
        if bars is None or bars.empty or len(bars) < min_bars:
            return []

        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        vol = bars["volume"].astype(float)

        L = int(self._s.VWAP_PULLBACK_LENGTH)
        z_th = float(self._s.VWAP_PULLBACK_Z_THRESHOLD)
        vwap, upper, lower, zscore, dist_pct, dev = rolling_vwap_zscore_bands(
            high,
            low,
            close,
            vol,
            length=L,
            z_threshold=z_th,
        )

        fast_n = int(self._s.VWAP_PULLBACK_TREND_FAST_SMA)
        slow_n = int(self._s.VWAP_PULLBACK_TREND_SLOW_SMA)
        sma_f = sma(close, fast_n)
        sma_s = sma(close, slow_n)
        adx_s = adx(high, low, close, length=int(self._s.ADX_LENGTH))
        atr_s = atr(high, low, close, length=int(self._s.ATR_LENGTH))
        rsi_s = rsi(close, length=14)

        last_close = float(close.iloc[-1])
        last_atr = float(atr_s.iloc[-1])
        vw = float(vwap.iloc[-1])
        up = float(upper.iloc[-1])
        z = float(zscore.iloc[-1])
        dp = float(dist_pct.iloc[-1])
        sf = float(sma_f.iloc[-1])
        ss = float(sma_s.iloc[-1])
        adxv = float(adx_s.iloc[-1])
        rv = float(rsi_s.iloc[-1])

        slope = float(sma_f.iloc[-1] - sma_f.iloc[-5]) if len(sma_f) >= 5 else 0.0
        max_dist = float(self._s.VWAP_PULLBACK_MAX_DISTANCE_PCT)

        meta_base = {
            "vwap": vw,
            "vwap_upper": up,
            "zscore": z,
            "distance_pct": dp,
            "sma_fast": sf,
            "sma_slow": ss,
            "adx": adxv,
            "atr": last_atr,
            "rsi": rv,
            "thresholds": {
                "max_distance_pct": max_dist,
                "rsi_min": float(self._s.VWAP_PULLBACK_RSI_MIN),
                "rsi_max": float(self._s.VWAP_PULLBACK_RSI_MAX),
                "adx_min": float(self._s.VWAP_PULLBACK_ADX_MIN),
                "adx_max": float(self._s.VWAP_PULLBACK_ADX_MAX),
            },
        }

        uptrend = sf > ss and last_close > ss and slope >= float(self._s.VWAP_PULLBACK_MIN_TREND_SLOPE)
        adx_ok = float(self._s.VWAP_PULLBACK_ADX_MIN) <= adxv <= float(self._s.VWAP_PULLBACK_ADX_MAX)
        rsi_ok = float(self._s.VWAP_PULLBACK_RSI_MIN) <= rv <= float(self._s.VWAP_PULLBACK_RSI_MAX)
        # Pullback: at or below VWAP (typical-price VWAP) within max distance
        near_vwap = vw > 0 and last_close <= vw * (1.0 + max_dist) and last_close >= vw * (1.0 - max_dist * 2)
        z_pullback = z <= float(self._s.VWAP_PULLBACK_MAX_ZSCORE)

        if ctx.has_position and ctx.position is not None and str(ctx.position.side).lower() == "long":
            entry_px = self._entry_price.get(sym, float(ctx.position.avg_entry_price))
            entry_ix = self._entry_bar_index.get(sym, len(bars) - 1)
            held = max(0, len(bars) - 1 - entry_ix)
            stop_px = entry_px - float(self._s.VWAP_PULLBACK_ATR_STOP_MULT) * last_atr
            target_px = entry_px + float(self._s.VWAP_PULLBACK_ATR_TARGET_MULT) * last_atr

            mean_recover = last_close >= vw - 1e-12 and z >= float(self._s.VWAP_PULLBACK_EXIT_Z_MIN)
            trend_bad = last_close < ss or sf < ss
            atr_stop = last_close <= stop_px + 1e-12
            target_hit = last_close >= target_px - 1e-12
            time_exit = held >= int(self._s.VWAP_PULLBACK_MAX_HOLD_BARS)

            if mean_recover or trend_bad or atr_stop or target_hit or time_exit:
                tag = (
                    "vwap_recover"
                    if mean_recover
                    else (
                        "trend_invalid"
                        if trend_bad
                        else ("atr_stop" if atr_stop else ("atr_target" if target_hit else "max_hold"))
                    )
                )
                self._entry_bar_index.pop(sym, None)
                self._entry_price.pop(sym, None)
                yield Signal(
                    symbol=sym,
                    action=SignalAction.EXIT_LONG,
                    reason=f"vwap_pullback_exit:{tag} close={last_close:.4f}",
                    reference_price=last_close,
                    atr=last_atr,
                    strategy_name=self.name,
                    confidence=0.8,
                    metadata={**meta_base, "exit_reason": tag, "held_bars": held},
                )
            return []

        if ctx.has_open_order:
            return []

        if not (uptrend and adx_ok and rsi_ok and near_vwap and z_pullback):
            return []

        self._entry_bar_index[sym] = len(bars) - 1
        self._entry_price[sym] = last_close
        yield Signal(
            symbol=sym,
            action=SignalAction.ENTER_LONG,
            reason=f"vwap_pullback_entry z={z:.3f} close={last_close:.4f} vwap={vw:.4f}",
            reference_price=last_close,
            atr=last_atr,
            strategy_name=self.name,
            confidence=0.7,
            metadata={**meta_base, "entry_checks": {"uptrend": uptrend, "adx_ok": adx_ok, "rsi_ok": rsi_ok}},
        )
