"""Long-only RSI mean reversion strategy.

Conservative defaults:
- 1 symbol position at a time (enforced by orchestrator MAX_OPEN_POSITIONS)
- Entries only when RSI(14) < RSI_OVERSOLD AND no current position / open order
- Exits when RSI > RSI_EXIT, ATR profit target hit, ATR stop breached, or
  MAX_HOLD_BARS exceeded.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext
from .indicators import atr, rsi


class RSIMeanReversionStrategy(Strategy):
    name = "rsi_meanrev"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._log = logging.getLogger(LOGGER_STRATEGY)
        self._entry_bar_index: dict[str, int] = {}

    def warmup_lookback(self) -> int:
        return max(
            self._settings.RSI_LENGTH * 4,
            self._settings.ATR_LENGTH * 4,
            150,
        )

    # ---------------------------------------------------------------- helpers

    def _compute(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if bars.empty or len(bars) < max(self._settings.RSI_LENGTH, self._settings.ATR_LENGTH) + 5:
            empty = pd.Series(dtype=float)
            return empty, empty
        rsi_series = rsi(bars["close"], length=self._settings.RSI_LENGTH)
        atr_series = atr(
            bars["high"],
            bars["low"],
            bars["close"],
            length=self._settings.ATR_LENGTH,
        )
        return rsi_series, atr_series

    # ---------------------------------------------------------------- evaluate

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        signals: list[Signal] = []
        bars = ctx.bars
        if bars is None or bars.empty:
            return signals

        # Use only completed bars: drop the in-progress one if the last bar
        # timestamp is within the last partial period.
        # The orchestrator should pass already-completed bars; we still tail.
        rsi_series, atr_series = self._compute(bars)
        if rsi_series.empty or atr_series.empty:
            return signals

        last_rsi = float(rsi_series.iloc[-1])
        last_atr = float(atr_series.iloc[-1])
        last_close = float(bars["close"].iloc[-1])

        if pd.isna(last_rsi) or pd.isna(last_atr) or last_atr <= 0 or last_close <= 0:
            return signals

        symbol = ctx.symbol.upper()

        # ----------------- exits first ---------------------------------------
        if ctx.has_position:
            position = ctx.position
            if position is None:
                return signals

            entry_price = float(position.avg_entry_price)
            stop_dist = self._settings.ATR_STOP_MULTIPLIER * last_atr
            tp_dist = self._settings.ATR_PROFIT_MULTIPLIER * last_atr

            stop_breached = last_close <= (entry_price - stop_dist) and position.side.lower() == "long"
            tp_hit = last_close >= (entry_price + tp_dist) and position.side.lower() == "long"
            rsi_exit = last_rsi >= self._settings.RSI_EXIT and position.side.lower() == "long"

            entry_idx = self._entry_bar_index.get(symbol, len(bars) - 1)
            held_bars = max(0, len(bars) - 1 - entry_idx)
            time_exit = held_bars >= self._settings.MAX_HOLD_BARS

            if stop_breached:
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=SignalAction.EMERGENCY_EXIT_LONG,
                        reason=f"atr_stop_breach close={last_close:.4f} entry={entry_price:.4f} stop_dist={stop_dist:.4f}",
                        reference_price=last_close,
                        atr=last_atr,
                    )
                )
                self._entry_bar_index.pop(symbol, None)
                return signals

            if tp_hit or rsi_exit or time_exit:
                reason = (
                    "tp_hit"
                    if tp_hit
                    else ("rsi_exit" if rsi_exit else "time_exit")
                )
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=SignalAction.EXIT_LONG,
                        reason=f"{reason} close={last_close:.4f} rsi={last_rsi:.2f}",
                        reference_price=last_close,
                        atr=last_atr,
                    )
                )
                self._entry_bar_index.pop(symbol, None)
                return signals

            return signals

        # ----------------- entries -------------------------------------------
        if ctx.has_open_order:
            return signals

        if ctx.quote is None:
            return signals

        if last_rsi >= self._settings.RSI_OVERSOLD:
            return signals

        signals.append(
            Signal(
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                reason=f"rsi_oversold rsi={last_rsi:.2f} close={last_close:.4f}",
                reference_price=last_close,
                atr=last_atr,
                metadata={"rsi": last_rsi, "close": last_close},
            )
        )
        self._entry_bar_index[symbol] = len(bars) - 1
        return signals
