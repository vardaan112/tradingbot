"""Canonical RSI mean-reversion strategy.

This is the authoritative implementation. The legacy
`src.strategies.rsi_mean_reversion` module is preserved as a thin re-export
so existing imports continue to work.

Design notes:
- Long-only.
- One position per symbol; the orchestrator enforces MAX_OPEN_POSITIONS.
- Signals derive only from completed bars. The orchestrator's bar warmup
  is responsible for delivering historical bars; this module additionally
  drops a trailing in-progress bar if its timestamp is younger than the
  configured timeframe (defense in depth against accidental lookahead).
- RSI/ATR computation is delegated to `strategies.indicators` - never duplicated.
- Every emitted Signal carries a rich `metadata` payload containing the
  inputs that drove the decision (rsi, atr, last_close, bar_timestamp,
  quote_bid, quote_ask, spread_pct, quote_age_seconds, strategy_name).
- Every emitted signal is also surfaced via a structured `event=strategy_signal`
  log line so downstream tooling can index decisions without parsing free text.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings
from core.market_data import Quote
from utils.price_utils import spread_pct as compute_spread_pct

from .base import Signal, SignalAction, Strategy, StrategyContext
from .indicators import atr, rsi


# Mapping from BAR_TIMEFRAME string to a timedelta used to drop an in-progress
# trailing bar. Keep in sync with `core.market_data._parse_timeframe`.
_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "5Min": timedelta(minutes=5),
    "15Min": timedelta(minutes=15),
    "1Hour": timedelta(hours=1),
    "1Day": timedelta(days=1),
}


class RSIMeanReversionStrategy(Strategy):
    """Long-only RSI mean reversion with ATR exits."""

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

    # ------------------------------------------------------------------ helpers

    def _completed_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return only completed bars.

        If the trailing bar's timestamp is within the current timeframe window,
        treat it as in-progress and drop it. Idempotent: never modifies input.
        """
        if bars is None or bars.empty:
            return bars
        delta = _TIMEFRAME_DELTAS.get(self._settings.BAR_TIMEFRAME)
        if delta is None:
            return bars
        try:
            last_ts = bars.index[-1]
            if hasattr(last_ts, "to_pydatetime"):
                last_dt = last_ts.to_pydatetime()
            elif isinstance(last_ts, datetime):
                last_dt = last_ts
            else:
                return bars
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            # The bar starting at last_dt completes at last_dt + delta. If now is
            # before that, the bar is still in progress.
            if now < last_dt + delta:
                return bars.iloc[:-1]
        except (IndexError, AttributeError, TypeError):
            return bars
        return bars

    def _compute(self, bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        min_len = max(self._settings.RSI_LENGTH, self._settings.ATR_LENGTH) + 5
        if bars is None or bars.empty or len(bars) < min_len:
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

    # ------------------------------------------------------------------ logging

    def _log_signal(
        self,
        signal: Signal,
        *,
        quote: Optional[Quote],
        last_close: float,
        rsi_value: float,
        atr_value: float,
    ) -> None:
        bid = quote.bid if quote is not None else 0.0
        ask = quote.ask if quote is not None else 0.0
        try:
            sp = (
                compute_spread_pct(bid, ask)
                if quote is not None and bid > 0 and ask > bid
                else 0.0
            )
        except ValueError:
            sp = 0.0
        age = quote.age_seconds() if quote is not None else 0.0
        self._log.info(
            "event=strategy_signal symbol=%s action=%s reason=%s "
            "rsi=%.4f atr=%.6f close=%.4f bid=%.4f ask=%.4f "
            "spread_pct=%.6f quote_age_seconds=%.4f strategy=%s",
            signal.symbol,
            signal.action.value,
            signal.reason,
            rsi_value,
            atr_value,
            last_close,
            bid,
            ask,
            sp,
            age,
            self.name,
            extra={"symbol": signal.symbol, "strategy": self.name},
        )

    def _build_metadata(
        self,
        *,
        rsi_value: float,
        atr_value: float,
        last_close: float,
        bar_timestamp: Optional[datetime],
        quote: Optional[Quote],
    ) -> dict:
        bid = quote.bid if quote is not None else 0.0
        ask = quote.ask if quote is not None else 0.0
        try:
            sp = (
                compute_spread_pct(bid, ask)
                if quote is not None and bid > 0 and ask > bid
                else 0.0
            )
        except ValueError:
            sp = 0.0
        age = quote.age_seconds() if quote is not None else 0.0
        return {
            "rsi": rsi_value,
            "atr": atr_value,
            "last_close": last_close,
            "bar_timestamp": bar_timestamp.isoformat() if bar_timestamp else "",
            "quote_bid": bid,
            "quote_ask": ask,
            "spread_pct": sp,
            "quote_age_seconds": age,
            "strategy_name": self.name,
        }

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        signals: list[Signal] = []

        bars = self._completed_bars(ctx.bars)
        if bars is None or bars.empty:
            return signals

        rsi_series, atr_series = self._compute(bars)
        if rsi_series.empty or atr_series.empty:
            return signals

        last_rsi = float(rsi_series.iloc[-1])
        last_atr = float(atr_series.iloc[-1])
        last_close = float(bars["close"].iloc[-1])
        bar_ts: Optional[datetime] = None
        try:
            ts_raw = bars.index[-1]
            if hasattr(ts_raw, "to_pydatetime"):
                bar_ts = ts_raw.to_pydatetime()
            elif isinstance(ts_raw, datetime):
                bar_ts = ts_raw
        except (IndexError, AttributeError):
            bar_ts = None

        if pd.isna(last_rsi) or pd.isna(last_atr) or last_atr <= 0 or last_close <= 0:
            return signals

        symbol = ctx.symbol.upper()

        # ----------------- exits first ---------------------------------------
        if ctx.has_position:
            position = ctx.position
            if position is None:
                return signals
            if position.side.lower() != "long":
                # This strategy is long-only; defer non-long management to
                # other code paths.
                return signals

            entry_price = float(position.avg_entry_price)
            stop_dist = self._settings.ATR_STOP_MULTIPLIER * last_atr
            tp_dist = self._settings.ATR_PROFIT_MULTIPLIER * last_atr

            stop_breached = last_close <= (entry_price - stop_dist)
            tp_hit = last_close >= (entry_price + tp_dist)
            rsi_exit = last_rsi >= self._settings.RSI_EXIT

            entry_idx = self._entry_bar_index.get(symbol, len(bars) - 1)
            held_bars = max(0, len(bars) - 1 - entry_idx)
            time_exit = held_bars >= self._settings.MAX_HOLD_BARS

            if stop_breached:
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.EMERGENCY_EXIT_LONG,
                    reason=(
                        f"atr_stop_breach close={last_close:.4f} "
                        f"entry={entry_price:.4f} stop_dist={stop_dist:.4f}"
                    ),
                    reference_price=last_close,
                    atr=last_atr,
                    metadata=self._build_metadata(
                        rsi_value=last_rsi,
                        atr_value=last_atr,
                        last_close=last_close,
                        bar_timestamp=bar_ts,
                        quote=ctx.quote,
                    ),
                )
                signals.append(signal)
                self._log_signal(
                    signal,
                    quote=ctx.quote,
                    last_close=last_close,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                )
                self._entry_bar_index.pop(symbol, None)
                return signals

            if tp_hit or rsi_exit or time_exit:
                if tp_hit:
                    reason = "tp_hit"
                elif rsi_exit:
                    reason = "rsi_exit"
                else:
                    reason = "time_exit"
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.EXIT_LONG,
                    reason=f"{reason} close={last_close:.4f} rsi={last_rsi:.2f}",
                    reference_price=last_close,
                    atr=last_atr,
                    metadata=self._build_metadata(
                        rsi_value=last_rsi,
                        atr_value=last_atr,
                        last_close=last_close,
                        bar_timestamp=bar_ts,
                        quote=ctx.quote,
                    ),
                )
                signals.append(signal)
                self._log_signal(
                    signal,
                    quote=ctx.quote,
                    last_close=last_close,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                )
                self._entry_bar_index.pop(symbol, None)
                return signals

            return signals

        # ----------------- entries -------------------------------------------
        # Pre-entry guards. The orchestrator's UniverseFilter is the
        # authoritative spread/quote-staleness/eligibility gate; this strategy
        # only suppresses obviously non-actionable cases here.
        if ctx.has_open_order:
            return signals
        if ctx.quote is None:
            return signals
        if last_rsi >= self._settings.RSI_OVERSOLD:
            return signals

        signal = Signal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            reason=f"rsi_oversold rsi={last_rsi:.2f} close={last_close:.4f}",
            reference_price=last_close,
            atr=last_atr,
            metadata=self._build_metadata(
                rsi_value=last_rsi,
                atr_value=last_atr,
                last_close=last_close,
                bar_timestamp=bar_ts,
                quote=ctx.quote,
            ),
        )
        signals.append(signal)
        self._log_signal(
            signal,
            quote=ctx.quote,
            last_close=last_close,
            rsi_value=last_rsi,
            atr_value=last_atr,
        )
        self._entry_bar_index[symbol] = len(bars) - 1
        return signals
