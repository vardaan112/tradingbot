"""Canonical RSI mean-reversion strategy (regime-aware, synthetic trailing-profit).

This is the authoritative implementation. `rsi_mean_reversion.py` stays a shim.

Phase Two adds:
- Regime filtering (ADX + 200 SMA / slope gating via `filters.compute_regime_snapshot`)
- Conviction-aware sizing cues (wired through ``Signal.metadata``)
- Profit-protecting synthetic trailing stop (no broker trailing orders)
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings
from config.strategy_runtime import StrategyRuntimeThresholds
from core.database import Database
from core.market_data import Quote
from core.state_store import StateStore, TrailTrailingRecord
from utils.price_utils import spread_pct as compute_spread_pct

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import RegimeSnapshot, compute_regime_snapshot
from .indicators import atr, rsi
from .sentiment import sentiment_overlay_neutral
from .skip_diagnostics import (
    SkipCodes,
    SkipDiagnosticsThrottle,
    SkipReason,
    emit_skip_diagnostic,
    regime_skip_reason,
)

# Mapping from BAR_TIMEFRAME string to a timedelta used to drop an in-progress
# trailing bar. Keep in sync with `core.market_data._parse_timeframe`.
_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "5Min": timedelta(minutes=5),
    "15Min": timedelta(minutes=15),
    "1Hour": timedelta(hours=1),
    "1Day": timedelta(days=1),
}


def _stub_regime(settings: Settings) -> RegimeSnapshot:
    """Conservative sentinel when insufficient bars prevent regime estimation."""
    return RegimeSnapshot(
        adx=0.0,
        adx_length=settings.ADX_LENGTH,
        sma200=0.0,
        sma_length=settings.SMA_FILTER_LENGTH,
        sma_slope=0.0,
        sma_slope_lookback=settings.SMA_SLOPE_LOOKBACK_BARS,
        price_above_sma200=False,
        regime_type="Range",
        high_conviction=False,
        allow_rsi_long=False,
        reason="insufficient_bars_for_regime_filters",
    )


@dataclass
class TrailState:
    """Per-symbol trailing-profit engine state."""

    avg_entry_price: float
    trailing_stop_active: bool
    locked_floor: float  # meaningful once active (>0)
    highest_close_since_activation: float
    trailing_stop_price: float
    target_a_hit: bool


class RSIMeanReversionStrategy(Strategy):
    """Long-only RSI mean reversion with catastrophe ATR stops + trailing profit."""

    name = "rsi_meanrev"

    def __init__(
        self,
        settings: Settings,
        state_store: StateStore | None = None,
        database: Database | None = None,
        runtime_thresholds: StrategyRuntimeThresholds | None = None,
        ml_filter: Any | None = None,
        discord_embed_fn: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._settings = settings
        thr = runtime_thresholds or StrategyRuntimeThresholds(
            rsi_oversold=float(settings.RSI_OVERSOLD),
            rsi_exit=float(settings.RSI_EXIT),
            adx_range_max=float(settings.ADX_RANGE_MAX),
            atr_stop_multiplier=float(settings.ATR_STOP_MULTIPLIER),
            trail_atr_multiplier=float(settings.TRAIL_ATR_MULTIPLIER),
        )
        self._thr = thr
        self._ml_filter = ml_filter
        self._discord_embed_fn = discord_embed_fn
        self._log = logging.getLogger(LOGGER_STRATEGY)
        self._entry_bar_index: dict[str, int] = {}
        self._state_store = state_store
        self._database = database
        self._trails_by_symbol: dict[str, TrailState] = {}
        self._skip_diag_throttle = SkipDiagnosticsThrottle()
        self._load_trailing_from_disk()

    def set_runtime_thresholds(self, rt: StrategyRuntimeThresholds) -> None:
        self._thr = rt

    def _risk_overlay_settings(self) -> Settings:
        return self._settings.model_copy(
            update={
                "RSI_OVERSOLD": float(self._thr.rsi_oversold),
                "RSI_EXIT": float(self._thr.rsi_exit),
                "ADX_RANGE_MAX": float(self._thr.adx_range_max),
                "ATR_STOP_MULTIPLIER": float(self._thr.atr_stop_multiplier),
                "TRAIL_ATR_MULTIPLIER": float(self._thr.trail_atr_multiplier),
            },
        )

    def warmup_lookback(self) -> int:
        return max(
            self._settings.RSI_LENGTH * 4,
            self._settings.ATR_LENGTH * 4,
            self._settings.ADX_LENGTH * 4,
            self._settings.SMA_FILTER_LENGTH + self._settings.SMA_SLOPE_LOOKBACK_BARS + 25,
            150,
        )

    # -------------------------------------------------------------- persistence

    def _load_trailing_from_disk(self) -> None:
        if self._state_store is None:
            return
        for sym, record in self._state_store.load_trailing_states().items():
            self._trails_by_symbol[sym.upper()] = TrailState(
                avg_entry_price=float(record.avg_entry_price),
                trailing_stop_active=bool(record.trailing_stop_active),
                locked_floor=float(record.locked_floor),
                highest_close_since_activation=float(record.highest_close_since_activation),
                trailing_stop_price=float(record.trailing_stop_price),
                target_a_hit=bool(record.target_a_hit),
            )

    def _persist_trailing_to_disk(self) -> None:
        if self._state_store is None:
            return
        payload: dict[str, TrailTrailingRecord] = {}
        for sym, st in self._trails_by_symbol.items():
            payload[sym.upper()] = TrailTrailingRecord(
                symbol=sym.upper(),
                avg_entry_price=st.avg_entry_price,
                trailing_stop_active=st.trailing_stop_active,
                locked_floor=st.locked_floor,
                highest_close_since_activation=st.highest_close_since_activation,
                trailing_stop_price=st.trailing_stop_price,
                target_a_hit=st.target_a_hit,
            )
        self._state_store.save_trailing_states(payload)

    # ------------------------------------------------------------------ helpers

    def _completed_bars(self, bars: pd.DataFrame) -> pd.DataFrame:
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
                last_dt = last_dt.replace(tzinfo=UTC)
            now = datetime.now(UTC)
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

    def _entry_anchor_epsilon(self, entry_price: float) -> float:
        return max(1e-6, abs(entry_price) * 1e-4)

    def _get_or_bind_trail(self, symbol: str, entry_price: float) -> TrailState:
        existing = self._trails_by_symbol.get(symbol.upper())
        if existing is None:
            st = TrailState(
                avg_entry_price=float(entry_price),
                trailing_stop_active=False,
                locked_floor=0.0,
                highest_close_since_activation=0.0,
                trailing_stop_price=0.0,
                target_a_hit=False,
            )
            self._trails_by_symbol[symbol.upper()] = st
            return st
        if abs(existing.avg_entry_price - float(entry_price)) > self._entry_anchor_epsilon(
            entry_price,
        ):
            st = TrailState(
                avg_entry_price=float(entry_price),
                trailing_stop_active=False,
                locked_floor=0.0,
                highest_close_since_activation=0.0,
                trailing_stop_price=0.0,
                target_a_hit=False,
            )
            self._trails_by_symbol[symbol.upper()] = st
            return st
        return existing

    def _clear_trail(self, symbol: str) -> None:
        self._trails_by_symbol.pop(symbol.upper(), None)

    def _regime_overlay(
        self,
        base_meta: dict[str, object],
        regime: RegimeSnapshot,
        *,
        trailing: Mapping[str, object],
        conviction_multiplier: float | None,
        ctx: StrategyContext | None = None,
    ) -> dict[str, object]:
        out = dict(base_meta)
        out.update(
            {
                "regime_type": regime.regime_type,
                "adx": regime.adx,
                "sma200": regime.sma200,
                "sma_slope": regime.sma_slope,
                "price_above_sma200": regime.price_above_sma200,
                "high_conviction": regime.high_conviction,
                "allow_rsi_long": regime.allow_rsi_long,
            }
        )
        out.update(dict(trailing))
        if conviction_multiplier is not None:
            out["conviction_risk_multiplier"] = conviction_multiplier
        if ctx is not None:
            ovl = ctx.sentiment_overlay or sentiment_overlay_neutral(ctx.symbol)
            out.update(dict(ovl))
            if ctx.anti_martingale_risk_mode:
                out["risk_mode"] = ctx.anti_martingale_risk_mode
            if ctx.anti_martingale_multiplier is not None:
                out["anti_martingale_multiplier"] = float(ctx.anti_martingale_multiplier)
            if ctx.recent_trade_outcomes_hint:
                out["recent_trade_outcomes"] = ctx.recent_trade_outcomes_hint
        return out

    def _sector_for_symbol(self, symbol: str) -> str:
        return self._settings.sector_for_symbol(symbol)

    def _resolve_rsi_entry_threshold(
        self,
        *,
        atr_value: float,
        last_close: float,
    ) -> tuple[float, float, str]:
        atr_pct = (atr_value / last_close) if last_close > 0 else 0.0
        high_vol = atr_pct > float(self._settings.HIGH_VOL_ATR_PCT_THRESHOLD)
        threshold = (
            float(self._settings.HIGH_VOL_RSI_ENTRY)
            if high_vol
            else float(self._settings.DEFAULT_RSI_ENTRY)
        )
        tier = "HIGH_VOL" if high_vol else "NORMAL_VOL"
        return threshold, atr_pct, tier

    # ------------------------------------------------------------------ logging

    def _log_signal(
        self,
        signal: Signal,
        *,
        quote: Quote | None,
        last_close: float,
        rsi_value: float,
        atr_value: float,
        regime_type: str,
        trailing_active: bool,
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
        ov = getattr(signal, "metadata", {}) or {}
        sentiment_score_txt = ov.get("sentiment_score")
        sentiment_label_txt = ov.get("sentiment_label")
        sentiment_score_repr = (
            f"{float(sentiment_score_txt):.4f}"
            if isinstance(sentiment_score_txt, int | float)
            else str(sentiment_score_txt if sentiment_score_txt is not None else "n_a")
        )
        sentiment_label_repr = (
            str(sentiment_label_txt) if sentiment_label_txt is not None else "n_a"
        )
        risk_mode_repr = str(ov.get("risk_mode") or "n_a")
        vol_tier_repr = str(ov.get("volatility_tier") or "n_a")
        atr_pct_repr = (
            f"{float(ov.get('atr_pct')):.6f}"
            if isinstance(ov.get("atr_pct"), int | float)
            else "n_a"
        )
        rsi_thr_repr = (
            f"{float(ov.get('rsi_threshold_used')):.4f}"
            if isinstance(ov.get("rsi_threshold_used"), int | float)
            else "n_a"
        )
        sector_repr = str(ov.get("sector") or "Unknown")
        sma_pass_repr = str(bool(ov.get("sma_filter_passed", True))).lower()
        aggr_bypass_repr = str(bool(ov.get("aggressive_sma_bypassed", False))).lower()
        self._log.info(
            "event=strategy_signal symbol=%s action=%s reason=%s "
            "rsi=%.4f atr=%.6f close=%.4f bid=%.4f ask=%.4f "
            "spread_pct=%.6f quote_age_seconds=%.4f regime_type=%s "
            "trailing_stop_active=%s sentiment_score=%s sentiment_label=%s "
            "risk_mode=%s sector=%s volatility_tier=%s atr_pct=%s rsi_threshold=%s "
            "sma_filter_passed=%s aggressive_sma_bypassed=%s strategy=%s",
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
            regime_type,
            str(trailing_active).lower(),
            sentiment_score_repr,
            sentiment_label_repr,
            risk_mode_repr,
            sector_repr,
            vol_tier_repr,
            atr_pct_repr,
            rsi_thr_repr,
            sma_pass_repr,
            aggr_bypass_repr,
            self.name,
            extra={"symbol": signal.symbol, "strategy": self.name},
        )

    def _emit_strategy_entry_skip(
        self,
        sr: SkipReason,
        *,
        log_event: str = "strategy_entry_skip",
        discord_title: str = "ENTRY_SKIP",
    ) -> None:
        if "decision_fn" not in sr.metadata:
            sr = replace(
                sr,
                metadata={"decision_fn": "RSIMeanReversionStrategy.evaluate", **dict(sr.metadata)},
            )
        actionable_for_discord = {
            SkipCodes.SIZE_ZERO,
            SkipCodes.STALE_BARS,
            SkipCodes.SPREAD_TOO_WIDE,
            SkipCodes.ORDER_REJECTED,
            SkipCodes.RISK_LIMIT_FAIL,
            SkipCodes.ADX_FILTER_FAIL,
            SkipCodes.SMA_FILTER_FAIL,
            SkipCodes.AGGRESSIVE_SMA_BYPASS,
        }
        emit_skip_diagnostic(
            settings=self._settings,
            logger=self._log,
            log_event=log_event,
            sr=sr,
            discord_enqueue=self._discord_embed_fn if sr.code in actionable_for_discord else None,
            throttle=self._skip_diag_throttle,
            strategy_name=self.name,
            phase="strategy",
            discord_title=discord_title,
        )

    def _log_regime_skip(
        self,
        *,
        symbol: str,
        reason: str,
        regime: RegimeSnapshot,
        rsi_value: float,
        atr_value: float,
        quote: Quote | None,
        last_close: float,
        bar_ts: datetime | None,
        extra_meta: Mapping[str, object] | None = None,
    ) -> None:
        sr = regime_skip_reason(
            symbol=symbol,
            regime_reason=reason,
            rsi_value=rsi_value,
            last_close=last_close,
            regime=regime,
            quote=quote,
        )
        slope_fail = float(regime.sma_slope) <= 0.0
        adx_fail = float(regime.adx) >= float(self._thr.adx_range_max)
        code = SkipCodes.ADX_FILTER_FAIL if adx_fail else SkipCodes.SMA_FILTER_FAIL
        msg = "regime blocked by adx/sma filter"
        if adx_fail and slope_fail:
            msg = "regime blocked by adx>=threshold and non-positive sma slope"
        elif adx_fail:
            msg = "regime blocked by adx filter"
        elif slope_fail:
            msg = "regime blocked by sma slope filter"
        sr = replace(
            sr,
            code=code,
            message=msg,
            atr=atr_value,
            strategy_bar_ts=bar_ts,
            metadata={
                **dict(sr.metadata),
                "adx_fail": adx_fail,
                "sma_slope_fail": slope_fail,
                "adx_threshold": float(self._thr.adx_range_max),
                **(dict(extra_meta) if extra_meta is not None else {}),
            },
        )
        self._emit_strategy_entry_skip(sr, log_event="strategy_skip_regime")

    def _build_metadata_core(
        self,
        *,
        rsi_value: float,
        atr_value: float,
        last_close: float,
        bar_timestamp: datetime | None,
        quote: Quote | None,
    ) -> dict[str, object]:
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

    def _log_sentiment_skip(
        self,
        *,
        symbol: str,
        overlay: Mapping[str, object],
        rsi_value: float,
        common_meta: Mapping[str, object] | None = None,
    ) -> None:
        ts = datetime.now(UTC).isoformat()
        headline_count = int(overlay.get("sentiment_headline_count") or 0)
        stale = bool(overlay.get("sentiment_stale_news", False))
        sr = SkipReason(
            code=SkipCodes.RISK_LIMIT_FAIL,
            message=str(overlay.get("sentiment_reason", "blocked_long")),
            symbol=symbol,
            rsi=rsi_value,
            metadata={
                **(dict(common_meta) if common_meta is not None else {}),
                "sentiment_score": overlay.get("sentiment_score"),
                "sentiment_label": overlay.get("sentiment_label"),
                "sentiment_blocks_long_entries": overlay.get("sentiment_blocks_long_entries"),
                "headline_count": headline_count,
                "stale_news": stale,
            },
        )
        self._emit_strategy_entry_skip(sr, log_event="strategy_skip_sentiment")
        if self._database is not None:
            with contextlib.suppress(Exception):
                self._database.record_execution_event(
                    event_type="strategy_skip_sentiment",
                    symbol=symbol,
                    side=None,
                    client_order_id=None,
                    order_id=None,
                    status=None,
                    price=None,
                    quantity=None,
                    metadata={
                        "sentiment_score": overlay.get("sentiment_score"),
                        "sentiment_label": overlay.get("sentiment_label"),
                        "headline_count": headline_count,
                        "stale_news": stale,
                        "timestamp": ts,
                    },
                )

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        signals: list[Signal] = []

        bars = self._completed_bars(ctx.bars)
        if bars is None or bars.empty:
            nrow = 0 if ctx.bars is None else len(ctx.bars)
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.NO_COMPLETED_BARS,
                    message="no_completed_closed_bars_ready_for_indicators",
                    symbol=ctx.symbol.upper(),
                    metadata={"raw_bar_rows": nrow},
                ),
            )
            return signals

        regime = compute_regime_snapshot(bars=bars, settings=self._risk_overlay_settings())
        regime_eff = regime or _stub_regime(self._settings)

        rsi_series, atr_series = self._compute(bars)
        if rsi_series.empty or atr_series.empty:
            min_len = max(self._settings.RSI_LENGTH, self._settings.ATR_LENGTH) + 5
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.INDICATOR_WARMUP,
                    message="rsi_or_atr_series_empty_need_more_completed_bars",
                    symbol=ctx.symbol.upper(),
                    metadata={"completed_bar_rows": len(bars), "min_rows": min_len},
                ),
            )
            return signals

        last_rsi = float(rsi_series.iloc[-1])
        last_atr = float(atr_series.iloc[-1])
        last_close = float(bars["close"].iloc[-1])
        bar_ts: datetime | None = None
        try:
            ts_raw = bars.index[-1]
            if hasattr(ts_raw, "to_pydatetime"):
                bar_ts = ts_raw.to_pydatetime()
            elif isinstance(ts_raw, datetime):
                bar_ts = ts_raw
        except (IndexError, AttributeError):
            bar_ts = None

        if pd.isna(last_rsi) or pd.isna(last_atr) or last_atr <= 0 or last_close <= 0:
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.INVALID_INDICATORS,
                    message="nan_or_non_positive_rsi_atr_or_close",
                    symbol=ctx.symbol.upper(),
                    rsi=None if pd.isna(last_rsi) else float(last_rsi),
                    price=last_close if last_close > 0 else None,
                    atr=last_atr if last_atr > 0 else None,
                    strategy_bar_ts=bar_ts,
                    metadata={
                        "last_atr": last_atr,
                        "last_close": last_close,
                    },
                ),
            )
            return signals

        symbol = ctx.symbol.upper()
        sector = self._sector_for_symbol(symbol)
        rsi_entry_threshold, atr_pct, volatility_tier = self._resolve_rsi_entry_threshold(
            atr_value=last_atr,
            last_close=last_close,
        )
        rsi_triggered = bool(last_rsi < rsi_entry_threshold)
        aggressive_mode = bool(self._settings.AGGRESSIVE_MODE)
        aggressive_bypass_threshold = float(self._settings.AGGRESSIVE_RSI_BYPASS_THRESHOLD)
        aggressive_bypass_candidate = aggressive_mode and (last_rsi < aggressive_bypass_threshold)

        self._log.info(
            "event=strategy_volatility_gate code=%s symbol=%s sector=%s price=%.4f atr=%.6f "
            "atr_pct=%.6f volatility_tier=%s rsi=%.4f rsi_threshold=%.4f triggered=%s",
            SkipCodes.VOLATILITY_THRESHOLD_USED,
            symbol,
            sector,
            last_close,
            last_atr,
            atr_pct,
            volatility_tier,
            last_rsi,
            rsi_entry_threshold,
            str(rsi_triggered).lower(),
            extra={"symbol": symbol, "strategy": self.name},
        )

        def trailing_view(tr_state: TrailState) -> MutableMapping[str, object]:
            return {
                "trailing_stop_active": tr_state.trailing_stop_active,
                "trailing_stop_price": tr_state.trailing_stop_price,
                "target_a_hit": tr_state.target_a_hit,
                "highest_close_since_activation": (
                    tr_state.highest_close_since_activation if tr_state.trailing_stop_active else 0.0
                ),
            }

        base_meta = lambda: self._build_metadata_core(  # noqa: E731
            rsi_value=last_rsi,
            atr_value=last_atr,
            last_close=last_close,
            bar_timestamp=bar_ts,
            quote=ctx.quote,
        )
        common_meta = {
            "sector": sector,
            "atr_pct": atr_pct,
            "volatility_tier": volatility_tier,
            "rsi_threshold_used": rsi_entry_threshold,
            "rsi_triggered": rsi_triggered,
            "aggressive_mode": aggressive_mode,
            "aggressive_rsi_bypass_threshold": aggressive_bypass_threshold,
        }

        # ----------------- exits first ------------------------------------
        if ctx.has_position:
            position = ctx.position
            if position is None or position.side.lower() != "long":
                self._clear_trail(symbol)
                self._persist_trailing_to_disk()
                return signals

            entry_price = float(position.avg_entry_price)
            stop_dist = self._thr.atr_stop_multiplier * last_atr
            tp_dist = self._settings.ATR_PROFIT_MULTIPLIER * last_atr

            stop_breached = last_close <= (entry_price - stop_dist)

            trail = self._get_or_bind_trail(symbol, entry_price)

            if stop_breached:
                md = self._regime_overlay(
                    base_meta(),
                    regime_eff,
                    trailing=trailing_view(trail),
                    conviction_multiplier=None,
                    ctx=ctx,
                )
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.EMERGENCY_EXIT_LONG,
                    reason=(
                        f"atr_stop_breach close={last_close:.4f} "
                        f"entry={entry_price:.4f} stop_dist={stop_dist:.4f}"
                    ),
                    reference_price=last_close,
                    atr=last_atr,
                    metadata=md,
                )
                signals.append(signal)
                self._log_signal(
                    signal,
                    quote=ctx.quote,
                    last_close=last_close,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                    regime_type=regime_eff.regime_type,
                    trailing_active=trail.trailing_stop_active,
                )
                self._clear_trail(symbol)
                self._persist_trailing_to_disk()
                self._entry_bar_index.pop(symbol, None)
                return signals

            # Update synthetic trailing BEFORE comparing breach for this closed bar.
            unreal_pct = safe_unreal_pct(last_close, entry_price)
            if not trail.trailing_stop_active:
                if unreal_pct >= self._settings.TRAIL_TRIGGER_PCT:
                    trail.target_a_hit = True
                    trail.trailing_stop_active = True
                    trail.locked_floor = entry_price * (1.0 + self._settings.TRAIL_LOCKED_PROFIT_PCT)
                    trail.highest_close_since_activation = max(last_close, entry_price)
                    atr_line = (
                        trail.highest_close_since_activation
                        - self._thr.trail_atr_multiplier * last_atr
                    )
                    trail.trailing_stop_price = max(trail.locked_floor, atr_line)
            else:
                trail.target_a_hit = True
                trail.highest_close_since_activation = max(
                    trail.highest_close_since_activation, last_close
                )
                atr_line = trail.highest_close_since_activation - self._thr.trail_atr_multiplier * last_atr
                trail.trailing_stop_price = max(trail.locked_floor, atr_line)

            self._trails_by_symbol[symbol] = trail

            if trail.trailing_stop_active and last_close <= trail.trailing_stop_price + 1e-12:
                md = self._regime_overlay(
                    base_meta(),
                    regime_eff,
                    trailing=trailing_view(trail),
                    conviction_multiplier=None,
                    ctx=ctx,
                )
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.EXIT_LONG,
                    reason=(
                        f"trailing_profit_breach stop={trail.trailing_stop_price:.4f} "
                        f"close={last_close:.4f}"
                    ),
                    reference_price=last_close,
                    atr=last_atr,
                    metadata=md,
                )
                signals.append(signal)
                self._log_signal(
                    signal,
                    quote=ctx.quote,
                    last_close=last_close,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                    regime_type=regime_eff.regime_type,
                    trailing_active=True,
                )
                self._clear_trail(symbol)
                self._persist_trailing_to_disk()
                self._entry_bar_index.pop(symbol, None)
                return signals

            tp_hit = last_close >= (entry_price + tp_dist)
            rsi_exit = last_rsi >= self._thr.rsi_exit
            entry_idx = self._entry_bar_index.get(symbol, len(bars) - 1)
            held_bars = max(0, len(bars) - 1 - entry_idx)
            time_exit = held_bars >= self._settings.MAX_HOLD_BARS

            if tp_hit or rsi_exit or time_exit:
                reason_tag = (
                    "tp_hit" if tp_hit else ("rsi_exit" if rsi_exit else "time_exit")
                )
                md = self._regime_overlay(
                    base_meta(),
                    regime_eff,
                    trailing=trailing_view(trail),
                    conviction_multiplier=None,
                    ctx=ctx,
                )
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.EXIT_LONG,
                    reason=f"{reason_tag} close={last_close:.4f} rsi={last_rsi:.2f}",
                    reference_price=last_close,
                    atr=last_atr,
                    metadata=md,
                )
                signals.append(signal)
                self._log_signal(
                    signal,
                    quote=ctx.quote,
                    last_close=last_close,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                    regime_type=regime_eff.regime_type,
                    trailing_active=trail.trailing_stop_active,
                )
                self._clear_trail(symbol)
                self._persist_trailing_to_disk()
                self._entry_bar_index.pop(symbol, None)
                return signals

            self._persist_trailing_to_disk()
            return signals

        # No position → ensure trail map does not stale rows for symbols not held here.
        if symbol in self._trails_by_symbol:
            self._trails_by_symbol.pop(symbol, None)
            self._persist_trailing_to_disk()

        # ----------------- entries ----------------------------------------
        sp_live: float | None = None
        qa_live: float | None = None
        if ctx.quote is not None:
            try:
                qa_live = float(ctx.quote.age_seconds())
            except (AttributeError, TypeError, ValueError):
                qa_live = None
            try:
                if ctx.quote.bid > 0 and ctx.quote.ask > ctx.quote.bid:
                    sp_live = float(compute_spread_pct(ctx.quote.bid, ctx.quote.ask))
            except ValueError:
                sp_live = None

        if ctx.has_open_order:
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.OPEN_ORDER_BLOCKS_ENTRY,
                    message="working_order_present_strategy_skips_additional_long_entry",
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    open_order_exists=True,
                    metadata=dict(common_meta),
                ),
            )
            return signals
        if ctx.quote is None:
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.QUOTE_INVALID,
                    message="no_quote_in_strategy_context_cannot_validate_spread_or_price",
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    strategy_bar_ts=bar_ts,
                    metadata=dict(common_meta),
                ),
            )
            return signals
        if last_rsi >= rsi_entry_threshold:
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.RSI_NOT_TRIGGERED,
                    message=(
                        f"rsi_not_strictly_below_oversold "
                        f"rsi={last_rsi:.4f} threshold={rsi_entry_threshold:.4f}"
                    ),
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    metadata={
                        **dict(common_meta),
                        "rsi_oversold_threshold": float(rsi_entry_threshold),
                    },
                ),
            )
            return signals

        aggressive_bypass_used = False
        if not regime_eff.allow_rsi_long:
            slope_fail = float(regime_eff.sma_slope) <= 0.0
            adx_fail = float(regime_eff.adx) >= float(self._thr.adx_range_max)
            if aggressive_bypass_candidate and slope_fail and not adx_fail:
                aggressive_bypass_used = True
                self._emit_strategy_entry_skip(
                    SkipReason(
                        code=SkipCodes.AGGRESSIVE_SMA_BYPASS,
                        message=(
                            "aggressive_mode_bypassed_sma_filter "
                            f"rsi={last_rsi:.4f} threshold={aggressive_bypass_threshold:.4f}"
                        ),
                        symbol=symbol,
                        decision="PASS",
                        rsi=last_rsi,
                        atr=last_atr,
                        adx=float(regime_eff.adx),
                        price=last_close,
                        sma_200=float(regime_eff.sma200),
                        sma_200_slope=float(regime_eff.sma_slope),
                        spread_pct=sp_live,
                        quote_age_seconds=qa_live,
                        strategy_bar_ts=bar_ts,
                        metadata={
                            **dict(common_meta),
                            "sma_filter_passed": False,
                            "aggressive_sma_bypassed": True,
                            "adx_fail": adx_fail,
                            "sma_slope_fail": slope_fail,
                        },
                    ),
                    log_event="strategy_aggressive_sma_bypass",
                    discord_title="AGGRESSIVE_SMA_BYPASS",
                )
            else:
                self._log_regime_skip(
                    symbol=symbol,
                    reason=regime_eff.reason,
                    regime=regime_eff,
                    rsi_value=last_rsi,
                    atr_value=last_atr,
                    quote=ctx.quote,
                    last_close=last_close,
                    bar_ts=bar_ts,
                    extra_meta={
                        **dict(common_meta),
                        "sma_filter_passed": False,
                        "aggressive_sma_bypassed": False,
                        "aggressive_bypass_candidate": aggressive_bypass_candidate,
                    },
                )
                return signals
        overlay_live = ctx.sentiment_overlay or sentiment_overlay_neutral(symbol)
        if overlay_live.get("sentiment_blocks_long_entries") or overlay_live.get(
            "sentiment_label"
        ) == "strong_negative":
            self._log_sentiment_skip(
                symbol=symbol,
                overlay=overlay_live,
                rsi_value=last_rsi,
                common_meta=common_meta,
            )
            return signals

        conviction_mult = (
            float(self._settings.HIGH_CONVICTION_RISK_MULTIPLIER)
            if regime_eff.price_above_sma200
            else float(self._settings.LOW_CONVICTION_RISK_MULTIPLIER)
        )

        md = self._regime_overlay(
            {
                **base_meta(),
                **common_meta,
                "sma_filter_passed": True,
                "aggressive_sma_bypassed": aggressive_bypass_used,
            },
            regime_eff,
            trailing={
                "trailing_stop_active": False,
                "trailing_stop_price": 0.0,
                "target_a_hit": False,
                "highest_close_since_activation": 0.0,
            },
            conviction_multiplier=conviction_mult,
            ctx=ctx,
        )

        if self._ml_filter is not None and self._settings.ENABLE_ML_FILTER:
            ctx_ml = dict(md)
            ctx_ml["symbol"] = symbol
            dec = self._ml_filter.should_allow_trade(signal_context=ctx_ml)
            if not dec.allowed:
                prob_txt = "n/a" if dec.probability is None else f"{dec.probability:.6f}"
                sr_ml = SkipReason(
                    code=SkipCodes.RISK_LIMIT_FAIL,
                    message=str(dec.reason or "ml_filter_disallowed"),
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    metadata={
                        **dict(common_meta),
                        "ml_model_trained": dec.model_trained,
                        "ml_probability": prob_txt,
                        "ml_threshold": float(self._settings.ML_FILTER_THRESHOLD),
                    },
                )
                self._emit_strategy_entry_skip(
                    sr_ml,
                    log_event="strategy_skip_ml",
                    discord_title="ML_ENTRY_SKIP",
                )
                return signals

        signal = Signal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            reason=f"rsi_oversold rsi={last_rsi:.2f} close={last_close:.4f}",
            reference_price=last_close,
            atr=last_atr,
            metadata=md,
        )
        signals.append(signal)
        self._log_signal(
            signal,
            quote=ctx.quote,
            last_close=last_close,
            rsi_value=last_rsi,
            atr_value=last_atr,
            regime_type=regime_eff.regime_type,
            trailing_active=False,
        )
        self._entry_bar_index[symbol] = len(bars) - 1
        return signals

    def adopt_long_position(self, symbol: str, avg_entry_price: float) -> None:
        """Snap synthetic trailing anchors to a reconciled broker long.

        The strategy's next completed-bar pass applies ATR catastrophe + profit trails.
        """
        sym = symbol.upper()
        self._trails_by_symbol[sym] = TrailState(
            avg_entry_price=float(avg_entry_price),
            trailing_stop_active=False,
            locked_floor=0.0,
            highest_close_since_activation=0.0,
            trailing_stop_price=0.0,
            target_a_hit=False,
        )
        self._persist_trailing_to_disk()


def safe_unreal_pct(last_close: float, entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return (last_close / entry_price) - 1.0


__all__ = ["RSIMeanReversionStrategy", "TrailState", "safe_unreal_pct"]
