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
from zoneinfo import ZoneInfo

import pandas as pd

from config.constants import LOGGER_STRATEGY
from config.settings import Settings
from config.strategy_runtime import StrategyRuntimeThresholds
from core.account import PositionSnapshot
from core.database import Database
from core.market_data import Quote
from core.state_store import StateStore, TrailTrailingRecord
from utils.price_utils import spread_pct as compute_spread_pct

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import RegimeSnapshot, compute_regime_snapshot
from .indicators import atr, bollinger_bands, rolling_vwap_zscore_bands, rsi
from .sentiment import sentiment_overlay_neutral
from .skip_diagnostics import (
    SkipCodes,
    SkipDiagnosticsThrottle,
    SkipReason,
    emit_skip_diagnostic,
    regime_skip_reason,
)
from .universe import compute_elastic_spread_cap

# Mapping from BAR_TIMEFRAME string to a timedelta used to drop an in-progress
# trailing bar. Keep in sync with `core.market_data._parse_timeframe`.
_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1Min": timedelta(minutes=1),
    "5Min": timedelta(minutes=5),
    "15Min": timedelta(minutes=15),
    "1Hour": timedelta(hours=1),
    "1Day": timedelta(days=1),
}

_ET = ZoneInfo("America/New_York")


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
        min_len = (
            max(
                self._settings.RSI_LENGTH,
                self._settings.ATR_LENGTH,
                self._settings.BOLLINGER_LENGTH if self._settings.BOLLINGER_ENABLED else 0,
                self._settings.VWAP_LENGTH if self._settings.VWAP_STRATEGY_ENABLED else 0,
                self._settings.DYNAMIC_RSI_LONG_ATR if self._settings.DYNAMIC_RSI_ENABLED else 0,
                self._settings.ATR_LOOKBACK if self._settings.DYNAMIC_RSI_ENABLED else 0,
            )
            + 5
        )
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
        atr_mean: float | None,
        last_close: float,
        atr_ratio_override: float | None = None,
    ) -> tuple[float, float, str, float | None]:
        atr_pct = (atr_value / last_close) if last_close > 0 else 0.0
        if bool(self._settings.DYNAMIC_RSI_ENABLED):
            base = float(self._settings.DYNAMIC_RSI_BASE)
            mn = float(self._settings.DYNAMIC_RSI_MIN)
            mx = float(self._settings.DYNAMIC_RSI_MAX)
            if atr_ratio_override is not None and atr_ratio_override > 0:
                atr_ratio = float(atr_ratio_override)
            else:
                ref = float(atr_mean) if atr_mean is not None and atr_mean > 0 else float(atr_value)
                atr_ratio = float(atr_value) / ref if ref > 0 else 1.0
            raw = base * atr_ratio
            threshold = min(mx, max(mn, raw))
            if atr_ratio > 1.05:
                tier = "DYNAMIC_HIGH_ATR"
            elif atr_ratio < 0.95:
                tier = "DYNAMIC_LOW_ATR"
            else:
                tier = "DYNAMIC_NORMAL_ATR"
            return threshold, atr_pct, tier, atr_ratio
        high_vol = atr_pct > float(self._settings.HIGH_VOL_ATR_PCT_THRESHOLD)
        threshold = (
            float(self._settings.HIGH_VOL_RSI_ENTRY)
            if high_vol
            else float(self._settings.DEFAULT_RSI_ENTRY)
        )
        tier = "HIGH_VOL" if high_vol else "NORMAL_VOL"
        return threshold, atr_pct, tier, None

    def _bollinger_snapshot(self, bars: pd.DataFrame) -> dict[str, float | bool] | None:
        basis, upper, lower, width = bollinger_bands(
            bars["close"],
            length=int(self._settings.BOLLINGER_LENGTH),
            num_std=float(self._settings.BOLLINGER_STD),
        )
        vals = {
            "basis": float(basis.iloc[-1]),
            "upper": float(upper.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "width_pct": float(width.iloc[-1]),
            "price": float(bars["close"].iloc[-1]),
        }
        if any(pd.isna(v) for v in vals.values()):
            return None
        min_width = float(self._settings.BOLLINGER_MIN_WIDTH_PCT)
        below_lower = bool(vals["price"] <= vals["lower"])
        above_upper = bool(vals["price"] >= vals["upper"])
        touch = below_lower or above_upper
        require_touch = bool(self._settings.BOLLINGER_REQUIRE_TOUCH)
        passed = bool(vals["width_pct"] >= min_width and (touch or not require_touch))
        vals["price_below_lower"] = below_lower
        vals["price_above_upper"] = above_upper
        vals["touch"] = touch
        vals["require_touch"] = require_touch
        vals["passed"] = passed
        vals["min_width_pct"] = min_width
        return vals

    def _vwap_snapshot(self, bars: pd.DataFrame) -> dict[str, float | bool] | None:
        if "volume" not in bars.columns:
            return None
        vwap, upper, lower, zscore, distance, deviation = rolling_vwap_zscore_bands(
            bars["high"],
            bars["low"],
            bars["close"],
            bars["volume"],
            length=int(self._settings.VWAP_LENGTH),
            z_threshold=float(self._settings.VWAP_Z_THRESHOLD),
        )
        vals = {
            "vwap": float(vwap.iloc[-1]),
            "upper": float(upper.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "zscore": float(zscore.iloc[-1]),
            "distance_pct": float(distance.iloc[-1]),
            "deviation": float(deviation.iloc[-1]),
            "price": float(bars["close"].iloc[-1]),
            "z_threshold": float(self._settings.VWAP_Z_THRESHOLD),
        }
        if any(pd.isna(v) for v in vals.values()):
            return None
        vals["passed"] = bool(vals["zscore"] <= -float(self._settings.VWAP_Z_THRESHOLD))
        return vals

    def _time_window_snapshot(self, now_utc_value: datetime) -> dict[str, object]:
        if not bool(self._settings.TIME_OF_DAY_FILTER_ENABLED):
            return {
                "enabled": False,
                "passed": True,
                "now_et": "",
                "start": self._settings.TIME_OF_DAY_TRADE_START,
                "end": self._settings.TIME_OF_DAY_TRADE_END,
            }
        now = now_utc_value
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        et_now = now.astimezone(_ET)

        def _mins(raw: str) -> int:
            hh, mm = raw.split(":", 1)
            return int(hh) * 60 + int(mm)

        cur_m = et_now.hour * 60 + et_now.minute
        start_m = _mins(self._settings.TIME_OF_DAY_TRADE_START)
        end_m = _mins(self._settings.TIME_OF_DAY_TRADE_END)
        if start_m <= end_m:
            passed = start_m <= cur_m <= end_m
        else:
            passed = cur_m >= start_m or cur_m <= end_m
        return {
            "enabled": True,
            "passed": bool(passed),
            "now_et": et_now.strftime("%H:%M"),
            "start": self._settings.TIME_OF_DAY_TRADE_START,
            "end": self._settings.TIME_OF_DAY_TRADE_END,
        }

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
        vwap_z_repr = (
            f"{float(ov.get('vwap_zscore')):.4f}"
            if isinstance(ov.get("vwap_zscore"), int | float)
            else "n_a"
        )
        vwap_repr = (
            f"{float(ov.get('vwap')):.4f}"
            if isinstance(ov.get("vwap"), int | float)
            else "n_a"
        )
        bb_width_repr = (
            f"{float(ov.get('bollinger_width_pct')):.6f}"
            if isinstance(ov.get("bollinger_width_pct"), int | float)
            else "n_a"
        )
        self._log.info(
            "event=strategy_signal symbol=%s action=%s reason=%s "
            "rsi=%.4f atr=%.6f close=%.4f bid=%.4f ask=%.4f "
            "spread_pct=%.6f quote_age_seconds=%.4f regime_type=%s "
            "trailing_stop_active=%s sentiment_score=%s sentiment_label=%s "
            "risk_mode=%s sector=%s volatility_tier=%s atr_pct=%s rsi_threshold=%s "
            "sma_filter_passed=%s aggressive_sma_bypassed=%s vwap=%s vwap_zscore=%s "
            "bollinger_width_pct=%s strategy=%s",
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
            vwap_repr,
            vwap_z_repr,
            bb_width_repr,
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
            SkipCodes.ORDER_REJECTED,
            SkipCodes.RISK_LIMIT_FAIL,
            SkipCodes.KILL_SWITCH_LATCHED,
            SkipCodes.COMPLIANCE_REJECTED,
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

    def _log_scale_in_candidate(
        self,
        *,
        symbol: str,
        current_qty: float,
        avg_entry_price: float,
        current_price: float,
        underwater_pct: float,
        rsi_value: float,
        bullet_count: int | None,
        max_bullets: int,
        spread_pct_value: float | None,
        spread_threshold_pct: float | None,
        quote_age_seconds: float | None,
        strategy_bar_ts: datetime | None,
    ) -> None:
        self._log.info(
            "event=scale_in_candidate symbol=%s current_qty=%.6f avg_entry_price=%.6f "
            "current_price=%.6f underwater_pct=%.6f rsi=%.6f bullet_count=%s max_bullets=%s "
            "spread_pct=%s spread_threshold_pct=%s quote_age_seconds=%s strategy_bar_ts=%s",
            symbol,
            current_qty,
            avg_entry_price,
            current_price,
            underwater_pct,
            rsi_value,
            str(bullet_count) if bullet_count is not None else "unknown",
            max_bullets,
            f"{spread_pct_value:.6f}" if spread_pct_value is not None else "n_a",
            f"{spread_threshold_pct:.6f}" if spread_threshold_pct is not None else "n_a",
            f"{quote_age_seconds:.3f}" if quote_age_seconds is not None else "n_a",
            strategy_bar_ts.isoformat() if strategy_bar_ts is not None else "n_a",
            extra={"symbol": symbol, "strategy": self.name},
        )

    def _log_scale_in_skip(
        self,
        *,
        symbol: str,
        skip_code: str,
        reason: str,
        current_qty: float,
        avg_entry_price: float,
        current_price: float,
        underwater_pct: float,
        rsi_value: float,
        rsi_threshold: float,
        bullet_count: int | None,
        max_bullets: int,
        spread_pct_value: float | None,
        spread_threshold_pct: float | None,
        quote_age_seconds: float | None,
        strategy_bar_ts: datetime | None,
    ) -> None:
        self._log.info(
            "event=scale_in_skip symbol=%s skip_code=%s reason=%s current_qty=%.6f "
            "avg_entry_price=%.6f current_price=%.6f underwater_pct=%.6f rsi=%.6f "
            "scale_in_rsi_threshold=%.6f bullet_count=%s max_bullets=%s "
            "spread_pct=%s spread_threshold_pct=%s quote_age_seconds=%s strategy_bar_ts=%s",
            symbol,
            skip_code,
            reason,
            current_qty,
            avg_entry_price,
            current_price,
            underwater_pct,
            rsi_value,
            rsi_threshold,
            str(bullet_count) if bullet_count is not None else "unknown",
            max_bullets,
            f"{spread_pct_value:.6f}" if spread_pct_value is not None else "n_a",
            f"{spread_threshold_pct:.6f}" if spread_threshold_pct is not None else "n_a",
            f"{quote_age_seconds:.3f}" if quote_age_seconds is not None else "n_a",
            strategy_bar_ts.isoformat() if strategy_bar_ts is not None else "n_a",
            extra={"symbol": symbol, "strategy": self.name, "skip_code": skip_code},
        )

    def _log_scale_in_signal(
        self,
        *,
        symbol: str,
        current_qty: float,
        add_qty: float,
        avg_entry_price: float,
        current_price: float,
        underwater_pct: float,
        rsi_value: float,
        bullet_number: int,
        max_bullets: int,
    ) -> None:
        self._log.info(
            "event=scale_in_signal symbol=%s action=BUY current_qty=%.6f add_qty=%.6f "
            "avg_entry_price=%.6f current_price=%.6f underwater_pct=%.6f rsi=%.6f "
            "bullet_number=%s max_bullets=%s reason=secondary_rsi_oversold_while_underwater",
            symbol,
            current_qty,
            add_qty,
            avg_entry_price,
            current_price,
            underwater_pct,
            rsi_value,
            bullet_number,
            max_bullets,
            extra={"symbol": symbol, "strategy": self.name},
        )

    def get_bullet_count(
        self,
        *,
        symbol: str,
        current_position: PositionSnapshot,
        trade_ledger: Database | None,
    ) -> int | None:
        """Infer bullet count conservatively; return ``None`` when uncertain."""

        _ = trade_ledger  # placeholder: future-proof hook for persisted bullet ledgers
        add_qty = float(self._settings.SCALE_IN_ADD_QTY)
        qty = abs(float(current_position.qty))
        if add_qty <= 0 or qty <= 0:
            return None
        ratio = qty / add_qty
        rounded = int(round(ratio))
        if rounded < 1:
            return None
        if abs(ratio - float(rounded)) > 1e-6:
            return None
        # Fail closed if quantity math is ambiguous in non-fractional mode.
        if (not bool(self._settings.ENABLE_FRACTIONAL)) and abs(qty - round(qty)) > 1e-6:
            return None
        return rounded

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
            min_len = (
                max(
                    self._settings.RSI_LENGTH,
                    self._settings.ATR_LENGTH,
                    self._settings.BOLLINGER_LENGTH if self._settings.BOLLINGER_ENABLED else 0,
                    self._settings.VWAP_LENGTH if self._settings.VWAP_STRATEGY_ENABLED else 0,
                    self._settings.DYNAMIC_RSI_LONG_ATR if self._settings.DYNAMIC_RSI_ENABLED else 0,
                    self._settings.ATR_LOOKBACK if self._settings.DYNAMIC_RSI_ENABLED else 0,
                )
                + 5
            )
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
        atr_mean: float | None = None
        dynamic_atr_ratio: float | None = None
        dynamic_atr_short: float | None = None
        dynamic_atr_long: float | None = None
        if bool(self._settings.DYNAMIC_RSI_ENABLED):
            atr_window = atr_series.tail(int(self._settings.ATR_LOOKBACK)).dropna()
            if not atr_window.empty:
                atr_mean = float(atr_window.mean())
            try:
                atr_short_series = atr(
                    bars["high"],
                    bars["low"],
                    bars["close"],
                    length=int(self._settings.DYNAMIC_RSI_SHORT_ATR),
                )
                atr_long_series = atr(
                    bars["high"],
                    bars["low"],
                    bars["close"],
                    length=int(self._settings.DYNAMIC_RSI_LONG_ATR),
                )
                dynamic_atr_short = float(atr_short_series.iloc[-1])
                dynamic_atr_long = float(atr_long_series.iloc[-1])
                if (
                    not pd.isna(dynamic_atr_short)
                    and not pd.isna(dynamic_atr_long)
                    and dynamic_atr_long > 0
                ):
                    dynamic_atr_ratio = dynamic_atr_short / dynamic_atr_long
            except (IndexError, KeyError, TypeError, ValueError):
                dynamic_atr_ratio = None
        rsi_entry_threshold, atr_pct, volatility_tier, atr_ratio = self._resolve_rsi_entry_threshold(
            atr_value=last_atr,
            atr_mean=atr_mean,
            last_close=last_close,
            atr_ratio_override=dynamic_atr_ratio,
        )
        rsi_triggered = bool(last_rsi < rsi_entry_threshold)
        aggressive_mode = bool(self._settings.AGGRESSIVE_MODE)
        aggressive_bypass_threshold = float(self._settings.AGGRESSIVE_RSI_BYPASS_THRESHOLD)
        aggressive_bypass_candidate = aggressive_mode and (last_rsi < aggressive_bypass_threshold)

        self._log.info(
            "event=strategy_volatility_gate code=%s symbol=%s sector=%s price=%.4f atr=%.6f "
            "atr_pct=%.6f atr_mean=%s atr_ratio=%s dynamic_rsi_enabled=%s "
            "volatility_tier=%s rsi=%.4f rsi_threshold=%.4f triggered=%s",
            SkipCodes.VOLATILITY_THRESHOLD_USED,
            symbol,
            sector,
            last_close,
            last_atr,
            atr_pct,
            f"{atr_mean:.6f}" if atr_mean is not None else "n_a",
            f"{atr_ratio:.6f}" if atr_ratio is not None else "n_a",
            str(self._settings.DYNAMIC_RSI_ENABLED).lower(),
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
            "atr_mean": atr_mean,
            "atr_ratio": atr_ratio,
            "dynamic_atr_short": dynamic_atr_short,
            "dynamic_atr_long": dynamic_atr_long,
            "dynamic_atr_short_length": int(self._settings.DYNAMIC_RSI_SHORT_ATR),
            "dynamic_atr_long_length": int(self._settings.DYNAMIC_RSI_LONG_ATR),
            "volatility_tier": volatility_tier,
            "rsi_threshold_used": rsi_entry_threshold,
            "rsi_triggered": rsi_triggered,
            "aggressive_mode": aggressive_mode,
            "aggressive_rsi_bypass_threshold": aggressive_bypass_threshold,
            "adx_low": float(self._settings.ADX_LOW),
            "adx_high": float(self._settings.ADX_HIGH),
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

            current_qty = abs(float(position.qty))
            underwater_pct = safe_unreal_pct(last_close, entry_price)
            scale_in_rsi_threshold = float(self._settings.SCALE_IN_RSI_THRESHOLD)
            underwater_threshold = float(self._settings.SCALE_IN_UNDERWATER_PCT)
            max_bullets = int(self._settings.MAX_BULLETS_PER_SYMBOL)
            proposed_add_qty = float(self._settings.SCALE_IN_ADD_QTY)
            bullet_count = self.get_bullet_count(
                symbol=symbol,
                current_position=position,
                trade_ledger=self._database,
            )

            quote_age_seconds: float | None = None
            spread_pct_value: float | None = None
            spread_threshold_pct: float | None = None
            if ctx.quote is not None:
                with contextlib.suppress(AttributeError, TypeError, ValueError):
                    quote_age_seconds = float(ctx.quote.age_seconds())
                with contextlib.suppress(ValueError):
                    if ctx.quote.bid > 0 and ctx.quote.ask > ctx.quote.bid:
                        spread_pct_value = float(compute_spread_pct(ctx.quote.bid, ctx.quote.ask))
                        spread_threshold_pct, _ = compute_elastic_spread_cap(
                            self._settings,
                            quote=ctx.quote,
                            ref_price=last_close,
                            quote_age_seconds=quote_age_seconds or 0.0,
                        )

            self._log_scale_in_candidate(
                symbol=symbol,
                current_qty=current_qty,
                avg_entry_price=entry_price,
                current_price=last_close,
                underwater_pct=underwater_pct,
                rsi_value=last_rsi,
                bullet_count=bullet_count,
                max_bullets=max_bullets,
                spread_pct_value=spread_pct_value,
                spread_threshold_pct=spread_threshold_pct,
                quote_age_seconds=quote_age_seconds,
                strategy_bar_ts=bar_ts,
            )

            if not bool(self._settings.SCALE_IN_ENABLED):
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_disabled",
                    reason="SCALE_IN_ENABLED=false",
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if ctx.has_open_order:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_open_order_exists",
                    reason="working_order_present",
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if ctx.quote is None:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_stale_quote",
                    reason="missing_quote",
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            max_quote_age = float(self._settings.QUOTE_STALENESS_SECONDS)
            if (
                quote_age_seconds is None
                or quote_age_seconds > max_quote_age
                or ctx.quote.bid <= 0
                or ctx.quote.ask <= ctx.quote.bid
            ):
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_stale_quote",
                    reason=(
                        "quote_missing_or_stale_for_scale_in "
                        f"age={quote_age_seconds} max_age={max_quote_age}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if spread_pct_value is None:
                with contextlib.suppress(ValueError):
                    spread_pct_value = float(compute_spread_pct(ctx.quote.bid, ctx.quote.ask))
            if spread_threshold_pct is None:
                spread_threshold_pct, _ = compute_elastic_spread_cap(
                    self._settings,
                    quote=ctx.quote,
                    ref_price=last_close,
                    quote_age_seconds=quote_age_seconds or 0.0,
                )
            if (
                spread_pct_value is None
                or spread_threshold_pct is None
                or spread_pct_value > spread_threshold_pct
            ):
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_spread_too_wide",
                    reason=(
                        "spread_above_threshold "
                        f"spread_pct={spread_pct_value} threshold={spread_threshold_pct}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if getattr(ctx, "qqq_regime_bear_volatile", False) and bool(
                self._settings.REGIME_BEAR_VOLATILE_BLOCK_ENTRIES,
            ):
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_macro_regime",
                    reason="QQQ_BearVolatile_block_entries",
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if self._settings.LIQUIDITY_GATE_ENABLED and "volume" in bars.columns and len(bars) >= 21:
                vol5m = float(bars["volume"].iloc[-1])
                avg_vol20 = float(bars["volume"].iloc[-20:].mean())
                thr = float(self._settings.LIQUIDITY_THRESHOLD)
                if avg_vol20 > 0 and vol5m < thr * avg_vol20:
                    self._log_scale_in_skip(
                        symbol=symbol,
                        skip_code="scale_in_low_liquidity",
                        reason=(
                            f"volume_gate vol5m={vol5m:.0f} avg20={avg_vol20:.0f} thr={thr}"
                        ),
                        current_qty=current_qty,
                        avg_entry_price=entry_price,
                        current_price=last_close,
                        underwater_pct=underwater_pct,
                        rsi_value=last_rsi,
                        rsi_threshold=scale_in_rsi_threshold,
                        bullet_count=bullet_count,
                        max_bullets=max_bullets,
                        spread_pct_value=spread_pct_value,
                        spread_threshold_pct=spread_threshold_pct,
                        quote_age_seconds=quote_age_seconds,
                        strategy_bar_ts=bar_ts,
                    )
                    self._persist_trailing_to_disk()
                    return signals

            if underwater_pct > underwater_threshold:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_not_underwater",
                    reason=(
                        "position_not_underwater_enough "
                        f"underwater_pct={underwater_pct:.6f} "
                        f"threshold={underwater_threshold:.6f}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if last_rsi > scale_in_rsi_threshold:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_rsi_not_low_enough",
                    reason=(
                        "secondary_rsi_not_triggered "
                        f"rsi={last_rsi:.6f} threshold={scale_in_rsi_threshold:.6f}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if bullet_count is None:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_bullet_count_unknown",
                    reason=(
                        "cannot_infer_bullets_safely_from_position_qty "
                        f"qty={current_qty:.6f} add_qty={proposed_add_qty:.6f}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            if bullet_count >= max_bullets:
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_max_bullets_reached",
                    reason=(
                        "max_bullets_reached "
                        f"bullet_count={bullet_count} max_bullets={max_bullets}"
                    ),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
                return signals

            overlay_live = ctx.sentiment_overlay or sentiment_overlay_neutral(symbol)
            if overlay_live.get("sentiment_blocks_long_entries") or overlay_live.get(
                "sentiment_label"
            ) == "strong_negative":
                self._log_scale_in_skip(
                    symbol=symbol,
                    skip_code="scale_in_sentiment_block",
                    reason=str(overlay_live.get("sentiment_reason", "sentiment_blocked")),
                    current_qty=current_qty,
                    avg_entry_price=entry_price,
                    current_price=last_close,
                    underwater_pct=underwater_pct,
                    rsi_value=last_rsi,
                    rsi_threshold=scale_in_rsi_threshold,
                    bullet_count=bullet_count,
                    max_bullets=max_bullets,
                    spread_pct_value=spread_pct_value,
                    spread_threshold_pct=spread_threshold_pct,
                    quote_age_seconds=quote_age_seconds,
                    strategy_bar_ts=bar_ts,
                )
                self._persist_trailing_to_disk()
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
                    "signal_type": "scale_in",
                    "scale_in": True,
                    "scale_in_rsi_threshold": scale_in_rsi_threshold,
                    "scale_in_underwater_threshold": underwater_threshold,
                    "underwater_pct": underwater_pct,
                    "position_qty": current_qty,
                    "proposed_add_qty": proposed_add_qty,
                    "bullet_count": bullet_count,
                    "bullet_number": bullet_count + 1,
                    "max_bullets": max_bullets,
                    "avg_entry_price": entry_price,
                    "scale_in_stop_distance": stop_dist,
                    "scale_in_stop_price": entry_price - stop_dist,
                    "spread_threshold_pct": spread_threshold_pct,
                },
                regime_eff,
                trailing=trailing_view(trail),
                conviction_multiplier=conviction_mult,
                ctx=ctx,
            )

            if self._ml_filter is not None and self._settings.ENABLE_ML_FILTER:
                ctx_ml = dict(md)
                ctx_ml["symbol"] = symbol
                dec = self._ml_filter.should_allow_trade(signal_context=ctx_ml)
                if not dec.allowed:
                    self._log_scale_in_skip(
                        symbol=symbol,
                        skip_code="scale_in_ml_filter_block",
                        reason=str(dec.reason or "ml_filter_disallowed"),
                        current_qty=current_qty,
                        avg_entry_price=entry_price,
                        current_price=last_close,
                        underwater_pct=underwater_pct,
                        rsi_value=last_rsi,
                        rsi_threshold=scale_in_rsi_threshold,
                        bullet_count=bullet_count,
                        max_bullets=max_bullets,
                        spread_pct_value=spread_pct_value,
                        spread_threshold_pct=spread_threshold_pct,
                        quote_age_seconds=quote_age_seconds,
                        strategy_bar_ts=bar_ts,
                    )
                    self._persist_trailing_to_disk()
                    return signals

            signal = Signal(
                symbol=symbol,
                action=SignalAction.ENTER_LONG,
                reason=(
                    f"scale_in_long rsi={last_rsi:.2f} underwater={underwater_pct:.4f} "
                    f"bullet={bullet_count + 1}/{max_bullets}"
                ),
                reference_price=last_close,
                atr=last_atr,
                metadata=md,
            )
            signals.append(signal)
            self._log_scale_in_signal(
                symbol=symbol,
                current_qty=current_qty,
                add_qty=proposed_add_qty,
                avg_entry_price=entry_price,
                current_price=last_close,
                underwater_pct=underwater_pct,
                rsi_value=last_rsi,
                bullet_number=bullet_count + 1,
                max_bullets=max_bullets,
            )
            self._log_signal(
                signal,
                quote=ctx.quote,
                last_close=last_close,
                rsi_value=last_rsi,
                atr_value=last_atr,
                regime_type=regime_eff.regime_type,
                trailing_active=trail.trailing_stop_active,
            )
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
        if getattr(ctx, "qqq_regime_bear_volatile", False) and bool(
            self._settings.REGIME_BEAR_VOLATILE_BLOCK_ENTRIES,
        ):
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.SKIP_MARKET_REGIME,
                    message="QQQ_BearVolatile_strategy_entry_blocked",
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    metadata={**dict(common_meta), "skip_code": "skip_market_regime"},
                ),
                log_event="strategy_skip",
            )
            return signals
        stale_lim = float(self._settings.QUOTE_STALENESS_SECONDS)
        if qa_live is not None and qa_live > stale_lim:
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.SKIP_STALE_QUOTE,
                    message=f"quote_age_exceeds_limit age={qa_live:.3f}s max={stale_lim:.3f}s",
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    metadata={**dict(common_meta), "skip_code": "skip_stale_quote"},
                ),
                log_event="strategy_skip",
            )
            return signals
        if self._settings.LIQUIDITY_GATE_ENABLED and "volume" in bars.columns and len(bars) >= 21:
            vol5m = float(bars["volume"].iloc[-1])
            avg_vol20 = float(bars["volume"].iloc[-20:].mean())
            thr = float(self._settings.LIQUIDITY_THRESHOLD)
            if avg_vol20 > 0 and vol5m < thr * avg_vol20:
                self._emit_strategy_entry_skip(
                    SkipReason(
                        code=SkipCodes.SKIP_LOW_LIQUIDITY,
                        message="current_bar_volume_below_threshold_vs_20bar_average",
                        symbol=symbol,
                        rsi=last_rsi,
                        atr=last_atr,
                        price=last_close,
                        spread_pct=sp_live,
                        quote_age_seconds=qa_live,
                        strategy_bar_ts=bar_ts,
                        metadata={
                            **dict(common_meta),
                            "vol5m": vol5m,
                            "avg_vol20": avg_vol20,
                            "threshold_fraction": thr,
                            "skip_code": "skip_low_liquidity",
                        },
                    ),
                    log_event="strategy_skip",
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

        time_gate = self._time_window_snapshot(ctx.now_utc)
        if not bool(time_gate["passed"]):
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.TIME_WINDOW_FAIL,
                    message=(
                        "outside_configured_entry_time_window "
                        f"now_et={time_gate['now_et']} start={time_gate['start']} end={time_gate['end']}"
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
                        "skip_code_short": "skip_time",
                        "time_window_enabled": time_gate["enabled"],
                        "time_window_now_et": time_gate["now_et"],
                        "time_window_start": time_gate["start"],
                        "time_window_end": time_gate["end"],
                    },
                ),
                log_event="strategy_skip_time",
            )
            return signals

        band_meta: dict[str, object] = {
            "time_window_enabled": time_gate["enabled"],
            "time_window_passed": time_gate["passed"],
            "time_window_now_et": time_gate["now_et"],
            "time_window_start": time_gate["start"],
            "time_window_end": time_gate["end"],
        }
        bb: dict[str, float | bool] | None = (
            self._bollinger_snapshot(bars) if bool(self._settings.BOLLINGER_ENABLED) else None
        )
        vw: dict[str, float | bool] | None = (
            self._vwap_snapshot(bars) if bool(self._settings.VWAP_STRATEGY_ENABLED) else None
        )
        self._log.info(
            "event=strategy_candidate symbol=%s ts=%s price=%.6f vwap=%s vwap_zscore=%s "
            "rsi=%.4f rsi_threshold=%.4f bollinger_bw=%s bollinger_bw_min=%.6f "
            "adx=%.4f atr=%.6f atr_pct=%.6f time_window_passed=%s",
            symbol,
            datetime.now(UTC).isoformat(),
            last_close,
            f"{float(vw['vwap']):.6f}" if vw else "n_a",
            f"{float(vw['zscore']):.6f}" if vw else "n_a",
            last_rsi,
            rsi_entry_threshold,
            f"{float(bb['width_pct']):.6f}" if bb else "n_a",
            float(self._settings.BOLLINGER_MIN_WIDTH_PCT),
            float(regime_eff.adx),
            last_atr,
            atr_pct,
            str(time_gate["passed"]).lower(),
            extra={"symbol": symbol, "strategy": self.name},
        )
        if bool(self._settings.BOLLINGER_ENABLED):
            bb_passed = bool(bb and bb.get("passed"))
            self._log.info(
                "event=strategy_bollinger_gate symbol=%s result=%s price=%.6f "
                "lower=%s upper=%s basis=%s width_pct=%s min_width_pct=%.6f "
                "touch=%s require_touch=%s rsi=%.4f rsi_threshold=%.4f",
                symbol,
                "PASS" if bb_passed else "SKIP",
                last_close,
                f"{float(bb['lower']):.6f}" if bb else "n_a",
                f"{float(bb['upper']):.6f}" if bb else "n_a",
                f"{float(bb['basis']):.6f}" if bb else "n_a",
                f"{float(bb['width_pct']):.6f}" if bb else "n_a",
                float(self._settings.BOLLINGER_MIN_WIDTH_PCT),
                str(bool(bb.get("touch"))).lower() if bb else "n_a",
                str(bool(self._settings.BOLLINGER_REQUIRE_TOUCH)).lower(),
                last_rsi,
                rsi_entry_threshold,
                extra={"symbol": symbol, "strategy": self.name},
            )
            if bb is None or not bb_passed:
                width_msg = (
                    "bollinger_snapshot_unavailable"
                    if bb is None
                    else (
                        f"Bandwidth {float(bb['width_pct']):.6f}<"
                        f"{float(self._settings.BOLLINGER_MIN_WIDTH_PCT):.6f}"
                        if float(bb["width_pct"]) < float(self._settings.BOLLINGER_MIN_WIDTH_PCT)
                        else "bollinger_touch_required_but_price_not_at_outer_band"
                    )
                )
                self._emit_strategy_entry_skip(
                    SkipReason(
                        code=SkipCodes.BOLLINGER_FILTER_FAIL,
                        message=width_msg,
                        symbol=symbol,
                        rsi=last_rsi,
                        atr=last_atr,
                        price=last_close,
                        spread_pct=sp_live,
                        quote_age_seconds=qa_live,
                        strategy_bar_ts=bar_ts,
                        metadata={
                            **dict(common_meta),
                            "bollinger_enabled": True,
                            "bollinger_length": int(self._settings.BOLLINGER_LENGTH),
                            "bollinger_std": float(self._settings.BOLLINGER_STD),
                            "bollinger_min_width_pct": float(self._settings.BOLLINGER_MIN_WIDTH_PCT),
                            "bollinger_require_touch": bool(self._settings.BOLLINGER_REQUIRE_TOUCH),
                            "skip_code_short": "skip_bollinger",
                            **(
                                {
                                    "bollinger_basis": bb["basis"],
                                    "bollinger_upper": bb["upper"],
                                    "bollinger_lower": bb["lower"],
                                    "bollinger_width_pct": bb["width_pct"],
                                    "bollinger_touch": bb["touch"],
                                }
                                if bb
                                else {}
                            ),
                        },
                    ),
                    log_event="strategy_skip_bollinger",
                    discord_title="BOLLINGER_ENTRY_SKIP",
                )
                return signals
            band_meta.update(
                {
                    "bollinger_enabled": True,
                    "bollinger_basis": bb["basis"],
                    "bollinger_upper": bb["upper"],
                    "bollinger_lower": bb["lower"],
                    "bollinger_width_pct": bb["width_pct"],
                    "bollinger_min_width_pct": bb["min_width_pct"],
                    "bollinger_touch": bb["touch"],
                    "bollinger_require_touch": bb["require_touch"],
                },
            )

        if bool(self._settings.VWAP_STRATEGY_ENABLED):
            vw_passed = bool(vw and vw.get("passed"))
            self._log.info(
                "event=strategy_vwap_gate symbol=%s result=%s price=%.6f "
                "vwap=%s lower=%s upper=%s zscore=%s z_threshold=%.6f "
                "distance_pct=%s rsi=%.4f rsi_threshold=%.4f",
                symbol,
                "PASS" if vw_passed else "SKIP",
                last_close,
                f"{float(vw['vwap']):.6f}" if vw else "n_a",
                f"{float(vw['lower']):.6f}" if vw else "n_a",
                f"{float(vw['upper']):.6f}" if vw else "n_a",
                f"{float(vw['zscore']):.6f}" if vw else "n_a",
                float(self._settings.VWAP_Z_THRESHOLD),
                f"{float(vw['distance_pct']):.6f}" if vw else "n_a",
                last_rsi,
                rsi_entry_threshold,
                extra={"symbol": symbol, "strategy": self.name},
            )
            if vw is None or not vw_passed:
                self._emit_strategy_entry_skip(
                    SkipReason(
                        code=SkipCodes.VWAP_FILTER_FAIL,
                        message=(
                            "vwap_snapshot_unavailable"
                            if vw is None
                            else (
                                f"VWAP_Z {float(vw['zscore']):.6f}>"
                                f"{-float(self._settings.VWAP_Z_THRESHOLD):.6f}"
                            )
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
                            "vwap_strategy_enabled": True,
                            "vwap_length": int(self._settings.VWAP_LENGTH),
                            "vwap_z_threshold": float(self._settings.VWAP_Z_THRESHOLD),
                            "skip_code_short": "skip_vwap",
                            **(
                                {
                                    "vwap": vw["vwap"],
                                    "vwap_upper": vw["upper"],
                                    "vwap_lower": vw["lower"],
                                    "vwap_zscore": vw["zscore"],
                                    "vwap_distance_pct": vw["distance_pct"],
                                    "vwap_deviation": vw["deviation"],
                                }
                                if vw
                                else {}
                            ),
                        },
                    ),
                    log_event="strategy_skip_vwap",
                    discord_title="VWAP_ENTRY_SKIP",
                )
                return signals
            band_meta.update(
                {
                    "vwap_strategy_enabled": True,
                    "vwap": vw["vwap"],
                    "vwap_upper": vw["upper"],
                    "vwap_lower": vw["lower"],
                    "vwap_zscore": vw["zscore"],
                    "vwap_z_threshold": vw["z_threshold"],
                    "vwap_distance_pct": vw["distance_pct"],
                    "vwap_deviation": vw["deviation"],
                },
            )

        vwap_z_for_adx = (
            float(vw["zscore"])
            if vw is not None and isinstance(vw.get("zscore"), int | float)
            else None
        )
        deep_vwap_threshold = -float(self._settings.VWAP_Z_THRESHOLD) * 1.5
        adx_high = float(self._settings.ADX_HIGH)
        if bool(self._settings.VWAP_STRATEGY_ENABLED) and float(regime_eff.adx) >= adx_high and (
            vwap_z_for_adx is None or vwap_z_for_adx > deep_vwap_threshold
        ):
            self._emit_strategy_entry_skip(
                SkipReason(
                    code=SkipCodes.ADX_FILTER_FAIL,
                    message=(
                        f"ADX {float(regime_eff.adx):.4f}>={adx_high:.4f} and "
                        f"VWAP_Z {vwap_z_for_adx if vwap_z_for_adx is not None else 'n_a'} "
                        f"> {deep_vwap_threshold:.4f}"
                    ),
                    symbol=symbol,
                    rsi=last_rsi,
                    atr=last_atr,
                    adx=float(regime_eff.adx),
                    price=last_close,
                    spread_pct=sp_live,
                    quote_age_seconds=qa_live,
                    strategy_bar_ts=bar_ts,
                    metadata={
                        "skip_code_short": "skip_adx",
                        "adx_high": adx_high,
                        "vwap_deep_z_required": deep_vwap_threshold,
                        "vwap_zscore": vwap_z_for_adx,
                        **dict(common_meta),
                        **dict(band_meta),
                    },
                ),
                log_event="strategy_skip_adx",
            )
            return signals

        aggressive_bypass_used = False
        hybrid_entry_mode = bool(self._settings.BOLLINGER_ENABLED and self._settings.VWAP_STRATEGY_ENABLED)
        hybrid_regime_softened = False
        if not regime_eff.allow_rsi_long and hybrid_entry_mode:
            hybrid_regime_softened = True
            self._log.info(
                "event=strategy_regime_softened symbol=%s old_allow=false reason=%s "
                "adx=%.4f adx_high=%.4f sma_slope=%.6f vwap_zscore=%s",
                symbol,
                regime_eff.reason,
                float(regime_eff.adx),
                adx_high,
                float(regime_eff.sma_slope),
                f"{vwap_z_for_adx:.6f}" if vwap_z_for_adx is not None else "n_a",
                extra={"symbol": symbol, "strategy": self.name},
            )
        elif not regime_eff.allow_rsi_long:
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
                **band_meta,
                "hybrid_entry_mode": hybrid_entry_mode,
                "hybrid_regime_softened": hybrid_regime_softened,
                "sma_filter_passed": bool(float(regime_eff.sma_slope) > 0.0),
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

        reason = f"rsi_oversold rsi={last_rsi:.2f} close={last_close:.4f}"
        if hybrid_entry_mode:
            reason = (
                f"hybrid_vwap_bollinger_reversion rsi={last_rsi:.2f} "
                f"vwap_z={vwap_z_for_adx if vwap_z_for_adx is not None else 'n_a'} "
                f"bb_width={float(band_meta.get('bollinger_width_pct', 0.0)):.4f}"
            )

        signal = Signal(
            symbol=symbol,
            action=SignalAction.ENTER_LONG,
            reason=reason,
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
