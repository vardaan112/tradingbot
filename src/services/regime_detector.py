"""Hourly QQQ macro regime: Bear-Volatile vs Normal for risk sizing / entry gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd

from config.constants import LOGGER_APP
from config.settings import Settings
from strategies.indicators import atr, rsi

if TYPE_CHECKING:
    from core.market_data import BarFetcher


@dataclass(frozen=True)
class QqqRegimeSnapshot:
    bear_volatile: bool
    close: float
    sma50: float
    atr1h: float
    atr_ma: float
    atr_ratio: float
    updated_at: datetime
    error: str = ""
    anchor_symbol: str = ""
    anchor_close: float = 0.0
    anchor_sma: float = 0.0
    anchor_rsi: float = 0.0
    anchor_state: str = "Unknown"


class QqqRegimeDetector:
    """Fetch QQQ 1h bars and classify Bear-Volatile macro regime."""

    def __init__(self, settings: Settings, bar_fetcher: BarFetcher) -> None:
        self._settings = settings
        self._bars = bar_fetcher
        self._log = logging.getLogger(LOGGER_APP)
        # Conservative default: treat as bear-volatile until the first successful
        # fetch.  Prevents trading through a gap-down open while the hourly
        # refresh hasn't fired yet (up to 59 min exposure window).
        self._snap = QqqRegimeSnapshot(
            bear_volatile=True,
            close=0.0,
            sma50=0.0,
            atr1h=0.0,
            atr_ma=0.0,
            atr_ratio=1.0,
            updated_at=datetime.now(UTC),
            error="init_pending",
            anchor_symbol=str(settings.REGIME_ANCHOR_SYMBOL).upper(),
        )
        self._last_refresh_hour_utc: Optional[tuple[int, int, int, int]] = None
        if settings.QQQ_REGIME_ENABLED:
            try:
                self.refresh()
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "event=regime_startup_refresh_failed err=%s will_stay_conservative=true", exc
                )

    @property
    def snapshot(self) -> QqqRegimeSnapshot:
        return self._snap

    def refresh_if_stale(self, *, now_utc: datetime) -> None:
        """At most once per clock hour (UTC), refresh hourly QQQ metrics."""

        if not self._settings.QQQ_REGIME_ENABLED:
            return
        hour_key = (now_utc.year, now_utc.month, now_utc.day, now_utc.hour)
        if self._last_refresh_hour_utc == hour_key:
            return
        self._last_refresh_hour_utc = hour_key
        self.refresh()

    def refresh(self) -> QqqRegimeSnapshot:
        sym = str(self._settings.REGIME_QQQ_SYMBOL or "QQQ").upper()
        lookback = max(
            120,
            self._settings.REGIME_ATR_MA_LENGTH + self._settings.REGIME_ATR_LENGTH + 55,
        )
        try:
            df = self._bars.fetch_bars(sym, "1Hour", lookback_bars=lookback)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("regime_detector fetch failed symbol=%s err=%s", sym, exc)
            self._snap = QqqRegimeSnapshot(
                bear_volatile=self._snap.bear_volatile,
                close=self._snap.close,
                sma50=self._snap.sma50,
                atr1h=self._snap.atr1h,
                atr_ma=self._snap.atr_ma,
                atr_ratio=self._snap.atr_ratio,
                updated_at=datetime.now(UTC),
                error=f"fetch:{exc}",
            )
            return self._snap

        snap = _compute_snapshot(df, self._settings, sym=sym)
        if self._settings.REGIME_ADAPTIVE_RSI_ENABLED:
            anchor_sym = str(self._settings.REGIME_ANCHOR_SYMBOL or sym).upper()
            try:
                anchor_df = (
                    df
                    if anchor_sym == sym and self._settings.REGIME_ANCHOR_TIMEFRAME == "1Hour"
                    else self._bars.fetch_bars(
                        anchor_sym,
                        str(self._settings.REGIME_ANCHOR_TIMEFRAME),
                        lookback_bars=max(120, int(self._settings.REGIME_SMA_PERIOD) + 20),
                    )
                )
                snap = _with_anchor_metrics(anchor_df, self._settings, snap, sym=anchor_sym)
            except Exception as exc:  # noqa: BLE001
                self._log.warning("regime_anchor fetch failed symbol=%s err=%s", anchor_sym, exc)
        self._snap = snap
        status = "BearVolatile" if snap.bear_volatile else "Normal"
        self._log.info(
            "event=regime_detect status=%s symbol=%s close=%.4f sma50=%.4f atr1h=%.6f "
            "atr_ma=%.6f QQQ_ATR_ratio=%.6f anchor_symbol=%s anchor_state=%s "
            "anchor_rsi=%.4f anchor_close=%.4f anchor_sma=%.4f err=%s",
            status,
            sym,
            snap.close,
            snap.sma50,
            snap.atr1h,
            snap.atr_ma,
            snap.atr_ratio,
            snap.anchor_symbol or "n_a",
            snap.anchor_state,
            snap.anchor_rsi,
            snap.anchor_close,
            snap.anchor_sma,
            snap.error or "n_a",
        )
        return self._snap


def _compute_snapshot(df: pd.DataFrame, settings: Settings, *, sym: str) -> QqqRegimeSnapshot:
    ts = datetime.now(UTC)
    if df is None or df.empty or len(df) < 55:
        return QqqRegimeSnapshot(
            bear_volatile=False,
            close=0.0,
            sma50=0.0,
            atr1h=0.0,
            atr_ma=0.0,
            atr_ratio=1.0,
            updated_at=ts,
            error="insufficient_bars",
        )
    close = df["close"].astype(float)
    last_px = float(close.iloc[-1])
    sma50_s = close.rolling(50, min_periods=50).mean()
    sma50 = float(sma50_s.iloc[-1]) if not pd.isna(sma50_s.iloc[-1]) else last_px

    hl = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float)
    atr_len = int(settings.REGIME_ATR_LENGTH)
    ma_len = int(settings.REGIME_ATR_MA_LENGTH)
    atr_s = atr(hl, lo, cl, length=atr_len)
    atr1h = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    atr_ma_s = atr_s.rolling(ma_len, min_periods=min(5, ma_len)).mean()
    atr_ma = float(atr_ma_s.iloc[-1]) if not pd.isna(atr_ma_s.iloc[-1]) else atr1h
    ratio = (atr1h / atr_ma) if atr_ma > 1e-12 else 1.0

    use_sma = bool(settings.REGIME_USE_SMA50)
    thresh = float(settings.REGIME_ATR_RATIO_THRESHOLD)
    cond_px = (last_px < sma50) if use_sma else True
    bear = bool(cond_px and ratio > thresh)

    return QqqRegimeSnapshot(
        bear_volatile=bear,
        close=last_px,
        sma50=sma50,
        atr1h=atr1h,
        atr_ma=atr_ma,
        atr_ratio=ratio,
        updated_at=ts,
        error="",
        anchor_symbol=str(settings.REGIME_ANCHOR_SYMBOL).upper(),
    )


def _with_anchor_metrics(
    df: pd.DataFrame,
    settings: Settings,
    snap: QqqRegimeSnapshot,
    *,
    sym: str,
) -> QqqRegimeSnapshot:
    if df is None or df.empty:
        return snap
    close = df["close"].astype(float)
    if len(close) < max(int(settings.REGIME_RSI_PERIOD), int(settings.REGIME_SMA_PERIOD)) + 2:
        return snap
    anchor_close = float(close.iloc[-1])
    sma_s = close.rolling(int(settings.REGIME_SMA_PERIOD), min_periods=int(settings.REGIME_SMA_PERIOD)).mean()
    anchor_sma = float(sma_s.iloc[-1]) if not pd.isna(sma_s.iloc[-1]) else anchor_close
    rsi_s = rsi(close, length=int(settings.REGIME_RSI_PERIOD))
    anchor_rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50.0

    above_sma = anchor_close >= anchor_sma
    if above_sma and anchor_rsi >= float(settings.REGIME_PARABOLIC_RSI_MIN):
        state = "ParabolicBull"
    elif above_sma and anchor_rsi >= float(settings.REGIME_BULL_RSI_MIN):
        state = "Bull"
    elif (not above_sma) or anchor_rsi <= float(settings.REGIME_BEAR_RSI_MAX):
        state = "Bear"
    else:
        state = "Neutral"

    return QqqRegimeSnapshot(
        bear_volatile=snap.bear_volatile,
        close=snap.close,
        sma50=snap.sma50,
        atr1h=snap.atr1h,
        atr_ma=snap.atr_ma,
        atr_ratio=snap.atr_ratio,
        updated_at=snap.updated_at,
        error=snap.error,
        anchor_symbol=sym.upper(),
        anchor_close=anchor_close,
        anchor_sma=anchor_sma,
        anchor_rsi=anchor_rsi,
        anchor_state=state,
    )


__all__ = ["QqqRegimeDetector", "QqqRegimeSnapshot"]
