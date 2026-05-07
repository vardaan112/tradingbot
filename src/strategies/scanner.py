"""Liquidity-ranked US equity scanner (daily refresh after market open).

Uses Alpaca ``get_all_assets`` + batched daily ``StockBars`` volume means.
Persists the last successful ranking under ``STATE_DIR/universe_scan_cache.json``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from config.constants import LOGGER_APP
from config.settings import Settings
from core.market_data import BarFetcher
from core.state_store import StateStore

_LOG = logging.getLogger(LOGGER_APP)

_ALLOWED_PRIMARY = frozenset(
    {
        AssetExchange.NYSE,
        AssetExchange.NASDAQ,
        AssetExchange.NYSEARCA,
        AssetExchange.ARCA,
        AssetExchange.AMEX,
        AssetExchange.BATS,
    },
)


@dataclass(frozen=True)
class UniverseScanRecord:
    """Last successful scanner artifact (Eastern calendar date keyed)."""

    as_of_et_date: str  # YYYY-MM-DD
    symbols: list[str]
    ts_utc: str
    reason: str = "ok"


def _scan_from_state_payload(raw: dict[str, Any]) -> UniverseScanRecord:
    return UniverseScanRecord(
        as_of_et_date=str(raw["as_of_et_date"]),
        symbols=list(raw["symbols"]),
        ts_utc=str(raw["ts_utc"]),
        reason=str(raw.get("reason", "ok")),
    )


def load_scan_record(state: StateStore) -> Optional[UniverseScanRecord]:
    raw = state.load_universe_scan_raw()
    if not raw:
        return None
    try:
        return _scan_from_state_payload(raw)
    except (KeyError, TypeError, ValueError):
        return None


def persist_scan_record(state: StateStore, record: UniverseScanRecord) -> None:
    from dataclasses import asdict

    state.save_universe_scan_raw(asdict(record))


def _asset_filter_ok(settings: Settings, asset: object) -> bool:
    """Tradable US equities on major exchanges at/above MIN_PRICE (last known)."""
    if getattr(asset, "asset_class", None) != AssetClass.US_EQUITY:
        return False
    if getattr(asset, "status", None) != AssetStatus.ACTIVE:
        return False
    if not bool(getattr(asset, "tradable", False)):
        return False
    ex = getattr(asset, "exchange", None)
    if ex not in _ALLOWED_PRIMARY:
        return False
    sym = str(getattr(asset, "symbol", "")).strip().upper()
    if "." in sym or len(sym) > 8:
        return False  # skip preferred / odd formats conservatively
    return True


def _collect_candidates(settings: Settings, trading: TradingClient) -> list[str]:
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    try:
        assets = trading.get_all_assets(filter=req)
    except Exception as exc:  # noqa: BLE001
        _LOG.error("scanner: get_all_assets failed: %s", exc)
        return []

    picks: list[str] = []
    for a in assets or []:
        if not _asset_filter_ok(settings, a):
            continue
        picks.append(str(a.symbol).upper())

    picks.sort()
    cap = settings.SCANNER_MAX_CANDIDATES
    if len(picks) > cap:
        # Deterministic: alphabetical truncation after filters (liquidity resolves ranking).
        picks = picks[:cap]
    return picks


def refresh_universe_now(
    settings: Settings,
    *,
    trading: TradingClient,
    bar_fetcher: BarFetcher,
    state: StateStore,
    force: bool = False,
) -> UniverseScanRecord:
    """Run a full scan, persist it, and return the record.

    When ``force`` is False, skips heavy work when today's ET row already matches
    persisted cache unless the persisted list is empty.
    """
    from utils.time_utils import now_eastern  # lazy: keeps tests patchable

    now_et = now_eastern()
    today_et = now_et.strftime("%Y-%m-%d")
    persisted = load_scan_record(state)
    if (
        not force
        and persisted is not None
        and persisted.as_of_et_date == today_et
        and persisted.symbols
    ):
        return persisted

    candidates = _collect_candidates(settings, trading)
    if not candidates:
        msg = "scanner: no_candidates_from_assets_api"
        _LOG.warning(msg)
        fall = UniverseScanRecord(
            as_of_et_date=today_et,
            symbols=list(dict.fromkeys(settings.symbols_list)),
            ts_utc=datetime.now(timezone.utc).isoformat(),
            reason=msg,
        )
        persist_scan_record(state, fall)
        return fall

    vol_map = bar_fetcher.fetch_mean_daily_volume_batch(
        candidates,
        calendar_lookback_days=settings.SCANNER_VOLUME_LOOKBACK_DAYS,
        min_rows=settings.SCANNER_MIN_HISTORY_DAYS,
    )

    ranked_pairs: list[tuple[str, float]] = []
    for sym, (avg_v, last_close) in vol_map.items():
        if avg_v <= 0 or last_close < settings.MIN_PRICE:
            continue
        if avg_v * last_close < settings.MIN_AVG_DOLLAR_VOLUME:
            continue
        ranked_pairs.append((sym, float(avg_v)))

    ranked_pairs.sort(key=lambda x: x[1], reverse=True)
    syms_out = [s for s, _ in ranked_pairs][: settings.SCANNER_TOP_N]

    if len(syms_out) < settings.SCANNER_TOP_N:
        _LOG.warning(
            "scanner: ranked=%d_below_top_n_requests date=%s",
            len(syms_out),
            today_et,
        )

    if not syms_out:
        msg = "scanner: liquidity_filter_yielded_empty"
        _LOG.error(msg)
        rec = UniverseScanRecord(
            as_of_et_date=today_et,
            symbols=list(dict.fromkeys(settings.symbols_list)),
            ts_utc=datetime.now(timezone.utc).isoformat(),
            reason=msg,
        )
        persist_scan_record(state, rec)
        return rec

    rec = UniverseScanRecord(
        as_of_et_date=today_et,
        symbols=syms_out,
        ts_utc=datetime.now(timezone.utc).isoformat(),
        reason="ok",
    )
    persist_scan_record(state, rec)
    _LOG.info(
        "event=universe_scan ranked=%s as_of_et=%s",
        ",".join(syms_out),
        today_et,
    )
    return rec


def maybe_refresh_after_open(
    settings: Settings,
    *,
    trading: TradingClient,
    bar_fetcher: BarFetcher,
    state: StateStore,
    session_is_open: bool,
) -> Optional[UniverseScanRecord]:
    """Refresh once per ET session-day after configured open clock if market open."""
    from utils.time_utils import now_eastern

    if not settings.DYNAMIC_UNIVERSE_ENABLED:
        return None
    if not session_is_open:
        return None

    now_et = now_eastern()
    today_et = now_et.strftime("%Y-%m-%d")
    gate = now_et.replace(
        hour=settings.SCANNER_REFRESH_HOUR_ET,
        minute=settings.SCANNER_REFRESH_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    if now_et < gate:
        return None

    cached = load_scan_record(state)
    if cached is not None and cached.as_of_et_date == today_et and cached.symbols:
        return cached

    return refresh_universe_now(
        settings,
        trading=trading,
        bar_fetcher=bar_fetcher,
        state=state,
        force=True,
    )


def merge_tradeable_universe(settings: Settings, scanned: Optional[list[str]]) -> list[str]:
    """Build the quote/subscription universe.

    When dynamic scanning is enabled and ``scanned`` is non-empty, that list drives
    the tradeable equity set; ``SYMBOLS`` is still merged first as a failover seed
    whenever the scanned list would otherwise fully replace diversification.
    """
    crit: list[str] = []
    crit.append(settings.CANARY_SYMBOL)
    crit.append(settings.BLACK_SWAN_SYMBOL)
    crit.append(settings.CORRELATION_LEADER_SYMBOL)
    crit.extend(settings.correlation_follower_symbols_list)

    if settings.DYNAMIC_UNIVERSE_ENABLED and scanned:
        core = list(dict.fromkeys(settings.symbols_list + scanned))
    else:
        core = list(dict.fromkeys(settings.symbols_list))

    merged: list[str] = []
    seen: set[str] = set()
    for s in core + crit:
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            merged.append(u)
    return merged


def symbols_for_strategy_ticks(
    settings: Settings,
    scanned: Optional[list[str]],
    *,
    broker_position_symbols: set[str],
) -> list[str]:
    """Symbols the RSI loop should evaluate each tick."""
    if settings.DYNAMIC_UNIVERSE_ENABLED and scanned:
        core = list(dict.fromkeys(scanned))
    else:
        core = list(dict.fromkeys(settings.symbols_list))
    for sym in sorted(s.upper() for s in broker_position_symbols if s):
        if sym not in core:
            core.append(sym)
    return core


__all__ = [
    "UniverseScanRecord",
    "merge_tradeable_universe",
    "maybe_refresh_after_open",
    "refresh_universe_now",
    "symbols_for_strategy_ticks",
    "load_scan_record",
    "persist_scan_record",
]
