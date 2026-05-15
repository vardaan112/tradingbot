"""Live shadow portfolios (Phase 7): virtual fills per strategy, no broker orders."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from config.constants import LOGGER_STRATEGY
from core.market_data import Quote
from services.ensemble import WeightedEnsembleEngine, votes_to_contributing_json
from sim.simulated_account import SimulatedAccount
from strategies.base import Signal, SignalAction
from strategies.registry import normalize_strategy_name

if TYPE_CHECKING:
    from config.settings import Settings
    from core.database import Database
    from services.ensemble import EnsembleDecision

_LOG = logging.getLogger(LOGGER_STRATEGY)

SHADOW_SOURCE = "shadow"


def shadow_buy_fill_price(quote: Quote | None, *, model: str) -> float:
    """Simulated long entry price (pay the offer or mid)."""

    if quote is None or quote.bid <= 0 or quote.ask <= quote.bid:
        return 0.0
    if model.strip().lower() == "midpoint":
        return float(quote.mid())
    return float(quote.ask)


def shadow_sell_fill_price(quote: Quote | None, *, model: str) -> float:
    """Simulated long exit price (hit the bid or mid)."""

    if quote is None or quote.bid <= 0 or quote.ask <= quote.bid:
        return 0.0
    if model.strip().lower() == "midpoint":
        return float(quote.mid())
    return float(quote.bid)


@dataclass
class _OpenLeg:
    """Track shadow entry for completed_trades on exit."""

    symbol: str
    opened_at: str
    entry_price: float
    quantity: float


class ShadowPortfolioManager:
    """One ``SimulatedAccount`` per active strategy plus optional ensemble book."""

    def __init__(
        self,
        settings: "Settings",
        database: "Database",
        *,
        run_id: str = "shadow_live",
    ) -> None:
        self._settings = settings
        self._database = database
        self._run_id = str(run_id).strip() or "shadow_live"
        init_eq = float(settings.SHADOW_INITIAL_EQUITY)
        self._accounts: dict[str, SimulatedAccount] = {}
        for name in settings.active_strategies_list:
            key = normalize_strategy_name(name)
            self._accounts[key] = SimulatedAccount(initial_equity=init_eq)
        self._ensemble: SimulatedAccount | None = None
        if settings.ENSEMBLE_ENABLED and settings.STRATEGY_RUN_MODE in ("ensemble", "both"):
            self._ensemble = SimulatedAccount(initial_equity=init_eq)
        self._open_legs: dict[str, dict[str, _OpenLeg]] = {}
        self._last_equity_snap_mono: dict[str, float] = {}
        self._fill_model = str(settings.SHADOW_FILL_MODEL).strip().lower()
        self._snap_interval = float(settings.SHADOW_RECORD_INTERVAL_SECONDS)

    @property
    def has_ensemble_book(self) -> bool:
        return self._ensemble is not None

    def _legs(self, book: str) -> dict[str, _OpenLeg]:
        return self._open_legs.setdefault(book, {})

    def _mark_prices(self, acct: SimulatedAccount, sym: str, px: float) -> dict[str, float]:
        u = sym.strip().upper()
        out: dict[str, float] = {u: float(px)}
        for s, pos in acct.positions.items():
            if s.upper() != u:
                out[s.upper()] = float(pos.avg_entry_price)
        return out

    def _equity_for_sizing(self, acct: SimulatedAccount, sym: str, px: float, ts_iso: str) -> float:
        _, eq = acct.mark_to_market(self._mark_prices(acct, sym, px), ts_iso=ts_iso)
        return float(eq)

    def _size_qty(self, acct: SimulatedAccount, symbol: str, ref_px: float, fill_px: float, ts_iso: str) -> float:
        if fill_px <= 0 or ref_px <= 0:
            return 0.0
        eq = self._equity_for_sizing(acct, symbol, ref_px, ts_iso=ts_iso)
        risk_pct = float(self._settings.MAX_RISK_PER_TRADE_PCT)
        cap = float(self._settings.MAX_EQUITY_USAGE_USD)
        notional = min(eq * risk_pct, cap)
        if notional <= 0:
            return 0.0
        qty = notional / float(fill_px)
        if self._settings.ENABLE_FRACTIONAL:
            return max(0.0, float(qty))
        q = int(qty)
        return float(q) if q >= 1 else 0.0

    def _maybe_equity_snapshot(self, book: str, acct: SimulatedAccount, prices: dict[str, float], ts_iso: str) -> None:
        now_m = time.monotonic()
        last = self._last_equity_snap_mono.get(book, 0.0)
        if self._snap_interval > 0 and (now_m - last) < self._snap_interval:
            return
        self._last_equity_snap_mono[book] = now_m
        unreal, eq = acct.mark_to_market(prices, ts_iso=ts_iso)
        gross = sum(float(acct.positions[s].quantity) * float(prices.get(s.upper(), 0.0)) for s in acct.positions)
        try:
            self._database.record_equity_snapshot(
                source=SHADOW_SOURCE,
                timestamp=ts_iso,
                run_id=self._run_id,
                strategy_name=book,
                cash=float(acct.cash),
                equity=float(eq),
                realized_pnl=float(acct.realized_pnl),
                unrealized_pnl=float(unreal),
                gross_exposure=float(gross),
                net_exposure=float(gross),
                benchmark_equity=None,
                metadata={"shadow": True, "book": book},
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("event=shadow_equity_snapshot_failed book=%s err=%s", book, exc)

    def _record_close(
        self,
        *,
        book: str,
        sym: str,
        leg: _OpenLeg,
        exit_px: float,
        closed_at: str,
        realized_pnl: float,
    ) -> None:
        qty = float(leg.quantity)
        ret = (realized_pnl / (leg.entry_price * qty)) if leg.entry_price * qty > 1e-12 else None
        try:
            self._database.record_completed_trade(
                trade_id=f"shadow-{book}-{sym}-{uuid.uuid4().hex[:12]}",
                symbol=sym,
                side="long",
                quantity=qty,
                entry_price=float(leg.entry_price),
                exit_price=float(exit_px),
                realized_pnl=float(realized_pnl),
                realized_return=float(ret) if ret is not None else None,
                opened_at=leg.opened_at,
                closed_at=closed_at,
                strategy_name=book,
                risk_mode="shadow",
                regime_type=None,
                sentiment_score=None,
                sentiment_label=None,
                is_canary=0,
                metadata={"shadow": True, "shadow_book": book},
                source=SHADOW_SOURCE,
                replay_run_id=None,
                invalid_for_ml=True,
                invalid_for_kelly=True,
                entry_fill_source=self._fill_model,
                exit_fill_source=self._fill_model,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("event=shadow_completed_trade_failed book=%s err=%s", book, exc)

    def _apply_to_book(
        self,
        *,
        book: str,
        acct: SimulatedAccount,
        signal: Signal,
        quote: Quote | None,
        ts_iso: str,
    ) -> None:
        sym = signal.symbol.strip().upper()
        act = signal.action
        if act == SignalAction.NONE:
            return
        ref = float(signal.reference_price or 0.0)
        mid = float(quote.mid()) if quote and quote.bid > 0 and quote.ask > quote.bid else ref
        mark_px = ref if ref > 0 else mid

        if act == SignalAction.ENTER_LONG:
            if sym in acct.positions:
                return
            buy_px = shadow_buy_fill_price(quote, model=self._fill_model)
            if buy_px <= 0:
                buy_px = mark_px
            if buy_px <= 0:
                return
            qty = self._size_qty(acct, sym, mark_px, buy_px, ts_iso)
            if qty <= 0:
                return
            try:
                acct.open_long(symbol=sym, quantity=qty, price=buy_px, fees_usd=0.0, ts_iso=ts_iso, metadata={"shadow": True})
            except ValueError as exc:
                _LOG.debug("event=shadow_enter_skipped book=%s sym=%s err=%s", book, sym, exc)
                return
            self._legs(book)[sym] = _OpenLeg(symbol=sym, opened_at=ts_iso, entry_price=buy_px, quantity=qty)
        elif act in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG):
            if sym not in acct.positions:
                return
            sell_px = shadow_sell_fill_price(quote, model=self._fill_model)
            if sell_px <= 0:
                sell_px = mark_px
            if sell_px <= 0:
                return
            leg = self._legs(book).pop(sym, None)
            try:
                _fil, pnl = acct.close_long(symbol=sym, quantity=None, price=sell_px, fees_usd=0.0, ts_iso=ts_iso, metadata={"shadow": True})
            except ValueError as exc:
                _LOG.debug("event=shadow_exit_skipped book=%s sym=%s err=%s", book, sym, exc)
                if leg is not None:
                    self._legs(book)[sym] = leg
                return
            if leg is not None:
                self._record_close(book=book, sym=sym, leg=leg, exit_px=sell_px, closed_at=ts_iso, realized_pnl=float(pnl))
        prices = self._mark_prices(acct, sym, mark_px)
        self._maybe_equity_snapshot(book, acct, prices, ts_iso)

    def on_symbol(
        self,
        *,
        symbol: str,
        timestamp_iso: str,
        raw_signals: list[Signal],
        quote: Quote | None,
        ensemble_decision: Optional["EnsembleDecision"] = None,
        ensemble_signal: Optional[Signal] = None,
    ) -> None:
        """Apply latest per-strategy signals and optional ensemble signal to shadow books."""

        sym_u = symbol.strip().upper()
        by_strat = WeightedEnsembleEngine._latest_signal_per_strategy(raw_signals, sym_u)
        for name in self._settings.active_strategies_list:
            key = normalize_strategy_name(name)
            acct = self._accounts.get(key)
            if acct is None:
                continue
            sig = by_strat.get(key)
            if sig is None:
                sig = Signal(sym_u, SignalAction.NONE, "shadow_missing", 0.0, 0.0, {}, key, 0.0)
            self._apply_to_book(book=key, acct=acct, signal=sig, quote=quote, ts_iso=timestamp_iso)

        if self._ensemble is not None:
            if (
                ensemble_decision is not None
                and ensemble_decision.final_action != SignalAction.NONE
                and self._database is not None
            ):
                try:
                    wscore = (
                        float(ensemble_decision.weighted_exit_score)
                        if ensemble_decision.final_action
                        in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                        else float(ensemble_decision.weighted_enter_score)
                    )
                    th = (
                        float(ensemble_decision.exit_threshold)
                        if ensemble_decision.final_action
                        in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG)
                        else float(ensemble_decision.enter_threshold)
                    )
                    self._database.record_strategy_decision(
                        source=SHADOW_SOURCE,
                        timestamp=timestamp_iso,
                        symbol=sym_u,
                        final_action=ensemble_decision.final_action.value,
                        run_id=self._run_id,
                        decision_type="weighted_ensemble",
                        weighted_score=wscore,
                        threshold=th,
                        contributing_signals_json=votes_to_contributing_json(ensemble_decision.contributing_votes),
                        metadata={
                            "shadow": True,
                            "weighted_enter_score": ensemble_decision.weighted_enter_score,
                            "weighted_exit_score": ensemble_decision.weighted_exit_score,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("event=shadow_strategy_decision_failed err=%s", exc)
            if ensemble_signal is not None:
                self._apply_to_book(
                    book="ensemble",
                    acct=self._ensemble,
                    signal=ensemble_signal,
                    quote=quote,
                    ts_iso=timestamp_iso,
                )

        px = float(quote.mid()) if quote and quote.bid > 0 and quote.ask > quote.bid else 0.0
        if px > 0:
            for book, acct in self._accounts.items():
                self._maybe_equity_snapshot(book, acct, self._mark_prices(acct, sym_u, px), timestamp_iso)
            if self._ensemble is not None:
                self._maybe_equity_snapshot(
                    "ensemble",
                    self._ensemble,
                    self._mark_prices(self._ensemble, sym_u, px),
                    timestamp_iso,
                )


__all__ = [
    "SHADOW_SOURCE",
    "ShadowPortfolioManager",
    "shadow_buy_fill_price",
    "shadow_sell_fill_price",
]
