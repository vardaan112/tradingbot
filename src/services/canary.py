"""Live canary check: a one-time tiny round-trip trade at startup.

Goal: prove credentials, order submission, fills, reconciliation, and clean
flatten work end-to-end before letting the main trading loop run.

Safety contract (non-negotiable):
- Skips entirely outside true live mode (paper, dry-run, or master switch off).
- Skips if the kill switch is latched (NEVER bypassed).
- Skips if a successful canary already ran today (persisted under STATE_DIR).
- Submits a LIMIT order only - no market orders.
- A whole-share path uses a conservative DAY limit buy and a marketable limit
  IOC sell at filled qty.
- A fractional path attempts a fractional DAY limit; if the broker rejects
  fractional limit orders, the canary aborts with explicit operator guidance
  rather than silently falling back to a market order.
- If any fragment remains open at timeout the canary tries to flatten via the
  same limit-IOC primitive; if that fails, the canary aborts startup loudly.

The canary is implemented as an isolated module so the existing strategy
order stack (`core.orders.OrderService`) remains unaltered.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from config.constants import LOGGER_APP, LOGGER_ORDERS
from config.settings import Settings
from core.account import AccountAdapter
from core.exceptions import (
    BrokerConnectionError,
    KillSwitchLatchedError,
    NonRetryableBrokerError,
    OrderPlacementError,
    QuoteUnavailableError,
)
from core.market_clock import MarketClock
from core.market_data import BarFetcher, Quote
from risk.compliance import ComplianceAdapter
from risk.killswitch import KillSwitch
from utils.ids import generate_client_order_id
from utils.price_utils import is_valid_quote, mid_price, round_to_tick, spread_pct
from utils.time_utils import now_utc, today_eastern


CANARY_STRATEGY_NAME = "canary"


# ---------------------------------------------------------------------------
# Result + persistence types
# ---------------------------------------------------------------------------


@dataclass
class CanaryResult:
    """Outcome of a canary attempt."""

    success: bool
    reason: str
    started_at: str = ""
    completed_at: str = ""
    symbol: str = ""
    buy_coid: str = ""
    sell_coid: str = ""
    buy_filled_qty: float = 0.0
    sell_filled_qty: float = 0.0
    buy_avg_price: float = 0.0
    sell_avg_price: float = 0.0
    notional_attempted: float = 0.0
    fractional: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class CanaryPersistedRecord:
    """What we persist to runtime to prevent re-running the same trading day."""

    date: str
    success: bool
    completed_at: str
    symbol: str
    notional_attempted: float
    fractional: bool


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def canary_state_path(settings: Settings) -> Path:
    return Path(settings.STATE_DIR) / settings.CANARY_PERSIST_FILENAME


def canary_already_succeeded_today(settings: Settings) -> bool:
    """True iff the persisted canary record is for today and was successful."""
    data = _read_json(canary_state_path(settings))
    if not data:
        return False
    try:
        return bool(data.get("success")) and str(data.get("date")) == today_eastern().isoformat()
    except (TypeError, ValueError):
        return False


def persist_canary_record(settings: Settings, record: CanaryPersistedRecord) -> None:
    _atomic_write_json(canary_state_path(settings), asdict(record))


# ---------------------------------------------------------------------------
# Canary service
# ---------------------------------------------------------------------------


class CanaryService:
    """Run the live-canary round trip with strict pre-checks and timeouts."""

    def __init__(
        self,
        settings: Settings,
        *,
        trading: TradingClient,
        account_adapter: AccountAdapter,
        bar_fetcher: BarFetcher,
        market_clock: MarketClock,
        kill_switch: KillSwitch,
        compliance: ComplianceAdapter,
    ) -> None:
        self._settings = settings
        self._client = trading
        self._account = account_adapter
        self._bar_fetcher = bar_fetcher
        self._clock = market_clock
        self._kill = kill_switch
        self._compliance = compliance
        self._log = logging.getLogger(LOGGER_APP)
        self._log_orders = logging.getLogger(LOGGER_ORDERS)

    # ----------------------------------------------------------------- public

    async def run(self) -> CanaryResult:
        """Execute the canary. Returns a CanaryResult describing the outcome."""
        symbol = self._settings.CANARY_SYMBOL
        notional = float(self._settings.CANARY_NOTIONAL_USD)
        started_at = now_utc().isoformat()
        log_extra = {"symbol": symbol, "strategy": CANARY_STRATEGY_NAME}

        self._log.info(
            "event=canary_start symbol=%s notional_usd=%.2f timeout_s=%.0f",
            symbol, notional, self._settings.CANARY_TIMEOUT_SECONDS,
            extra=log_extra,
        )

        # Pre-check 1: kill switch
        if self._kill.is_latched():
            return self._abort(
                "kill_switch_latched",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        # Pre-check 2: market open + entry window allowed
        try:
            session = self._clock.get_session(force_refresh=True)
        except BrokerConnectionError as exc:
            return self._precheck_fail(
                f"clock_fetch_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if not session.is_open:
            return self._precheck_fail(
                "market_closed",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if not self._clock.can_open_new_position(session):
            return self._precheck_fail(
                "outside_entry_window",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        # Pre-check 3: account snapshot + compliance
        try:
            account = self._account.fetch_account()
        except (BrokerConnectionError, NonRetryableBrokerError) as exc:
            return self._precheck_fail(
                f"account_fetch_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        compliance_decision = self._compliance.decide(account)
        if not compliance_decision.allow_new_entries:
            return self._precheck_fail(
                f"compliance_block:{compliance_decision.reason}",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        # Pre-check 4: no existing position or open order in the canary symbol
        try:
            positions = self._account.fetch_positions()
            open_orders = self._account.fetch_open_orders() or []
        except (BrokerConnectionError, NonRetryableBrokerError) as exc:
            return self._precheck_fail(
                f"position_fetch_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if any(p.symbol.upper() == symbol.upper() for p in positions):
            return self._precheck_fail(
                "existing_position_in_canary_symbol",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if any(
            str(getattr(o, "symbol", "")).upper() == symbol.upper()
            for o in open_orders
        ):
            return self._precheck_fail(
                "existing_open_order_in_canary_symbol",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        # Pre-check 5: asset is tradable; cache fractionable flag for sizing
        try:
            asset = self._client.get_asset(symbol)
        except Exception as exc:  # noqa: BLE001
            return self._precheck_fail(
                f"asset_fetch_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if not bool(getattr(asset, "tradable", False)):
            return self._precheck_fail(
                "asset_not_tradable",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        fractionable = bool(getattr(asset, "fractionable", False))

        # Pre-check 6: fresh, in-spec quote
        try:
            quote = self._bar_fetcher.fetch_latest_quote(symbol)
        except (BrokerConnectionError, QuoteUnavailableError) as exc:
            return self._precheck_fail(
                f"quote_fetch_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if not is_valid_quote(
            quote.bid,
            quote.ask,
            quote_age_seconds=quote.age_seconds(),
            max_age_seconds=self._settings.QUOTE_STALENESS_SECONDS,
        ):
            return self._precheck_fail(
                "quote_invalid_or_stale",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        try:
            sp = spread_pct(quote.bid, quote.ask)
        except ValueError as exc:
            return self._precheck_fail(
                f"spread_compute_failed:{exc}",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        if sp > self._settings.SPREAD_FILTER_PCT:
            return self._precheck_fail(
                f"spread_too_wide:{sp:.6f}>{self._settings.SPREAD_FILTER_PCT:.6f}",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        # Decide whole-share vs fractional path.
        share_price = quote.ask
        if share_price <= 0:
            return self._precheck_fail(
                "non_positive_ask",
                started_at=started_at, symbol=symbol, notional=notional,
            )
        whole_shares_possible = notional >= share_price

        if whole_shares_possible:
            qty: float = float(int(notional // share_price))
            if qty < 1:
                qty = 1.0
            return await self._run_round_trip(
                symbol=symbol,
                qty=qty,
                quote=quote,
                fractional=False,
                notional_attempted=notional,
                started_at=started_at,
            )

        # Fractional path - require fractionable asset support.
        if not fractionable:
            self._log.error(
                "event=canary_abort reason=asset_not_fractionable symbol=%s "
                "notional_usd=%.4f share_price=%.4f. To proceed: increase "
                "CANARY_NOTIONAL_USD above one share price OR pick a "
                "fractionable canary symbol.",
                symbol, notional, share_price,
                extra=log_extra,
            )
            return self._abort(
                "asset_not_fractionable",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        frac_qty = round(notional / share_price, 6)
        if frac_qty <= 0:
            return self._precheck_fail(
                "fractional_qty_non_positive",
                started_at=started_at, symbol=symbol, notional=notional,
            )

        return await self._run_round_trip(
            symbol=symbol,
            qty=float(frac_qty),
            quote=quote,
            fractional=True,
            notional_attempted=notional,
            started_at=started_at,
        )

    # ----------------------------------------------------------------- core

    async def _run_round_trip(
        self,
        *,
        symbol: str,
        qty: float,
        quote: Quote,
        fractional: bool,
        notional_attempted: float,
        started_at: str,
    ) -> CanaryResult:
        log_extra = {"symbol": symbol, "strategy": CANARY_STRATEGY_NAME}

        # Conservative buy limit: bid + 0.25 * spread, rounded down to a tick.
        spread = quote.ask - quote.bid
        if spread <= 0:
            return self._precheck_fail(
                "invalid_spread_at_submit",
                started_at=started_at, symbol=symbol, notional=notional_attempted,
            )
        raw_buy_price = quote.bid + 0.25 * spread
        buy_limit = round_to_tick(raw_buy_price, mode="down")

        buy_coid = generate_client_order_id(CANARY_STRATEGY_NAME, symbol, "buy")

        buy_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=buy_limit,
            client_order_id=buy_coid,
        )

        self._log_orders.info(
            "event=canary_buy_submitted symbol=%s qty=%.6f limit=%.4f "
            "fractional=%s coid=%s",
            symbol, qty, buy_limit, fractional, buy_coid,
            extra={
                "symbol": symbol,
                "strategy": CANARY_STRATEGY_NAME,
                "client_order_id": buy_coid,
            },
        )

        try:
            buy_placed = self._client.submit_order(buy_request)
        except Exception as exc:  # noqa: BLE001
            # Distinguish fractional-limit rejection so operators get clear guidance.
            err_text = str(exc).lower()
            if fractional and (
                "fractional" in err_text
                or "limit" in err_text and "not allowed" in err_text
                or "must be a positive integer" in err_text
                or "qty" in err_text and "integer" in err_text
            ):
                self._log.error(
                    "event=canary_abort reason=fractional_limit_rejected_by_broker "
                    "symbol=%s qty=%.6f limit=%.4f underlying=%s. "
                    "To proceed: enable a supported fractional-limit path OR "
                    "raise CANARY_NOTIONAL_USD above one whole share price "
                    "of the canary symbol so the bot can use a whole-share "
                    "DAY limit instead.",
                    symbol, qty, buy_limit, exc,
                    extra=log_extra,
                )
                return self._abort(
                    "fractional_limit_rejected_by_broker",
                    started_at=started_at,
                    symbol=symbol,
                    notional=notional_attempted,
                    fractional=True,
                )
            # Reconcile via COID before assuming the order didn't land.
            return await self._reconcile_or_abort(
                buy_coid,
                started_at=started_at,
                symbol=symbol,
                notional=notional_attempted,
                fractional=fractional,
                stage="buy_submit",
                exc=exc,
                qty=qty,
                limit=buy_limit,
                quote=quote,
            )

        buy_broker_id: Optional[str] = (
            getattr(buy_placed, "id", None) or getattr(buy_placed, "order_id", None)
        )
        buy_broker_id_str = str(buy_broker_id) if buy_broker_id is not None else None

        # Wait for the buy to fill (or partial-fill). We poll get_order_by_id
        # because the trading-stream is not necessarily wired up here.
        deadline = asyncio.get_event_loop().time() + float(self._settings.CANARY_TIMEOUT_SECONDS)
        buy_filled_qty, buy_avg_price = await self._wait_for_fill(
            broker_id=buy_broker_id_str,
            coid=buy_coid,
            deadline=deadline,
            symbol=symbol,
        )

        if buy_filled_qty <= 0:
            self._log.error(
                "event=canary_abort reason=buy_unfilled symbol=%s coid=%s",
                symbol, buy_coid,
                extra=log_extra,
            )
            # Best-effort cancel; harmless if already terminal.
            await self._best_effort_cancel(buy_broker_id_str)
            return self._abort(
                "buy_unfilled",
                started_at=started_at,
                symbol=symbol,
                notional=notional_attempted,
                fractional=fractional,
                buy_coid=buy_coid,
            )

        self._log_orders.info(
            "event=canary_buy_filled symbol=%s coid=%s filled_qty=%.6f avg=%.4f",
            symbol, buy_coid, buy_filled_qty, buy_avg_price,
            extra={
                "symbol": symbol,
                "strategy": CANARY_STRATEGY_NAME,
                "client_order_id": buy_coid,
            },
        )

        # If the buy was partial-filled and the rest is still open, cancel it.
        await self._best_effort_cancel(buy_broker_id_str)

        # Submit matching sell at marketable limit IOC for exactly the filled qty.
        # Refresh the quote first so we price aggressively against the latest book.
        try:
            sell_quote = self._bar_fetcher.fetch_latest_quote(symbol)
        except (BrokerConnectionError, QuoteUnavailableError):
            sell_quote = quote  # fall back to the buy-side quote

        try:
            sell_mid = mid_price(sell_quote.bid, sell_quote.ask)
        except ValueError:
            sell_mid = (quote.bid + quote.ask) / 2.0
        offset = abs(self._settings.EMERGENCY_AGGRESSIVENESS_PCT) * sell_mid
        raw_sell_price = max(0.01, sell_quote.bid - offset)
        sell_limit = round_to_tick(raw_sell_price, mode="down")
        sell_coid = generate_client_order_id(CANARY_STRATEGY_NAME, symbol, "sell")

        sell_qty: float = (
            float(int(buy_filled_qty)) if not fractional else round(float(buy_filled_qty), 6)
        )
        if sell_qty <= 0:
            # The buy filled less than one whole share but we are in
            # whole-share mode: we cannot flatten; fail loudly.
            self._log.critical(
                "event=canary_abort reason=cannot_flatten_partial_buy symbol=%s "
                "buy_filled_qty=%.6f fractional=%s",
                symbol, buy_filled_qty, fractional,
                extra=log_extra,
            )
            return self._abort(
                "cannot_flatten_partial_buy",
                started_at=started_at,
                symbol=symbol,
                notional=notional_attempted,
                fractional=fractional,
                buy_coid=buy_coid,
            )

        sell_request = LimitOrderRequest(
            symbol=symbol,
            qty=sell_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.IOC,
            limit_price=sell_limit,
            client_order_id=sell_coid,
        )

        self._log_orders.info(
            "event=canary_sell_submitted symbol=%s qty=%.6f limit=%.4f coid=%s",
            symbol, sell_qty, sell_limit, sell_coid,
            extra={
                "symbol": symbol,
                "strategy": CANARY_STRATEGY_NAME,
                "client_order_id": sell_coid,
            },
        )

        try:
            sell_placed = self._client.submit_order(sell_request)
        except Exception as exc:  # noqa: BLE001
            return await self._reconcile_or_abort(
                sell_coid,
                started_at=started_at,
                symbol=symbol,
                notional=notional_attempted,
                fractional=fractional,
                stage="sell_submit",
                exc=exc,
                qty=sell_qty,
                limit=sell_limit,
                quote=sell_quote,
                buy_coid=buy_coid,
                buy_filled_qty=buy_filled_qty,
                buy_avg_price=buy_avg_price,
            )

        sell_broker_id = (
            getattr(sell_placed, "id", None) or getattr(sell_placed, "order_id", None)
        )
        sell_broker_id_str = str(sell_broker_id) if sell_broker_id is not None else None

        sell_filled_qty, sell_avg_price = await self._wait_for_fill(
            broker_id=sell_broker_id_str,
            coid=sell_coid,
            deadline=deadline,
            symbol=symbol,
        )

        # IOC orders that don't fully fill are auto-cancelled by the broker.
        if abs(sell_filled_qty - buy_filled_qty) > 1e-6:
            self._log.critical(
                "event=canary_abort reason=position_not_flat symbol=%s "
                "buy_filled=%.6f sell_filled=%.6f",
                symbol, buy_filled_qty, sell_filled_qty,
                extra=log_extra,
            )
            return self._abort(
                "position_not_flat_after_sell",
                started_at=started_at,
                symbol=symbol,
                notional=notional_attempted,
                fractional=fractional,
                buy_coid=buy_coid,
                sell_coid=sell_coid,
                buy_filled_qty=buy_filled_qty,
                sell_filled_qty=sell_filled_qty,
                buy_avg_price=buy_avg_price,
                sell_avg_price=sell_avg_price,
            )

        self._log_orders.info(
            "event=canary_sell_filled symbol=%s coid=%s filled_qty=%.6f avg=%.4f",
            symbol, sell_coid, sell_filled_qty, sell_avg_price,
            extra={
                "symbol": symbol,
                "strategy": CANARY_STRATEGY_NAME,
                "client_order_id": sell_coid,
            },
        )

        completed_at = now_utc().isoformat()
        result = CanaryResult(
            success=True,
            reason="ok",
            started_at=started_at,
            completed_at=completed_at,
            symbol=symbol,
            buy_coid=buy_coid,
            sell_coid=sell_coid,
            buy_filled_qty=buy_filled_qty,
            sell_filled_qty=sell_filled_qty,
            buy_avg_price=buy_avg_price,
            sell_avg_price=sell_avg_price,
            notional_attempted=notional_attempted,
            fractional=fractional,
        )
        self._log.info(
            "event=canary_complete symbol=%s buy_filled=%.6f sell_filled=%.6f "
            "buy_avg=%.4f sell_avg=%.4f fractional=%s",
            symbol, buy_filled_qty, sell_filled_qty, buy_avg_price,
            sell_avg_price, fractional,
            extra=log_extra,
        )
        return result

    # ----------------------------------------------------------------- helpers

    async def _wait_for_fill(
        self,
        *,
        broker_id: Optional[str],
        coid: str,
        deadline: float,
        symbol: str,
    ) -> tuple[float, float]:
        """Poll order state until fill / timeout. Returns (filled_qty, avg_price)."""
        last_qty = 0.0
        last_avg = 0.0
        while asyncio.get_event_loop().time() < deadline:
            try:
                if broker_id is not None:
                    order = self._client.get_order_by_id(broker_id)
                else:
                    order = self._client.get_order_by_client_id(coid)
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "canary order poll failed symbol=%s coid=%s err=%s",
                    symbol, coid, exc,
                    extra={
                        "symbol": symbol,
                        "strategy": CANARY_STRATEGY_NAME,
                        "client_order_id": coid,
                    },
                )
                await asyncio.sleep(0.5)
                continue

            status = str(getattr(order, "status", "")).lower()
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
            filled_avg = float(getattr(order, "filled_avg_price", 0) or 0)
            if filled_qty > last_qty:
                last_qty = filled_qty
            if filled_avg > 0:
                last_avg = filled_avg

            if status in {"filled"}:
                return last_qty, last_avg
            if status in {"canceled", "expired", "rejected", "done_for_day"}:
                # Even on terminal cancel/expire, we may still have a partial fill.
                return last_qty, last_avg

            await asyncio.sleep(0.5)
        return last_qty, last_avg

    async def _best_effort_cancel(self, broker_id: Optional[str]) -> None:
        if broker_id is None:
            return
        try:
            self._client.cancel_order_by_id(broker_id)
        except Exception:  # noqa: BLE001
            # The order may already be terminal; that's fine.
            return

    async def _reconcile_or_abort(
        self,
        coid: str,
        *,
        started_at: str,
        symbol: str,
        notional: float,
        fractional: bool,
        stage: str,
        exc: BaseException,
        qty: float,
        limit: float,
        quote: Quote,
        buy_coid: str = "",
        buy_filled_qty: float = 0.0,
        buy_avg_price: float = 0.0,
    ) -> CanaryResult:
        """When submission raised, attempt COID reconciliation before aborting."""
        log_extra = {"symbol": symbol, "strategy": CANARY_STRATEGY_NAME}
        try:
            existing = self._client.get_order_by_client_id(coid)
        except Exception as get_exc:  # noqa: BLE001
            self._log.error(
                "event=canary_abort reason=ambiguous_submission stage=%s symbol=%s "
                "submit_err=%s lookup_err=%s coid=%s qty=%.6f limit=%.4f",
                stage, symbol, exc, get_exc, coid, qty, limit,
                extra=log_extra,
            )
            return self._abort(
                f"ambiguous_submission:{stage}",
                started_at=started_at,
                symbol=symbol,
                notional=notional,
                fractional=fractional,
                buy_coid=buy_coid,
                sell_coid=coid if stage == "sell_submit" else "",
                buy_filled_qty=buy_filled_qty,
                buy_avg_price=buy_avg_price,
            )
        broker_id = getattr(existing, "id", None) or getattr(existing, "order_id", None)
        if broker_id is None:
            self._log.error(
                "event=canary_abort reason=submission_failed stage=%s symbol=%s "
                "err=%s coid=%s",
                stage, symbol, exc, coid, extra=log_extra,
            )
            return self._abort(
                f"submission_failed:{stage}",
                started_at=started_at,
                symbol=symbol,
                notional=notional,
                fractional=fractional,
                buy_coid=buy_coid,
                sell_coid=coid if stage == "sell_submit" else "",
                buy_filled_qty=buy_filled_qty,
                buy_avg_price=buy_avg_price,
            )
        # Order did land; resume the wait-for-fill path. The caller can re-enter
        # the standard waiting flow but at this point the integration is already
        # too tangled - safer to log a critical and abort startup so an operator
        # inspects the order manually.
        self._log.critical(
            "event=canary_abort reason=reconciled_but_unverified stage=%s symbol=%s "
            "broker_id=%s coid=%s. Inspect Alpaca for status before re-running.",
            stage, symbol, broker_id, coid, extra=log_extra,
        )
        return self._abort(
            f"reconciled_but_unverified:{stage}",
            started_at=started_at,
            symbol=symbol,
            notional=notional,
            fractional=fractional,
            buy_coid=buy_coid,
            sell_coid=coid if stage == "sell_submit" else "",
            buy_filled_qty=buy_filled_qty,
            buy_avg_price=buy_avg_price,
        )

    def _precheck_fail(
        self,
        reason: str,
        *,
        started_at: str,
        symbol: str,
        notional: float,
        fractional: bool = False,
    ) -> CanaryResult:
        self._log.error(
            "event=canary_precheck_fail symbol=%s reason=%s notional=%.4f",
            symbol, reason, notional,
            extra={"symbol": symbol, "strategy": CANARY_STRATEGY_NAME},
        )
        return CanaryResult(
            success=False,
            reason=reason,
            started_at=started_at,
            completed_at=now_utc().isoformat(),
            symbol=symbol,
            notional_attempted=notional,
            fractional=fractional,
        )

    def _abort(
        self,
        reason: str,
        *,
        started_at: str,
        symbol: str,
        notional: float,
        fractional: bool = False,
        buy_coid: str = "",
        sell_coid: str = "",
        buy_filled_qty: float = 0.0,
        sell_filled_qty: float = 0.0,
        buy_avg_price: float = 0.0,
        sell_avg_price: float = 0.0,
    ) -> CanaryResult:
        return CanaryResult(
            success=False,
            reason=reason,
            started_at=started_at,
            completed_at=now_utc().isoformat(),
            symbol=symbol,
            buy_coid=buy_coid,
            sell_coid=sell_coid,
            buy_filled_qty=buy_filled_qty,
            sell_filled_qty=sell_filled_qty,
            buy_avg_price=buy_avg_price,
            sell_avg_price=sell_avg_price,
            notional_attempted=notional,
            fractional=fractional,
        )


# ---------------------------------------------------------------------------
# Top-level guarded entry called by main.py
# ---------------------------------------------------------------------------


async def maybe_run_canary(settings: Settings) -> bool:
    """Run the canary if all gates allow it. Returns True iff main loop may proceed.

    Gating order (fail-closed where it matters):
      1. Skip if not on the live endpoint.
      2. Skip if DRY_RUN is true.
      3. Skip if LIVE_TRADING_ENABLED is false.
      4. Skip if RUN_LIVE_CANARY_ON_STARTUP is false.
      5. Abort startup if kill switch is latched (NEVER bypassed).
      6. Skip if the canary already succeeded today.
      7. Otherwise: build minimal clients, run the canary, persist result.

    The canary uses an isolated submission path so the strategy order stack
    in `core.orders.OrderService` is not weakened.
    """
    log = logging.getLogger(LOGGER_APP)
    log_extra = {"symbol": settings.CANARY_SYMBOL, "strategy": CANARY_STRATEGY_NAME}

    if not settings.is_live_endpoint:
        log.info("canary skipped: not live endpoint", extra=log_extra)
        return True
    if settings.DRY_RUN:
        log.info("canary skipped: DRY_RUN=true", extra=log_extra)
        return True
    if not settings.LIVE_TRADING_ENABLED:
        log.info("canary skipped: LIVE_TRADING_ENABLED=false", extra=log_extra)
        return True
    if not settings.RUN_LIVE_CANARY_ON_STARTUP:
        log.info("canary skipped: RUN_LIVE_CANARY_ON_STARTUP=false", extra=log_extra)
        return True

    # Kill switch must be honored before any live action.
    from core.state_store import StateStore  # local import: avoid cycles at module import

    state = StateStore(Path(settings.STATE_DIR))
    kill = KillSwitch(state, drawdown_pct=settings.KILL_SWITCH_DRAWDOWN_PCT)
    if kill.is_latched():
        log.critical(
            "event=canary_abort reason=kill_switch_latched symbol=%s",
            settings.CANARY_SYMBOL, extra=log_extra,
        )
        raise KillSwitchLatchedError("kill switch latched at canary entry")

    if canary_already_succeeded_today(settings):
        log.info(
            "canary skipped: already succeeded today (%s)",
            today_eastern().isoformat(),
            extra=log_extra,
        )
        return True

    # Build minimal clients for the canary. We avoid building streams here.
    from core.alpaca_clients import build_alpaca_clients

    clients = build_alpaca_clients(settings)
    account_adapter = AccountAdapter(
        clients.trading,
        max_attempts=settings.RETRY_MAX_ATTEMPTS,
        base_delay=settings.RETRY_BASE_DELAY_SECONDS,
        max_delay=settings.RETRY_MAX_DELAY_SECONDS,
    )
    market_clock = MarketClock(
        clients.trading,
        max_attempts=settings.RETRY_MAX_ATTEMPTS,
        base_delay=settings.RETRY_BASE_DELAY_SECONDS,
        max_delay=settings.RETRY_MAX_DELAY_SECONDS,
    )
    bar_fetcher = BarFetcher(
        clients.historical_data,
        feed=clients.resolved_feed,
        max_attempts=settings.RETRY_MAX_ATTEMPTS,
        base_delay=settings.RETRY_BASE_DELAY_SECONDS,
        max_delay=settings.RETRY_MAX_DELAY_SECONDS,
    )
    compliance = ComplianceAdapter(settings)

    service = CanaryService(
        settings,
        trading=clients.trading,
        account_adapter=account_adapter,
        bar_fetcher=bar_fetcher,
        market_clock=market_clock,
        kill_switch=kill,
        compliance=compliance,
    )

    try:
        result = await service.run()
    except KillSwitchLatchedError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("canary crashed: %s", exc, extra=log_extra)
        return False

    # Persist either way so we have an audit trail; only success blocks re-run.
    persist_canary_record(
        settings,
        CanaryPersistedRecord(
            date=today_eastern().isoformat(),
            success=result.success,
            completed_at=result.completed_at or now_utc().isoformat(),
            symbol=result.symbol,
            notional_attempted=result.notional_attempted,
            fractional=result.fractional,
        ),
    )

    if not result.success:
        log.error(
            "event=canary_abort reason=%s symbol=%s notional=%.4f",
            result.reason, result.symbol, result.notional_attempted,
            extra=log_extra,
        )
    return bool(result.success)
