"""Account snapshot + position adapters with PDT-deprecation tolerance.

The `pattern_day_trader`, `daytrade_count`, and `daytrading_buying_power`
fields are being phased out by Alpaca (FINRA Rule 4210, effective 2026-06-04).
They are read here as optional / deprecated metadata only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from config.constants import LOGGER_APP
from utils.time_utils import now_utc

from .exceptions import AccountStateError, BrokerConnectionError, NonRetryableBrokerError
from .retries import retry_call


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return default


@dataclass(frozen=True)
class AccountSnapshot:
    """Strongly typed account snapshot.

    Required fields are buying_power and equity. PDT-related fields are
    optional and may be absent post-2026-06-04 transition.
    """

    equity: float
    last_equity: float
    cash: float
    buying_power: float
    regt_buying_power: float
    portfolio_value: float
    long_market_value: float
    short_market_value: float
    initial_margin: float
    maintenance_margin: float
    multiplier: float
    status: str
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool

    pattern_day_trader: Optional[bool] = None  # deprecated
    daytrade_count: Optional[int] = None  # deprecated
    daytrading_buying_power: Optional[float] = None  # deprecated

    fetched_at: datetime = field(default_factory=now_utc)


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    avg_entry_price: float
    side: str
    market_value: float
    cost_basis: float
    unrealized_pl: float
    current_price: float


def _to_dict(obj: Any) -> dict:
    """Coerce alpaca-py models / dicts / random objects into a dict."""
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # noqa: BLE001
                continue
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise AccountStateError(f"cannot coerce {type(obj).__name__} to dict")


def parse_account(raw: Any) -> AccountSnapshot:
    """Parse a raw account object/dict into an AccountSnapshot.

    Tolerant to missing PDT fields per Rule 4210 transition.
    """
    data = _to_dict(raw)

    try:
        return AccountSnapshot(
            equity=_safe_float(data.get("equity")),
            last_equity=_safe_float(data.get("last_equity")),
            cash=_safe_float(data.get("cash")),
            buying_power=_safe_float(data.get("buying_power")),
            regt_buying_power=_safe_float(data.get("regt_buying_power")),
            portfolio_value=_safe_float(data.get("portfolio_value")),
            long_market_value=_safe_float(data.get("long_market_value")),
            short_market_value=_safe_float(data.get("short_market_value")),
            initial_margin=_safe_float(data.get("initial_margin")),
            maintenance_margin=_safe_float(data.get("maintenance_margin")),
            multiplier=_safe_float(data.get("multiplier"), default=1.0),
            status=str(data.get("status") or "UNKNOWN"),
            trading_blocked=_safe_bool(data.get("trading_blocked")),
            transfers_blocked=_safe_bool(data.get("transfers_blocked")),
            account_blocked=_safe_bool(data.get("account_blocked")),
            pattern_day_trader=(
                _safe_bool(data["pattern_day_trader"])
                if "pattern_day_trader" in data and data["pattern_day_trader"] is not None
                else None
            ),
            daytrade_count=(
                _safe_int(data["daytrade_count"])
                if "daytrade_count" in data and data["daytrade_count"] is not None
                else None
            ),
            daytrading_buying_power=(
                _safe_float(data["daytrading_buying_power"])
                if "daytrading_buying_power" in data
                and data["daytrading_buying_power"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise AccountStateError(f"failed to parse account snapshot: {exc}") from exc


def parse_position(raw: Any) -> PositionSnapshot:
    """Parse a raw position object/dict into a PositionSnapshot."""
    data = _to_dict(raw)
    return PositionSnapshot(
        symbol=str(data.get("symbol", "")).upper(),
        qty=_safe_float(data.get("qty")),
        avg_entry_price=_safe_float(data.get("avg_entry_price")),
        side=str(data.get("side") or "long"),
        market_value=_safe_float(data.get("market_value")),
        cost_basis=_safe_float(data.get("cost_basis")),
        unrealized_pl=_safe_float(data.get("unrealized_pl")),
        current_price=_safe_float(data.get("current_price")),
    )


class AccountAdapter:
    """Thin REST adapter for snapshots + reconciliation."""

    def __init__(
        self,
        trading: TradingClient,
        *,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
    ) -> None:
        self._client = trading
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._log = logging.getLogger(LOGGER_APP)

    def _retry(self, op_name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return retry_call(
            fn,
            *args,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            op_name=op_name,
            logger=self._log,
            **kwargs,
        )

    def fetch_account(self) -> AccountSnapshot:
        try:
            raw = self._retry("get_account", self._client.get_account)
        except NonRetryableBrokerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrokerConnectionError(f"get_account failed: {exc}") from exc
        return parse_account(raw)

    def fetch_positions(self) -> list[PositionSnapshot]:
        try:
            raw = self._retry("get_all_positions", self._client.get_all_positions)
        except NonRetryableBrokerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrokerConnectionError(f"get_all_positions failed: {exc}") from exc
        return [parse_position(p) for p in (raw or [])]

    def fetch_open_orders(self) -> list[Any]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
        try:
            return self._retry("get_orders_open", self._client.get_orders, filter=request)
        except NonRetryableBrokerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrokerConnectionError(f"get_orders failed: {exc}") from exc
