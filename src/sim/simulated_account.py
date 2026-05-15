"""In-memory portfolio state for historical replay (no broker APIs)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from core.account import AccountSnapshot, PositionSnapshot


@dataclass
class SimulatedFill:
    """One execution fill."""

    fill_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fees_usd: float
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedOrder:
    """Order record (replay bookkeeping)."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str
    created_at: str
    filled_at: Optional[str] = None
    avg_fill_price: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatedPosition:
    """Open long position."""

    symbol: str
    quantity: float
    avg_entry_price: float
    opened_at: str


@dataclass
class SimulatedAccount:
    """Cash + long positions + PnL for one virtual portfolio."""

    initial_equity: float
    cash: float = 0.0
    realized_pnl: float = 0.0
    positions: dict[str, SimulatedPosition] = field(default_factory=dict)
    orders: list[SimulatedOrder] = field(default_factory=list)
    fills: list[SimulatedFill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash == 0.0 and self.initial_equity > 0:
            self.cash = float(self.initial_equity)

    def mark_to_market(self, prices: dict[str, float], *, ts_iso: str) -> tuple[float, float]:
        """Return (unrealized_pnl, equity) using mark prices for open positions."""

        unreal = 0.0
        mv_total = 0.0
        for sym, pos in self.positions.items():
            px = float(prices.get(sym.upper(), pos.avg_entry_price))
            mv = px * float(pos.quantity)
            cost = float(pos.avg_entry_price) * float(pos.quantity)
            unreal += mv - cost
            mv_total += mv
        equity = float(self.cash) + mv_total
        return unreal, equity

    def open_long(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
        fees_usd: float,
        ts_iso: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SimulatedFill:
        sym = symbol.upper()
        q = float(quantity)
        cost = q * float(price) + float(fees_usd)
        if cost > self.cash + 1e-9:
            raise ValueError(f"insufficient_cash need={cost} have={self.cash}")
        self.cash -= cost
        existing = self.positions.get(sym)
        if existing is not None:
            new_q = float(existing.quantity) + q
            new_avg = (
                float(existing.avg_entry_price) * float(existing.quantity) + float(price) * q
            ) / new_q
            self.positions[sym] = SimulatedPosition(sym, new_q, new_avg, existing.opened_at)
        else:
            self.positions[sym] = SimulatedPosition(sym, q, float(price), ts_iso)
        oid = str(uuid.uuid4())
        fil = SimulatedFill(
            fill_id=oid,
            symbol=sym,
            side="buy",
            quantity=q,
            price=float(price),
            fees_usd=float(fees_usd),
            timestamp=ts_iso,
            metadata=dict(metadata or {}),
        )
        self.fills.append(fil)
        self.orders.append(
            SimulatedOrder(
                order_id=oid,
                symbol=sym,
                side="buy",
                quantity=q,
                status="filled",
                created_at=ts_iso,
                filled_at=ts_iso,
                avg_fill_price=float(price),
                metadata=dict(metadata or {}),
            ),
        )
        return fil

    def close_long(
        self,
        *,
        symbol: str,
        quantity: Optional[float],
        price: float,
        fees_usd: float,
        ts_iso: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[SimulatedFill, float]:
        """Close long (full or partial). Returns (fill, realized_pnl_this_close)."""

        sym = symbol.upper()
        pos = self.positions.get(sym)
        if pos is None:
            raise ValueError(f"no_position symbol={sym}")
        q_close = float(pos.quantity) if quantity is None else min(float(quantity), float(pos.quantity))
        if q_close <= 0:
            raise ValueError("close quantity must be > 0")
        proceeds = q_close * float(price) - float(fees_usd)
        cost_basis = q_close * float(pos.avg_entry_price)
        pnl = proceeds - cost_basis
        self.cash += proceeds
        self.realized_pnl += pnl
        oid = str(uuid.uuid4())
        fil = SimulatedFill(
            fill_id=oid,
            symbol=sym,
            side="sell",
            quantity=q_close,
            price=float(price),
            fees_usd=float(fees_usd),
            timestamp=ts_iso,
            metadata=dict(metadata or {}),
        )
        self.fills.append(fil)
        self.orders.append(
            SimulatedOrder(
                order_id=oid,
                symbol=sym,
                side="sell",
                quantity=q_close,
                status="filled",
                created_at=ts_iso,
                filled_at=ts_iso,
                avg_fill_price=float(price),
                metadata=dict(metadata or {}),
            ),
        )
        rem = float(pos.quantity) - q_close
        if rem <= 1e-12:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = SimulatedPosition(sym, rem, float(pos.avg_entry_price), pos.opened_at)
        return fil, pnl

    def account_snapshot(self, *, prices: dict[str, float], ts: datetime) -> AccountSnapshot:
        unreal, equity = self.mark_to_market(prices, ts_iso=ts.isoformat())
        long_mv = 0.0
        for sym, pos in self.positions.items():
            px = float(prices.get(sym.upper(), pos.avg_entry_price))
            long_mv += px * float(pos.quantity)
        return AccountSnapshot(
            equity=float(equity),
            last_equity=float(equity),
            cash=float(self.cash),
            buying_power=float(self.cash),
            regt_buying_power=float(self.cash),
            portfolio_value=float(equity),
            long_market_value=float(long_mv),
            short_market_value=0.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            multiplier=1.0,
            status="ACTIVE",
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            fetched_at=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
        )

    def positions_snapshot(self, prices: dict[str, float]) -> dict[str, PositionSnapshot]:
        out: dict[str, PositionSnapshot] = {}
        for sym, pos in self.positions.items():
            px = float(prices.get(sym.upper(), pos.avg_entry_price))
            mv = px * float(pos.quantity)
            cost = float(pos.avg_entry_price) * float(pos.quantity)
            out[sym] = PositionSnapshot(
                symbol=sym,
                qty=float(pos.quantity),
                avg_entry_price=float(pos.avg_entry_price),
                side="long",
                market_value=mv,
                cost_basis=cost,
                unrealized_pl=mv - cost,
                current_price=px,
            )
        return out


__all__ = [
    "SimulatedAccount",
    "SimulatedFill",
    "SimulatedOrder",
    "SimulatedPosition",
]
