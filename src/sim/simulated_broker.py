"""Pending fills and execution against ``SimulatedAccount`` (replay only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from strategies.base import SignalAction

from .fill_model import FillModelParams, entry_long_fill_price, exit_long_fill_price, fees_usd
from .simulated_account import SimulatedAccount


@dataclass
class PendingFill:
    execute_at_bar_index: int
    symbol: str
    action: SignalAction
    quantity: float
    strategy_name: str
    reason: str = ""


class SimulatedBroker:
    """Queues next-bar fills and applies them at subsequent bar opens."""

    def __init__(
        self,
        account: SimulatedAccount,
        *,
        fill_params: FillModelParams,
        prevent_same_bar_fills: bool = True,
    ) -> None:
        self._acct = account
        self._fill = fill_params
        self._prevent_same = prevent_same_bar_fills
        self._pending: list[PendingFill] = []

    @property
    def account(self) -> SimulatedAccount:
        return self._acct

    @property
    def pending(self) -> list[PendingFill]:
        return list(self._pending)

    def cancel_symbol_pending(self, symbol: str) -> None:
        sym = symbol.upper()
        self._pending = [p for p in self._pending if p.symbol.upper() != sym]

    def schedule(self, pending: PendingFill) -> None:
        self._pending.append(pending)

    def process_bar_open(
        self,
        *,
        bar_index: int,
        open_by_symbol: dict[str, float],
        ts_iso: str,
        on_skip: Optional[Callable[[str, str, str], None]] = None,
        volume_by_symbol: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        """Execute all pending fills targeting ``bar_index``; return event dicts.

        When ``volume_by_symbol`` is set (replay master-clock path), fills are skipped
        if that bar's volume is <= 0 so we never execute on synthetic minutes with no
        reported trade activity.
        """

        events: list[dict[str, Any]] = []
        to_run = [p for p in self._pending if p.execute_at_bar_index == bar_index]
        self._pending = [p for p in self._pending if p.execute_at_bar_index != bar_index]
        for p in to_run:
            sym_u = p.symbol.upper()
            if volume_by_symbol is not None:
                try:
                    vol = float(volume_by_symbol.get(sym_u, float("nan")))
                except (TypeError, ValueError):
                    vol = float("nan")
                if not (vol > 0.0):
                    if on_skip:
                        on_skip(
                            p.symbol,
                            "ghost_bar_zero_volume",
                            f"pending {p.action} skipped bar_volume={vol}",
                        )
                    events.append({"kind": "skip", "reason": "ghost_bar_zero_volume", "pending": p})
                    continue
            opx = float(open_by_symbol.get(sym_u, 0.0))
            if opx <= 0:
                if on_skip:
                    on_skip(p.symbol, "no_next_bar_open", f"pending {p.action} skipped")
                events.append({"kind": "skip", "reason": "no_next_bar_open", "pending": p})
                continue
            try:
                if p.action == SignalAction.ENTER_LONG:
                    px = entry_long_fill_price(opx, params=self._fill)
                    notional = px * float(p.quantity)
                    fee = fees_usd(notional, params=self._fill)
                    self._acct.open_long(
                        symbol=p.symbol,
                        quantity=p.quantity,
                        price=px,
                        fees_usd=fee,
                        ts_iso=ts_iso,
                        metadata={"strategy": p.strategy_name, "reason": p.reason},
                    )
                    events.append(
                        {
                            "kind": "fill",
                            "action": "enter_long",
                            "symbol": p.symbol,
                            "qty": p.quantity,
                            "price": px,
                            "strategy": p.strategy_name,
                        },
                    )
                elif p.action in (SignalAction.EXIT_LONG, SignalAction.EMERGENCY_EXIT_LONG):
                    px = exit_long_fill_price(opx, params=self._fill)
                    notional = px * float(p.quantity)
                    fee = fees_usd(notional, params=self._fill)
                    _, pnl = self._acct.close_long(
                        symbol=p.symbol,
                        quantity=p.quantity,
                        price=px,
                        fees_usd=fee,
                        ts_iso=ts_iso,
                        metadata={"strategy": p.strategy_name, "reason": p.reason},
                    )
                    events.append(
                        {
                            "kind": "fill",
                            "action": "exit_long",
                            "symbol": p.symbol,
                            "qty": p.quantity,
                            "price": px,
                            "pnl": pnl,
                            "strategy": p.strategy_name,
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                if on_skip:
                    on_skip(p.symbol, "fill_error", str(exc))
                events.append({"kind": "error", "error": str(exc), "pending": p})
        return events


__all__ = ["PendingFill", "SimulatedBroker"]
