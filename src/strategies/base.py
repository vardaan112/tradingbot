"""Strategy base interface.

Concrete strategies subclass `Strategy` and produce `Signal` objects from a
`StrategyContext` snapshot. The orchestrator owns the per-tick context build
and is responsible for risk, sizing, and order placement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Optional

import pandas as pd

from core.account import AccountSnapshot, PositionSnapshot
from core.market_data import Quote


class SignalAction(str, Enum):
    """Discrete actions a strategy can request."""

    NONE = "none"
    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"
    EMERGENCY_EXIT_LONG = "emergency_exit_long"


@dataclass(frozen=True)
class Signal:
    """A request from the strategy layer to the orchestrator."""

    symbol: str
    action: SignalAction
    reason: str
    reference_price: float = 0.0
    atr: float = 0.0
    metadata: dict = field(default_factory=dict)
    strategy_name: Optional[str] = None
    confidence: float = 1.0


@dataclass
class StrategyContext:
    """Per-tick snapshot passed into `Strategy.evaluate`."""

    symbol: str
    bars: pd.DataFrame  # OHLCV indexed by timestamp ascending
    quote: Optional[Quote]
    account: AccountSnapshot
    positions_by_symbol: dict[str, PositionSnapshot]
    open_order_symbols: set[str]
    now_utc: datetime
    feed: str
    sentiment_overlay: Optional[dict[str, Any]] = field(default=None)
    qqq_regime_bear_volatile: bool = False
    regime_anchor_state: str = "Unknown"
    regime_anchor_rsi: Optional[float] = None
    regime_anchor_close: Optional[float] = None
    regime_anchor_sma: Optional[float] = None
    anti_martingale_risk_mode: Optional[str] = None
    anti_martingale_multiplier: Optional[float] = None
    recent_trade_outcomes_hint: str = ""
    all_bars_by_symbol: dict[str, pd.DataFrame] = field(default_factory=dict)
    all_quotes_by_symbol: dict[str, Quote] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.symbol.upper() in self.positions_by_symbol

    @property
    def has_open_order(self) -> bool:
        return self.symbol.upper() in self.open_order_symbols

    @property
    def position(self) -> Optional[PositionSnapshot]:
        return self.positions_by_symbol.get(self.symbol.upper())

    def quote_age_vs_eval_seconds(self) -> Optional[float]:
        """Seconds from quote timestamp to the evaluation clock (``now_utc``).

        Live ticks set ``now_utc`` to the orchestrator's evaluation instant; replay
        sets it to the simulated bar time while the synthetic quote is stamped with
        that same bar. Using this for staleness keeps real-time freshness checks
        meaningful without treating historical replay quotes as wall-clock stale.
        """
        if self.quote is None:
            return None
        try:
            return float(self.quote.age_seconds(reference=self.now_utc))
        except (AttributeError, TypeError, ValueError):
            return None


class Strategy(ABC):
    """Base class for trading strategies."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        """Return zero or more Signals for this evaluation cycle."""
        raise NotImplementedError

    def warmup_lookback(self) -> int:
        """How many historical bars the strategy needs to be ready."""
        return 200
