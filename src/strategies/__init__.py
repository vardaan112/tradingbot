"""Strategy package: base interface, indicators, universe, and concrete strategies."""

from .base import Signal, SignalAction, Strategy, StrategyContext
from .filters import RegimeSnapshot, adx, compute_regime_snapshot, sma
from .indicators import atr, rsi
from .rsi_strategy import RSIMeanReversionStrategy
from .universe import UniverseFilter

__all__ = [
    "Signal",
    "SignalAction",
    "Strategy",
    "StrategyContext",
    "RegimeSnapshot",
    "adx",
    "sma",
    "compute_regime_snapshot",
    "atr",
    "rsi",
    "RSIMeanReversionStrategy",
    "UniverseFilter",
]
