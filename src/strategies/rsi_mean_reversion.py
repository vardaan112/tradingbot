"""Backward-compatibility shim for the legacy import path.

The canonical implementation now lives in `strategies.rsi_strategy`. This
module re-exports the same `RSIMeanReversionStrategy` class so existing
imports such as

    from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

continue to work without modification.

Do not add new behavior here. Edit `rsi_strategy.py` instead.
"""

from __future__ import annotations

from .rsi_strategy import RSIMeanReversionStrategy

__all__ = ["RSIMeanReversionStrategy"]
