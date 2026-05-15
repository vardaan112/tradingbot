"""Strategy name registry and constructors (Phase 2 multi-strategy)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Optional

from strategies.base import Strategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.etf_rotation_strategy import ETFRotationStrategy
from strategies.momentum_strategy import MomentumTrendStrategy
from strategies.pairs_mean_reversion_strategy import PairsMeanReversionStrategy
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from strategies.vwap_pullback_strategy import VWAPPullbackStrategy

if TYPE_CHECKING:
    from config.settings import Settings
    from config.strategy_runtime import StrategyRuntimeThresholds
    from core.database import Database
    from core.state_store import StateStore


# Canonical keys registered for build_strategy / Settings validation.
_STRATEGY_BUILDERS: dict[str, type[Strategy]] = {
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "momentum": MomentumTrendStrategy,
    "breakout": BreakoutStrategy,
    "vwap_pullback": VWAPPullbackStrategy,
    "etf_rotation": ETFRotationStrategy,
    "pairs_mean_reversion": PairsMeanReversionStrategy,
}

# User-facing aliases -> canonical name used in weights / persistence.
_STRATEGY_ALIASES: dict[str, str] = {
    "rsi": "rsi_mean_reversion",
    "rsi_strategy": "rsi_mean_reversion",
    "rsi_meanrev": "rsi_mean_reversion",
    "momentum_trend": "momentum",
}


def normalize_strategy_name(name: str) -> str:
    """Return canonical registry key for ``name`` (lowercase, aliased)."""

    raw = (name or "").strip().lower()
    if not raw:
        raise ValueError("strategy name must not be empty")
    return _STRATEGY_ALIASES.get(raw, raw)


def supported_strategy_names() -> frozenset[str]:
    """Return canonical strategy keys that ``build_strategy`` accepts."""

    return frozenset(_STRATEGY_BUILDERS.keys())


def _rsi_kwargs(
    settings: "Settings",
    *,
    state_store: Optional["StateStore"] = None,
    database: Optional["Database"] = None,
    runtime_thresholds: Optional["StrategyRuntimeThresholds"] = None,
    ml_filter: Any = None,
    discord_embed_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    return {
        "settings": settings,
        "state_store": state_store,
        "database": database,
        "runtime_thresholds": runtime_thresholds,
        "ml_filter": ml_filter,
        "discord_embed_fn": discord_embed_fn,
    }


def build_strategy(
    name: str,
    settings: "Settings",
    *,
    state_store: Optional["StateStore"] = None,
    database: Optional["Database"] = None,
    runtime_thresholds: Optional["StrategyRuntimeThresholds"] = None,
    ml_filter: Any = None,
    discord_embed_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Strategy:
    """Construct a single strategy instance by name or alias."""

    key = normalize_strategy_name(name)
    cls = _STRATEGY_BUILDERS.get(key)
    if cls is None:
        raise ValueError(f"unknown strategy name: {name!r} (normalized={key!r})")
    if cls is RSIMeanReversionStrategy:
        return RSIMeanReversionStrategy(**_rsi_kwargs(
            settings,
            state_store=state_store,
            database=database,
            runtime_thresholds=runtime_thresholds,
            ml_filter=ml_filter,
            discord_embed_fn=discord_embed_fn,
        ))
    return cls(
        settings,
        state_store=state_store,
        database=database,
        runtime_thresholds=runtime_thresholds,
        ml_filter=ml_filter,
        discord_embed_fn=discord_embed_fn,
    )


def build_strategies(
    names: Sequence[str],
    settings: "Settings",
    *,
    state_store: Optional["StateStore"] = None,
    database: Optional["Database"] = None,
    runtime_thresholds: Optional["StrategyRuntimeThresholds"] = None,
    ml_filter: Any = None,
    discord_embed_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[Strategy]:
    """Build an ordered list of strategies (one instance per name entry)."""

    out: list[Strategy] = []
    for raw in names:
        out.append(
            build_strategy(
                raw,
                settings,
                state_store=state_store,
                database=database,
                runtime_thresholds=runtime_thresholds,
                ml_filter=ml_filter,
                discord_embed_fn=discord_embed_fn,
            ),
        )
    return out


__all__ = [
    "build_strategies",
    "build_strategy",
    "normalize_strategy_name",
    "supported_strategy_names",
]
