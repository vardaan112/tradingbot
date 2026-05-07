"""Merged strategy thresholds: static Settings + optional autotuned JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


from config.constants import LOGGER_APP

_LOG = logging.getLogger(LOGGER_APP)


@dataclass(frozen=True)
class StrategyRuntimeThresholds:
    """Live thresholds used by ``RSIMeanReversionStrategy`` (subset of Settings)."""

    rsi_oversold: float
    rsi_exit: float
    adx_range_max: float
    atr_stop_multiplier: float
    trail_atr_multiplier: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rsi_entry_threshold": self.rsi_oversold,
            "rsi_exit_threshold": self.rsi_exit,
            "adx_threshold": self.adx_range_max,
            "atr_stop_multiplier": self.atr_stop_multiplier,
            "atr_trailing_multiplier": self.trail_atr_multiplier,
        }


def default_dynamic_params_path() -> Path:
    return Path(__file__).resolve().parent / "dynamic_params.json"


def load_dynamic_params_file(path: Path) -> Optional[dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        return obj
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("dynamic_params unreadable path=%s err=%s", path, exc)
        return None

def resolve_dynamic_params_path(settings: Any) -> Path:
    p = Path(getattr(settings, "DYNAMIC_PARAMS_PATH", "") or "")
    root = Path.cwd()
    candidate = Path(p).expanduser()
    if not candidate.parts:
        return default_dynamic_params_path()
    if candidate.is_absolute():
        return candidate
    parts = tuple(Path(c) for c in candidate.parts[:3])
    if parts and parts[0].name == "src" or str(candidate).startswith("src"):
        rep = Path(__file__).resolve().parents[2]
        return (rep / candidate).resolve()
    return (root / candidate).resolve()


def merge_strategy_thresholds(settings: Any, dyn_path: Optional[Path] = None) -> StrategyRuntimeThresholds:
    """Prefer ``dynamic_params.json`` when ``ENABLE_AUTOTUNE`` is True and file is valid."""
    rp = dyn_path if dyn_path is not None else resolve_dynamic_params_path(settings)
    base_oversold = float(settings.RSI_OVERSOLD)
    base_exit = float(settings.RSI_EXIT)
    base_adx = float(settings.ADX_RANGE_MAX)
    base_atr_stop = float(settings.ATR_STOP_MULTIPLIER)
    base_trail = float(settings.TRAIL_ATR_MULTIPLIER)

    if not bool(getattr(settings, "ENABLE_AUTOTUNE", False)):
        return StrategyRuntimeThresholds(
            rsi_oversold=base_oversold,
            rsi_exit=base_exit,
            adx_range_max=base_adx,
            atr_stop_multiplier=base_atr_stop,
            trail_atr_multiplier=base_trail,
        )

    d = load_dynamic_params_file(rp)
    if not d:
        _LOG.info("event=strategy_runtime dynamic_params missing; using Settings")
        return StrategyRuntimeThresholds(
            rsi_oversold=base_oversold,
            rsi_exit=base_exit,
            adx_range_max=base_adx,
            atr_stop_multiplier=base_atr_stop,
            trail_atr_multiplier=base_trail,
        )

    try:
        src = str(d.get("source", "")).strip().lower()
        if src and src != "autotune":
            _LOG.warning("event=strategy_runtime unknown_source=%s; using Settings", d.get("source"))
            raise ValueError("bad source")

        rsi_oversold = float(d.get("rsi_entry_threshold", d.get("rsi_oversold", base_oversold)))
        rsi_exit = float(d.get("rsi_exit_threshold", d.get("rsi_exit", base_exit)))
        adx_mx = float(d.get("adx_threshold", d.get("adx_range_max", base_adx)))
        atr_stop = float(d["atr_stop_multiplier"])
        atr_trail = float(d["atr_trailing_multiplier"])
        if rsi_oversold >= rsi_exit or adx_mx <= 0 or atr_stop <= 0 or atr_trail <= 0:
            raise ValueError("invalid ranges")
        if not (1.0 <= rsi_oversold <= 99.0 and 1.0 <= rsi_exit <= 99.0):
            raise ValueError("rsi bounds")
        return StrategyRuntimeThresholds(
            rsi_oversold=rsi_oversold,
            rsi_exit=rsi_exit,
            adx_range_max=adx_mx,
            atr_stop_multiplier=atr_stop,
            trail_atr_multiplier=atr_trail,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.warning("event=strategy_runtime dynamic_params_invalid err=%s; using Settings", exc)
        return StrategyRuntimeThresholds(
            rsi_oversold=base_oversold,
            rsi_exit=base_exit,
            adx_range_max=base_adx,
            atr_stop_multiplier=base_atr_stop,
            trail_atr_multiplier=base_trail,
        )


__all__ = [
    "StrategyRuntimeThresholds",
    "default_dynamic_params_path",
    "load_dynamic_params_file",
    "merge_strategy_thresholds",
    "resolve_dynamic_params_path",
]
