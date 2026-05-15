"""Long-only pairs mean reversion on follower vs leader spread (Phase 3)."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config.settings import Settings

from .base import Signal, SignalAction, Strategy, StrategyContext


class PairsMeanReversionStrategy(Strategy):
    """When follower is cheap vs leader (low spread z), enter long follower."""

    name = "pairs_mean_reversion"

    def __init__(self, settings: Settings, **_kwargs: Any) -> None:
        self._s = settings

    def _pairs(self) -> dict[str, str]:
        try:
            raw = json.loads(self._s.PAIRS_CONFIG_JSON or "{}")
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            fk = str(k).strip().upper()
            lv = str(v).strip().upper()
            if fk and lv:
                out[fk] = lv
        return out

    def warmup_lookback(self) -> int:
        return int(self._s.PAIRS_LOOKBACK_BARS) + 15

    def evaluate(self, ctx: StrategyContext) -> Iterable[Signal]:
        sym = ctx.symbol.upper()
        if not self._s.PAIRS_ENABLED:
            return []

        pairs = self._pairs()
        leader = pairs.get(sym)
        if leader is None:
            return []

        all_bars = ctx.all_bars_by_symbol or {}
        df_f = ctx.bars
        df_l = all_bars.get(leader)
        if df_f is None or df_l is None or df_f.empty or df_l.empty:
            return []

        lookback = int(self._s.PAIRS_LOOKBACK_BARS)
        if len(df_f) < lookback + 5 or len(df_l) < lookback + 5:
            return []

        # Align on common timestamps
        cf = df_f["close"].astype(float)
        cl = df_l["close"].astype(float)
        joined = pd.DataFrame({"f": cf, "l": cl}).dropna()
        if len(joined) < lookback + 5:
            return []

        lf = np.log(joined["f"].values)
        ll = np.log(joined["l"].values)
        win_f = lf[-lookback:]
        win_l = ll[-lookback:]
        cov = np.cov(win_f, win_l, ddof=0)
        var_l = float(np.var(win_l))
        if var_l < 1e-8:
            beta = 1.0
        else:
            raw_beta = float(cov[0, 1] / var_l)
            if not np.isfinite(raw_beta) or abs(raw_beta) < 0.05:
                beta = 1.0
            else:
                beta = float(np.clip(raw_beta, -3.0, 3.0))
        spread = win_f - beta * win_l
        mu = float(np.mean(spread))
        sig = float(np.std(spread, ddof=0))
        last_spread = float(spread[-1])
        z = (last_spread - mu) / sig if sig > 1e-12 else 0.0

        last_px = float(joined["f"].iloc[-1])
        meta = {
            "leader": leader,
            "beta": beta,
            "spread_z": z,
            "spread_mean": mu,
            "spread_std": sig,
            "thresholds": {
                "entry_z": float(self._s.PAIRS_ENTRY_Z),
                "exit_z": float(self._s.PAIRS_EXIT_Z),
            },
        }

        if ctx.has_position and ctx.position is not None and str(ctx.position.side).lower() == "long":
            if z >= float(self._s.PAIRS_EXIT_Z):
                yield Signal(
                    symbol=sym,
                    action=SignalAction.EXIT_LONG,
                    reason=f"pairs_exit_convergence z={z:.3f}",
                    reference_price=last_px,
                    atr=0.0,
                    strategy_name=self.name,
                    confidence=0.8,
                    metadata=meta,
                )
            return []

        if ctx.has_open_order:
            return []

        if z <= float(self._s.PAIRS_ENTRY_Z):
            yield Signal(
                symbol=sym,
                action=SignalAction.ENTER_LONG,
                reason=f"pairs_entry_follower_weak z={z:.3f} vs {leader}",
                reference_price=last_px,
                atr=0.0,
                strategy_name=self.name,
                confidence=min(1.0, max(0.3, 0.5 - 0.15 * (z - float(self._s.PAIRS_ENTRY_Z)))),
                metadata=meta,
            )
