"""Anti-martingale sizing: deflate risk after a loss streak via SQLite ledger."""

from __future__ import annotations

from enum import Enum

from config.settings import Settings
from core.database import CompletedTradeRow


class RiskMode(str, Enum):
    NORMAL = "normal"
    DEFENSIVE = "defensive"


def resolve_anti_martingale(
    settings: Settings,
    recent_newest_first: list[CompletedTradeRow],
) -> tuple[RiskMode, float, str]:
    """Return `(mode, multiplier, rationale_token)`.

    `recent_newest_first` must exclude canary trades (caller filters DB rows).

    Rules (in order):
    1. If the newest ``ANTI_MARTINGALE_WIN_RECOVERY`` trades all have positive PnL ->
       NORMAL with normal multiplier (wins erase defensive posture).
    2. Else if newest ``ANTI_MARTINGALE_LOSS_STREAK`` trades all have negative PnL ->
       DEFENSIVE with defensive multiplier.
    3. Otherwise NORMAL with normal multiplier ("mixed").
    """

    if not settings.ANTI_MARTINGALE_ENABLED:
        return RiskMode.NORMAL, 1.0, "anti_martingale_disabled"

    nw = settings.ANTI_MARTINGALE_WIN_RECOVERY
    if len(recent_newest_first) >= nw:
        wins = recent_newest_first[:nw]
        if all((r.realized_pnl or 0.0) > 1e-9 for r in wins):
            return (
                RiskMode.NORMAL,
                float(settings.ANTI_MARTINGALE_NORMAL_MULTIPLIER),
                f"recovery_last_{nw}_wins",
            )

    ls = settings.ANTI_MARTINGALE_LOSS_STREAK
    if len(recent_newest_first) >= ls:
        losses = recent_newest_first[:ls]
        if all((r.realized_pnl or 0.0) < -1e-9 for r in losses):
            return (
                RiskMode.DEFENSIVE,
                float(settings.ANTI_MARTINGALE_DEFENSIVE_MULTIPLIER),
                f"loss_streak_{ls}",
            )

    return (
        RiskMode.NORMAL,
        float(settings.ANTI_MARTINGALE_NORMAL_MULTIPLIER),
        "mixed_or_neutral",
    )


def recent_trade_pnls_preview(recent_newest_first: list[CompletedTradeRow], max_n: int) -> str:
    """Compact string like ``W/-/L`` for rationales."""

    vals: list[str] = []
    for r in recent_newest_first[:max_n]:
        p = r.realized_pnl
        if p is None:
            vals.append("?")
        elif p > 1e-9:
            vals.append("W")
        elif p < -1e-9:
            vals.append("L")
        else:
            vals.append("0")
    return "".join(vals) if vals else "none"


__all__ = ["RiskMode", "resolve_anti_martingale", "recent_trade_pnls_preview"]
