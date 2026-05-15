"""Conservative next-bar fill prices for replay (spread + slippage + fees)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillModelParams:
    spread_pct: float = 0.0005
    slippage_bps: float = 1.0
    fee_bps_per_side: float = 0.0
    prevent_same_bar_fills: bool = True

    def slip_frac(self) -> float:
        return float(self.slippage_bps) / 10_000.0

    def fee_frac(self) -> float:
        return float(self.fee_bps_per_side) / 10_000.0

    def half_spread(self) -> float:
        return float(self.spread_pct) / 2.0


def entry_long_fill_price(
    next_bar_open: float,
    *,
    params: FillModelParams,
) -> float:
    """Buy at next bar open + half spread + slippage (adverse for buyer)."""

    o = float(next_bar_open)
    if o <= 0:
        return o
    return o * (1.0 + params.half_spread() + params.slip_frac())


def exit_long_fill_price(
    next_bar_open: float,
    *,
    params: FillModelParams,
) -> float:
    """Sell at next bar open - half spread - slippage."""

    o = float(next_bar_open)
    if o <= 0:
        return o
    return max(1e-12, o * (1.0 - params.half_spread() - params.slip_frac()))


def fees_usd(notional: float, *, params: FillModelParams) -> float:
    return abs(float(notional)) * params.fee_frac()


def same_bar_stop_vs_target_long(
    *,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    stop_price: float,
    target_price: float,
) -> str | None:
    """If both stop (below) and target (above) are touched, return 'stop' or 'target'.

    Adverse outcome first for long: **stop** (loss) before **target** (profit).
    Assumes stop_price < entry < target_price typical long setup.
    Returns None if neither or only one is touched (ambiguous single touch uses touched).
    """

    lo = float(bar_low)
    hi = float(bar_high)
    sp = float(stop_price)
    tp = float(target_price)
    hit_stop = lo <= sp + 1e-12
    hit_target = hi >= tp - 1e-12
    if hit_stop and hit_target:
        return "stop"
    if hit_stop:
        return "stop"
    if hit_target:
        return "target"
    return None


__all__ = [
    "FillModelParams",
    "entry_long_fill_price",
    "exit_long_fill_price",
    "fees_usd",
    "same_bar_stop_vs_target_long",
]
