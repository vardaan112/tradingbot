"""Numeric helpers shared across risk and strategy code."""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


def clamp(value: Number, lower: Number, upper: Number) -> Number:
    """Clamp `value` to the inclusive range [lower, upper]."""
    if lower > upper:
        raise ValueError(f"clamp: lower {lower} > upper {upper}")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def safe_div(numerator: Number, denominator: Number, *, default: float = 0.0) -> float:
    """Return `numerator / denominator`, or `default` if denominator is ~0.

    Treats values with absolute magnitude < 1e-12 as zero to avoid blow-ups
    from float jitter.
    """
    if abs(denominator) < 1e-12:
        return float(default)
    return float(numerator) / float(denominator)
