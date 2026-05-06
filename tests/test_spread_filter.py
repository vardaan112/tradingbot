"""Tests for the spread filter, quote validation, tick rounding, and IDs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from utils.ids import generate_client_order_id, short_uuid
from utils.price_utils import (
    is_valid_quote,
    mid_price,
    round_to_tick,
    spread_pct,
    tick_size_for,
)


def test_spread_pct_basic():
    assert spread_pct(100.00, 100.10) == pytest.approx((0.10) / 100.05)


def test_spread_threshold_5bps():
    sp = spread_pct(100.00, 100.05)
    assert sp == pytest.approx(0.05 / 100.025)
    # 0.0005 threshold: this 5-cent spread on a $100 stock is 4.998 bps -> passes.
    assert sp <= 0.0005


def test_invalid_quote_rejected_when_inverted():
    assert not is_valid_quote(100.10, 100.00)


def test_invalid_quote_rejected_when_zero_or_negative():
    assert not is_valid_quote(0.0, 100.00)
    assert not is_valid_quote(100.0, 0.0)
    assert not is_valid_quote(-1.0, 1.0)
    assert not is_valid_quote(None, 1.0)


def test_quote_stale_rejected():
    assert not is_valid_quote(100.0, 100.05, quote_age_seconds=10.0, max_age_seconds=5.0)
    assert is_valid_quote(100.0, 100.05, quote_age_seconds=2.0, max_age_seconds=5.0)


def test_mid_price_invalid_inputs_raise():
    with pytest.raises(ValueError):
        mid_price(100.0, 99.0)
    with pytest.raises(ValueError):
        mid_price(0.0, 1.0)


def test_tick_size_above_one_is_penny():
    assert tick_size_for(10.0) == 0.01
    assert tick_size_for(1.0) == 0.01


def test_tick_size_below_one_is_subpenny():
    assert tick_size_for(0.99) == 0.0001
    assert tick_size_for(0.5) == 0.0001


def test_round_to_tick_modes():
    assert round_to_tick(10.123, mode="down") == 10.12
    assert round_to_tick(10.123, mode="up") == 10.13
    assert round_to_tick(10.125, mode="nearest") in (10.12, 10.13)


def test_round_to_tick_subpenny():
    assert round_to_tick(0.12345, mode="down") == 0.1234
    assert round_to_tick(0.12345, mode="up") == 0.1235


def test_short_uuid_unique():
    s = {short_uuid(8) for _ in range(200)}
    assert len(s) > 190  # collisions extremely unlikely


def test_client_order_id_format_and_uniqueness():
    ids = {generate_client_order_id("rsi", "AAPL", "buy") for _ in range(1000)}
    assert len(ids) == 1000
    sample = next(iter(ids))
    parts = sample.split("-")
    assert len(parts) == 5
    assert parts[0] == "rsi"
    assert parts[1] == "AAPL"
    assert parts[2] == "buy"


def test_client_order_id_sanitization():
    coid = generate_client_order_id("strat one", "AAPL$", "buy", short_id="abc def")
    # No spaces or invalid chars survive.
    for ch in coid:
        assert ch.isalnum() or ch in {"-", "_", "."}
