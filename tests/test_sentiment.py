"""Sentiment scoring and overlays (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from strategies.sentiment import (
    CachedSentimentProvider,
    SentimentSnapshot,
    VaderSentimentBackend,
    classify_label,
    sentiment_blocks_enter_long,
    sentiment_overlay_neutral,
)


def test_classify_label_buckets() -> None:
    thr = -0.5
    assert classify_label(-0.6, thr) == "strong_negative"
    assert classify_label(-0.2, thr) == "negative"
    assert classify_label(0.0, thr) == "neutral"
    assert classify_label(0.2, thr) == "positive"


def test_strong_negative_blocks_long() -> None:
    snap = SentimentSnapshot(
        symbol="SPY",
        sentiment_score=-0.9,
        sentiment_label="strong_negative",
        headline_count=3,
        source_count=1,
        latest_headline_timestamp=None,
        stale_news=False,
        reason="unit",
    )
    assert sentiment_blocks_enter_long(snap)


def test_neutral_overlay_not_blocking() -> None:
    ov = sentiment_overlay_neutral("aapl")
    assert ov["sentiment_label"] == "neutral"
    assert not ov["sentiment_blocks_long_entries"]


def test_cached_provider_no_news_is_neutral(make_settings_factory, monkeypatch):
    settings = make_settings_factory(
        SENTIMENT_ENABLED=True,
        TEARSHEET_PRIMARY="sqlite",
        DATABASE_PATH=Path("./runtime/tradingbot.sqlite3"),
    )

    class _F:
        def fetch_headlines(self, symbol: str, *, limit: int):
            return [], {"error": None, "sources": set(), "latest_ts": None}

    class _B:
        def score_compound(self, texts):
            return 0.0, "noop"

    prov = CachedSentimentProvider(settings, _F(), _B())  # type: ignore[arg-type]
    s = prov.snapshot_for_symbol("SPY")
    assert s.headline_count == 0
    assert s.sentiment_label == "neutral"


def test_vader_backend_headlines_positive() -> None:
    pytest.importorskip("vaderSentiment")
    b = VaderSentimentBackend()
    score, bk = b.score_compound(["Stellar earnings beat expectations great outlook"])
    assert bk == "vader"
    assert score >= 0.05
