"""News-linked sentiment overlays (VADER) with Alpaca News API fetch.

Pure scoring helpers are deterministic. Network access is confined to fetchers.

An optional future ``LLMSentimentBackend`` can implement the same ``score_texts``
protocol without changing trading gates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Sequence

from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from config.constants import LOGGER_STRATEGY
from config.settings import Settings

_LOG = logging.getLogger(LOGGER_STRATEGY)


class SentimentTextBackend(Protocol):
    """Pluggable scorer (VADER today, optional LLM later)."""

    def score_compound(self, texts: Sequence[str]) -> tuple[float, str]:
        ...


class VaderSentimentBackend:
    """vaderSentiment compound score aggregator over concatenated snippets."""

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    def score_compound(self, texts: Sequence[str]) -> tuple[float, str]:
        if not texts:
            return 0.0, "no_text"
        blob = " ".join(t.strip() for t in texts if t and str(t).strip())
        if not blob:
            return 0.0, "no_text"
        scores = self._analyzer.polarity_scores(blob)
        compound = float(scores.get("compound", 0.0))
        return compound, "vader"


@dataclass(frozen=True)
class SentimentSnapshot:
    symbol: str
    sentiment_score: float
    sentiment_label: str
    headline_count: int
    source_count: int
    latest_headline_timestamp: Optional[str]
    stale_news: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sentiment_symbol": self.symbol,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "sentiment_headline_count": self.headline_count,
            "sentiment_source_count": self.source_count,
            "sentiment_latest_headline_ts": self.latest_headline_timestamp or "",
            "sentiment_stale_news": self.stale_news,
            "sentiment_reason": self.reason,
            "sentiment_blocks_long": self.sentiment_label == "strong_negative",
        }

    def to_overlay_dict(self) -> dict[str, Any]:
        d = self.as_dict()
        d.pop("sentiment_blocks_long", None)
        d["sentiment_blocks_long_entries"] = self.sentiment_label == "strong_negative"
        return d


def classify_label(compound: float, strong_negative_threshold: float) -> str:
    if compound <= strong_negative_threshold:
        return "strong_negative"
    if compound <= -0.05:
        return "negative"
    if compound >= 0.05:
        return "positive"
    return "neutral"


class AlpacaNewsSentimentFetcher:
    """Thin wrapper around alpaca-py ``NewsClient``."""

    def __init__(self, client: NewsClient) -> None:
        self._client = client

    def fetch_headlines(self, symbol: str, *, limit: int) -> tuple[list[str], dict[str, Any]]:
        meta: dict[str, Any] = {"error": None, "sources": set(), "latest_ts": None}
        sym = symbol.strip().upper()
        try:
            req = NewsRequest(symbols=sym, limit=limit)
            nset = self._client.get_news(req)
            articles = []
            news_list = []
            raw = getattr(nset, "data", None)
            if isinstance(raw, dict):
                news_list = raw.get("news", []) or []
            latest_dt: Optional[datetime] = None
            for art in news_list:
                headline = str(getattr(art, "headline", "") or "").strip()
                summary = str(getattr(art, "summary", "") or "").strip()
                text = f"{headline}. {summary}".strip()
                if text:
                    articles.append(text)
                src = getattr(art, "source", None)
                if src:
                    meta["sources"].add(str(src))
                for ts_raw in (getattr(art, "updated_at", None), getattr(art, "created_at", None)):
                    if ts_raw is None:
                        continue
                    if isinstance(ts_raw, datetime):
                        c = ts_raw.astimezone(timezone.utc)
                        latest_dt = c if latest_dt is None or c > latest_dt else latest_dt
            if latest_dt is not None:
                meta["latest_ts"] = latest_dt.isoformat()
            return articles, meta
        except Exception as exc:  # noqa: BLE001
            meta["error"] = str(exc)
            return [], meta


class CachedSentimentProvider:
    """Per-symbol TTL cache with consecutive-failure tracking for fail-closed."""

    def __init__(
        self,
        settings: Settings,
        fetcher: AlpacaNewsSentimentFetcher,
        backend: SentimentTextBackend,
        *,
        record_fn: Optional[Callable[[SentimentSnapshot], None]] = None,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._backend = backend
        self._record_fn = record_fn
        self._cache: dict[str, tuple[float, SentimentSnapshot]] = {}
        self._fail_streak: dict[str, int] = {}

    def invalidate(self, symbol: str) -> None:
        self._cache.pop(symbol.upper(), None)

    def snapshot_for_symbol(self, symbol: str) -> SentimentSnapshot:
        sym = symbol.strip().upper()
        now = time.monotonic()
        ttl = float(self._settings.SENTIMENT_CACHE_TTL_SECONDS)
        ent = self._cache.get(sym)
        if ent and (now - ent[0]) < ttl:
            return ent[1]

        snap = self._build_snapshot(sym)
        self._cache[sym] = (now, snap)
        if self._record_fn is not None:
            try:
                self._record_fn(snap)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("sentiment persistence callback failed %s", exc)
        return snap

    def _build_snapshot(self, sym: str) -> SentimentSnapshot:
        thr = float(self._settings.SENTIMENT_STRONG_NEGATIVE_THRESHOLD)
        if not self._settings.SENTIMENT_ENABLED:
            return SentimentSnapshot(
                symbol=sym,
                sentiment_score=0.0,
                sentiment_label="neutral",
                headline_count=0,
                source_count=0,
                latest_headline_timestamp=None,
                stale_news=False,
                reason="sentiment_disabled",
            )

        texts, meta = self._fetcher.fetch_headlines(
            sym,
            limit=int(self._settings.SENTIMENT_HEADLINE_LIMIT),
        )
        err = meta.get("error")
        if err:
            streak = self._fail_streak.get(sym, 0) + 1
            self._fail_streak[sym] = streak
            block = bool(self._settings.SENTIMENT_FAIL_CLOSED) and streak >= int(
                self._settings.SENTIMENT_FAIL_CONSECUTIVE_THRESHOLD
            )
            label = "strong_negative" if block else "neutral"
            label_reason = "api_fail_closed" if block else "api_error_neutral"
            snap = SentimentSnapshot(
                symbol=sym,
                sentiment_score=0.0,
                sentiment_label=label,
                headline_count=0,
                source_count=0,
                latest_headline_timestamp=None,
                stale_news=True,
                reason=f"{label_reason}:{err}",
            )
            return snap

        self._fail_streak[sym] = 0

        if not texts:
            return SentimentSnapshot(
                symbol=sym,
                sentiment_score=0.0,
                sentiment_label="neutral",
                headline_count=0,
                source_count=0,
                latest_headline_timestamp=None,
                stale_news=False,
                reason="no_headlines_default_neutral",
            )

        compound, _bk = self._backend.score_compound(texts)
        label = classify_label(compound, thr)
        sources = meta.get("sources") or set()
        latest_ts = meta.get("latest_ts")
        stale = False
        if latest_ts:
            try:
                ts_dt = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
                age_sec = (
                    datetime.now(timezone.utc) - ts_dt.astimezone(timezone.utc)
                ).total_seconds()
                stale = age_sec > float(self._settings.SENTIMENT_STALE_AFTER_SECONDS)
            except (ValueError, TypeError, OSError):
                stale = True

        snap = SentimentSnapshot(
            symbol=sym,
            sentiment_score=float(compound),
            sentiment_label=label,
            headline_count=len(texts),
            source_count=len(sources),
            latest_headline_timestamp=str(latest_ts) if latest_ts else None,
            stale_news=stale,
            reason="ok_with_headlines",
        )
        # Stale-but-readable news stays informational unless fail-closed is enabled.
        if stale and self._settings.SENTIMENT_FAIL_CLOSED:
            snap = SentimentSnapshot(
                symbol=snap.symbol,
                sentiment_score=snap.sentiment_score,
                sentiment_label="strong_negative",
                headline_count=snap.headline_count,
                source_count=snap.source_count,
                latest_headline_timestamp=snap.latest_headline_timestamp,
                stale_news=True,
                reason="stale_news_fail_closed",
            )
        return snap


def sentiment_blocks_enter_long(snap: SentimentSnapshot) -> bool:
    return snap.sentiment_label == "strong_negative"


def merge_sentiment_metadata(meta: dict[str, Any], snap: SentimentSnapshot) -> dict[str, Any]:
    out = dict(meta)
    out.update(snap.to_overlay_dict())
    return out


def sentiment_overlay_neutral(symbol: str) -> dict[str, Any]:
    """Explicit neutral overlay when sentiment is disabled (for strategy logs)."""

    return SentimentSnapshot(
        symbol=symbol.strip().upper(),
        sentiment_score=0.0,
        sentiment_label="neutral",
        headline_count=0,
        source_count=0,
        latest_headline_timestamp=None,
        stale_news=False,
        reason="sentiment_disabled",
    ).to_overlay_dict()


__all__ = [
    "AlpacaNewsSentimentFetcher",
    "CachedSentimentProvider",
    "SentimentSnapshot",
    "VaderSentimentBackend",
    "classify_label",
    "merge_sentiment_metadata",
    "sentiment_blocks_enter_long",
    "sentiment_overlay_neutral",
]
