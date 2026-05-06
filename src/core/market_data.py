"""Quote cache + historical-bar fetching for indicator warm-up.

`QuoteCache` holds the latest top-of-book per symbol with timestamps and feed
metadata. `BarFetcher` retrieves enough historical bars for indicators to be
ready immediately after startup so the bot does not need to wait through a
warm-up period before evaluating signals.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.constants import LOGGER_DATA
from utils.price_utils import is_valid_quote, mid_price, spread_pct
from utils.time_utils import seconds_since

from .exceptions import BrokerConnectionError, QuoteUnavailableError
from .retries import retry_call


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    feed: str

    def age_seconds(self, *, reference: Optional[datetime] = None) -> float:
        return seconds_since(self.timestamp, reference=reference)

    def mid(self) -> float:
        return mid_price(self.bid, self.ask)

    def spread_pct(self) -> float:
        return spread_pct(self.bid, self.ask)

    def is_fresh(self, max_age_seconds: float) -> bool:
        return is_valid_quote(
            self.bid,
            self.ask,
            quote_age_seconds=self.age_seconds(),
            max_age_seconds=max_age_seconds,
        )


def _parse_timeframe(spec: str) -> TimeFrame:
    """Convert env value like '5Min' / '1Hour' / '1Day' into a TimeFrame."""
    spec = spec.strip()
    mapping: dict[str, tuple[int, TimeFrameUnit]] = {
        "1Min": (1, TimeFrameUnit.Minute),
        "5Min": (5, TimeFrameUnit.Minute),
        "15Min": (15, TimeFrameUnit.Minute),
        "1Hour": (1, TimeFrameUnit.Hour),
        "1Day": (1, TimeFrameUnit.Day),
    }
    if spec not in mapping:
        raise ValueError(f"unsupported BAR_TIMEFRAME: {spec!r}")
    amount, unit = mapping[spec]
    return TimeFrame(amount, unit)


class QuoteCache:
    """Thread-safe latest-quote store keyed by symbol."""

    def __init__(self, *, max_age_seconds: float, feed: str) -> None:
        self._lock = threading.RLock()
        self._quotes: dict[str, Quote] = {}
        self._max_age_seconds = max_age_seconds
        self._feed = feed
        self._log = logging.getLogger(LOGGER_DATA)

    def update_from_event(self, event: object) -> None:
        """Update cache from a streaming Quote event published by alpaca-py."""
        symbol = getattr(event, "symbol", None)
        bid = getattr(event, "bid_price", None)
        ask = getattr(event, "ask_price", None)
        bid_size = getattr(event, "bid_size", 0.0) or 0.0
        ask_size = getattr(event, "ask_size", 0.0) or 0.0
        timestamp = getattr(event, "timestamp", None)
        if not symbol or bid is None or ask is None or timestamp is None:
            return
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        with self._lock:
            self._quotes[symbol.upper()] = Quote(
                symbol=symbol.upper(),
                bid=float(bid),
                ask=float(ask),
                bid_size=float(bid_size),
                ask_size=float(ask_size),
                timestamp=timestamp,
                feed=self._feed,
            )

    def set_quote(self, q: Quote) -> None:
        with self._lock:
            self._quotes[q.symbol.upper()] = q

    def get(self, symbol: str) -> Optional[Quote]:
        with self._lock:
            return self._quotes.get(symbol.upper())

    def latest_age_seconds(self) -> Optional[float]:
        with self._lock:
            if not self._quotes:
                return None
            youngest = max(self._quotes.values(), key=lambda q: q.timestamp)
            return youngest.age_seconds()

    def fresh_quote(self, symbol: str) -> Quote:
        q = self.get(symbol)
        if q is None:
            raise QuoteUnavailableError(f"no quote cached for {symbol}")
        if not q.is_fresh(self._max_age_seconds):
            raise QuoteUnavailableError(
                f"stale quote for {symbol}: age={q.age_seconds():.2f}s "
                f"max={self._max_age_seconds:.2f}s"
            )
        return q

    @property
    def feed(self) -> str:
        return self._feed

    @property
    def max_age_seconds(self) -> float:
        return self._max_age_seconds


class BarFetcher:
    """Fetch historical bars for indicator warm-up via the historical client."""

    def __init__(
        self,
        client: StockHistoricalDataClient,
        *,
        feed: str,
        max_attempts: int,
        base_delay: float,
        max_delay: float,
    ) -> None:
        self._client = client
        self._feed_enum = DataFeed.SIP if feed == "sip" else DataFeed.IEX
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._log = logging.getLogger(LOGGER_DATA)

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Fetch the most recent `lookback_bars` for `symbol`.

        Returns a DataFrame indexed by timestamp with columns:
        open, high, low, close, volume, trade_count, vwap.
        Empty DataFrame if no data is available.
        """
        tf = _parse_timeframe(timeframe)
        # Pad lookback to absorb non-trading days/holidays.
        if "Min" in timeframe:
            window = timedelta(days=max(5, int(lookback_bars / 78) + 3))
        elif "Hour" in timeframe:
            window = timedelta(days=max(10, int(lookback_bars / 7) + 5))
        else:
            window = timedelta(days=max(60, lookback_bars * 2))

        end = datetime.now(timezone.utc)
        start = end - window

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start,
            end=end,
            feed=self._feed_enum,
            limit=10_000,
        )

        def _do() -> pd.DataFrame:
            try:
                resp = self._client.get_stock_bars(request)
            except Exception as exc:  # noqa: BLE001
                raise BrokerConnectionError(f"get_stock_bars({symbol}): {exc}") from exc
            try:
                df = resp.df
            except Exception as exc:  # noqa: BLE001
                raise BrokerConnectionError(f"bars df conversion failed: {exc}") from exc
            if df is None or df.empty:
                return pd.DataFrame()
            # Multi-symbol responses come back with a (symbol, timestamp) MultiIndex.
            if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
                try:
                    df = df.xs(symbol, level="symbol")
                except KeyError:
                    return pd.DataFrame()
            return df.tail(lookback_bars).copy()

        return retry_call(
            _do,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            op_name=f"fetch_bars[{symbol}]",
            logger=self._log,
        )

    def fetch_latest_quote(self, symbol: str) -> Quote:
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=self._feed_enum)

        def _do() -> Quote:
            try:
                resp = self._client.get_stock_latest_quote(request)
            except Exception as exc:  # noqa: BLE001
                raise BrokerConnectionError(f"get_stock_latest_quote: {exc}") from exc
            if symbol not in resp:
                raise QuoteUnavailableError(f"no latest quote for {symbol}")
            raw = resp[symbol]
            ts = getattr(raw, "timestamp", None)
            if ts is None:
                raise QuoteUnavailableError(f"missing timestamp on latest quote for {symbol}")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return Quote(
                symbol=symbol.upper(),
                bid=float(getattr(raw, "bid_price", 0.0) or 0.0),
                ask=float(getattr(raw, "ask_price", 0.0) or 0.0),
                bid_size=float(getattr(raw, "bid_size", 0.0) or 0.0),
                ask_size=float(getattr(raw, "ask_size", 0.0) or 0.0),
                timestamp=ts,
                feed=self._feed_enum.value,
            )

        return retry_call(
            _do,
            max_attempts=self._max_attempts,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            op_name=f"latest_quote[{symbol}]",
            logger=self._log,
        )
