"""Alpaca client construction and capability detection.

Centralizes the construction of:
    - alpaca.trading.client.TradingClient
    - alpaca.trading.stream.TradingStream
    - alpaca.data.historical.stock.StockHistoricalDataClient
    - alpaca.data.live.stock.StockDataStream

Capability detection probes whether the account is entitled to SIP data so
`ALPACA_FEED=auto` can resolve to `sip` when available, otherwise `iex`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

from config.constants import FEED_IEX, FEED_SIP, LOGGER_APP, LOGGER_STREAM
from config.settings import Settings


@dataclass
class AlpacaClients:
    """Bundle of constructed Alpaca clients plus resolved feed metadata."""

    trading: TradingClient
    trading_stream: TradingStream
    historical_data: StockHistoricalDataClient
    market_stream: StockDataStream
    resolved_feed: str
    sip_supported: bool


def _resolve_feed_enum(feed: str) -> DataFeed:
    if feed == FEED_SIP:
        return DataFeed.SIP
    if feed == FEED_IEX:
        return DataFeed.IEX
    raise ValueError(f"unsupported feed: {feed!r}")


def detect_sip_supported(
    historical: StockHistoricalDataClient,
    probe_symbol: str,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Return True iff the account is entitled to SIP for the given symbol.

    Implementation: try a single SIP-feed quote request; on failure assume IEX.
    """
    log = logger or logging.getLogger(LOGGER_APP)
    try:
        request = StockLatestQuoteRequest(
            symbol_or_symbols=[probe_symbol], feed=DataFeed.SIP
        )
        historical.get_stock_latest_quote(request)
        return True
    except Exception as exc:  # noqa: BLE001 - capability probe is best-effort
        log.info("SIP capability probe failed (%s); falling back to IEX.", exc)
        return False


def build_alpaca_clients(settings: Settings) -> AlpacaClients:
    """Construct all Alpaca client objects for the configured environment.

    The trading client is forced to `paper=True` only when `ALPACA_ENV=paper`.
    """
    paper = settings.is_paper

    trading = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
        paper=paper,
    )

    trading_stream = TradingStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
        paper=paper,
    )

    historical_data = StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
    )

    # Determine effective feed: probe SIP only when caller selected `auto`.
    log_app = logging.getLogger(LOGGER_APP)
    log_stream = logging.getLogger(LOGGER_STREAM)
    sip_supported = False
    if settings.feed_preference == "auto":
        probe = settings.symbols_list[0] if settings.symbols_list else "SPY"
        sip_supported = detect_sip_supported(historical_data, probe, logger=log_app)
    elif settings.feed_preference == FEED_SIP:
        sip_supported = True

    resolved_feed = settings.feed_resolved(sip_supported)

    if resolved_feed == FEED_IEX:
        log_app.warning(
            "Market data feed resolved to IEX; spread filtering runs in "
            "DEGRADED CONFIDENCE mode (single-venue quotes)."
        )
    else:
        log_app.info("Market data feed resolved to SIP.")

    market_stream = StockDataStream(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_API_SECRET,
        feed=_resolve_feed_enum(resolved_feed),
    )
    log_stream.info("StockDataStream initialized on feed=%s", resolved_feed)

    return AlpacaClients(
        trading=trading,
        trading_stream=trading_stream,
        historical_data=historical_data,
        market_stream=market_stream,
        resolved_feed=resolved_feed,
        sip_supported=sip_supported,
    )


def shutdown_clients(clients: AlpacaClients) -> None:
    """Best-effort shutdown for streams. Safe to call multiple times."""
    log = logging.getLogger(LOGGER_STREAM)
    for label, stream in (
        ("trading_stream", clients.trading_stream),
        ("market_stream", clients.market_stream),
    ):
        try:
            close: Any = getattr(stream, "stop", None) or getattr(stream, "close", None)
            if close is not None:
                close()
        except Exception as exc:  # noqa: BLE001
            log.warning("Error shutting down %s: %s", label, exc)
