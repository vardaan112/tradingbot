"""Bot-specific exception hierarchy.

Errors are organized so retry policy can act on broad classes (transient vs.
permanent) without inspecting third-party exception strings.
"""

from __future__ import annotations


class TradingBotError(Exception):
    """Base class for all trading-bot errors."""


class BrokerConnectionError(TradingBotError):
    """Transient connectivity / 5xx / timeout from Alpaca - safe to retry."""


class RateLimitedError(BrokerConnectionError):
    """Alpaca returned HTTP 429 or equivalent throttling response."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class NonRetryableBrokerError(TradingBotError):
    """4xx (other than 429) - the request itself is wrong; do not retry blindly."""


class OrderPlacementError(TradingBotError):
    """Order submission ambiguous or rejected; reconciliation may be needed."""


class OrderRejectedError(NonRetryableBrokerError):
    """Order definitively rejected by broker; do not auto-retry."""


class QuoteUnavailableError(TradingBotError):
    """No quote, stale quote, or malformed quote for the requested symbol."""


class AccountStateError(TradingBotError):
    """Account snapshot was missing required fields or failed validation."""


class StaleStateError(TradingBotError):
    """Local state has not been refreshed within tolerance; trading must pause."""


class KillSwitchLatchedError(TradingBotError):
    """Operation was attempted while the kill switch is latched."""
