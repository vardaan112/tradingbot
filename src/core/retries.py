"""Exponential-backoff retry policy with jitter, 429-aware.

`retry_call` wraps a synchronous callable and re-runs it on transient failures.
`retry_call_async` is the async equivalent. Both honor `Retry-After` style
hints surfaced via `RateLimitedError.retry_after_seconds`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Tuple, Type

from config.constants import LOGGER_APP

from .exceptions import (
    BrokerConnectionError,
    NonRetryableBrokerError,
    RateLimitedError,
)

_RETRYABLE_DEFAULT: Tuple[Type[BaseException], ...] = (
    BrokerConnectionError,
    RateLimitedError,
    TimeoutError,
    ConnectionError,
)


def _compute_backoff(
    attempt: int,
    *,
    base: float,
    cap: float,
) -> float:
    """Decorrelated jittered exponential backoff."""
    expo = base * (2 ** (attempt - 1))
    capped = min(cap, expo)
    return random.uniform(base, max(base, capped))


def _resolve_delay(
    attempt: int,
    exc: BaseException,
    *,
    base: float,
    cap: float,
) -> float:
    if isinstance(exc, RateLimitedError) and exc.retry_after_seconds:
        return min(cap, max(base, float(exc.retry_after_seconds)))
    return _compute_backoff(attempt, base=base, cap=cap)


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    retryable: Tuple[Type[BaseException], ...] = _RETRYABLE_DEFAULT,
    op_name: str = "call",
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous retry wrapper. Re-raises on non-retryable failure or budget exhaustion."""
    log = logger or logging.getLogger(LOGGER_APP)
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except NonRetryableBrokerError:
            raise
        except retryable as exc:
            last_exc = exc
            if attempt >= max_attempts:
                log.error(
                    "%s exhausted retries (%d attempts): %s",
                    op_name,
                    attempt,
                    exc,
                )
                raise
            delay = _resolve_delay(attempt, exc, base=base_delay, cap=max_delay)
            log.warning(
                "%s failed (attempt %d/%d): %s. Sleeping %.2fs.",
                op_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{op_name} retry loop exited without success or exception")


async def retry_call_async(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    retryable: Tuple[Type[BaseException], ...] = _RETRYABLE_DEFAULT,
    op_name: str = "call",
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    """Async variant of retry_call."""
    log = logger or logging.getLogger(LOGGER_APP)
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except NonRetryableBrokerError:
            raise
        except retryable as exc:
            last_exc = exc
            if attempt >= max_attempts:
                log.error(
                    "%s exhausted retries (%d attempts): %s",
                    op_name,
                    attempt,
                    exc,
                )
                raise
            delay = _resolve_delay(attempt, exc, base=base_delay, cap=max_delay)
            log.warning(
                "%s failed (attempt %d/%d): %s. Sleeping %.2fs.",
                op_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{op_name} retry loop exited without success or exception")


def classify_http_status(status: int, message: str = "") -> BrokerConnectionError | NonRetryableBrokerError:
    """Translate an HTTP status code into the right exception class.

    Use this to map raw alpaca-py / httpx errors into our retry-aware tree.
    """
    if status == 429:
        return RateLimitedError(message or "rate limited")
    if 500 <= status < 600:
        return BrokerConnectionError(message or f"broker 5xx: {status}")
    if 400 <= status < 500:
        return NonRetryableBrokerError(message or f"broker 4xx: {status}")
    return BrokerConnectionError(message or f"unexpected status: {status}")
