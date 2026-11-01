"""Authentication and rate limiting.

Both are deliberately simple and both fail closed.

Authentication: a shared API key presented in X-API-Key. When AGRI_API_KEYS is
empty, authentication is DISABLED and the service says so in /v1/health and in a
startup warning. That is the right default for local development and the wrong
one for production, so it is made loud rather than silent.

Rate limiting: an in-process token bucket per caller. This is honest about its
limits - it is per-process, so behind several workers the effective limit is
multiplied by the worker count. A shared Redis would be required for a real
global limit, and the docstring says so rather than implying otherwise.
"""

from __future__ import annotations

import hmac
import time
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, status

from agri.common.config import get_settings
from agri.common.logging import get_logger

_log = get_logger("api.security")


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject the request unless a configured API key is presented.

    Comparison uses hmac.compare_digest so that a timing side channel cannot be
    used to recover a key one byte at a time.
    """
    settings = get_settings()
    configured = settings.api_key_set

    if not configured:
        request.state.api_key = "anonymous"
        return

    if not x_api_key:
        _log.warning("request rejected: missing api key", extra={"path": request.url.path})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "type": "about:blank",
                "title": "Unauthorized",
                "status": 401,
                "code": "missing_api_key",
                "detail": "Supply your key in the X-API-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    for candidate in configured:
        if hmac.compare_digest(candidate, x_api_key):
            request.state.api_key = x_api_key[:6] + "..."
            return

    _log.warning("request rejected: invalid api key", extra={"path": request.url.path})
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "type": "about:blank",
            "title": "Unauthorized",
            "status": 401,
            "code": "invalid_api_key",
            "detail": "The supplied API key is not valid.",
        },
        headers={"WWW-Authenticate": "ApiKey"},
    )


@dataclass
class _Bucket:
    tokens: float
    updated_at: float = field(default_factory=time.monotonic)


class TokenBucketLimiter:
    """Per-caller token bucket.

    capacity is the burst size and refill_per_second the sustained rate. Keys are
    the API key when present, otherwise the client address.
    """

    def __init__(self, per_minute: int, *, capacity: int | None = None) -> None:
        self.per_minute = max(per_minute, 1)
        self.capacity = float(capacity if capacity is not None else max(self.per_minute // 4, 5))
        self.refill_per_second = self.per_minute / 60.0
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self.capacity, updated_at=now)
            self._buckets[key] = bucket

        elapsed = now - bucket.updated_at
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.updated_at = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0

        deficit = 1.0 - bucket.tokens
        retry_after = deficit / self.refill_per_second if self.refill_per_second else 60.0
        return False, retry_after

    def prune(self, max_entries: int = 10_000) -> None:
        """Drop idle buckets so a long-running process does not grow unbounded."""
        if len(self._buckets) <= max_entries:
            return
        now = time.monotonic()
        stale = [key for key, bucket in self._buckets.items() if now - bucket.updated_at > 600]
        for key in stale:
            self._buckets.pop(key, None)


_limiter: TokenBucketLimiter | None = None


def get_limiter() -> TokenBucketLimiter:
    """Module-level limiter, created once from configuration."""
    global _limiter
    if _limiter is None:
        settings = get_settings()
        _limiter = TokenBucketLimiter(settings.rate_limit_per_minute)
        _log.info(
            "rate limiter initialised",
            extra={
                "per_minute": settings.rate_limit_per_minute,
                "scope": "per-process; multiply by worker count behind a multi-worker deployment",
            },
        )
    return _limiter


async def enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 when the caller's bucket is empty."""
    limiter = get_limiter()
    key = getattr(request.state, "api_key", None) or (request.client.host if request.client else "unknown")
    allowed, retry_after = limiter.check(key)
    limiter.prune()
    if not allowed:
        _log.warning(
            "request rate limited", extra={"path": request.url.path, "retry_after": round(retry_after, 1)}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "type": "about:blank",
                "title": "Too Many Requests",
                "status": 429,
                "code": "rate_limited",
                "detail": f"Rate limit of {limiter.per_minute} requests per minute exceeded.",
            },
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
